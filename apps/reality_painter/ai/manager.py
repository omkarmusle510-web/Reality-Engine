"""AI subsystem orchestration for Reality Painter.

`AIManager` is the single entry point Reality Painter uses to interact
with the AI subsystem - the AI-side equivalent of
`engine.core.engine.Engine`. It performs no image generation, no HTTP
requests, and contains no provider-specific logic of any kind: it does
not know about Gemini, Groq, OpenAI, Meshy, TRELLIS, or any other
backend, current or future. Its only job is to coordinate the AI
workflow - building prompts, analyzing sketches, selecting a provider,
dispatching requests, tracking their lifecycle, consulting a cache,
recording history, and translating failures into a uniform response -
through small, structurally-typed collaborator interfaces (see the
`Protocol` classes below) that concrete implementations plug into later
without ever requiring a change to this file.

Reality Painter talks only to `AIManager`. `AIManager` talks only to its
injected collaborators. A future provider, prompt builder, sketch
analyzer, cache, or history store is added by constructing an object
that satisfies the relevant protocol and passing it in - never by
modifying this module.

Shared data contracts (`AICapability`, `RequestStatus`, `AIRequest`,
`AIResponse`, `GenerationRecord`) live in `apps.reality_painter.ai.models`
and are re-exported here unchanged, so existing imports of these names
from this module (e.g. `apps.reality_painter.ai.prompt_builder`)
continue to work without modification.
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import replace
from typing import Any, Deque, Dict, List, Optional, Protocol, runtime_checkable

from apps.reality_painter.ai.models import (
    AICapability,
    AIRequest,
    AIResponse,
    GenerationRecord,
    RequestStatus,
)
from engine.core.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_HISTORY_MAX_ENTRIES = 100


# --- Collaborator interfaces -----------------------------------------------
#
# Every interface below is a structural (`Protocol`) type, not an
# abstract base class: a concrete implementation satisfies one of these
# simply by having the right methods, with no inheritance and no
# dependency on this module at import time. This is what lets a future
# provider, prompt builder, sketch analyzer, cache, or history store
# live entirely outside `apps/reality_painter/ai/` (or even outside
# Reality Painter) while still plugging into `AIManager` unmodified.


@runtime_checkable
class PromptBuilder(Protocol):
    """Builds a finished prompt string from user input and sketch data."""

    def build(
        self,
        capability: AICapability,
        user_input: Optional[str],
        sketch_analysis: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> str:
        """Returns the finished prompt text for one request."""
        ...


@runtime_checkable
class SketchAnalyzer(Protocol):
    """Extracts structured information from a user's sketch."""

    def analyze(self, sketch: Any) -> Dict[str, Any]:
        """Returns structured data describing `sketch`.

        `sketch` is opaque to `AIManager` - it is whatever the caller
        passes to `AIManager.generate()` (e.g. canvas pixel data), and
        only this analyzer needs to know its shape.
        """
        ...


@runtime_checkable
class AIProvider(Protocol):
    """A single AI backend capable of servicing one or more capabilities.

    `AIManager` never imports or knows about a concrete provider - it
    only ever talks to objects satisfying this shape, registered at
    runtime via `AIManager.register_provider`.
    """

    name: str

    def supports(self, capability: AICapability) -> bool:
        """Returns True if this provider can service `capability`."""
        ...

    def generate(self, request: AIRequest) -> AIResponse:
        """Executes `request` and returns its outcome.

        May raise - `AIManager` catches any exception raised here and
        converts it into a failed `AIResponse`, so a provider
        implementation is free to let underlying errors (network,
        auth, quota, ...) propagate rather than handling every case
        itself.
        """
        ...


@runtime_checkable
class GenerationCache(Protocol):
    """Caches `AIResponse`s by an opaque string key."""

    def get(self, key: str) -> Optional[AIResponse]:
        """Returns a previously cached response for `key`, or `None`."""
        ...

    def set(self, key: str, response: AIResponse) -> None:
        """Stores `response` under `key`."""
        ...


@runtime_checkable
class GenerationHistory(Protocol):
    """Persists completed `GenerationRecord`s."""

    def record(self, record: GenerationRecord) -> None:
        """Appends `record` to history."""
        ...

    def recent(self, limit: int) -> List[GenerationRecord]:
        """Returns up to `limit` most recent records, newest first."""
        ...


# --- Errors ---------------------------------------------------------------


class AIManagerError(Exception):
    """Base class for errors raised directly by `AIManager`."""


class NoProviderAvailableError(AIManagerError):
    """Raised when no registered provider supports a requested capability."""


# --- Default history --------------------------------------------------


class _InMemoryHistory:
    """Default `GenerationHistory` used when the caller injects none.

    A minimal, bounded, thread-safe, in-memory implementation, so
    `AIManager` always has somewhere to record generation history even
    before any persistent history store exists. A caller that wants
    persistence (disk, a database, ...) injects its own object
    satisfying `GenerationHistory` instead; this class is never
    referenced outside `AIManager`.
    """

    def __init__(self, max_entries: int = _DEFAULT_HISTORY_MAX_ENTRIES) -> None:
        self._entries: Deque[GenerationRecord] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def record(self, record: GenerationRecord) -> None:
        with self._lock:
            self._entries.append(record)

    def recent(self, limit: int) -> List[GenerationRecord]:
        with self._lock:
            entries = list(self._entries)
        return entries[-limit:][::-1]


# --- Manager --------------------------------------------------------------


class AIManager:
    """Central orchestrator for Reality Painter's AI subsystem.

    The AI-side equivalent of `Engine`: Reality Painter talks only to
    `AIManager`, never to a concrete provider, prompt builder, sketch
    analyzer, cache, or history store directly. `AIManager` performs no
    generation, HTTP, or provider-specific work itself - it only builds
    prompts (via an injected `PromptBuilder`), analyzes sketches (via an
    injected `SketchAnalyzer`), selects a registered `AIProvider` by
    capability, dispatches the request, consults an optional cache,
    records history, and translates provider failures into a uniform
    `AIResponse` rather than letting exceptions escape to the caller.

    Every collaborator is optional and injected, never imported by
    concrete type. A new provider - or a future capability such as 3D
    generation or image editing - is added by registering an object
    that satisfies `AIProvider` and advertises the relevant
    `AICapability`; `AIManager` itself never changes.

    No global state and no singleton: every application owns its own
    `AIManager` instance. Provider registration is guarded by a lock so
    registering/unregistering from one thread is safe while `generate()`
    runs on another; `generate()` does not hold that lock while a
    provider is running, so concurrent requests are never serialized
    against each other by the registry lock alone. `generate()` is
    synchronous by design in this phase - an async provider is free to
    run its own I/O internally, and a future async entry point can be
    added alongside this one without changing how providers, prompt
    builders, or sketch analyzers are written.
    """

    def __init__(
        self,
        prompt_builder: Optional[PromptBuilder] = None,
        sketch_analyzer: Optional[SketchAnalyzer] = None,
        cache: Optional[GenerationCache] = None,
        history: Optional[GenerationHistory] = None,
    ) -> None:
        """Creates an AI manager with optional, injected collaborators.

        Every argument is optional so `AIManager` is fully usable the
        moment it's constructed, even before any provider, prompt
        builder, sketch analyzer, cache, or history store exists - later
        phases inject each one without touching this class.

        Args:
            prompt_builder: Builds finished prompt strings. If omitted,
                `generate()` uses `user_input` verbatim as the prompt.
            sketch_analyzer: Extracts structured data from a sketch. If
                omitted, sketches are passed through unanalyzed.
            cache: Caches responses by capability/prompt/provider. If
                omitted, every request reaches a provider.
            history: Records completed requests/responses. If omitted,
                a bounded in-memory history is used internally.
        """
        self._prompt_builder = prompt_builder
        self._sketch_analyzer = sketch_analyzer
        self._cache = cache
        self._history: GenerationHistory = history if history is not None else _InMemoryHistory()

        self._providers: Dict[str, AIProvider] = {}
        self._registry_lock = threading.Lock()

    # --- Provider registry -----------------------------------------

    def register_provider(self, provider: AIProvider) -> None:
        """Registers a provider so it becomes eligible for selection.

        Registering under a name that is already registered replaces
        the previous provider - this is what lets a caller swap a
        provider implementation (e.g. a new backend or API key) without
        recreating `AIManager`.

        Args:
            provider: Any object satisfying the `AIProvider` protocol.
        """
        with self._registry_lock:
            self._providers[provider.name] = provider
        logger.info("AI provider '%s' registered.", provider.name)

    def unregister_provider(self, name: str) -> None:
        """Removes a provider by name, if registered. Safe to call otherwise.

        Args:
            name: The provider's `name`, as passed to `register_provider`.
        """
        with self._registry_lock:
            removed = self._providers.pop(name, None) is not None
        if removed:
            logger.info("AI provider '%s' unregistered.", name)

    def list_providers(self) -> List[str]:
        """Returns the names of all currently registered providers."""
        with self._registry_lock:
            return list(self._providers.keys())

    def select_provider(self, capability: AICapability, preferred: Optional[str] = None) -> AIProvider:
        """Selects a registered provider that supports the given capability.

        Args:
            capability: The capability the returned provider must
                support.
            preferred: A provider name to prefer, used if it is
                registered and supports `capability`. Falls back to the
                first registered provider (in registration order) that
                supports it otherwise.

        Returns:
            A provider satisfying `capability`.

        Raises:
            NoProviderAvailableError: If no registered provider supports
                `capability`.
        """
        with self._registry_lock:
            if preferred is not None:
                candidate = self._providers.get(preferred)
                if candidate is not None and candidate.supports(capability):
                    return candidate

            for candidate in self._providers.values():
                if candidate.supports(capability):
                    return candidate

        raise NoProviderAvailableError(f"No registered provider supports capability {capability.value!r}.")

    # --- Generation ---------------------------------------------------

    def generate(
        self,
        capability: AICapability,
        *,
        user_input: Optional[str] = None,
        sketch: Any = None,
        context: Optional[Dict[str, Any]] = None,
        preferred_provider: Optional[str] = None,
        use_cache: bool = True,
    ) -> AIResponse:
        """Runs one full AI request end to end and returns its outcome.

        Coordinates the entire workflow behind a single call: analyzes
        `sketch` (if given and a `SketchAnalyzer` is configured), builds
        a prompt (via the configured `PromptBuilder`, or `user_input`
        verbatim if none is configured), checks the cache, selects a
        provider for `capability`, dispatches the request, and records
        the outcome to history - regardless of whether it succeeded.

        Never raises for a workflow or provider-side failure: a missing
        provider (`NoProviderAvailableError`), a provider exception, or
        any other error during analysis/prompt-building is caught and
        returned as an `AIResponse` with `success=False` and `error`
        set, so callers never need a try/except around this call.

        Args:
            capability: The kind of work being requested.
            user_input: Raw user-provided text, e.g. a typed prompt.
                Ignored if a `PromptBuilder` is configured and derives
                its own prompt from `sketch`/`context` instead.
            sketch: Opaque sketch data (e.g. canvas pixels), passed to
                the configured `SketchAnalyzer`, if any. `AIManager`
                never interprets this itself.
            context: Arbitrary caller-supplied context passed through to
                the prompt builder and attached to the resulting
                `AIRequest`.
            preferred_provider: A provider name to prefer for selection.
                See `select_provider`.
            use_cache: If False, bypasses the cache entirely for this
                call (both read and write) - useful when a caller
                explicitly wants a fresh generation.

        Returns:
            The completed `AIResponse`, whether it succeeded or failed.
        """
        request_context = dict(context) if context else {}
        request_id = str(uuid.uuid4())

        try:
            sketch_analysis = self._analyze_sketch(sketch)
            prompt = self._build_prompt(capability, user_input, sketch_analysis, request_context)
        except Exception as exc:
            logger.exception("AI request %s failed before dispatch.", request_id)
            return AIResponse(request_id=request_id, success=False, error=str(exc))

        request = AIRequest(
            request_id=request_id,
            capability=capability,
            prompt=prompt,
            sketch_analysis=sketch_analysis,
            context=request_context,
        )

        cache_key = self._cache_key(capability, prompt, preferred_provider)
        if use_cache and self._cache is not None and cache_key is not None:
            cached_response = self._cache.get(cache_key)
            if cached_response is not None:
                logger.debug("AI request %s served from cache.", request_id)
                request.status = RequestStatus.SUCCEEDED
                served_response = replace(cached_response, request_id=request_id, cached=True)
                self._history.record(GenerationRecord(request=request, response=served_response))
                return served_response

        request.status = RequestStatus.RUNNING

        try:
            provider = self.select_provider(capability, preferred=preferred_provider)
        except NoProviderAvailableError as exc:
            request.status = RequestStatus.FAILED
            response = AIResponse(request_id=request_id, success=False, error=str(exc))
            self._history.record(GenerationRecord(request=request, response=response))
            return response

        try:
            response = provider.generate(request)
        except Exception as exc:
            logger.exception("Provider '%s' raised during AI request %s.", provider.name, request_id)
            request.status = RequestStatus.FAILED
            response = AIResponse(
                request_id=request_id,
                success=False,
                provider_name=provider.name,
                error=str(exc),
            )
            self._history.record(GenerationRecord(request=request, response=response))
            return response

        request.status = RequestStatus.SUCCEEDED if response.success else RequestStatus.FAILED

        if use_cache and self._cache is not None and cache_key is not None and response.success:
            self._cache.set(cache_key, response)

        self._history.record(GenerationRecord(request=request, response=response))
        return response

    # --- History --------------------------------------------------------

    def get_history(self, limit: int = 20) -> List[GenerationRecord]:
        """Returns the most recent generation records, newest first.

        Args:
            limit: Maximum number of records to return.
        """
        return self._history.recent(limit)

    # --- Internal helpers -------------------------------------------------

    def _analyze_sketch(self, sketch: Any) -> Optional[Dict[str, Any]]:
        """Runs the configured `SketchAnalyzer`, if any and if `sketch` is given."""
        if sketch is None or self._sketch_analyzer is None:
            return None
        return self._sketch_analyzer.analyze(sketch)

    def _build_prompt(
        self,
        capability: AICapability,
        user_input: Optional[str],
        sketch_analysis: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> str:
        """Builds the final prompt string for a request.

        Delegates to the configured `PromptBuilder`, passing along
        whatever sketch analysis was just produced, if any. Falls back
        to `user_input` verbatim (empty string if `user_input` is
        `None`) when no `PromptBuilder` is configured, so `AIManager`
        remains fully usable before one exists.
        """
        if self._prompt_builder is not None:
            return self._prompt_builder.build(capability, user_input, sketch_analysis, context)
        return user_input or ""

    def _cache_key(
        self,
        capability: AICapability,
        prompt: str,
        preferred_provider: Optional[str],
    ) -> Optional[str]:
        """Derives a cache key for a request, or `None` if nothing is cacheable.

        Deliberately simple: capability, prompt text, and the preferred
        provider name are the only inputs that determine whether two
        requests are "the same" for caching purposes. An empty prompt is
        never cacheable, since it carries no information to key on. A
        future cache implementation is free to hash or namespace this
        further - `AIManager` only ever needs a stable string.
        """
        if not prompt:
            return None
        return f"{capability.value}:{preferred_provider or 'auto'}:{prompt}"