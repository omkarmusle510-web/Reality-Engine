"""Provider contract for Reality Painter's AI subsystem.

This module defines the interface every concrete AI backend (Gemini,
Groq, OpenAI, Meshy, TRELLIS, or any future provider) must implement to
plug into `apps.reality_painter.ai.manager.AIManager`. It contains no
provider-specific logic, no HTTP calls, no authentication, and no
generation logic of any kind - only the contract.

`AIManager` already defines a minimal structural `AIProvider` Protocol
(`name`, `supports`, `generate`) in `manager.py`. `AIProviderBase` here
satisfies that same Protocol - a provider built on it is usable by
`AIManager` without any change to `manager.py` - while giving concrete
providers a common, opinionated base to inherit from instead of
re-implementing capability discovery and health checking themselves.

Reality Painter never talks to a provider directly:

    Reality Painter -> AIManager -> AIProvider interface -> concrete provider

A concrete provider only needs to:
    1. Declare which `AICapability` values it supports.
    2. Implement `generate()`.
    3. Optionally override any of the auxiliary capability hooks below
       (image editing, variations, sketch understanding, prompt
       optimization, 3D generation) that go beyond plain generation.
    4. Optionally override `check_health()`.
Everything else is provided by this base class.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional

from apps.reality_painter.ai.manager import AICapability, AIRequest, AIResponse


# --- Health -----------------------------------------------------------


class ProviderHealthStatus(str, Enum):
    """Coarse-grained health state of a provider."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderHealth:
    """The result of a single provider health check.

    Attributes:
        status: Coarse-grained health state.
        message: Human-readable detail (e.g. an error summary), or
            `None` if there's nothing more to say than `status`.
        checked_at: Wall-clock time the check was performed
            (`time.time()`).
    """

    status: ProviderHealthStatus
    message: Optional[str] = None
    checked_at: float = field(default_factory=time.time)


# --- Provider base --------------------------------------------------------


class AIProviderBase(ABC):
    """Base class for a single AI backend usable by `AIManager`.

    Satisfies `apps.reality_painter.ai.manager.AIProvider` structurally
    (`name`, `supports`, `generate`), so any subclass plugs into
    `AIManager.register_provider()` unmodified. Capability discovery is
    handled once, here, from a `capabilities` set declared by the
    subclass at construction time - a concrete provider never needs to
    re-implement `supports()`.

    Only `generate()` is required. Every other capability - image
    editing, variations, sketch understanding, prompt optimization, and
    3D generation (image-to-3D, text-to-3D) - is an optional hook a
    provider overrides only if it supports that capability. The default
    implementation raises `NotImplementedError`, so calling an
    unsupported hook fails loudly and specifically rather than silently
    doing nothing.

    None of this class touches HTTP, authentication, caching, retries,
    or prompt construction - those remain the concern of a concrete
    provider (transport/auth) or `AIManager`'s injected collaborators
    (caching, prompt building).
    """

    def __init__(self, name: str, capabilities: FrozenSet[AICapability]) -> None:
        """Creates a provider with a fixed name and declared capabilities.

        Args:
            name: Stable, unique provider name (e.g. "gemini", "groq").
                Used by `AIManager` for registration and preferred-
                provider selection.
            capabilities: The set of `AICapability` values this provider
                supports. Drives `supports()`; never mutated after
                construction, so a provider's advertised capabilities
                are stable for its lifetime.
        """
        self._name = name
        self._capabilities = frozenset(capabilities)

    # --- Identity and capability discovery ------------------------

    @property
    def name(self) -> str:
        """Stable, unique name identifying this provider."""
        return self._name

    @property
    def capabilities(self) -> FrozenSet[AICapability]:
        """The set of `AICapability` values this provider supports."""
        return self._capabilities

    def supports(self, capability: AICapability) -> bool:
        """Returns True if this provider was declared to support `capability`.

        Args:
            capability: The capability to check.
        """
        return capability in self._capabilities

    # --- Core generation (required) ---------------------------------

    @abstractmethod
    def generate(self, request: AIRequest) -> AIResponse:
        """Executes `request` and returns its outcome.

        `request.capability` is guaranteed by `AIManager` to be a
        capability this provider was selected for (i.e. `supports()`
        returned True). A provider is free to let underlying transport,
        auth, or quota errors propagate as exceptions - `AIManager`
        catches them and converts them into a failed `AIResponse`.

        Args:
            request: The provider-agnostic request to execute.

        Returns:
            The completed `AIResponse`.
        """
        raise NotImplementedError

    # --- Optional capability hooks -----------------------------------
    #
    # These are convenience entry points for capabilities that don't
    # map cleanly onto a single `generate()` call, or that a caller may
    # want to invoke directly (e.g. sketch analysis ahead of building a
    # prompt). A provider overrides only the hooks it supports; the
    # default raises `NotImplementedError` naming the provider and
    # capability, so an unsupported call fails clearly instead of
    # silently returning a placeholder.

    def edit_image(self, image: Any, instructions: str, context: Optional[Dict[str, Any]] = None) -> AIResponse:
        """Edits an existing image according to `instructions`.

        Args:
            image: Opaque provider-agnostic image data (e.g. canvas
                pixels or an encoded image). Never inspected by
                `AIManager` or this base class.
            instructions: Natural-language description of the edit.
            context: Optional arbitrary caller-supplied context.

        Returns:
            The completed `AIResponse`.

        Raises:
            NotImplementedError: If this provider does not support
                `AICapability.IMAGE_EDIT`.
        """
        raise NotImplementedError(f"Provider '{self._name}' does not support image editing.")

    def generate_variations(
        self, image: Any, count: int = 1, context: Optional[Dict[str, Any]] = None
    ) -> AIResponse:
        """Generates variations of an existing image.

        Args:
            image: Opaque provider-agnostic image data.
            count: Number of variations requested.
            context: Optional arbitrary caller-supplied context.

        Returns:
            The completed `AIResponse`.

        Raises:
            NotImplementedError: If this provider does not support
                generating image variations.
        """
        raise NotImplementedError(f"Provider '{self._name}' does not support image variations.")

    def analyze_sketch(self, sketch: Any) -> Dict[str, Any]:
        """Extracts structured information from a user's sketch.

        This is a provider-backed alternative to (or complement of) the
        `SketchAnalyzer` collaborator `AIManager` already supports -
        useful for providers whose sketch understanding is itself a
        model call rather than local analysis.

        Args:
            sketch: Opaque sketch data (e.g. canvas pixels).

        Returns:
            Structured data describing the sketch.

        Raises:
            NotImplementedError: If this provider does not support
                sketch understanding.
        """
        raise NotImplementedError(f"Provider '{self._name}' does not support sketch understanding.")

    def optimize_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Rewrites or enriches a prompt to improve generation quality.

        A provider-backed alternative to (or complement of) the
        `PromptBuilder` collaborator `AIManager` already supports -
        useful for providers that offer their own prompt-optimization
        model or endpoint.

        Args:
            prompt: The prompt text to optimize.
            context: Optional arbitrary caller-supplied context.

        Returns:
            The optimized prompt text.

        Raises:
            NotImplementedError: If this provider does not support
                prompt optimization.
        """
        raise NotImplementedError(f"Provider '{self._name}' does not support prompt optimization.")

    def generate_3d_model(
        self, source: Any, capability: AICapability = AICapability.MODEL_3D_GENERATION,
        context: Optional[Dict[str, Any]] = None,
    ) -> AIResponse:
        """Generates a 3D model from `source` (image-to-3D or text-to-3D).

        `source` is deliberately opaque and `capability` deliberately
        parameterized: today this covers image-to-3D via
        `AICapability.MODEL_3D_GENERATION`; a future text-to-3D
        capability can reuse this same hook by declaring a new
        `AICapability` member and passing it here, with no change to
        this base class.

        Args:
            source: Opaque provider-agnostic source data (e.g. an image
                for image-to-3D, or text for a future text-to-3D path).
            capability: The specific 3D-generation capability being
                requested.
            context: Optional arbitrary caller-supplied context.

        Returns:
            The completed `AIResponse`.

        Raises:
            NotImplementedError: If this provider does not support
                `capability`.
        """
        raise NotImplementedError(f"Provider '{self._name}' does not support 3D model generation.")

    # --- Health -----------------------------------------------------------

    def check_health(self) -> ProviderHealth:
        """Checks whether this provider is currently able to service requests.

        The default implementation reports `HEALTHY` unconditionally -
        a provider with no meaningful health signal (no network calls,
        no external dependency) is healthy by definition. A concrete
        provider backed by a remote API should override this to perform
        a lightweight reachability/auth check and report `DEGRADED` or
        `UNAVAILABLE` accordingly, so a future caller (e.g. provider
        selection in `AIManager`) can route around an unhealthy
        provider without first failing a real generation request.

        Returns:
            The current `ProviderHealth`.
        """
        return ProviderHealth(status=ProviderHealthStatus.HEALTHY)