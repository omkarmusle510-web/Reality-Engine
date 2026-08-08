"""In-memory response caching for Reality Painter's AI subsystem.

`InMemoryGenerationCache` satisfies the structural `GenerationCache`
Protocol already declared in `apps.reality_painter.ai.manager`
(`get(key) -> Optional[AIResponse]`, `set(key, response) -> None`), so
an instance can be passed directly to `AIManager(cache=...)` without
any change to that module.

This module performs no AI orchestration, no provider selection, no
prompt building, and no network or API calls of any kind - it only
stores and retrieves previously computed `AIResponse` objects by an
opaque string key. `AIManager` already derives that key itself (see
`AIManager._cache_key`) and treats it as an unstructured string; the
key-building helper here (`build_key`) is an optional convenience for
callers that want a deterministic, collision-resistant key from
several distinct pieces of request-defining information, rather than
`AIManager`'s simpler `capability:provider:prompt` scheme. Both are
valid strings from this cache's point of view - it never inspects or
parses a key.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from apps.reality_painter.ai.models import AIResponse


# --- Cache entry ------------------------------------------------------


@dataclass
class _CacheEntry:
    """One stored response plus its optional expiration time.

    Attributes:
        response: The cached `AIResponse`.
        expires_at: Wall-clock time (`time.time()`) after which this
            entry is considered expired, or `None` if it never expires.
    """

    response: AIResponse
    expires_at: Optional[float]

    def is_expired(self, now: float) -> bool:
        """True if `now` is at or past this entry's expiration time."""
        return self.expires_at is not None and now >= self.expires_at


# --- Cache --------------------------------------------------------------


class InMemoryGenerationCache:
    """A small, thread-safe, in-memory cache of `AIResponse`s by string key.

    Satisfies `apps.reality_painter.ai.manager.GenerationCache`
    structurally. Storage is a plain dict guarded by a lock, so
    concurrent `get`/`set` calls from multiple threads (e.g. future
    asynchronous generation tasks) never corrupt internal state.

    Entries may carry a per-entry TTL (time-to-live), set at write time
    via `set(..., ttl_seconds=...)`. Expired entries are treated as
    absent by `get`/`exists` and are lazily evicted the next time
    they're looked up or `purge_expired()` is called - there is no
    background sweep thread, keeping this implementation deliberately
    simple.

    Persistent caching (disk, Redis, a database, ...) can be added
    later as a separate class satisfying the same `GenerationCache`
    shape, without any change to `AIManager` or to callers of this
    class.
    """

    def __init__(self, default_ttl_seconds: Optional[float] = None) -> None:
        """Creates an empty cache.

        Args:
            default_ttl_seconds: TTL applied to entries whose `set()`
                call doesn't specify its own `ttl_seconds`. `None`
                (the default) means entries never expire unless a
                per-call TTL is given.
        """
        self._default_ttl_seconds = default_ttl_seconds
        self._entries: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    # --- Protocol entry points (apps.reality_painter.ai.manager.GenerationCache) ---

    def get(self, key: str) -> Optional[AIResponse]:
        """Returns the cached response for `key`, or `None` if absent/expired.

        Args:
            key: The cache key, as produced by `AIManager._cache_key`
                or `build_key`.
        """
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.is_expired(now):
                del self._entries[key]
                return None
            return entry.response

    def set(self, key: str, response: AIResponse, ttl_seconds: Optional[float] = None) -> None:
        """Stores `response` under `key`.

        Args:
            key: The cache key to store under.
            response: The `AIResponse` to cache.
            ttl_seconds: Per-entry TTL overriding
                `default_ttl_seconds` for this entry. `None` uses
                `default_ttl_seconds` (which may itself be `None`,
                meaning no expiration).
        """
        effective_ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        expires_at = time.time() + effective_ttl if effective_ttl is not None else None
        with self._lock:
            self._entries[key] = _CacheEntry(response=response, expires_at=expires_at)

    # --- Additional cache operations -----------------------------------

    def exists(self, key: str) -> bool:
        """True if `key` is present and not expired."""
        return self.get(key) is not None

    def remove(self, key: str) -> bool:
        """Removes `key` if present.

        Returns:
            True if an entry was removed, False if `key` wasn't cached.
        """
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        """Removes every cached entry."""
        with self._lock:
            self._entries.clear()

    def purge_expired(self) -> int:
        """Evicts all currently expired entries.

        Not required for correctness (expired entries are already
        treated as absent by `get`/`exists`) - useful for callers that
        want to bound memory use proactively rather than only on
        next-access.

        Returns:
            The number of entries removed.
        """
        now = time.time()
        with self._lock:
            expired_keys = [key for key, entry in self._entries.items() if entry.is_expired(now)]
            for key in expired_keys:
                del self._entries[key]
        return len(expired_keys)

    def __len__(self) -> int:
        """Number of entries currently stored, including any not-yet-purged expired ones."""
        with self._lock:
            return len(self._entries)

    # --- Deterministic key construction ---------------------------------

    @staticmethod
    def build_key(
        capability: str,
        prompt: str,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        reference_id: Optional[str] = None,
    ) -> str:
        """Builds a deterministic, collision-resistant cache key.

        A SHA-256 digest over every piece of information that actually
        affects the result - never Python's built-in `hash()`, which
        is randomized per-process and unsuitable as a stable key.
        Components are joined with a separator unlikely to appear
        naturally and sorted where order is otherwise ambiguous
        (`parameters`), so semantically identical requests always
        produce the same key regardless of argument order or process.

        No API secrets are ever accepted or embedded here - only
        request-shaping values (capability, prompt/input, provider and
        model identity, generation parameters, and an optional
        reference/sketch identity) belong in a cache key.

        Args:
            capability: The `AICapability` value (as a string) being
                requested.
            prompt: The finished prompt/input text for this request.
            provider_name: The provider expected to service this
                request, if the result is provider-specific. Omitted
                (`None`) if the result is provider-agnostic.
            model_name: The specific model identity, if a provider
                exposes multiple models and results differ by model.
            parameters: Additional generation parameters that affect
                the result (e.g. size, seed, temperature). Serialized
                in sorted key order so key construction is
                order-independent.
            reference_id: A stable identifier for reference/sketch
                input (e.g. a hash of sketch pixels), if the request is
                keyed on more than just its prompt text.

        Returns:
            A hex-encoded SHA-256 digest string, safe to use directly
            as a `get`/`set` key.
        """
        parts = [
            f"capability={capability}",
            f"provider={provider_name or ''}",
            f"model={model_name or ''}",
            f"reference={reference_id or ''}",
            f"prompt={prompt}",
        ]
        if parameters:
            sorted_params = ",".join(f"{k}={parameters[k]}" for k in sorted(parameters))
            parts.append(f"params={sorted_params}")

        digest_input = "|".join(parts).encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()