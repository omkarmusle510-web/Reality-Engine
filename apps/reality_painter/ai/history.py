"""In-memory generation history for Reality Painter's AI subsystem.

`InMemoryGenerationHistory` satisfies the structural `GenerationHistory`
Protocol already declared in `apps.reality_painter.ai.manager`
(`record(record) -> None`, `recent(limit) -> List[GenerationRecord]`),
so an instance can be passed directly to `AIManager(history=...)`
without any change to that module. It reuses `GenerationRecord` (and
transitively `AIRequest`/`AIResponse`) exactly as already defined in
`apps.reality_painter.ai.models` - no new record type is introduced
here.

This module performs no AI requests, no caching, no provider selection,
no prompt building, and knows nothing about any concrete provider - it
only stores and retrieves `GenerationRecord`s that `AIManager` has
already produced.

`AIManager` already contains a private, equivalent implementation
(`_InMemoryHistory`) used as its default when no `history` collaborator
is injected. This module exists to make an equivalent implementation
available as a public, reusable, independently-constructible class -
e.g. for a caller that wants to hold a reference to history outside
`AIManager`, or configure it (max size) before injection - without
duplicating `AIManager`'s internal default or changing its behavior.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, List, Optional

from apps.reality_painter.ai.models import GenerationRecord

_DEFAULT_MAX_ENTRIES = 100


class InMemoryGenerationHistory:
    """A small, thread-safe, in-memory, bounded history of `GenerationRecord`s.

    Satisfies `apps.reality_painter.ai.manager.GenerationHistory`
    structurally. Storage is a `collections.deque` with a fixed
    `maxlen`, guarded by a lock, so once `max_entries` is reached the
    oldest record is dropped automatically as new ones are added -
    chronological order is preserved without unbounded memory growth.

    Persistent history (disk, a database, ...) can be added later as a
    separate class satisfying the same `GenerationHistory` shape,
    without any change to `AIManager` or to callers of this class.
    """

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        """Creates an empty history.

        Args:
            max_entries: Maximum number of records retained. Once
                exceeded, the oldest record is dropped as each new one
                is added.
        """
        self._entries: Deque[GenerationRecord] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    # --- Protocol entry points (apps.reality_painter.ai.manager.GenerationHistory) ---

    def record(self, record: GenerationRecord) -> None:
        """Appends `record` to history, evicting the oldest if at capacity."""
        with self._lock:
            self._entries.append(record)

    def recent(self, limit: int) -> List[GenerationRecord]:
        """Returns up to `limit` most recent records, newest first.

        Args:
            limit: Maximum number of records to return. A `limit` of
                zero or less returns an empty list.
        """
        with self._lock:
            entries = list(self._entries)
        if limit <= 0:
            return []
        return entries[-limit:][::-1]

    # --- Additional history operations -----------------------------------

    def get_latest(self) -> Optional[GenerationRecord]:
        """Returns the single most recent record, or `None` if history is empty."""
        with self._lock:
            if not self._entries:
                return None
            return self._entries[-1]

    def get_by_request_id(self, request_id: str) -> Optional[GenerationRecord]:
        """Returns the record whose request has `request_id`, or `None`.

        Searches newest-first, so if a `request_id` were ever recorded
        more than once (not expected in normal operation, since
        `AIManager` generates a fresh UUID per request), the most
        recent match is returned.

        Args:
            request_id: The `AIRequest.request_id` to look up.
        """
        with self._lock:
            for entry in reversed(self._entries):
                if entry.request.request_id == request_id:
                    return entry
        return None

    def clear(self) -> None:
        """Removes every recorded entry."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        """Number of records currently retained."""
        with self._lock:
            return len(self._entries)