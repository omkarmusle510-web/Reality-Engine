"""Recognized-label -> registered-asset resolution for Reality Painter.

`resolve_asset` is the sole bridge between a recognition label (e.g.
"flower") and the existing `AssetRegistry` (Phase 12A/12B) - it
performs no retrieval and no caching itself; it only decides whether a
registered asset exists for a label, using the registry's own existing,
deterministic `search_assets` lookup. It never invents an asset for an
unmatched label, and it never introduces AI/embedding-based or fuzzy
matching beyond what `AssetRegistry` already provides.

Block 11C Phase 2 adds exactly one fallback, in this order:
    1. Already-registered asset (Phase 1's normalized-label lookup,
       unchanged).
    2. On a miss, run the existing Phase 1 auto-discovery
       (`apps.reality_painter.assets.auto_discovery.ensure_discovered`)
       across the configured GitHub repositories, and retry the same
       normalized-label lookup exactly once more.
No second discovery implementation, registry, or matching algorithm is
introduced - this only calls the existing Phase 1 orchestration
(itself unmodified) on a miss. `ensure_discovered` already isolates a
failing repository (never raises `GitHubSourceError` past it) and
never re-scans a repository already represented in the registry, so a
label that remains unresolved after discovery triggers no repeated
network calls on subsequent lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

from apps.reality_painter.assets import auto_discovery
from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.schema import Asset


class AssetResolutionStatus(str, Enum):
    """Outcome of resolving a recognized label against the asset registry."""

    RESOLVED = "resolved"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class AssetResolution:
    """The result of resolving one recognized label.

    Attributes:
        status: Whether a matching registered asset was found.
        asset: The matched `Asset`, or `None` if `status` is
            `UNAVAILABLE`.
        label: The label that was resolved, carried through for
            logging/diagnostics.
    """

    status: AssetResolutionStatus
    asset: Optional[Asset]
    label: str


def _normalized_label_candidates(label: str) -> List[str]:
    """Generates deterministic label variants to try against the registry.

    Broadest-first: the raw (lowercased, stripped) label, then
    progressively shorter word-suffixes of it - e.g. "red flower" ->
    ["red flower", "flower"] - so a descriptive recognition label
    still resolves against a registry entry named/tagged with just its
    final noun. Purely string manipulation: no NLP model, no stemming,
    no embeddings. A single-word label (e.g. "car") yields exactly one
    candidate, identical to the label itself - existing single-word
    resolution behavior is unchanged.

    Args:
        label: The raw recognized label.

    Returns:
        Ordered, deduplicated candidate strings to try, most specific
        first. Empty if `label` is empty/whitespace-only.
    """
    normalized = label.strip().lower()
    words = normalized.split()
    candidates = [normalized] if normalized else []
    for start in range(1, len(words)):
        candidate = " ".join(words[start:])
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _search_candidates(label: str, registry: AssetRegistry) -> Optional[Asset]:
    """Tries each normalized candidate against `registry.search_assets`, first match wins."""
    for candidate in _normalized_label_candidates(label):
        matches = registry.search_assets(candidate)
        if matches:
            return matches[0]
    return None


def resolve_asset(
    label: str,
    registry: AssetRegistry,
    session: Optional[Any] = None,
    registry_path: Optional[Path] = None,
) -> AssetResolution:
    """Resolves a recognized label to a registered asset, if one exists.

    Deterministic: tries each of `_normalized_label_candidates(label)`,
    most specific first, against the registry's own existing
    `search_assets` (case-insensitive substring match against
    name/tags) - no AI/embedding-based or fuzzy matching is introduced
    here. The first candidate that yields any match wins, and its
    first match (in the registry's own `id`-sorted order) is used, so
    a repeated call with the same label and registry state always
    resolves to the same asset. A single-word label behaves exactly as
    before, since it produces exactly one candidate.

    If nothing matches yet, the existing Phase 1
    `auto_discovery.ensure_discovered` is run once (scanning every
    configured repository not already represented in `registry`,
    isolating any repository that fails - see that function's own
    docstring) and the same normalized-label lookup is retried exactly
    once more. This never raises for a discovery failure: an
    unreachable/rate-limited/empty scan simply leaves `registry`
    unchanged and resolution falls through to `UNAVAILABLE`, same as
    before this phase.

    Args:
        label: A recognized object label (e.g. "flower", or a more
            descriptive "red flower").
        registry: The `AssetRegistry` to search (and, on a miss,
            discover newly available assets into via `ensure_discovered`).
        session: Optional `requests`-compatible session forwarded to
            `ensure_discovered`/`github.ingest_repository`, so tests can
            inject a fake session and no real HTTP call is required.
            Defaults to a real network call.
        registry_path: Forwarded to `ensure_discovered` for where a
            registry with newly discovered assets is persisted.
            Defaults to the bundled `registry.json`.

    Returns:
        `AssetResolution(status=RESOLVED, asset=...)` if any candidate
        matched (before or after discovery), or
        `AssetResolution(status=UNAVAILABLE, asset=None)` otherwise.
        Never raises for "no match" - that is an expected,
        non-exceptional outcome.
    """
    match = _search_candidates(label, registry)
    if match is not None:
        return AssetResolution(status=AssetResolutionStatus.RESOLVED, asset=match, label=label)

    # Fallback 1: Primary repository discovery
    primary = auto_discovery.primary_repository()
    auto_discovery.ensure_discovered(
        registry,
        repositories=[primary],
        session=session,
        registry_path=registry_path,
    )
    match = _search_candidates(label, registry)
    if match is not None:
        return AssetResolution(status=AssetResolutionStatus.RESOLVED, asset=match, label=label)

    # Fallback 2: Configured external repositories discovery (in order)
    external_repos = auto_discovery.external_repositories()
    for repo in external_repos:
        if repo.get("repository") == primary.get("repository"):
            continue
        auto_discovery.ensure_discovered(
            registry,
            repositories=[repo],
            session=session,
            registry_path=registry_path,
        )
        match = _search_candidates(label, registry)
        if match is not None:
            return AssetResolution(status=AssetResolutionStatus.RESOLVED, asset=match, label=label)

    return AssetResolution(status=AssetResolutionStatus.UNAVAILABLE, asset=None, label=label)

