"""Recognized-label -> registered-asset resolution for Reality Painter.

`resolve_asset` is the sole bridge between a recognition label (e.g.
"flower") and the existing `AssetRegistry` (Phase 12A/12B) - it
performs no network access, no retrieval, and no caching itself; it
only decides whether a registered asset already exists for a label,
using the registry's own existing, deterministic `search_assets`
lookup. It never invents an asset for an unmatched label, and it never
introduces AI/embedding-based or fuzzy matching beyond what
`AssetRegistry` already provides.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

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


def resolve_asset(label: str, registry: AssetRegistry) -> AssetResolution:
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

    Args:
        label: A recognized object label (e.g. "flower", or a more
            descriptive "red flower").
        registry: The `AssetRegistry` to search.

    Returns:
        `AssetResolution(status=RESOLVED, asset=...)` if any candidate
        matched, or `AssetResolution(status=UNAVAILABLE, asset=None)`
        otherwise. Never raises for "no match" - that is an expected,
        non-exceptional outcome.
    """
    for candidate in _normalized_label_candidates(label):
        matches = registry.search_assets(candidate)
        if matches:
            return AssetResolution(status=AssetResolutionStatus.RESOLVED, asset=matches[0], label=label)
    return AssetResolution(status=AssetResolutionStatus.UNAVAILABLE, asset=None, label=label)
