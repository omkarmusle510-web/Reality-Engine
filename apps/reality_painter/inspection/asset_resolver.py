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
from typing import Optional

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


def resolve_asset(label: str, registry: AssetRegistry) -> AssetResolution:
    """Resolves a recognized label to a registered asset, if one exists.

    Deterministic: delegates entirely to the registry's own
    `search_assets` (case-insensitive substring match against
    name/tags) - no new or fuzzier matching logic is introduced here.
    The first match, in the registry's own `id`-sorted order, is used,
    so a repeated call with the same label and registry state always
    resolves to the same asset.

    Args:
        label: A recognized object label (e.g. "flower").
        registry: The `AssetRegistry` to search.

    Returns:
        `AssetResolution(status=RESOLVED, asset=...)` if a match was
        found, or `AssetResolution(status=UNAVAILABLE, asset=None)`
        otherwise. Never raises for "no match" - that is an expected,
        non-exceptional outcome.
    """
    matches = registry.search_assets(label)
    if not matches:
        return AssetResolution(status=AssetResolutionStatus.UNAVAILABLE, asset=None, label=label)
    return AssetResolution(status=AssetResolutionStatus.RESOLVED, asset=matches[0], label=label)
