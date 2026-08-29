"""Safe repository promotion policies for Reality Painter assets (Block 11C, Phase 2).

Defines promotion eligibility and license-verification rules for
promoting externally sourced assets into our primary repository
(`omkarmusle510-web/reality-engine-assets`).

CRITICAL POLICY:
- Never automatically publish or copy a third-party asset unless its
  license explicitly permits redistribution.
- If license information is missing, None, or unclear, do not promote
  and do not invent a license.
- Reuses existing environment credentials (e.g. GITHUB_TOKEN) without
  creating a separate auth mechanism.
"""

from __future__ import annotations

import logging
from typing import Optional, Set, Tuple

from apps.reality_painter.assets.schema import Asset

logger = logging.getLogger(__name__)

# Standard SPDX identifiers that explicitly permit redistribution and reuse
PERMISSIVE_SPDX_LICENSES: Set[str] = {
    "cc0-1.0",
    "cc0",
    "cc-by-4.0",
    "cc-by-3.0",
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "unlicense",
    "0bsd",
    "isc",
}


class PromotionLicenseError(ValueError):
    """Raised when an asset cannot be safely promoted due to license restrictions."""


def is_promotable(asset: Asset) -> bool:
    """Checks whether `asset` has an explicit license permitting redistribution.

    Returns False if `asset.license` is None, empty, unknown, or
    non-permissive. Never guesses or invents a license.
    """
    if not asset.license or not isinstance(asset.license, str):
        return False

    normalized_license = asset.license.strip().lower()
    return normalized_license in PERMISSIVE_SPDX_LICENSES


def check_promotion_eligibility(asset: Asset) -> Tuple[bool, str]:
    """Validates whether `asset` can be safely promoted to our primary repository.

    Returns:
        `(True, reason)` if eligible under a permissive license, or
        `(False, reason)` explaining why promotion is rejected.
    """
    if not asset.license or not isinstance(asset.license, str):
        return (
            False,
            f"Asset '{asset.id}' cannot be promoted: license metadata is missing or unspecified.",
        )

    normalized_license = asset.license.strip().lower()
    if normalized_license in PERMISSIVE_SPDX_LICENSES:
        return (
            True,
            f"Asset '{asset.id}' is eligible for promotion under permissive license '{asset.license}'.",
        )

    return (
        False,
        f"Asset '{asset.id}' cannot be promoted: license '{asset.license}' does not permit automated redistribution.",
    )


def validate_asset_for_promotion(asset: Asset) -> None:
    """Enforces safety rules for promoting `asset` to the primary repository.

    Raises:
        PromotionLicenseError: If the asset lacks an explicit permissive license.
    """
    eligible, reason = check_promotion_eligibility(asset)
    if not eligible:
        raise PromotionLicenseError(reason)

