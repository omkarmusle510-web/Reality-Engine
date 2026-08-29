"""Safe repository promotion policies and local staging for Reality Painter assets.

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

PROMOTION FLOW:
  external repository
  → existing local cache (AssetRetriever)
  → promotion staging directory (this module)
  → updated Asset with source pointing to our primary repository
  → AssetRegistry

The staging directory mirrors the directory structure of our primary
repository (category/filename.glb) so staged assets are ready for
eventual push to omkarmusle510-web/reality-engine-assets.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from apps.reality_painter.assets.schema import Asset, AssetSource

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

# Default primary repository for promoted assets
_DEFAULT_PRIMARY_REPOSITORY = "omkarmusle510-web/reality-engine-assets"

# Staging directory for promoted assets (sits alongside the existing cache)
_DEFAULT_STAGING_DIR = Path(__file__).parent / "promoted"

# Metadata file tracking all promotions
_DEFAULT_PROMOTION_LOG = Path(__file__).parent / "promoted" / "promotions.json"


class PromotionLicenseError(ValueError):
    """Raised when an asset cannot be safely promoted due to license restrictions."""


class PromotionError(Exception):
    """Raised when promotion fails for a non-license reason (e.g. missing cache file)."""


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


def _safe_filename(name: str) -> str:
    """Converts an asset name to a filesystem-safe filename."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.lower()).strip("_")
    return safe or "asset"


def _infer_promoted_path(asset: Asset) -> str:
    """Determines the target path within the primary repository for a promoted asset.

    Uses the asset's category as the directory and the asset name as filename,
    mirroring the structure of omkarmusle510-web/reality-engine-assets
    (e.g. 'nature/sun.glb', 'vehicles/car.glb').
    """
    category = asset.category if asset.category != "unknown" else "objects"
    filename = _safe_filename(asset.name) + f".{asset.format}"
    return f"{category}/{filename}"


def _load_promotion_log(log_path: Path) -> Dict[str, Any]:
    """Loads the promotion log, returning an empty structure if absent."""
    if log_path.is_file():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("promoted"), list):
                    return data
        except (ValueError, OSError):
            pass
    return {"promoted": []}


def _save_promotion_log(log_path: Path, data: Dict[str, Any]) -> None:
    """Persists the promotion log to disk."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def is_already_promoted(
    asset: Asset,
    log_path: Optional[Path] = None,
) -> bool:
    """Returns True if `asset` has already been promoted (prevents duplicate promotion)."""
    path = log_path or _DEFAULT_PROMOTION_LOG
    log = _load_promotion_log(path)
    return any(entry.get("original_id") == asset.id for entry in log["promoted"])


def promote_asset(
    asset: Asset,
    cached_file_path: Path,
    staging_dir: Optional[Path] = None,
    log_path: Optional[Path] = None,
    primary_repository: Optional[str] = None,
) -> Asset:
    """Promotes an externally sourced, cached asset into our primary repository staging.

    FLOW:
      1. Validate license permits redistribution (raises PromotionLicenseError if not).
      2. Check the asset hasn't already been promoted (prevents duplicates).
      3. Copy the cached model file into the staging directory, structured
         to mirror the primary repository (category/filename.glb).
      4. Record the promotion in a persistent log with source and license metadata.
      5. Return a new Asset with its source pointing to the primary repository,
         so future lookups resolve from our own repository rather than the external one.

    Args:
        asset: The externally discovered Asset to promote.
        cached_file_path: Local path to the already-downloaded model file
            (from AssetRetriever's cache).
        staging_dir: Directory to stage promoted assets into. Defaults to
            `apps/reality_painter/assets/promoted/`.
        log_path: Path to the promotion log file. Defaults to
            `promoted/promotions.json`.
        primary_repository: The target repository name. Defaults to
            `omkarmusle510-web/reality-engine-assets`.

    Returns:
        A new `Asset` with source updated to point to the primary repository.

    Raises:
        PromotionLicenseError: If the asset's license doesn't permit redistribution.
        PromotionError: If the cached file doesn't exist or promotion fails.
    """
    validate_asset_for_promotion(asset)

    stage = staging_dir or _DEFAULT_STAGING_DIR
    log = log_path or _DEFAULT_PROMOTION_LOG
    repo = primary_repository or _DEFAULT_PRIMARY_REPOSITORY

    if is_already_promoted(asset, log):
        logger.info("Asset '%s' already promoted; skipping duplicate.", asset.id)
        promoted_path = _infer_promoted_path(asset)
        return _build_promoted_asset(asset, repo, promoted_path)

    if not cached_file_path.is_file():
        raise PromotionError(
            f"Cannot promote asset '{asset.id}': cached file not found at '{cached_file_path}'."
        )

    promoted_path = _infer_promoted_path(asset)
    dest = stage / promoted_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(cached_file_path), str(dest))

    log_data = _load_promotion_log(log)
    log_data["promoted"].append({
        "original_id": asset.id,
        "original_source": asset.source.to_dict(),
        "promoted_path": promoted_path,
        "promoted_repository": repo,
        "license": asset.license,
        "name": asset.name,
        "category": asset.category,
        "format": asset.format,
        "tags": list(asset.tags),
    })
    _save_promotion_log(log, log_data)

    logger.info(
        "Promoted asset '%s' (license: %s) → %s/%s",
        asset.id,
        asset.license,
        repo,
        promoted_path,
    )

    return _build_promoted_asset(asset, repo, promoted_path)


def _build_promoted_asset(asset: Asset, repository: str, promoted_path: str) -> Asset:
    """Builds a new Asset with source pointing to the primary repository."""
    promoted_id = re.sub(r"[^a-z0-9]+", "_", f"github_{repository}_{promoted_path}".lower()).strip("_")
    return Asset(
        id=promoted_id,
        name=asset.name,
        category=asset.category,
        format=asset.format,
        source=AssetSource(type="github", details={"repository": repository, "path": promoted_path}),
        tags=asset.tags,
        license=asset.license,
    )


def list_promoted_assets(
    log_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Returns a list of all promoted asset entries from the promotion log."""
    path = log_path or _DEFAULT_PROMOTION_LOG
    log = _load_promotion_log(path)
    return log["promoted"]
