"""Safe repository promotion and GitHub publishing for Reality Painter assets.

Defines promotion eligibility, license-verification rules, and GitHub
publishing for promoting externally sourced assets into our primary
repository (`omkarmusle510-web/reality-engine-assets`).

CRITICAL POLICY:
- Never automatically publish or copy a third-party asset unless its
  license explicitly permits redistribution.
- If license information is missing, None, or unclear, do not promote
  and do not invent a license.
- Reuses existing environment credentials (`GITHUB_TOKEN`) without
  creating a separate auth mechanism or exposing tokens.
- Only promote assets that are actually downloaded/used.
- Prevents duplicate promotions.

PROMOTION FLOW:
  external repository
  → asset downloaded/used (AssetRetriever)
  → verify redistribution license (validate_asset_for_promotion)
  → prevent duplicate promotion (is_already_promoted)
  → stage locally in mirrored structure (category/filename.glb)
  → upload/commit to our GitHub repository via GitHub Contents API
  → record metadata in promotion log (promotions.json)
  → register promoted Asset in AssetRegistry
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

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

# Local staging directory for promoted assets
_DEFAULT_STAGING_DIR = Path(__file__).parent / "promoted"

# Metadata file tracking all promotions
_DEFAULT_PROMOTION_LOG = Path(__file__).parent / "promoted" / "promotions.json"

_API_BASE_URL = "https://api.github.com"
_DEFAULT_TIMEOUT_SECONDS = 30.0


class PromotionLicenseError(ValueError):
    """Raised when an asset cannot be safely promoted due to license restrictions."""


class PromotionError(Exception):
    """Raised when promotion fails for a non-license reason (e.g. missing cache file)."""


class GitHubUploadError(PromotionError):
    """Raised when publishing a file to GitHub fails."""


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


def upload_file_to_github(
    repository: str,
    target_path: str,
    file_bytes: bytes,
    commit_message: str,
    session: Optional[Any] = None,
    branch: Optional[str] = None,
) -> Dict[str, Any]:
    """Uploads/commits a file to a GitHub repository via the GitHub Contents API.

    Endpoint: `PUT /repos/{owner}/{repo}/contents/{path}`
    Uses `GITHUB_TOKEN` from the environment if present. Never logs or prints tokens.

    Args:
        repository: `"owner/name"` target repository.
        target_path: Relative path in the repo (e.g. `"nature/tree.glb"`).
        file_bytes: Raw binary content of the file.
        commit_message: Git commit message.
        session: Optional `requests`-compatible session (used for mock testing).
        branch: Target branch name (defaults to repository's default branch if omitted).

    Returns:
        The parsed GitHub API JSON response.

    Raises:
        GitHubUploadError: If the upload is rejected or encounters a network error.
    """
    owner, separator, repo_name = repository.partition("/")
    if not separator or not owner.strip() or not repo_name.strip():
        raise PromotionError(f"Malformed repository name: {repository!r}")

    http = session if session is not None else requests
    url = f"{_API_BASE_URL}/repos/{owner.strip()}/{repo_name.strip()}/contents/{target_path}"

    token = os.getenv("GITHUB_TOKEN")
    headers: Dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Check if file already exists in remote repo to retrieve sha (needed for update)
    params = {"ref": branch} if branch else {}
    existing_sha: Optional[str] = None
    try:
        get_resp = http.get(url, headers=headers, params=params, timeout=_DEFAULT_TIMEOUT_SECONDS)
        if get_resp.status_code == 200:
            try:
                data = get_resp.json()
                if isinstance(data, dict) and "sha" in data:
                    existing_sha = data["sha"]
            except Exception:
                pass
    except Exception as exc:
        logger.debug("Failed to check existing file sha: %s", exc)

    encoded_content = base64.b64encode(file_bytes).decode("utf-8")
    body: Dict[str, Any] = {
        "message": commit_message,
        "content": encoded_content,
    }
    if branch:
        body["branch"] = branch
    if existing_sha:
        body["sha"] = existing_sha

    try:
        response = http.put(url, headers=headers, json=body, timeout=_DEFAULT_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as exc:
        raise GitHubUploadError(
            f"Network error during GitHub upload for '{target_path}': {type(exc).__name__}"
        ) from exc

    if response.status_code in (200, 201):
        try:
            return response.json()
        except Exception:
            return {"status": "ok", "code": response.status_code}

    if response.status_code == 401:
        raise GitHubUploadError("GitHub upload unauthorized. Check GITHUB_TOKEN permissions.")
    if response.status_code == 403:
        raise GitHubUploadError("GitHub upload forbidden. Token lacks write permissions to target repository.")
    if response.status_code == 404:
        raise GitHubUploadError(
            f"GitHub repository '{repository}' not found or not writable with current credentials."
        )

    error_detail = ""
    try:
        error_detail = response.json().get("message", "")
    except Exception:
        error_detail = f"HTTP {response.status_code}"

    raise GitHubUploadError(f"GitHub upload failed ({error_detail}).")


def promote_asset(
    asset: Asset,
    cached_file_path: Path,
    staging_dir: Optional[Path] = None,
    log_path: Optional[Path] = None,
    primary_repository: Optional[str] = None,
    push_to_remote: bool = True,
    session: Optional[Any] = None,
    branch: Optional[str] = None,
    registry: Optional[Any] = None,
    registry_path: Optional[Path] = None,
) -> Asset:
    """Promotes an externally sourced, cached asset into our primary repository.

    FLOW:
      1. Validate license permits redistribution (raises PromotionLicenseError if not).
      2. Check if asset hasn't already been promoted (prevents duplicate promotion).
      3. Stage model file locally mirroring primary repo structure (category/filename.glb).
      4. Upload file to our GitHub repository using GitHub Contents API (if push_to_remote).
      5. Record promotion metadata in persistent log (promotions.json).
      6. If registry is provided, register promoted Asset and save registry.
      7. Return updated Asset pointing to primary repository.

    Args:
        asset: The externally discovered Asset to promote.
        cached_file_path: Local path to the already-downloaded model file.
        staging_dir: Directory to stage promoted assets into. Defaults to
            `apps/reality_painter/assets/promoted/`.
        log_path: Path to the promotion log file. Defaults to
            `promoted/promotions.json`.
        primary_repository: The target repository name. Defaults to
            `omkarmusle510-web/reality-engine-assets`.
        push_to_remote: Whether to upload to GitHub via API (default: True).
        session: Optional HTTP session for mock testing.
        branch: Target GitHub branch (optional).
        registry: Optional AssetRegistry to register promoted asset into.
        registry_path: Optional path to save registry to.

    Returns:
        A new `Asset` with source updated to point to the primary repository.

    Raises:
        PromotionLicenseError: If the asset's license doesn't permit redistribution.
        PromotionError / GitHubUploadError: If file is missing or GitHub upload fails.
    """
    validate_asset_for_promotion(asset)

    stage = staging_dir or _DEFAULT_STAGING_DIR
    log = log_path or _DEFAULT_PROMOTION_LOG
    repo = primary_repository or _DEFAULT_PRIMARY_REPOSITORY

    if is_already_promoted(asset, log):
        logger.info("Asset '%s' already promoted; skipping duplicate.", asset.id)
        promoted_path = _infer_promoted_path(asset)
        promoted_asset = _build_promoted_asset(asset, repo, promoted_path)
        if registry is not None:
            registry.register(promoted_asset)
        return promoted_asset

    if not cached_file_path.is_file():
        raise PromotionError(
            f"Cannot promote asset '{asset.id}': cached file not found at '{cached_file_path}'."
        )

    file_bytes = cached_file_path.read_bytes()
    if len(file_bytes) == 0:
        raise PromotionError(
            f"Cannot promote asset '{asset.id}': cached file at '{cached_file_path}' is empty."
        )

    promoted_path = _infer_promoted_path(asset)
    dest = stage / promoted_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(cached_file_path), str(dest))

    uploaded = False
    if push_to_remote:
        commit_message = (
            f"Promote {asset.name} ({asset.id}) from {asset.source.details.get('repository', 'external')}"
        )
        upload_file_to_github(
            repository=repo,
            target_path=promoted_path,
            file_bytes=file_bytes,
            commit_message=commit_message,
            session=session,
            branch=branch,
        )
        uploaded = True

    promoted_asset = _build_promoted_asset(asset, repo, promoted_path)

    log_data = _load_promotion_log(log)
    log_data["promoted"].append({
        "original_id": asset.id,
        "original_source": asset.source.to_dict(),
        "promoted_id": promoted_asset.id,
        "promoted_path": promoted_path,
        "promoted_repository": repo,
        "license": asset.license,
        "name": asset.name,
        "category": asset.category,
        "format": asset.format,
        "tags": list(asset.tags),
        "uploaded_to_github": uploaded,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_promotion_log(log, log_data)

    if registry is not None:
        registry.register(promoted_asset)
        if registry_path is not None:
            registry.save(registry_path)

    logger.info(
        "Promoted asset '%s' (license: %s) → %s/%s (uploaded=%s)",
        asset.id,
        asset.license,
        repo,
        promoted_path,
        uploaded,
    )

    return promoted_asset


def maybe_promote_asset(
    asset: Asset,
    cached_file_path: Path,
    registry: Optional[Any] = None,
    registry_path: Optional[Path] = None,
    staging_dir: Optional[Path] = None,
    log_path: Optional[Path] = None,
    primary_repository: Optional[str] = None,
    push_to_remote: bool = True,
    session: Optional[Any] = None,
) -> Optional[Asset]:
    """Promotes `asset` if it comes from an external repository and is eligible.

    Safe helper for automatic promotion on asset use:
    - No-op if asset is already from our primary repository.
    - No-op if asset lacks permissive redistribution license.
    - No-op if asset is already promoted.
    - Promotes and returns updated `Asset` if eligible.
    - Never raises: logs failures and returns None so caller is never blocked.
    """
    repo = primary_repository or _DEFAULT_PRIMARY_REPOSITORY
    if asset.source.type == "github" and asset.source.details.get("repository") == repo:
        return None

    if not is_promotable(asset):
        logger.info(
            "Asset '%s' from '%s' not promoted (license '%s' does not permit redistribution).",
            asset.id,
            asset.source.details.get("repository"),
            asset.license,
        )
        return None

    if is_already_promoted(asset, log_path=log_path):
        return None

    try:
        return promote_asset(
            asset=asset,
            cached_file_path=cached_file_path,
            staging_dir=staging_dir,
            log_path=log_path,
            primary_repository=repo,
            push_to_remote=push_to_remote,
            session=session,
            registry=registry,
            registry_path=registry_path,
        )
    except Exception as exc:
        logger.warning("Automatic promotion failed for asset '%s': %s", asset.id, exc)
        return None


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
