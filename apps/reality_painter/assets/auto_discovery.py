"""Automatic multi-repository asset discovery for Reality Painter (Block 11C, Phase 1).

Connects the existing GitHub discovery module
(`apps.reality_painter.assets.github`, Phase 12B) to a configurable
list of asset repositories, so `registry.json` no longer needs every
object entered by hand. This module introduces no second discovery
implementation, no second registry, and no second retrieval path - it
only orchestrates `github.ingest_repository()` and
`AssetRegistry`/`AssetRegistry.save()`, both unmodified in behavior,
across a list of configured repositories.

FLOW: configured_repositories() -> github.ingest_repository() (per
repository, skipped if already represented in the registry) ->
AssetRegistry.save() (only if something new was actually discovered).

Repeated calls are cheap: a repository already represented in the
registry is never re-scanned (no GitHub API call at all), and
`AssetRegistry.save()` is only invoked when at least one asset was
newly added - so a steady-state call across an already-fully-scanned
repository list touches neither the network nor disk.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from apps.reality_painter.assets import github
from apps.reality_painter.assets.registry import AssetRegistry

logger = logging.getLogger(__name__)

_PRIMARY_ENV_VAR = "REALITY_PAINTER_PRIMARY_REPOSITORY"
_EXTERNAL_ENV_VAR = "REALITY_PAINTER_EXTERNAL_REPOSITORIES"
_REPOSITORIES_ENV_VAR = "REALITY_PAINTER_ASSET_REPOSITORIES"

_DEFAULT_PRIMARY_REPOSITORY: Dict[str, str] = {
    "repository": "omkarmusle510-web/reality-engine-assets",
    "path": "",
}

_DEFAULT_EXTERNAL_REPOSITORIES: List[Dict[str, str]] = [
    {"repository": "KhronosGroup/glTF-Sample-Assets", "path": ""},
]

_DEFAULT_REPOSITORIES: List[Dict[str, str]] = [
    _DEFAULT_PRIMARY_REPOSITORY,
    *_DEFAULT_EXTERNAL_REPOSITORIES,
]

_SCANNED_REPOSITORIES: set[str] = set()


def reset_discovery_state() -> None:
    """Resets in-memory scanned repository tracking (for test isolation)."""
    _SCANNED_REPOSITORIES.clear()


def primary_repository() -> Dict[str, str]:
    """Returns the configured primary asset repository (always first in discovery)."""
    raw = os.environ.get(_PRIMARY_ENV_VAR)
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("repository"), str) and parsed["repository"].strip():
                return {"repository": parsed["repository"].strip(), "path": parsed.get("path", "")}
            if isinstance(parsed, str) and parsed.strip():
                return {"repository": parsed.strip(), "path": ""}
        except ValueError:
            if raw.strip():
                return {"repository": raw.strip(), "path": ""}
    return dict(_DEFAULT_PRIMARY_REPOSITORY)


def external_repositories() -> List[Dict[str, str]]:
    """Returns the configured external fallback asset repositories."""
    raw = os.environ.get(_EXTERNAL_ENV_VAR)
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                repos: List[Dict[str, str]] = []
                for entry in parsed:
                    if isinstance(entry, dict) and isinstance(entry.get("repository"), str) and entry["repository"].strip():
                        repos.append({"repository": entry["repository"].strip(), "path": entry.get("path", "")})
                    elif isinstance(entry, str) and entry.strip():
                        repos.append({"repository": entry.strip(), "path": ""})
                if repos:
                    return repos
        except ValueError:
            logger.warning("%s is not valid JSON; using default external repositories.", _EXTERNAL_ENV_VAR)

    return [dict(repo) for repo in _DEFAULT_EXTERNAL_REPOSITORIES]


def configured_repositories() -> List[Dict[str, str]]:
    """Returns the ordered list of asset repositories to auto-discover.

    Primary repository is always first, followed by trusted external
    fallback repositories. Reads from environment variables if set,
    falling back to defaults.
    """
    raw = os.environ.get(_REPOSITORIES_ENV_VAR)
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                repositories: List[Dict[str, str]] = []
                for entry in parsed:
                    if isinstance(entry, dict) and isinstance(entry.get("repository"), str) and entry["repository"].strip():
                        repositories.append({"repository": entry["repository"].strip(), "path": entry.get("path", "")})
                    elif isinstance(entry, str) and entry.strip():
                        repositories.append({"repository": entry.strip(), "path": ""})
                if repositories:
                    return repositories
        except ValueError:
            logger.warning("%s is not valid JSON; using default asset repositories.", _REPOSITORIES_ENV_VAR)

    primary = primary_repository()
    external = external_repositories()
    
    # Ensure primary is first and deduplicated
    combined = [primary]
    for ext in external:
        if ext["repository"] != primary["repository"]:
            combined.append(ext)

    return combined


def _already_scanned(registry: Optional[AssetRegistry], repository: str) -> bool:
    """True if `repository` has already been scanned into `registry`."""
    if registry is None:
        return repository in _SCANNED_REPOSITORIES
    scanned = getattr(registry, "_scanned_repositories", None)
    if scanned is None:
        registry._scanned_repositories = set()
        return False
    return repository in scanned


def ensure_discovered(
    registry: AssetRegistry,
    repositories: Optional[List[Dict[str, str]]] = None,
    session: Optional[Any] = None,
    registry_path: Optional[Path] = None,
) -> int:
    """Scans every not-yet-scanned repository in `repositories` into `registry`.

    Thin orchestration over `github.ingest_repository()` and
    `AssetRegistry.save()` - no discovery or retrieval logic is
    duplicated. A repository already scanned in the active session
    is skipped without any network call. Failing repositories are
    isolated and logged rather than raised.

    Args:
        registry: The `AssetRegistry` to discover into and persist.
        repositories: Repositories to scan. Defaults to
            `configured_repositories()`.
        session: Forwarded to `github.ingest_repository` as its
            `session` (a `requests`-compatible object).
        registry_path: Where to persist newly discovered assets.
            Defaults to the bundled `registry.json` if omitted.

    Returns:
        The number of newly added assets across every scanned
        repository.
    """
    active_repositories = repositories if repositories is not None else configured_repositories()

    if not hasattr(registry, "_scanned_repositories"):
        registry._scanned_repositories = set()

    total_added = 0
    for entry in active_repositories:
        repository = entry["repository"]
        if _already_scanned(registry, repository):
            continue

        try:
            added, _skipped = github.ingest_repository(
                registry, repository=repository, path=entry.get("path", ""), session=session
            )
        except github.GitHubSourceError as exc:
            logger.warning("Auto-discovery skipped repository '%s': %s", repository, exc)
            continue
        except Exception as exc:
            logger.warning("Auto-discovery encountered unexpected error for '%s': %s", repository, exc)
            continue

        _SCANNED_REPOSITORIES.add(repository)
        registry._scanned_repositories.add(repository)
        total_added += added

    if total_added > 0:
        registry.save(registry_path)

    return total_added

