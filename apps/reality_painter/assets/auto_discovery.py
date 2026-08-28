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

_REPOSITORIES_ENV_VAR = "REALITY_PAINTER_ASSET_REPOSITORIES"

# Used only when REALITY_PAINTER_ASSET_REPOSITORIES is unset/empty/
# malformed - a default, not a hard-coded assumption baked into the
# discovery logic itself (see configured_repositories()). This is the
# same repository registry.json's existing hand-entered "flower" asset
# already points at, so automatic discovery is a strict superset of
# today's registry contents, never a behavior change for it.
_DEFAULT_REPOSITORIES: List[Dict[str, str]] = [
    {"repository": "omkarmusle510-web/reality-engine-assets", "path": ""},
]


def configured_repositories() -> List[Dict[str, str]]:
    """Returns the configured list of asset repositories to auto-discover.

    Reads a JSON array from the `REALITY_PAINTER_ASSET_REPOSITORIES`
    environment variable, e.g.:

        [{"repository": "owner/name", "path": "models"}, ...]

    so repository sources are configurable without editing code or
    hard-coding a single repository anywhere in the discovery path.
    Falls back to `_DEFAULT_REPOSITORIES` if the variable is unset,
    not valid JSON, not a list, or contains no usable entries; never
    raises.

    Returns:
        A list of `{"repository": ..., "path": ...}` mappings. `path`
        defaults to `""` (the whole repository) per entry if omitted.
    """
    raw = os.environ.get(_REPOSITORIES_ENV_VAR)
    if not raw:
        return list(_DEFAULT_REPOSITORIES)

    try:
        parsed = json.loads(raw)
    except ValueError:
        logger.warning("%s is not valid JSON; using default asset repositories.", _REPOSITORIES_ENV_VAR)
        return list(_DEFAULT_REPOSITORIES)

    if not isinstance(parsed, list):
        logger.warning("%s must be a JSON array; using default asset repositories.", _REPOSITORIES_ENV_VAR)
        return list(_DEFAULT_REPOSITORIES)

    repositories: List[Dict[str, str]] = []
    for entry in parsed:
        if isinstance(entry, dict) and isinstance(entry.get("repository"), str) and entry["repository"].strip():
            repositories.append({"repository": entry["repository"], "path": entry.get("path", "")})

    return repositories or list(_DEFAULT_REPOSITORIES)


def _already_scanned(registry: AssetRegistry, repository: str) -> bool:
    """True if `registry` already holds at least one asset sourced from `repository`.

    A deterministic, registry-only check - no separate "already
    scanned" state file is introduced. A repository whose prior scan
    matched zero files is rescanned on the next call; that is an
    inexpensive, acceptable edge case rather than added bookkeeping.
    """
    return any(
        asset.source.type == "github" and asset.source.details.get("repository") == repository
        for asset in registry
    )


def ensure_discovered(
    registry: AssetRegistry,
    repositories: Optional[List[Dict[str, str]]] = None,
    session: Optional[Any] = None,
    registry_path: Optional[Path] = None,
) -> int:
    """Scans every not-yet-scanned configured repository into `registry`.

    Thin orchestration over the existing `github.ingest_repository()`
    and `AssetRegistry.save()` - no discovery, retrieval, or registry
    logic is duplicated here. A repository already represented in
    `registry` (see `_already_scanned`) is skipped without any network
    call. A repository whose scan fails for any reason
    `github.GitHubSourceError` covers (not found, rate limited,
    network failure, malformed response, ...) is logged and skipped
    rather than raised, so one bad repository can never prevent the
    others - or the caller - from proceeding.

    Args:
        registry: The `AssetRegistry` to discover into and persist.
        repositories: Repositories to scan. Defaults to
            `configured_repositories()`.
        session: Forwarded to `github.ingest_repository` as its
            `session` - a `requests`-compatible object. Tests inject a
            fake session so no real HTTP call is ever made.
        registry_path: Where to persist newly discovered assets (see
            `AssetRegistry.save`). Defaults to the bundled
            `registry.json` next to `registry.py` if omitted.

    Returns:
        The number of newly added assets across every scanned
        repository (`0` if every repository was already scanned, or
        every scan attempted failed).
    """
    active_repositories = repositories if repositories is not None else configured_repositories()

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

        total_added += added

    if total_added > 0:
        registry.save(registry_path)

    return total_added
