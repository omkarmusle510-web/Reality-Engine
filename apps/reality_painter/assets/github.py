"""GitHub public-repository asset source for Reality Painter's asset registry.

Connects the asset metadata layer (`schema.py`/`registry.py`) to public
GitHub repositories: given a repository, optional path, and optional
ref, `discover_assets()` walks that repository's contents via the
public GitHub REST API (no token required) and returns validated
`Asset` objects for every candidate 3D model file found (`.glb`,
`.gltf` by default).

This module performs metadata discovery only. It never downloads a
model's file contents, never loads or renders a model, and never
executes anything from the target repository - only the repository's
directory listing and (best-effort) license metadata are read.

Category/tag extraction is purely deterministic (directory names,
filename, extension) - no LLM or embedding-based inference happens
here. A category that can't be matched against a small known set is
recorded as `"unknown"` rather than guessed.

`ingest_repository()` is the sanctioned way to get discovered assets
into an existing `AssetRegistry`: it calls `discover_assets()` and
registers each result via `AssetRegistry.register()`, so a repository
can be re-scanned any number of times without creating duplicate
registry entries.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.schema import Asset

_API_BASE_URL = "https://api.github.com"
_DEFAULT_EXTENSIONS: Tuple[str, ...] = (".glb", ".gltf")
_DEFAULT_TIMEOUT_SECONDS = 15.0
_DEFAULT_MAX_DEPTH = 8

# Deterministic category vocabulary - a directory name matching one of
# these (case-insensitive) becomes the asset's category. Never guessed
# or expanded via inference; unmatched assets get `_UNKNOWN_CATEGORY`.
_KNOWN_CATEGORIES = {
    "furniture", "plants", "vehicles", "characters", "props",
    "architecture", "nature", "animals", "weapons", "food",
}
_UNKNOWN_CATEGORY = "unknown"


# --- Errors ---------------------------------------------------------------


class GitHubSourceError(Exception):
    """Base class for errors raised by the GitHub asset source."""


class RepositoryNotFoundError(GitHubSourceError):
    """Raised when the target repository does not exist (or is private)."""


class PathNotFoundError(GitHubSourceError):
    """Raised when the requested path does not exist in the repository."""


class RateLimitError(GitHubSourceError):
    """Raised when GitHub's (unauthenticated) API rate limit is exhausted."""


class GitHubNetworkError(GitHubSourceError):
    """Raised when a network-level failure prevents reaching GitHub."""


class MalformedResponseError(GitHubSourceError):
    """Raised when GitHub returns a response that can't be parsed as expected."""


# --- Public API -------------------------------------------------------------


def discover_assets(
    repository: str,
    path: str = "",
    ref: Optional[str] = None,
    extensions: Sequence[str] = _DEFAULT_EXTENSIONS,
    session: Optional[Any] = None,
) -> List[Asset]:
    """Discovers candidate 3D asset files in a public GitHub repository.

    Recursively walks `path` (the whole repository if omitted),
    matching files against `extensions`, and converts each match into
    a validated `Asset` via deterministic metadata rules (see module
    docstring). Never downloads a matched file's contents.

    Args:
        repository: `"owner/name"` of a public GitHub repository.
        path: Directory (or single file) to start discovery from.
            Defaults to the repository root.
        ref: Branch, tag, or commit SHA to read from. Defaults to the
            repository's default branch.
        extensions: File extensions treated as candidate 3D assets.
            Defaults to `(".glb", ".gltf")`.
        session: An object exposing a `requests`-compatible `.get()`
            method. Defaults to the `requests` module itself; tests
            inject a fake session so no real HTTP call is ever made.

    Returns:
        A list of validated `Asset` objects, one per matched file.

    Raises:
        ValueError: If `repository` isn't `"owner/name"` shaped.
        RepositoryNotFoundError: If the repository doesn't exist.
        PathNotFoundError: If `path` doesn't exist in the repository.
        RateLimitError: If GitHub's API rate limit is exhausted.
        GitHubNetworkError: If a network-level failure occurs.
        MalformedResponseError: If GitHub's response can't be parsed.
    """
    owner, repo = _parse_repository(repository)
    http = session if session is not None else requests

    license_id = _fetch_license(http, owner, repo)

    matched_entries: List[Dict[str, Any]] = []
    _walk(http, owner, repo, path, ref, extensions, matched_entries)

    return [_build_asset(entry, repository, license_id) for entry in matched_entries]


def ingest_repository(
    registry: AssetRegistry,
    repository: str,
    path: str = "",
    ref: Optional[str] = None,
    extensions: Sequence[str] = _DEFAULT_EXTENSIONS,
    session: Optional[Any] = None,
) -> Tuple[int, int]:
    """Discovers assets in a repository and registers them into `registry`.

    Thin composition of `discover_assets()` and
    `AssetRegistry.register()` - this function holds no registry logic
    of its own. Safe to call repeatedly against the same repository:
    previously-registered assets (same deterministic id - see
    `_make_asset_id`) are skipped, never duplicated or overwritten.

    Args:
        registry: The `AssetRegistry` to ingest discovered assets into.
        repository: `"owner/name"` of a public GitHub repository.
        path: Directory (or single file) to start discovery from.
        ref: Branch, tag, or commit SHA to read from.
        extensions: File extensions treated as candidate 3D assets.
        session: See `discover_assets`.

    Returns:
        `(added_count, skipped_count)` - `added_count` is the number of
        newly registered assets, `skipped_count` the number already
        present in `registry`.
    """
    discovered = discover_assets(repository, path=path, ref=ref, extensions=extensions, session=session)

    added = 0
    skipped = 0
    for asset in discovered:
        if registry.register(asset):
            added += 1
        else:
            skipped += 1
    return added, skipped


# --- Repository / path parsing -----------------------------------------


def _parse_repository(repository: str) -> Tuple[str, str]:
    """Splits `"owner/name"` into `(owner, name)`.

    Raises:
        ValueError: If `repository` isn't shaped `"owner/name"`.
    """
    owner, separator, name = repository.partition("/")
    if not separator or not owner.strip() or not name.strip():
        raise ValueError(f"repository must be 'owner/name', got {repository!r}.")
    return owner.strip(), name.strip()


# --- HTTP -----------------------------------------------------------------


def _get(http: Any, url: str, params: Optional[Dict[str, str]] = None) -> Any:
    """Issues one GET request, translating transport failures.

    Never raises a raw `requests` exception past this function - every
    network-level failure becomes a `GitHubNetworkError`, so callers
    only ever need to handle this module's own exception hierarchy.
    """
    try:
        return http.get(
            url,
            params=params,
            headers={"Accept": "application/vnd.github+json"},
            timeout=_DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout as exc:
        raise GitHubNetworkError(f"GitHub request timed out: {url}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise GitHubNetworkError(f"Could not reach GitHub: {url}") from exc
    except requests.exceptions.RequestException as exc:
        raise GitHubNetworkError(f"GitHub request failed ({type(exc).__name__}): {url}") from exc


def _raise_for_rate_limit(response: Any) -> None:
    """Raises `RateLimitError` if `response` indicates GitHub's rate limit was hit."""
    if response.status_code == 429:
        raise RateLimitError("GitHub API rate limit exceeded (HTTP 429).")
    if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
        raise RateLimitError("GitHub API rate limit exceeded (HTTP 403, X-RateLimit-Remaining=0).")


def _parse_json(response: Any, description: str) -> Any:
    """Parses `response.json()`, converting failures to `MalformedResponseError`."""
    try:
        return response.json()
    except ValueError as exc:
        raise MalformedResponseError(f"{description} was not valid JSON.") from exc


def _fetch_license(http: Any, owner: str, repo: str) -> Optional[str]:
    """Best-effort fetch of a repository's SPDX license id, or `None`.

    Also serves as the repository-existence check: a 404 here means
    the repository itself doesn't exist, distinct from a 404 on a
    contents lookup (which means the *path* doesn't exist within an
    existing repository).

    Returns:
        The repository's SPDX license id (e.g. `"MIT"`), or `None` if
        no license is set or it can't be determined. Never invents a
        value - absence here is what `Asset.license` already treats as
        "unknown" (see `schema.py`).

    Raises:
        RepositoryNotFoundError: If the repository doesn't exist.
        RateLimitError: If GitHub's rate limit is exhausted.
        GitHubNetworkError: On a network-level failure.
        MalformedResponseError: If the response can't be parsed.
    """
    url = f"{_API_BASE_URL}/repos/{owner}/{repo}"
    response = _get(http, url)

    if response.status_code == 404:
        raise RepositoryNotFoundError(f"Repository '{owner}/{repo}' not found.")
    _raise_for_rate_limit(response)
    if response.status_code != 200:
        raise MalformedResponseError(f"Unexpected response for repository '{owner}/{repo}' (HTTP {response.status_code}).")

    payload = _parse_json(response, f"Repository '{owner}/{repo}' info response")
    if not isinstance(payload, dict):
        raise MalformedResponseError(f"Repository '{owner}/{repo}' info response had an unexpected shape.")

    license_info = payload.get("license")
    if isinstance(license_info, dict):
        spdx_id = license_info.get("spdx_id")
        if isinstance(spdx_id, str) and spdx_id and spdx_id != "NOASSERTION":
            return spdx_id
    return None


def _list_contents(http: Any, owner: str, repo: str, path: str, ref: Optional[str]) -> List[Dict[str, Any]]:
    """Lists one directory's (or one file's) contents via the GitHub API.

    A single-file `path` returns a JSON object rather than an array;
    that shape is normalized to a one-element list here so callers
    never need to branch on it.

    Raises:
        PathNotFoundError: If `path` doesn't exist in the repository.
        RateLimitError: If GitHub's rate limit is exhausted.
        GitHubNetworkError: On a network-level failure.
        MalformedResponseError: If the response can't be parsed.
    """
    url = f"{_API_BASE_URL}/repos/{owner}/{repo}/contents/{path}".rstrip("/")
    params = {"ref": ref} if ref else None
    response = _get(http, url, params=params)

    if response.status_code == 404:
        raise PathNotFoundError(f"Path '{path}' not found in repository '{owner}/{repo}'.")
    _raise_for_rate_limit(response)
    if response.status_code != 200:
        raise MalformedResponseError(f"Unexpected response listing '{path}' (HTTP {response.status_code}).")

    payload = _parse_json(response, f"Contents listing for '{path}'")
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(entry, dict) for entry in payload):
        raise MalformedResponseError(f"Contents listing for '{path}' had an unexpected shape.")
    return payload


def _matches_extension(file_path: str, extensions: Sequence[str]) -> bool:
    """True if `file_path`'s suffix matches one of `extensions` (case-insensitive)."""
    lower_path = file_path.lower()
    return any(lower_path.endswith(extension.lower()) for extension in extensions)


def _walk(
    http: Any,
    owner: str,
    repo: str,
    path: str,
    ref: Optional[str],
    extensions: Sequence[str],
    matched_entries: List[Dict[str, Any]],
    depth: int = 0,
) -> None:
    """Recursively lists `path`, collecting entries matching `extensions`.

    Directories are descended into regardless of depth up to
    `_DEFAULT_MAX_DEPTH`, so nested asset folders are discovered the
    same as top-level ones. Non-matching files are silently skipped -
    not every file in a repository is a candidate 3D asset.
    """
    if depth > _DEFAULT_MAX_DEPTH:
        return

    for entry in _list_contents(http, owner, repo, path, ref):
        entry_type = entry.get("type")
        entry_path = entry.get("path")
        if not entry_path:
            continue

        if entry_type == "dir":
            _walk(http, owner, repo, entry_path, ref, extensions, matched_entries, depth + 1)
        elif entry_type == "file" and _matches_extension(entry_path, extensions):
            matched_entries.append(entry)


# --- Asset construction -----------------------------------------------


def _make_asset_id(repository: str, file_path: str) -> str:
    """Builds a deterministic, stable asset id from repository + file path.

    Deterministic so re-discovering the same file always produces the
    same id - this is what lets `AssetRegistry.register()` recognize a
    re-scanned asset as already present instead of creating a
    duplicate entry.
    """
    slug_source = f"github_{repository}_{file_path}".lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug_source).strip("_")
    return slug


def _infer_category(directories: Sequence[str]) -> str:
    """Matches a path's directory names against the known category vocabulary.

    Returns the first matching known category (closest to the file, to
    prefer more specific directories), or `_UNKNOWN_CATEGORY` if none
    match. Never inferred beyond this fixed vocabulary - see module
    docstring.
    """
    for directory in reversed(directories):
        if directory in _KNOWN_CATEGORIES:
            return directory
    return _UNKNOWN_CATEGORY


def _infer_tags(directories: Sequence[str], file_stem: str) -> List[str]:
    """Builds a deterministic tag list from directory names and the filename."""
    stem_parts = [part for part in re.split(r"[_\-\s]+", file_stem.lower()) if part]
    candidates = list(directories) + stem_parts
    return sorted({tag for tag in candidates if tag})


def _build_asset(entry: Dict[str, Any], repository: str, license_id: Optional[str]) -> Asset:
    """Converts one matched GitHub content entry into a validated `Asset`."""
    file_path = entry["path"]
    path_object = PurePosixPath(file_path)
    directories = [part.lower() for part in path_object.parts[:-1]]
    file_stem = path_object.stem
    asset_format = path_object.suffix.lstrip(".").lower()

    data: Dict[str, Any] = {
        "id": _make_asset_id(repository, file_path),
        "name": file_stem.replace("_", " ").replace("-", " ").strip().title() or file_path,
        "category": _infer_category(directories),
        "format": asset_format,
        "tags": _infer_tags(directories, file_stem),
        "source": {"type": "github", "repository": repository, "path": file_path},
    }
    if license_id:
        data["license"] = license_id

    return Asset.from_dict(data)
