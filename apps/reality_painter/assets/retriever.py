"""Local retrieval and caching of registered 3D assets for Reality Painter.

`AssetRetriever` is the one place that turns an already-discovered
`Asset` (produced by the existing GitHub asset source in
`apps.reality_painter.assets.github`, Phase 12B) into a local file
path, downloading and caching its `.glb`/`.gltf` file exactly once.
This module never discovers or registers assets itself - it accepts
`Asset` objects exactly as `discover_assets()`/`AssetRegistry` already
produce them - so it introduces no second GitHub discovery path and
has no import-time dependency on `AssetRegistry`.

FLOW: AssetRegistry -> GitHub asset source (12B) -> Asset -> AssetRetriever -> local cache -> local path.

Storage independence: only `AssetSource.type == "github"` is supported
today, dispatched via `_DOWNLOADERS`. A future source type (R2, S3,
Hugging Face, ...) is added by registering another entry there, never
by changing `retrieve()` itself, and never by hard-coding a specific
cloud provider here.

Security: cache filenames are derived only from `Asset.id`/`Asset.format`
(sanitized), never from `AssetSource.details` - remote metadata can
never influence where a file is written on disk, which is what
prevents path traversal via a malicious/malformed source. Downloaded
files are never executed or parsed by this module, only written to
disk and returned as a path.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

from apps.reality_painter.assets.schema import Asset

_API_BASE_URL = "https://api.github.com"
_DEFAULT_TIMEOUT_SECONDS = 15.0
_DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"
_SUPPORTED_FORMATS = (".glb", ".gltf")
_DOWNLOAD_CHUNK_SIZE = 65536


# --- Errors ---------------------------------------------------------------


class AssetRetrievalError(Exception):
    """Base class for errors raised by `AssetRetriever`."""


class UnsupportedFormatError(AssetRetrievalError):
    """Raised when an asset's format/extension isn't supported."""


class AssetNotFoundError(AssetRetrievalError):
    """Raised when the remote asset file cannot be found."""


class RetrievalRateLimitError(AssetRetrievalError):
    """Raised when the remote source's rate limit is exhausted."""


class RetrievalNetworkError(AssetRetrievalError):
    """Raised when a network-level failure prevents retrieval."""


class InvalidCachePathError(AssetRetrievalError):
    """Raised when an asset's identity would resolve outside the cache directory."""


# --- Retriever --------------------------------------------------------------


class AssetRetriever:
    """Retrieves a registered `Asset`'s 3D file into a local, deduplicated cache.

    Cache paths are deterministic, derived only from `Asset.id` and
    `Asset.format`, and validated to stay within the configured cache
    directory. A cached file already present and non-empty is returned
    as-is, without any network call - this is what makes a repeated
    `retrieve()` call for the same asset a cache hit rather than a
    duplicate download. A previously cached file that is missing or
    empty (e.g. a prior download was interrupted) is treated as invalid
    and re-downloaded.
    """

    def __init__(
        self,
        cache_dir: Optional[Any] = None,
        session: Optional[Any] = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Creates a retriever bound to a local cache directory.

        Args:
            cache_dir: Directory to store downloaded asset files under.
                Defaults to a `cache/` directory next to this module -
                deliberately separate from `registry.json` and never
                Git-tracked (downloaded assets are never committed).
            session: An object exposing a `requests`-compatible `.get()`
                method. Defaults to the `requests` module itself; tests
                inject a fake session so no real HTTP call is ever made.
            timeout_seconds: Per-request HTTP timeout.
        """
        self._cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._http = session if session is not None else requests
        self._timeout_seconds = timeout_seconds

    def retrieve(self, asset: Asset) -> Path:
        """Returns a local path to `asset`'s 3D file, downloading if needed.

        Args:
            asset: A validated `Asset`, exactly as produced by
                `apps.reality_painter.assets.github.discover_assets`
                (or read back from an `AssetRegistry`).

        Returns:
            The local filesystem path to the asset's cached file.

        Raises:
            UnsupportedFormatError: If `asset.format` isn't `.glb`/`.gltf`.
            InvalidCachePathError: If the resolved cache path would
                escape the cache directory.
            AssetNotFoundError: If the remote file cannot be found.
            RetrievalRateLimitError: If the remote source's rate limit
                is exhausted.
            RetrievalNetworkError: On a network-level failure.
            AssetRetrievalError: On any other unrecoverable retrieval
                failure (e.g. an unsupported source type, or an empty
                download).
        """
        cache_path = self._cache_path_for(asset)

        if self._is_valid_cached_file(cache_path):
            return cache_path

        downloader = _DOWNLOADERS.get(asset.source.type)
        if downloader is None:
            raise AssetRetrievalError(f"No retrieval support for source type {asset.source.type!r}.")

        downloader(self, asset, cache_path)
        return cache_path

    # --- Cache path -------------------------------------------------

    def _cache_path_for(self, asset: Asset) -> Path:
        """Builds a deterministic, sandboxed cache path for `asset`.

        The filename is derived only from `asset.id` (sanitized to
        filesystem-safe characters) and `asset.format` - never from
        `asset.source.details`, which may originate from remote,
        untrusted metadata. The result is additionally verified to
        resolve inside the cache directory as defense in depth.

        Raises:
            UnsupportedFormatError: If `asset.format` isn't supported.
            InvalidCachePathError: If the resolved path escapes the
                cache directory.
        """
        extension = f".{asset.format.lower()}"
        if extension not in _SUPPORTED_FORMATS:
            raise UnsupportedFormatError(f"Unsupported asset format {asset.format!r} for asset {asset.id!r}.")

        safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", asset.id)
        cache_root = self._cache_dir.resolve()
        resolved = (cache_root / f"{safe_stem}{extension}").resolve()

        if resolved.parent != cache_root:
            raise InvalidCachePathError(f"Resolved cache path for asset {asset.id!r} escapes the cache directory.")

        return resolved

    def _is_valid_cached_file(self, path: Path) -> bool:
        """True if `path` exists and is a non-empty file.

        A zero-byte file is treated as corrupt/invalid - the same
        signal an interrupted or failed prior download would leave -
        and triggers a fresh download rather than being returned as-is.
        """
        return path.is_file() and path.stat().st_size > 0

    # --- GitHub download --------------------------------------------------

    def _download_from_github(self, asset: Asset, cache_path: Path) -> None:
        """Downloads `asset`'s file from GitHub into `cache_path`.

        Looks up the file's `download_url` via the GitHub Contents API
        (using `source.details["repository"]`/`["path"]`, exactly as
        `apps.reality_painter.assets.github` already populates them)
        rather than guessing a raw-content URL, since no ref/branch is
        stored in `AssetSource.details` today.

        Raises:
            AssetRetrievalError: If the asset's source metadata is
                incomplete/malformed, or the metadata response is
                malformed or missing a download URL.
            AssetNotFoundError: If the repository/path/file doesn't exist.
            RetrievalRateLimitError: If GitHub's rate limit is exhausted.
            RetrievalNetworkError: On a network-level failure.
        """
        repository = asset.source.details.get("repository")
        file_path = asset.source.details.get("path")
        if not repository or not file_path:
            raise AssetRetrievalError(f"Asset {asset.id!r} has incomplete GitHub source metadata.")

        owner, separator, repo = str(repository).partition("/")
        if not separator or not owner or not repo:
            raise AssetRetrievalError(f"Asset {asset.id!r} has a malformed GitHub repository {repository!r}.")

        metadata_url = f"{_API_BASE_URL}/repos/{owner}/{repo}/contents/{file_path}"
        response = self._get(metadata_url)

        if response.status_code == 404:
            raise AssetNotFoundError(f"Asset {asset.id!r} not found at '{repository}:{file_path}'.")
        self._raise_for_rate_limit(response)
        if response.status_code != 200:
            raise AssetRetrievalError(
                f"Unexpected GitHub response fetching metadata for asset {asset.id!r} (HTTP {response.status_code})."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise AssetRetrievalError(f"Malformed GitHub metadata response for asset {asset.id!r}.") from exc

        download_url = payload.get("download_url") if isinstance(payload, dict) else None
        if not download_url:
            raise AssetRetrievalError(f"GitHub metadata for asset {asset.id!r} has no download_url.")

        self._stream_download(download_url, cache_path, asset.id)

    def _stream_download(self, url: str, cache_path: Path, asset_id: str) -> None:
        """Streams `url`'s content into `cache_path` without holding it fully in memory.

        Writes to a temporary sibling file first and only replaces the
        final cache path once the download completes successfully, so
        a failed or partial download never leaves a corrupt file at
        `cache_path` for a later `retrieve()` call to mistake as a
        valid cache hit.

        Raises:
            AssetNotFoundError: If the download URL itself 404s.
            RetrievalRateLimitError: If the remote rate limit is hit.
            RetrievalNetworkError: On a network-level failure mid-download.
            AssetRetrievalError: If the response is otherwise
                unexpected, or the downloaded content is empty.
        """
        temp_path = cache_path.with_name(cache_path.name + ".part")
        response = self._get(url, stream=True)

        if response.status_code == 404:
            raise AssetNotFoundError(f"Asset {asset_id!r} file not found at its download URL.")
        self._raise_for_rate_limit(response)
        if response.status_code != 200:
            raise AssetRetrievalError(f"Unexpected response downloading asset {asset_id!r} (HTTP {response.status_code}).")

        total_bytes = 0
        try:
            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        file.write(chunk)
                        total_bytes += len(chunk)
        except (requests.exceptions.RequestException, OSError) as exc:
            temp_path.unlink(missing_ok=True)
            raise RetrievalNetworkError(f"Download failed for asset {asset_id!r}: {type(exc).__name__}.") from exc

        if total_bytes == 0:
            temp_path.unlink(missing_ok=True)
            raise AssetRetrievalError(f"Downloaded file for asset {asset_id!r} was empty.")

        temp_path.replace(cache_path)

    # --- HTTP helpers -------------------------------------------------

    def _get(self, url: str, stream: bool = False) -> Any:
        """Issues one GET request, translating transport failures.

        Never raises a raw `requests` exception past this method -
        every network-level failure becomes a `RetrievalNetworkError`.
        """
        try:
            token = os.getenv("GITHUB_TOKEN")
            headers = {"Accept": "application/vnd.github+json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            return self._http.get(
                url,
                headers=headers,
                timeout=self._timeout_seconds,
                stream=stream,
            )
        except requests.exceptions.Timeout as exc:
            raise RetrievalNetworkError("Request timed out.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise RetrievalNetworkError("Could not reach remote source.") from exc
        except requests.exceptions.RequestException as exc:
            raise RetrievalNetworkError(f"Request failed ({type(exc).__name__}).") from exc

    def _raise_for_rate_limit(self, response: Any) -> None:
        """Raises `RetrievalRateLimitError` if `response` indicates a rate limit hit."""
        if response.status_code == 429:
            raise RetrievalRateLimitError("Rate limit exceeded (HTTP 429).")
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            raise RetrievalRateLimitError("Rate limit exceeded (HTTP 403, X-RateLimit-Remaining=0).")

    # --- Poly Pizza download -----------------------------------------------

    def _download_from_poly_pizza(self, asset: Asset, cache_path: Path) -> None:
        """Downloads `asset`'s GLB file from a Poly Pizza CDN URL.

        The direct download URL is stored in ``source.details["download_url"]``
        by the ``polypizza`` module. No authentication header is required for
        CDN downloads — only the initial API search uses the API key.

        Raises:
            AssetRetrievalError: If the source metadata is incomplete.
            AssetNotFoundError: If the CDN URL 404s.
            RetrievalRateLimitError: If rate-limited.
            RetrievalNetworkError: On a network failure.
        """
        download_url = asset.source.details.get("download_url")
        if not download_url:
            raise AssetRetrievalError(
                f"Asset {asset.id!r} has no Poly Pizza download URL.")

        # Poly Pizza CDN downloads don't need GitHub auth — use a plain GET.
        try:
            response = self._http.get(
                download_url, timeout=self._timeout_seconds, stream=True)
        except requests.exceptions.Timeout as exc:
            raise RetrievalNetworkError("Poly Pizza download timed out.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise RetrievalNetworkError(
                "Could not reach Poly Pizza CDN.") from exc
        except requests.exceptions.RequestException as exc:
            raise RetrievalNetworkError(
                f"Poly Pizza download failed ({type(exc).__name__}).") from exc

        if response.status_code == 404:
            raise AssetNotFoundError(
                f"Asset {asset.id!r} not found at Poly Pizza CDN URL.")
        if response.status_code == 429:
            raise RetrievalRateLimitError("Poly Pizza CDN rate limit exceeded.")
        if response.status_code != 200:
            raise AssetRetrievalError(
                f"Unexpected Poly Pizza CDN response (HTTP {response.status_code}).")

        # Reuse the safe streaming download (atomic write via .part file)
        temp_path = cache_path.with_name(cache_path.name + ".part")
        total_bytes = 0
        try:
            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        file.write(chunk)
                        total_bytes += len(chunk)
        except (requests.exceptions.RequestException, OSError) as exc:
            temp_path.unlink(missing_ok=True)
            raise RetrievalNetworkError(
                f"Download failed for Poly Pizza asset {asset.id!r}: "
                f"{type(exc).__name__}.") from exc

        if total_bytes == 0:
            temp_path.unlink(missing_ok=True)
            raise AssetRetrievalError(
                f"Downloaded Poly Pizza file for asset {asset.id!r} was empty.")

        temp_path.replace(cache_path)


# --- Source-type dispatch ---------------------------------------------

# Maps `AssetSource.type` to the `AssetRetriever` method that knows how
# to download it. Adding a future source type (e.g. "s3", "r2") means
# registering another entry here - `retrieve()` itself never changes.
_DOWNLOADERS: Dict[str, Callable[["AssetRetriever", Asset, Path], None]] = {
    "github": AssetRetriever._download_from_github,
    "poly_pizza": AssetRetriever._download_from_poly_pizza,
}
