"""Persistent optimization cache for Reality Painter's Asset Optimizer.

Block 5 stores and retrieves the *output* of Blocks 1-4: an already
optimized/selected GLB file, keyed by a deterministic identity derived
from a source asset's identity plus the optimizer configuration that
produced it. It performs no analysis (Block 1), no candidate
generation (Block 2), no benchmarking (Block 3), and no selection
(Block 4) itself - it only persists and validates a result those
blocks already computed.

DISTINCT FROM THE EXISTING SOURCE CACHE
-----------------------------------------
`apps.reality_painter.assets.retriever.AssetRetriever` already caches
*downloaded source* GLB/GLTF files (remote asset -> local file). This
module is a second, independent layer sitting logically downstream of
that one:

    SOURCE CACHE (AssetRetriever):   remote source  -> local source GLB
    OPTIMIZATION CACHE (this module): source identity + optimizer
                                       configuration -> optimized GLB

This module never imports, modifies, wraps, or duplicates
`AssetRetriever`, and performs zero network access - it operates
entirely on local files already present on disk, exactly like
`AssetRetriever`'s own cache-hit path.

CACHE KEY / IDENTITY
---------------------
A cache key is never a human-readable label like `"flower"` - it is a
deterministic SHA-256 digest (see `CacheKey.build`) over the source
asset's identity, the `OPTIMIZER_VERSION`, and (optionally) a
selection-configuration mapping. The same logical inputs always
produce the same key; different inputs are, in practice, collision-
free. No timestamp or random UUID is ever part of the key.

OPTIMIZER_VERSION
-------------------
`OPTIMIZER_VERSION` is an explicit schema/algorithm version stamped
into every cache entry's metadata and folded into every cache key.
Bumping it after a future change to the optimization pipeline makes
every previously-cached entry naturally stop matching (a key built
under the new version never equals one built under the old version,
and `lookup()` additionally rejects an entry whose stored
`optimizer_version` doesn't match the cache's current one even in the
event of an artificial key collision). No automatic migration is
implemented - that is explicitly out of scope for this block.

VALIDATION
-----------
A cache hit is never just "a file exists at this path." `lookup()`
requires: parseable metadata, a cache-key match, a matching
`optimizer_version`, a matching metadata schema version, and a
referenced asset file that exists, is a regular file, and is
non-empty. Any failure is reported as `CacheStatus.INVALID` (or
`CacheStatus.MISS` if the entry simply doesn't exist) rather than
raising or returning something unrelated.

ATOMICITY
----------
`store()` writes both the copied asset and its metadata to temporary
`.part` siblings first, then promotes the asset into place, then the
metadata. Metadata existing at its final path is what `lookup()`
treats as "entry present" - so a crash or interruption at any point
before the metadata rename completes leaves, at worst, an orphaned
temp/asset file and a clean `CACHE_MISS` on the next lookup, never a
half-valid or corrupted hit.

PATH SAFETY
------------
Cache filenames are derived only from the cache key's own hex digest
(sanitized defensively regardless), never from a source label,
identity string, or any other caller-controlled text. Every resolved
path is verified to remain inside the configured cache directory,
preventing path traversal.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

#: Explicit optimizer/cache schema version. Folded into every cache
#: key and stamped into every entry's metadata - see module docstring.
OPTIMIZER_VERSION = "v1"

#: Metadata file shape version, independent of `OPTIMIZER_VERSION`:
#: this changes only if `CacheEntryMetadata`'s own JSON shape changes,
#: not when the optimization algorithm changes.
_METADATA_SCHEMA_VERSION = 1

_DEFAULT_CACHE_DIR = Path(__file__).parent / "optimized_cache"
_ASSET_EXTENSION = ".glb"
_METADATA_EXTENSION = ".json"


class CacheStatus(str, Enum):
    """The outcome of one `OptimizationCache.lookup()` call."""

    HIT = "hit"
    MISS = "miss"
    INVALID = "invalid"


class OptimizationCacheError(Exception):
    """Base class for errors raised directly by this module's public API."""


class InvalidCacheKeyError(OptimizationCacheError):
    """Raised when a cache key would resolve outside the cache directory."""


class SourceAssetNotFoundError(OptimizationCacheError):
    """Raised by `store()` when the optimized asset to cache does not exist."""


# --- Cache key --------------------------------------------------------


@dataclass(frozen=True)
class CacheKey:
    """A deterministic cache identity.

    `value` is expected to be a SHA-256 hex digest, as produced by
    `build()` - the sanctioned constructor. Direct construction with an
    arbitrary string is still accepted (e.g. for testing path-safety
    behavior), but every use of `value` as a filename component is
    sanitized defensively regardless - see `OptimizationCache`.

    Attributes:
        value: The opaque, deterministic key string.
    """

    value: str

    @staticmethod
    def build(
        source_identity: str,
        optimizer_version: str = OPTIMIZER_VERSION,
        selection_config: Optional[Dict[str, Any]] = None,
    ) -> "CacheKey":
        """Builds a deterministic key from stable, logical inputs.

        The same `(source_identity, optimizer_version,
        selection_config)` always produces the same key; a different
        source identity, a different optimizer version, or a
        different selection configuration all produce a different key.
        Never derived from a timestamp or random value.

        Args:
            source_identity: A stable identity for the *source* asset
                being optimized (e.g. an `Asset.id`, a repository +
                path, or a content hash - never merely a display
                label). Must be a non-empty string.
            optimizer_version: The optimizer schema/algorithm version
                this key is being built under. Defaults to the
                current `OPTIMIZER_VERSION`.
            selection_config: Optional additional configuration that
                affects the optimized result (e.g. target/minimum FPS
                from a `PerformancePolicy`, or a selected candidate
                spec name) - serialized in sorted key order so key
                construction is order-independent. Values must be
                JSON-serializable.

        Returns:
            A new `CacheKey`.

        Raises:
            ValueError: If `source_identity` is empty/not a string.
        """
        if not isinstance(source_identity, str) or not source_identity.strip():
            raise ValueError("source_identity must be a non-empty string.")

        parts = [f"source={source_identity}", f"optimizer_version={optimizer_version}"]
        if selection_config:
            serialized = json.dumps(selection_config, sort_keys=True, default=str)
            parts.append(f"selection_config={serialized}")

        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        return CacheKey(value=digest)


# --- Metadata -----------------------------------------------------------


@dataclass(frozen=True)
class CacheEntryMetadata:
    """Everything needed to validate and identify one cached optimized asset.

    Deliberately excludes benchmark history or full candidate details -
    only enough is kept to answer "does this cached file actually
    belong to this source/configuration?" (see module docstring).

    Attributes:
        cache_key: The entry's own cache key (hex digest), duplicated
            here (not just implied by the filename) so `lookup()` can
            detect a mismatch even if a file were ever misplaced.
        source_identity: The source asset identity this entry was
            built from (see `CacheKey.build`).
        optimizer_version: The `OPTIMIZER_VERSION` this entry was
            produced under.
        asset_filename: The cached optimized GLB's filename (relative
            to the cache directory).
        selected_candidate: The name of the candidate Block 4 selected
            to produce this entry, if known. Purely informational.
        created_at: Wall-clock time (`time.time()`) this entry was
            first stored. Preserved across a `store()` that overwrites
            an existing entry.
        updated_at: Wall-clock time this entry was last (re)written.
        schema_version: This metadata shape's own version - see
            `_METADATA_SCHEMA_VERSION`.
    """

    cache_key: str
    source_identity: str
    optimizer_version: str
    asset_filename: str
    selected_candidate: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    schema_version: int = _METADATA_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Returns a plain, JSON-serializable dict of this metadata."""
        return asdict(self)

    @staticmethod
    def from_dict(data: Any) -> "CacheEntryMetadata":
        """Parses and validates a raw metadata mapping.

        Args:
            data: Parsed JSON content of a metadata file.

        Returns:
            A validated `CacheEntryMetadata`.

        Raises:
            ValueError: If `data` is not a mapping, is missing a
                required field, or a field has the wrong shape.
        """
        if not isinstance(data, dict):
            raise ValueError("Cache metadata must be a JSON object.")

        required_fields = (
            "cache_key",
            "source_identity",
            "optimizer_version",
            "asset_filename",
            "created_at",
            "updated_at",
            "schema_version",
        )
        for field_name in required_fields:
            if field_name not in data:
                raise ValueError(f"Cache metadata missing required field {field_name!r}.")

        try:
            return CacheEntryMetadata(
                cache_key=str(data["cache_key"]),
                source_identity=str(data["source_identity"]),
                optimizer_version=str(data["optimizer_version"]),
                asset_filename=str(data["asset_filename"]),
                selected_candidate=(
                    str(data["selected_candidate"]) if data.get("selected_candidate") is not None else None
                ),
                created_at=float(data["created_at"]),
                updated_at=float(data["updated_at"]),
                schema_version=int(data["schema_version"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cache metadata has an invalid field: {exc}") from exc


# --- Lookup result --------------------------------------------------------


@dataclass(frozen=True)
class CacheLookupResult:
    """The outcome of one `OptimizationCache.lookup()` call.

    Attributes:
        status: `HIT`, `MISS`, or `INVALID` - see `CacheStatus`.
        cache_key: The key that was looked up, echoed back for
            convenience.
        asset_path: The cached optimized GLB's local path, only if
            `status` is `HIT`.
        metadata: The validated entry metadata, only if `status` is
            `HIT`.
        reason: A human-readable explanation of the outcome - always
            non-empty.
    """

    status: CacheStatus
    cache_key: str
    asset_path: Optional[Path]
    metadata: Optional[CacheEntryMetadata]
    reason: str


# --- Cache ----------------------------------------------------------------


class OptimizationCache:
    """Persistent, on-disk cache of optimized/selected GLB assets.

    Every path this class writes to or reads from is derived only from
    a `CacheKey`'s own (sanitized) hex value and validated to resolve
    inside `cache_dir` - never from caller-supplied labels or asset
    metadata, which may be untrusted (see module docstring's "PATH
    SAFETY" section).
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None) -> None:
        """Creates a cache bound to a local cache directory.

        Args:
            cache_dir: Directory to store optimized assets and their
                metadata under. Defaults to an `optimized_cache/`
                directory next to this module, deliberately distinct
                from `AssetRetriever`'s own `cache/` directory.
        """
        self._cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Path resolution (sandboxed) -----------------------------------

    def _safe_key_stem(self, cache_key: CacheKey) -> str:
        """Sanitizes a cache key's value into a filesystem-safe stem.

        Defensive in depth: `CacheKey.build()` already only ever
        produces a hex digest, but this never trusts that a `CacheKey`
        passed to this class was necessarily built that way.
        """
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", cache_key.value) or "_"

    def _paths_for(self, cache_key: CacheKey) -> "tuple[Path, Path]":
        """Resolves `(asset_path, metadata_path)` for `cache_key`, sandboxed to the cache dir.

        Raises:
            InvalidCacheKeyError: If the resolved paths would escape
                `cache_dir` (defense in depth against path traversal).
        """
        stem = self._safe_key_stem(cache_key)
        cache_root = self._cache_dir.resolve()
        asset_path = (cache_root / f"{stem}{_ASSET_EXTENSION}").resolve()
        metadata_path = (cache_root / f"{stem}{_METADATA_EXTENSION}").resolve()

        if asset_path.parent != cache_root or metadata_path.parent != cache_root:
            raise InvalidCacheKeyError(f"Cache key {cache_key.value!r} resolves outside the cache directory.")

        return asset_path, metadata_path

    # --- Lookup -----------------------------------------------------

    def lookup(self, cache_key: CacheKey) -> CacheLookupResult:
        """Looks up and fully validates a cache entry. Never raises.

        A `HIT` requires every one of: parseable metadata, a matching
        `cache_key`, a matching `OPTIMIZER_VERSION`, a matching
        metadata schema version, and an existing, non-empty, regular
        asset file. Any other outcome is `MISS` (nothing stored) or
        `INVALID` (something is stored but doesn't check out) - never
        an exception, and never a mismatched/unrelated asset.

        Args:
            cache_key: The key to look up.

        Returns:
            A `CacheLookupResult` describing the outcome.
        """
        try:
            asset_path, metadata_path = self._paths_for(cache_key)
        except InvalidCacheKeyError as exc:
            return CacheLookupResult(CacheStatus.INVALID, cache_key.value, None, None, str(exc))

        if not metadata_path.is_file():
            return CacheLookupResult(CacheStatus.MISS, cache_key.value, None, None, "No cache entry found.")

        try:
            raw = metadata_path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            metadata = CacheEntryMetadata.from_dict(parsed)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return CacheLookupResult(
                CacheStatus.INVALID, cache_key.value, None, None, f"Malformed cache metadata: {exc}"
            )

        if metadata.cache_key != cache_key.value:
            return CacheLookupResult(
                CacheStatus.INVALID, cache_key.value, None, None, "Cache metadata's cache_key does not match."
            )
        if metadata.schema_version != _METADATA_SCHEMA_VERSION:
            return CacheLookupResult(
                CacheStatus.INVALID,
                cache_key.value,
                None,
                None,
                f"Cache metadata schema version {metadata.schema_version} does not match "
                f"expected {_METADATA_SCHEMA_VERSION}.",
            )
        if metadata.optimizer_version != OPTIMIZER_VERSION:
            return CacheLookupResult(
                CacheStatus.INVALID,
                cache_key.value,
                None,
                None,
                f"Cache entry optimizer_version {metadata.optimizer_version!r} does not match "
                f"current {OPTIMIZER_VERSION!r}.",
            )

        if not asset_path.exists():
            return CacheLookupResult(
                CacheStatus.INVALID, cache_key.value, None, None, "Cached asset file is missing."
            )
        if not asset_path.is_file():
            return CacheLookupResult(
                CacheStatus.INVALID, cache_key.value, None, None, "Cached asset path is not a regular file."
            )
        if asset_path.stat().st_size == 0:
            return CacheLookupResult(
                CacheStatus.INVALID, cache_key.value, None, None, "Cached asset file is empty."
            )

        return CacheLookupResult(CacheStatus.HIT, cache_key.value, asset_path, metadata, "Cache hit.")

    def exists(self, cache_key: CacheKey) -> bool:
        """True if `lookup(cache_key)` would report `CacheStatus.HIT`."""
        return self.lookup(cache_key).status == CacheStatus.HIT

    # --- Store --------------------------------------------------------

    def store(
        self,
        cache_key: CacheKey,
        source_identity: str,
        optimized_asset_path: Union[str, Path],
        selected_candidate: Optional[str] = None,
    ) -> CacheEntryMetadata:
        """Stores an already-optimized GLB (produced by Blocks 1-4) under `cache_key`.

        `optimized_asset_path` is only ever read (via `shutil.copy2`)
        - it is never modified, moved, or deleted. The asset and its
        metadata are written to temporary siblings first and promoted
        into place only once complete (see module docstring's
        "ATOMICITY" section), so a failure partway through never
        leaves a corrupt or half-written entry visible to `lookup()`.

        Overwriting an existing entry preserves that entry's original
        `created_at`.

        Args:
            cache_key: The key to store under (see `CacheKey.build`).
            source_identity: The source asset identity this entry was
                built from, recorded in the entry's metadata.
            optimized_asset_path: Path to the already-optimized local
                GLB file to cache. Must exist and be a regular file.
            selected_candidate: The Block 4 candidate name that
                produced this result, if known. Purely informational.

        Returns:
            The stored `CacheEntryMetadata`.

        Raises:
            InvalidCacheKeyError: If `cache_key` would resolve outside
                the cache directory.
            SourceAssetNotFoundError: If `optimized_asset_path` does
                not exist or is not a regular file.
        """
        asset_path, metadata_path = self._paths_for(cache_key)

        source = Path(optimized_asset_path)
        if not source.is_file():
            raise SourceAssetNotFoundError(f"Optimized asset not found: {source}.")

        now = time.time()
        existing = self.lookup(cache_key)
        created_at = existing.metadata.created_at if existing.metadata is not None else now

        metadata = CacheEntryMetadata(
            cache_key=cache_key.value,
            source_identity=source_identity,
            optimizer_version=OPTIMIZER_VERSION,
            asset_filename=asset_path.name,
            selected_candidate=selected_candidate,
            created_at=created_at,
            updated_at=now,
        )

        temp_asset_path = asset_path.with_name(asset_path.name + ".part")
        temp_metadata_path = metadata_path.with_name(metadata_path.name + ".part")

        try:
            shutil.copy2(source, temp_asset_path)
            temp_metadata_path.write_text(json.dumps(metadata.to_dict(), sort_keys=True), encoding="utf-8")
        except OSError:
            temp_asset_path.unlink(missing_ok=True)
            temp_metadata_path.unlink(missing_ok=True)
            raise

        # Asset is promoted first, metadata last: `lookup()` only
        # considers an entry present once metadata exists at its final
        # path, so a crash between these two renames leaves a clean
        # MISS (plus a harmless orphaned asset file overwritten by the
        # next successful store), never a partially-valid HIT.
        temp_asset_path.replace(asset_path)
        temp_metadata_path.replace(metadata_path)

        return metadata

    # --- Removal --------------------------------------------------------

    def invalidate(self, cache_key: CacheKey) -> bool:
        """Removes the cache entry (asset + metadata) for `cache_key`, if present.

        Removing only the metadata (or only the asset) would leave a
        entry `lookup()` already treats as `INVALID`/`MISS` rather than
        a phantom `HIT`, but both are removed here regardless, so
        `invalidate()` also cleans up an already-broken entry rather
        than leaving orphaned files behind.

        Args:
            cache_key: The key to remove.

        Returns:
            True if anything was removed, False if nothing was present.
        """
        asset_path, metadata_path = self._paths_for(cache_key)
        removed = False
        if metadata_path.exists():
            metadata_path.unlink()
            removed = True
        if asset_path.exists():
            asset_path.unlink()
            removed = True
        return removed

    def clear(self) -> int:
        """Removes every entry in this cache.

        Returns:
            The number of metadata entries removed (one per logical
            cache entry, regardless of whether its asset file was also
            present).
        """
        removed = 0
        for metadata_path in self._cache_dir.glob(f"*{_METADATA_EXTENSION}"):
            asset_path = metadata_path.with_suffix(_ASSET_EXTENSION)
            metadata_path.unlink()
            if asset_path.exists():
                asset_path.unlink()
            removed += 1
        # Sweep any orphaned asset files (e.g. from an interrupted
        # store that never reached the metadata rename) too, so
        # `clear()` genuinely empties the cache directory.
        for asset_path in self._cache_dir.glob(f"*{_ASSET_EXTENSION}"):
            asset_path.unlink()
        return removed
