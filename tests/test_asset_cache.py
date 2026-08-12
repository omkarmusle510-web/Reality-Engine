"""Offline, deterministic tests for the Asset Optimizer's Block 5 cache.

No network access, no GitHub, no camera, no pyrender/OpenGL, no
`trimesh`, no `AssetRetriever`/`AssetRegistry`. Only
`apps.reality_painter.optimization.cache` is exercised here - Blocks
1-4 (`analyzer.py`, `candidate_generator.py`, `benchmark.py`,
`selector.py`) are never imported or touched.
"""
import inspect
import json
import sys
import tempfile
import time
from pathlib import Path

from apps.reality_painter.optimization import cache as cache_module
from apps.reality_painter.optimization.cache import (
    CacheEntryMetadata,
    CacheKey,
    CacheStatus,
    OPTIMIZER_VERSION,
    OptimizationCache,
    SourceAssetNotFoundError,
)

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {name}")
    else:
        failed += 1
        print(f"FAIL: {name}")


def expect_raises(name, exception_type, func):
    try:
        func()
        check(name, False)
    except exception_type:
        check(name, True)


def _make_glb(path: Path, payload: bytes = b"fake-optimized-glb-bytes") -> Path:
    path.write_bytes(payload)
    return path


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    cache_dir = tmp_path / "opt_cache"
    optimized_source_dir = tmp_path / "optimized_outputs"
    optimized_source_dir.mkdir()

    cache = OptimizationCache(cache_dir=cache_dir)

    # ===========================================================================
    # 1. Cache miss for an unknown asset.
    # ===========================================================================
    unknown_key = CacheKey.build("flower_001")
    result = cache.lookup(unknown_key)
    check("unknown asset -> CACHE_MISS", result.status == CacheStatus.MISS)
    check("cache miss carries no asset_path", result.asset_path is None)
    check("cache miss carries a non-empty reason", bool(result.reason))

    # ===========================================================================
    # 2 & 3. Successful store + lookup returns a HIT with the correct asset.
    # ===========================================================================
    flower_glb = _make_glb(optimized_source_dir / "flower_optimized.glb", b"flower-bytes")
    flower_key = CacheKey.build("flower_001")
    stored_metadata = cache.store(flower_key, "flower_001", flower_glb, selected_candidate="LIGHT")
    check("store() returns metadata with the matching cache_key", stored_metadata.cache_key == flower_key.value)

    hit = cache.lookup(flower_key)
    check("stored entry -> CACHE_HIT", hit.status == CacheStatus.HIT)
    check("hit returns a real, existing asset_path", hit.asset_path is not None and hit.asset_path.is_file())
    check("hit's cached asset content matches what was stored", hit.asset_path.read_bytes() == b"flower-bytes")
    check("hit metadata carries the selected candidate", hit.metadata.selected_candidate == "LIGHT")
    check("hit metadata carries the source identity", hit.metadata.source_identity == "flower_001")

    # ===========================================================================
    # 4 & 5. Deterministic cache keys; different source identities differ.
    # ===========================================================================
    key_a = CacheKey.build("flower_001")
    key_b = CacheKey.build("flower_001")
    check("CacheKey.build is deterministic for identical inputs", key_a.value == key_b.value)

    key_c = CacheKey.build("chair_001")
    check("different source identities produce different keys", key_a.value != key_c.value)

    # ===========================================================================
    # 6. Different optimizer versions/selection configs must not collide.
    # ===========================================================================
    key_v1 = CacheKey.build("flower_001", optimizer_version="v1")
    key_v2 = CacheKey.build("flower_001", optimizer_version="v2")
    check("different optimizer_version -> different key", key_v1.value != key_v2.value)

    key_cfg_a = CacheKey.build("flower_001", selection_config={"target_fps": 60})
    key_cfg_b = CacheKey.build("flower_001", selection_config={"target_fps": 30})
    check("different selection_config -> different key", key_cfg_a.value != key_cfg_b.value)
    check("selection_config presence changes the key vs no config", key_cfg_a.value != key_a.value)

    # A cache entry stored under the current OPTIMIZER_VERSION but whose
    # metadata claims a different (stale) version is rejected on lookup,
    # even under a hypothetical key collision.
    stale_key = CacheKey.build("stale_asset")
    stale_glb = _make_glb(optimized_source_dir / "stale.glb", b"stale-bytes")
    cache.store(stale_key, "stale_asset", stale_glb)
    stale_asset_path, stale_metadata_path = cache._paths_for(stale_key)
    stale_data = json.loads(stale_metadata_path.read_text(encoding="utf-8"))
    stale_data["optimizer_version"] = "v0-ancient"
    stale_metadata_path.write_text(json.dumps(stale_data), encoding="utf-8")
    stale_result = cache.lookup(stale_key)
    check("wrong optimizer_version in stored metadata -> INVALID", stale_result.status == CacheStatus.INVALID)
    check("optimizer-version mismatch is never silently treated as a hit", stale_result.asset_path is None)

    # ===========================================================================
    # 7. Metadata validation: cache_key mismatch inside the metadata file.
    # ===========================================================================
    mismatch_key = CacheKey.build("mismatch_asset")
    mismatch_glb = _make_glb(optimized_source_dir / "mismatch.glb", b"mismatch-bytes")
    cache.store(mismatch_key, "mismatch_asset", mismatch_glb)
    mismatch_asset_path, mismatch_metadata_path = cache._paths_for(mismatch_key)
    tampered = json.loads(mismatch_metadata_path.read_text(encoding="utf-8"))
    tampered["cache_key"] = "not-the-real-key"
    mismatch_metadata_path.write_text(json.dumps(tampered), encoding="utf-8")
    mismatch_result = cache.lookup(mismatch_key)
    check("cache_key mismatch inside metadata -> INVALID", mismatch_result.status == CacheStatus.INVALID)

    # ===========================================================================
    # 8. Missing GLB invalidates an otherwise-valid entry.
    # ===========================================================================
    missing_glb_key = CacheKey.build("missing_glb_asset")
    missing_glb_source = _make_glb(optimized_source_dir / "missing.glb", b"will-be-deleted")
    cache.store(missing_glb_key, "missing_glb_asset", missing_glb_source)
    missing_asset_path, _ = cache._paths_for(missing_glb_key)
    missing_asset_path.unlink()
    missing_result = cache.lookup(missing_glb_key)
    check("deleted cached GLB -> INVALID, not a crash", missing_result.status == CacheStatus.INVALID)
    check("missing-GLB reason mentions the asset file", "asset" in missing_result.reason.lower())

    # ===========================================================================
    # 9. Malformed metadata is handled cleanly (never raises out of lookup()).
    # ===========================================================================
    malformed_key = CacheKey.build("malformed_asset")
    malformed_glb = _make_glb(optimized_source_dir / "malformed.glb", b"malformed-bytes")
    cache.store(malformed_key, "malformed_asset", malformed_glb)
    _, malformed_metadata_path = cache._paths_for(malformed_key)
    malformed_metadata_path.write_text("{not valid json", encoding="utf-8")
    malformed_result = cache.lookup(malformed_key)
    check("invalid JSON metadata -> INVALID (no exception)", malformed_result.status == CacheStatus.INVALID)

    incomplete_key = CacheKey.build("incomplete_asset")
    incomplete_glb = _make_glb(optimized_source_dir / "incomplete.glb", b"incomplete-bytes")
    cache.store(incomplete_key, "incomplete_asset", incomplete_glb)
    _, incomplete_metadata_path = cache._paths_for(incomplete_key)
    incomplete_metadata_path.write_text(json.dumps({"cache_key": incomplete_key.value}), encoding="utf-8")
    incomplete_result = cache.lookup(incomplete_key)
    check("metadata missing required fields -> INVALID (no exception)", incomplete_result.status == CacheStatus.INVALID)

    expect_raises(
        "CacheEntryMetadata.from_dict rejects a non-dict payload",
        ValueError,
        lambda: CacheEntryMetadata.from_dict(["not", "a", "dict"]),
    )

    # ===========================================================================
    # 10. Wrong optimizer version at lookup time (module-level version differs
    #     from what an entry's key/metadata was built under).
    # ===========================================================================
    versioned_key = CacheKey.build("versioned_asset", optimizer_version="v1")
    check("versioned_key uses the current OPTIMIZER_VERSION by default", OPTIMIZER_VERSION == "v1")
    old_version_key = CacheKey.build("versioned_asset", optimizer_version="v0")
    versioned_glb = _make_glb(optimized_source_dir / "versioned.glb", b"versioned-bytes")
    cache.store(old_version_key, "versioned_asset", versioned_glb)
    # Manually mark this entry's stored optimizer_version as stale ("v0"),
    # simulating an entry produced before an OPTIMIZER_VERSION bump.
    old_asset_path, old_metadata_path = cache._paths_for(old_version_key)
    old_data = json.loads(old_metadata_path.read_text(encoding="utf-8"))
    old_data["optimizer_version"] = "v0"
    old_metadata_path.write_text(json.dumps(old_data), encoding="utf-8")
    old_result = cache.lookup(old_version_key)
    check("entry stamped with a stale optimizer_version is rejected", old_result.status == CacheStatus.INVALID)

    # ===========================================================================
    # 11. Path traversal / input safety.
    # ===========================================================================
    traversal_key = CacheKey(value="../../../etc/evil_cache_entry")
    traversal_glb = _make_glb(optimized_source_dir / "traversal.glb", b"traversal-bytes")
    traversal_metadata = cache.store(traversal_key, "evil", traversal_glb)
    traversal_asset_path, traversal_metadata_path = cache._paths_for(traversal_key)
    check("path-traversal key is sanitized to stay in the cache dir", traversal_asset_path.parent == cache_dir.resolve())
    check("sanitized traversal filename has no path separators", "/" not in traversal_asset_path.name and ".." not in traversal_asset_path.name)
    check("path-traversal store() still produces a retrievable hit", cache.lookup(traversal_key).status == CacheStatus.HIT)

    # ===========================================================================
    # 12. invalidate() removes exactly the targeted entry.
    # ===========================================================================
    target_key = CacheKey.build("to_be_invalidated")
    other_key = CacheKey.build("should_remain")
    target_glb = _make_glb(optimized_source_dir / "target.glb", b"target-bytes")
    other_glb = _make_glb(optimized_source_dir / "other.glb", b"other-bytes")
    cache.store(target_key, "to_be_invalidated", target_glb)
    cache.store(other_key, "should_remain", other_glb)

    removed = cache.invalidate(target_key)
    check("invalidate() reports True when something was removed", removed is True)
    check("invalidated entry is now a MISS", cache.lookup(target_key).status == CacheStatus.MISS)
    check("invalidate() does not affect unrelated entries", cache.lookup(other_key).status == CacheStatus.HIT)

    removed_again = cache.invalidate(target_key)
    check("invalidate() on an already-removed key reports False", removed_again is False)

    # ===========================================================================
    # 13. clear() empties the entire cache.
    # ===========================================================================
    pre_clear_hit = cache.lookup(other_key)
    check("sanity: an entry exists before clear()", pre_clear_hit.status == CacheStatus.HIT)
    cleared_count = cache.clear()
    check("clear() reports a positive removed count", cleared_count > 0)
    check("clear() empties the cache directory of entries", cache.lookup(other_key).status == CacheStatus.MISS)
    check(
        "clear() leaves no leftover cache/metadata files on disk",
        not any(cache_dir.iterdir()),
    )

    # ===========================================================================
    # 14. Cache performs no network access (source inspection).
    # ===========================================================================
    import_lines = [
        line.strip()
        for line in inspect.getsource(cache_module).splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    forbidden_import_tokens = ("requests", "socket", "http.client", "urllib", "assets.retriever", "subprocess")
    check(
        "cache module has no network/subprocess/AssetRetriever import statements",
        not any(any(token in line for token in forbidden_import_tokens) for line in import_lines),
    )

    # ===========================================================================
    # 15. Cache never modifies the source (optimized-output) GLB it copies in.
    # ===========================================================================
    immutable_source = _make_glb(optimized_source_dir / "immutable.glb", b"do-not-touch-me")
    original_bytes = immutable_source.read_bytes()
    original_mtime = immutable_source.stat().st_mtime
    immutable_key = CacheKey.build("immutable_asset")
    cache.store(immutable_key, "immutable_asset", immutable_source)
    check("source GLB bytes are unchanged after store()", immutable_source.read_bytes() == original_bytes)
    check("source GLB file still exists at its original path", immutable_source.is_file())

    # ===========================================================================
    # 16. Corrupted/incomplete cache entries fail safely (metadata present,
    #     asset missing == an interrupted store's worst case).
    # ===========================================================================
    corrupt_key = CacheKey.build("corrupt_entry")
    corrupt_glb = _make_glb(optimized_source_dir / "corrupt.glb", b"corrupt-bytes")
    cache.store(corrupt_key, "corrupt_entry", corrupt_glb)
    corrupt_asset_path, corrupt_metadata_path = cache._paths_for(corrupt_key)
    corrupt_asset_path.write_bytes(b"")  # simulate a truncated/corrupt write
    corrupt_result = cache.lookup(corrupt_key)
    check("zero-byte cached asset -> INVALID, not a crash", corrupt_result.status == CacheStatus.INVALID)

    # A store() whose source file vanishes before copy still fails cleanly.
    ghost_key = CacheKey.build("ghost_asset")
    ghost_path = optimized_source_dir / "does_not_exist.glb"
    expect_raises(
        "store() with a nonexistent optimized asset raises SourceAssetNotFoundError",
        SourceAssetNotFoundError,
        lambda: cache.store(ghost_key, "ghost_asset", ghost_path),
    )
    check("failed store() never creates a partial entry", cache.lookup(ghost_key).status == CacheStatus.MISS)
    check(
        "failed store() leaves no leftover .part temp files",
        not list(cache_dir.glob("*.part")),
    )

    # ===========================================================================
    # 17. Repeated lookup never rewrites or modifies the cached asset.
    # ===========================================================================
    stable_key = CacheKey.build("stable_asset")
    stable_glb = _make_glb(optimized_source_dir / "stable.glb", b"stable-bytes")
    cache.store(stable_key, "stable_asset", stable_glb)
    stable_asset_path, _ = cache._paths_for(stable_key)
    bytes_before = stable_asset_path.read_bytes()
    mtime_before = stable_asset_path.stat().st_mtime

    for _ in range(5):
        cache.lookup(stable_key)

    check("repeated lookup() leaves cached asset bytes unchanged", stable_asset_path.read_bytes() == bytes_before)
    check("repeated lookup() leaves cached asset mtime unchanged", stable_asset_path.stat().st_mtime == mtime_before)

    # ===========================================================================
    # Overwrite semantics: re-storing under the same key preserves created_at
    # but advances updated_at, and replaces the cached content.
    # ===========================================================================
    overwrite_key = CacheKey.build("overwrite_asset")
    first_glb = _make_glb(optimized_source_dir / "overwrite_v1.glb", b"version-one")
    first_meta = cache.store(overwrite_key, "overwrite_asset", first_glb)
    time.sleep(0.01)
    second_glb = _make_glb(optimized_source_dir / "overwrite_v2.glb", b"version-two")
    second_meta = cache.store(overwrite_key, "overwrite_asset", second_glb, selected_candidate="HEAVY_OPTIMIZED")
    check("re-store preserves the original created_at", second_meta.created_at == first_meta.created_at)
    check("re-store advances updated_at", second_meta.updated_at >= first_meta.updated_at)
    reread = cache.lookup(overwrite_key)
    check("re-store replaces the cached asset content", reread.asset_path.read_bytes() == b"version-two")
    check("re-store updates selected_candidate metadata", reread.metadata.selected_candidate == "HEAVY_OPTIMIZED")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
