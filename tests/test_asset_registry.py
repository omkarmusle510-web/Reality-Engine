"""Offline, deterministic tests for the Phase 12A asset registry.

No network access, no GitHub/Cloudflare/AWS calls - only local JSON
fixtures and in-memory dicts.
"""
import json
import sys
import tempfile
from pathlib import Path

from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.schema import Asset, AssetSource, AssetValidationError

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


def _write_registry(tmp_dir: Path, payload: dict) -> Path:
    path = tmp_dir / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


FLOWER = {
    "id": "flower_001",
    "name": "Flower",
    "category": "plants",
    "tags": ["flower", "plant", "garden"],
    "format": "glb",
    "source": {"type": "github", "repository": "owner/repository", "path": "models/flower.glb"},
    "license": "CC0",
}

CHAIR = {
    "id": "chair_001",
    "name": "Wooden Chair",
    "category": "furniture",
    "tags": ["chair", "furniture", "wood"],
    "format": "glb",
    "source": {"type": "local", "path": "/local/models/chair.glb"},
}

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    # 1. Loading an empty registry.
    empty_path = _write_registry(tmp_path, {"assets": []})
    empty_registry = AssetRegistry.load(empty_path)
    check("loads an empty registry", len(empty_registry) == 0)
    check("list_assets() on empty registry returns []", empty_registry.list_assets() == [])

    # Bundled default registry.json also loads and is empty.
    default_registry = AssetRegistry.load()
    check("bundled default registry.json loads cleanly", len(default_registry) == 0)

    # 2. Loading valid assets.
    valid_path = _write_registry(tmp_path, {"assets": [FLOWER, CHAIR]})
    registry = AssetRegistry.load(valid_path)
    check("loads valid assets", len(registry) == 2)
    check("parsed asset preserves id/name", registry.get_asset("flower_001").name == "Flower")
    check("parsed source round-trips flat shape", registry.get_asset("flower_001").source.to_dict() == FLOWER["source"])
    check("parsed asset.to_dict() round-trips", registry.get_asset("flower_001").to_dict()["tags"] == FLOWER["tags"])

    # 3. Getting an asset by ID.
    check("get_asset() returns the right asset", registry.get_asset("chair_001").category == "furniture")

    # 4. Searching by name.
    name_hits = registry.search_assets("Flower")
    check("search_assets() matches by name (case-insensitive)", [a.id for a in name_hits] == ["flower_001"])

    # 5. Searching by tags.
    tag_hits = registry.search_assets("wood")
    check("search_assets() matches by tag substring", [a.id for a in tag_hits] == ["chair_001"])

    # 6. Filtering by category.
    plant_assets = registry.list_assets(category="plants")
    check("list_assets(category=) filters correctly", [a.id for a in plant_assets] == ["flower_001"])
    check("list_assets() with unknown category returns []", registry.list_assets(category="vehicles") == [])

    # Deterministic ordering / empty query behavior.
    check("search_assets('') returns nothing (not the whole registry)", registry.search_assets("   ") == [])
    check("list_assets() ordering is deterministic (sorted by id)", [a.id for a in registry.list_assets()] == ["chair_001", "flower_001"])

    # 7. Rejecting malformed entries.
    missing_field = dict(FLOWER)
    del missing_field["category"]
    try:
        Asset.from_dict(missing_field)
        check("rejects entry missing required field", False)
    except AssetValidationError:
        check("rejects entry missing required field", True)

    bad_source = dict(FLOWER, source={"repository": "owner/repo"})  # missing "type"
    try:
        Asset.from_dict(bad_source)
        check("rejects source missing 'type'", False)
    except AssetValidationError:
        check("rejects source missing 'type'", True)

    bad_source_shape = dict(FLOWER, source="not-an-object")
    try:
        AssetSource.from_dict(bad_source_shape["source"])
        check("rejects non-object source", False)
    except AssetValidationError:
        check("rejects non-object source", True)

    bad_tags = dict(FLOWER, id="flower_002", tags="not-a-list")
    try:
        Asset.from_dict(bad_tags)
        check("rejects non-list tags", False)
    except AssetValidationError:
        check("rejects non-list tags", True)

    malformed_path = _write_registry(tmp_path, {"assets": [missing_field]})
    try:
        AssetRegistry.load(malformed_path)
        check("AssetRegistry.load() propagates validation errors", False)
    except AssetValidationError:
        check("AssetRegistry.load() propagates validation errors", True)

    # 8. Rejecting duplicate IDs.
    duplicate = dict(FLOWER)
    dup_path = _write_registry(tmp_path, {"assets": [FLOWER, duplicate]})
    try:
        AssetRegistry.load(dup_path)
        check("rejects duplicate asset ids", False)
    except AssetValidationError:
        check("rejects duplicate asset ids", True)

    # 9. Missing asset id returns a clean result (None), not an exception.
    check("get_asset() on missing id returns None cleanly", registry.get_asset("does_not_exist") is None)

    # Top-level shape validation.
    bad_shape_path = _write_registry(tmp_path, {"items": []})
    try:
        AssetRegistry.load(bad_shape_path)
        check("rejects registry file missing 'assets' key", False)
    except AssetValidationError:
        check("rejects registry file missing 'assets' key", True)

    # Missing file surfaces a clean, standard error.
    try:
        AssetRegistry.load(tmp_path / "does_not_exist.json")
        check("missing registry file raises FileNotFoundError", False)
    except FileNotFoundError:
        check("missing registry file raises FileNotFoundError", True)

    # from_list() works directly on in-memory dicts (no file I/O).
    in_memory = AssetRegistry.from_list([FLOWER])
    check("from_list() builds a registry without touching disk", len(in_memory) == 1)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
