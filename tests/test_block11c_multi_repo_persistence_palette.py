"""Tests for Block 11C Multi-Repo + Permanent Asset Storage + 12-Color Palette.

Verifies:
1. All 4 configured repositories are scanned in deterministic order.
2. One failed repository does not stop later repositories.
3. No duplicate registry entries across multiple repositories.
4. External asset can be promoted only when redistribution is allowed.
5. Prohibited/unknown-license asset is not promoted.
6. Promoted asset is available through our existing repository/cache path.
7. Duplicate promotion is prevented.
8. All 12 palette colors exist and update ToolState correctly.
9. Real GitHub integration check.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import apps.reality_painter.assets.auto_discovery as auto_discovery_module
from apps.reality_painter.assets.auto_discovery import (
    configured_repositories,
    ensure_discovered,
    external_repositories,
    primary_repository,
    reset_discovery_state,
)
from apps.reality_painter.assets.promotion import (
    PromotionError,
    PromotionLicenseError,
    check_promotion_eligibility,
    is_already_promoted,
    is_promotable,
    list_promoted_assets,
    promote_asset,
    validate_asset_for_promotion,
)
from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.schema import Asset, AssetSource
from apps.reality_painter.inspection.asset_resolver import (
    AssetResolutionStatus,
    resolve_asset,
)
from apps.reality_painter.sketch import ToolState, _COMPACT_PALETTE

passed = 0
failed = 0


def check(name: str, condition: bool) -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {name}")
    else:
        failed += 1
        print(f"FAIL: {name}")


class FakeResponse:
    def __init__(self, status_code: int, json_data: Any = None, headers: Any = None) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}

    def json(self) -> Any:
        return self._json_data


class FakeSession:
    def __init__(self, routes: Dict[str, Any]) -> None:
        self._routes = routes
        self.requested_urls: List[str] = []

    def get(self, url: str, params: Any = None, headers: Any = None, timeout: Any = None, stream: bool = False) -> Any:
        self.requested_urls.append(url)
        route = self._routes.get(url)
        if route is None:
            raise AssertionError(f"Unexpected URL in test: {url}")
        return route


# --- Mock routes for 4 repositories ---

PRIMARY_REPO = "omkarmusle510-web/reality-engine-assets"
REPO_KHRONOS = "KhronosGroup/glTF-Sample-Assets"
REPO_BABYLON = "BabylonJS/Assets"
REPO_THREEJS = "mrdoob/three.js"
REPO_FAILING = "failing/nonexistent-repo"

mock_routes = {
    # Primary
    f"https://api.github.com/repos/{PRIMARY_REPO}": FakeResponse(200, {"default_branch": "main", "license": None}),
    f"https://api.github.com/repos/{PRIMARY_REPO}/git/trees/main": FakeResponse(200, {
        "tree": [
            {"path": "nature/sun.glb", "type": "blob"},
            {"path": "nature/flower.glb", "type": "blob"},
            {"path": "vehicles/simple_sport_car.glb", "type": "blob"},
        ]
    }),
    # Khronos
    f"https://api.github.com/repos/{REPO_KHRONOS}": FakeResponse(200, {"default_branch": "main", "license": {"spdx_id": "CC-BY-4.0"}}),
    f"https://api.github.com/repos/{REPO_KHRONOS}/git/trees/main": FakeResponse(200, {
        "tree": [
            {"path": "Models/Avocado/glTF-Binary/Avocado.glb", "type": "blob"},
            {"path": "Models/Box/glTF-Binary/Box.glb", "type": "blob"},
        ]
    }),
    # BabylonJS
    f"https://api.github.com/repos/{REPO_BABYLON}": FakeResponse(200, {"default_branch": "master", "license": {"spdx_id": "Apache-2.0"}}),
    f"https://api.github.com/repos/{REPO_BABYLON}/git/trees/master": FakeResponse(200, {
        "tree": [
            {"path": "meshes/chair.glb", "type": "blob"},
            {"path": "meshes/shark.glb", "type": "blob"},
        ]
    }),
    # Three.js
    f"https://api.github.com/repos/{REPO_THREEJS}": FakeResponse(200, {"default_branch": "dev", "license": {"spdx_id": "MIT"}}),
    f"https://api.github.com/repos/{REPO_THREEJS}/git/trees/dev": FakeResponse(200, {
        "tree": [
            {"path": "examples/models/gltf/Soldier.glb", "type": "blob"},
            {"path": "examples/models/gltf/Horse.glb", "type": "blob"},
        ]
    }),
    # Failing
    f"https://api.github.com/repos/{REPO_FAILING}": FakeResponse(404, {}),
}


def test_1_all_repos_scanned_in_order():
    """1. All 4 configured repositories are scanned in deterministic order."""
    reset_discovery_state()
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry_path = Path(tmp_dir) / "registry.json"
        repos = [
            {"repository": PRIMARY_REPO, "path": ""},
            {"repository": REPO_KHRONOS, "path": ""},
            {"repository": REPO_BABYLON, "path": "meshes"},
            {"repository": REPO_THREEJS, "path": "examples/models/gltf"},
        ]

        orig = auto_discovery_module.configured_repositories
        auto_discovery_module.configured_repositories = lambda: repos
        try:
            registry = AssetRegistry()
            session = FakeSession(mock_routes)
            added = ensure_discovered(registry, session=session, registry_path=registry_path)

            check("all 4 repositories were scanned", added == 9)  # 3+2+2+2
            check("primary repo scanned first", session.requested_urls[0] == f"https://api.github.com/repos/{PRIMARY_REPO}")

            # Verify order: Primary URLs before Khronos, Khronos before Babylon, Babylon before Three.js
            primary_idx = next(i for i, u in enumerate(session.requested_urls) if PRIMARY_REPO in u)
            khronos_idx = next(i for i, u in enumerate(session.requested_urls) if REPO_KHRONOS in u)
            babylon_idx = next(i for i, u in enumerate(session.requested_urls) if REPO_BABYLON in u)
            threejs_idx = next(i for i, u in enumerate(session.requested_urls) if REPO_THREEJS in u)
            check("deterministic order: primary < khronos < babylon < threejs",
                  primary_idx < khronos_idx < babylon_idx < threejs_idx)

            check("all assets registered", len(registry) == 9)
        finally:
            auto_discovery_module.configured_repositories = orig


def test_2_failed_repo_does_not_stop_others():
    """2. One failed repository does not stop later repositories."""
    reset_discovery_state()
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry_path = Path(tmp_dir) / "registry.json"
        repos = [
            {"repository": PRIMARY_REPO, "path": ""},
            {"repository": REPO_FAILING, "path": ""},
            {"repository": REPO_KHRONOS, "path": ""},
            {"repository": REPO_BABYLON, "path": "meshes"},
        ]

        orig = auto_discovery_module.configured_repositories
        auto_discovery_module.configured_repositories = lambda: repos
        try:
            registry = AssetRegistry()
            session = FakeSession(mock_routes)
            added = ensure_discovered(registry, session=session, registry_path=registry_path)

            check("failing repo was attempted", f"https://api.github.com/repos/{REPO_FAILING}" in session.requested_urls)
            check("other repos still scanned after failure", added == 7)  # 3+2+2 (failing skipped)
            check("khronos assets present", len(registry.search_assets("avocado")) > 0)
            check("babylon assets present", len(registry.search_assets("chair")) > 0)
        finally:
            auto_discovery_module.configured_repositories = orig


def test_3_no_duplicate_entries():
    """3. No duplicate registry entries across multiple scan calls."""
    reset_discovery_state()
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry_path = Path(tmp_dir) / "registry.json"
        repos = [
            {"repository": PRIMARY_REPO, "path": ""},
            {"repository": REPO_KHRONOS, "path": ""},
        ]

        orig = auto_discovery_module.configured_repositories
        auto_discovery_module.configured_repositories = lambda: repos
        try:
            registry = AssetRegistry()
            session = FakeSession(mock_routes)

            added_1 = ensure_discovered(registry, session=session, registry_path=registry_path)
            count_1 = len(registry)

            added_2 = ensure_discovered(registry, session=session, registry_path=registry_path)
            count_2 = len(registry)

            check("first scan adds assets", added_1 == 5)
            check("second scan adds zero", added_2 == 0)
            check("no duplicate entries", count_1 == count_2)
        finally:
            auto_discovery_module.configured_repositories = orig


def test_4_promotable_asset_can_be_promoted():
    """4. External asset can be promoted only when redistribution is allowed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        staging = Path(tmp_dir) / "promoted"
        log_path = staging / "promotions.json"

        # Create a mock cached file
        cache_file = Path(tmp_dir) / "cached_model.glb"
        cache_file.write_bytes(b"FAKE_GLB_DATA")

        asset = Asset.from_dict({
            "id": "github_khronos_avocado",
            "name": "Avocado",
            "category": "nature",
            "format": "glb",
            "tags": ["avocado"],
            "source": {"type": "github", "repository": REPO_KHRONOS, "path": "Models/Avocado.glb"},
            "license": "CC-BY-4.0",
        })

        check("CC-BY-4.0 is promotable", is_promotable(asset))

        promoted = promote_asset(asset, cache_file, staging_dir=staging, log_path=log_path, push_to_remote=False)
        check("promoted asset has source pointing to primary repo",
              promoted.source.details.get("repository") == "omkarmusle510-web/reality-engine-assets")
        check("promoted asset preserves license", promoted.license == "CC-BY-4.0")
        check("promoted asset preserves name", promoted.name == "Avocado")

        staged_file = staging / promoted.source.details["path"]
        check("staged file exists on disk", staged_file.is_file())
        check("staged file matches source", staged_file.read_bytes() == b"FAKE_GLB_DATA")

        promotions = list_promoted_assets(log_path=log_path)
        check("promotion logged", len(promotions) == 1)
        check("promotion log records original source",
              promotions[0]["original_source"]["repository"] == REPO_KHRONOS)


def test_5_prohibited_license_not_promoted():
    """5. Prohibited/unknown-license asset is not promoted."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        staging = Path(tmp_dir) / "promoted"
        log_path = staging / "promotions.json"
        cache_file = Path(tmp_dir) / "cached_model.glb"
        cache_file.write_bytes(b"FAKE_GLB_DATA")

        no_license_asset = Asset.from_dict({
            "id": "unknown_model",
            "name": "Unknown Model",
            "category": "objects",
            "format": "glb",
            "tags": ["model"],
            "source": {"type": "github", "repository": "external/repo", "path": "model.glb"},
        })
        check("no-license asset is NOT promotable", not is_promotable(no_license_asset))

        rejected = False
        try:
            promote_asset(no_license_asset, cache_file, staging_dir=staging, log_path=log_path)
        except PromotionLicenseError:
            rejected = True
        check("no-license asset raises PromotionLicenseError", rejected)

        nc_asset = Asset.from_dict({
            "id": "nc_model",
            "name": "NC Model",
            "category": "objects",
            "format": "glb",
            "tags": ["model"],
            "source": {"type": "github", "repository": "external/repo", "path": "model.glb"},
            "license": "CC-BY-NC-4.0",
        })
        check("non-commercial license is NOT promotable", not is_promotable(nc_asset))

        rejected_nc = False
        try:
            promote_asset(nc_asset, cache_file, staging_dir=staging, log_path=log_path)
        except PromotionLicenseError:
            rejected_nc = True
        check("non-commercial asset raises PromotionLicenseError", rejected_nc)


def test_6_promoted_asset_available_from_primary():
    """6. Promoted asset is available through our primary repository path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        staging = Path(tmp_dir) / "promoted"
        log_path = staging / "promotions.json"
        cache_file = Path(tmp_dir) / "cached.glb"
        cache_file.write_bytes(b"GLB_MODEL_DATA")

        asset = Asset.from_dict({
            "id": "external_chair",
            "name": "Chair",
            "category": "furniture",
            "format": "glb",
            "tags": ["chair"],
            "source": {"type": "github", "repository": REPO_BABYLON, "path": "meshes/chair.glb"},
            "license": "Apache-2.0",
        })

        promoted = promote_asset(asset, cache_file, staging_dir=staging, log_path=log_path, push_to_remote=False)

        # Register promoted asset into a registry
        registry = AssetRegistry()
        registered = registry.register(promoted)
        check("promoted asset registers into AssetRegistry", registered)

        # Search for it by name
        matches = registry.search_assets("Chair")
        check("promoted asset found by search", len(matches) > 0)
        check("promoted source is primary repo",
              matches[0].source.details.get("repository") == "omkarmusle510-web/reality-engine-assets")


def test_7_duplicate_promotion_prevented():
    """7. Duplicate promotion is prevented."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        staging = Path(tmp_dir) / "promoted"
        log_path = staging / "promotions.json"
        cache_file = Path(tmp_dir) / "cached.glb"
        cache_file.write_bytes(b"GLB_DATA")

        asset = Asset.from_dict({
            "id": "dup_asset",
            "name": "Dup Asset",
            "category": "nature",
            "format": "glb",
            "tags": ["dup"],
            "source": {"type": "github", "repository": "ext/repo", "path": "dup.glb"},
            "license": "MIT",
        })

        promoted_1 = promote_asset(asset, cache_file, staging_dir=staging, log_path=log_path, push_to_remote=False)
        promoted_2 = promote_asset(asset, cache_file, staging_dir=staging, log_path=log_path, push_to_remote=False)

        check("duplicate promotion returns same promoted id", promoted_1.id == promoted_2.id)
        check("promotion log has exactly 1 entry", len(list_promoted_assets(log_path=log_path)) == 1)
        check("is_already_promoted returns True", is_already_promoted(asset, log_path=log_path))


def test_8_twelve_palette_colors():
    """8. All 12 palette colors exist and update ToolState correctly."""
    check("palette has exactly 12 colors", len(_COMPACT_PALETTE) == 12)

    expected_names = [
        "Red", "Orange", "Yellow", "Green", "Sky Blue", "Blue",
        "Purple", "Pink", "Brown", "Black", "White", "Gray"
    ]
    actual_names = [name for name, _color in _COMPACT_PALETTE]
    check("palette names match expected 12", actual_names == expected_names)

    # All colors are valid BGR tuples
    for name, color in _COMPACT_PALETTE:
        check(f"{name} is a 3-tuple", isinstance(color, tuple) and len(color) == 3)

    # ToolState correctly applies each color
    ts = ToolState()
    for name, color in _COMPACT_PALETTE:
        ts.select_color(name, color)
        check(f"ToolState.select_color('{name}') updates color", ts.color == color)
        check(f"ToolState.color_name after '{name}' is correct", ts.color_name == name)


def test_9_configured_repositories_structure():
    """9. Verify configured_repositories returns all 4 repos in order."""
    reset_discovery_state()
    # Clear env vars to use defaults
    for var in ["REALITY_PAINTER_ASSET_REPOSITORIES", "REALITY_PAINTER_PRIMARY_REPOSITORY", "REALITY_PAINTER_EXTERNAL_REPOSITORIES"]:
        os.environ.pop(var, None)

    repos = configured_repositories()
    check("configured_repositories returns 4 repos by default", len(repos) == 4)

    repo_names = [r["repository"] for r in repos]
    check("primary is first", repo_names[0] == "omkarmusle510-web/reality-engine-assets")
    check("khronos is second", repo_names[1] == "KhronosGroup/glTF-Sample-Assets")
    check("babylon is third", repo_names[2] == "BabylonJS/Assets")
    check("threejs is fourth", repo_names[3] == "mrdoob/three.js")

    # Verify paths
    check("babylon path is meshes", repos[2]["path"] == "meshes")
    check("threejs path is examples/models/gltf", repos[3]["path"] == "examples/models/gltf")


def test_10_real_github_integration():
    """10. Real GitHub integration check against primary repository."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("SKIP: Real GitHub integration check (GITHUB_TOKEN not set)")
        return

    print("Running real GitHub integration check...")
    reset_discovery_state()
    registry = AssetRegistry()
    added = ensure_discovered(
        registry,
        repositories=[{"repository": "omkarmusle510-web/reality-engine-assets", "path": ""}],
    )
    check("real discovery found assets", added > 0)
    check("real discovery found sun", len(registry.search_assets("sun")) > 0)


if __name__ == "__main__":
    print("--- Running Block 11C Multi-Repo + Persistence + Palette Tests ---\n")
    test_1_all_repos_scanned_in_order()
    test_2_failed_repo_does_not_stop_others()
    test_3_no_duplicate_entries()
    test_4_promotable_asset_can_be_promoted()
    test_5_prohibited_license_not_promoted()
    test_6_promoted_asset_available_from_primary()
    test_7_duplicate_promotion_prevented()
    test_8_twelve_palette_colors()
    test_9_configured_repositories_structure()
    test_10_real_github_integration()

    print(f"\n==========================================")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"==========================================")
    sys.exit(1 if failed else 0)

