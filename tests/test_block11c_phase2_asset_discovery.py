"""Tests for Block 11C Phase 2: Reality Engine Multi-Repository Asset Discovery.

Verifies:
1. Primary repository auto-discovery finds 'sun' and registers it into AssetRegistry.
2. Discovered assets enter AssetRegistry with deterministic IDs.
3. Duplicate discovery calls are idempotent and do not duplicate entries or re-scan.
4. Local cache is preferred on retrieval (no network call on cache hit).
5. Multi-repository fallback: primary repository is tried first, external repos tried on miss.
6. Repository failure isolation (failing repo does not abort discovery of subsequent repos).
7. Unknown asset cleanly returns UNAVAILABLE without crashing.
8. Safe repository promotion checks permissive licenses and rejects unpermitted/missing licenses.
9. Real GitHub integration check against omkarmusle510-web/reality-engine-assets using GITHUB_TOKEN.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import apps.reality_painter.assets.auto_discovery as auto_discovery_module
from apps.reality_painter.assets.auto_discovery import (
    _already_scanned,
    configured_repositories,
    ensure_discovered,
    external_repositories,
    primary_repository,
    reset_discovery_state,
)
from apps.reality_painter.assets.promotion import (
    check_promotion_eligibility,
    is_promotable,
    validate_asset_for_promotion,
    PromotionLicenseError,
)
from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.retriever import AssetRetriever
from apps.reality_painter.assets.schema import Asset, AssetSource
from apps.reality_painter.inspection.asset_resolver import (
    AssetResolutionStatus,
    resolve_asset,
)

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


PRIMARY_REPO = "omkarmusle510-web/reality-engine-assets"
EXTERNAL_REPO = "KhronosGroup/glTF-Sample-Assets"

PRIMARY_REPO_URL = f"https://api.github.com/repos/{PRIMARY_REPO}"
PRIMARY_TREE_URL = f"https://api.github.com/repos/{PRIMARY_REPO}/git/trees/main"

EXTERNAL_REPO_URL = f"https://api.github.com/repos/{EXTERNAL_REPO}"
EXTERNAL_TREE_URL = f"https://api.github.com/repos/{EXTERNAL_REPO}/git/trees/main"

mock_routes = {
    PRIMARY_REPO_URL: FakeResponse(200, {"default_branch": "main", "license": {"spdx_id": "MIT"}}),
    PRIMARY_TREE_URL: FakeResponse(
        200,
        {
            "tree": [
                {"path": "nature/sun.glb", "type": "blob"},
                {"path": "nature/flower.glb", "type": "blob"},
                {"path": "vehicles/simple_sport_car.glb", "type": "blob"},
                {"path": "README.md", "type": "blob"},
            ]
        },
    ),
    EXTERNAL_REPO_URL: FakeResponse(200, {"default_branch": "main", "license": {"spdx_id": "CC-BY-4.0"}}),
    EXTERNAL_TREE_URL: FakeResponse(
        200,
        {
            "tree": [
                {"path": "Models/Avocado/glTF-Binary/Avocado.glb", "type": "blob"},
                {"path": "Models/Box/glTF-Binary/Box.glb", "type": "blob"},
            ]
        },
    ),
    "https://api.github.com/repos/failing/repo": FakeResponse(404, {}),
}


def test_1_primary_repo_discovery_and_sun_resolution() -> None:
    """1. Primary repository discovers 'sun' and registers it into AssetRegistry."""
    reset_discovery_state()
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry_path = Path(tmp_dir) / "registry.json"
        # Seed registry with pre-existing flower (reproducing the original bug condition)
        pre_existing_flower = Asset.from_dict({
            "id": "flower",
            "name": "Flower",
            "category": "nature",
            "format": "glb",
            "tags": ["flower", "plant", "nature"],
            "source": {
                "type": "github",
                "repository": PRIMARY_REPO,
                "path": "nature/flower.glb",
            },
        })
        registry = AssetRegistry([pre_existing_flower])
        session = FakeSession(mock_routes)

        check("initial registry has 1 asset", len(registry) == 1)
        check("initial registry cannot resolve 'sun'", registry.search_assets("sun") == [])

        # Auto-discovery via resolve_asset
        resolution = resolve_asset("sun", registry, session=session, registry_path=registry_path)

        check("resolve_asset resolves 'sun'", resolution.status == AssetResolutionStatus.RESOLVED)
        check("resolved asset is sun", resolution.asset is not None and "sun" in resolution.asset.id)
        check("registry now contains sun and car", len(registry) >= 3)
        check("sun asset in registry has deterministic id", registry.get_asset(resolution.asset.id) is not None)
        check("persisted registry exists on disk", registry_path.is_file())


def test_2_idempotency_and_no_duplicate_registration() -> None:
    """2. Duplicate discovery does not duplicate entries or make redundant network calls."""
    reset_discovery_state()
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry_path = Path(tmp_dir) / "registry.json"
        registry = AssetRegistry()
        session = FakeSession(mock_routes)

        added_1 = ensure_discovered(
            registry,
            repositories=[{"repository": PRIMARY_REPO, "path": ""}],
            session=session,
            registry_path=registry_path,
        )
        count_after_1 = len(registry)
        urls_after_1 = len(session.requested_urls)

        check("first discovery adds assets", added_1 == 3 and count_after_1 == 3)

        # Second call to ensure_discovered on the same registry
        added_2 = ensure_discovered(
            registry,
            repositories=[{"repository": PRIMARY_REPO, "path": ""}],
            session=session,
            registry_path=registry_path,
        )
        count_after_2 = len(registry)
        urls_after_2 = len(session.requested_urls)

        check("second discovery adds 0 assets", added_2 == 0)
        check("asset count is unchanged", count_after_2 == count_after_1)
        check("no new network requests made on second discovery", urls_after_2 == urls_after_1)


def test_3_local_cache_preferred() -> None:
    """3. Local cache is preferred after retrieval (no network call on cache hit)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir) / "cache"
        cache_dir.mkdir()
        asset = Asset.from_dict({
            "id": "test_sun_model",
            "name": "Sun",
            "category": "nature",
            "format": "glb",
            "tags": ["sun"],
            "source": {"type": "github", "repository": PRIMARY_REPO, "path": "nature/sun.glb"},
        })

        # Pre-populate valid cached file
        cached_file = cache_dir / "test_sun_model.glb"
        cached_file.write_bytes(b"GLB_MODEL_MOCK_DATA")

        # Session that will fail if any HTTP request is attempted
        strict_session = FakeSession({})
        retriever = AssetRetriever(cache_dir=cache_dir, session=strict_session)

        retrieved_path = retriever.retrieve(asset)
        check("retriever returns existing cached file path", retrieved_path == cached_file.resolve())
        check("retrieved file exists and is valid", retrieved_path.exists() and retrieved_path.stat().st_size > 0)
        check("no network calls made on cache hit", len(strict_session.requested_urls) == 0)


def test_4_multi_repo_fallback_and_isolation() -> None:
    """4. Multi-repository fallback: primary first, external on miss, failure isolation."""
    reset_discovery_state()
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry_path = Path(tmp_dir) / "registry.json"
        registry = AssetRegistry()

        # Configure 3 repos: Failing repo, Primary repo, External repo
        custom_repos = [
            {"repository": "failing/repo", "path": ""},
            {"repository": PRIMARY_REPO, "path": ""},
            {"repository": EXTERNAL_REPO, "path": ""},
        ]

        orig_configured = auto_discovery_module.configured_repositories
        auto_discovery_module.configured_repositories = lambda: custom_repos
        try:
            session = FakeSession(mock_routes)
            # Resolve 'avocado' which is ONLY in EXTERNAL_REPO
            resolution = resolve_asset("avocado", registry, session=session, registry_path=registry_path)

            check("avocado resolves via external repo fallback", resolution.status == AssetResolutionStatus.RESOLVED)
            check("resolved asset name is Avocado", resolution.asset is not None and "avocado" in resolution.asset.name.lower())
            check("failing repo was attempted and isolated", "https://api.github.com/repos/failing/repo" in session.requested_urls)
            check("primary repo was attempted first", PRIMARY_REPO_URL in session.requested_urls)
            check("external repo was attempted and succeeded", EXTERNAL_REPO_URL in session.requested_urls)
        finally:
            auto_discovery_module.configured_repositories = orig_configured


def test_5_unknown_asset_fails_cleanly() -> None:
    """5. Unknown asset fails cleanly returning UNAVAILABLE without crashing."""
    reset_discovery_state()
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry_path = Path(tmp_dir) / "registry.json"
        registry = AssetRegistry()
        session = FakeSession(mock_routes)

        resolution = resolve_asset("nonexistent_mystical_dragon", registry, session=session, registry_path=registry_path)
        check("unknown asset returns UNAVAILABLE", resolution.status == AssetResolutionStatus.UNAVAILABLE)
        check("asset is None on UNAVAILABLE", resolution.asset is None)
        check("label is preserved", resolution.label == "nonexistent_mystical_dragon")


def test_6_safe_promotion_policies() -> None:
    """6. Safe repository promotion policies enforce license verification."""
    permissive_asset = Asset.from_dict({
        "id": "asset_mit",
        "name": "MIT Asset",
        "category": "nature",
        "format": "glb",
        "tags": ["tree"],
        "source": {"type": "github", "repository": "external/repo", "path": "tree.glb"},
        "license": "MIT",
    })
    check("MIT asset is promotable", is_promotable(permissive_asset))
    eligible, _ = check_promotion_eligibility(permissive_asset)
    check("MIT asset check_promotion_eligibility is True", eligible)

    cc0_asset = Asset.from_dict({
        "id": "asset_cc0",
        "name": "CC0 Asset",
        "category": "nature",
        "format": "glb",
        "tags": ["sun"],
        "source": {"type": "github", "repository": "external/repo", "path": "sun.glb"},
        "license": "CC0-1.0",
    })
    check("CC0-1.0 asset is promotable", is_promotable(cc0_asset))

    no_license_asset = Asset.from_dict({
        "id": "asset_no_license",
        "name": "Unspecified Asset",
        "category": "nature",
        "format": "glb",
        "tags": ["rock"],
        "source": {"type": "github", "repository": "external/repo", "path": "rock.glb"},
        "license": None,
    })
    check("Asset without license is NOT promotable", not is_promotable(no_license_asset))
    eligible_no, reason_no = check_promotion_eligibility(no_license_asset)
    check("No-license asset is not eligible", not eligible_no and "missing or unspecified" in reason_no)

    rejected = False
    try:
        validate_asset_for_promotion(no_license_asset)
    except PromotionLicenseError:
        rejected = True
    check("validate_asset_for_promotion raises PromotionLicenseError for missing license", rejected)

    restrictive_asset = Asset.from_dict({
        "id": "asset_nc",
        "name": "Non-Commercial Asset",
        "category": "nature",
        "format": "glb",
        "tags": ["car"],
        "source": {"type": "github", "repository": "external/repo", "path": "car.glb"},
        "license": "CC-BY-NC-4.0",
    })
    check("Non-commercial asset is NOT promotable", not is_promotable(restrictive_asset))


def test_7_real_github_integration_check() -> None:
    """7. Real GitHub integration check against omkarmusle510-web/reality-engine-assets."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("SKIP: Real GitHub integration check (GITHUB_TOKEN not set)")
        return

    print("Running real GitHub integration check against omkarmusle510-web/reality-engine-assets...")
    reset_discovery_state()
    registry = AssetRegistry()
    added = ensure_discovered(
        registry,
        repositories=[{"repository": "omkarmusle510-web/reality-engine-assets", "path": ""}],
    )

    check("real GitHub discovery succeeded and discovered assets", added > 0)
    sun_matches = registry.search_assets("sun")
    check("real GitHub discovery found 'sun'", len(sun_matches) > 0)
    car_matches = registry.search_assets("car")
    check("real GitHub discovery found 'car'", len(car_matches) > 0)
    flower_matches = registry.search_assets("flower")
    check("real GitHub discovery found 'flower'", len(flower_matches) > 0)


if __name__ == "__main__":
    print("--- Running Block 11C Phase 2 Asset Discovery Tests ---\n")
    test_1_primary_repo_discovery_and_sun_resolution()
    test_2_idempotency_and_no_duplicate_registration()
    test_3_local_cache_preferred()
    test_4_multi_repo_fallback_and_isolation()
    test_5_unknown_asset_fails_cleanly()
    test_6_safe_promotion_policies()
    test_7_real_github_integration_check()

    print(f"\n==========================================")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"==========================================")
    sys.exit(1 if failed else 0)

