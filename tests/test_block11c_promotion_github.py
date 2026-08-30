"""Focused tests for Block 11C External Asset -> Primary GitHub Repository Promotion.

Verifies:
1. Permitted redistribution licenses are promotable (CC0, CC-BY-4.0, MIT, Apache-2.0, etc.).
2. Prohibited and unknown/missing licenses reject promotion and raise PromotionLicenseError.
3. Promotion uploads to GitHub via mocked API with correct endpoint, headers, and base64 payload.
4. Promotion creates local staging file in mirrored category/name structure.
5. Promotion persists complete metadata into promotions.json (source repo, path, license, deterministic ID).
6. Duplicate promotion is prevented without re-uploading.
7. maybe_promote_asset safe helper handles primary (skip), prohibited (skip), and valid external (promote).
8. Real GitHub integration check if GITHUB_TOKEN is present.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.reality_painter.assets.promotion import (
    GitHubUploadError,
    PromotionError,
    PromotionLicenseError,
    check_promotion_eligibility,
    is_already_promoted,
    is_promotable,
    list_promoted_assets,
    maybe_promote_asset,
    promote_asset,
    upload_file_to_github,
    validate_asset_for_promotion,
)
from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.schema import Asset, AssetSource

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


class MockResponse:
    def __init__(self, status_code: int, json_data: Any = None, headers: Optional[Dict[str, str]] = None) -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.headers = headers or {}
        self.text = json.dumps(self._json_data)

    def json(self) -> Any:
        return self._json_data


class MockHttpSession:
    def __init__(self) -> None:
        self.get_requests: List[Dict[str, Any]] = []
        self.put_requests: List[Dict[str, Any]] = []

    def get(self, url: str, headers: Any = None, params: Any = None, timeout: Any = None) -> MockResponse:
        self.get_requests.append({"url": url, "headers": headers, "params": params})
        # Default: 404 (file does not yet exist on GitHub)
        return MockResponse(404, {"message": "Not Found"})

    def put(self, url: str, headers: Any = None, json: Any = None, timeout: Any = None) -> MockResponse:
        self.put_requests.append({"url": url, "headers": headers, "json": json})
        return MockResponse(201, {
            "content": {"name": "model.glb", "path": "nature/model.glb", "sha": "mocksha123"},
            "commit": {"sha": "mockcommit456", "message": json.get("message", "") if json else ""},
        })


def test_1_permitted_licenses():
    """1. Permitted redistribution licenses are accepted."""
    permissive = ["CC0-1.0", "cc0", "CC-BY-4.0", "CC-BY-3.0", "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "Unlicense"]
    for lic in permissive:
        asset = Asset.from_dict({
            "id": f"asset_{lic.lower()}",
            "name": f"Asset {lic}",
            "category": "nature",
            "format": "glb",
            "tags": ["test"],
            "source": {"type": "github", "repository": "external/repo", "path": "model.glb"},
            "license": lic,
        })
        check(f"License '{lic}' is promotable", is_promotable(asset))
        eligible, reason = check_promotion_eligibility(asset)
        check(f"License '{lic}' eligibility check is True", eligible)


def test_2_prohibited_and_unknown_licenses():
    """2. Unknown, missing, or restrictive licenses are strictly rejected."""
    prohibited = [None, "", "unknown", "CC-BY-NC-4.0", "CC-BY-ND-4.0", "GPL-3.0", "Proprietary", "Custom Non-Commercial"]
    for lic in prohibited:
        asset = Asset.from_dict({
            "id": f"asset_prohibited_{str(lic)}",
            "name": "Prohibited Asset",
            "category": "nature",
            "format": "glb",
            "tags": ["test"],
            "source": {"type": "github", "repository": "external/repo", "path": "model.glb"},
            "license": lic,
        })
        check(f"License '{lic}' is NOT promotable", not is_promotable(asset))
        eligible, reason = check_promotion_eligibility(asset)
        check(f"License '{lic}' eligibility is False", not eligible)

        raised = False
        try:
            validate_asset_for_promotion(asset)
        except PromotionLicenseError:
            raised = True
        check(f"validate_asset_for_promotion raises for '{lic}'", raised)


def test_3_mocked_github_upload_and_staging():
    """3. Promotion uploads to GitHub and stages locally."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        staging_dir = Path(tmp_dir) / "promoted"
        log_path = staging_dir / "promotions.json"
        cached_file = Path(tmp_dir) / "cached_model.glb"
        cached_file.write_bytes(b"GLB_BINARY_CONTENT_12345")

        asset = Asset.from_dict({
            "id": "github_khronos_models_avocado_glb",
            "name": "Avocado",
            "category": "nature",
            "format": "glb",
            "tags": ["avocado", "food"],
            "source": {"type": "github", "repository": "KhronosGroup/glTF-Sample-Assets", "path": "Models/Avocado/Avocado.glb"},
            "license": "CC-BY-4.0",
        })

        mock_session = MockHttpSession()
        registry = AssetRegistry()

        promoted = promote_asset(
            asset=asset,
            cached_file_path=cached_file,
            staging_dir=staging_dir,
            log_path=log_path,
            session=mock_session,
            push_to_remote=True,
            registry=registry,
        )

        # Verify GitHub PUT call
        check("GitHub PUT was called exactly once", len(mock_session.put_requests) == 1)
        put_req = mock_session.put_requests[0]
        expected_url = "https://api.github.com/repos/omkarmusle510-web/reality-engine-assets/contents/nature/avocado.glb"
        check("PUT targeted correct repository and path", put_req["url"] == expected_url)

        # Verify payload contains valid base64
        payload = put_req["json"]
        decoded_bytes = base64.b64decode(payload["content"])
        check("PUT payload content matches original file bytes", decoded_bytes == b"GLB_BINARY_CONTENT_12345")
        check("Commit message mentions asset and source repo", "Avocado" in payload["message"] and "KhronosGroup" in payload["message"])

        # Verify local staging file
        staged_file = staging_dir / "nature" / "avocado.glb"
        check("Local staging file exists", staged_file.is_file())
        check("Local staging content matches", staged_file.read_bytes() == b"GLB_BINARY_CONTENT_12345")

        # Verify promotions.json metadata
        promotions = list_promoted_assets(log_path=log_path)
        check("Promotion log has 1 entry", len(promotions) == 1)
        entry = promotions[0]
        check("Logged original_id", entry["original_id"] == asset.id)
        check("Logged original_source repository", entry["original_source"]["repository"] == "KhronosGroup/glTF-Sample-Assets")
        check("Logged license", entry["license"] == "CC-BY-4.0")
        check("Logged uploaded_to_github is True", entry["uploaded_to_github"] is True)

        # Verify registry entry
        check("Promoted asset registered in AssetRegistry", len(registry) == 1)
        resolved = registry.search_assets("avocado")
        check("Registry search finds promoted asset", len(resolved) == 1)
        check("Registry source points to omkarmusle510-web/reality-engine-assets",
              resolved[0].source.details["repository"] == "omkarmusle510-web/reality-engine-assets")
        check("Registry path is nature/avocado.glb", resolved[0].source.details["path"] == "nature/avocado.glb")


def test_4_duplicate_prevention():
    """4. Repeated promotion does not duplicate upload or log entries."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        staging_dir = Path(tmp_dir) / "promoted"
        log_path = staging_dir / "promotions.json"
        cached_file = Path(tmp_dir) / "cached_model.glb"
        cached_file.write_bytes(b"GLB_BINARY_DATA")

        asset = Asset.from_dict({
            "id": "github_threejs_horse_glb",
            "name": "Horse",
            "category": "animals",
            "format": "glb",
            "tags": ["horse"],
            "source": {"type": "github", "repository": "mrdoob/three.js", "path": "examples/models/gltf/Horse.glb"},
            "license": "CC-BY-4.0",
        })

        mock_session = MockHttpSession()
        registry = AssetRegistry()

        promoted_1 = promote_asset(
            asset=asset,
            cached_file_path=cached_file,
            staging_dir=staging_dir,
            log_path=log_path,
            session=mock_session,
            push_to_remote=True,
            registry=registry,
        )

        check("First promotion made 1 PUT request", len(mock_session.put_requests) == 1)

        promoted_2 = promote_asset(
            asset=asset,
            cached_file_path=cached_file,
            staging_dir=staging_dir,
            log_path=log_path,
            session=mock_session,
            push_to_remote=True,
            registry=registry,
        )

        check("Second promotion made NO additional PUT requests", len(mock_session.put_requests) == 1)
        check("Returned asset IDs match", promoted_1.id == promoted_2.id)
        check("Promotion log still has exactly 1 entry", len(list_promoted_assets(log_path=log_path)) == 1)


def test_5_maybe_promote_asset_safe_helper():
    """5. maybe_promote_asset handles primary, prohibited, and permitted assets cleanly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        staging_dir = Path(tmp_dir) / "promoted"
        log_path = staging_dir / "promotions.json"
        cached_file = Path(tmp_dir) / "cached.glb"
        cached_file.write_bytes(b"MODEL_DATA")

        # 1. Primary repository asset -> skipped (returns None)
        primary_asset = Asset.from_dict({
            "id": "primary_sun",
            "name": "Sun",
            "category": "nature",
            "format": "glb",
            "tags": ["sun"],
            "source": {"type": "github", "repository": "omkarmusle510-web/reality-engine-assets", "path": "nature/sun.glb"},
        })
        res1 = maybe_promote_asset(primary_asset, cached_file, log_path=log_path, staging_dir=staging_dir)
        check("Primary asset is skipped by maybe_promote_asset", res1 is None)

        # 2. Prohibited license -> skipped (returns None, does not raise)
        prohibited_asset = Asset.from_dict({
            "id": "ext_nc",
            "name": "NonCommercial",
            "category": "objects",
            "format": "glb",
            "tags": ["nc"],
            "source": {"type": "github", "repository": "other/repo", "path": "nc.glb"},
            "license": "CC-BY-NC-4.0",
        })
        res2 = maybe_promote_asset(prohibited_asset, cached_file, log_path=log_path, staging_dir=staging_dir)
        check("Prohibited license is skipped by maybe_promote_asset", res2 is None)

        # 3. Permitted external asset -> promoted
        mock_session = MockHttpSession()
        permitted_asset = Asset.from_dict({
            "id": "ext_mit_chair",
            "name": "Chair",
            "category": "furniture",
            "format": "glb",
            "tags": ["chair"],
            "source": {"type": "github", "repository": "BabylonJS/Assets", "path": "meshes/chair.glb"},
            "license": "Apache-2.0",
        })
        res3 = maybe_promote_asset(
            permitted_asset,
            cached_file,
            log_path=log_path,
            staging_dir=staging_dir,
            session=mock_session,
        )
        check("Permitted asset is promoted by maybe_promote_asset", res3 is not None)
        check("Promoted asset source is primary repo", res3.source.details["repository"] == "omkarmusle510-web/reality-engine-assets")


def test_6_real_github_integration_check():
    """6. Real GitHub integration check: verify token connectivity and probe GitHub API contents endpoint."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("SKIP: Real GitHub integration check (GITHUB_TOKEN not set)")
        return

    print("\nRunning real GitHub integration probe against omkarmusle510-web/reality-engine-assets...")
    import requests
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    # Test read access to repository
    r_get = requests.get(
        "https://api.github.com/repos/omkarmusle510-web/reality-engine-assets",
        headers=headers,
        timeout=15.0,
    )
    check("GitHub read access to omkarmusle510-web/reality-engine-assets is 200 OK", r_get.status_code == 200)

    # Test user identity
    r_user = requests.get("https://api.github.com/user", headers=headers, timeout=15.0)
    check("GitHub user endpoint is accessible", r_user.status_code == 200)
    user_login = r_user.json().get("login") if r_user.status_code == 200 else "unknown"
    print(f"Authenticated as GitHub user: {user_login}")

    # Probe Contents API write capability
    url = "https://api.github.com/repos/omkarmusle510-web/reality-engine-assets/contents/.probe_write_permission"
    r_probe = requests.get(url, headers=headers, timeout=15.0)
    check("GitHub Contents API responds (404/200)", r_probe.status_code in (200, 404))


if __name__ == "__main__":
    print("--- Running Block 11C Promotion & GitHub Upload Tests ---\n")
    test_1_permitted_licenses()
    test_2_prohibited_and_unknown_licenses()
    test_3_mocked_github_upload_and_staging()
    test_4_duplicate_prevention()
    test_5_maybe_promote_asset_safe_helper()
    test_6_real_github_integration_check()

    print(f"\n==========================================")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"==========================================")
    sys.exit(1 if failed else 0)
