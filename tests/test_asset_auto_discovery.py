"""Offline, deterministic tests for Block 11C Phase 1 (automatic asset discovery).

No real network access - every GitHub HTTP response is mocked via a
fake session object, exactly like `tests/test_github_asset_source.py`
already does for `apps.reality_painter.assets.github`.
"""
import json
import sys
import tempfile
from pathlib import Path

from apps.reality_painter.assets.auto_discovery import (
    _already_scanned,
    configured_repositories,
    ensure_discovered,
)
from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.schema import Asset, AssetSource
from apps.reality_painter.inspection.asset_resolver import (
    AssetResolutionStatus,
    resolve_asset,
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


class FakeResponse:
    def __init__(self, status_code, json_data=None, headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}

    def json(self):
        return self._json_data


class FakeSession:
    """Maps exact URLs to canned `FakeResponse`s; records every call made."""

    def __init__(self, routes):
        self._routes = routes
        self.requested_urls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.requested_urls.append(url)
        route = self._routes.get(url)
        if route is None:
            raise AssertionError(f"Unexpected URL requested in test: {url}")
        return route


_REPO = "owner/assets-repo"
_REPO_URL = f"https://api.github.com/repos/{_REPO}"
_TREE_URL = f"https://api.github.com/repos/{_REPO}/git/trees/main"

_REPO_INFO = FakeResponse(200, json_data={"license": None, "default_branch": "main"})
_TREE = FakeResponse(
    200,
    json_data={
        "tree": [
            {"path": "nature/flower.glb", "type": "blob"},
            {"path": "vehicles/car.glb", "type": "blob"},
            {"path": "README.md", "type": "blob"},
        ]
    },
)


def _make_session():
    return FakeSession({_REPO_URL: _REPO_INFO, _TREE_URL: _TREE})


# Every ensure_discovered() call below must pass an explicit
# registry_path so this test never writes to the real, bundled
# apps/reality_painter/assets/registry.json - a fresh, per-test-run
# temp path is used throughout, mirroring how AssetRegistry.load()'s
# own tests never touch the bundled file either.
_scratch_dir = tempfile.TemporaryDirectory()
_scratch_registry_path = Path(_scratch_dir.name) / "registry.json"

# --- 1. repository scan discovers supported assets; deterministic IDs -----

registry = AssetRegistry()
added = ensure_discovered(
    registry,
    repositories=[{"repository": _REPO, "path": ""}],
    session=_make_session(),
    registry_path=_scratch_registry_path,
)
check("scan discovers both .glb assets", added == 2)
check("scan skips non-3D files (README.md)", len(registry) == 2)

registry_again = AssetRegistry()
added_again = ensure_discovered(
    registry_again,
    repositories=[{"repository": _REPO, "path": ""}],
    session=_make_session(),
    registry_path=_scratch_registry_path,
)
check(
    "same repository scan twice produces identical (deterministic) asset ids",
    sorted(a.id for a in registry) == sorted(a.id for a in registry_again),
)

# --- 2. registry already representing a repository skips re-scanning ------

skip_session = FakeSession({})  # any .get() call raises AssertionError
added_second_call = ensure_discovered(
    registry,
    repositories=[{"repository": _REPO, "path": ""}],
    session=skip_session,
    registry_path=_scratch_registry_path,
)
check("already-scanned repository is not re-scanned (no network call)", added_second_call == 0)
check("_already_scanned reports True for a scanned repository", _already_scanned(registry, _REPO))
check("_already_scanned reports False for an unrelated repository", not _already_scanned(registry, "owner/other"))

# --- 3. registry persistence (save/load round-trip) ------------------------

with tempfile.TemporaryDirectory() as tmp_dir:
    registry_path = Path(tmp_dir) / "registry.json"
    fresh_registry = AssetRegistry()
    ensure_discovered(
        fresh_registry,
        repositories=[{"repository": _REPO, "path": ""}],
        session=_make_session(),
        registry_path=registry_path,
    )
    check("registry is persisted to disk after new assets are discovered", registry_path.is_file())

    reloaded = AssetRegistry.load(registry_path)
    check("persisted registry reloads with the same asset count", len(reloaded) == len(fresh_registry))

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    check("persisted file uses the existing flat registry.json schema", "assets" in payload)

# --- 4. a repository that fails never crashes discovery of the others -----

failing_session = FakeSession({_REPO_URL: FakeResponse(404, json_data={})})
mixed_registry = AssetRegistry()
mixed_added = ensure_discovered(
    mixed_registry,
    repositories=[{"repository": "owner/missing-repo", "path": ""}, {"repository": _REPO, "path": ""}],
    session=FakeSession(
        {
            f"https://api.github.com/repos/owner/missing-repo": FakeResponse(404, json_data={}),
            _REPO_URL: _REPO_INFO,
            _TREE_URL: _TREE,
        }
    ),
    registry_path=_scratch_registry_path,
)
check("a failing repository is skipped, not raised", mixed_added == 2)
check("a later, working repository still gets scanned", len(mixed_registry) == 2)

_scratch_dir.cleanup()

# --- 5. configuration is not a single hard-coded repository ----------------

import os

os.environ["REALITY_PAINTER_ASSET_REPOSITORIES"] = json.dumps([{"repository": "a/b", "path": "models"}])
try:
    configured = configured_repositories()
    check("configured_repositories() reads the env var override", configured == [{"repository": "a/b", "path": "models"}])
finally:
    del os.environ["REALITY_PAINTER_ASSET_REPOSITORIES"]

check("configured_repositories() falls back to a default when unset", len(configured_repositories()) >= 1)

os.environ["REALITY_PAINTER_ASSET_REPOSITORIES"] = "not json"
try:
    check("malformed env var falls back to default instead of raising", len(configured_repositories()) >= 1)
finally:
    del os.environ["REALITY_PAINTER_ASSET_REPOSITORIES"]

# --- 6. normalized label lookup --------------------------------------------

flower_asset = Asset.from_dict(
    {
        "id": "flower_001",
        "name": "Flower",
        "category": "nature",
        "format": "glb",
        "tags": ["flower", "plant"],
        "source": {"type": "github", "repository": _REPO, "path": "nature/flower.glb"},
    }
)
label_registry = AssetRegistry([flower_asset])

exact = resolve_asset("flower", label_registry)
check("exact single-word label still resolves (unchanged behavior)", exact.status == AssetResolutionStatus.RESOLVED)

descriptive = resolve_asset("red flower", label_registry)
check(
    "descriptive label normalizes to its final word and resolves ('red flower' -> 'flower')",
    descriptive.status == AssetResolutionStatus.RESOLVED and descriptive.asset.id == "flower_001",
)
check("resolution carries the original, un-normalized label through", descriptive.label == "red flower")

unmatched = resolve_asset("a red flying dinosaur", label_registry)
check("a label with no matching normalized candidate resolves UNAVAILABLE", unmatched.status == AssetResolutionStatus.UNAVAILABLE)


print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
