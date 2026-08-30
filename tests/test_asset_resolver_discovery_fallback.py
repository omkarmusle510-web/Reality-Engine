"""Offline, deterministic test for Block 11C Phase 2 (resolver discovery fallback).

No real network access - every GitHub HTTP response is mocked via a
fake session object, matching the existing convention in
tests/test_github_asset_source.py and tests/test_asset_auto_discovery.py.
"""
import sys
import tempfile
from pathlib import Path

from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.schema import Asset
from apps.reality_painter.inspection.asset_resolver import AssetResolutionStatus, resolve_asset

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
    """Maps exact URLs to canned responses; unmapped URLs raise AssertionError."""

    def __init__(self, routes):
        self._routes = routes
        self.requested_urls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.requested_urls.append(url)
        route = self._routes.get(url)
        if route is None:
            raise AssertionError(f"Unexpected/unreachable URL requested in test: {url}")
        return route


GOOD_REPO = "owner/good-repo"
GOOD_REPO_URL = f"https://api.github.com/repos/{GOOD_REPO}"
GOOD_TREE_URL = f"https://api.github.com/repos/{GOOD_REPO}/git/trees/main"
BAD_REPO = "owner/missing-repo"
BAD_REPO_URL = f"https://api.github.com/repos/{BAD_REPO}"

good_routes = {
    GOOD_REPO_URL: FakeResponse(200, {"default_branch": "main", "license": None}),
    GOOD_TREE_URL: FakeResponse(
        200,
        {"tree": [{"path": "nature/sun.glb", "type": "blob"}, {"path": "README.md", "type": "blob"}]},
    ),
    BAD_REPO_URL: FakeResponse(404, {}),
}

repos = [{"repository": BAD_REPO, "path": ""}, {"repository": GOOD_REPO, "path": ""}]

scratch_dir = tempfile.TemporaryDirectory()
registry_path = Path(scratch_dir.name) / "registry.json"

# --- 1. registry miss triggers discovery across multiple configured repos --
# Since resolve_asset has no `repositories` param (matches auto_discovery's
# own contract - configured via env var / default), point discovery at our
# fake repos for this test by monkeypatching configured_repositories().
import apps.reality_painter.assets.auto_discovery as auto_discovery_module

_original_configured = auto_discovery_module.configured_repositories
_original_primary = auto_discovery_module.primary_repository
_original_external = auto_discovery_module.external_repositories
auto_discovery_module.configured_repositories = lambda: repos
auto_discovery_module.primary_repository = lambda: repos[0]
auto_discovery_module.external_repositories = lambda: repos[1:]

try:
    registry = AssetRegistry()
    session = FakeSession(good_routes)
    resolution = resolve_asset("sun", registry, session=session, registry_path=registry_path)

    check("resolver discovers and resolves 'sun' on first miss", resolution.status == AssetResolutionStatus.RESOLVED)
    check(
        "bad repo was attempted (isolated failure) alongside good repo",
        BAD_REPO_URL in session.requested_urls and GOOD_REPO_URL in session.requested_urls,
    )
    check("README.md (non-3D file) was not registered", len(registry) == 1)

    requests_before = len(session.requested_urls)
    resolution_again = resolve_asset("sun", registry, session=session, registry_path=registry_path)
    check("second resolve for an already-registered label still resolves", resolution_again.status == AssetResolutionStatus.RESOLVED)
    check(
        "already-scanned repos are not re-scanned (no duplicate network calls / no duplicate registration)",
        len(session.requested_urls) == requests_before and len(registry) == 1,
    )

    check("discovered asset persisted to registry_path", registry_path.is_file())
    reloaded = AssetRegistry.load(registry_path)
    check("persisted registry reloads with the discovered asset", len(reloaded) == 1)

    # --- cache-first: an already-registered asset never triggers discovery ---
    cached_registry = AssetRegistry(
        [
            Asset.from_dict(
                {
                    "id": "flower_001",
                    "name": "Flower",
                    "category": "nature",
                    "format": "glb",
                    "tags": ["flower"],
                    "source": {"type": "github", "repository": "owner/other", "path": "flower.glb"},
                }
            )
        ]
    )
    no_network_session = FakeSession({})  # any .get() call raises AssertionError
    cached_resolution = resolve_asset("flower", cached_registry, session=no_network_session, registry_path=registry_path)
    check(
        "an already-registered asset resolves without touching discovery/network at all",
        cached_resolution.status == AssetResolutionStatus.RESOLVED,
    )

    # --- unresolvable label after discovery still cleanly reports UNAVAILABLE ---
    unmatched = resolve_asset("unicorn", registry, session=session, registry_path=registry_path)
    check("unmatched label after discovery is UNAVAILABLE, not raised", unmatched.status == AssetResolutionStatus.UNAVAILABLE)
finally:
    auto_discovery_module.configured_repositories = _original_configured
    auto_discovery_module.primary_repository = _original_primary
    auto_discovery_module.external_repositories = _original_external
    scratch_dir.cleanup()

print(f"\n{passed} passed, {failed} failed")
if failed:
    sys.exit(1)
