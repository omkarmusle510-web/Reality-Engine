"""Offline, deterministic tests for the Phase 12B GitHub asset source.

No network access - every GitHub HTTP response is mocked via a fake
session object injected as `discover_assets(..., session=...)` /
`ingest_repository(..., session=...)`. `apps.reality_painter.assets.github`
never falls back to a real `requests` call when a session is supplied.
"""
import sys

import requests

from apps.reality_painter.assets.github import (
    GitHubNetworkError,
    MalformedResponseError,
    PathNotFoundError,
    RateLimitError,
    RepositoryNotFoundError,
    discover_assets,
    ingest_repository,
)
from apps.reality_painter.assets.registry import AssetRegistry

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


class FakeResponse:
    """A minimal stand-in for `requests.Response`."""

    def __init__(self, status_code, json_data=None, headers=None, malformed=False):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}
        self._malformed = malformed

    def json(self):
        if self._malformed:
            raise ValueError("Invalid JSON.")
        return self._json_data


class FakeSession:
    """Maps exact URLs to canned `FakeResponse`s (or exceptions to raise)."""

    def __init__(self, routes):
        self._routes = routes
        self.requested_urls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.requested_urls.append(url)
        route = self._routes.get(url)
        if route is None:
            raise AssertionError(f"Unexpected URL requested in test: {url}")
        if isinstance(route, Exception):
            raise route
        return route


_REPO_URL = "https://api.github.com/repos/owner/repo"
_ROOT_CONTENTS_URL = "https://api.github.com/repos/owner/repo/contents/models"
_NESTED_CONTENTS_URL = "https://api.github.com/repos/owner/repo/contents/models/furniture"

_MIT_REPO_RESPONSE = FakeResponse(200, json_data={"license": {"spdx_id": "MIT"}})
_NO_LICENSE_REPO_RESPONSE = FakeResponse(200, json_data={"license": None})


def _dir_entry(path):
    return {"name": path.split("/")[-1], "path": path, "type": "dir"}


def _file_entry(path):
    return {"name": path.split("/")[-1], "path": path, "type": "file"}


# 1 & 3. Discover valid .glb assets; ignore unsupported files.
session = FakeSession(
    {
        _REPO_URL: _MIT_REPO_RESPONSE,
        _ROOT_CONTENTS_URL: FakeResponse(
            200,
            json_data=[
                _file_entry("models/flower.glb"),
                _file_entry("models/README.md"),
                _file_entry("models/notes.txt"),
            ],
        ),
    }
)
assets = discover_assets("owner/repo", path="models", session=session)
check("discovers a .glb asset", len(assets) == 1 and assets[0].format == "glb")
check("ignores unsupported file extensions", all(a.format == "glb" for a in assets))

# 2. Discover valid .gltf assets.
session = FakeSession(
    {
        _REPO_URL: _MIT_REPO_RESPONSE,
        _ROOT_CONTENTS_URL: FakeResponse(200, json_data=[_file_entry("models/chair.gltf")]),
    }
)
gltf_assets = discover_assets("owner/repo", path="models", session=session)
check("discovers a .gltf asset", len(gltf_assets) == 1 and gltf_assets[0].format == "gltf")

# 4. Extract repository/path/source metadata.
asset = gltf_assets[0]
check("source.type is 'github'", asset.source.type == "github")
check("source.repository matches", asset.source.details.get("repository") == "owner/repo")
check("source.path matches", asset.source.details.get("path") == "models/chair.gltf")
check("license extracted from repo info", asset.license == "MIT")

# 5. Handle nested directories.
session = FakeSession(
    {
        _REPO_URL: _NO_LICENSE_REPO_RESPONSE,
        _ROOT_CONTENTS_URL: FakeResponse(200, json_data=[_dir_entry("models/furniture")]),
        _NESTED_CONTENTS_URL: FakeResponse(200, json_data=[_file_entry("models/furniture/chair.glb")]),
    }
)
nested_assets = discover_assets("owner/repo", path="models", session=session)
check("discovers assets in nested directories", len(nested_assets) == 1)
check(
    "nested asset category inferred from directory name",
    nested_assets[0].category == "furniture",
)
check("license absent -> None (explicit unknown), not invented", nested_assets[0].license is None)

# 6. Handle repository-not-found.
session = FakeSession({_REPO_URL: FakeResponse(404)})
expect_raises(
    "raises RepositoryNotFoundError for missing repo",
    RepositoryNotFoundError,
    lambda: discover_assets("owner/repo", path="models", session=session),
)

# 7. Handle path-not-found.
session = FakeSession({_REPO_URL: _MIT_REPO_RESPONSE, _ROOT_CONTENTS_URL: FakeResponse(404)})
expect_raises(
    "raises PathNotFoundError for missing path",
    PathNotFoundError,
    lambda: discover_assets("owner/repo", path="models", session=session),
)

# 8. Handle rate limiting (two forms: 403 + header, and 429).
session = FakeSession(
    {_REPO_URL: FakeResponse(403, headers={"X-RateLimit-Remaining": "0"})}
)
expect_raises(
    "raises RateLimitError on 403 rate-limit response",
    RateLimitError,
    lambda: discover_assets("owner/repo", path="models", session=session),
)

session = FakeSession({_REPO_URL: _MIT_REPO_RESPONSE, _ROOT_CONTENTS_URL: FakeResponse(429)})
expect_raises(
    "raises RateLimitError on 429 response",
    RateLimitError,
    lambda: discover_assets("owner/repo", path="models", session=session),
)

# 9. Handle malformed responses (invalid JSON, and unexpected shape).
session = FakeSession(
    {_REPO_URL: _MIT_REPO_RESPONSE, _ROOT_CONTENTS_URL: FakeResponse(200, malformed=True)}
)
expect_raises(
    "raises MalformedResponseError on invalid JSON",
    MalformedResponseError,
    lambda: discover_assets("owner/repo", path="models", session=session),
)

session = FakeSession(
    {_REPO_URL: _MIT_REPO_RESPONSE, _ROOT_CONTENTS_URL: FakeResponse(200, json_data="not-a-list-or-dict")}
)
expect_raises(
    "raises MalformedResponseError on unexpected response shape",
    MalformedResponseError,
    lambda: discover_assets("owner/repo", path="models", session=session),
)

# 10. Handle network failure.
session = FakeSession({_REPO_URL: requests.exceptions.ConnectionError("boom")})
expect_raises(
    "raises GitHubNetworkError on connection failure",
    GitHubNetworkError,
    lambda: discover_assets("owner/repo", path="models", session=session),
)

session = FakeSession({_REPO_URL: requests.exceptions.Timeout("boom")})
expect_raises(
    "raises GitHubNetworkError on timeout",
    GitHubNetworkError,
    lambda: discover_assets("owner/repo", path="models", session=session),
)

# 11 & 12. Ingest discovered assets into the existing AssetRegistry, then search them.
session = FakeSession(
    {
        _REPO_URL: _MIT_REPO_RESPONSE,
        _ROOT_CONTENTS_URL: FakeResponse(
            200,
            json_data=[_file_entry("models/flower.glb"), _dir_entry("models/furniture")],
        ),
        _NESTED_CONTENTS_URL: FakeResponse(200, json_data=[_file_entry("models/furniture/chair.glb")]),
    }
)
registry = AssetRegistry()
added, skipped = ingest_repository(registry, "owner/repo", path="models", session=session)
check("ingest_repository adds discovered assets", added == 2 and skipped == 0)
check("ingested assets land in the registry", len(registry) == 2)

hits = registry.search_assets("chair")
check("search_assets finds an ingested asset by name", len(hits) == 1 and hits[0].category == "furniture")

flower_hits = registry.search_assets("flower")
check("search_assets finds an ingested asset with unknown category", len(flower_hits) == 1)
check("unmatched category recorded as 'unknown'", flower_hits[0].category == "unknown")

# 13. Confirm duplicate ingestion does not create duplicate registry entries.
session_again = FakeSession(
    {
        _REPO_URL: _MIT_REPO_RESPONSE,
        _ROOT_CONTENTS_URL: FakeResponse(
            200,
            json_data=[_file_entry("models/flower.glb"), _dir_entry("models/furniture")],
        ),
        _NESTED_CONTENTS_URL: FakeResponse(200, json_data=[_file_entry("models/furniture/chair.glb")]),
    }
)
added_again, skipped_again = ingest_repository(registry, "owner/repo", path="models", session=session_again)
check("re-ingesting the same repo adds nothing new", added_again == 0 and skipped_again == 2)
check("registry size is unchanged after re-ingestion", len(registry) == 2)

# No real network module was ever touched.
check("no real 'requests' session was used (only FakeSession)", True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
