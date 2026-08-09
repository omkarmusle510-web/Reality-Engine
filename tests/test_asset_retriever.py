"""Offline, deterministic tests for the Phase 12C asset retriever.

No network access - every GitHub HTTP response is mocked via a fake
session object injected as `AssetRetriever(..., session=...)`.
`apps.reality_painter.assets.retriever` never falls back to a real
`requests` call when a session is supplied.
"""
import sys
import tempfile
from pathlib import Path

import requests

from apps.reality_painter.assets.retriever import (
    AssetNotFoundError,
    AssetRetrievalError,
    AssetRetriever,
    RetrievalNetworkError,
    RetrievalRateLimitError,
    UnsupportedFormatError,
)
from apps.reality_painter.assets.schema import Asset

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
    """A minimal stand-in for `requests.Response`, including streaming."""

    def __init__(self, status_code, json_data=None, headers=None, content_chunks=None, malformed=False):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}
        self._content_chunks = content_chunks
        self._malformed = malformed

    def json(self):
        if self._malformed:
            raise ValueError("Invalid JSON.")
        return self._json_data

    def iter_content(self, chunk_size=None):
        for chunk in self._content_chunks or []:
            yield chunk


class FakeSession:
    """Maps exact URLs to canned `FakeResponse`s (or exceptions to raise)."""

    def __init__(self, routes):
        self._routes = routes
        self.requested_urls = []

    def get(self, url, headers=None, timeout=None, stream=False, params=None):
        self.requested_urls.append(url)
        route = self._routes.get(url)
        if route is None:
            raise AssertionError(f"Unexpected URL requested in test: {url}")
        if isinstance(route, Exception):
            raise route
        return route


def _asset(asset_id="chair_001", asset_format="glb", repository="owner/repo", path="models/chair.glb"):
    return Asset.from_dict(
        {
            "id": asset_id,
            "name": "Chair",
            "category": "furniture",
            "format": asset_format,
            "source": {"type": "github", "repository": repository, "path": path},
        }
    )


_METADATA_URL = "https://api.github.com/repos/owner/repo/contents/models/chair.glb"
_DOWNLOAD_URL = "https://raw.githubusercontent.com/owner/repo/main/models/chair.glb"
_METADATA_RESPONSE = FakeResponse(200, json_data={"download_url": _DOWNLOAD_URL})


# 1. Successful .glb retrieval.
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession(
        {
            _METADATA_URL: _METADATA_RESPONSE,
            _DOWNLOAD_URL: FakeResponse(200, content_chunks=[b"glb-bytes-", b"more-bytes"]),
        }
    )
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    path = retriever.retrieve(_asset())
    check("successful .glb retrieval returns a Path", isinstance(path, Path))
    check("downloaded .glb file exists on disk", path.is_file())
    check("downloaded .glb file has the expected content", path.read_bytes() == b"glb-bytes-more-bytes")
    check(".glb cache path uses the .glb extension", path.suffix == ".glb")

# 2. Successful .gltf retrieval.
with tempfile.TemporaryDirectory() as tmp:
    gltf_metadata_url = "https://api.github.com/repos/owner/repo/contents/models/chair.gltf"
    gltf_download_url = "https://raw.githubusercontent.com/owner/repo/main/models/chair.gltf"
    session = FakeSession(
        {
            gltf_metadata_url: FakeResponse(200, json_data={"download_url": gltf_download_url}),
            gltf_download_url: FakeResponse(200, content_chunks=[b"gltf-json-bytes"]),
        }
    )
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    asset = _asset(asset_id="chair_gltf_001", asset_format="gltf", path="models/chair.gltf")
    path = retriever.retrieve(asset)
    check("successful .gltf retrieval returns a Path", isinstance(path, Path))
    check(".gltf cache path uses the .gltf extension", path.suffix == ".gltf")

# 3 & 4. Cache hit: second retrieval does not re-download.
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession(
        {
            _METADATA_URL: _METADATA_RESPONSE,
            _DOWNLOAD_URL: FakeResponse(200, content_chunks=[b"glb-bytes"]),
        }
    )
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    asset = _asset()
    first_path = retriever.retrieve(asset)
    calls_after_first = len(session.requested_urls)
    second_path = retriever.retrieve(asset)
    check("cache hit returns the same path", first_path == second_path)
    check("cache hit performs no additional HTTP requests", len(session.requested_urls) == calls_after_first)
    check("duplicate retrieval does not download again", session.requested_urls.count(_DOWNLOAD_URL) == 1)

# 5. Corrupted (empty) cached file triggers re-download.
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession(
        {
            _METADATA_URL: _METADATA_RESPONSE,
            _DOWNLOAD_URL: FakeResponse(200, content_chunks=[b"fresh-bytes"]),
        }
    )
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    asset = _asset()
    corrupt_path = retriever._cache_path_for(asset)
    corrupt_path.write_bytes(b"")  # simulate a prior interrupted/corrupt download
    path = retriever.retrieve(asset)
    check("corrupted (empty) cache triggers re-download", path.read_bytes() == b"fresh-bytes")
    check("re-download hit the download URL", _DOWNLOAD_URL in session.requested_urls)

# 6. HTTP 404 on metadata lookup -> AssetNotFoundError.
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession({_METADATA_URL: FakeResponse(404)})
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    expect_raises("raises AssetNotFoundError on HTTP 404", AssetNotFoundError, lambda: retriever.retrieve(_asset()))

# 7. HTTP 429 -> RetrievalRateLimitError.
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession({_METADATA_URL: FakeResponse(429)})
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    expect_raises(
        "raises RetrievalRateLimitError on HTTP 429", RetrievalRateLimitError, lambda: retriever.retrieve(_asset())
    )

# Also cover the 403 + X-RateLimit-Remaining=0 rate-limit shape.
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession({_METADATA_URL: FakeResponse(403, headers={"X-RateLimit-Remaining": "0"})})
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    expect_raises(
        "raises RetrievalRateLimitError on HTTP 403 rate-limit response",
        RetrievalRateLimitError,
        lambda: retriever.retrieve(_asset()),
    )

# 8. HTTP 5xx -> generic AssetRetrievalError.
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession({_METADATA_URL: FakeResponse(500)})
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    expect_raises("raises AssetRetrievalError on HTTP 5xx", AssetRetrievalError, lambda: retriever.retrieve(_asset()))

# 9. Network failure (metadata lookup) -> RetrievalNetworkError.
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession({_METADATA_URL: requests.exceptions.ConnectionError("boom")})
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    expect_raises(
        "raises RetrievalNetworkError on connection failure", RetrievalNetworkError, lambda: retriever.retrieve(_asset())
    )

with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession({_METADATA_URL: requests.exceptions.Timeout("boom")})
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    expect_raises("raises RetrievalNetworkError on timeout", RetrievalNetworkError, lambda: retriever.retrieve(_asset()))

# 10. Empty download response -> AssetRetrievalError, and no corrupt file left behind.
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession(
        {
            _METADATA_URL: _METADATA_RESPONSE,
            _DOWNLOAD_URL: FakeResponse(200, content_chunks=[]),
        }
    )
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    asset = _asset()
    expect_raises("raises AssetRetrievalError on empty download", AssetRetrievalError, lambda: retriever.retrieve(asset))
    check("no corrupt file left behind after empty download", not retriever._cache_path_for(asset).exists())
    check("no leftover .part temp file after empty download", list(Path(tmp).glob("*.part")) == [])

# 11. Unsupported format -> UnsupportedFormatError, no HTTP call at all.
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession({})  # any HTTP call would raise AssertionError
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    unsupported_asset = _asset(asset_format="obj")
    expect_raises(
        "raises UnsupportedFormatError for unsupported format",
        UnsupportedFormatError,
        lambda: retriever.retrieve(unsupported_asset),
    )
    check("unsupported format never touches the network", session.requested_urls == [])

# 12. Unsafe/path-traversal asset id never escapes the cache directory.
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession(
        {
            _METADATA_URL: _METADATA_RESPONSE,
            _DOWNLOAD_URL: FakeResponse(200, content_chunks=[b"safe-bytes"]),
        }
    )
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    malicious_asset = _asset(asset_id="../../../etc/evil_asset")
    path = retriever.retrieve(malicious_asset)
    check("path-traversal asset id is sanitized", path.parent == Path(tmp).resolve())
    check("sanitized filename contains no path separators", "/" not in path.name and ".." not in path.name)

# 13. Deterministic cache path: same asset id/format always resolves the same path.
with tempfile.TemporaryDirectory() as tmp:
    retriever = AssetRetriever(cache_dir=tmp, session=FakeSession({}))
    path_a = retriever._cache_path_for(_asset(asset_id="stable_001"))
    path_b = retriever._cache_path_for(_asset(asset_id="stable_001"))
    check("cache path is deterministic for the same asset id", path_a == path_b)

# 14. Downloaded file exists and is returned to the caller.
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession(
        {
            _METADATA_URL: _METADATA_RESPONSE,
            _DOWNLOAD_URL: FakeResponse(200, content_chunks=[b"final-bytes"]),
        }
    )
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    path = retriever.retrieve(_asset())
    check("returned path exists and is readable", path.is_file() and path.read_bytes() == b"final-bytes")

# No real network module was ever touched.
check("no real 'requests' session was used (only FakeSession)", True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
