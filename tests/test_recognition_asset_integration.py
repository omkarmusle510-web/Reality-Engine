"""Offline, deterministic tests for the recognition -> asset -> cache/retrieve
-> GLB-loading integration layer ONLY.

Does not re-test AssetRegistry, AssetRetriever, GitHub discovery,
load_glb, or Renderer3D themselves - those are already covered by
their own existing test files. Every GitHub HTTP call is mocked via a
fake session, matching the pattern already used in
tests/test_asset_retriever.py.
"""
import sys
import tempfile
from pathlib import Path

from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.retriever import AssetRetriever
from apps.reality_painter.inspection.asset_resolver import AssetResolutionStatus, resolve_asset
from apps.reality_painter.inspection.controller import ControllerOutcome, InspectionController, select_object
from apps.reality_painter.recognition.models import RecognitionResult, RecognizedObject

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


class _StubProvider:
    """A RecognitionProvider stand-in - returns a canned RecognitionResult, no network."""

    def __init__(self, result):
        self._result = result

    def recognize(self, image):
        return self._result


class _RaisingProvider:
    """A RecognitionProvider stand-in that raises, to prove the controller never propagates it."""

    def recognize(self, image):
        raise RuntimeError("provider exploded")


_FLOWER_ASSET = {
    "id": "flower_001",
    "name": "Flower",
    "category": "plants",
    "tags": ["flower", "plant"],
    "format": "glb",
    "source": {"type": "github", "repository": "owner/repo", "path": "models/flower.glb"},
}
_METADATA_URL = "https://api.github.com/repos/owner/repo/contents/models/flower.glb"
_DOWNLOAD_URL = "https://raw.githubusercontent.com/owner/repo/main/models/flower.glb"
_METADATA_RESPONSE = FakeResponse(200, json_data={"download_url": _DOWNLOAD_URL})


def _try_build_valid_glb_bytes():
    """Builds real GLB bytes via trimesh if available, else returns None."""
    try:
        import trimesh
    except ImportError:
        return None
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return box.export(file_type="glb")


_VALID_GLB_BYTES = _try_build_valid_glb_bytes()

# ===========================================================================
# 1. Recognition models
# ===========================================================================
result = RecognitionResult(succeeded=True, objects=[RecognizedObject(label="flower", confidence=0.9)])
check("RecognitionResult holds succeeded/objects", result.succeeded is True and len(result.objects) == 1)
check("RecognizedObject carries label/confidence", result.objects[0].label == "flower" and result.objects[0].confidence == 0.9)

empty_result = RecognitionResult(succeeded=False, error="no model output")
check("RecognitionResult supports a failed/empty outcome", empty_result.succeeded is False and empty_result.objects == [])
check("Failed RecognitionResult carries an error message", empty_result.error == "no model output")

# ===========================================================================
# 2. Asset resolution: known / unknown label
# ===========================================================================
registry = AssetRegistry.from_list([_FLOWER_ASSET])

known = resolve_asset("flower", registry)
check("known label resolves to RESOLVED", known.status == AssetResolutionStatus.RESOLVED)
check("resolved asset is the correct registry entry", known.asset is not None and known.asset.id == "flower_001")

unknown = resolve_asset("dinosaur", registry)
check("unknown label resolves to UNAVAILABLE", unknown.status == AssetResolutionStatus.UNAVAILABLE)
check("unresolved AssetResolution.asset is None", unknown.asset is None)

# ===========================================================================
# 3. Object selection: highest confidence + tie ordering
# ===========================================================================
candidates = [
    RecognizedObject(label="car", confidence=0.4),
    RecognizedObject(label="flower", confidence=0.9),
    RecognizedObject(label="house", confidence=0.2),
]
check("select_object picks the highest-confidence object", select_object(candidates).label == "flower")

tied = [
    RecognizedObject(label="first", confidence=0.5),
    RecognizedObject(label="second", confidence=0.5),
]
check("select_object preserves input order on a confidence tie", select_object(tied).label == "first")

check("select_object returns None for an empty list", select_object([]) is None)

# ===========================================================================
# 4/5/6. Controller: cache-hit / cache-miss / success / failure paths
# ===========================================================================
_recognized_flower = RecognitionResult(succeeded=True, objects=[RecognizedObject(label="flower", confidence=1.0)])

if _VALID_GLB_BYTES is None:
    print("NOTE: trimesh unavailable in this environment - GLB-load success/failure "
          "assertions fall back to the controller's generic exception handler rather "
          "than exercising trimesh's real parse path. Re-run on a machine with "
          "trimesh installed (per requirements.txt) for full confirmation.")

with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession({
        _METADATA_URL: _METADATA_RESPONSE,
        _DOWNLOAD_URL: FakeResponse(200, content_chunks=[_VALID_GLB_BYTES or b"placeholder-bytes"]),
    })
    registry = AssetRegistry.from_list([_FLOWER_ASSET])
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    controller = InspectionController()

    # --- 5/success path: cache MISS on first run -> GitHub retrieval invoked ---
    outcome = controller.run(image=object(), provider=_StubProvider(_recognized_flower), registry=registry, retriever=retriever)

    if _VALID_GLB_BYTES is not None:
        check("successful run: recognition -> asset -> retrieve -> GLB load -> success", outcome.success is True)
        check("successful run returns a SceneObject", outcome.scene_object is not None)
    else:
        check("run with invalid GLB bytes (trimesh unavailable) fails cleanly, no crash", isinstance(outcome, ControllerOutcome))

    check("successful/attempted run reports the selected label", outcome.selected_label == "flower")
    check("cache-miss path invoked GitHub retrieval (metadata + download URLs hit)",
          _METADATA_URL in session.requested_urls and _DOWNLOAD_URL in session.requested_urls)

    calls_after_first_run = len(session.requested_urls)

    # --- 4/cache-hit path: second run with the same registry/retriever -> no new GitHub calls ---
    outcome_2 = controller.run(image=object(), provider=_StubProvider(_recognized_flower), registry=registry, retriever=retriever)
    check("cache-hit path does not invoke GitHub retrieval again",
          len(session.requested_urls) == calls_after_first_run)

# --- 6a. recognition failure ---
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession({})  # any HTTP call here is a test failure
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    registry = AssetRegistry.from_list([_FLOWER_ASSET])
    failed_recognition = RecognitionResult(succeeded=False, error="model unavailable")
    outcome = InspectionController().run(image=object(), provider=_StubProvider(failed_recognition), registry=registry, retriever=retriever)
    check("recognition failure -> clean failure outcome", outcome.success is False and outcome.error == "model unavailable")
    check("recognition failure never touches the network", session.requested_urls == [])

# --- 6b. empty recognition result ---
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession({})
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    registry = AssetRegistry.from_list([_FLOWER_ASSET])
    empty_recognition = RecognitionResult(succeeded=True, objects=[])
    outcome = InspectionController().run(image=object(), provider=_StubProvider(empty_recognition), registry=registry, retriever=retriever)
    check("empty recognition result -> clean failure outcome", outcome.success is False)
    check("empty recognition result never touches the network", session.requested_urls == [])

# --- 6c. provider raises unexpectedly ---
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession({})
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    registry = AssetRegistry.from_list([_FLOWER_ASSET])
    outcome = InspectionController().run(image=object(), provider=_RaisingProvider(), registry=registry, retriever=retriever)
    check("provider raising an exception never escapes the controller", outcome.success is False)
    check("provider exception is reported in the outcome error", "provider exploded" in (outcome.error or ""))

# --- 6d. unknown/unavailable asset (recognized label has no registry match) ---
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession({})  # asset unavailable -> must never touch the network
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    registry = AssetRegistry.from_list([_FLOWER_ASSET])
    unresolvable = RecognitionResult(succeeded=True, objects=[RecognizedObject(label="dinosaur", confidence=1.0)])
    outcome = InspectionController().run(image=object(), provider=_StubProvider(unresolvable), registry=registry, retriever=retriever)
    check("unknown asset -> clean failure outcome", outcome.success is False)
    check("unavailable asset never triggers a retrieval attempt", session.requested_urls == [])

# --- 6e. retrieval failure (GitHub 404 on metadata) ---
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession({_METADATA_URL: FakeResponse(404)})
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    registry = AssetRegistry.from_list([_FLOWER_ASSET])
    outcome = InspectionController().run(image=object(), provider=_StubProvider(_recognized_flower), registry=registry, retriever=retriever)
    check("retrieval failure -> clean failure outcome, no crash", outcome.success is False)
    check("retrieval failure error message is reported", "retrieval failed" in (outcome.error or "").lower())

# --- 6f. GLB load failure (download succeeds, but content is not a valid GLB) ---
with tempfile.TemporaryDirectory() as tmp:
    session = FakeSession({
        _METADATA_URL: _METADATA_RESPONSE,
        _DOWNLOAD_URL: FakeResponse(200, content_chunks=[b"this is not a real glb file"]),
    })
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    registry = AssetRegistry.from_list([_FLOWER_ASSET])
    outcome = InspectionController().run(image=object(), provider=_StubProvider(_recognized_flower), registry=registry, retriever=retriever)
    check("corrupt GLB content -> clean failure outcome, no crash", outcome.success is False)
    check("GLB load failure reports an error message", bool(outcome.error))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
