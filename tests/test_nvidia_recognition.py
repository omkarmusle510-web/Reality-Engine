"""Offline, deterministic tests for the Phase 3 NVIDIA recognition provider.

No real NVIDIA API calls - `requests.post` is mocked directly on the
`apps.reality_painter.recognition.providers.nvidia` module. Does not
re-test AssetRegistry, AssetRetriever, load_glb, or runtime_mode/mode_router
themselves - those are already covered by their own existing test files;
this file only proves the new NVIDIA provider and its wiring through the
existing, unmodified InspectionController and mode_router.
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import apps.reality_painter.recognition.providers.nvidia as nvidia_module
from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.retriever import AssetRetriever
from apps.reality_painter.inspection.controller import InspectionController
from apps.reality_painter.mode_router import create_mode_router_stage
from apps.reality_painter.recognition.models import RecognitionResult, RecognizedObject
from apps.reality_painter.recognition.providers.nvidia import NvidiaRecognitionProvider
from apps.reality_painter.runtime_mode import ModeController, RuntimeMode

import requests

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


_DUMMY_IMAGE = np.full((8, 8, 3), 200, dtype=np.uint8)


def _fake_response(status_code, json_data=None, json_raises=False):
    response = Mock()
    response.status_code = status_code
    if json_raises:
        response.json.side_effect = ValueError("bad json")
    else:
        response.json.return_value = json_data
    return response


# ===========================================================================
# 1. Request-building: endpoint, model, auth header, one image, text+image.
# ===========================================================================
with patch.object(nvidia_module.requests, "post") as mock_post:
    mock_post.return_value = _fake_response(200, {"choices": [{"message": {"content": "flower\nlooks like petals"}}]})
    provider = NvidiaRecognitionProvider(api_key="nvapi-test-key")
    provider.recognize(_DUMMY_IMAGE)

    check("requests.post was called exactly once", mock_post.call_count == 1)
    call_args, call_kwargs = mock_post.call_args
    check("correct endpoint used", call_args[0] == "https://integrate.api.nvidia.com/v1/chat/completions")
    check("correct model in payload", call_kwargs["json"]["model"] == "meta/llama-3.2-11b-vision-instruct")
    check("Authorization header uses the supplied API key", call_kwargs["headers"]["Authorization"] == "Bearer nvapi-test-key")

    content = call_kwargs["json"]["messages"][0]["content"]
    image_blocks = [block for block in content if block.get("type") == "image_url"]
    text_blocks = [block for block in content if block.get("type") == "text"]
    check("exactly one image included in the request", len(image_blocks) == 1)
    check("text block present alongside the image", len(text_blocks) == 1)
    check("image sent as a base64 data URI", image_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,"))

# ===========================================================================
# 2. Successful mocked response -> RecognitionResult(succeeded=True).
# ===========================================================================
with patch.object(nvidia_module.requests, "post") as mock_post:
    mock_post.return_value = _fake_response(
        200, {"choices": [{"message": {"content": "Flower\nRounded petals arranged in a circle."}}]}
    )
    provider = NvidiaRecognitionProvider(api_key="nvapi-test-key")
    result = provider.recognize(_DUMMY_IMAGE)

    check("successful response -> RecognitionResult(succeeded=True)", result.succeeded is True)
    check("label parsed and lowercased", len(result.objects) == 1 and result.objects[0].label == "flower")
    check("reasoning preserved when returned", "petals" in (result.objects[0].reasoning or ""))

# ===========================================================================
# 3. Malformed/empty model response -> clean failure, no exception escapes.
# ===========================================================================
with patch.object(nvidia_module.requests, "post") as mock_post:
    mock_post.return_value = _fake_response(200, json_raises=True)
    provider = NvidiaRecognitionProvider(api_key="nvapi-test-key")
    result = provider.recognize(_DUMMY_IMAGE)
    check("invalid JSON body -> clean failed RecognitionResult", result.succeeded is False and bool(result.error))

with patch.object(nvidia_module.requests, "post") as mock_post:
    mock_post.return_value = _fake_response(200, {"choices": []})
    provider = NvidiaRecognitionProvider(api_key="nvapi-test-key")
    result = provider.recognize(_DUMMY_IMAGE)
    check("empty 'choices' -> clean failed RecognitionResult", result.succeeded is False and bool(result.error))

with patch.object(nvidia_module.requests, "post") as mock_post:
    mock_post.return_value = _fake_response(200, {"choices": [{"message": {"content": "   "}}]})
    provider = NvidiaRecognitionProvider(api_key="nvapi-test-key")
    result = provider.recognize(_DUMMY_IMAGE)
    check("whitespace-only content -> clean failed RecognitionResult", result.succeeded is False)

# ===========================================================================
# 4. HTTP 401/403 -> clean failure, key never exposed.
# ===========================================================================
with patch.object(nvidia_module.requests, "post") as mock_post:
    mock_post.return_value = _fake_response(401)
    provider = NvidiaRecognitionProvider(api_key="nvapi-super-secret-key")
    result = provider.recognize(_DUMMY_IMAGE)
    check("HTTP 401 -> clean failed RecognitionResult", result.succeeded is False)
    check("API key never appears in the 401 error message", "nvapi-super-secret-key" not in (result.error or ""))

with patch.object(nvidia_module.requests, "post") as mock_post:
    mock_post.return_value = _fake_response(403)
    provider = NvidiaRecognitionProvider(api_key="nvapi-super-secret-key")
    result = provider.recognize(_DUMMY_IMAGE)
    check("HTTP 403 -> clean failed RecognitionResult", result.succeeded is False)
    check("API key never appears in the 403 error message", "nvapi-super-secret-key" not in (result.error or ""))

# ===========================================================================
# 5. Timeout / network failure -> clean failure, no exception escapes.
# ===========================================================================
with patch.object(nvidia_module.requests, "post", side_effect=requests.exceptions.Timeout("boom")):
    provider = NvidiaRecognitionProvider(api_key="nvapi-test-key")
    result = provider.recognize(_DUMMY_IMAGE)
    check("timeout -> clean failed RecognitionResult, no exception escapes", result.succeeded is False)

with patch.object(nvidia_module.requests, "post", side_effect=requests.exceptions.ConnectionError("boom")):
    provider = NvidiaRecognitionProvider(api_key="nvapi-test-key")
    result = provider.recognize(_DUMMY_IMAGE)
    check("connection error -> clean failed RecognitionResult, no exception escapes", result.succeeded is False)

# ===========================================================================
# 6. Unknown/unmapped label -> controller clean failure, no retrieval attempt.
# ===========================================================================
class _FakeGetSession:
    """Any HTTP call here fails the test - proves no retrieval was attempted."""

    def get(self, *args, **kwargs):
        raise AssertionError("Unexpected network call during unknown-label test.")


with patch.object(nvidia_module.requests, "post") as mock_post:
    mock_post.return_value = _fake_response(200, {"choices": [{"message": {"content": "dinosaur"}}]})
    provider = NvidiaRecognitionProvider(api_key="nvapi-test-key")

    with tempfile.TemporaryDirectory() as tmp:
        empty_registry = AssetRegistry()
        retriever = AssetRetriever(cache_dir=tmp, session=_FakeGetSession())
        outcome = InspectionController().run(image=_DUMMY_IMAGE, provider=provider, registry=empty_registry, retriever=retriever)
        check("unmapped label -> controller reports clean failure", outcome.success is False)
        check("unmapped label never triggers a retrieval attempt", "No registered asset" in (outcome.error or ""))

# ===========================================================================
# 7. Known label -> full flow reaches ASSET_READY-equivalent success.
# ===========================================================================
_FLOWER_ASSET = {
    "id": "flower_001",
    "name": "Flower",
    "category": "plants",
    "tags": ["flower"],
    "format": "glb",
    "source": {"type": "github", "repository": "owner/repo", "path": "models/flower.glb"},
}


class _FakeDownloadResponse:
    def __init__(self, status_code, json_data=None, content_chunks=None):
        self.status_code = status_code
        self._json_data = json_data
        self._content_chunks = content_chunks or []
        self.headers = {}

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size=None):
        for chunk in self._content_chunks:
            yield chunk


def _try_build_valid_glb_bytes():
    try:
        import trimesh
    except ImportError:
        return None
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return box.export(file_type="glb")


_VALID_GLB_BYTES = _try_build_valid_glb_bytes()
_METADATA_URL = "https://api.github.com/repos/owner/repo/contents/models/flower.glb"
_DOWNLOAD_URL = "https://raw.githubusercontent.com/owner/repo/main/models/flower.glb"


class _FakeGitHubSession:
    def get(self, url, headers=None, timeout=None, stream=False, params=None):
        if url == _METADATA_URL:
            return _FakeDownloadResponse(200, json_data={"download_url": _DOWNLOAD_URL})
        if url == _DOWNLOAD_URL:
            return _FakeDownloadResponse(200, content_chunks=[_VALID_GLB_BYTES or b"placeholder"])
        raise AssertionError(f"Unexpected URL requested: {url}")


with patch.object(nvidia_module.requests, "post") as mock_post:
    mock_post.return_value = _fake_response(200, {"choices": [{"message": {"content": "flower\nfive petals visible"}}]})
    provider = NvidiaRecognitionProvider(api_key="nvapi-test-key")

    with tempfile.TemporaryDirectory() as tmp:
        registry = AssetRegistry.from_list([_FLOWER_ASSET])
        retriever = AssetRetriever(cache_dir=tmp, session=_FakeGitHubSession())
        outcome = InspectionController().run(image=_DUMMY_IMAGE, provider=provider, registry=registry, retriever=retriever)

        if _VALID_GLB_BYTES is not None:
            check("known label: full flow reaches ASSET_READY-equivalent success", outcome.success is True)
            check("known label: outcome carries a loaded SceneObject", outcome.scene_object is not None)
        else:
            print("NOTE: trimesh unavailable in this environment - GLB-load assertions skipped.")
        check("known label: selected label recorded on outcome", outcome.selected_label == "flower")

# ===========================================================================
# 8. Recognition success does NOT enter INSPECTING_3D.
# ===========================================================================
mode_controller = ModeController()
router_stage = create_mode_router_stage(mode_controller, analyze_fn=lambda ctx: True)
router_stage({"key_pressed": ord("n")})
check("successful analyze_fn drives PAINTING -> ASSET_READY (not INSPECTING_3D)", mode_controller.mode == RuntimeMode.ASSET_READY)
check("mode is explicitly not INSPECTING_3D after a successful analyze", mode_controller.mode != RuntimeMode.INSPECTING_3D)

# A subsequent, separate 'I' key press is required to actually enter 3D.
router_stage({"key_pressed": ord("i")})
check("explicit 3D key is still required to enter INSPECTING_3D", mode_controller.mode == RuntimeMode.INSPECTING_3D)

# ===========================================================================
# 9. Controller remains provider-agnostic: a fake provider works identically.
# ===========================================================================
class _FakeProvider:
    def recognize(self, image):
        return RecognitionResult(succeeded=True, objects=[RecognizedObject(label="flower", confidence=1.0)])


with tempfile.TemporaryDirectory() as tmp:
    registry = AssetRegistry.from_list([_FLOWER_ASSET])
    retriever = AssetRetriever(cache_dir=tmp, session=_FakeGitHubSession())
    outcome = InspectionController().run(image=_DUMMY_IMAGE, provider=_FakeProvider(), registry=registry, retriever=retriever)
    if _VALID_GLB_BYTES is not None:
        check("controller works identically with a fake (non-NVIDIA) provider", outcome.success is True)
    check("fake provider path resolves the same label", outcome.selected_label == "flower")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
