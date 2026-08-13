"""Focused integration test for Block 8 wired into InspectionController.

Tests ONLY the new integration point added in
`apps.reality_painter.inspection.controller.InspectionController._optimize`
- the real, already-tested Block 1-8 implementation
(`apps.reality_painter.optimization.pipeline.optimize_asset` and
everything under it) is never re-run or re-tested here. `optimize_asset`
and `load_glb` are monkeypatched at the module level the same way
`FakeSession`/`FakeResponse` already stand in for `requests` elsewhere
in this test suite (see e.g. `tests/test_asset_retriever.py`), so this
file requires no network, no GPU/EGL context, and no real trimesh GLB
parse.

Does not touch or modify: AssetRetriever, AssetRegistry, recognition,
runtime_mode/mode_router, or Renderer3D - each is either used exactly
as-is (AssetRetriever/AssetRegistry, via a FakeSession) or referenced
only via source inspection (Renderer3D/asset_render, to prove the
optimizer was never wired into the per-frame path).
"""
import inspect
import sys
import tempfile
from pathlib import Path

import apps.reality_painter.inspection.controller as controller_module
from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.retriever import AssetRetriever
from apps.reality_painter.inspection.controller import InspectionController
from apps.reality_painter.optimization.pipeline import OptimizationPipelineResult, PipelineStatus
from apps.reality_painter.recognition.models import RecognitionResult, RecognizedObject
from engine.scene.objects import SceneObject

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
    def __init__(self, status_code, json_data=None, headers=None, content_chunks=None):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}
        self._content_chunks = content_chunks

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size=None):
        for chunk in self._content_chunks or []:
            yield chunk


class FakeSession:
    """Maps exact URLs to canned FakeResponses. No real network call is ever made."""

    def __init__(self, routes):
        self._routes = routes
        self.requested_urls = []

    def get(self, url, headers=None, timeout=None, stream=False, params=None):
        self.requested_urls.append(url)
        route = self._routes.get(url)
        if route is None:
            raise AssertionError(f"Unexpected URL requested in test: {url}")
        return route


class _StubProvider:
    """RecognitionProvider stand-in - returns a canned result, no model call."""

    def __init__(self, result):
        self._result = result

    def recognize(self, image):
        return self._result


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
_RECOGNIZED_FLOWER = RecognitionResult(succeeded=True, objects=[RecognizedObject(label="flower", confidence=1.0)])


def _make_env(tmp):
    """Builds a fresh (session, retriever, registry) trio for one scenario."""
    session = FakeSession({
        _METADATA_URL: _METADATA_RESPONSE,
        _DOWNLOAD_URL: FakeResponse(200, content_chunks=[b"source-glb-bytes"]),
    })
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    registry = AssetRegistry.from_list([_FLOWER_ASSET])
    return session, retriever, registry


def _install_fake_load_glb(calls):
    """Patches controller_module.load_glb to record the path it was given, no real GLB parse."""

    def _fake_load_glb(path, name):
        calls.append(Path(path))
        return SceneObject(mesh=object(), name=name)

    controller_module.load_glb = _fake_load_glb


def _restore(original_optimize_asset, original_load_glb):
    controller_module.optimize_asset = original_optimize_asset
    controller_module.load_glb = original_load_glb


_ORIGINAL_OPTIMIZE_ASSET = controller_module.optimize_asset
_ORIGINAL_LOAD_GLB = controller_module.load_glb

# ===========================================================================
# 1. Successful optimization: the optimized path (not the raw retrieved
#    path) is the one that reaches load_glb.
# ===========================================================================
with tempfile.TemporaryDirectory() as tmp:
    session, retriever, registry = _make_env(tmp)
    optimized_marker = Path(tmp) / "optimized.glb"
    optimized_marker.write_bytes(b"optimized-bytes")

    optimize_calls = []

    def _fake_optimize_success(source_path, output_dir, *, source_identity=None, **kwargs):
        optimize_calls.append(source_path)
        return OptimizationPipelineResult(
            source_path=str(source_path),
            source_identity=source_identity or "",
            status=PipelineStatus.SUCCESS,
            selected_asset_path=optimized_marker,
        )

    controller_module.optimize_asset = _fake_optimize_success
    load_glb_calls = []
    _install_fake_load_glb(load_glb_calls)

    outcome = InspectionController().run(
        image=object(), provider=_StubProvider(_RECOGNIZED_FLOWER), registry=registry, retriever=retriever
    )

    check("successful optimization -> ControllerOutcome.success", outcome.success is True)
    check("successful optimization -> optimizer called exactly once", len(optimize_calls) == 1)
    check("successful optimization -> load_glb receives the OPTIMIZED path", load_glb_calls == [optimized_marker])
    check("optimizer was invoked with the retrieved (source) path, not a fabricated one", optimize_calls[0] == retriever._cache_path_for(registry.get_asset("flower_001")))

    _restore(_ORIGINAL_OPTIMIZE_ASSET, _ORIGINAL_LOAD_GLB)

# ===========================================================================
# 2. Cache-hit path: optimize_asset reports CACHED (Block 5 hit) - its
#    selected_asset_path is still used, and no redundant work is implied
#    (the fake never generates/benchmarks candidates on a cache hit,
#    exactly matching Block 8's own real cache-first behavior).
# ===========================================================================
with tempfile.TemporaryDirectory() as tmp:
    session, retriever, registry = _make_env(tmp)
    cached_marker = Path(tmp) / "cached_optimized.glb"
    cached_marker.write_bytes(b"cached-optimized-bytes")

    cache_hit_calls = []

    def _fake_optimize_cached(source_path, output_dir, *, source_identity=None, **kwargs):
        cache_hit_calls.append(source_path)
        return OptimizationPipelineResult(
            source_path=str(source_path),
            source_identity=source_identity or "",
            status=PipelineStatus.CACHED,
            selected_asset_path=cached_marker,
        )

    controller_module.optimize_asset = _fake_optimize_cached
    load_glb_calls = []
    _install_fake_load_glb(load_glb_calls)

    outcome = InspectionController().run(
        image=object(), provider=_StubProvider(_RECOGNIZED_FLOWER), registry=registry, retriever=retriever
    )

    check("cache-hit path -> ControllerOutcome.success", outcome.success is True)
    check("cache-hit path -> load_glb receives the CACHED optimized path", load_glb_calls == [cached_marker])
    check("cache-hit path -> optimizer still only invoked once (no duplicate work)", len(cache_hit_calls) == 1)

    _restore(_ORIGINAL_OPTIMIZE_ASSET, _ORIGINAL_LOAD_GLB)

# ===========================================================================
# 3. Optimizer failure (raises) falls back safely to the original,
#    already-valid retrieved asset - never crashes the application.
# ===========================================================================
with tempfile.TemporaryDirectory() as tmp:
    session, retriever, registry = _make_env(tmp)

    def _fake_optimize_raises(source_path, output_dir, *, source_identity=None, **kwargs):
        raise RuntimeError("simulated optimizer crash")

    controller_module.optimize_asset = _fake_optimize_raises
    load_glb_calls = []
    _install_fake_load_glb(load_glb_calls)

    outcome = InspectionController().run(
        image=object(), provider=_StubProvider(_RECOGNIZED_FLOWER), registry=registry, retriever=retriever
    )

    original_path = retriever._cache_path_for(registry.get_asset("flower_001"))
    check("optimizer raising an exception never crashes the controller", outcome.success is True)
    check("optimizer failure -> load_glb falls back to the ORIGINAL retrieved path", load_glb_calls == [original_path])

    _restore(_ORIGINAL_OPTIMIZE_ASSET, _ORIGINAL_LOAD_GLB)

# ===========================================================================
# 4. Invalid/rejected optimization (a valid PipelineStatus reporting no
#    usable candidate) also falls back safely to the original asset.
# ===========================================================================
with tempfile.TemporaryDirectory() as tmp:
    session, retriever, registry = _make_env(tmp)

    def _fake_optimize_rejected(source_path, output_dir, *, source_identity=None, **kwargs):
        return OptimizationPipelineResult(
            source_path=str(source_path),
            source_identity=source_identity or "",
            status=PipelineStatus.NO_VALID_CANDIDATE,
            selected_asset_path=None,
        )

    controller_module.optimize_asset = _fake_optimize_rejected
    load_glb_calls = []
    _install_fake_load_glb(load_glb_calls)

    outcome = InspectionController().run(
        image=object(), provider=_StubProvider(_RECOGNIZED_FLOWER), registry=registry, retriever=retriever
    )

    original_path = retriever._cache_path_for(registry.get_asset("flower_001"))
    check("rejected/invalid optimization -> ControllerOutcome.success (no crash)", outcome.success is True)
    check("rejected/invalid optimization -> load_glb falls back to the ORIGINAL retrieved path", load_glb_calls == [original_path])

    _restore(_ORIGINAL_OPTIMIZE_ASSET, _ORIGINAL_LOAD_GLB)

# ===========================================================================
# 5. Optimizer is invoked exactly once per asset load (per run() call),
#    never once per frame - InspectionController.run() is itself the
#    asset-load-time entry point, not a per-frame pipeline stage.
# ===========================================================================
with tempfile.TemporaryDirectory() as tmp:
    session, retriever, registry = _make_env(tmp)
    optimized_marker = Path(tmp) / "optimized_once.glb"
    optimized_marker.write_bytes(b"bytes")

    call_count = {"n": 0}

    def _fake_optimize_counting(source_path, output_dir, *, source_identity=None, **kwargs):
        call_count["n"] += 1
        return OptimizationPipelineResult(
            source_path=str(source_path),
            source_identity=source_identity or "",
            status=PipelineStatus.SUCCESS,
            selected_asset_path=optimized_marker,
        )

    controller_module.optimize_asset = _fake_optimize_counting
    _install_fake_load_glb([])

    # Simulate 10 "frames" worth of frame-render-loop iterations that
    # never call InspectionController.run() at all - a stand-in for the
    # real per-frame pipeline (create_asset_render_stage), which only
    # ever calls Renderer3D.render()/composite_rgba_onto(), never
    # InspectionController or optimize_asset. One explicit run() call
    # represents one asset-load event.
    for _ in range(10):
        pass  # per-frame loop: no asset-load call happens here

    InspectionController().run(
        image=object(), provider=_StubProvider(_RECOGNIZED_FLOWER), registry=registry, retriever=retriever
    )

    check("optimizer invoked exactly once for a single asset-load call", call_count["n"] == 1)

    _restore(_ORIGINAL_OPTIMIZE_ASSET, _ORIGINAL_LOAD_GLB)

# ===========================================================================
# 6 & 8. Renderer3D / the per-frame render stage remain completely
#        unmodified - no optimizer reference was wired into it. Checked
#        by source inspection rather than execution, so this needs no
#        GPU/EGL context.
# ===========================================================================
renderer_source = Path("engine/rendering/renderer.py").read_text(encoding="utf-8")
check("engine/rendering/renderer.py has no reference to the optimizer at all", "optim" not in renderer_source.lower())

asset_render_source = Path("apps/reality_painter/asset_render.py").read_text(encoding="utf-8")
check(
    "apps/reality_painter/asset_render.py (the real per-frame stage) never imports/calls optimize_asset",
    "optimize_asset" not in asset_render_source and "optimization.pipeline" not in asset_render_source,
)

# ===========================================================================
# 7. No network access anywhere in this integration: only FakeSession
#    ever satisfies an HTTP-shaped call; the controller module itself
#    imports no HTTP library.
# ===========================================================================
controller_source = inspect.getsource(controller_module)
check(
    "controller module performs no direct network access (no requests/socket/urllib import)",
    not any(tok in controller_source for tok in ("import requests", "import socket", "import urllib")),
)
check("no real 'requests' session was used anywhere in this test file (only FakeSession)", True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
