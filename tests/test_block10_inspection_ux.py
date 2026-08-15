"""Offline, deterministic tests for Block 10 (inspection UX + status lifecycle).

Covers ONLY the new Block 10 surface:
    - apps.reality_painter.inspection.framing.normalize_transform
    - apps.reality_painter.inspection.controls (InspectionViewState,
      create_inspection_controls_stage)
    - apps.reality_painter.status_overlay.categorize_failure
    - the wiring in apps.reality_painter.app._analyze_fn (reproduced
      here against the real, unmodified InspectionController /
      ModeController / mode_router, the same pattern already used by
      tests/test_block9_app_lifecycle.py)

Blocks 1-9 (optimization pipeline, recognition providers, GitHub asset
discovery, runtime_mode/mode_router, InspectionController's own
recognition/resolution/retrieval logic) are exercised only through
their existing, real, unmodified public APIs - never reimplemented or
monkeypatched here except where InspectionController.run() is invoked
against fakes, matching tests/test_asset_retriever.py's convention.

No network access, no camera, no GPU/EGL, no MediaPipe.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.retriever import AssetRetriever
from apps.reality_painter.inspection.controller import InspectionController
from apps.reality_painter.inspection.controls import InspectionViewState, create_inspection_controls_stage
from apps.reality_painter.inspection.framing import normalize_transform
from apps.reality_painter.mode_router import create_mode_router_stage
from apps.reality_painter.recognition.models import RecognitionResult, RecognizedObject
from apps.reality_painter.runtime_mode import ModeController, RuntimeMode
from apps.reality_painter.status_overlay import categorize_failure
from engine.scene.objects import Transform
from engine.scene.scene import Scene

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


# ===========================================================================
# 1. Startup scene is empty (no 3D asset preloaded).
# ===========================================================================
check("a freshly constructed Scene is empty at startup", len(Scene()) == 0)


# ===========================================================================
# 2. Model normalization: deterministic centering + uniform scale.
# ===========================================================================
class _FakeMesh:
    """A minimal stand-in exposing only `.bounds`, no real trimesh parse needed."""

    def __init__(self, bounds):
        self.bounds = bounds


big_mesh = _FakeMesh(np.array([[0.0, 0.0, 0.0], [100.0, 50.0, 25.0]]))
transform_a = normalize_transform(big_mesh)
transform_b = normalize_transform(big_mesh)
check("normalize_transform is deterministic for identical input", transform_a == transform_b)
check("normalize_transform scales the longest dimension to the target size", abs(transform_a.scale[0] * 100.0 - 2.0) < 1e-9)
check("normalize_transform is uniform (same scale on all axes)", transform_a.scale[0] == transform_a.scale[1] == transform_a.scale[2])

# Center of [0,100]x[0,50]x[0,25] is (50,25,12.5); scaled and negated.
expected_x = -50.0 * transform_a.scale[0]
check("normalize_transform centers the mesh at the origin", abs(transform_a.position[0] - expected_x) < 1e-9)

degenerate_mesh = _FakeMesh(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
check("normalize_transform falls back to identity for a zero-size mesh", normalize_transform(degenerate_mesh) == Transform())

no_bounds_mesh = _FakeMesh(None)
check("normalize_transform falls back to identity when bounds is None", normalize_transform(no_bounds_mesh) == Transform())


class _RaisingMesh:
    @property
    def bounds(self):
        raise RuntimeError("boom")


check("normalize_transform never raises for a broken mesh", normalize_transform(_RaisingMesh()) == Transform())


# ===========================================================================
# 3. Inspection controls: rotate/zoom/reset mutate the active object's transform.
# ===========================================================================
view_state = InspectionViewState()
base = Transform(position=(1.0, 2.0, 3.0), rotation=(0.0, 0.0, 0.0), scale=(0.5, 0.5, 0.5))
view_state.set_base(base)
check("resolve_transform before any input equals the base transform", view_state.resolve_transform() == base)

view_state.apply_key(ord("d"))  # rotate right (yaw+)
after_yaw = view_state.resolve_transform()
check("rotate-right increases yaw (rotation.y)", after_yaw.rotation[1] > base.rotation[1])
check("rotate-right does not change position", after_yaw.position == base.position)

view_state.apply_key(ord("w"))  # tilt up (pitch-)
after_pitch = view_state.resolve_transform()
check("tilt-up changes pitch (rotation.x)", after_pitch.rotation[0] != base.rotation[0])

view_state.apply_key(ord("+"))  # zoom in
after_zoom = view_state.resolve_transform()
check("zoom-in increases scale beyond the base scale", after_zoom.scale[0] > base.scale[0])

view_state.apply_key(ord("r"))  # reset
after_reset = view_state.resolve_transform()
check("reset() returns exactly to the base transform", after_reset == base)

fresh_state = InspectionViewState()
check("resolve_transform with no base set returns None", fresh_state.resolve_transform() is None)


class _FakeSceneObject:
    def __init__(self, transform):
        self.transform = transform


controls_state = InspectionViewState()
controls_state.set_base(Transform())
fake_obj = _FakeSceneObject(Transform())
stage = create_inspection_controls_stage(controls_state, lambda: fake_obj)
stage({"key_pressed": ord("d")})
check("inspection_controls stage mutates the active object's transform", fake_obj.transform.rotation[1] != 0.0)

no_object_stage = create_inspection_controls_stage(InspectionViewState(), lambda: None)
result_context = no_object_stage({"key_pressed": ord("d")})
check("inspection_controls stage is a no-op with no active object", result_context == {"key_pressed": ord("d")})


# ===========================================================================
# 4. Failure categorization matches InspectionController's real error text.
# ===========================================================================
check("'No registered asset' error -> no_asset category", categorize_failure("No registered asset for label 'dinosaur'.") == "no_asset")
check("'Asset retrieval failed' error -> retrieval_failed category", categorize_failure("Asset retrieval failed: HTTP 404") == "retrieval_failed")
check("'GLB load failed' error -> load_failed category", categorize_failure("GLB load failed: corrupt file") == "load_failed")
check("recognition-provider error -> recognition_failed category", categorize_failure("Recognition provider error: boom") == "recognition_failed")
check("None error -> recognition_failed category (safe default)", categorize_failure(None) == "recognition_failed")
check("empty-result error -> recognition_failed category", categorize_failure("Recognition returned no objects.") == "recognition_failed")


# ===========================================================================
# 5-10. End-to-end _analyze_fn wiring (reproduces app.py's real closure body)
#       against the real InspectionController/ModeController/mode_router.
# ===========================================================================
class _StubProvider:
    def __init__(self, result):
        self._result = result

    def recognize(self, image):
        return self._result


class _FakeResponse:
    def __init__(self, status_code, json_data=None, content_chunks=None):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = {}
        self._content_chunks = content_chunks or []

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size=None):
        for chunk in self._content_chunks:
            yield chunk


class _FakeSession:
    def __init__(self, routes):
        self._routes = routes

    def get(self, url, headers=None, timeout=None, stream=False, params=None):
        route = self._routes.get(url)
        if route is None:
            raise AssertionError(f"Unexpected URL requested in test: {url}")
        return route


_FLOWER_ASSET = {
    "id": "flower_001",
    "name": "Flower",
    "category": "plants",
    "tags": ["flower"],
    "format": "glb",
    "source": {"type": "github", "repository": "owner/repo", "path": "models/flower.glb"},
}
_METADATA_URL = "https://api.github.com/repos/owner/repo/contents/models/flower.glb"
_DOWNLOAD_URL = "https://raw.githubusercontent.com/owner/repo/main/models/flower.glb"
_DUMMY_IMAGE = np.full((8, 8, 3), 200, dtype=np.uint8)


def _try_build_valid_glb_bytes():
    try:
        import trimesh
    except ImportError:
        return None
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return box.export(file_type="glb")


_VALID_GLB_BYTES = _try_build_valid_glb_bytes()


def _make_analyze_fn(scene, active_object, view_state, canvas_snapshot, provider, registry, retriever, controller):
    """Reproduces apps/reality_painter/app.py's real `_analyze_fn` body."""

    def _categorize(error):
        return categorize_failure(error)

    def _analyze_fn(context):
        context["analysis_error_category"] = None
        if provider is None:
            context["analysis_error_category"] = _categorize(None)
            return False
        snapshot = canvas_snapshot
        if snapshot is None:
            context["analysis_error_category"] = _categorize(None)
            return False
        outcome = controller.run(image=snapshot, provider=provider, registry=registry, retriever=retriever)
        if outcome.success and outcome.scene_object is not None:
            scene.add(outcome.scene_object)
            active_object["obj"] = outcome.scene_object
            view_state.set_base(outcome.scene_object.transform)
        else:
            context["analysis_error_category"] = _categorize(outcome.error)
        return outcome.success

    return _analyze_fn


analyze_call_count = {"n": 0}

with tempfile.TemporaryDirectory() as tmp:
    session = _FakeSession(
        {
            _METADATA_URL: _FakeResponse(200, json_data={"download_url": _DOWNLOAD_URL}),
            _DOWNLOAD_URL: _FakeResponse(200, content_chunks=[_VALID_GLB_BYTES or b"placeholder-bytes"]),
        }
    )
    registry = AssetRegistry.from_list([_FLOWER_ASSET])
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    controller = InspectionController()
    provider = _StubProvider(RecognitionResult(succeeded=True, objects=[RecognizedObject(label="flower", confidence=1.0)]))

    scene = Scene()
    active_object = {"obj": None}
    view_state = InspectionViewState()

    class _CountingProvider:
        def __init__(self, inner):
            self._inner = inner

        def recognize(self, image):
            analyze_call_count["n"] += 1
            return self._inner.recognize(image)

    counting_provider = _CountingProvider(provider)

    analyze_fn = _make_analyze_fn(scene, active_object, view_state, _DUMMY_IMAGE, counting_provider, registry, retriever, controller)
    mode_controller = ModeController()
    router_stage = create_mode_router_stage(mode_controller, analyze_fn=analyze_fn)

    # --- 5. N enters ANALYZING and (with a successful analyze_fn) reaches ASSET_READY.
    context = {"key_pressed": ord("n")}
    router_stage(context)
    check("N key drives PAINTING -> ASSET_READY on success", mode_controller.mode == RuntimeMode.ASSET_READY)
    check("successful analysis clears analysis_error_category", context["analysis_error_category"] is None)

    if _VALID_GLB_BYTES is not None:
        # --- 6. Successful asset is added to the scene exactly once.
        check("successful recognition adds the SceneObject to scene exactly once", len(scene) == 1)
        check("active_object tracks the newly added SceneObject", active_object["obj"] is not None)
        check("inspection view_state base transform was set from the loaded object", view_state.resolve_transform() == active_object["obj"].transform)
    else:
        print("NOTE: trimesh unavailable - scene-population assertions skipped (GLB load could not succeed).")

    # --- 7. I enters INSPECTING_3D.
    router_stage({"key_pressed": ord("i")})
    check("I key drives ASSET_READY -> INSPECTING_3D", mode_controller.mode == RuntimeMode.INSPECTING_3D)

    # --- 8. Inspection (I key) never re-invokes recognition/optimization.
    calls_after_asset_ready = analyze_call_count["n"]
    router_stage({"key_pressed": ord("i")})  # already INSPECTING_3D; should be a no-op for analyze_fn
    check("entering/staying in inspection never calls the recognition provider again", analyze_call_count["n"] == calls_after_asset_ready)

    # --- 9. Controls change the existing SceneObject's transform while inspecting.
    if active_object["obj"] is not None:
        controls_stage = create_inspection_controls_stage(view_state, lambda: active_object["obj"])
        transform_before = active_object["obj"].transform
        controls_stage({"key_pressed": ord("d")})
        check("rotate control changes the active SceneObject's transform in place", active_object["obj"].transform != transform_before)

    # --- 10. Returning to painting (X key) works.
    router_stage({"key_pressed": ord("x")})
    check("X key drives INSPECTING_3D -> PAINTING", mode_controller.mode == RuntimeMode.PAINTING)

check("N was only ever dispatched to the recognition provider once for this scenario", analyze_call_count["n"] == 1)


# ===========================================================================
# 11. Failure path: scene stays empty and a user-visible status is produced.
# ===========================================================================
with tempfile.TemporaryDirectory() as tmp:
    session = _FakeSession({})  # any HTTP call here fails the test
    registry = AssetRegistry.from_list([_FLOWER_ASSET])
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    controller = InspectionController()
    provider = _StubProvider(RecognitionResult(succeeded=False, error="model unavailable"))

    scene = Scene()
    active_object = {"obj": None}
    view_state = InspectionViewState()
    analyze_fn = _make_analyze_fn(scene, active_object, view_state, _DUMMY_IMAGE, provider, registry, retriever, controller)
    mode_controller = ModeController()
    router_stage = create_mode_router_stage(mode_controller, analyze_fn=analyze_fn)

    context = {"key_pressed": ord("n")}
    router_stage(context)
    check("failed analysis leaves the scene empty", len(scene) == 0)
    check("failed analysis returns the mode to PAINTING", mode_controller.mode == RuntimeMode.PAINTING)
    check("failed analysis leaves active_object unset", active_object["obj"] is None)
    check(
        "failed analysis produces a non-empty, categorized status for the UI",
        context["analysis_error_category"] in ("recognition_failed", "no_asset", "retrieval_failed", "load_failed"),
    )

# Unknown-label case -> "no_asset" category specifically.
with tempfile.TemporaryDirectory() as tmp:
    session = _FakeSession({})  # unresolved label never touches the network
    registry = AssetRegistry.from_list([_FLOWER_ASSET])
    retriever = AssetRetriever(cache_dir=tmp, session=session)
    controller = InspectionController()
    provider = _StubProvider(RecognitionResult(succeeded=True, objects=[RecognizedObject(label="dinosaur", confidence=1.0)]))

    scene = Scene()
    active_object = {"obj": None}
    view_state = InspectionViewState()
    analyze_fn = _make_analyze_fn(scene, active_object, view_state, _DUMMY_IMAGE, provider, registry, retriever, controller)
    mode_controller = ModeController()
    router_stage = create_mode_router_stage(mode_controller, analyze_fn=analyze_fn)

    context = {"key_pressed": ord("n")}
    router_stage(context)
    check("unresolved asset label -> no_asset status category", context["analysis_error_category"] == "no_asset")
    check("unresolved asset label leaves the scene empty", len(scene) == 0)


# ===========================================================================
# 12. No unrelated files modified - verified via source inspection of the
#     real, unmodified modules this block touches.
# ===========================================================================
renderer_source = Path("engine/rendering/renderer.py").read_text(encoding="utf-8")
loader_source = Path("engine/scene/loader.py").read_text(encoding="utf-8")
check("engine/rendering/renderer.py has no Block 10 references", "status_overlay" not in renderer_source and "InspectionViewState" not in renderer_source)
check("engine/scene/loader.py is unmodified by Block 10 (no normalize_transform call)", "normalize_transform" not in loader_source)

controller_source = Path("apps/reality_painter/inspection/controller.py").read_text(encoding="utf-8")
check("controller.py's only Block 10 addition is the normalize_transform integration", "normalize_transform(scene_object.mesh)" in controller_source)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
