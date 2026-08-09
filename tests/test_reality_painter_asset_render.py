"""Focused integration test for Phase 12E.

Proves the Reality Painter pipeline integration added in
`apps.reality_painter.asset_render`: a loaded `SceneObject` in a
`Scene`, rendered via the existing `Renderer3D` and composited onto
the existing camera `Frame` through the actual pipeline stage
function `create_asset_render_stage` - the same stage
`apps/reality_painter/app.py` registers into its real pipeline.

Uses a deterministic local GLB fixture generated offline with
`trimesh` (no network, no GitHub, no AssetRetriever/registry
involvement) - Phase 12A/12B/12C are not touched or re-tested here.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from apps.reality_painter.asset_render import create_asset_render_stage
from engine.rendering.renderer import RenderError, Renderer3D
from engine.scene.loader import load_glb
from engine.scene.scene import Scene
from engine.vision.frame import Frame

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


try:
    renderer = Renderer3D(width=64, height=64)
except RenderError as exc:
    print(f"SKIPPED: Renderer3D unavailable in this environment ({exc})")
    sys.exit(0)

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    # Deterministic local GLB fixture - mirrors Phase 12D's fixture, not
    # a re-test of AssetRetriever/registry.
    import trimesh

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    glb_path = tmp_path / "box.glb"
    box.export(glb_path, file_type="glb")

    # cached asset (local file) -> load_glb -> Scene
    scene_object = load_glb(glb_path, name="test_box")
    scene = Scene()
    scene.add(scene_object)

    stage = create_asset_render_stage(scene, renderer)

    # Scene -> Renderer3D -> composite onto camera frame, via the real
    # pipeline stage function app.py registers.
    frame = Frame(image=np.full((64, 64, 3), 127, dtype=np.uint8), timestamp=0.0)
    context = {"frame": frame}
    result_context = stage(context)

    check("stage returns the same context", result_context is context)
    check("frame is modified by the rendered model", bool(np.any(frame.image != 127)))
    check("frame stays valid BGR uint8", frame.image.shape == (64, 64, 3) and frame.image.dtype == np.uint8)

    # Empty scene (no asset loaded) is a no-op - never forces a render.
    empty_scene = Scene()
    empty_stage = create_asset_render_stage(empty_scene, renderer)
    untouched_frame = Frame(image=np.full((64, 64, 3), 127, dtype=np.uint8), timestamp=0.0)
    empty_stage({"frame": untouched_frame})
    check("empty scene leaves the frame untouched", bool(np.all(untouched_frame.image == 127)))

    # Missing frame (stage runs before vision has produced one) is a no-op.
    no_frame_context = {}
    stage(no_frame_context)
    check("missing frame is a no-op", "frame" not in no_frame_context)

    renderer.close()

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
