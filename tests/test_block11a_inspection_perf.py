"""Focused, offline tests for Block 11A (inspection performance + live camera).

Covers ONLY the new Block 11A contract:
    - apps.reality_painter.asset_render's 3D render cache: unchanged
      scene-object transforms reuse the cached render; a changed
      transform (as the 3D inspection controls in
      apps.reality_painter.inspection.controls already produce)
      invalidates it.
    - apps.reality_painter.mode_router no longer freezes/replaces
      context["frame"] on entering INSPECTING_3D (the camera stays
      live - see apps/reality_painter/app.py's now-ungated vision/
      mirror stages).

No network, no camera, no GPU/EGL/pyrender - a fake renderer (the same
dependency-injection convention already used by
tests/test_asset_benchmark.py/test_optimization_pipeline.py) stands in
for Renderer3D so this file needs no display/OpenGL context. Does not
re-test Blocks 1-10's own recognition/asset/optimizer/framing logic.
"""
import sys

import numpy as np

from apps.reality_painter.asset_render import create_asset_render_stage
from apps.reality_painter.mode_router import create_mode_router_stage
from apps.reality_painter.runtime_mode import ModeController, RuntimeMode
from engine.scene.objects import SceneObject, Transform
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


class _FakeRenderer:
    """Fake Renderer3D-shaped renderer: counts calls, returns a tiny RGBA buffer."""

    def __init__(self):
        self.render_calls = 0

    def render(self, scene):
        self.render_calls += 1
        return np.full((4, 4, 4), 255, dtype=np.uint8)


# ===========================================================================
# 1 & 2. Render cache: unchanged transform -> renderer not called again;
#        changed transform -> renderer is called again.
# ===========================================================================
scene = Scene()
obj = SceneObject(mesh=object(), transform=Transform(), name="test_obj")
scene.add(obj)

fake_renderer = _FakeRenderer()
stage = create_asset_render_stage(scene, fake_renderer)

frame = Frame(image=np.full((8, 8, 3), 100, dtype=np.uint8), timestamp=0.0)

stage({"frame": frame})
check("first cycle renders once", fake_renderer.render_calls == 1)

stage({"frame": frame})
stage({"frame": frame})
check("unchanged transform across further cycles never re-renders", fake_renderer.render_calls == 1)

# A fresh camera frame every cycle (Block 11A) must not itself
# invalidate the cache - only a transform change should.
fresh_frame = Frame(image=np.full((8, 8, 3), 50, dtype=np.uint8), timestamp=1.0)
stage({"frame": fresh_frame})
check("a new camera frame object alone does not invalidate the render cache", fake_renderer.render_calls == 1)

obj.transform = Transform(rotation=(0.0, 0.2, 0.0))
stage({"frame": fresh_frame})
check("a changed transform invalidates the cache and re-renders", fake_renderer.render_calls == 2)

stage({"frame": fresh_frame})
check("cache is populated again after the re-render", fake_renderer.render_calls == 2)

# Empty scene / missing frame are no-ops, never touch the renderer.
empty_scene = Scene()
empty_stage = create_asset_render_stage(empty_scene, fake_renderer)
empty_stage({"frame": fresh_frame})
check("empty scene never calls the renderer", fake_renderer.render_calls == 2)

no_frame_calls_before = fake_renderer.render_calls
stage({})
check("missing frame never calls the renderer", fake_renderer.render_calls == no_frame_calls_before)


# ===========================================================================
# 3. Compositing actually touches the frame pixels (cache reuse still composites).
# ===========================================================================
composited_frame = Frame(image=np.full((4, 4, 3), 10, dtype=np.uint8), timestamp=2.0)
composite_scene = Scene()
composite_scene.add(SceneObject(mesh=object(), transform=Transform(), name="c"))
composite_stage = create_asset_render_stage(composite_scene, _FakeRenderer())
composite_stage({"frame": composited_frame})
check("compositing (even via cache) still writes the rendered pixels onto the frame", bool(np.any(composited_frame.image != 10)))


# ===========================================================================
# 4 & 5. Camera stays live: entering INSPECTING_3D no longer freezes/
#        replaces context["frame"] - mode_router only flips the mode.
# ===========================================================================
mode_controller = ModeController()
mode_controller.request_analyze()
mode_controller.analysis_succeeded()  # -> ASSET_READY

router_stage = create_mode_router_stage(mode_controller)
live_frame = Frame(image=np.full((4, 4, 3), 9, dtype=np.uint8), timestamp=3.0)
enter_context = {"key_pressed": ord("i"), "frame": live_frame}
router_stage(enter_context)

check("entering INSPECTING_3D via 'I' still transitions the mode", mode_controller.mode == RuntimeMode.INSPECTING_3D)
check("entering INSPECTING_3D no longer replaces context['frame'] (camera stays live)", enter_context["frame"] is live_frame)
check("entering INSPECTING_3D never mutates the frame's pixels itself", np.array_equal(enter_context["frame"].image, live_frame.image))


print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
