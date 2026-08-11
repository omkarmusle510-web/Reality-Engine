"""Focused, offline, deterministic tests for the runtime-mode architecture.

Covers ONLY:
    1. legal state transitions
    2. illegal transitions rejected
    3. explicit Analyze -> ANALYZING
    4. successful analysis -> ASSET_READY
    5. explicit 3D action -> INSPECTING_3D
    6. explicit exit key -> PAINTING
    7. PAINTING executes the painting pipeline, not the inspection pipeline
    8. INSPECTING_3D executes the inspection pipeline, not the painting pipeline
    9. entering 3D freezes/copies the current frame
    10. inspection mode performs no continuous camera reads

No network access, no camera, no MediaPipe, no pyrender - stages are
exercised directly as plain functions against a fake context.
"""
import sys

import numpy as np

from apps.reality_painter.mode_router import (
    INSPECTION_MODES,
    PAINTING_MODES,
    create_mode_router_stage,
    gate,
)
from apps.reality_painter.runtime_mode import ModeController, RuntimeMode
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


# ===========================================================================
# 1 & 2. Legal transitions accepted, illegal transitions rejected.
# ===========================================================================
controller = ModeController()
check("initial mode is PAINTING", controller.mode == RuntimeMode.PAINTING)

check("PAINTING -> ANALYZING is legal", controller.transition_to(RuntimeMode.ANALYZING))
check("mode is now ANALYZING", controller.mode == RuntimeMode.ANALYZING)

check("ANALYZING -> INSPECTING_3D is illegal", controller.transition_to(RuntimeMode.INSPECTING_3D) is False)
check("illegal transition left mode unchanged", controller.mode == RuntimeMode.ANALYZING)

check("ANALYZING -> ASSET_READY is legal", controller.transition_to(RuntimeMode.ASSET_READY))
check("PAINTING -> PAINTING (self) is illegal", ModeController().transition_to(RuntimeMode.PAINTING) is False)

check("ASSET_READY -> INSPECTING_3D is legal", controller.transition_to(RuntimeMode.INSPECTING_3D))
check("INSPECTING_3D -> ASSET_READY is illegal", controller.transition_to(RuntimeMode.ASSET_READY) is False)
check("INSPECTING_3D -> PAINTING is legal", controller.transition_to(RuntimeMode.PAINTING))

# Full required flow, start to finish, via the convenience methods.
flow_controller = ModeController()
check("flow: request_analyze -> ANALYZING", flow_controller.request_analyze())
check("flow: analysis_succeeded -> ASSET_READY", flow_controller.analysis_succeeded())
check("flow: enter_inspection -> INSPECTING_3D", flow_controller.enter_inspection())
check("flow: exit_inspection -> PAINTING", flow_controller.exit_inspection())
check("flow ends back at PAINTING", flow_controller.mode == RuntimeMode.PAINTING)

# ===========================================================================
# 3 & 4. Explicit Analyze -> ANALYZING, successful analysis -> ASSET_READY.
# ===========================================================================
analyze_controller = ModeController()
router_stage = create_mode_router_stage(analyze_controller, analyze_fn=lambda ctx: True)
context = {"key_pressed": ord("n")}
router_stage(context)
check("explicit Analyze key drives PAINTING -> ANALYZING -> ASSET_READY", analyze_controller.mode == RuntimeMode.ASSET_READY)
check("router stage writes context['runtime_mode']", context["runtime_mode"] == RuntimeMode.ASSET_READY.value)

# A failed analysis returns to PAINTING instead.
fail_controller = ModeController()
fail_router = create_mode_router_stage(fail_controller, analyze_fn=lambda ctx: False)
fail_router({"key_pressed": ord("N")})
check("failed analysis returns ANALYZING -> PAINTING", fail_controller.mode == RuntimeMode.PAINTING)

# With no analyze_fn wired, ANALYZING is entered and held (no auto-advance).
held_controller = ModeController()
held_router = create_mode_router_stage(held_controller, analyze_fn=None)
held_router({"key_pressed": ord("n")})
check("Analyze key with no analyze_fn holds at ANALYZING", held_controller.mode == RuntimeMode.ANALYZING)

# Analyze key is ignored outside PAINTING.
ignored_controller = ModeController()
ignored_controller.request_analyze()  # -> ANALYZING
ignored_router = create_mode_router_stage(ignored_controller, analyze_fn=lambda ctx: True)
ignored_router({"key_pressed": ord("n")})
check("Analyze key ignored while already ANALYZING", ignored_controller.mode == RuntimeMode.ANALYZING)

# ===========================================================================
# 5. Explicit 3D action -> INSPECTING_3D.
# ===========================================================================
ready_controller = ModeController()
ready_controller.request_analyze()
ready_controller.analysis_succeeded()  # -> ASSET_READY
enter_router = create_mode_router_stage(ready_controller)
frame_before = Frame(image=np.full((4, 4, 3), 9, dtype=np.uint8), timestamp=1.0)
enter_context = {"key_pressed": ord("i"), "frame": frame_before}
enter_router(enter_context)
check("explicit 3D key drives ASSET_READY -> INSPECTING_3D", ready_controller.mode == RuntimeMode.INSPECTING_3D)

# Ignored outside ASSET_READY.
early_controller = ModeController()
early_router = create_mode_router_stage(early_controller)
early_router({"key_pressed": ord("i")})
check("3D key ignored while PAINTING", early_controller.mode == RuntimeMode.PAINTING)

# ===========================================================================
# 6. Explicit exit key -> PAINTING.
# ===========================================================================
exit_router = create_mode_router_stage(ready_controller)  # ready_controller is already INSPECTING_3D
exit_router({"key_pressed": ord("x")})
check("explicit exit key drives INSPECTING_3D -> PAINTING", ready_controller.mode == RuntimeMode.PAINTING)

# ===========================================================================
# 7 & 8. Pipeline exclusivity: exactly one gated group executes per mode.
# ===========================================================================
calls = {"painting_stage": 0, "inspection_stage": 0}


def _painting_stage(ctx):
    calls["painting_stage"] += 1
    return ctx


def _inspection_stage(ctx):
    calls["inspection_stage"] += 1
    return ctx


exclusivity_controller = ModeController()
gated_painting = gate(_painting_stage, exclusivity_controller, PAINTING_MODES)
gated_inspection = gate(_inspection_stage, exclusivity_controller, INSPECTION_MODES)

# In PAINTING: painting-side runs, inspection-side does not.
gated_painting({})
gated_inspection({})
check("PAINTING executes the painting pipeline", calls["painting_stage"] == 1)
check("PAINTING does not execute the inspection pipeline", calls["inspection_stage"] == 0)

# Drive to INSPECTING_3D and repeat.
exclusivity_controller.request_analyze()
exclusivity_controller.analysis_succeeded()
exclusivity_controller.enter_inspection()
gated_painting({})
gated_inspection({})
check("INSPECTING_3D executes the inspection pipeline", calls["inspection_stage"] == 1)
check("INSPECTING_3D does not execute the painting pipeline", calls["painting_stage"] == 1)  # unchanged from before

# ===========================================================================
# 9. Entering 3D freezes/copies the current frame.
# ===========================================================================
check("frame object was replaced (not mutated in place) on entering 3D", enter_context["frame"] is not frame_before)
check("frozen frame pixels match the source frame at freeze time", np.array_equal(enter_context["frame"].image, frame_before.image))
frame_before.image[:] = 255
check("mutating the original frame after freeze does not affect the frozen copy", not np.array_equal(enter_context["frame"].image, frame_before.image))

# ===========================================================================
# 10. Inspection mode performs no continuous camera reads.
# ===========================================================================
camera_reads = {"count": 0}


def _fake_vision_stage(ctx):
    camera_reads["count"] += 1
    return ctx


camera_controller = ModeController()
gated_vision = gate(_fake_vision_stage, camera_controller, PAINTING_MODES)

# Several PAINTING cycles do read the camera.
for _ in range(3):
    gated_vision({})
check("camera is read during PAINTING", camera_reads["count"] == 3)

# Drive to INSPECTING_3D; further cycles must not call camera.read().
camera_controller.request_analyze()
camera_controller.analysis_succeeded()
camera_controller.enter_inspection()
for _ in range(5):
    gated_vision({})
check("no additional camera reads occur while INSPECTING_3D", camera_reads["count"] == 3)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
