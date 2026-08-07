"""RealityPainter application entry point.

Boots the engine, wires the full continuous pipeline (vision through
display), runs it until stopped, and shuts down cleanly.
"""

from __future__ import annotations

from apps.reality_painter.sketch import Canvas, ToolState, create_painting_stage
from engine.core.config import config as EngineConfig
from engine.core.emergency_exit import EmergencyExit, create_emergency_exit_stage
from engine.core.engine import Engine
from engine.core.fps import FPSCounter, create_fps_stage
from engine.core.logger import get_logger
from engine.interaction.action_mapper import ActionMapper, create_action_stage
from engine.interaction.cursor_mapper import CursorSmoother, create_cursor_stage
from engine.interaction.gesture_recognizer import create_gesture_stage
from engine.interaction.mouse_controller import MouseController, create_mouse_controller_stage
from engine.interaction.mouse_toggle import MouseToggle, create_mouse_toggle_stage
from engine.rendering.display import DisplayWindow, create_display_stage
from engine.rendering.overlay import create_overlay_stage
from engine.tracking.hand_tracker import HandTracker, create_tracking_stage
from engine.vision.camera import Camera
from engine.vision.mirror import create_mirror_stage
from engine.vision.pipeline import create_vision_stage

logger = get_logger(__name__)


def run() -> None:
    """Boots the engine, runs the full pipeline until stopped, and shuts down."""
    engine_config = EngineConfig(app_name="RealityPainter")
    engine = Engine(engine_config)
    engine.initialize()

    camera = Camera()
    try:
        camera.open()
    except RuntimeError:
        logger.error("Camera failed to open - shutting down.")
        engine.shutdown()
        return

    tracker = HandTracker()
    cursor_smoother = CursorSmoother()
    action_mapper = ActionMapper()
    mouse_toggle = MouseToggle()
    mouse_controller = MouseController()
    canvas = Canvas()
    tool_state = ToolState()
    fps_counter = FPSCounter()
    display = DisplayWindow()
    emergency_exit = EmergencyExit()

    engine.pipeline.register_stage("emergency_exit", create_emergency_exit_stage(emergency_exit))
    engine.pipeline.register_stage("vision", create_vision_stage(camera))
    engine.pipeline.register_stage("mirror", create_mirror_stage())
    engine.pipeline.register_stage("fps", create_fps_stage(fps_counter))
    engine.pipeline.register_stage("tracking", create_tracking_stage(tracker))
    engine.pipeline.register_stage("gesture", create_gesture_stage())
    engine.pipeline.register_stage("cursor", create_cursor_stage(cursor_smoother))
    engine.pipeline.register_stage("action", create_action_stage(action_mapper))
    engine.pipeline.register_stage("mouse_toggle", create_mouse_toggle_stage(mouse_toggle))
    engine.pipeline.register_stage("mouse_controller", create_mouse_controller_stage(mouse_controller))
    engine.pipeline.register_stage("painting", create_painting_stage(canvas, tool_state))
    engine.pipeline.register_stage("overlay", create_overlay_stage())
    engine.pipeline.register_stage("display", create_display_stage(display))

    logger.info("Starting engine.")

    try:
        engine.start()
    finally:
        display.close()
        tracker.close()
        camera.release()
        mouse_controller.release()
        emergency_exit.close()
        engine.shutdown()


if __name__ == "__main__":
    run()