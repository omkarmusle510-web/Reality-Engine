"""RealityPainter application entry point.

Boots the engine, wires the full continuous pipeline (vision through
display), runs it until stopped, and shuts down cleanly.
"""

from __future__ import annotations

import os

from apps.reality_painter.ai.cache import InMemoryGenerationCache
from apps.reality_painter.ai.manager import AIManager
from apps.reality_painter.ai.models import AICapability
from apps.reality_painter.ai.prompt_builder import PromptBuilder
from apps.reality_painter.ai.sketch_analyzer import SketchAnalyzer
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


def _build_ai_manager() -> AIManager:
    """Builds the application's single `AIManager`.

    Constructed exactly once, here, and passed into the painting stage
    - never recreated per frame. Registers `GeminiProvider`/
    `GroqProvider` only when their API key is present in the
    environment; a missing key, a missing optional SDK dependency
    (`google-generativeai`, `groq` - neither is a hard requirement of
    this repository), or any other provider construction error is
    logged and skipped rather than raised. AI is an optional subsystem:
    Reality Painter must start and run normally with zero providers
    registered, in which case `AIManager.generate()` still returns a
    clean failed `AIResponse` rather than the app ever seeing an
    exception.
    """
    ai_manager = AIManager(
        prompt_builder=PromptBuilder(),
        sketch_analyzer=SketchAnalyzer(),
        cache=InMemoryGenerationCache(),
    )

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        try:
            from apps.reality_painter.ai.providers.gemini import GeminiProvider

            ai_manager.register_provider(
                GeminiProvider(
                    api_key=gemini_api_key,
                    model_name=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
                    capabilities=frozenset({AICapability.IMAGE_GENERATION}),
                )
            )
        except Exception:
            logger.exception("Gemini provider unavailable - continuing without it.")

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key:
        try:
            from apps.reality_painter.ai.providers.groq import GroqProvider

            ai_manager.register_provider(
                GroqProvider(
                    api_key=groq_api_key,
                    model_name=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
                )
            )
        except Exception:
            logger.exception("Groq provider unavailable - continuing without it.")

    return ai_manager


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
    ai_manager = _build_ai_manager()
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
    engine.pipeline.register_stage("painting", create_painting_stage(canvas, tool_state, ai_manager))
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