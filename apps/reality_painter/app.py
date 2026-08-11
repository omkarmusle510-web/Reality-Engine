"""RealityPainter application entry point.

Boots the engine, wires the full continuous pipeline (vision through
display), runs it until stopped, and shuts down cleanly.
"""

from __future__ import annotations

import os

from apps.reality_painter.ai.manager import AIManager
from apps.reality_painter.ai.prompt_builder import PromptBuilder
from apps.reality_painter.ai.providers.cloudflare import CloudflareProvider
from apps.reality_painter.ai.providers.gemini import GeminiProvider
from apps.reality_painter.ai.sketch_analyzer import SketchAnalyzer
from apps.reality_painter.asset_render import create_asset_render_stage
from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.retriever import AssetRetrievalError, AssetRetriever
from apps.reality_painter.mode_router import (
    INSPECTION_MODES,
    PAINTING_MODES,
    create_mode_router_stage,
    gate,
)
from apps.reality_painter.runtime_mode import ModeController
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
from engine.rendering.renderer import RenderError, Renderer3D
from engine.scene.loader import ModelLoadError, load_glb
from engine.scene.scene import Scene
from engine.tracking.hand_tracker import HandTracker, create_tracking_stage
from engine.vision.camera import Camera
from engine.vision.mirror import create_mirror_stage
from engine.vision.pipeline import create_vision_stage

logger = get_logger(__name__)


def _load_display_asset(scene: Scene) -> None:
    """Retrieves and loads the first registered asset into `scene`, if any.

    Best-effort: an empty registry, a retrieval failure, or a load
    failure all leave `scene` empty rather than raising, so a missing
    or not-yet-retrievable 3D asset never prevents Reality Painter from
    starting. Retrieval (Phase 12C) and loading (Phase 12D) both
    happen here, once, at startup - never inside the pipeline stage
    itself (see `apps.reality_painter.asset_render`).
    """
    registry = AssetRegistry.load()
    available_assets = registry.list_assets()
    if not available_assets:
        logger.info("Asset registry is empty - no 3D asset to display.")
        return

    asset = available_assets[0]
    try:
        local_path = AssetRetriever().retrieve(asset)
        scene_object = load_glb(local_path, name=asset.id)
    except (AssetRetrievalError, ModelLoadError) as exc:
        logger.warning("Could not load 3D asset '%s' for display: %s", asset.id, exc)
        return

    scene.add(scene_object)
    logger.info("Loaded 3D asset '%s' for display.", asset.id)


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

    scene = Scene()
    try:
        renderer_3d = Renderer3D()
    except RenderError as exc:
        renderer_3d = None
        logger.warning("3D renderer unavailable - 3D asset display disabled: %s", exc)

    if renderer_3d is not None:
        _load_display_asset(scene)

    ai_manager = AIManager(prompt_builder=PromptBuilder(), sketch_analyzer=SketchAnalyzer())

    # Registered first so AIManager.select_provider() (called with no
    # `preferred` argument by sketch.py's AI trigger) picks Cloudflare
    # over Gemini whenever both are available - registration order is
    # the existing preference mechanism (see AIManager.select_provider),
    # so this needs no change to the manager itself.
    cloudflare_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    cloudflare_api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if cloudflare_account_id and cloudflare_api_token:
        ai_manager.register_provider(CloudflareProvider(account_id=cloudflare_account_id, api_token=cloudflare_api_token))
        logger.info("Cloudflare provider registered - AI generation ('A' key) will prefer Cloudflare.")
    else:
        logger.warning("CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_API_TOKEN not set - Cloudflare provider unavailable.")

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        ai_manager.register_provider(GeminiProvider(api_key=gemini_api_key))
        logger.info("Gemini provider registered - available as fallback for AI generation.")
    else:
        logger.warning("GEMINI_API_KEY not set - Gemini fallback provider unavailable.")

    # Runtime-mode router: owns PAINTING/ANALYZING/ASSET_READY/INSPECTING_3D
    # transitions (N=analyze, I=enter 3D, X=exit 3D). Registered first and
    # ungated so a mode change always takes effect before any gated stage
    # below runs in the same cycle. Every stage that belongs to the
    # painting side or the 3D-inspection side is wrapped with `gate()` so
    # only one side's stages do real work on any given cycle - see
    # apps.reality_painter.mode_router for how exclusivity is enforced.
    mode_controller = ModeController()
    engine.pipeline.register_stage("mode_router", create_mode_router_stage(mode_controller))

    engine.pipeline.register_stage("emergency_exit", create_emergency_exit_stage(emergency_exit))
    engine.pipeline.register_stage("vision", gate(create_vision_stage(camera), mode_controller, PAINTING_MODES))
    engine.pipeline.register_stage("mirror", gate(create_mirror_stage(), mode_controller, PAINTING_MODES))
    if renderer_3d is not None:
        engine.pipeline.register_stage(
            "asset_render", gate(create_asset_render_stage(scene, renderer_3d), mode_controller, PAINTING_MODES)
        )
        engine.pipeline.register_stage(
            "inspection_render",
            gate(create_asset_render_stage(scene, renderer_3d), mode_controller, INSPECTION_MODES),
        )
    engine.pipeline.register_stage("fps", create_fps_stage(fps_counter))
    engine.pipeline.register_stage("tracking", gate(create_tracking_stage(tracker), mode_controller, PAINTING_MODES))
    engine.pipeline.register_stage("gesture", gate(create_gesture_stage(), mode_controller, PAINTING_MODES))
    engine.pipeline.register_stage("cursor", gate(create_cursor_stage(cursor_smoother), mode_controller, PAINTING_MODES))
    engine.pipeline.register_stage("action", gate(create_action_stage(action_mapper), mode_controller, PAINTING_MODES))
    engine.pipeline.register_stage(
        "mouse_toggle", gate(create_mouse_toggle_stage(mouse_toggle), mode_controller, PAINTING_MODES)
    )
    engine.pipeline.register_stage(
        "mouse_controller", gate(create_mouse_controller_stage(mouse_controller), mode_controller, PAINTING_MODES)
    )
    engine.pipeline.register_stage(
        "painting", gate(create_painting_stage(canvas, tool_state, ai_manager), mode_controller, PAINTING_MODES)
    )
    engine.pipeline.register_stage("overlay", gate(create_overlay_stage(), mode_controller, PAINTING_MODES))
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
        if renderer_3d is not None:
            renderer_3d.close()
        engine.shutdown()


if __name__ == "__main__":
    run()