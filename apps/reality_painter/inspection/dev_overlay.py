"""Developer diagnostics overlay for Reality Painter's 3D inspection view.

`engine.rendering.overlay.create_overlay_stage` (the existing developer
HUD) is gated entirely off during `INSPECTING_3D` (see
`apps.reality_painter.mode_router.PAINTING_MODES`), so INSPECTING_3D is
clean by default - no radial menu, no color palette, no brush/eraser
UI, and no developer HUD. This module adds back an *optional* developer
HUD for that mode only, reusing the existing developer-HUD renderer
(`engine.rendering.overlay.draw_debug_hud`) rather than building a
second debug architecture - it only owns a small on/off toggle and
calls that same function with whatever diagnostic values are available
while inspecting (no hand tracking, no gestures, no mouse control in
this mode).
"""

from __future__ import annotations

from engine.core.pipeline import PipelineContext, StageFunc
from engine.rendering.overlay import draw_debug_hud
from engine.vision.frame import Frame

#: Toggles the 3D-inspection developer HUD. Chosen to match the
#: README's existing "H = Toggle Developer HUD" hint; unused elsewhere
#: in Reality Painter's key bindings.
_DEV_HUD_TOGGLE_KEYS = (ord("h"), ord("H"))


class DevHudToggle:
    """Tracks a persistent visible/hidden flag for the 3D-inspection developer HUD.

    Off by default, mirroring `engine.interaction.mouse_toggle
    .MouseToggle`'s on/off pattern but starting hidden, since a clean
    view is the required default for INSPECTING_3D.
    """

    def __init__(self) -> None:
        self._visible = False

    def update(self, toggle_requested: bool) -> bool:
        """Applies a pending toggle request (if any) and returns current state."""
        if toggle_requested:
            self._visible = not self._visible
        return self._visible


def create_inspection_dev_overlay_stage(toggle: DevHudToggle) -> StageFunc:
    """Builds a stage that draws the existing developer HUD while inspecting 3D, if toggled on.

    Reads `context["frame"]` and `context["key_pressed"]`. A no-op
    (developer HUD hidden, per `DevHudToggle`'s default) unless 'H' has
    been pressed an odd number of times since this stage's owning
    `DevHudToggle` was created. Intended to be registered gated to
    `apps.reality_painter.mode_router.INSPECTION_MODES` only - see
    `apps/reality_painter/app.py`.

    Args:
        toggle: A `DevHudToggle` instance, owned by the caller so its
            state persists across pipeline executions.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _inspection_dev_overlay_stage(context: PipelineContext) -> PipelineContext:
        frame = context.get("frame")
        if not isinstance(frame, Frame):
            return context

        key_pressed = context.get("key_pressed")
        visible = toggle.update(key_pressed in _DEV_HUD_TOGGLE_KEYS)
        if not visible:
            return context

        draw_debug_hud(
            frame.image,
            fps=context.get("fps", 0.0),
            mouse_enabled=False,
            gesture=None,
            action=None,
            cursor=None,
            hand_count=0,
            tracking_label="N/A",
            transition_text=None,
            ai_status=context.get("ai_status"),
        )
        return context

    return _inspection_dev_overlay_stage
