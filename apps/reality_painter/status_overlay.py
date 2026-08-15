"""User-facing status overlay for Reality Painter's runtime-mode lifecycle.

Draws a single, always-visible status line summarizing the current
`apps.reality_painter.runtime_mode.RuntimeMode` in plain language -
never the internal enum name - plus, after a recognition/asset
failure, a short retry hint. This is Reality-Painter-specific UI,
drawn onto `context["frame"]` before `engine.rendering.overlay`'s
generic HUD runs - the same "bake application UI into the frame first"
pattern `apps.reality_painter.menu` already uses (see
`engine.rendering.overlay`'s module docstring).

This module performs no recognition, retrieval, optimization, or
network access - it only reads `context["runtime_mode"]` (written by
`apps.reality_painter.mode_router`) and an optional
`context["analysis_error_category"]` to choose which message to draw.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2

from apps.reality_painter.runtime_mode import RuntimeMode
from engine.core.pipeline import PipelineContext, StageFunc
from engine.vision.frame import Frame

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_COLOR = (255, 255, 255)
_BG_COLOR = (0, 0, 0)
_BG_ALPHA = 0.55
_MARGIN_X = 10
_MARGIN_Y = 10
_LINE_HEIGHT = 24
_FONT_SCALE = 0.6
_FONT_THICKNESS = 2

#: Plain-language status text per `RuntimeMode` - never the raw enum
#: value, so normal users never see internal names like "ASSET_READY".
_MODE_MESSAGES: Dict[RuntimeMode, str] = {
    RuntimeMode.PAINTING: "Painting - press N to find a 3D model",
    RuntimeMode.ANALYZING: "Analyzing drawing... finding 3D asset",
    RuntimeMode.ASSET_READY: "3D model ready - press I to inspect",
    RuntimeMode.INSPECTING_3D: "Inspecting 3D - A/D rotate, W/S tilt, +/- zoom, R reset, X back",
}

#: Plain-language failure text per category - see `categorize_failure`.
_FAILURE_MESSAGES: Dict[str, str] = {
    "recognition_failed": "Couldn't recognize the drawing. Try again.",
    "no_asset": "Recognized, but no 3D model is available for it.",
    "retrieval_failed": "3D model found but could not be downloaded.",
    "load_failed": "3D model was found but could not be loaded.",
}

_DEFAULT_FAILURE_CATEGORY = "recognition_failed"


def categorize_failure(error: Optional[str]) -> str:
    """Maps a raw `ControllerOutcome.error` message to a status category.

    Category keys match `_FAILURE_MESSAGES` above. Matching is done by
    substring against the exact wording
    `apps.reality_painter.inspection.controller.InspectionController`
    already produces for each failure path, so this never invents or
    guesses a new failure taxonomy. An unrecognized or empty error
    message defaults to the broadest failure class, "recognition_failed"
    (covers a missing/failing recognition provider and an
    empty/unsuccessful recognition result).

    Args:
        error: `ControllerOutcome.error`, or `None`.

    Returns:
        One of `"no_asset"`, `"retrieval_failed"`, `"load_failed"`, or
        `"recognition_failed"`.
    """
    message = error or ""
    if "No registered asset" in message:
        return "no_asset"
    if "Asset retrieval failed" in message:
        return "retrieval_failed"
    if "GLB load failed" in message:
        return "load_failed"
    return _DEFAULT_FAILURE_CATEGORY


def _status_text(runtime_mode_value: Optional[str], failure_category: Optional[str]) -> Optional[str]:
    """Resolves the user-facing status line for the current state."""
    if runtime_mode_value == RuntimeMode.PAINTING.value and failure_category:
        return _FAILURE_MESSAGES.get(failure_category, _FAILURE_MESSAGES[_DEFAULT_FAILURE_CATEGORY])

    for mode, message in _MODE_MESSAGES.items():
        if runtime_mode_value == mode.value:
            return message
    return None


def create_status_overlay_stage() -> StageFunc:
    """Builds a pipeline stage that draws the current lifecycle status onto the frame.

    Reads `context["frame"]`, `context["runtime_mode"]`, and
    `context["analysis_error_category"]`. A no-op if no frame is
    present yet. Always runs (regardless of runtime mode) so the user
    is never left without a status line - including immediately after
    a failure, since failure text is shown only while back in
    `PAINTING` and only until the next analysis attempt clears it.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _status_overlay_stage(context: PipelineContext) -> PipelineContext:
        frame = context.get("frame")
        if not isinstance(frame, Frame):
            return context

        text = _status_text(context.get("runtime_mode"), context.get("analysis_error_category"))
        if not text:
            return context

        height, width = frame.image.shape[:2]
        (text_width, _text_height), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        panel_right = min(width, _MARGIN_X + text_width + 2 * _MARGIN_X)
        panel_bottom = min(height, _MARGIN_Y + _LINE_HEIGHT + _MARGIN_Y)
        if panel_right <= _MARGIN_X or panel_bottom <= _MARGIN_Y:
            return context

        roi = frame.image[_MARGIN_Y:panel_bottom, _MARGIN_X:panel_right]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), _BG_COLOR, -1)
        cv2.addWeighted(overlay, _BG_ALPHA, roi, 1 - _BG_ALPHA, 0, dst=roi)

        text_origin = (_MARGIN_X + _MARGIN_X // 2, _MARGIN_Y + _LINE_HEIGHT - 6)
        cv2.putText(frame.image, text, text_origin, _FONT, _FONT_SCALE, _TEXT_COLOR, _FONT_THICKNESS, cv2.LINE_AA)

        return context

    return _status_overlay_stage
