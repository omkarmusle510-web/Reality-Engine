"""Hand landmark overlay drawing for the Reality Engine rendering layer.

Draws landmarks, their connections, and a small developer debug HUD
(FPS, mouse-control state, active gesture) using OpenCV drawing functions
only. No window display, no gesture/action/toggle decision logic - this
module only renders values that other stages have already computed and
placed in the pipeline context.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from engine.core.pipeline import PipelineContext, StageFunc
from engine.interaction.Gesture import Gesture
from engine.tracking.hand import Hand
from engine.vision.frame import Frame

_LANDMARK_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky finger
    (0, 17),                                   # palm base
]

_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_COLOR = (0, 255, 0)
_HUD_LINE_HEIGHT_PX = 25
_HUD_ORIGIN_X = 10
_HUD_ORIGIN_Y = 30


def draw_hands(image: np.ndarray, hands: List[Hand]) -> np.ndarray:
    """Draws landmarks and connections for each hand onto the image.

    Args:
        image: BGR image to draw on. Mutated in place.
        hands: Hands to draw.

    Returns:
        The same image, with landmarks and connections drawn on it.
    """
    height, width = image.shape[:2]

    for hand in hands:
        points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in hand.landmarks]

        for start_index, end_index in _LANDMARK_CONNECTIONS:
            cv2.line(image, points[start_index], points[end_index], (0, 255, 0), 2)

        for point in points:
            cv2.circle(image, point, 4, (0, 0, 255), -1)

    return image


def draw_debug_hud(image: np.ndarray, fps: float, mouse_enabled: bool, gesture: Optional[Gesture]) -> np.ndarray:
    """Draws developer debug text in the upper-left corner.

    Purely a rendering function: it draws whatever values it is given
    and makes no decisions about FPS, mouse-control state, or gesture
    recognition itself.

    Args:
        image: BGR image to draw on. Mutated in place.
        fps: Current smoothed frames-per-second value.
        mouse_enabled: Whether OS mouse control is currently enabled.
        gesture: The primary hand's recognized gesture, or `None` if no
            hand is currently tracked.

    Returns:
        The same image, with debug text drawn on it.
    """
    lines = [
        f"FPS: {fps:.1f}",
        f"Mouse: {'ON' if mouse_enabled else 'OFF'}",
        f"Gesture: {gesture.name if gesture is not None else 'NONE'}",
    ]

    for line_index, line in enumerate(lines):
        y = _HUD_ORIGIN_Y + line_index * _HUD_LINE_HEIGHT_PX
        cv2.putText(image, line, (_HUD_ORIGIN_X, y), _HUD_FONT, 0.7, _HUD_COLOR, 2)

    return image


def create_overlay_stage() -> StageFunc:
    """Builds a pipeline stage that draws hands and the debug HUD onto the current frame.

    Reads `context["frame"]`, `context["hands"]`, `context["fps"]`,
    `context["mouse_enabled"]`, and `context["gestures"]`. Hand landmarks
    are drawn only if both a frame and hands are present, matching prior
    behavior. The debug HUD is drawn whenever a frame is present, even if
    no hand is currently detected, since FPS and mouse-control state are
    useful to see at all times.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _overlay_stage(context: PipelineContext) -> PipelineContext:
        frame = context.get("frame")
        if not isinstance(frame, Frame):
            return context

        hands = context.get("hands")
        if hands is not None:
            draw_hands(frame.image, hands)

        fps = context.get("fps", 0.0)
        mouse_enabled = context.get("mouse_enabled", True)
        gestures = context.get("gestures")
        primary_gesture = gestures[0] if gestures else None
        draw_debug_hud(frame.image, fps, mouse_enabled, primary_gesture)

        return context

    return _overlay_stage