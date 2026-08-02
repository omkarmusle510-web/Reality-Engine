"""Hand landmark overlay drawing for the Reality Engine rendering layer.

Draws landmarks and their connections using OpenCV drawing functions only.
No window display, no gesture logic - purely image mutation.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from engine.core.pipeline import PipelineContext, StageFunc
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


def create_overlay_stage() -> StageFunc:
    """Builds a pipeline stage that draws detected hands onto the current frame.

    Reads `context["frame"]` and `context["hands"]`. If either is missing,
    the stage is a no-op.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _overlay_stage(context: PipelineContext) -> PipelineContext:
        frame = context.get("frame")
        hands = context.get("hands")
        if isinstance(frame, Frame) and hands is not None:
            draw_hands(frame.image, hands)
        return context

    return _overlay_stage