"""MediaPipe-backed hand tracking.

MediaPipe is a dependency, not part of the engine. This module owns
MediaPipe Hands entirely and converts every result into engine `Hand`
objects before returning - no MediaPipe type ever leaves this module.
"""

from __future__ import annotations

from typing import List

import mediapipe as mp

from engine.core.logger import get_logger
from engine.core.pipeline import PipelineContext, StageFunc
from engine.tracking.hand import Hand, Landmark
from engine.vision.frame import Frame

logger = get_logger(__name__)


class HandTracker:
    """Detects hands in a `Frame` and returns engine `Hand` objects."""

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.5) -> None:
        """Creates a hand tracker backed by MediaPipe Hands.

        Args:
            max_num_hands: Maximum number of hands to detect per frame.
            min_detection_confidence: Minimum confidence for a detection
                to be considered valid.
        """
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame: Frame) -> List[Hand]:
        """Detects hands in the given frame.

        Args:
            frame: Camera frame to analyze (BGR image).

        Returns:
            Detected hands as engine `Hand` objects. Empty if none found.
        """
        rgb_image = frame.image[:, :, ::-1]
        results = self._hands.process(rgb_image)

        if not results.multi_hand_landmarks:
            return []

        handedness_list = results.multi_handedness or []
        hands: List[Hand] = []

        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = [Landmark(x=point.x, y=point.y, z=point.z) for point in hand_landmarks.landmark]

            label = "Unknown"
            confidence = 0.0
            if index < len(handedness_list):
                classification = handedness_list[index].classification[0]
                label = classification.label
                confidence = classification.score

            hands.append(Hand(handedness=label, confidence=confidence, landmarks=landmarks))

        return hands

    def close(self) -> None:
        """Releases MediaPipe resources."""
        self._hands.close()


def create_tracking_stage(tracker: HandTracker) -> StageFunc:
    """Builds a pipeline stage that detects hands in the current frame.

    Reads `context["frame"]` and stores detected hands in `context["hands"]`.
    If no frame is present, the stage is a no-op.

    Args:
        tracker: A `HandTracker` instance.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _tracking_stage(context: PipelineContext) -> PipelineContext:
        frame = context.get("frame")
        if isinstance(frame, Frame):
            context["hands"] = tracker.detect(frame)
        return context

    return _tracking_stage