"""MediaPipe-backed hand tracking.

MediaPipe is a dependency, not part of the engine. This module owns
MediaPipe Hands entirely and converts every result into engine `Hand`
objects before returning - no MediaPipe type ever leaves this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from engine.core.logger import get_logger
from engine.core.pipeline import PipelineContext, StageFunc
from engine.tracking.hand import Hand, Landmark
from engine.vision.frame import Frame

logger = get_logger(__name__)

_DEFAULT_MODEL_ASSET_PATH = "assets/models/hand_landmarker.task"


class HandTracker:
    """Detects hands in a `Frame` and returns engine `Hand` objects."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        model_asset_path: str = _DEFAULT_MODEL_ASSET_PATH,
    ) -> None:
        """Creates a hand tracker backed by MediaPipe's Hand Landmarker task.

        Args:
            max_num_hands: Maximum number of hands to detect per frame.
            min_detection_confidence: Minimum confidence for a detection
                to be considered valid.
            model_asset_path: Path to the `hand_landmarker.task` model
                file. Unlike the legacy `mp.solutions.hands` API this
                replaces, MediaPipe's current Tasks API does not bundle
                a model internally - one must be supplied explicitly.

        Raises:
            FileNotFoundError: If `model_asset_path` does not exist.
                Checked here, at construction, rather than at first
                frame - consistent with "initialize once" and so a
                missing asset fails loudly before the pipeline starts,
                not mid-stream on the first camera frame.
        """
        if not Path(model_asset_path).is_file():
            raise FileNotFoundError(
                f"Hand Landmarker model asset not found at '{model_asset_path}'. "
                "Obtain 'hand_landmarker.task' from MediaPipe's official model "
                "index and place it at this path, or pass a different "
                "model_asset_path to HandTracker."
            )

        base_options = mp_tasks.BaseOptions(model_asset_path=model_asset_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame: Frame) -> List[Hand]:
        """Detects hands in the given frame.

        Args:
            frame: Camera frame to analyze (BGR image).

        Returns:
            Detected hands as engine `Hand` objects. Empty if none found.
        """
        rgb_image = frame.image[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # VIDEO running mode requires strictly increasing millisecond
        # timestamps. `frame.timestamp` is `time.monotonic()`-based and
        # therefore already monotonic, but frames faster than 1ms apart
        # (not expected from a camera) could collide after rounding; the
        # increment fallback guarantees strict monotonicity regardless.
        timestamp_ms = int(frame.timestamp * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return []

        handedness_list = result.handedness or []
        hands: List[Hand] = []

        for index, hand_landmarks in enumerate(result.hand_landmarks):
            landmarks = [Landmark(x=point.x, y=point.y, z=point.z) for point in hand_landmarks]

            label = "Unknown"
            confidence = 0.0
            if index < len(handedness_list) and handedness_list[index]:
                classification = handedness_list[index][0]
                label = classification.category_name
                confidence = classification.score

            hands.append(Hand(handedness=label, confidence=confidence, landmarks=landmarks))

        return hands

    def close(self) -> None:
        """Releases MediaPipe resources."""
        self._landmarker.close()


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