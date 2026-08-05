"""Horizontal frame mirroring for the Reality Engine vision layer.

A separate stage from `vision/pipeline.py`'s frame-capture stage: `Camera`
and the capture stage are lifecycle/capture only and never touch pixel
data (see `camera.py`). Mirroring is a distinct, optional transform that
runs immediately after capture and before tracking, so every downstream
stage - tracking, gesture, cursor - already sees a mirrored frame and
never needs to know mirroring happened at all.
"""

from __future__ import annotations

import cv2

from engine.core.pipeline import PipelineContext, StageFunc
from engine.vision.frame import Frame


def create_mirror_stage() -> StageFunc:
    """Builds a pipeline stage that flips the current frame horizontally.

    Reads and mutates `context["frame"]` in place so the camera feed (and
    therefore hand tracking and cursor mapping) behaves like a mirror -
    moving a hand right on screen moves it right in the image.

    `Camera.read()` can return `None`, in which case the vision stage
    leaves `context["frame"]` pointing at the same `Frame` object as the
    previous cycle rather than a new one. This stage tracks the identity
    of the last frame it flipped and skips re-flipping that same object,
    so a stale frame is mirrored exactly once - not flipped back and
    forth on every cycle the camera fails to produce a fresh frame.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """
    last_flipped_frame_id = {"id": None}

    def _mirror_stage(context: PipelineContext) -> PipelineContext:
        frame = context.get("frame")
        if isinstance(frame, Frame) and id(frame) != last_flipped_frame_id["id"]:
            frame.image = cv2.flip(frame.image, 1)
            last_flipped_frame_id["id"] = id(frame)
        return context

    return _mirror_stage