"""Gesture recognition for the Reality Engine interaction layer.

Pure engine logic - operates only on `Hand`, `Landmark`, and `FingerState`.
No MediaPipe, no drawing, no cursor control, no input control of any kind.
"""

from __future__ import annotations

from typing import List

from engine.core.pipeline import PipelineContext, StageFunc
from engine.interaction.finger_state import FingerState, get_finger_states
from engine.interaction.Gesture import Gesture
from engine.tracking.hand import Hand

_PINCH_DISTANCE_THRESHOLD = 0.05


def _is_pinching(hand: Hand) -> bool:
    """True if the thumb tip and index tip are close enough to count as a pinch."""
    thumb_tip = hand.landmarks[4]
    index_tip = hand.landmarks[8]
    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    return distance < _PINCH_DISTANCE_THRESHOLD


def recognize_gesture(hand: Hand) -> Gesture:
    """Classifies a single hand's gesture from its landmarks.

    Args:
        hand: The hand to classify.

    Returns:
        The recognized `Gesture`.
    """
    if _is_pinching(hand):
        return Gesture.PINCH

    states = get_finger_states(hand)

    if all(state == FingerState.CLOSED for state in states.values()):
        return Gesture.FIST

    if all(state == FingerState.OPEN for state in states.values()):
        return Gesture.OPEN_HAND

    if (
        states["index"] == FingerState.OPEN
        and states["thumb"] == FingerState.CLOSED
        and states["middle"] == FingerState.CLOSED
        and states["ring"] == FingerState.CLOSED
        and states["pinky"] == FingerState.CLOSED
    ):
        return Gesture.POINT

    return Gesture.UNKNOWN


def create_gesture_stage() -> StageFunc:
    """Builds a pipeline stage that recognizes a gesture for each detected hand.

    Reads `context["hands"]` and stores recognized gestures in
    `context["gestures"]`, in the same order as the hands list. If no
    hands are present, the stage is a no-op.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _gesture_stage(context: PipelineContext) -> PipelineContext:
        hands = context.get("hands")
        if hands is not None:
            context["gestures"] = [recognize_gesture(hand) for hand in hands]
        return context

    return _gesture_stage