"""Finger open/closed state detection.

Pure geometry over engine `Hand`/`Landmark` data - no MediaPipe, no
drawing, no external dependencies beyond the standard library.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple

from engine.tracking.hand import Hand, Landmark

_FINGER_JOINTS: Dict[str, Tuple[int, int]] = {
    "thumb": (4, 2),
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}


class FingerState(str, Enum):
    """Whether a single finger is extended or curled."""

    OPEN = "open"
    CLOSED = "closed"


def _squared_distance(a: Landmark, b: Landmark) -> float:
    """Squared Euclidean distance between two landmarks (x/y plane)."""
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def get_finger_states(hand: Hand) -> Dict[str, FingerState]:
    """Determines the open/closed state of each finger.

    A finger is OPEN if its tip is farther from the wrist than its
    reference joint, and CLOSED otherwise. This is orientation-agnostic -
    it works regardless of how the hand is rotated in the image plane.

    Args:
        hand: The hand to analyze.

    Returns:
        A mapping of finger name ("thumb", "index", "middle", "ring",
        "pinky") to its `FingerState`.
    """
    wrist = hand.landmarks[0]
    states: Dict[str, FingerState] = {}

    for finger_name, (tip_index, joint_index) in _FINGER_JOINTS.items():
        tip = hand.landmarks[tip_index]
        joint = hand.landmarks[joint_index]
        is_open = _squared_distance(tip, wrist) > _squared_distance(joint, wrist)
        states[finger_name] = FingerState.OPEN if is_open else FingerState.CLOSED

    return states