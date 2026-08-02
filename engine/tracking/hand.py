"""Hand model for the Reality Engine tracking layer.

Contains only plain engine data - never a MediaPipe object of any kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Landmark:
    """A single hand landmark position, normalized to [0, 1] image coordinates.

    Attributes:
        x: Horizontal position, 0 (left) to 1 (right).
        y: Vertical position, 0 (top) to 1 (bottom).
        z: Relative depth (smaller is closer to the camera).
    """

    x: float
    y: float
    z: float


@dataclass
class Hand:
    """A single detected hand.

    Attributes:
        handedness: "Left" or "Right".
        confidence: Detection confidence in [0, 1].
        landmarks: Exactly 21 hand landmarks.
    """

    handedness: str
    confidence: float
    landmarks: List[Landmark]