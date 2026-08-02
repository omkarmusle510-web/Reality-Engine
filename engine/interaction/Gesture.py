"""Gesture model for the Reality Engine interaction layer."""

from __future__ import annotations

from enum import Enum


class Gesture(str, Enum):
    """A recognized hand gesture."""

    OPEN_HAND = "open_hand"
    FIST = "fist"
    POINT = "point"
    PINCH = "pinch"
    UNKNOWN = "unknown"