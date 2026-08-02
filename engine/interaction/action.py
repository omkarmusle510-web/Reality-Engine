"""Action model for the Reality Engine interaction layer.

Represents a high-level, OS-agnostic user intent derived from gestures
(e.g. "the user wants to left-click"). This is pure data - no mouse, OS,
or input APIs are touched here. See `mouse_controller.py` for execution.
"""

from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    """A high-level action implied by a gesture (or gesture transition)."""

    NONE = "none"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    RELEASE = "release"