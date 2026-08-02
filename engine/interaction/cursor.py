"""Cursor model for the Reality Engine interaction layer.

Represents where the cursor SHOULD be, in normalized coordinates. This is
not an OS cursor and never touches screen or input APIs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Cursor:
    """A normalized cursor position.

    Attributes:
        x: Horizontal position, 0 (left) to 1 (right).
        y: Vertical position, 0 (top) to 1 (bottom).
    """

    x: float
    y: float