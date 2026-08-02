"""Frame model for the Reality Engine vision layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """A single captured camera frame.

    Attributes:
        image: Raw image data as captured by the camera (BGR, uint8).
        timestamp: Capture time in seconds, from `time.monotonic()`.
    """

    image: np.ndarray
    timestamp: float