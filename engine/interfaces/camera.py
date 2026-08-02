"""Camera contract used by the vision layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from engine.vision.frame import Frame


class CameraInterface(ABC):
    """Minimal contract for a camera source."""

    @abstractmethod
    def open(self) -> None:
        """Opens the camera device."""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Reads a single frame.

        Returns:
            A `Frame`, or `None` if no frame was available.
        """

    @abstractmethod
    def release(self) -> None:
        """Releases the camera device."""