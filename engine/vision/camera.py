"""OpenCV-backed camera implementation.

Lifecycle only: open, read, release. No preprocessing, flipping, resizing,
or drawing happens here.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2

from engine.core.logger import get_logger
from engine.interfaces.camera import CameraInterface
from engine.vision.frame import Frame

logger = get_logger(__name__)


class Camera(CameraInterface):
    """Captures raw frames from a local camera device via OpenCV."""

    def __init__(self, device_index: int = 0) -> None:
        """Creates a camera bound to the given device index.

        Args:
            device_index: OS-level camera device index (0 is typically the
                default webcam).
        """
        self._device_index = device_index
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Opens the camera device.

        Raises:
            RuntimeError: If the device cannot be opened.
        """
        self._capture = cv2.VideoCapture(self._device_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera device {self._device_index}.")
        logger.info("Camera device %d opened.", self._device_index)

    def read(self) -> Optional[Frame]:
        """Reads a single frame from the camera.

        Returns:
            A `Frame` wrapping the captured image, or `None` if no frame
            was available.

        Raises:
            RuntimeError: If called before `open()`.
        """
        if self._capture is None:
            raise RuntimeError("Camera.read() called before open().")

        success, image = self._capture.read()
        if not success:
            logger.warning("Camera device %d returned no frame.", self._device_index)
            return None

        return Frame(image=image, timestamp=time.monotonic())

    def release(self) -> None:
        """Releases the camera device. Safe to call even if never opened."""
        if self._capture is not None:
            self._capture.release()
            logger.info("Camera device %d released.", self._device_index)
            self._capture = None