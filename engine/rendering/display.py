"""Window display and exit-signal detection for the Reality Engine.

Shows the current frame in an OpenCV window and detects the two
window-driven exit conditions: ESC key and window close. This is purely a
presentation concern, distinct from `overlay.py`, which only draws into
the frame buffer and never touches a window. `overlay.py`'s "no UI"
contract is unchanged by this file's existence.

This module communicates with `Engine` only through the shared
`PipelineContext` - it has no reference to the engine itself, and the
engine has no reference to OpenCV, windows, or keys.
"""

from __future__ import annotations

import cv2

from engine.core.logger import get_logger
from engine.core.pipeline import PipelineContext, StageFunc
from engine.vision.frame import Frame

logger = get_logger(__name__)

_ESC_KEY = 27


class DisplayWindow:
    """Owns a single OpenCV display window."""

    def __init__(self, window_name: str = "Reality Engine") -> None:
        """Creates and shows a display window.

        Args:
            window_name: Title of the OpenCV window.
        """
        self._window_name = window_name
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

    def show(self, frame: Frame) -> bool:
        """Displays a frame and checks whether the user requested an exit.

        `cv2.waitKey(1)` both pumps the window's event queue (required for
        the window to actually redraw and remain responsive) and yields
        roughly a millisecond of wall-clock time per call. That alone is
        enough pacing to keep this loop from busy-waiting; no separate
        sleep or frame-rate limiter is needed.

        Args:
            frame: The frame to display (already annotated by upstream
                stages, if any).

        Returns:
            True if ESC was pressed or the window was closed this cycle.
        """
        cv2.imshow(self._window_name, frame.image)

        key = cv2.waitKey(1) & 0xFF
        if key == _ESC_KEY:
            logger.info("ESC pressed - requesting engine stop.")
            return True

        if cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
            logger.info("Display window closed - requesting engine stop.")
            return True

        return False

    def close(self) -> None:
        """Destroys the OpenCV window. Safe to call even if already closed."""
        cv2.destroyWindow(self._window_name)


def create_display_stage(window: DisplayWindow) -> StageFunc:
    """Builds a pipeline stage that shows the current frame and detects exit signals.

    Reads `context["frame"]`. If present, displays it and checks for ESC
    or window-close; either sets `context["stop_requested"] = True`, a
    reserved key that `Engine.start()`'s loop checks after every pipeline
    execution to decide whether to stop. No-op if no frame is present.

    Args:
        window: A `DisplayWindow` instance, owned by the caller.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _display_stage(context: PipelineContext) -> PipelineContext:
        frame = context.get("frame")
        if isinstance(frame, Frame) and window.show(frame):
            context["stop_requested"] = True
        return context

    return _display_stage