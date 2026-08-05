"""Window display and exit-signal detection for the Reality Engine.

Shows the current frame in an OpenCV window and detects window-driven
developer signals: ESC (stop), window close (stop), and M (toggle mouse
control). This is purely a presentation and input-detection concern,
distinct from `overlay.py`, which only draws into the frame buffer and
never touches a window. `overlay.py`'s "no UI" contract is unchanged by
this file's existence.

This module communicates with `Engine` only through the shared
`PipelineContext` - it has no reference to the engine itself, and the
engine has no reference to OpenCV, windows, or keys.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes

import cv2

from engine.core.logger import get_logger
from engine.core.pipeline import PipelineContext, StageFunc
from engine.vision.frame import Frame

logger = get_logger(__name__)

_ESC_KEY = 27
_MOUSE_TOGGLE_KEY_LOWER = ord("m")
_MOUSE_TOGGLE_KEY_UPPER = ord("M")


@dataclass
class DisplaySignal:
    """Developer-facing signals detected while showing a frame this cycle.

    Attributes:
        stop_requested: True if ESC was pressed or the window was closed.
        toggle_mouse_requested: True if the mouse-control toggle key (M)
            was pressed this cycle.
    """

    stop_requested: bool
    toggle_mouse_requested: bool


class DisplayWindow:
    """Owns a single OpenCV display window."""

    def __init__(self, window_name: str = "Reality Engine") -> None:
        """Creates and shows a display window.

        Args:
            window_name: Title of the OpenCV window.
        """
        self._window_name = window_name

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

        # -------------------------------------------------------------
        # Small developer preview window
        # -------------------------------------------------------------
        window_width = 480
        window_height = 270

        cv2.resizeWindow(self._window_name, window_width, window_height)

        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)

        cv2.moveWindow(
            self._window_name,
            screen_width - window_width - 20,
            screen_height - window_height - 80,
        )

    def show(self, frame: Frame) -> DisplaySignal:
        """Displays a frame and checks for developer key/window signals.

        `cv2.waitKey(1)` both pumps the window's event queue (required for
        the window to actually redraw and remain responsive) and yields
        roughly a millisecond of wall-clock time per call. That alone is
        enough pacing to keep this loop from busy-waiting; no separate
        sleep or frame-rate limiter is needed.

        Args:
            frame: The frame to display (already annotated by upstream
                stages, if any).

        Returns:
            A `DisplaySignal` describing what was detected this cycle.
        """
        cv2.imshow(self._window_name, frame.image)

        key = cv2.waitKey(1) & 0xFF

        stop_requested = False
        if key == _ESC_KEY:
            logger.info("ESC pressed - requesting engine stop.")
            stop_requested = True
        elif cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
            logger.info("Display window closed - requesting engine stop.")
            stop_requested = True

        toggle_mouse_requested = key in (
            _MOUSE_TOGGLE_KEY_LOWER,
            _MOUSE_TOGGLE_KEY_UPPER,
        )

        if toggle_mouse_requested:
            logger.info("M pressed - requesting mouse-control toggle.")

        return DisplaySignal(
            stop_requested=stop_requested,
            toggle_mouse_requested=toggle_mouse_requested,
        )

    def close(self) -> None:
        """Destroys all OpenCV windows. Safe to call even if already closed."""
        cv2.destroyAllWindows()


def create_display_stage(window: DisplayWindow) -> StageFunc:
    """Builds a pipeline stage that shows the current frame and detects developer signals.

    Reads `context["frame"]`. If present, displays it and checks for
    ESC/window-close, setting `context["stop_requested"] = True` - a
    reserved key that `Engine.start()`'s loop checks after every pipeline
    execution to decide whether to stop - and checks for the M key,
    setting `context["toggle_mouse_requested"] = True`, a reserved key
    the mouse-toggle stage consumes (and clears) on the next execution.
    No-op if no frame is present.

    Args:
        window: A `DisplayWindow` instance, owned by the caller.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _display_stage(context: PipelineContext) -> PipelineContext:
        frame = context.get("frame")
        if isinstance(frame, Frame):
            signal = window.show(frame)

            if signal.stop_requested:
                context["stop_requested"] = True

            if signal.toggle_mouse_requested:
                context["toggle_mouse_requested"] = True

        return context

    return _display_stage