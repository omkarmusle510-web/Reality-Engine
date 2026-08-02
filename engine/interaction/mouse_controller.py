"""OS mouse control for the Reality Engine interaction layer.

`MouseController` is the ONLY module in Reality Engine allowed to touch
operating system input APIs. No other layer imports ctypes, win32api,
pyautogui, or pynput - every such call is isolated here.

Movement only: never clicks, drags, scrolls, or sends key input. Uses the
Windows User32 API directly via `ctypes` - no third-party automation
library.
"""

from __future__ import annotations

import ctypes

from engine.core.logger import get_logger
from engine.core.pipeline import PipelineContext, StageFunc
from engine.interaction.cursor import Cursor

logger = get_logger(__name__)

_SM_CXSCREEN = 0
_SM_CYSCREEN = 1


class MouseController:
    """Moves the OS cursor to match normalized `Cursor` positions."""

    def __init__(self) -> None:
        """Creates a mouse controller bound to the current screen resolution.

        Screen size is read dynamically via `GetSystemMetrics` - never
        hardcoded.
        """
        self._user32 = ctypes.windll.user32
        self._screen_width = self._user32.GetSystemMetrics(_SM_CXSCREEN)
        self._screen_height = self._user32.GetSystemMetrics(_SM_CYSCREEN)
        logger.info(
            "MouseController bound to screen resolution %dx%d.",
            self._screen_width,
            self._screen_height,
        )

    def move_to(self, cursor: Cursor) -> None:
        """Moves the OS cursor to the given normalized position.

        Args:
            cursor: Normalized cursor position, 0.0-1.0 on each axis.
        """
        pixel_x = int(cursor.x * self._screen_width)
        pixel_y = int(cursor.y * self._screen_height)
        self._user32.SetCursorPos(pixel_x, pixel_y)


def create_mouse_controller_stage(controller: MouseController) -> StageFunc:
    """Builds a pipeline stage that moves the OS cursor to match context["cursor"].

    Reads `context["cursor"]` and moves the OS mouse to that position.
    Returns the context unchanged. No-op if no cursor is present this frame.

    Args:
        controller: A `MouseController` instance, owned by the caller.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _mouse_controller_stage(context: PipelineContext) -> PipelineContext:
        cursor = context.get("cursor")
        if isinstance(cursor, Cursor):
            controller.move_to(cursor)
        return context

    return _mouse_controller_stage