"""OS mouse control for the Reality Engine interaction layer.

`MouseController` is the ONLY module in Reality Engine allowed to touch
operating system input APIs. No other layer imports ctypes, win32api,
pyautogui, or pynput - every such call is isolated here.

Two responsibilities, both OS-level mouse control: moving the cursor to
match a normalized `Cursor` position, and executing a high-level `Action`
(click, drag, release) decided upstream by `ActionMapper`. Uses the
Windows User32 API directly via `ctypes` - no third-party automation
library. `MouseController` only executes actions it is given; it never
decides what action should happen - that decision is `ActionMapper`'s.
"""

from __future__ import annotations

import ctypes

from engine.core.logger import get_logger
from engine.core.pipeline import PipelineContext, StageFunc
from engine.interaction.action import Action
from engine.interaction.cursor import Cursor

logger = get_logger(__name__)

_SM_CXSCREEN = 0
_SM_CYSCREEN = 1

_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010


class MouseController:
    """Moves the OS cursor and executes high-level mouse actions."""

    def __init__(self) -> None:
        """Creates a mouse controller bound to the current screen resolution.

        Screen size is read dynamically via `GetSystemMetrics` - never
        hardcoded.
        """
        self._user32 = ctypes.windll.user32
        self._screen_width = self._user32.GetSystemMetrics(_SM_CXSCREEN)
        self._screen_height = self._user32.GetSystemMetrics(_SM_CYSCREEN)
        self._left_button_down = False
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

    def execute_action(self, action: Action) -> None:
        """Executes a high-level mouse action via the Windows User32 API.

        `Action.DRAG` deliberately issues no OS event of its own: the
        left button is already held down from the `Action.LEFT_CLICK`
        that started the drag, and `move_to` is called every frame
        regardless of action, so the cursor keeps tracking the hand while
        the button stays down - that combination *is* the drag.

        Args:
            action: The action to execute. `Action.NONE` is a no-op.
        """
        if action == Action.LEFT_CLICK:
            self._user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self._left_button_down = True
            logger.debug("Left mouse button pressed.")
        elif action == Action.DRAG:
            pass
        elif action == Action.RELEASE:
            self._user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self._left_button_down = False
            logger.debug("Left mouse button released.")
        elif action == Action.RIGHT_CLICK:
            self._user32.mouse_event(_MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            self._user32.mouse_event(_MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            logger.debug("Right mouse button clicked.")
        # Action.NONE: no OS event.

    def release(self) -> None:
        """Force-releases the left mouse button if currently held down.

        Safe to call unconditionally, including when no button is held.
        Intended for shutdown, so exiting mid-drag never leaves the OS
        mouse button stuck down.
        """
        if self._left_button_down:
            self._user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self._left_button_down = False
            logger.info("Left mouse button forcibly released on cleanup.")


def create_mouse_controller_stage(controller: MouseController) -> StageFunc:
    """Builds a pipeline stage that drives the OS mouse from context state.

    Reads `context["cursor"]` and moves the OS mouse to that position.
    Reads `context["action"]` and executes it via the Windows API. Either
    is a no-op if its context key is absent this frame. Returns the
    context unchanged.

    Args:
        controller: A `MouseController` instance, owned by the caller.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _mouse_controller_stage(context: PipelineContext) -> PipelineContext:
        cursor = context.get("cursor")
        if isinstance(cursor, Cursor):
            controller.move_to(cursor)

        action = context.get("action")
        if isinstance(action, Action):
            controller.execute_action(action)

        return context

    return _mouse_controller_stage