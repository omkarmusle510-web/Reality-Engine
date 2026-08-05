"""Global emergency-exit hotkey detection for Reality Engine.

Polls the Windows keyboard state on a background thread for a fixed
hotkey combination (default Ctrl+Shift+Q), independent of which window
currently has focus. ESC (handled in `display.py`) only works while the
OpenCV window is focused; this does not depend on focus at all, which
is the entire point of an "emergency" exit.

This module knows nothing about the engine, the pipeline, mouse APIs,
or windows. It only detects the hotkey and feeds a stop signal into the
pipeline through the same reserved `context["stop_requested"]` key that
`display.py` already uses - so `Engine.start()`'s loop, and the
application's existing shutdown/cleanup sequence in `app.py`, are
identical regardless of whether ESC or this hotkey triggered the stop.
"""

from __future__ import annotations

import ctypes
import threading
import time

from engine.core.logger import get_logger
from engine.core.pipeline import PipelineContext, StageFunc

logger = get_logger(__name__)

_VK_CONTROL = 0x11
_VK_SHIFT = 0x10
_VK_Q = 0x51

_POLL_INTERVAL_SECONDS = 0.05


def _key_is_down(virtual_key_code: int) -> bool:
    """True if the given virtual key is currently held down, system-wide."""
    return bool(ctypes.windll.user32.GetAsyncKeyState(virtual_key_code) & 0x8000)


class EmergencyExit:
    """Detects a global emergency-exit hotkey (default Ctrl+Shift+Q).

    Runs a lightweight background polling thread so the hotkey is caught
    no matter which application currently has keyboard focus - Chrome,
    VS Code, Paint, Explorer, etc. This is deliberately a dumb poller
    rather than a `RegisterHotKey`/message-loop hook, since Reality
    Engine has no window of its own to pump messages for, and polling a
    3-key combo every 50ms is negligible overhead.
    """

    def __init__(self) -> None:
        """Creates and immediately arms the hotkey watcher."""
        self._triggered = threading.Event()
        self._stop_polling = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Emergency exit hotkey armed (Ctrl+Shift+Q).")

    def _poll_loop(self) -> None:
        while not self._stop_polling.is_set():
            if _key_is_down(_VK_CONTROL) and _key_is_down(_VK_SHIFT) and _key_is_down(_VK_Q):
                if not self._triggered.is_set():
                    logger.info("Emergency exit hotkey (Ctrl+Shift+Q) detected.")
                self._triggered.set()
            time.sleep(_POLL_INTERVAL_SECONDS)

    def is_triggered(self) -> bool:
        """True if the emergency-exit hotkey has been pressed since creation."""
        return self._triggered.is_set()

    def close(self) -> None:
        """Stops the background polling thread. Safe to call multiple times."""
        self._stop_polling.set()


def create_emergency_exit_stage(emergency_exit: EmergencyExit) -> StageFunc:
    """Builds a pipeline stage that checks for the global emergency-exit hotkey.

    Always runs (no upstream context key required, mirroring
    `mouse_toggle_stage`/`fps_stage`). If the hotkey has fired, sets
    `context["stop_requested"] = True` - the same reserved key
    `display.py`'s ESC/window-close handling sets - so this shares
    `Engine.start()`'s existing stop path exactly, with no separate
    cleanup logic to maintain.

    Args:
        emergency_exit: An `EmergencyExit` instance, owned by the caller
            so its background thread persists across pipeline executions.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _emergency_exit_stage(context: PipelineContext) -> PipelineContext:
        if emergency_exit.is_triggered():
            context["stop_requested"] = True
        return context

    return _emergency_exit_stage