"""Runtime mouse-control toggle for the Reality Engine interaction layer.

Tracks whether OS mouse control is currently enabled, in response to a
one-shot toggle request detected by the display layer (the M key). This
is pure on/off state - it never touches Win32 APIs itself; it only
decides, each frame, whether `MouseController` is allowed to.
"""

from __future__ import annotations

from engine.core.logger import get_logger
from engine.core.pipeline import PipelineContext, StageFunc

logger = get_logger(__name__)


class MouseToggle:
    """Tracks a persistent enabled/disabled flag for OS mouse control."""

    def __init__(self, enabled: bool = True) -> None:
        """Creates a mouse toggle.

        Args:
            enabled: Initial state. Mouse control is enabled by default.
        """
        self._enabled = enabled

    def update(self, toggle_requested: bool) -> bool:
        """Applies a pending toggle request (if any) and returns current state.

        Args:
            toggle_requested: True if the toggle key was pressed since
                this was last checked.

        Returns:
            The current enabled state, after applying the toggle.
        """
        if toggle_requested:
            self._enabled = not self._enabled
            logger.info("Mouse control %s.", "enabled" if self._enabled else "disabled")
        return self._enabled


def create_mouse_toggle_stage(toggle: MouseToggle) -> StageFunc:
    """Builds a pipeline stage that applies pending mouse-toggle requests.

    Reads and clears `context["toggle_mouse_requested"]` (set by the
    display stage when M was pressed) and writes the resulting state to
    `context["mouse_enabled"]` for the mouse-controller stage to respect.
    Always runs; treats a missing key as "no toggle requested".

    Args:
        toggle: A `MouseToggle` instance, owned by the caller so its
            state persists across pipeline executions.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _mouse_toggle_stage(context: PipelineContext) -> PipelineContext:
        toggle_requested = context.pop("toggle_mouse_requested", False)
        context["mouse_enabled"] = toggle.update(toggle_requested)
        return context

    return _mouse_toggle_stage