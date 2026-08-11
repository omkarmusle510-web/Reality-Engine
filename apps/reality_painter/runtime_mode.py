"""Runtime mode state machine for Reality Painter.

Reality Painter has exactly two mutually exclusive runtime modes:
`PAINTING` (webcam, hand tracking, drawing, AI/vision, the painting
pipeline) and `INSPECTING_3D` (frozen frame, 3D renderer, model
inspection - painting, hand tracking, and continuous camera reads all
inactive). `ANALYZING` and `ASSET_READY` are the explicit, user-driven
states between them:

    PAINTING -> ANALYZING -> ASSET_READY -> INSPECTING_3D -> PAINTING

This module owns only the state machine itself - which transitions are
legal and the current mode. It performs no camera access, no
rendering, no key handling, and no recognition/asset logic; see
`apps.reality_painter.mode_router` for how pipeline stages react to
mode changes.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set

from engine.core.logger import get_logger

logger = get_logger(__name__)


class RuntimeMode(str, Enum):
    """A Reality Painter runtime mode."""

    PAINTING = "painting"
    ANALYZING = "analyzing"
    ASSET_READY = "asset_ready"
    INSPECTING_3D = "inspecting_3d"


# The only legal transitions. Anything not listed here (including
# self-transitions) is rejected by `transition_to`.
_LEGAL_TRANSITIONS: Dict[RuntimeMode, Set[RuntimeMode]] = {
    RuntimeMode.PAINTING: {RuntimeMode.ANALYZING},
    RuntimeMode.ANALYZING: {RuntimeMode.ASSET_READY, RuntimeMode.PAINTING},
    RuntimeMode.ASSET_READY: {RuntimeMode.INSPECTING_3D, RuntimeMode.PAINTING},
    RuntimeMode.INSPECTING_3D: {RuntimeMode.PAINTING},
}


class ModeController:
    """Tracks the current `RuntimeMode` and enforces legal transitions.

    Never raises on an illegal transition request - `transition_to`
    (and every convenience method built on it) returns `False` and
    leaves the current mode unchanged, so pipeline stages can call it
    unconditionally (e.g. on every key press) without needing a
    try/except.
    """

    def __init__(self, initial: RuntimeMode = RuntimeMode.PAINTING) -> None:
        self._mode = initial

    @property
    def mode(self) -> RuntimeMode:
        """The current runtime mode."""
        return self._mode

    def can_transition(self, target: RuntimeMode) -> bool:
        """True if moving from the current mode to `target` is legal."""
        return target in _LEGAL_TRANSITIONS.get(self._mode, set())

    def transition_to(self, target: RuntimeMode) -> bool:
        """Attempts to move to `target`.

        Args:
            target: The mode to transition to.

        Returns:
            True if the transition was legal and applied, False if it
            was rejected (current mode is unchanged).
        """
        if not self.can_transition(target):
            logger.warning("Rejected illegal mode transition: %s -> %s.", self._mode.value, target.value)
            return False

        logger.info("Mode transition: %s -> %s.", self._mode.value, target.value)
        self._mode = target
        return True

    # --- Convenience methods (one per explicit user action) -------------

    def request_analyze(self) -> bool:
        """PAINTING -> ANALYZING. Requires an explicit user action to call."""
        return self.transition_to(RuntimeMode.ANALYZING)

    def analysis_succeeded(self) -> bool:
        """ANALYZING -> ASSET_READY."""
        return self.transition_to(RuntimeMode.ASSET_READY)

    def analysis_failed(self) -> bool:
        """ANALYZING -> PAINTING."""
        return self.transition_to(RuntimeMode.PAINTING)

    def enter_inspection(self) -> bool:
        """ASSET_READY -> INSPECTING_3D. Requires an explicit 3D action to call."""
        return self.transition_to(RuntimeMode.INSPECTING_3D)

    def exit_inspection(self) -> bool:
        """INSPECTING_3D -> PAINTING. Requires an explicit exit key to call."""
        return self.transition_to(RuntimeMode.PAINTING)
