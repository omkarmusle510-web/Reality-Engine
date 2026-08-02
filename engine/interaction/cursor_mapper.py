"""Coordinate mapping and smoothing for the Reality Engine cursor.

Converts the index fingertip landmark into a normalized `Cursor` position
and applies lightweight temporal smoothing. No screen APIs, no input
control, no MediaPipe - pure engine data in, pure engine data out.
"""

from __future__ import annotations

from typing import Optional

from engine.core.pipeline import PipelineContext, StageFunc
from engine.interaction.cursor import Cursor
from engine.tracking.hand import Hand

_INDEX_FINGERTIP = 8


def map_hand_to_cursor(hand: Hand) -> Cursor:
    """Maps a hand's index fingertip to a normalized cursor position.

    Args:
        hand: The hand to read the index fingertip from.

    Returns:
        A `Cursor` at the index fingertip's normalized position.
    """
    fingertip = hand.landmarks[_INDEX_FINGERTIP]
    return Cursor(x=fingertip.x, y=fingertip.y)


class CursorSmoother:
    """Applies exponential moving average smoothing to cursor positions.

    Deterministic and stateful: each call blends the new raw position with
    the previous smoothed position, so small jitter is damped while real
    movement is still tracked frame to frame.
    """

    def __init__(self, smoothing_factor: float = 0.5) -> None:
        """Creates a smoother.

        Args:
            smoothing_factor: Weight given to the new raw position, in
                (0, 1]. Lower values smooth more aggressively (slower to
                react); 1.0 disables smoothing entirely.

        Raises:
            ValueError: If `smoothing_factor` is not in (0, 1].
        """
        if not 0.0 < smoothing_factor <= 1.0:
            raise ValueError(f"smoothing_factor must be in (0, 1], got {smoothing_factor!r}.")
        self._smoothing_factor = smoothing_factor
        self._previous: Optional[Cursor] = None

    def update(self, raw: Cursor) -> Cursor:
        """Smooths a new raw cursor position against the previous one.

        Args:
            raw: The newly computed, unsmoothed cursor position.

        Returns:
            The smoothed `Cursor` position. Also becomes the new baseline
            for the next call.
        """
        if self._previous is None:
            self._previous = raw
            return raw

        alpha = self._smoothing_factor
        smoothed = Cursor(
            x=self._previous.x + alpha * (raw.x - self._previous.x),
            y=self._previous.y + alpha * (raw.y - self._previous.y),
        )
        self._previous = smoothed
        return smoothed


def create_cursor_stage(smoother: CursorSmoother) -> StageFunc:
    """Builds a pipeline stage that computes a smoothed cursor position.

    Reads `context["hands"]` and, if at least one hand is present, maps
    the first hand's index fingertip to a cursor position, smooths it, and
    stores the result in `context["cursor"]`. If no hands are present, the
    stage is a no-op (the previous cursor value, if any, is left as-is).

    Args:
        smoother: A `CursorSmoother` instance, owned by the caller so its
            state persists across pipeline executions.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _cursor_stage(context: PipelineContext) -> PipelineContext:
        hands = context.get("hands")
        if hands:
            raw_cursor = map_hand_to_cursor(hands[0])
            context["cursor"] = smoother.update(raw_cursor)
        return context

    return _cursor_stage