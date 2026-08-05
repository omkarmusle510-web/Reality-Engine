"""Frame-rate measurement for the Reality Engine core layer.

Pure timing utility, reusable across any application built on the
engine. Computes a smoothed frames-per-second value from wall-clock time
between pipeline executions - no rendering, no OS, no vision/tracking
concerns.
"""

from __future__ import annotations

import time
from typing import Optional

from engine.core.pipeline import PipelineContext, StageFunc


class FPSCounter:
    """Tracks a smoothed frames-per-second value across pipeline executions."""

    def __init__(self, smoothing_factor: float = 0.9) -> None:
        """Creates an FPS counter.

        Args:
            smoothing_factor: Weight given to the running average on each
                tick, in [0, 1). Higher values smooth more aggressively
                (slower to react to sudden frame-rate changes).

        Raises:
            ValueError: If `smoothing_factor` is not in [0, 1).
        """
        if not 0.0 <= smoothing_factor < 1.0:
            raise ValueError(f"smoothing_factor must be in [0, 1), got {smoothing_factor!r}.")
        self._smoothing_factor = smoothing_factor
        self._previous_time: Optional[float] = None
        self._fps = 0.0

    def tick(self) -> float:
        """Records one frame and returns the current smoothed FPS.

        Returns:
            The smoothed FPS value. `0.0` on the very first call, since
            no interval exists yet to measure.
        """
        now = time.monotonic()
        if self._previous_time is not None:
            delta = now - self._previous_time
            if delta > 0:
                instantaneous_fps = 1.0 / delta
                alpha = self._smoothing_factor
                self._fps = (alpha * self._fps) + ((1.0 - alpha) * instantaneous_fps)
        self._previous_time = now
        return self._fps


def create_fps_stage(counter: FPSCounter) -> StageFunc:
    """Builds a pipeline stage that measures FPS once per pipeline execution.

    Always runs (no upstream context key required) and writes the
    current smoothed FPS to `context["fps"]`.

    Args:
        counter: An `FPSCounter` instance, owned by the caller so its
            state persists across pipeline executions.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _fps_stage(context: PipelineContext) -> PipelineContext:
        context["fps"] = counter.tick()
        return context

    return _fps_stage