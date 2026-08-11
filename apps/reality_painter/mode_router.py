"""Runtime-mode routing and pipeline-exclusivity gating for Reality Painter.

Reality Engine's `Pipeline` (see `engine.core.pipeline`) runs a single,
fixed list of stages every cycle - it has no concept of "modes." This
module keeps that architecture unchanged: every stage from both the
painting side and the 3D-inspection side is still registered on the
one `Pipeline`, but each is wrapped with `gate()`, a thin no-op-unless-
allowed wrapper (the same pattern `mirror.py`/`mouse_controller.py`
already use for stage-local no-ops). Exactly one group of gated stages
is ever active on a given cycle, decided entirely by
`ModeController.mode` - so only one "pipeline" (painting or
inspection) genuinely executes per engine loop, without a second
`Pipeline` instance and without changing `Engine`/`Pipeline`
themselves.

`create_mode_router_stage` is the only stage that reacts to explicit
user actions (key presses) and drives `ModeController` transitions. It
is registered ungated, first, so mode changes always take effect
before any gated stage runs in the same cycle.
"""

from __future__ import annotations

from typing import Callable, FrozenSet, Optional

from apps.reality_painter.runtime_mode import ModeController, RuntimeMode
from engine.core.logger import get_logger
from engine.core.pipeline import PipelineContext, StageFunc
from engine.vision.frame import Frame

logger = get_logger(__name__)

# Explicit key bindings for the three user-driven transitions this
# module owns. Chosen to avoid every key already bound in
# `apps.reality_painter.sketch` (b/g/e/u/r/c/s/a/q/[/]/1-5) and
# `engine.rendering.display` (ESC/M).
_ANALYZE_KEYS = (ord("n"), ord("N"))
_ENTER_3D_KEYS = (ord("i"), ord("I"))
_EXIT_3D_KEYS = (ord("x"), ord("X"))

#: Modes in which the painting-side stages (vision, tracking, gesture,
#: cursor, action, mouse control, painting, overlay) are allowed to run.
#: Deliberately everything except `INSPECTING_3D`.
PAINTING_MODES: FrozenSet[RuntimeMode] = frozenset(
    {RuntimeMode.PAINTING, RuntimeMode.ANALYZING, RuntimeMode.ASSET_READY}
)

#: Modes in which 3D-inspection stages are allowed to run.
INSPECTION_MODES: FrozenSet[RuntimeMode] = frozenset({RuntimeMode.INSPECTING_3D})

AnalyzeFn = Callable[[PipelineContext], bool]


def gate(stage: StageFunc, mode_controller: ModeController, allowed_modes: FrozenSet[RuntimeMode]) -> StageFunc:
    """Wraps `stage` so it only executes while the current mode is allowed.

    This is what enforces pipeline exclusivity: a stage wrapped with
    `gate(..., PAINTING_MODES)` is a pure no-op (context passed through
    unchanged, no camera read, no tracking, no rendering) whenever
    `mode_controller.mode` is `INSPECTING_3D`, and vice versa for a
    stage wrapped with `gate(..., INSPECTION_MODES)`. Since both groups
    are registered on the same single `Pipeline`, only one group's
    stages ever do real work on a given cycle.

    Args:
        stage: The stage function to gate.
        mode_controller: The shared `ModeController` whose current mode
            decides whether `stage` runs this cycle.
        allowed_modes: Modes in which `stage` is allowed to execute.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _gated_stage(context: PipelineContext) -> PipelineContext:
        if mode_controller.mode not in allowed_modes:
            return context
        return stage(context)

    return _gated_stage


def _freeze_frame(context: PipelineContext) -> None:
    """Replaces `context["frame"]` with a standalone copy of itself.

    This is the entire "freeze" mechanism: it does not introduce a
    second, separate frozen-frame key. Because the vision stage that
    would normally overwrite `context["frame"]` every cycle is gated to
    `PAINTING_MODES` (see `PAINTING_MODES`) and therefore never runs
    while `INSPECTING_3D`, replacing `context["frame"].image` with a
    private `.copy()` at the moment of entering 3D guarantees no
    downstream stage - including a stage that mutates `frame.image` in
    place - can ever affect the live camera frame or vice versa, and
    the same frozen pixels are seen on every subsequent cycle without
    any further camera reads.
    """
    frame = context.get("frame")
    if isinstance(frame, Frame):
        context["frame"] = Frame(image=frame.image.copy(), timestamp=frame.timestamp)


def create_mode_router_stage(mode_controller: ModeController, analyze_fn: Optional[AnalyzeFn] = None) -> StageFunc:
    """Builds the ungated stage that drives explicit runtime-mode transitions.

    Reads `context["key_pressed"]` (the same generic raw-key
    passthrough `apps.reality_painter.sketch` already consumes; no
    collision - see the key bindings above) and reacts only while the
    mode that key is valid for is current:

        - 'N' while PAINTING: requests ANALYZING. If `analyze_fn` was
          supplied, it is invoked synchronously in the same cycle and
          its boolean result decides ASSET_READY (success) or back to
          PAINTING (failure) - recognition/asset-resolution logic
          itself is never implemented here, only wired through this
          optional hook. If `analyze_fn` is `None`, the mode simply
          stays ANALYZING until something else (e.g. a future
          asynchronous caller) calls `analysis_succeeded()` /
          `analysis_failed()`.
        - 'I' while ASSET_READY: freezes the current frame (see
          `_freeze_frame`) and enters INSPECTING_3D. This is the only
          place a 3D entry happens - recognition succeeding never
          triggers it on its own.
        - 'X' while INSPECTING_3D: returns to PAINTING. This is the
          only exit path; it is driven purely by a keyboard key, never
          by any hand-tracking/gesture signal (tracking is gated off
          while INSPECTING_3D anyway - see `PAINTING_MODES`).

    Always writes `context["runtime_mode"]` (the current mode's string
    value) so other stages (e.g. a future HUD) can read it without any
    dependency on this module or `ModeController`.

    Args:
        mode_controller: The shared `ModeController` instance, owned by
            the caller so runtime mode persists across pipeline
            executions.
        analyze_fn: Optional synchronous analysis hook. See above.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _mode_router_stage(context: PipelineContext) -> PipelineContext:
        key_pressed = context.get("key_pressed")
        mode = mode_controller.mode

        if mode == RuntimeMode.PAINTING and key_pressed in _ANALYZE_KEYS:
            mode_controller.request_analyze()
            if analyze_fn is not None:
                try:
                    succeeded = bool(analyze_fn(context))
                except Exception:
                    logger.exception("analyze_fn raised during ANALYZING.")
                    succeeded = False
                if succeeded:
                    mode_controller.analysis_succeeded()
                else:
                    mode_controller.analysis_failed()

        elif mode == RuntimeMode.ASSET_READY and key_pressed in _ENTER_3D_KEYS:
            _freeze_frame(context)
            mode_controller.enter_inspection()

        elif mode == RuntimeMode.INSPECTING_3D and key_pressed in _EXIT_3D_KEYS:
            mode_controller.exit_inspection()

        context["runtime_mode"] = mode_controller.mode.value
        return context

    return _mode_router_stage
