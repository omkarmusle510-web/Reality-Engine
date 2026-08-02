"""Gesture-to-action mapping for the Reality Engine interaction layer.

Converts recognized gestures into high-level `Action`s. Pure decision
logic - never touches mouse, OS, or input APIs. No MediaPipe, no drawing.

Gesture recognition (`gesture_recognizer.py`) is stateless and per-frame:
it classifies a hand's pose in isolation, with no notion of how long that
pose has been held. Several target actions here - a held pinch becoming a
drag, or a pinch ending becoming a release - are inherently about state
*transitions* between frames, not single-frame poses. `ActionMapper` is
therefore intentionally stateful: it remembers the previous frame's
gesture (for the primary/first tracked hand, matching the convention
`cursor_mapper.py` already uses) and uses that transition to distinguish
a fresh click from a sustained drag from a release.
"""

from __future__ import annotations

from typing import Optional

from engine.core.pipeline import PipelineContext, StageFunc
from engine.interaction.action import Action
from engine.interaction.Gesture import Gesture


class ActionMapper:
    """Maps a sequence of per-frame gestures to high-level actions.

    Stateful: keeps track of the previous frame's gesture so that
    sustained gestures (hold -> drag) and transitions (pinch ending ->
    release) can be distinguished from single-frame ones (click).
    """

    def __init__(self) -> None:
        self._previous_gesture: Optional[Gesture] = None

    def map(self, gesture: Optional[Gesture]) -> Action:
        """Maps one frame's gesture for the primary hand to an action.

        Args:
            gesture: The current frame's recognized gesture for the
                primary hand, or `None` if no hand is currently tracked
                (e.g. the hand left the frame).

        Returns:
            The `Action` implied by this gesture, given the previous
            frame's gesture.
        """
        previous_gesture = self._previous_gesture
        self._previous_gesture = gesture

        if gesture == Gesture.PINCH:
            return Action.DRAG if previous_gesture == Gesture.PINCH else Action.LEFT_CLICK

        if previous_gesture == Gesture.PINCH:
            # The pinch ended this frame - including the hand disappearing
            # mid-pinch. Releasing here (rather than staying silent) keeps
            # the OS mouse button from ever getting stuck down.
            return Action.RELEASE

        if gesture == Gesture.FIST:
            return Action.NONE if previous_gesture == Gesture.FIST else Action.RIGHT_CLICK

        return Action.NONE


def create_action_stage(mapper: ActionMapper) -> StageFunc:
    """Builds a pipeline stage that maps the primary hand's gesture to an action.

    Reads `context["gestures"]` and maps the first hand's gesture (the
    same hand `cursor_mapper` treats as primary) to a high-level
    `Action`, stored in `context["action"]`.

    If `context["gestures"]` is an empty list (tracking ran this cycle
    but found no hand), the mapper is still invoked with `None` so that a
    hand disappearing mid-drag safely resolves to `Action.RELEASE`
    instead of leaving the mapper's internal state - and the OS button -
    stuck. If `context["gestures"]` is absent entirely (no tracking data
    this cycle), the stage is a no-op, consistent with how `cursor_stage`
    and `gesture_stage` treat a missing upstream key.

    Args:
        mapper: An `ActionMapper` instance, owned by the caller so its
            state persists across pipeline executions.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _action_stage(context: PipelineContext) -> PipelineContext:
        gestures = context.get("gestures")
        if gestures is None:
            return context

        primary_gesture = gestures[0] if gestures else None
        context["action"] = mapper.map(primary_gesture)
        return context

    return _action_stage