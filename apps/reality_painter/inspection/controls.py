"""3D inspection view controls for Reality Painter.

Reads the generic `key_pressed` passthrough already provided by
`engine.rendering.display.DisplayWindow` and mutates the currently
inspected `SceneObject`'s transform: rotation (yaw/pitch), zoom
(scale), and reset back to its normalized base transform (see
`apps.reality_painter.inspection.framing.normalize_transform`).

`engine.rendering.display.DisplayWindow.show()` masks every key to a
single byte (`cv2.waitKey(1) & 0xFF`), which drops the extended codes
OpenCV reports for arrow keys on most platforms. To stay reliable
without touching that engine module, rotation/zoom use plain
single-byte letter keys instead of arrow keys - conceptually mapped as
Left/Right -> A/D and Up/Down -> W/S. Exiting back to painting reuses
the existing 'X' key already owned by `apps.reality_painter.mode_router`.

This module never touches recognition, retrieval, optimization, or the
network - it only mutates a `Transform` already attached to a
`SceneObject` already present in the `Scene`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

from engine.core.pipeline import PipelineContext, StageFunc
from engine.interaction.cursor import Cursor
from engine.interaction.Gesture import Gesture
from engine.scene.objects import SceneObject, Transform

_ROTATE_LEFT_KEYS = (ord("a"), ord("A"))
_ROTATE_RIGHT_KEYS = (ord("d"), ord("D"))
_ROTATE_UP_KEYS = (ord("w"), ord("W"))
_ROTATE_DOWN_KEYS = (ord("s"), ord("S"))
_ZOOM_IN_KEYS = (ord("+"), ord("="))
_ZOOM_OUT_KEYS = (ord("-"), ord("_"))
_RESET_KEYS = (ord("r"), ord("R"))

_YAW_STEP_RADIANS = 0.1
_PITCH_STEP_RADIANS = 0.1
_ZOOM_STEP_FACTOR = 1.1
_MIN_ZOOM_FACTOR = 0.1
_MAX_ZOOM_FACTOR = 10.0

# --- Lightweight hand interaction (rotate/tilt via movement, zoom via pinch) ---
# Deliberately simple: reuses whatever normalized cursor/gesture the
# existing tracking/gesture/cursor stages already compute (see
# apps/reality_painter/app.py's INSPECTION_MODES-gated stages) - no new
# gesture framework, no extra tracking pass, no heavy dependency. A
# tracked hand's frame-to-frame movement drives yaw/pitch continuously
# (like a natural "grab and turn" feel); a PINCH gesture zooms in while
# held. Small multipliers keep this responsive without large jumps.
_HAND_ROTATE_SENSITIVITY = 4.0
_HAND_PINCH_ZOOM_STEP = 1.01


class InspectionViewState:
    """Tracks per-object rotation/zoom offsets applied on top of a base transform.

    `base_transform` is the object's normalized transform (see
    `apps.reality_painter.inspection.framing.normalize_transform`) -
    never mutated in place. Rotation/zoom are tracked as independent
    offsets so `reset()` always returns to exactly that base,
    regardless of how many control inputs happened since.
    """

    def __init__(self) -> None:
        self._base_transform: Optional[Transform] = None
        self._yaw = 0.0
        self._pitch = 0.0
        self._zoom = 1.0
        self._last_hand_cursor: Optional[Cursor] = None

    def set_base(self, transform: Transform) -> None:
        """Sets the base (normalized) transform for a newly inspected object.

        Also clears any rotation/zoom offsets left over from a
        previously inspected object.
        """
        self._base_transform = transform
        self._yaw = 0.0
        self._pitch = 0.0
        self._zoom = 1.0

    def reset(self) -> None:
        """Clears rotation/zoom offsets back to the base transform."""
        self._yaw = 0.0
        self._pitch = 0.0
        self._zoom = 1.0

    def apply_key(self, key_pressed: Optional[int]) -> None:
        """Applies one key press's effect on rotation/zoom state, if any."""
        if key_pressed in _ROTATE_LEFT_KEYS:
            self._yaw -= _YAW_STEP_RADIANS
        elif key_pressed in _ROTATE_RIGHT_KEYS:
            self._yaw += _YAW_STEP_RADIANS
        elif key_pressed in _ROTATE_UP_KEYS:
            self._pitch -= _PITCH_STEP_RADIANS
        elif key_pressed in _ROTATE_DOWN_KEYS:
            self._pitch += _PITCH_STEP_RADIANS
        elif key_pressed in _ZOOM_IN_KEYS:
            self._zoom = min(_MAX_ZOOM_FACTOR, self._zoom * _ZOOM_STEP_FACTOR)
        elif key_pressed in _ZOOM_OUT_KEYS:
            self._zoom = max(_MIN_ZOOM_FACTOR, self._zoom / _ZOOM_STEP_FACTOR)
        elif key_pressed in _RESET_KEYS:
            self.reset()

    def apply_hand_tracking(self, cursor: Optional[Cursor], gesture: Optional[Gesture]) -> None:
        """Applies lightweight hand-driven rotation/zoom for this cycle.

        Natural hand movement (frame-to-frame delta of the normalized
        index-fingertip cursor, the same value the painting pipeline
        already computes) drives yaw/pitch continuously while a hand is
        tracked. A `Gesture.PINCH` drives a simple continuous zoom-in
        while held - deliberately not a full pinch-distance gesture
        framework, just reusing the gesture the engine already
        recognizes. A no-op whenever no hand is currently tracked
        (`cursor` is `None`), and the movement baseline resets so the
        hand reappearing doesn't jump the model from a stale delta.

        Args:
            cursor: This cycle's normalized cursor position (index
                fingertip), or `None` if no hand is tracked.
            gesture: This cycle's primary-hand gesture, or `None`.
        """
        if cursor is None:
            self._last_hand_cursor = None
            return

        if self._last_hand_cursor is not None:
            self._yaw += (cursor.x - self._last_hand_cursor.x) * _HAND_ROTATE_SENSITIVITY
            self._pitch += (cursor.y - self._last_hand_cursor.y) * _HAND_ROTATE_SENSITIVITY
        self._last_hand_cursor = cursor

        if gesture == Gesture.PINCH:
            self._zoom = min(_MAX_ZOOM_FACTOR, self._zoom * _HAND_PINCH_ZOOM_STEP)

    def resolve_transform(self) -> Optional[Transform]:
        """Computes the current transform (base transform + rotation/zoom offsets).

        Returns:
            `None` if no base transform has been set yet (nothing is
            currently being inspected).
        """
        if self._base_transform is None:
            return None
        base = self._base_transform
        rotation = (base.rotation[0] + self._pitch, base.rotation[1] + self._yaw, base.rotation[2])
        scale = tuple(component * self._zoom for component in base.scale)
        return replace(base, rotation=rotation, scale=scale)


def create_inspection_controls_stage(
    view_state: InspectionViewState, get_active_object: Callable[[], Optional[SceneObject]]
) -> StageFunc:
    """Builds a pipeline stage that applies rotate/zoom/reset controls.

    Reads `context["key_pressed"]`. A no-op if no object is currently
    active (`get_active_object()` returns `None`) or no key was
    pressed this cycle. Mutates the active `SceneObject`'s `.transform`
    in place - it never re-runs recognition, retrieval, optimization,
    or GLB loading, and never touches the network.

    Args:
        view_state: The `InspectionViewState` tracking rotation/zoom
            offsets, owned by the caller so it persists across
            pipeline executions.
        get_active_object: Returns the currently inspected
            `SceneObject`, or `None` if nothing is active. Injected so
            this stage never needs to own scene/object lifecycle
            itself.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _inspection_controls_stage(context: PipelineContext) -> PipelineContext:
        active_object = get_active_object()
        if active_object is None:
            return context

        key_pressed = context.get("key_pressed")
        if key_pressed is not None:
            view_state.apply_key(key_pressed)

        cursor = context.get("cursor")
        gestures = context.get("gestures")
        primary_gesture = gestures[0] if gestures else None
        view_state.apply_hand_tracking(cursor if isinstance(cursor, Cursor) else None, primary_gesture)

        transform = view_state.resolve_transform()
        if transform is not None:
            active_object.transform = transform

        return context

    return _inspection_controls_stage
