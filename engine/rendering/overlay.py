"""Hand landmark overlay drawing for the Reality Engine rendering layer.

Draws landmarks, their connections, the engine cursor, a short fading
cursor trail, a brief gesture transition banner, and a grouped
developer debug HUD (FPS, tracking confidence, gesture, action, cursor
position, mouse state, hand count) using OpenCV drawing functions only.
No window display, no gesture/action/toggle/cursor/tracking decision
logic - this module only renders values that other stages have already
computed and placed in the pipeline context.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from engine.core.logger import get_logger
from engine.core.pipeline import PipelineContext, StageFunc
from engine.interaction.action import Action
from engine.interaction.cursor import Cursor
from engine.interaction.Gesture import Gesture
from engine.tracking.hand import Hand
from engine.vision.frame import Frame

logger = get_logger(__name__)

_LANDMARK_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky finger
    (0, 17),                                   # palm base
]

# --- HUD panel layout -------------------------------------------------
_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_TITLE_COLOR = (255, 255, 255)
_HUD_LABEL_COLOR = (0, 255, 0)
_HUD_SEPARATOR_COLOR = (90, 90, 90)
_HUD_TRANSITION_COLOR = (0, 255, 255)
_HUD_LINE_HEIGHT_PX = 22
_HUD_ORIGIN_X = 10
_HUD_ORIGIN_Y = 15
_HUD_PANEL_PADDING_PX = 10
_HUD_PANEL_COLOR = (0, 0, 0)
_HUD_PANEL_ALPHA = 0.45
_HUD_PANEL_WIDTH_PX = 250

# --- Cursor visualization ----------------------------------------------
_CURSOR_RADIUS_PX = 8
_CURSOR_CROSSHAIR_LENGTH_PX = 14
_CURSOR_IDLE_COLOR = (0, 255, 255)   # yellow
_CURSOR_DRAG_COLOR = (0, 0, 255)     # red
_DRAGGING_ACTIONS = (Action.LEFT_CLICK, Action.DRAG)

# --- Cursor trail --------------------------------------------------------
_TRAIL_MAX_AGE_SECONDS = 0.35
_TRAIL_MAX_POINTS = 12
_TRAIL_BASE_RADIUS_PX = 6
_TRAIL_BASE_ALPHA = 0.5
_TRAIL_COLOR = (0, 255, 255)  # yellow, same family as the idle cursor

# --- Gesture transition feedback ---------------------------------------
_GESTURE_TRANSITION_DISPLAY_SECONDS = 0.5

# --- Tracking confidence buckets ----------------------------------------
_CONFIDENCE_EXCELLENT_THRESHOLD = 0.90
_CONFIDENCE_GOOD_THRESHOLD = 0.75
_CONFIDENCE_POOR_THRESHOLD = 0.50


def draw_hands(image: np.ndarray, hands: List[Hand]) -> np.ndarray:
    """Draws landmarks and connections for each hand onto the image.

    Args:
        image: BGR image to draw on. Mutated in place.
        hands: Hands to draw.

    Returns:
        The same image, with landmarks and connections drawn on it.
    """
    height, width = image.shape[:2]

    for hand in hands:
        points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in hand.landmarks]

        for start_index, end_index in _LANDMARK_CONNECTIONS:
            cv2.line(image, points[start_index], points[end_index], (0, 255, 0), 2)

        for point in points:
            cv2.circle(image, point, 4, (0, 0, 255), -1)

    return image


def draw_cursor(image: np.ndarray, cursor: Cursor, dragging: bool) -> np.ndarray:
    """Draws the engine cursor (circle + crosshair) at its current position.

    Purely visual - this has no effect on the OS cursor, which is owned
    entirely by `MouseController`. Color communicates state at a glance:
    red while dragging, yellow while idle.

    Args:
        image: BGR image to draw on. Mutated in place.
        cursor: The current smoothed engine cursor position (normalized).
        dragging: True if the current action implies the mouse button is
            currently held down (`Action.LEFT_CLICK` or `Action.DRAG`).

    Returns:
        The same image, with the cursor drawn on it.
    """
    height, width = image.shape[:2]
    center = (int(cursor.x * width), int(cursor.y * height))
    color = _CURSOR_DRAG_COLOR if dragging else _CURSOR_IDLE_COLOR

    cv2.circle(image, center, _CURSOR_RADIUS_PX, color, 2)
    cv2.circle(image, center, 2, color, -1)

    half_length = _CURSOR_CROSSHAIR_LENGTH_PX
    cv2.line(image, (center[0] - half_length, center[1]), (center[0] + half_length, center[1]), color, 1)
    cv2.line(image, (center[0], center[1] - half_length), (center[0], center[1] + half_length), color, 1)

    return image


def _draw_faded_circle(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    """Alpha-blends a small filled circle onto a cropped region of the image.

    Blends only the circle's own bounding box, not the full frame - the
    same region-of-interest approach already used for the HUD panel
    background, so drawing several trail points per frame stays cheap
    regardless of camera resolution.

    Args:
        image: BGR image to draw onto. Mutated in place.
        center: Pixel-space (x, y) center of the circle.
        radius: Circle radius in pixels. No-op if <= 0.
        color: BGR color of the circle.
        alpha: Blend strength in [0, 1]. No-op if <= 0.
    """
    if radius <= 0 or alpha <= 0:
        return

    height, width = image.shape[:2]
    x1 = max(0, center[0] - radius)
    x2 = min(width, center[0] + radius + 1)
    y1 = max(0, center[1] - radius)
    y2 = min(height, center[1] + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return

    roi = image[y1:y2, x1:x2]
    roi_overlay = roi.copy()
    local_center = (center[0] - x1, center[1] - y1)
    cv2.circle(roi_overlay, local_center, radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(roi_overlay, alpha, roi, 1 - alpha, 0, dst=roi)


class _CursorTrailTracker:
    """Tracks recent cursor positions so Overlay can draw a short fading trail.

    Purely a rendering aid, independent of the painted canvas: it does
    not compute cursor positions (that remains `CursorSmoother`'s job)
    and has no effect on anything drawn to the persistent painting
    layer - it only remembers the last few frames' cursor positions and
    how long ago each was recorded, fading and expiring them
    automatically. State lives here, inside the rendering layer, the
    same pattern already used by `_GestureTransitionTracker`.
    """

    def __init__(
        self,
        max_age_seconds: float = _TRAIL_MAX_AGE_SECONDS,
        max_points: int = _TRAIL_MAX_POINTS,
    ) -> None:
        self._max_age_seconds = max_age_seconds
        self._max_points = max_points
        self._points: List[Tuple[Tuple[int, int], float]] = []

    def record(self, cursor: Cursor, width: int, height: int) -> None:
        """Records the current cursor position and prunes expired ones.

        Args:
            cursor: The current smoothed cursor position (normalized).
            width: Frame width in pixels, for normalized-to-pixel conversion.
            height: Frame height in pixels, for normalized-to-pixel conversion.
        """
        now = time.monotonic()
        point = (int(cursor.x * width), int(cursor.y * height))
        self._points.append((point, now))

        cutoff = now - self._max_age_seconds
        self._points = [(p, t) for (p, t) in self._points if t >= cutoff]
        if len(self._points) > self._max_points:
            self._points = self._points[-self._max_points :]

    def draw(self, image: np.ndarray) -> None:
        """Draws every currently-live trail point, fading with age.

        Automatically skips points that have expired since `record()`
        was last called - the trail never becomes permanent and requires
        no separate cleanup step.

        Args:
            image: BGR image to draw onto. Mutated in place.
        """
        if not self._points:
            return

        now = time.monotonic()
        for point, recorded_at in self._points:
            age = now - recorded_at
            if age > self._max_age_seconds:
                continue
            fade = max(0.0, 1.0 - age / self._max_age_seconds)
            radius = max(1, int(_TRAIL_BASE_RADIUS_PX * fade))
            alpha = _TRAIL_BASE_ALPHA * fade
            _draw_faded_circle(image, point, radius, _TRAIL_COLOR, alpha)


class _GestureTransitionTracker:
    """Tracks recent gesture changes so Overlay can show a brief transition banner.

    Purely a rendering aid: it does not classify or recognize gestures
    (that remains `GestureRecognizer`'s job) - it only remembers the
    previous frame's gesture and how long ago it changed, so Overlay can
    display e.g. "POINT -> PINCH" for about half a second and then let it
    disappear on its own. State lives here, inside the rendering layer,
    rather than in `GestureRecognizer` or `ActionMapper`, since no other
    stage needs it.
    """

    def __init__(self, display_seconds: float = _GESTURE_TRANSITION_DISPLAY_SECONDS) -> None:
        self._display_seconds = display_seconds
        self._previous_gesture: Optional[Gesture] = None
        self._transition_text: Optional[str] = None
        self._expires_at: Optional[float] = None

    def update(self, current_gesture: Optional[Gesture]) -> Optional[str]:
        """Records this frame's gesture and returns the active transition text, if any.

        Args:
            current_gesture: This frame's primary-hand gesture, or `None`
                if no hand is currently tracked.

        Returns:
            A "PREVIOUS -> CURRENT" string while within the display
            window of a just-occurred transition, otherwise `None`.
        """
        now = time.monotonic()

        if current_gesture != self._previous_gesture and self._previous_gesture is not None:
            previous_name = self._previous_gesture.name
            current_name = current_gesture.name if current_gesture is not None else "NONE"
            self._transition_text = f"{previous_name} -> {current_name}"
            self._expires_at = now + self._display_seconds

        self._previous_gesture = current_gesture

        if self._expires_at is not None and now >= self._expires_at:
            self._transition_text = None
            self._expires_at = None

        return self._transition_text


class _HUDVisibilityToggle:
    """Tracks a persistent visible/hidden flag for the developer HUD panel.

    Pure on/off state, following the same pattern as
    `engine.interaction.mouse_toggle.MouseToggle`. It never touches
    OpenCV, windows, or keys itself - it only decides, each frame,
    whether `draw_debug_hud` is allowed to draw. Hand landmarks, the
    cursor, and the gesture-transition banner are drawn independently of
    this flag and are unaffected by it.
    """

    def __init__(self, visible: bool = True) -> None:
        """Creates a HUD visibility toggle.

        Args:
            visible: Initial state. The HUD is visible by default.
        """
        self._visible = visible

    def update(self, toggle_requested: bool) -> bool:
        """Applies a pending toggle request (if any) and returns current state.

        Args:
            toggle_requested: True if the HUD-toggle key was pressed
                since this was last checked.

        Returns:
            The current visibility state, after applying the toggle.
        """
        if toggle_requested:
            self._visible = not self._visible
            logger.info("Debug HUD %s.", "shown" if self._visible else "hidden")
        return self._visible


def get_tracking_confidence_label(hands: Optional[List[Hand]]) -> str:
    """Classifies tracking quality from the primary hand's detection confidence.

    Reads `Hand.confidence`, which `HandTracker.detect()` already
    populates from MediaPipe's classification score - no new tracking
    computation happens here, this only buckets an existing value into a
    human-readable label. Falls back to a safe string rather than
    crashing if no hand or no confidence value is available.

    Args:
        hands: The current frame's detected hands, or `None` if the
            tracking stage did not run this cycle.

    Returns:
        One of "EXCELLENT", "GOOD", "POOR", "LOST", or "N/A".
    """
    if not hands:
        return "LOST"

    confidence = getattr(hands[0], "confidence", None)
    if confidence is None:
        return "N/A"

    if confidence >= _CONFIDENCE_EXCELLENT_THRESHOLD:
        return "EXCELLENT"
    if confidence >= _CONFIDENCE_GOOD_THRESHOLD:
        return "GOOD"
    if confidence >= _CONFIDENCE_POOR_THRESHOLD:
        return "POOR"
    return "POOR"


def draw_debug_hud(
    image: np.ndarray,
    fps: float,
    mouse_enabled: bool,
    gesture: Optional[Gesture],
    action: Optional[Action],
    cursor: Optional[Cursor],
    hand_count: int,
    tracking_label: str,
    transition_text: Optional[str],
) -> np.ndarray:
    """Draws a grouped developer debug panel in the upper-left corner.

    Purely a rendering function: it draws whatever values it is given
    and makes no decisions about FPS, mouse-control state, gesture
    recognition, action mapping, cursor position, or tracking quality
    itself. A semi-transparent panel is drawn behind the text so the HUD
    stays readable and visually distinct from hand landmarks and the
    cursor, which are drawn directly onto the camera feed elsewhere.

    Args:
        image: BGR image to draw on. Mutated in place.
        fps: Current smoothed frames-per-second value.
        mouse_enabled: Whether OS mouse control is currently enabled.
        gesture: The primary hand's recognized gesture, or `None`.
        action: The current frame's mapped action, or `None`.
        cursor: The current smoothed cursor position, or `None`.
        hand_count: Number of hands detected this frame.
        tracking_label: Pre-classified tracking confidence label (e.g.
            "EXCELLENT", "GOOD", "POOR", "LOST", "N/A").
        transition_text: An active "PREVIOUS -> CURRENT" gesture
            transition string, or `None` if none is currently active.

    Returns:
        The same image, with the debug panel drawn on it.
    """
    cursor_text = f"({cursor.x:.2f}, {cursor.y:.2f})" if cursor is not None else "NONE"

    body_lines = [
        f"FPS:      {fps:.1f}",
        f"Tracking: {tracking_label}",
        f"Gesture:  {gesture.name if gesture is not None else 'NONE'}",
        f"Action:   {action.name if action is not None else 'NONE'}",
        f"Cursor:   {cursor_text}",
        f"Mouse:    {'ON' if mouse_enabled else 'OFF'}",
        f"Hands:    {hand_count}",
    ]

    # Title + separator + body lines (+ optional transition line).
    total_rows = 2 + len(body_lines) + (1 if transition_text else 0)
    panel_height = _HUD_PANEL_PADDING_PX * 2 + total_rows * _HUD_LINE_HEIGHT_PX
    panel_top_left = (_HUD_ORIGIN_X - _HUD_PANEL_PADDING_PX, _HUD_ORIGIN_Y - _HUD_PANEL_PADDING_PX)
    panel_bottom_right = (
        _HUD_ORIGIN_X + _HUD_PANEL_WIDTH_PX,
        _HUD_ORIGIN_Y - _HUD_PANEL_PADDING_PX + panel_height,
    )

    panel_roi = image[panel_top_left[1]:panel_bottom_right[1], panel_top_left[0]:panel_bottom_right[0]]
    roi_overlay = panel_roi.copy()
    cv2.rectangle(roi_overlay, (0, 0), (roi_overlay.shape[1], roi_overlay.shape[0]), _HUD_PANEL_COLOR, -1)
    cv2.addWeighted(roi_overlay, _HUD_PANEL_ALPHA, panel_roi, 1 - _HUD_PANEL_ALPHA, 0, dst=panel_roi)

    row = 0

    # Title
    row += 1
    title_y = _HUD_ORIGIN_Y + row * _HUD_LINE_HEIGHT_PX
    cv2.putText(image, "Reality Engine", (_HUD_ORIGIN_X, title_y), _HUD_FONT, 0.6, _HUD_TITLE_COLOR, 2)

    # Separator
    separator_y = title_y + (_HUD_LINE_HEIGHT_PX // 2)
    cv2.line(
        image,
        (_HUD_ORIGIN_X, separator_y),
        (_HUD_ORIGIN_X + _HUD_PANEL_WIDTH_PX - 2 * _HUD_PANEL_PADDING_PX, separator_y),
        _HUD_SEPARATOR_COLOR,
        1,
    )
    row += 1

    # Body
    for line in body_lines:
        row += 1
        y = _HUD_ORIGIN_Y + row * _HUD_LINE_HEIGHT_PX
        cv2.putText(image, line, (_HUD_ORIGIN_X, y), _HUD_FONT, 0.55, _HUD_LABEL_COLOR, 2)

    # Gesture transition banner (auto-disappears after ~0.5s)
    if transition_text:
        row += 1
        y = _HUD_ORIGIN_Y + row * _HUD_LINE_HEIGHT_PX
        cv2.putText(image, f"Gesture: {transition_text}", (_HUD_ORIGIN_X, y), _HUD_FONT, 0.55, _HUD_TRANSITION_COLOR, 2)

    return image


def create_overlay_stage() -> StageFunc:
    """Builds a pipeline stage that draws hands, cursor, trail, and the debug HUD.

    Reads `context["frame"]`, `context["hands"]`, `context["fps"]`,
    `context["mouse_enabled"]`, `context["gestures"]`,
    `context["action"]`, `context["cursor"]`, and
    `context["toggle_debug_requested"]`. Hand landmarks are drawn only
    if both a frame and hands are present, matching prior behavior. The
    cursor trail and the engine cursor are drawn whenever a cursor
    position is present, regardless of whether a hand is currently
    detected. The debug HUD (including the gesture-transition banner) is
    drawn only while the internal HUD-visibility toggle is on - hiding
    the HUD never affects hand landmarks, the trail, or the cursor, and
    never touches the pipeline or the camera feed itself.

    Gesture-transition, HUD-visibility, and cursor-trail state are
    tracked internally via closure-owned helpers, matching the existing
    pattern used by `mirror.py` (`last_flipped_frame_id`) and
    `mouse_controller.py` (`previously_enabled`) for stage-local state
    that must persist across pipeline executions but has no reason to be
    owned by the calling application.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """
    transition_tracker = _GestureTransitionTracker()
    hud_toggle = _HUDVisibilityToggle()
    trail_tracker = _CursorTrailTracker()

    def _overlay_stage(context: PipelineContext) -> PipelineContext:
        frame = context.get("frame")
        if not isinstance(frame, Frame):
            return context

        hands = context.get("hands")
        if hands is not None:
            draw_hands(frame.image, hands)

        fps = context.get("fps", 0.0)
        mouse_enabled = context.get("mouse_enabled", True)
        gestures = context.get("gestures")
        primary_gesture = gestures[0] if gestures else None
        action = context.get("action")
        cursor = context.get("cursor")
        hand_count = len(hands) if hands is not None else 0

        if isinstance(cursor, Cursor):
            height, width = frame.image.shape[:2]
            trail_tracker.record(cursor, width, height)
            trail_tracker.draw(frame.image)
            dragging = action in _DRAGGING_ACTIONS
            draw_cursor(frame.image, cursor, dragging)

        # Always updated, regardless of HUD visibility, so a transition
        # that occurs while the HUD is hidden is still timed correctly
        # and appears immediately if the HUD is re-shown mid-transition.
        transition_text = transition_tracker.update(primary_gesture)

        toggle_debug_requested = context.pop("toggle_debug_requested", False)
        hud_visible = hud_toggle.update(toggle_debug_requested)

        if hud_visible:
            tracking_label = get_tracking_confidence_label(hands)
            draw_debug_hud(
                frame.image,
                fps,
                mouse_enabled,
                primary_gesture,
                action,
                cursor,
                hand_count,
                tracking_label,
                transition_text,
            )

        return context

    return _overlay_stage