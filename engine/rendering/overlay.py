"""Hand landmark overlay drawing for the Reality Engine rendering layer.

Draws landmarks, their connections, the engine cursor, a brush preview,
a short fading cursor trail, a brief gesture transition banner, and a
grouped developer debug HUD (FPS, tracking confidence, gesture, action,
cursor position, mouse state, hand count) using OpenCV drawing functions
only. No window display, no gesture/action/toggle/cursor/tracking
decision logic - this module only renders values that other stages have
already computed and placed in the pipeline context.

Phase 10 splits the HUD into two independent layers:
    - A "user HUD" (current tool: brush/shape type, size, color, eraser
      state) that stays on by default while drawing, since that is the
      information relevant to painting itself.
    - A "developer HUD" (FPS, tracking confidence, gesture, action,
      cursor coordinates, hand count) that remains optional/toggleable,
      exactly as before.

This module intentionally has NO dependency on any `apps.*` package
(e.g. Reality Painter's brush/shape/menu modules) - Reality Engine must
stay reusable by any application built on it. Any application-specific
value this module renders (tool name, brush type, shape mode, ...) is
read from the shared `PipelineContext` as a plain, optionally-present
value, never imported directly. In particular, a radial menu (or any
other app-specific overlay UI) is expected to be drawn by the owning
application's own pipeline stage before `overlay` runs - not by this
module - so Reality Engine never needs to know such a thing exists.
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

# --- User HUD (top-right, drawing-relevant info only) ---------------------
_USER_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_USER_HUD_LINE_HEIGHT_PX = 22
_USER_HUD_PANEL_PADDING_PX = 10
_USER_HUD_PANEL_WIDTH_PX = 210
_USER_HUD_PANEL_COLOR = (0, 0, 0)
_USER_HUD_PANEL_ALPHA = 0.45
_USER_HUD_LABEL_COLOR = (255, 255, 255)
_USER_HUD_MARGIN_PX = 10
_USER_HUD_SWATCH_SIZE_PX = 14

# --- Cursor visualization ----------------------------------------------
_CURSOR_RADIUS_PX = 8
_CURSOR_CROSSHAIR_LENGTH_PX = 14
_CURSOR_IDLE_COLOR = (0, 255, 255)   # yellow
_CURSOR_DRAG_COLOR = (0, 0, 255)     # red
_CURSOR_OUTER_RING_RADIUS_PX = 16
_CURSOR_CENTER_DOT_RADIUS_PX = 2
_CURSOR_PULSE_PERIOD_SECONDS = 0.6
_CURSOR_PULSE_RING_MIN_ALPHA = 0.15
_CURSOR_PULSE_RING_MAX_ALPHA = 0.55
_DRAGGING_ACTIONS = (Action.LEFT_CLICK, Action.DRAG)

# --- Brush preview ---------------------------------------------------------
# Renders already-decided values from context (brush_size, brush_color,
# eraser_active, and the optional brush_type_name/shape_type - see the
# module docstring) - Overlay makes no decisions about brush size,
# color, tool, or shape itself, the same way it already only visualizes
# cursor/action without computing them.
_ERASER_PREVIEW_COLOR = (255, 255, 255)  # white ring while erasing
_BRUSH_PREVIEW_SMOOTHING = 0.35  # eases the preview ring radius toward brush_size
_BRUSH_PREVIEW_LABEL_COLOR = (255, 255, 255)
_BRUSH_PREVIEW_LABEL_OFFSET_PX = 10

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


def draw_cursor(image: np.ndarray, cursor: Cursor, dragging: bool, pulse_phase: float = 0.0) -> np.ndarray:
    """Draws the engine cursor (outer ring, crosshair, and center dot).

    Purely visual - this has no effect on the OS cursor, which is owned
    entirely by `MouseController`. Color communicates state at a glance:
    red while dragging, yellow while idle. A soft outer ring pulses
    gently while dragging, giving lightweight, continuous feedback that
    a stroke/shape/selection is actively being held without needing any
    extra state beyond `dragging` and a time-based phase.

    Args:
        image: BGR image to draw on. Mutated in place.
        cursor: The current smoothed engine cursor position (normalized).
        dragging: True if the current action implies the mouse button is
            currently held down (`Action.LEFT_CLICK` or `Action.DRAG`).
        pulse_phase: A value in [0, 1) driving the outer ring's pulse
            animation while dragging. Time-based, not frame-count-based,
            so pulse speed is independent of frame rate.

    Returns:
        The same image, with the cursor drawn on it.
    """
    height, width = image.shape[:2]
    center = (int(cursor.x * width), int(cursor.y * height))
    color = _CURSOR_DRAG_COLOR if dragging else _CURSOR_IDLE_COLOR

    if dragging:
        pulse = 0.5 - 0.5 * np.cos(2 * np.pi * pulse_phase)  # smooth 0->1->0
        alpha = _CURSOR_PULSE_RING_MIN_ALPHA + pulse * (_CURSOR_PULSE_RING_MAX_ALPHA - _CURSOR_PULSE_RING_MIN_ALPHA)
        _draw_faded_circle(image, center, _CURSOR_OUTER_RING_RADIUS_PX, color, alpha)

    cv2.circle(image, center, _CURSOR_RADIUS_PX, color, 2, cv2.LINE_AA)
    cv2.circle(image, center, _CURSOR_CENTER_DOT_RADIUS_PX, color, -1, cv2.LINE_AA)

    half_length = _CURSOR_CROSSHAIR_LENGTH_PX
    cv2.line(image, (center[0] - half_length, center[1]), (center[0] + half_length, center[1]), color, 1, cv2.LINE_AA)
    cv2.line(image, (center[0], center[1] - half_length), (center[0], center[1] + half_length), color, 1, cv2.LINE_AA)

    return image


class _BrushPreviewSmoother:
    """Eases the rendered brush-preview radius toward the target brush size.

    Purely a rendering aid - mirrors the same EMA smoothing pattern
    `apps.reality_painter.sketch.ToolState` already applies to the brush
    size *value*, but applied here to the *drawn* radius, so a brush
    size change (e.g. from `[`/`]` or the radial menu) eases visually
    instead of snapping, without Overlay needing to know how or why the
    size changed.
    """

    def __init__(self, smoothing_factor: float = _BRUSH_PREVIEW_SMOOTHING) -> None:
        self._smoothing_factor = smoothing_factor
        self._current_radius: Optional[float] = None

    def update(self, target_radius: float) -> float:
        """Advances the smoothed radius one step toward `target_radius`."""
        if self._current_radius is None:
            self._current_radius = target_radius
        else:
            self._current_radius += self._smoothing_factor * (target_radius - self._current_radius)
        return self._current_radius


def draw_brush_preview(
    image: np.ndarray,
    cursor: Cursor,
    brush_radius: float,
    color: Tuple[int, int, int],
    eraser_active: bool,
    label: Optional[str] = None,
) -> np.ndarray:
    """Draws a ring around the cursor showing the current brush/eraser footprint.

    Purely visual: the ring's radius mirrors whatever brush size was
    already decided upstream (by `ToolState` in Reality Painter) - this
    function makes no decision about size, color, or tool selection, it
    only renders the values it is given. While the eraser is active, the
    ring is drawn in a neutral white regardless of the selected palette
    color, so it's visually distinct from a paint preview. An optional
    short label (e.g. a brush or shape name) can be drawn just below the
    ring when the caller has one available - Overlay never invents this
    text itself, it only draws a string it was given.

    Args:
        image: BGR image to draw on. Mutated in place.
        cursor: The current smoothed cursor position (normalized).
        brush_radius: Current (already-smoothed) brush radius in pixels.
        color: Current brush BGR color.
        eraser_active: Whether the eraser tool is currently active.
        label: Optional short text (e.g. brush type or shape name) to
            draw beneath the preview ring. Omitted if `None`.

    Returns:
        The same image, with the brush preview ring (and optional label)
        drawn on it.
    """
    height, width = image.shape[:2]
    center = (int(cursor.x * width), int(cursor.y * height))
    radius = max(1, int(round(brush_radius)))
    ring_color = _ERASER_PREVIEW_COLOR if eraser_active else color
    cv2.circle(image, center, radius, ring_color, 1, cv2.LINE_AA)

    if label:
        text_origin = (center[0] - len(label) * 3, center[1] + radius + _BRUSH_PREVIEW_LABEL_OFFSET_PX)
        cv2.putText(image, label, text_origin, _HUD_FONT, 0.45, _BRUSH_PREVIEW_LABEL_COLOR, 1, cv2.LINE_AA)

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
    backgrounds, so drawing several trail points or a pulsing ring per
    frame stays cheap regardless of camera resolution.

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
    cursor, the user HUD, and the gesture-transition banner are drawn
    independently of this flag and are unaffected by it.
    """

    def __init__(self, visible: bool = True) -> None:
        """Creates a HUD visibility toggle.

        Args:
            visible: Initial state. The developer HUD is visible by
                default.
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
    """Draws the developer debug panel (upper-left corner).

    Purely a rendering function: it draws whatever values it is given
    and makes no decisions about FPS, mouse-control state, gesture
    recognition, action mapping, cursor position, or tracking quality
    itself. A semi-transparent panel is drawn behind the text so the HUD
    stays readable and visually distinct from hand landmarks and the
    cursor, which are drawn directly onto the camera feed elsewhere.

    Tool/brush/color information intentionally does NOT live here - see
    `draw_user_hud` - since that is drawing-relevant information the
    user HUD shows regardless of whether the developer HUD is toggled
    on.

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


def draw_user_hud(
    image: np.ndarray,
    brush_size: Optional[int],
    brush_color_name: Optional[str],
    brush_color: Optional[Tuple[int, int, int]],
    eraser_active: Optional[bool],
    tool_name: Optional[str] = None,
    shape_name: Optional[str] = None,
) -> np.ndarray:
    """Draws the always-available user HUD (upper-right corner).

    Shows only what's relevant while actively drawing: the active tool,
    brush size, color (as a name plus a small swatch), and eraser state.
    A no-op if no brush/tool information is present in context yet
    (e.g. before the first frame reaches the painting stage), so this
    never draws a panel of placeholder values.

    `tool_name` and `shape_name` are optional, forward-compatible
    values: this module has no dependency on any specific application's
    tool/shape types, so it draws them only if the caller's pipeline
    context happens to include them, and simply omits those lines
    otherwise.

    Args:
        image: BGR image to draw on. Mutated in place.
        brush_size: Current brush size in pixels, or `None` if unknown.
        brush_color_name: Current brush color's display name, or `None`.
        brush_color: Current brush BGR color, or `None`.
        eraser_active: Whether the eraser tool is active, or `None`.
        tool_name: Optional display name of the active brush/tool type.
        shape_name: Optional display name of the active shape tool, if
            any is currently selected.

    Returns:
        The same image, with the user HUD drawn on it (if applicable).
    """
    if brush_size is None:
        return image

    body_lines = []
    if shape_name:
        body_lines.append(f"Shape: {shape_name}")
    elif tool_name:
        body_lines.append(f"Tool:  {'Eraser' if eraser_active else tool_name}")
    else:
        body_lines.append(f"Tool:  {'Eraser' if eraser_active else 'Brush'}")
    body_lines.append(f"Size:  {brush_size}px")
    if brush_color_name and not eraser_active:
        body_lines.append(f"Color: {brush_color_name}")

    panel_height = _USER_HUD_PANEL_PADDING_PX * 2 + len(body_lines) * _USER_HUD_LINE_HEIGHT_PX
    _, width = image.shape[:2]
    panel_right = width - _USER_HUD_MARGIN_PX
    panel_left = panel_right - _USER_HUD_PANEL_WIDTH_PX
    panel_top = _USER_HUD_MARGIN_PX
    panel_bottom = panel_top + panel_height

    panel_roi = image[panel_top:panel_bottom, panel_left:panel_right]
    if panel_roi.size == 0:
        return image
    roi_overlay = panel_roi.copy()
    cv2.rectangle(roi_overlay, (0, 0), (roi_overlay.shape[1], roi_overlay.shape[0]), _USER_HUD_PANEL_COLOR, -1)
    cv2.addWeighted(roi_overlay, _USER_HUD_PANEL_ALPHA, panel_roi, 1 - _USER_HUD_PANEL_ALPHA, 0, dst=panel_roi)

    text_x = panel_left + _USER_HUD_PANEL_PADDING_PX
    for index, line in enumerate(body_lines):
        y = panel_top + _USER_HUD_PANEL_PADDING_PX + (index + 1) * _USER_HUD_LINE_HEIGHT_PX - 6
        cv2.putText(image, line, (text_x, y), _USER_HUD_FONT, 0.5, _USER_HUD_LABEL_COLOR, 1, cv2.LINE_AA)

    if brush_color is not None and not eraser_active:
        swatch_y = panel_top + panel_height - _USER_HUD_PANEL_PADDING_PX - _USER_HUD_SWATCH_SIZE_PX
        swatch_x = panel_right - _USER_HUD_PANEL_PADDING_PX - _USER_HUD_SWATCH_SIZE_PX
        cv2.rectangle(
            image,
            (swatch_x, swatch_y),
            (swatch_x + _USER_HUD_SWATCH_SIZE_PX, swatch_y + _USER_HUD_SWATCH_SIZE_PX),
            brush_color,
            -1,
        )
        cv2.rectangle(
            image,
            (swatch_x, swatch_y),
            (swatch_x + _USER_HUD_SWATCH_SIZE_PX, swatch_y + _USER_HUD_SWATCH_SIZE_PX),
            (255, 255, 255),
            1,
        )

    return image


def create_overlay_stage() -> StageFunc:
    """Builds a pipeline stage that draws hands, cursor, previews, and HUDs.

    Reads `context["frame"]`, `context["hands"]`, `context["fps"]`,
    `context["mouse_enabled"]`, `context["gestures"]`,
    `context["action"]`, `context["cursor"]`,
    `context["toggle_debug_requested"]`, `context["brush_size"]`,
    `context["brush_color"]`, `context["brush_color_name"]`, and
    `context["eraser_active"]`. It also reads two optional,
    forward-compatible keys - `context["brush_type_name"]` and
    `context["shape_type"]` - purely as opaque display strings, without
    any dependency on where they came from.

    Hand landmarks are drawn only if both a frame and hands are present,
    matching prior behavior. The cursor trail, brush preview, and engine
    cursor are drawn whenever a cursor position is present, regardless
    of whether a hand is currently detected. Any application-specific
    overlay (e.g. Reality Painter's radial menu) is expected to already
    be baked into `context["frame"]` by an earlier stage in that
    application's own pipeline - this module never imports or renders
    such UI itself, keeping the engine reusable by any application.

    The user HUD (tool/size/color/eraser) is always drawn once brush
    information is available in context, independent of the developer
    HUD's visibility. The developer HUD (FPS, tracking confidence,
    gesture, action, cursor position, mouse state, hand count, plus the
    gesture-transition banner) is drawn only while the internal
    HUD-visibility toggle is on. Neither HUD crops, resizes, or reduces
    the usable camera/drawing area - both are drawn as translucent
    panels over the existing frame, matching the "maximum drawing area"
    requirement.

    Gesture-transition, HUD-visibility, cursor-trail, and brush-preview
    smoothing state are tracked internally via closure-owned helpers,
    matching the existing pattern used by `mirror.py`
    (`last_flipped_frame_id`) and `mouse_controller.py`
    (`previously_enabled`) for stage-local state that must persist
    across pipeline executions but has no reason to be owned by the
    calling application.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """
    transition_tracker = _GestureTransitionTracker()
    hud_toggle = _HUDVisibilityToggle()
    trail_tracker = _CursorTrailTracker()
    brush_preview_smoother = _BrushPreviewSmoother()

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
        brush_size = context.get("brush_size")
        brush_color = context.get("brush_color")
        brush_color_name = context.get("brush_color_name")
        eraser_active = context.get("eraser_active")
        brush_type_name = context.get("brush_type_name")
        shape_type = context.get("shape_type")

        if isinstance(cursor, Cursor):
            height, width = frame.image.shape[:2]
            trail_tracker.record(cursor, width, height)
            trail_tracker.draw(frame.image)

            if brush_size is not None and brush_color is not None:
                smoothed_radius = brush_preview_smoother.update(max(1, brush_size) / 2.0)
                preview_label = shape_type or brush_type_name
                draw_brush_preview(
                    frame.image,
                    cursor,
                    smoothed_radius,
                    brush_color,
                    bool(eraser_active),
                    preview_label,
                )

            dragging = action in _DRAGGING_ACTIONS
            pulse_phase = (time.monotonic() % _CURSOR_PULSE_PERIOD_SECONDS) / _CURSOR_PULSE_PERIOD_SECONDS
            draw_cursor(frame.image, cursor, dragging, pulse_phase)

        draw_user_hud(
            frame.image,
            brush_size,
            brush_color_name,
            brush_color,
            eraser_active,
            tool_name=brush_type_name.title() if isinstance(brush_type_name, str) else None,
            shape_name=shape_type.title() if isinstance(shape_type, str) else None,
        )

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