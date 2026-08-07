"""Radial selection menu for Reality Painter.

A hand-tracked cursor is not pixel-precise, so this menu favors a small
number of large, well-separated targets arranged in a circle around the
cursor rather than a conventional pixel-precise UI. It is hidden by
default, opens only when explicitly requested by the caller, and closes
itself immediately after a single confirmed selection.

This module owns menu state, geometry, hover detection, and animation
only. It has no knowledge of the camera, tracking, gestures, painting,
canvas, undo, brushes, or saving - it only ever produces a currently
hovered item id and a currently selected item id. The caller decides
what "hover" and "confirm" mean in terms of gestures or input, and is
entirely responsible for acting on a selection once it is reported.

Rendering is intentionally separate from state management: `Menu`
exposes read-only, precomputed render data via `iter_render_items()`,
and `render_menu()` is a pure function that draws that data with
OpenCV. Neither needs to know about the other's internals.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from engine.core.logger import get_logger

logger = get_logger(__name__)

Point = Tuple[float, float]

# --- Animation timing -------------------------------------------------
_DEFAULT_OPEN_DURATION_SECONDS = 0.18
_DEFAULT_CLOSE_DURATION_SECONDS = 0.15
_DEFAULT_SELECTION_FLASH_SECONDS = 0.12

# --- Geometry -----------------------------------------------------------
_DEFAULT_RADIUS_PX = 140.0
_DEFAULT_ITEM_HIT_RADIUS_PX = 46.0
_DEFAULT_ITEM_VISUAL_RADIUS_PX = 40.0

# --- Rendering ------------------------------------------------------------
_CENTER_DOT_RADIUS_PX = 6
_ITEM_FILL_COLOR = (60, 60, 60)
_ITEM_HOVER_FILL_COLOR = (60, 150, 255)
_ITEM_SELECTED_FILL_COLOR = (60, 220, 120)
_ITEM_BORDER_COLOR = (230, 230, 230)
_ITEM_LABEL_COLOR = (255, 255, 255)
_CENTER_DOT_COLOR = (230, 230, 230)
_CONNECTOR_COLOR = (110, 110, 110)
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_FONT_SCALE = 0.55
_LABEL_FONT_THICKNESS = 2


def _ease_out_cubic(t: float) -> float:
    """Fast-start, slow-finish easing, used for the menu's opening animation."""
    clamped = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - clamped) ** 3


def _ease_in_cubic(t: float) -> float:
    """Slow-start, fast-finish easing, used for the menu's closing animation."""
    clamped = min(1.0, max(0.0, t))
    return clamped ** 3


class MenuState(Enum):
    """Lifecycle state of a `Menu`."""

    HIDDEN = "hidden"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"


@dataclass(frozen=True)
class MenuItem:
    """A single selectable entry in a `Menu`.

    Pure data - a `MenuItem` has no behavior of its own. Adding a new
    option anywhere `Menu` is constructed (Brush, Color, Eraser, Undo,
    Save, Clear, or any future entry) never requires changes to `Menu`
    or `render_menu`; it is purely a matter of listing another
    `MenuItem` in the items passed to `Menu.__init__`.

    Attributes:
        item_id: Stable identifier the caller uses to interpret a
            selection (e.g. "brush", "undo"). Never displayed.
        label: Short display text shown on the item.
    """

    item_id: str
    label: str


@dataclass(frozen=True)
class RenderItem:
    """Precomputed, read-only data describing how to draw one menu item.

    Produced by `Menu.iter_render_items()`. `render_menu()` (or any
    other renderer) only ever needs this - it never reaches into
    `Menu`'s internal state directly.
    """

    item: MenuItem
    position: Point
    visual_radius: float
    is_hovered: bool
    is_selected: bool


@dataclass
class _ItemGeometry:
    """Precomputed, center-independent layout for one item.

    The unit direction depends only on the item's index and the total
    item count, never on where the menu happens to open - so it is
    computed once per `Menu` instance rather than every frame.
    """

    item: MenuItem
    direction: Point  # unit vector (dx, dy)


class Menu:
    """Radial menu state machine: geometry, hover detection, and animation.

    Usage:
        menu = Menu(items)
        menu.open(cursor_position)
        ...
        menu.update(cursor_position)          # every frame
        if caller_detects_confirmation():
            menu.confirm()
        selected = menu.consume_selection()    # act on it once, if any
    """

    def __init__(
        self,
        items: List[MenuItem],
        radius: float = _DEFAULT_RADIUS_PX,
        item_hit_radius: float = _DEFAULT_ITEM_HIT_RADIUS_PX,
        item_visual_radius: float = _DEFAULT_ITEM_VISUAL_RADIUS_PX,
        open_duration_seconds: float = _DEFAULT_OPEN_DURATION_SECONDS,
        close_duration_seconds: float = _DEFAULT_CLOSE_DURATION_SECONDS,
    ) -> None:
        """Creates a radial menu for a fixed set of items.

        Args:
            items: The menu's entries. Must be non-empty.
            radius: Distance in pixels from the cursor to each item's
                center once fully open.
            item_hit_radius: Distance in pixels from an item's center
                within which the cursor counts as hovering it.
            item_visual_radius: Radius in pixels of each item's drawn
                circle, at full scale.
            open_duration_seconds: Time for the expand-from-cursor
                animation to complete.
            close_duration_seconds: Time for the collapse animation to
                complete.

        Raises:
            ValueError: If `items` is empty.
        """
        if not items:
            raise ValueError("Menu requires at least one MenuItem.")

        self._radius = radius
        self._item_hit_radius = item_hit_radius
        self._item_visual_radius = item_visual_radius
        self._open_duration_seconds = open_duration_seconds
        self._close_duration_seconds = close_duration_seconds

        self._geometry: List[_ItemGeometry] = self._precompute_geometry(items)

        self._state = MenuState.HIDDEN
        self._center: Point = (0.0, 0.0)
        self._animation_started_at: float = 0.0
        self._progress: float = 0.0

        self._hovered_item_id: Optional[str] = None
        self._committed_item_id: Optional[str] = None
        self._pending_selection_id: Optional[str] = None

    @staticmethod
    def _precompute_geometry(items: List[MenuItem]) -> List[_ItemGeometry]:
        """Computes each item's unit direction from the menu center.

        Items are spaced evenly around a full circle, starting at the
        top and proceeding clockwise, matching the layout callers expect
        (e.g. Undo at top, Clear at bottom). This depends only on item
        count, so it is computed once and reused for every open.
        """
        count = len(items)
        geometry: List[_ItemGeometry] = []
        for index, item in enumerate(items):
            angle = -math.pi / 2 + (2 * math.pi * index / count)
            direction = (math.cos(angle), math.sin(angle))
            geometry.append(_ItemGeometry(item=item, direction=direction))
        return geometry

    # --- Lifecycle -------------------------------------------------

    @property
    def state(self) -> MenuState:
        """Current lifecycle state."""
        return self._state

    @property
    def is_visible(self) -> bool:
        """True whenever the menu occupies any screen space at all."""
        return self._state != MenuState.HIDDEN

    def open(self, center: Point, now: Optional[float] = None) -> None:
        """Opens the menu, centered on the given position.

        A no-op if the menu is already open or opening, so a caller can
        call this every frame a "hold to open" gesture is active without
        restarting the animation or moving an already-open menu.

        Args:
            center: Pixel-space (x, y) to center the menu on, typically
                the current cursor position at the moment of request.
            now: Current time in seconds (`time.monotonic()` semantics).
                Defaults to `time.monotonic()` if omitted; exposed for
                deterministic testing.
        """
        if self._state in (MenuState.OPEN, MenuState.OPENING):
            return

        self._center = center
        self._state = MenuState.OPENING
        self._animation_started_at = now if now is not None else time.monotonic()
        self._progress = 0.0
        self._hovered_item_id = None
        self._committed_item_id = None
        self._pending_selection_id = None
        logger.info("Menu opened at (%.0f, %.0f).", center[0], center[1])

    def close(self, now: Optional[float] = None) -> None:
        """Requests the menu collapse and disappear, without selecting anything.

        A no-op if the menu is already hidden or closing. Intended for a
        caller-detected "cancel" (e.g. releasing the open gesture
        without confirming).

        Args:
            now: Current time in seconds. Defaults to `time.monotonic()`.
        """
        if self._state in (MenuState.HIDDEN, MenuState.CLOSING):
            return

        self._state = MenuState.CLOSING
        self._animation_started_at = now if now is not None else time.monotonic()
        self._progress = 0.0
        self._hovered_item_id = None
        logger.info("Menu closed without selection.")

    def update(self, cursor: Point, now: Optional[float] = None) -> None:
        """Advances animation and hover state for the current frame.

        Must be called once per frame while the menu is visible. Hover
        detection only applies while the menu is fully `OPEN` - during
        `OPENING`/`CLOSING`, items are still animating into or out of
        position, so hovering them is not yet (or no longer) meaningful.

        Args:
            cursor: Current cursor position, in the same pixel space as
                the `center` passed to `open()`.
            now: Current time in seconds. Defaults to `time.monotonic()`.
        """
        if self._state == MenuState.HIDDEN:
            return

        current_time = now if now is not None else time.monotonic()
        elapsed = current_time - self._animation_started_at

        if self._state == MenuState.OPENING:
            duration = max(1e-6, self._open_duration_seconds)
            self._progress = min(1.0, elapsed / duration)
            if self._progress >= 1.0:
                self._state = MenuState.OPEN
        elif self._state == MenuState.CLOSING:
            duration = max(1e-6, self._close_duration_seconds)
            self._progress = min(1.0, elapsed / duration)
            if self._progress >= 1.0:
                self._state = MenuState.HIDDEN
                self._progress = 0.0
                self._committed_item_id = None
                return

        if self._state == MenuState.OPEN:
            self._hovered_item_id = self._hit_test(cursor)
        else:
            self._hovered_item_id = None

    def _hit_test(self, cursor: Point) -> Optional[str]:
        """Returns the id of the item whose hit radius contains `cursor`, if any."""
        closest_id: Optional[str] = None
        closest_distance = self._item_hit_radius
        for geometry in self._geometry:
            item_x = self._center[0] + geometry.direction[0] * self._radius
            item_y = self._center[1] + geometry.direction[1] * self._radius
            distance = math.hypot(cursor[0] - item_x, cursor[1] - item_y)
            if distance <= closest_distance:
                closest_distance = distance
                closest_id = geometry.item.item_id
        return closest_id

    # --- Selection ---------------------------------------------------

    @property
    def hovered_item_id(self) -> Optional[str]:
        """The item id currently under the cursor, or `None` if none is hovered.

        Only ever non-`None` while `state` is `OPEN`.
        """
        return self._hovered_item_id

    def confirm(self, now: Optional[float] = None) -> Optional[str]:
        """Commits the currently hovered item as the selection, if any.

        This module makes no assumption about what a "confirmation
        action" is (a pinch, a dwell timer, a key press, etc.) - the
        caller alone decides when to call this. A no-op, returning
        `None`, if the menu is not fully `OPEN` or nothing is currently
        hovered. On a successful confirm, the menu immediately begins
        its close animation so control returns to drawing without
        further input.

        Args:
            now: Current time in seconds. Defaults to `time.monotonic()`.

        Returns:
            The confirmed item id, or `None` if there was nothing to
            confirm.
        """
        if self._state != MenuState.OPEN or self._hovered_item_id is None:
            return None

        selected_id = self._hovered_item_id
        self._committed_item_id = selected_id
        self._pending_selection_id = selected_id
        logger.info("Menu selection confirmed: '%s'.", selected_id)
        self.close(now)
        return selected_id

    def consume_selection(self) -> Optional[str]:
        """Returns and clears the most recently confirmed selection, if unread.

        A caller should call this once per frame (e.g. right after
        `update()`) and act on a non-`None` result exactly once. Once
        consumed, the same selection is never returned again, so a
        selection can never be accidentally applied twice.

        Returns:
            The confirmed item id, or `None` if there is nothing new to
            act on.
        """
        selection = self._pending_selection_id
        self._pending_selection_id = None
        return selection

    # --- Rendering data ------------------------------------------------

    def iter_render_items(self) -> Iterator[RenderItem]:
        """Yields precomputed, read-only render data for each menu item.

        Positions and scale already account for the current open/close
        animation progress, so a renderer never needs to know about
        `MenuState`, easing functions, or animation timing - it only
        draws exactly what it is given. Yields nothing while `state` is
        `HIDDEN`.
        """
        if self._state == MenuState.HIDDEN:
            return

        if self._state == MenuState.OPENING:
            scale = _ease_out_cubic(self._progress)
        elif self._state == MenuState.CLOSING:
            scale = 1.0 - _ease_in_cubic(self._progress)
        else:
            scale = 1.0

        current_radius = self._radius * scale
        current_visual_radius = self._item_visual_radius * scale

        for geometry in self._geometry:
            item = geometry.item
            position = (
                self._center[0] + geometry.direction[0] * current_radius,
                self._center[1] + geometry.direction[1] * current_radius,
            )
            is_hovered = self._state == MenuState.OPEN and item.item_id == self._hovered_item_id
            is_selected = item.item_id == self._committed_item_id
            yield RenderItem(
                item=item,
                position=position,
                visual_radius=current_visual_radius,
                is_hovered=is_hovered,
                is_selected=is_selected,
            )

    @property
    def center(self) -> Point:
        """The menu's current center position (the cursor position at open)."""
        return self._center


def render_menu(image: np.ndarray, menu: Menu) -> np.ndarray:
    """Draws a `Menu`'s current animation frame using OpenCV primitives.

    Pure rendering: reads only `menu.iter_render_items()` and
    `menu.center`, and makes no state decisions of its own. A no-op if
    the menu is not currently visible.

    Args:
        image: BGR image to draw onto. Mutated in place.
        menu: The `Menu` instance to render.

    Returns:
        The same image, with the menu drawn on it.
    """
    if not menu.is_visible:
        return image

    center_point = (int(round(menu.center[0])), int(round(menu.center[1])))

    render_items = list(menu.iter_render_items())

    for render_item in render_items:
        item_point = (int(round(render_item.position[0])), int(round(render_item.position[1])))
        cv2.line(image, center_point, item_point, _CONNECTOR_COLOR, 1, cv2.LINE_AA)

    cv2.circle(image, center_point, _CENTER_DOT_RADIUS_PX, _CENTER_DOT_COLOR, -1, cv2.LINE_AA)

    for render_item in render_items:
        radius = max(1, int(round(render_item.visual_radius)))
        item_point = (int(round(render_item.position[0])), int(round(render_item.position[1])))

        if render_item.is_selected:
            fill_color = _ITEM_SELECTED_FILL_COLOR
        elif render_item.is_hovered:
            fill_color = _ITEM_HOVER_FILL_COLOR
        else:
            fill_color = _ITEM_FILL_COLOR

        cv2.circle(image, item_point, radius, fill_color, -1, cv2.LINE_AA)
        cv2.circle(image, item_point, radius, _ITEM_BORDER_COLOR, 2, cv2.LINE_AA)

        label = render_item.item.label
        (text_width, text_height), _ = cv2.getTextSize(label, _LABEL_FONT, _LABEL_FONT_SCALE, _LABEL_FONT_THICKNESS)
        text_origin = (item_point[0] - text_width // 2, item_point[1] + text_height // 2)
        cv2.putText(
            image,
            label,
            text_origin,
            _LABEL_FONT,
            _LABEL_FONT_SCALE,
            _ITEM_LABEL_COLOR,
            _LABEL_FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return image