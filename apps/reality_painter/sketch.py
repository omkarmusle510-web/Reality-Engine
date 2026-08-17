"""Persistent painting canvas and tools for Reality Painter.

Painting is specific to Reality Painter, not a generic Reality Engine
capability - Reality Engine only provides frames, cursor positions,
high-level actions (LEFT_CLICK, DRAG, RELEASE, ...), and a generic raw
key passthrough from Display. Interpreting a "drag" as "draw a stroke,"
a specific key as "increase brush size" or "undo," are entirely
Reality-Painter-specific decisions, so this module stays here rather
than moving into engine/rendering/ or engine/interaction/. No
MediaPipe, no OS mouse APIs, no gesture recognition happens here - this
module only draws BGR pixels onto a persistent buffer using OpenCV
(plus stdlib file I/O for saving), driven by values other stages
already computed.

Phase 10 integrates three standalone, previously-accepted modules
without duplicating their logic:
    - `apps.reality_painter.brushes`: brush stroke rendering. `Canvas`
      delegates paint-stroke drawing to whichever `Brush` `ToolState`
      currently has selected; Canvas itself no longer decides how a
      stroke looks, only when and where one happens.
    - `apps.reality_painter.shapes`: drag-to-draw shapes. `Canvas`
      delegates both preview and final rendering to whichever `Shape`
      `ToolState` currently has selected.
    - `apps.reality_painter.menu`: a radial menu used as the entry
      point for selecting Brush, Color, Eraser, Undo, Save, and Clear.
      The menu only ever reports a hovered/selected item id; this
      module alone decides what each id means in terms of Canvas and
      ToolState calls.
"""

from __future__ import annotations

import math
import os
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from apps.reality_painter.ai.manager import AIManager
from apps.reality_painter.ai.models import AICapability
from apps.reality_painter.brushes import Brush, create_brush
from apps.reality_painter.menu import Menu, MenuItem, render_menu
from apps.reality_painter.shapes import Shape, create_shape
from engine.core.logger import get_logger
from engine.core.pipeline import PipelineContext, StageFunc
from engine.interaction.action import Action
from engine.interaction.cursor import Cursor
from engine.vision.frame import Frame

logger = get_logger(__name__)

_DRAWING_ACTIONS = (Action.LEFT_CLICK, Action.DRAG)

# --- Brush size -----------------------------------------------------------
_BRUSH_MIN_SIZE_PX = 2
_BRUSH_MAX_SIZE_PX = 40
_BRUSH_DEFAULT_SIZE_PX = 6
_BRUSH_SIZE_STEP_PX = 2
# EMA smoothing factor applied to the displayed/used brush size every
# frame, the same pattern engine.interaction.cursor_mapper.CursorSmoother
# already uses for cursor position - this is what makes size changes
# ("Smooth Transitions") ease toward their target instead of snapping.
_BRUSH_SIZE_SMOOTHING = 0.35
_BRUSH_SIZE_DECREASE_KEY = ord("[")
_BRUSH_SIZE_INCREASE_KEY = ord("]")

# --- Color palette ----------------------------------------------------------
# Ordered list of (name, BGR color) pairs. Extending the palette later is
# just appending an entry here - no branching logic depends on the list's
# length or contents beyond its size, so this stays open for future
# custom colors without any structural change.
_PALETTE: List[Tuple[str, Tuple[int, int, int]]] = [
    ("Orange", (60, 180, 255)),
    ("Red", (50, 50, 220)),
    ("Green", (80, 200, 80)),
    ("Blue", (220, 140, 60)),
    ("Magenta", (200, 80, 200)),
]
_COLOR_SELECT_KEYS: Dict[int, int] = {ord(str(i + 1)): i for i in range(len(_PALETTE))}

# --- Eraser -----------------------------------------------------------------
_ERASER_TOGGLE_KEYS = (ord("e"), ord("E"))

# --- Brush type ---------------------------------------------------------
# Selectable `apps.reality_painter.brushes.Brush` implementations. Cycling
# (rather than a dedicated key per type, as the palette uses) keeps this
# open to future brush types without needing new keys reserved for them.
_BRUSH_TYPES: List[str] = ["hard", "soft", "marker", "highlighter"]
_BRUSH_CYCLE_KEYS = (ord("b"), ord("B"))

# --- Shape tool -----------------------------------------------------------
# `None` represents "shape tool off" (freehand brush drawing applies
# instead); the remaining entries are `apps.reality_painter.shapes.Shape`
# registry keys. Cycling through a single list keeps "off" a first-class,
# equally-reachable state rather than a separate toggle plus a separate
# cycle.
_SHAPE_TYPES: List[Optional[str]] = [None, "line", "rectangle", "circle"]
_SHAPE_CYCLE_KEYS = (ord("g"), ord("G"))

# --- Undo / Redo --------------------------------------------------------------
# Bounded history depth. Each entry is a full copy of the canvas's color
# layer and paint mask (~1.2MB combined at a typical 640x480 feed), so a
# deque(maxlen=...) caps worst-case memory at roughly
# _UNDO_MAX_LEVELS * ~1.2MB per stack (undo and redo are each bounded
# independently), regardless of how long a session runs. See the
# Canvas class docstring for the full memory strategy.
_UNDO_MAX_LEVELS = 20
_UNDO_KEYS = (ord("u"), ord("U"))
_REDO_KEYS = (ord("r"), ord("R"))

# --- Clear canvas -------------------------------------------------------------
_CLEAR_KEYS = (ord("c"), ord("C"))

# --- Save canvas ----------------------------------------------------------------
_SAVE_KEYS = (ord("s"), ord("S"))
_SAVE_DIRECTORY = "saved_canvases"
_SAVE_FILENAME_PREFIX = "reality_painter"
_SAVE_FILENAME_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# --- AI generation (true image-to-image) -------------------------------
# A single key trigger for the first implementation - see
# `create_painting_stage`. Deliberately synchronous/blocking (like the
# save-to-disk path above), matching "safest minimal behavior" for this
# phase; a future phase can move this off the pipeline thread.
_AI_GENERATE_KEYS = (ord("a"), ord("A"))
_AI_DEFAULT_PROMPT = (
    "Transform this sketch into a finished, polished piece of artwork, "
    "preserving its composition and subject."
)

# --- Radial menu ------------------------------------------------------------
# The menu is the single entry point for these six actions; each item's
# id is interpreted only in `_apply_menu_selection`, never inside
# `menu.py` itself. Brush size and the shape tool remain key-driven
# (see `_BRUSH_CYCLE_KEYS`/`_SHAPE_CYCLE_KEYS`/bracket keys above) since
# they weren't part of the requested menu surface.
_MENU_TOGGLE_KEYS = (ord("q"), ord("Q"))
_MENU_ITEMS: List[MenuItem] = [
    MenuItem("undo", "Undo"),
    MenuItem("brush", "Brush"),
    MenuItem("save", "Save"),
    MenuItem("color", "Color"),
    MenuItem("eraser", "Eraser"),
    MenuItem("clear", "Clear"),
]

# --- Radial color palette -----------------------------------------------
# A second-level picker shown after selecting "Color" from the existing
# radial menu. The radial menu itself (`apps.reality_painter.menu.Menu`)
# is completely untouched - this only extends what selecting its
# existing "color" item does, and follows the same radial interaction
# language (a center point, wedges distributed evenly around it,
# hover/confirm) rather than a linear row of swatches. Selection still
# writes into the same `ToolState` color state used everywhere else
# (see `ToolState.select_color`); no second color manager is introduced.
_COMPACT_PALETTE: List[Tuple[str, Tuple[int, int, int]]] = [
    ("Red", (50, 50, 220)),
    ("Orange", (60, 180, 255)),
    ("Yellow", (60, 230, 255)),
    ("Green", (80, 200, 80)),
    ("Blue", (220, 140, 60)),
    ("Purple", (204, 50, 153)),
    ("Pink", (180, 105, 255)),
    ("White", (255, 255, 255)),
]
_PALETTE_INNER_RADIUS_PX = 34.0
_PALETTE_OUTER_RADIUS_PX = 115.0
_PALETTE_BORDER_COLOR = (230, 230, 230)
_PALETTE_HOVER_BORDER_COLOR = (60, 150, 255)
_PALETTE_CENTER_COLOR = (40, 40, 40)
_PALETTE_CENTER_HOVER_COLOR = (90, 90, 90)


class _ColorPalette:
    """A compact radial wedge picker for the 8 colors above.

    Visually and interactionally matches `apps.reality_painter.menu.Menu`
    (a center point with items distributed evenly around it, hover
    tracked by cursor position, confirm on click) without touching that
    class's geometry/animation/layout at all - this is a standalone,
    minimal helper using the same open/update/confirm/consume_selection
    shape `Menu` already establishes, so it plugs into the painting
    stage the same way. Owns no reference to `Canvas`/`ToolState`
    itself - it only ever reports a selected color name, same
    separation of concerns as `Menu` + `_apply_menu_selection`.

    Hovering the small center circle and confirming acts as cancel/back
    - the palette closes with nothing selected, mirroring how the
    outer radial `Menu` is dismissed by closing without a confirm.
    """

    def __init__(self) -> None:
        self._visible = False
        self._center: Tuple[int, int] = (0, 0)
        self._hovered_index: Optional[int] = None  # -1 means "center/cancel"
        self._pending_selection: Optional[int] = None

    @property
    def is_visible(self) -> bool:
        return self._visible

    def open(self, center: Tuple[int, int]) -> None:
        self._visible = True
        self._center = center
        self._hovered_index = None
        self._pending_selection = None

    def close(self) -> None:
        self._visible = False
        self._hovered_index = None

    def _wedge_angle_range(self, index: int) -> Tuple[float, float]:
        """Returns the (start, end) angle in degrees for wedge `index`.

        Wedges are spaced evenly starting at the top, proceeding
        clockwise - the same layout convention `Menu._precompute_geometry`
        already uses for its own items.
        """
        count = len(_COMPACT_PALETTE)
        wedge_width = 360.0 / count
        start = -90.0 - wedge_width / 2.0 + index * wedge_width
        return start, start + wedge_width

    def update(self, cursor_point: Tuple[int, int]) -> None:
        if not self._visible:
            return

        dx = cursor_point[0] - self._center[0]
        dy = cursor_point[1] - self._center[1]
        distance = (dx * dx + dy * dy) ** 0.5

        if distance > _PALETTE_OUTER_RADIUS_PX:
            self._hovered_index = None
            return
        if distance <= _PALETTE_INNER_RADIUS_PX:
            self._hovered_index = -1
            return

        angle = math.degrees(math.atan2(dy, dx))
        count = len(_COMPACT_PALETTE)
        wedge_width = 360.0 / count
        # Shift so wedge 0 (centered at -90) starts at angle 0 for indexing.
        shifted = (angle + 90.0 + wedge_width / 2.0) % 360.0
        self._hovered_index = int(shifted // wedge_width) % count

    def confirm(self) -> None:
        if not self._visible or self._hovered_index is None:
            return
        if self._hovered_index >= 0:
            self._pending_selection = self._hovered_index
        self.close()

    def consume_selection(self) -> Optional[Tuple[str, Tuple[int, int, int]]]:
        if self._pending_selection is None:
            return None
        index = self._pending_selection
        self._pending_selection = None
        return _COMPACT_PALETTE[index]


def render_color_palette(image: np.ndarray, palette: "_ColorPalette") -> None:
    """Draws the radial color palette. Pure rendering, mirrors `render_menu`."""
    if not palette.is_visible:
        return

    center = palette._center  # noqa: SLF001 - same-module helper
    hovered = palette._hovered_index  # noqa: SLF001 - same-module helper
    outer = int(_PALETTE_OUTER_RADIUS_PX)
    inner = int(_PALETTE_INNER_RADIUS_PX)
    axes = (outer, outer)

    for index, (_name, color) in enumerate(_COMPACT_PALETTE):
        start_angle, end_angle = palette._wedge_angle_range(index)  # noqa: SLF001
        cv2.ellipse(image, center, axes, 0, start_angle, end_angle, color, -1, cv2.LINE_AA)
        border_color = _PALETTE_HOVER_BORDER_COLOR if index == hovered else _PALETTE_BORDER_COLOR
        cv2.ellipse(image, center, axes, 0, start_angle, end_angle, border_color, 2, cv2.LINE_AA)

    center_color = _PALETTE_CENTER_HOVER_COLOR if hovered == -1 else _PALETTE_CENTER_COLOR
    cv2.circle(image, center, inner, center_color, -1, cv2.LINE_AA)
    cv2.circle(image, center, inner, _PALETTE_BORDER_COLOR, 2, cv2.LINE_AA)


class ToolState:
    """Tracks the currently active painting tool settings.

    Owns brush size (with smoothing), the selected palette color,
    whether the eraser is active, the selected `Brush` implementation,
    and the selected `Shape` tool (or none, for freehand brush mode).
    Pure state - no drawing happens here, no gesture or cursor
    computation happens here. Reacts to raw key codes handed to it by
    the painting stage, and to menu selections applied via its public
    `cycle_*`/`toggle_*` methods; it has no dependency on Display,
    OpenCV windows, `Menu`, or any input mechanism itself.
    """

    def __init__(self) -> None:
        self._target_size = float(_BRUSH_DEFAULT_SIZE_PX)
        self._current_size = float(_BRUSH_DEFAULT_SIZE_PX)
        self._color_index = 0
        self._eraser_active = False
        # An explicit color chosen from the compact palette (Block 11B),
        # overriding `_color_index` when set. The existing numeric-key
        # shortcuts and `cycle_color()` both clear it, so they continue
        # to behave exactly as before once used.
        self._custom_color: Optional[Tuple[str, Tuple[int, int, int]]] = None

        # Brush instances are stateless renderers (see brushes.py), so
        # one of each is created once and reused for the tool state's
        # lifetime rather than reallocated on every selection change.
        self._brush_index = 0
        self._brushes: Dict[str, Brush] = {name: create_brush(name) for name in _BRUSH_TYPES}

        self._shape_index = 0
        self._shapes: Dict[str, Shape] = {
            name: create_shape(name) for name in _SHAPE_TYPES if name is not None
        }

    def update(self, key_pressed: Optional[int]) -> None:
        """Applies a pending key press (if any) and advances size smoothing.

        Always advances the current size toward the target size by one
        smoothing step, even when no key was pressed this frame, so a
        single key press produces a gradual size change across several
        frames rather than an instant jump.

        Args:
            key_pressed: The raw key code pressed this cycle, or `None`.
        """
        if key_pressed == _BRUSH_SIZE_INCREASE_KEY:
            self._target_size = min(_BRUSH_MAX_SIZE_PX, self._target_size + _BRUSH_SIZE_STEP_PX)
            logger.debug("Brush size target increased to %.0f.", self._target_size)
        elif key_pressed == _BRUSH_SIZE_DECREASE_KEY:
            self._target_size = max(_BRUSH_MIN_SIZE_PX, self._target_size - _BRUSH_SIZE_STEP_PX)
            logger.debug("Brush size target decreased to %.0f.", self._target_size)
        elif key_pressed in _COLOR_SELECT_KEYS:
            self._color_index = _COLOR_SELECT_KEYS[key_pressed]
            self._custom_color = None
            logger.info("Color selected: %s.", self.color_name)
        elif key_pressed in _ERASER_TOGGLE_KEYS:
            self.toggle_eraser()
        elif key_pressed in _BRUSH_CYCLE_KEYS:
            self.cycle_brush()
        elif key_pressed in _SHAPE_CYCLE_KEYS:
            self.cycle_shape()

        self._current_size += _BRUSH_SIZE_SMOOTHING * (self._target_size - self._current_size)

    def cycle_color(self) -> None:
        """Advances to the next palette color, wrapping around.

        Equivalent to pressing a numeric palette key, but position-
        independent - this is what lets the radial menu's "Color" item
        change color without knowing the palette's key bindings.
        """
        self._color_index = (self._color_index + 1) % len(_PALETTE)
        self._custom_color = None
        logger.info("Color selected: %s.", self.color_name)

    def select_color(self, name: str, color: Tuple[int, int, int]) -> None:
        """Sets an explicit color (e.g. from the compact palette), overriding the index-based palette.

        The existing numeric-key shortcuts and `cycle_color()` both
        clear this override, so pressing them afterward still behaves
        exactly as before - this is purely an additional way to set
        the same underlying color state, not a second color system.
        """
        self._custom_color = (name, color)
        logger.info("Color selected: %s.", name)

    def toggle_eraser(self) -> None:
        """Toggles the eraser tool on/off."""
        self._eraser_active = not self._eraser_active
        logger.info("Eraser %s.", "activated" if self._eraser_active else "deactivated")

    def cycle_brush(self) -> None:
        """Advances to the next brush type, wrapping around."""
        self._brush_index = (self._brush_index + 1) % len(_BRUSH_TYPES)
        logger.info("Brush selected: %s.", self.brush_type_name)

    def cycle_shape(self) -> None:
        """Advances to the next shape tool, wrapping around.

        Includes `None` (shape tool off, freehand brush drawing applies)
        as one of the positions in the cycle.
        """
        self._shape_index = (self._shape_index + 1) % len(_SHAPE_TYPES)
        logger.info("Shape tool selected: %s.", self.shape_type or "off (brush)")

    @property
    def brush_size(self) -> int:
        """Current smoothed brush size in pixels, rounded to the nearest integer."""
        return max(1, int(round(self._current_size)))

    @property
    def color(self) -> Tuple[int, int, int]:
        """Currently selected color, as a BGR tuple.

        Returns the compact-palette override (see `select_color`) if
        one is set, otherwise the index-based palette color.
        """
        if self._custom_color is not None:
            return self._custom_color[1]
        return _PALETTE[self._color_index][1]

    @property
    def color_name(self) -> str:
        """Currently selected color's display name."""
        if self._custom_color is not None:
            return self._custom_color[0]
        return _PALETTE[self._color_index][0]

    @property
    def eraser_active(self) -> bool:
        """True if the eraser tool is currently active."""
        return self._eraser_active

    @property
    def brush_type_name(self) -> str:
        """Currently selected brush type's registry name (e.g. "hard")."""
        return _BRUSH_TYPES[self._brush_index]

    @property
    def brush(self) -> Brush:
        """The currently selected `Brush` instance.

        Returns the same cached instance across calls (see `__init__`),
        so selecting a brush never allocates.
        """
        return self._brushes[self.brush_type_name]

    @property
    def shape_type(self) -> Optional[str]:
        """Currently selected shape tool's registry name, or `None` if off."""
        return _SHAPE_TYPES[self._shape_index]

    @property
    def shape(self) -> Optional[Shape]:
        """The currently selected `Shape` instance, or `None` if shape mode is off."""
        shape_type = self.shape_type
        return self._shapes[shape_type] if shape_type is not None else None


_CanvasSnapshot = Tuple[np.ndarray, np.ndarray]


class Canvas:
    """A persistent drawing surface that painted strokes accumulate onto.

    The camera frame itself is never the canvas: a separate buffer (plus
    a paint mask marking which pixels have been painted) is kept alive
    across pipeline executions, so strokes remain visible after the hand
    moves elsewhere, and is composited onto each new camera frame every
    cycle. Painted pixels persist; the camera feed underneath keeps
    changing frame to frame - undo, redo, and clear only ever mutate
    this canvas's own buffers, never the frame the camera produced.

    Brush strokes are rendered by delegating to whichever `Brush` the
    caller supplies (see `extend_stroke`) - Canvas owns stroke lifecycle
    (history snapshots, last-point tracking) but never brush-specific
    drawing logic. Erasing remains Canvas's own responsibility, since it
    isn't a "how does this look" decision a `Brush` makes but a direct
    mask/layer clear.

    Shape tools (`Shape` instances) render into a second, scratch
    layer/mask pair owned by this class, so an in-progress shape drag
    can be redrawn as a live preview every frame (`preview_shape`)
    without touching committed artwork, and only reaches the persistent
    buffers once the drag ends (`commit_shape`).

    Memory strategy for undo/redo (see also module-level
    `_UNDO_MAX_LEVELS`): each history entry is a full copy of the color
    layer and paint mask. Both the undo and redo stacks are bounded
    `deque`s, so total history memory is hard-capped regardless of
    session length - the oldest entry is evicted in O(1) once the cap is
    reached. A snapshot is only captured once per stroke (at its first
    drawn segment) or once per committed shape, not once per frame, so
    holding a drag for many pipeline cycles costs exactly one snapshot.
    """

    def __init__(self) -> None:
        self._layer: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._last_point: Optional[Tuple[int, int]] = None
        self._undo_stack: Deque[_CanvasSnapshot] = deque(maxlen=_UNDO_MAX_LEVELS)
        self._redo_stack: Deque[_CanvasSnapshot] = deque(maxlen=_UNDO_MAX_LEVELS)
        self._stroke_snapshotted = False

        # Scratch buffers for the in-progress shape-tool preview. Never
        # part of undo/redo history and never composited into `_layer`
        # directly - only ever blended onto the outgoing frame in
        # `composite_onto`, so a preview can be redrawn from scratch
        # every frame without any persistent side effect.
        self._scratch_layer: Optional[np.ndarray] = None
        self._scratch_mask: Optional[np.ndarray] = None
        self._shape_start_point: Optional[Tuple[int, int]] = None

    def prepare(self, height: int, width: int) -> None:
        """Lazily (re)creates the canvas buffer to match the frame size.

        The camera's frame size isn't known until the first frame
        arrives. If it changes (e.g. a different camera device), the
        canvas - and its undo/redo history - is reset, since history
        snapshots for a different resolution can't be meaningfully
        restored. Safe to call every frame; it's a no-op once the size
        matches.

        Args:
            height: Frame height in pixels.
            width: Frame width in pixels.
        """
        if self._layer is not None and self._layer.shape[:2] == (height, width):
            return

        self._layer = np.zeros((height, width, 3), dtype=np.uint8)
        self._mask = np.zeros((height, width), dtype=np.uint8)
        self._scratch_layer = np.zeros((height, width, 3), dtype=np.uint8)
        self._scratch_mask = np.zeros((height, width), dtype=np.uint8)
        self._last_point = None
        self._shape_start_point = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._stroke_snapshotted = False
        logger.info("Canvas (re)initialized at %dx%d.", width, height)

    def _snapshot(self) -> _CanvasSnapshot:
        """Captures a copy of the current layer and mask for history storage."""
        return (self._layer.copy(), self._mask.copy())  # type: ignore[union-attr]

    def _restore(self, snapshot: _CanvasSnapshot) -> None:
        """Restores the canvas buffers from a history snapshot."""
        self._layer, self._mask = snapshot
        self._last_point = None

    def _begin_stroke_if_needed(self) -> None:
        """Pushes one undo snapshot at the start of a new stroke, clearing redo.

        Called from `_draw_segment` before any pixels are modified.
        Guarded by `_stroke_snapshotted` so a single stroke - which may
        span many pipeline executions while a drag is held - only ever
        contributes one history entry, at the moment it begins. Starting
        a new stroke is exactly the "new drawing" event that invalidates
        any pending redo history, per the undo/redo requirement.
        """
        if self._stroke_snapshotted or self._layer is None:
            return
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()
        self._stroke_snapshotted = True

    def _draw_segment(
        self,
        point: Tuple[int, int],
        color: Tuple[int, int, int],
        mask_value: int,
        thickness: int,
        antialiased: bool,
    ) -> None:
        """Draws one segment of the current erase stroke directly.

        Used by `erase_stroke` only - paint strokes are now rendered by
        delegating to a `Brush` (see `extend_stroke`), since how a paint
        stroke looks is a brush decision, not a Canvas one. Erasing has
        no equivalent "look" to delegate: it always clears the mask to
        exactly 0 with a hard edge, so it stays Canvas's own direct
        responsibility. A single `cv2.line` call per pair of consecutive
        points handles arbitrarily long segments in roughly constant
        overhead - this is what keeps fast movement from producing gaps.

        Args:
            point: Pixel-space (x, y) position to extend the stroke to.
            color: BGR color to draw onto the color layer.
            mask_value: Value to write onto the paint mask (255 marks
                "painted," 0 marks "unpainted"/erased).
            thickness: Stroke thickness in pixels.
            antialiased: Whether to draw with antialiasing. Erase
                strokes use a hard (non-antialiased) edge so the mask
                clears to exactly 0 rather than leaving a faint
                partially-painted edge from blended antialiasing.
        """
        if self._layer is None or self._mask is None:
            return

        self._begin_stroke_if_needed()

        line_type = cv2.LINE_AA if antialiased else cv2.LINE_8

        if self._last_point is not None:
            cv2.line(self._layer, self._last_point, point, color, thickness, line_type)
            cv2.line(self._mask, self._last_point, point, mask_value, thickness, line_type)
        else:
            radius = max(1, thickness // 2)
            cv2.circle(self._layer, point, radius, color, -1, line_type)
            cv2.circle(self._mask, point, radius, mask_value, -1, line_type)

        self._last_point = point

    def extend_stroke(self, point: Tuple[int, int], brush: Brush, color: Tuple[int, int, int], thickness: int) -> None:
        """Extends the current stroke to a new pixel position using the given brush.

        Delegates the actual rendering (shape, opacity, blending) to
        `brush.stroke_segment` - see `apps.reality_painter.brushes` - so
        Canvas never hardcodes how a stroke looks. Canvas retains only
        stroke lifecycle: opening one undo entry per stroke via
        `_begin_stroke_if_needed`, and tracking the last drawn point so
        the brush can interpolate between consecutive segments.

        Args:
            point: Pixel-space (x, y) position to extend the stroke to.
            brush: The `Brush` responsible for rendering this segment.
            color: BGR brush color for this stroke segment.
            thickness: Brush thickness in pixels.
        """
        if self._layer is None or self._mask is None:
            return

        self._begin_stroke_if_needed()
        brush.stroke_segment(self._layer, self._mask, self._last_point, point, color, thickness)
        self._last_point = point

    def erase_stroke(self, point: Tuple[int, int], thickness: int) -> None:
        """Extends the current stroke to a new pixel position, erasing paint.

        Only the pixels along this stroke are cleared - the rest of the
        canvas is untouched. Uses the same interpolated-line mechanism as
        `extend_stroke`, so fast erasing motion doesn't leave gaps either.
        An erase stroke begins a new history entry exactly like a paint
        stroke does.

        Args:
            point: Pixel-space (x, y) position to extend the erase to.
            thickness: Eraser thickness in pixels (follows brush size).
        """
        self._draw_segment(point, (0, 0, 0), 0, thickness, antialiased=False)

    def end_stroke(self) -> None:
        """Marks the current stroke as finished.

        Called whenever drawing/erasing is not currently active, so the
        next stroke starts fresh instead of connecting to wherever the
        last stroke ended, and so the next stroke will push its own
        fresh undo snapshot rather than being folded into this one.
        """
        self._last_point = None
        self._stroke_snapshotted = False

    def has_active_shape(self) -> bool:
        """True if a shape drag is currently in progress (an anchor point is set)."""
        return self._shape_start_point is not None

    def begin_shape(self, point: Tuple[int, int]) -> None:
        """Marks the start of a new shape drag at the given pixel position.

        Args:
            point: Pixel-space (x, y) anchor position for the shape.
        """
        self._shape_start_point = point

    def preview_shape(
        self,
        shape: Shape,
        point: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Renders a temporary preview of the in-progress shape.

        Drawn into the scratch layer/mask, never the persistent canvas
        buffers - the scratch buffers are cleared and redrawn from
        scratch each call, so the preview tracks the current drag
        position without leaving any trace of earlier preview frames.
        `composite_onto` blends the scratch buffers over the outgoing
        frame after the persistent canvas, so the preview is visible
        without ever being committed.

        Args:
            shape: The `Shape` to render.
            point: Pixel-space (x, y) of the current drag position.
            color: BGR shape color.
            thickness: Outline thickness in pixels.
        """
        if self._scratch_layer is None or self._scratch_mask is None or self._shape_start_point is None:
            return
        self._scratch_layer[:] = 0
        self._scratch_mask[:] = 0
        shape.draw_preview(self._scratch_layer, self._scratch_mask, self._shape_start_point, point, color, thickness)

    def commit_shape(
        self,
        shape: Shape,
        point: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Commits the in-progress shape onto the persistent canvas.

        Pushes one undo snapshot before committing - the same "one
        history entry per edit" rule strokes and `clear()` already
        follow - then renders the final shape directly onto the
        persistent layer/mask via `shape.draw_final`, and clears the
        scratch preview and drag state so the next shape starts fresh.

        Args:
            shape: The `Shape` to render.
            point: Pixel-space (x, y) where the drag ended.
            color: BGR shape color.
            thickness: Outline thickness in pixels.
        """
        if self._layer is None or self._mask is None or self._shape_start_point is None:
            return

        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()
        shape.draw_final(self._layer, self._mask, self._shape_start_point, point, color, thickness)
        self._shape_start_point = None
        self._clear_scratch()
        logger.info("Shape committed (%d undo entries).", len(self._undo_stack))

    def cancel_shape(self) -> None:
        """Abandons the in-progress shape drag without committing anything.

        Used when the drag ends without a valid end position to commit
        to (e.g. the tracked hand disappeared mid-drag).
        """
        self._shape_start_point = None
        self._clear_scratch()

    def _clear_scratch(self) -> None:
        """Clears the shape-preview scratch buffers, if allocated."""
        if self._scratch_layer is not None and self._scratch_mask is not None:
            self._scratch_layer[:] = 0
            self._scratch_mask[:] = 0

    def undo(self) -> bool:
        """Reverts the canvas to its state before the most recent stroke.

        Only affects this canvas's own buffers - the camera frame is
        never touched, since it's produced fresh from the camera every
        cycle and only ever read (never mutated) by `Canvas`. The
        reverted state is pushed onto the redo stack, so a subsequent
        `redo()` can restore it.

        Returns:
            True if an undo was performed, False if there was nothing to
            undo (empty history, or canvas not yet initialized).
        """
        if not self._undo_stack or self._layer is None or self._mask is None:
            logger.info("Undo requested but no history is available.")
            return False

        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        logger.info("Undo applied (%d undo entries remaining).", len(self._undo_stack))
        return True

    def redo(self) -> bool:
        """Re-applies the most recently undone stroke, if any.

        Returns:
            True if a redo was performed, False if there was nothing to
            redo (empty redo stack, or canvas not yet initialized).
        """
        if not self._redo_stack or self._layer is None or self._mask is None:
            logger.info("Redo requested but no redo history is available.")
            return False

        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        logger.info("Redo applied (%d redo entries remaining).", len(self._redo_stack))
        return True

    def clear(self) -> bool:
        """Clears all painted strokes from the canvas.

        Only the drawing canvas is cleared - the camera preview is
        produced independently every frame and is never touched here.
        Brush settings (size, color, eraser state) live entirely in
        `ToolState`, not in `Canvas`, so clearing the canvas has no
        effect on them.

        History behavior: the pre-clear canvas state is pushed onto the
        undo stack before clearing, so `undo()` immediately after a
        clear restores everything that was on the canvas. Clearing is
        treated as a state-changing edit like a stroke, so the redo
        stack is cleared too - the same "new edit invalidates pending
        redo" rule that applies to starting a fresh stroke.

        Returns:
            True if the canvas was cleared, False if it hasn't been
            initialized yet (nothing to clear).
        """
        if self._layer is None or self._mask is None:
            return False

        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()
        self._layer[:] = 0
        self._mask[:] = 0
        self._last_point = None
        self._stroke_snapshotted = False
        logger.info("Canvas cleared.")
        return True

    def export_snapshot(self) -> Optional[np.ndarray]:
        """Exports the current painted artwork as a standalone BGR image.

        Renders only what the user has actually painted, onto a plain
        white background - the live camera feed underneath is never
        included. This is the "user drawing" referenced by the AI
        subsystem's true image-to-image flow: the returned pixels are
        what gets sent to a provider (e.g. Gemini) as the source image,
        completely decoupled from whatever the camera happens to be
        showing at generation time.

        Returns a fresh copy; the persistent canvas buffers are never
        mutated or exposed directly by this method.

        Returns:
            A BGR `numpy.ndarray` snapshot of the painted artwork, or
            `None` if the canvas hasn't been initialized yet (no frame
            has reached `prepare()` yet).
        """
        if self._layer is None or self._mask is None:
            return None

        snapshot = np.full_like(self._layer, 255)
        painted = self._mask.astype(bool)
        snapshot[painted] = self._layer[painted]
        return snapshot

    def composite_onto(self, image: np.ndarray) -> None:
        """Blends painted strokes and any in-progress shape preview onto the given image.

        Only pixels the user has actually painted (mask nonzero) are
        copied - unpainted or erased canvas area leaves the camera image
        untouched, so the canvas never obscures the live feed except
        where strokes currently exist. The shape-preview scratch buffer
        is blended on top of that, using the same nonzero-mask rule, so
        an in-progress shape drag is visible without ever touching the
        persistent canvas buffers.

        Args:
            image: BGR image to draw onto (typically the current
                frame). Mutated in place.
        """
        if self._layer is None or self._mask is None:
            return
        if image.shape[:2] != self._layer.shape[:2]:
            return

        painted = self._mask.astype(bool)
        image[painted] = self._layer[painted]

        if self._scratch_layer is not None and self._scratch_mask is not None:
            preview_painted = self._scratch_mask.astype(bool)
            if preview_painted.any():
                image[preview_painted] = self._scratch_layer[preview_painted]


def save_canvas_image(merged_image: np.ndarray) -> Optional[str]:
    """Saves the current merged (camera + painted artwork) frame to disk.

    `Canvas.composite_onto` already overwrites the frame in place with
    painted strokes blended over the live camera feed before this
    function is ever called - there is no separate "artwork-only, no
    camera" frame produced anywhere downstream of `Canvas`, so the
    merged frame passed in here already is the "painted artwork" this
    feature saves. Filenames are timestamp-based so repeated saves never
    collide. Uses only the Python standard library (`os`, `datetime`)
    plus OpenCV, already a project dependency - no new third-party
    library is introduced.

    Args:
        merged_image: The current frame's BGR pixel data, after
            `Canvas.composite_onto` has already run this cycle.

    Returns:
        The path the image was written to, or `None` if the write
        failed.
    """
    try:
        os.makedirs(_SAVE_DIRECTORY, exist_ok=True)
    except OSError:
        logger.exception("Could not create save directory '%s'.", _SAVE_DIRECTORY)
        return None

    timestamp = datetime.now().strftime(_SAVE_FILENAME_TIMESTAMP_FORMAT)
    filename = f"{_SAVE_FILENAME_PREFIX}_{timestamp}.png"
    path = os.path.join(_SAVE_DIRECTORY, filename)

    success = cv2.imwrite(path, merged_image)
    if not success:
        logger.error("Failed to write canvas image to '%s'.", path)
        return None

    logger.info("Canvas saved to '%s'.", path)
    return path


def save_ai_image(image_bytes: bytes, mime_type: Optional[str]) -> Optional[str]:
    """Saves AI-generated image bytes to disk, alongside user-saved canvases.

    Uses the same `_SAVE_DIRECTORY` as `save_canvas_image` but a
    distinct filename prefix, so AI-generated results are easy to tell
    apart from the user's own saved canvases and never overwrite them.
    The user's live canvas (`Canvas`'s persistent buffers) is never
    touched by this function - saving an AI result is purely a
    side-effect write to disk.

    Args:
        image_bytes: Raw encoded image bytes, as returned by a
            provider's `AIResponse.data["image_bytes"]`.
        mime_type: The image's MIME type (e.g. "image/png"), used only
            to pick a matching file extension. Falls back to `.png` for
            an unrecognized or missing MIME type.

    Returns:
        The path the image was written to, or `None` if the write
        failed.
    """
    try:
        os.makedirs(_SAVE_DIRECTORY, exist_ok=True)
    except OSError:
        logger.exception("Could not create save directory '%s'.", _SAVE_DIRECTORY)
        return None

    extension = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime_type or "", ".png")
    timestamp = datetime.now().strftime(_SAVE_FILENAME_TIMESTAMP_FORMAT)
    filename = f"{_SAVE_FILENAME_PREFIX}_ai_{timestamp}{extension}"
    path = os.path.join(_SAVE_DIRECTORY, filename)

    try:
        with open(path, "wb") as file:
            file.write(image_bytes)
    except OSError:
        logger.exception("Failed to write AI-generated image to '%s'.", path)
        return None

    logger.info("AI-generated image saved to '%s'.", path)
    return path


def _apply_menu_selection(selection: str, canvas: Canvas, tool_state: ToolState) -> bool:
    """Executes a confirmed radial-menu selection.

    `menu.py` reports selection as pure state - an item id and nothing
    more (see `Menu.consume_selection`) - so this is the one place that
    interpretation happens, keeping the menu itself completely unaware
    of Canvas, ToolState, brushes, or saving.

    "save" is deferred rather than performed here: it returns True so
    the caller can perform the actual file write only after this
    cycle's `canvas.composite_onto` has run, the same ordering the
    key-based save shortcut already relies on.

    Args:
        selection: The confirmed menu item's id.
        canvas: The active `Canvas`.
        tool_state: The active `ToolState`.

    Returns:
        True if "save" was selected (deferred to the caller), False
        otherwise.
    """
    if selection == "undo":
        canvas.undo()
    elif selection == "clear":
        canvas.clear()
    elif selection == "eraser":
        tool_state.toggle_eraser()
    elif selection == "brush":
        tool_state.cycle_brush()
    elif selection == "save":
        return True

    return False


def create_painting_stage(
    canvas: Canvas, tool_state: ToolState, ai_manager: Optional[AIManager] = None
) -> StageFunc:
    """Builds a pipeline stage that paints, erases, and manages canvas history.

    Reads `context["frame"]`, `context["cursor"]`, `context["action"]`,
    and `context["key_pressed"]`. `context["key_pressed"]` (a generic
    raw key code from Display) drives brush size, brush type, the shape
    tool, undo, redo, clear, save, AI generation, and the radial menu's
    open/close toggle; Display itself has no knowledge of what any of
    these keys mean, keeping that interpretation entirely inside
    Reality Painter.

    If `ai_manager` is supplied, pressing the AI-generate key
    (`_AI_GENERATE_KEYS`) exports the current drawing via
    `Canvas.export_snapshot()` and runs a synchronous, true
    image-to-image `AIManager.generate()` call - the actual canvas
    pixels plus a generated prompt are sent to whichever provider is
    registered (e.g. Gemini), never a text-only approximation. This
    blocks the pipeline thread for the duration of the call, matching
    the existing synchronous save-to-disk path; a future phase can move
    it off-thread. The result is never composited onto the user's live
    canvas automatically - a successful generation is instead written
    to `saved_canvases/` under a distinct filename via
    `save_ai_image()`, and `context["ai_status"]`
    (`"succeeded"`/`"failed"`) plus `context["ai_result"]` (the saved
    path, or an error message on failure) are set so a caller can
    surface it. If `ai_manager` is `None` (the default), the AI-generate
    key is a no-op - this stage remains fully usable without any AI
    subsystem wired in, exactly as before this capability existed.

    A `Menu` (see `apps.reality_painter.menu`) is owned internally by
    this stage's closure, the same pattern `overlay.py` already uses for
    stage-local state (`_GestureTransitionTracker`, `_HUDVisibilityToggle`,
    etc.) that must persist across pipeline executions without being
    wired through `app.py`. While the menu is open, it is the sole
    consumer of `context["action"]` (a pinch/`Action.LEFT_CLICK`
    confirms the hovered item) and drawing is suspended entirely; the
    menu itself only ever reports a hovered/selected item id via its
    public interface, never touching Canvas or ToolState directly -
    `_apply_menu_selection` is what turns a selection into a concrete
    action.

    Drawing (when the menu is closed) is driven by the same `Action`
    values `MouseController` already reacts to (`LEFT_CLICK` starts a
    stroke/shape, `DRAG` continues it; anything else ends it) - this
    reuses the existing gesture -> action decision already made by
    `ActionMapper`. Freehand strokes are rendered by delegating to
    `tool_state.brush` (paint) or `Canvas.erase_stroke` (erase); when
    `tool_state.shape` is set, drags render through `Canvas`'s
    preview/commit shape methods instead, delegating to
    `tool_state.shape` for the actual rendering.

    Writes `context["brush_size"]`, `context["brush_color"]`,
    `context["brush_color_name"]`, and `context["eraser_active"]` every
    cycle so the overlay stage can display them, without Overlay needing
    any import-level dependency on this module.

    The canvas (and any open menu) is composited/drawn onto the frame
    every cycle, so painted content, an in-progress shape preview, and
    the menu are all visible immediately and stay visible on every
    subsequent frame. The camera frame itself is never the canvas and is
    never persistently modified; it's re-read fresh from the camera
    every cycle by the vision stage.

    Args:
        canvas: A `Canvas` instance, owned by the caller so painted
            strokes and history persist across pipeline executions.
        tool_state: A `ToolState` instance, owned by the caller so tool
            settings persist across pipeline executions.
        ai_manager: An `AIManager` instance to use for the AI-generate
            key, or `None` to disable that key entirely. Owned by the
            caller (see `apps/reality_painter/app.py`).

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """
    menu = Menu(_MENU_ITEMS)
    color_palette = _ColorPalette()

    def _painting_stage(context: PipelineContext) -> PipelineContext:
        frame = context.get("frame")
        if not isinstance(frame, Frame):
            return context

        height, width = frame.image.shape[:2]
        canvas.prepare(height, width)

        key_pressed = context.pop("key_pressed", None)
        tool_state.update(key_pressed)

        if key_pressed in _UNDO_KEYS:
            canvas.undo()
        elif key_pressed in _REDO_KEYS:
            canvas.redo()
        elif key_pressed in _CLEAR_KEYS:
            canvas.clear()

        cursor = context.get("cursor")
        action = context.get("action")
        cursor_point = (int(cursor.x * width), int(cursor.y * height)) if isinstance(cursor, Cursor) else None

        # --- Radial menu: open/close on demand, hover, confirm ----------
        if key_pressed in _MENU_TOGGLE_KEYS:
            if menu.is_visible:
                menu.close()
            elif cursor_point is not None:
                menu.open(cursor_point)

        save_requested = key_pressed in _SAVE_KEYS

        if menu.is_visible:
            if cursor_point is not None:
                menu.update(cursor_point)
            if action == Action.LEFT_CLICK:
                menu.confirm()

        selection = menu.consume_selection()
        if selection == "color":
            # Extends the existing "Color" menu item: instead of
            # cycling, it now opens the compact 8-swatch palette. The
            # radial menu itself (geometry, other items, hover/confirm
            # behavior) is unchanged.
            if cursor_point is not None:
                color_palette.open(cursor_point)
        elif selection is not None:
            save_requested = _apply_menu_selection(selection, canvas, tool_state) or save_requested

        if color_palette.is_visible:
            if cursor_point is not None:
                color_palette.update(cursor_point)
            if action == Action.LEFT_CLICK:
                color_palette.confirm()

        picked_color = color_palette.consume_selection()
        if picked_color is not None:
            tool_state.select_color(*picked_color)

        # --- Drawing (suspended while the menu or color palette is open) ---
        if not menu.is_visible and not color_palette.is_visible:
            shape = tool_state.shape
            if shape is not None:
                if cursor_point is not None and action in _DRAWING_ACTIONS:
                    if not canvas.has_active_shape():
                        canvas.begin_shape(cursor_point)
                    canvas.preview_shape(shape, cursor_point, tool_state.color, tool_state.brush_size)
                elif canvas.has_active_shape():
                    if cursor_point is not None:
                        canvas.commit_shape(shape, cursor_point, tool_state.color, tool_state.brush_size)
                    else:
                        canvas.cancel_shape()
            elif cursor_point is not None and action in _DRAWING_ACTIONS:
                if tool_state.eraser_active:
                    canvas.erase_stroke(cursor_point, tool_state.brush_size)
                else:
                    canvas.extend_stroke(cursor_point, tool_state.brush, tool_state.color, tool_state.brush_size)
            else:
                canvas.end_stroke()

        # Captured before compositing the drawing on top, so entering
        # 3D inspection (see apps.reality_painter.mode_router._freeze_frame)
        # can freeze the live camera view without the painted layer
        # baked in. Purely additional context state - frame.image itself
        # is composited exactly as before for the painting display.
        context["clean_camera_frame"] = frame.image.copy()

        canvas.composite_onto(frame.image)

        if menu.is_visible:
            render_menu(frame.image, menu)

        if color_palette.is_visible:
            render_color_palette(frame.image, color_palette)

        if save_requested:
            saved_path = save_canvas_image(frame.image)
            if saved_path is not None:
                context["canvas_saved_path"] = saved_path

        # --- AI generation (true image-to-image; see docstring) ---------
        if ai_manager is not None and key_pressed in _AI_GENERATE_KEYS:
            snapshot = canvas.export_snapshot()
            if snapshot is None:
                context["ai_status"] = "failed"
                context["ai_result"] = "Canvas not yet initialized."
                logger.warning("AI generation requested before canvas initialization.")
            else:
                logger.info("AI generation requested; dispatching canvas snapshot to AIManager.")
                response = ai_manager.generate(
                    AICapability.IMAGE_GENERATION,
                    user_input=_AI_DEFAULT_PROMPT,
                    sketch=snapshot,
                    context={"canvas": {"width": width, "height": height}},
                )
                image_bytes = response.data.get("image_bytes") if isinstance(response.data, dict) else None
                if response.success and image_bytes:
                    mime_type = response.data.get("mime_type") if isinstance(response.data, dict) else None
                    saved_ai_path = save_ai_image(image_bytes, mime_type)
                    context["ai_status"] = "succeeded" if saved_ai_path else "failed"
                    context["ai_result"] = saved_ai_path or "Generated image could not be saved to disk."
                else:
                    context["ai_status"] = "failed"
                    context["ai_result"] = response.error or "AI generation returned no image."
                    logger.warning("AI generation failed: %s", context["ai_result"])

        context["brush_size"] = tool_state.brush_size
        context["brush_color"] = tool_state.color
        context["brush_color_name"] = tool_state.color_name
        context["eraser_active"] = tool_state.eraser_active

        return context

    return _painting_stage