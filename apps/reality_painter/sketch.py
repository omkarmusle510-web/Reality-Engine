"""Persistent painting canvas for Reality Painter.

Painting is specific to Reality Painter, not a generic Reality Engine
capability - Reality Engine only provides frames, cursor positions, and
high-level actions (LEFT_CLICK, DRAG, RELEASE, ...). Interpreting a
"drag" as "draw a stroke" is a Reality-Painter-specific decision, so
this module stays here rather than moving into engine/rendering/ or
engine/interaction/. No MediaPipe, no OS mouse APIs, no gesture
recognition happens here - this module only draws BGR pixels onto a
persistent buffer using OpenCV, driven by values other stages already
computed.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from engine.core.logger import get_logger
from engine.core.pipeline import PipelineContext, StageFunc
from engine.interaction.action import Action
from engine.interaction.cursor import Cursor
from engine.vision.frame import Frame

logger = get_logger(__name__)

_BRUSH_COLOR = (60, 180, 255)  # BGR - warm orange
_BRUSH_THICKNESS_PX = 4
_DRAWING_ACTIONS = (Action.LEFT_CLICK, Action.DRAG)


class Canvas:
    """A persistent drawing surface that painted strokes accumulate onto.

    The camera frame itself is never the canvas: a separate buffer (plus
    a paint mask marking which pixels have been painted) is kept alive
    across pipeline executions, so strokes remain visible after the hand
    moves elsewhere, and is composited onto each new camera frame every
    cycle. Painted pixels persist; the camera feed underneath keeps
    changing frame to frame.
    """

    def __init__(self) -> None:
        self._layer: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._last_point: Optional[Tuple[int, int]] = None

    def prepare(self, height: int, width: int) -> None:
        """Lazily (re)creates the canvas buffer to match the frame size.

        The camera's frame size isn't known until the first frame
        arrives. If it changes (e.g. a different camera device), the
        canvas is recreated - existing strokes are not preserved across
        a resolution change, since there's no sensible way to resize
        freehand pixel strokes without distortion. Safe to call every
        frame; it's a no-op once the size matches.

        Args:
            height: Frame height in pixels.
            width: Frame width in pixels.
        """
        if self._layer is not None and self._layer.shape[:2] == (height, width):
            return

        self._layer = np.zeros((height, width, 3), dtype=np.uint8)
        self._mask = np.zeros((height, width), dtype=np.uint8)
        self._last_point = None
        logger.info("Canvas (re)initialized at %dx%d.", width, height)

    def extend_stroke(self, point: Tuple[int, int]) -> None:
        """Extends the current stroke to a new pixel position.

        Draws a single antialiased line segment from the previous point
        to `point` when a previous point already exists (the stroke is
        in progress). A single `cv2.line` call interpolates every pixel
        along the segment internally - this is the "smooth stroke
        interpolation": rather than manually sampling and drawing many
        intermediate points (which would cost more per frame the faster
        the hand moves), one line-draw call handles arbitrarily long
        segments in roughly constant overhead, so fast movement between
        two sampled frames still produces a continuous line instead of a
        gap, cheaply enough for low-end hardware.

        Args:
            point: Pixel-space (x, y) position to extend the stroke to.
        """
        if self._layer is None or self._mask is None:
            return

        if self._last_point is not None:
            cv2.line(self._layer, self._last_point, point, _BRUSH_COLOR, _BRUSH_THICKNESS_PX, cv2.LINE_AA)
            cv2.line(self._mask, self._last_point, point, 255, _BRUSH_THICKNESS_PX, cv2.LINE_AA)
        else:
            # First point of a new stroke: draw a dot so a quick tap
            # still leaves a mark even with no second point yet.
            cv2.circle(self._layer, point, _BRUSH_THICKNESS_PX // 2, _BRUSH_COLOR, -1, cv2.LINE_AA)
            cv2.circle(self._mask, point, _BRUSH_THICKNESS_PX // 2, 255, -1, cv2.LINE_AA)

        self._last_point = point

    def end_stroke(self) -> None:
        """Marks the current stroke as finished.

        Called whenever drawing is not currently active, so the next
        stroke starts fresh instead of connecting to wherever the last
        stroke ended.
        """
        self._last_point = None

    def composite_onto(self, image: np.ndarray) -> None:
        """Blends all painted strokes onto the given image, in place.

        Only pixels the user has actually painted are copied - unpainted
        canvas area leaves the camera image untouched, so the canvas
        never obscures the live feed except where strokes exist.

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


def create_painting_stage(canvas: Canvas) -> StageFunc:
    """Builds a pipeline stage that paints strokes onto a persistent canvas.

    Reads `context["frame"]`, `context["cursor"]`, and
    `context["action"]`. Drawing is driven entirely by the same `Action`
    values `MouseController` already reacts to (`LEFT_CLICK` starts a
    stroke, `DRAG` continues it; anything else ends the current stroke) -
    this reuses the existing gesture -> action decision already made by
    `ActionMapper` rather than inventing a second, painting-specific
    interpretation of gestures. No gesture or cursor computation happens
    here; this stage only decides "should the brush currently be down,"
    which is a Reality-Painter-specific rendering decision, not a
    generic engine concern.

    The canvas is composited onto the frame every cycle - after any
    stroke update - so painted content is visible immediately and stays
    visible on every subsequent frame even after the hand moves away.
    The camera frame itself is never the canvas.

    Args:
        canvas: A `Canvas` instance, owned by the caller so painted
            strokes persist across pipeline executions.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _painting_stage(context: PipelineContext) -> PipelineContext:
        frame = context.get("frame")
        if not isinstance(frame, Frame):
            return context

        height, width = frame.image.shape[:2]
        canvas.prepare(height, width)

        cursor = context.get("cursor")
        action = context.get("action")

        if isinstance(cursor, Cursor) and action in _DRAWING_ACTIONS:
            point = (int(cursor.x * width), int(cursor.y * height))
            canvas.extend_stroke(point)
        else:
            canvas.end_stroke()

        canvas.composite_onto(frame.image)

        return context

    return _painting_stage