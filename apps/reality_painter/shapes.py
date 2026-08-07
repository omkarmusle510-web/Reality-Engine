"""Shape rendering system for Reality Painter.

Defines a common shape interface so drag-to-draw shape tools (Canvas, in
a future phase) can render previews and commit final shapes without
knowing which concrete shape is active. Each shape owns only its own
rendering behavior - it never manages canvas buffers, undo history, or
tool selection; that remains the caller's responsibility (see
apps/reality_painter/sketch.py). This module is standalone and performs
no application wiring.

Every shape renders onto a persistent BGR color layer and a persistent
single-channel coverage mask, both supplied by the caller. Preview
rendering draws into a caller-provided scratch pair (so a shape can be
redrawn every frame while dragging without disturbing the committed
canvas), while final rendering commits directly onto the persistent
buffers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type

import cv2
import numpy as np

BGRColor = Tuple[int, int, int]
Point = Tuple[int, int]

#: Coverage value written to the mask for fully-drawn shape pixels,
#: matching the convention used by apps.reality_painter.sketch.Canvas
#: (255 marks "painted").
_MASK_PAINTED_VALUE = 255


class Shape(ABC):
    """Common interface every shape implements.

    A caller drives shape drawing purely through `draw_preview` and
    `draw_final` - it never needs to know which concrete shape is
    active or how it renders. New shapes are added by subclassing and
    registering, never by branching on shape type elsewhere.
    """

    #: Human-readable name, for any future UI/menu that lists shapes.
    name: str = "shape"

    @abstractmethod
    def _render(
        self,
        layer: np.ndarray,
        mask: np.ndarray,
        start_point: Point,
        end_point: Point,
        color: BGRColor,
        thickness: int,
        line_type: int,
    ) -> None:
        """Renders this shape's outline onto the given buffers.

        Shared by both preview and final rendering - the only
        difference between the two is which buffers are passed in and
        which line type is used (see `draw_preview`/`draw_final`), not
        how the shape itself is drawn.

        Args:
            layer: BGR color buffer, mutated in place.
            mask: Single-channel coverage buffer (uint8, 0-255),
                mutated in place.
            start_point: Pixel-space (x, y) where the drag began.
            end_point: Pixel-space (x, y) of the current/final drag
                position.
            color: BGR shape color.
            thickness: Outline thickness in pixels.
            line_type: OpenCV line type to draw with (antialiased for
                final rendering, non-antialiased for cheap per-frame
                preview redraws).
        """

    def draw_preview(
        self,
        layer: np.ndarray,
        mask: np.ndarray,
        start_point: Point,
        end_point: Point,
        color: BGRColor,
        thickness: int,
    ) -> None:
        """Draws a temporary preview of this shape while a drag is in progress.

        Intended to be called against a caller-owned scratch layer/mask
        pair - not the persistent canvas buffers - so the preview can be
        cheaply redrawn from scratch every frame (the caller clears its
        scratch buffers first) without touching committed artwork.
        Non-antialiased for speed, since this may run once per frame for
        the duration of the drag.

        Args:
            layer: Scratch BGR buffer to draw the preview into.
            mask: Scratch coverage buffer to draw the preview into.
            start_point: Pixel-space (x, y) where the drag began.
            end_point: Pixel-space (x, y) of the current drag position.
            color: BGR shape color.
            thickness: Outline thickness in pixels.
        """
        self._render(layer, mask, start_point, end_point, color, thickness, cv2.LINE_8)

    def draw_final(
        self,
        layer: np.ndarray,
        mask: np.ndarray,
        start_point: Point,
        end_point: Point,
        color: BGRColor,
        thickness: int,
    ) -> None:
        """Commits this shape onto the persistent canvas buffers.

        Intended to be called once, when the drag ends, against the
        canvas's real color layer and paint mask. Antialiased, since
        this is drawn only once rather than every frame.

        Args:
            layer: Persistent BGR color buffer, mutated in place.
            mask: Persistent coverage buffer, mutated in place.
            start_point: Pixel-space (x, y) where the drag began.
            end_point: Pixel-space (x, y) where the drag ended.
            color: BGR shape color.
            thickness: Outline thickness in pixels.
        """
        self._render(layer, mask, start_point, end_point, color, thickness, cv2.LINE_AA)


class Line(Shape):
    """A straight line from the drag's start point to its end point."""

    name = "Line"

    def _render(self, layer, mask, start_point, end_point, color, thickness, line_type) -> None:
        cv2.line(layer, start_point, end_point, color, thickness, line_type)
        cv2.line(mask, start_point, end_point, _MASK_PAINTED_VALUE, thickness, line_type)


class Rectangle(Shape):
    """An axis-aligned rectangle spanning the drag's start and end points."""

    name = "Rectangle"

    def _render(self, layer, mask, start_point, end_point, color, thickness, line_type) -> None:
        cv2.rectangle(layer, start_point, end_point, color, thickness, line_type)
        cv2.rectangle(mask, start_point, end_point, _MASK_PAINTED_VALUE, thickness, line_type)


class Circle(Shape):
    """A circle centered on the drag's start point, sized by the drag distance."""

    name = "Circle"

    def _render(self, layer, mask, start_point, end_point, color, thickness, line_type) -> None:
        radius = int(round(((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2) ** 0.5))
        radius = max(1, radius)
        cv2.circle(layer, start_point, radius, color, thickness, line_type)
        cv2.circle(mask, start_point, radius, _MASK_PAINTED_VALUE, thickness, line_type)


_SHAPE_REGISTRY: Dict[str, Type[Shape]] = {
    "line": Line,
    "rectangle": Rectangle,
    "circle": Circle,
}


def create_shape(shape_type: str) -> Shape:
    """Instantiates a shape by its registry key.

    Adding a new shape later means creating a `Shape` subclass and
    adding one entry to `_SHAPE_REGISTRY` - no other code in this
    module changes.

    Args:
        shape_type: One of "line", "rectangle", "circle"
            (case-insensitive).

    Returns:
        A new instance of the requested shape.

    Raises:
        ValueError: If `shape_type` isn't a registered shape.
    """
    key = shape_type.strip().lower()
    shape_class = _SHAPE_REGISTRY.get(key)
    if shape_class is None:
        raise ValueError(f"Unknown shape type {shape_type!r}. Available: {sorted(_SHAPE_REGISTRY)}.")
    return shape_class()