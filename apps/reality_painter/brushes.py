"""Brush system for Reality Painter.

Defines a common brush interface so painting logic (Canvas, in a future
phase) can draw strokes without knowing which concrete brush is active.
Each brush owns only its own rendering behavior - it never manages
canvas buffers, undo history, or tool selection; that remains the
caller's responsibility (see apps/reality_painter/sketch.py). This
module is standalone and performs no application wiring.

Every brush draws onto a persistent BGR color layer and a persistent
single-channel coverage mask, both supplied by the caller, using
standard "over" alpha compositing. This lets brushes range from fully
opaque (Hard Brush) to heavily translucent (Highlighter) while sharing
one blending path, and lets overlapping strokes composite correctly
regardless of which brush drew them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type

import cv2
import numpy as np

BGRColor = Tuple[int, int, int]
Point = Tuple[int, int]


def _blend_over(
    layer: np.ndarray,
    mask: np.ndarray,
    stroke_layer: np.ndarray,
    stroke_alpha: np.ndarray,
) -> None:
    """Alpha-composites a single-segment stroke onto the persistent buffers.

    Standard "source over destination" compositing, applied only where
    `stroke_alpha` is nonzero so untouched pixels are never re-blended
    (cheap no-op elsewhere). `layer`/`mask` are mutated in place.

    Args:
        layer: Persistent BGR color buffer (uint8), mutated in place.
        mask: Persistent single-channel coverage buffer (uint8, 0-255),
            mutated in place.
        stroke_layer: BGR buffer holding this segment's drawn color.
        stroke_alpha: Single-channel float32 buffer in [0, 1] giving
            this segment's opacity at each pixel.
    """
    touched = stroke_alpha > 0
    if not np.any(touched):
        return

    alpha = stroke_alpha[touched][..., None]
    layer_f = layer[touched].astype(np.float32)
    stroke_f = stroke_layer[touched].astype(np.float32)
    blended = stroke_f * alpha + layer_f * (1.0 - alpha)
    layer[touched] = blended.astype(np.uint8)

    mask_f = mask[touched].astype(np.float32)
    new_mask = stroke_alpha[touched] * 255.0 + mask_f * (1.0 - stroke_alpha[touched])
    mask[touched] = np.clip(new_mask, 0, 255).astype(np.uint8)


def _draw_line_or_dot(image: np.ndarray, last_point, point, color, thickness: int, line_type: int) -> None:
    """Draws a connecting line, or a filled dot if this is the stroke's start.

    Shared by every brush's `_render_segment` so a stroke's very first
    point (where there is no previous point to connect from) still
    leaves a visible mark, matching how `extend_stroke`/`erase_stroke`
    behave elsewhere in the painting pipeline.
    """
    if last_point is not None:
        cv2.line(image, last_point, point, color, thickness, line_type)
    else:
        radius = max(1, thickness // 2)
        cv2.circle(image, point, radius, color, -1, line_type)


class Brush(ABC):
    """Common interface every brush implements.

    A caller drives strokes purely through `stroke_segment` - it never
    needs to know which concrete brush is active, how it renders, or
    whether it feathers, blends, or hard-edges its strokes. New brushes
    are added by subclassing and registering, never by branching on
    brush type elsewhere.
    """

    #: Human-readable name, for any future UI/menu that lists brushes.
    name: str = "brush"

    @abstractmethod
    def stroke_segment(
        self,
        layer: np.ndarray,
        mask: np.ndarray,
        last_point: Optional[Point],
        point: Point,
        color: BGRColor,
        thickness: int,
    ) -> None:
        """Draws one stroke segment onto the persistent canvas buffers.

        Args:
            layer: Persistent BGR color buffer (uint8), mutated in place.
            mask: Persistent single-channel coverage buffer (uint8,
                0-255), mutated in place.
            last_point: Pixel-space (x, y) of the previous point in this
                stroke, or `None` if this is the stroke's first point.
            point: Pixel-space (x, y) to extend the stroke to.
            color: BGR brush color.
            thickness: Brush thickness in pixels.
        """


class _ScratchStrokeBrush(Brush):
    """Base for brushes that render a segment into a scratch buffer first.

    Shared by every concrete brush below: each draws its own shape and
    opacity into a segment-local scratch pair (`stroke_layer`,
    `stroke_alpha`) sized to the persistent buffers, then hands off to
    `_blend_over` for compositing. This is what lets hard, soft, marker,
    and highlighter brushes share one blending path while only
    differing in how they fill the scratch buffer.
    """

    @abstractmethod
    def _render_segment(
        self,
        stroke_layer: np.ndarray,
        stroke_alpha: np.ndarray,
        last_point: Optional[Point],
        point: Point,
        color: BGRColor,
        thickness: int,
    ) -> None:
        """Renders this segment's shape/opacity into the scratch buffers."""

    def stroke_segment(
        self,
        layer: np.ndarray,
        mask: np.ndarray,
        last_point: Optional[Point],
        point: Point,
        color: BGRColor,
        thickness: int,
    ) -> None:
        stroke_layer = np.zeros_like(layer)
        stroke_alpha = np.zeros(layer.shape[:2], dtype=np.float32)
        self._render_segment(stroke_layer, stroke_alpha, last_point, point, color, thickness)
        _blend_over(layer, mask, stroke_layer, stroke_alpha)


class HardBrush(_ScratchStrokeBrush):
    """Solid, fully opaque brush with crisp, non-antialiased edges."""

    name = "Hard Brush"

    def _render_segment(self, stroke_layer, stroke_alpha, last_point, point, color, thickness) -> None:
        _draw_line_or_dot(stroke_layer, last_point, point, color, thickness, cv2.LINE_8)
        _draw_line_or_dot(stroke_alpha, last_point, point, 1.0, thickness, cv2.LINE_8)


class SoftBrush(_ScratchStrokeBrush):
    """Feathered brush with a Gaussian-falloff edge.

    Draws an antialiased, fully opaque core, then blurs the alpha
    channel to spread opacity outward with a soft falloff, so strokes
    fade toward their perimeter instead of cutting off sharply.
    """

    name = "Soft Brush"

    #: Blur kernel size relative to brush thickness - scales the
    #: feather width with brush size.
    _FEATHER_RATIO = 0.75
    #: Core (pre-blur) thickness relative to the requested thickness,
    #: so the blur has room to feather outward without overshooting it.
    _CORE_RATIO = 0.6

    def _render_segment(self, stroke_layer, stroke_alpha, last_point, point, color, thickness) -> None:
        core_thickness = max(1, int(thickness * self._CORE_RATIO))
        _draw_line_or_dot(stroke_layer, last_point, point, color, core_thickness, cv2.LINE_AA)
        _draw_line_or_dot(stroke_alpha, last_point, point, 1.0, core_thickness, cv2.LINE_AA)

        kernel_size = int(thickness * self._FEATHER_RATIO)
        kernel_size += 1 - (kernel_size % 2)  # force odd, as GaussianBlur requires
        kernel_size = max(3, kernel_size)
        stroke_alpha[:] = cv2.GaussianBlur(stroke_alpha, (kernel_size, kernel_size), 0)
        np.clip(stroke_alpha, 0.0, 1.0, out=stroke_alpha)


class Marker(_ScratchStrokeBrush):
    """Flat, near-opaque brush with slight translucency, like a felt-tip pen."""

    name = "Marker"

    _OPACITY = 0.85

    def _render_segment(self, stroke_layer, stroke_alpha, last_point, point, color, thickness) -> None:
        _draw_line_or_dot(stroke_layer, last_point, point, color, thickness, cv2.LINE_AA)
        _draw_line_or_dot(stroke_alpha, last_point, point, self._OPACITY, thickness, cv2.LINE_AA)


class Highlighter(_ScratchStrokeBrush):
    """Wide, heavily translucent brush that lets existing strokes show through."""

    name = "Highlighter"

    _OPACITY = 0.35
    _WIDTH_MULTIPLIER = 2.5

    def _render_segment(self, stroke_layer, stroke_alpha, last_point, point, color, thickness) -> None:
        wide_thickness = max(1, int(thickness * self._WIDTH_MULTIPLIER))
        _draw_line_or_dot(stroke_layer, last_point, point, color, wide_thickness, cv2.LINE_8)
        _draw_line_or_dot(stroke_alpha, last_point, point, self._OPACITY, wide_thickness, cv2.LINE_8)


_BRUSH_REGISTRY: Dict[str, Type[Brush]] = {
    "hard": HardBrush,
    "soft": SoftBrush,
    "marker": Marker,
    "highlighter": Highlighter,
}


def create_brush(brush_type: str) -> Brush:
    """Instantiates a brush by its registry key.

    Adding a new brush later means creating a `Brush` subclass and
    adding one entry to `_BRUSH_REGISTRY` - no other code in this
    module changes.

    Args:
        brush_type: One of "hard", "soft", "marker", "highlighter"
            (case-insensitive).

    Returns:
        A new instance of the requested brush.

    Raises:
        ValueError: If `brush_type` isn't a registered brush.
    """
    key = brush_type.strip().lower()
    brush_class = _BRUSH_REGISTRY.get(key)
    if brush_class is None:
        raise ValueError(f"Unknown brush type {brush_type!r}. Available: {sorted(_BRUSH_REGISTRY)}.")
    return brush_class()