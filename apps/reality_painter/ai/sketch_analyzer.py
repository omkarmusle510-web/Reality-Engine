"""Sketch analysis for Reality Painter's AI subsystem.

`SketchAnalyzer` turns raw sketch data (canvas pixels) into the
structured `Dict[str, Any]` that `apps.reality_painter.ai.manager`
already expects from anything satisfying its `SketchAnalyzer` Protocol
(`analyze(self, sketch: Any) -> Dict[str, Any]`), and that
`apps.reality_painter.ai.prompt_builder.PromptBuilder` already knows how
to render (see `_build_sketch_section`).

This module is provider-agnostic in the strictest sense: it never
imports or knows about any AI backend, never builds a prompt, never
performs generation, and never touches the network. Its only concern is
local, deterministic image analysis - purely geometric, based on
OpenCV/NumPy, the same dependencies `engine/vision` and
`engine/rendering` already use elsewhere in this project.

Analysis is composed from small, independent components, mirroring the
section-composition pattern already established in `prompt_builder.py`:
each component inspects the sketch and contributes one key to the
result dict, or nothing at all if it has nothing useful to say. A
missing or malformed sketch never raises - `analyze()` simply returns
an empty (or partial) dict.

This intentionally leaves room for future, more sophisticated
components - real object detection, shape classification via a trained
model, OCR, color analysis, semantic segmentation, stroke-direction
estimation, depth estimation, or a preprocessing step - to be dropped
in via `register_component()` (or by swapping a built-in component for
a smarter implementation of the same `SketchAnalysisComponent`
Protocol) without any change to `AIManager`, `PromptBuilder`, or the
components already here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import cv2
import numpy as np

from engine.core.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_BACKGROUND_THRESHOLD = 250  # grayscale value; pixels >= this are background
_DEFAULT_MIN_COMPONENT_AREA = 8.0    # px^2; smaller connected components are noise
_DEFAULT_COMPOSITION_GRID = 3        # 3x3 grid for composition/region density
_DEFAULT_MAX_SHAPE_CANDIDATES = 25   # cap on contours classified per analysis


# --- Configuration ------------------------------------------------------


@dataclass
class SketchAnalyzerConfig:
    """Static, reusable tuning parameters for a `SketchAnalyzer` instance.

    Attributes:
        background_threshold: Grayscale intensity (0-255) at or above
            which a pixel is treated as empty canvas rather than a
            stroke. Assumes a light/white canvas background, consistent
            with Reality Painter's default canvas.
        min_component_area: Minimum connected-component area, in
            pixels, to be counted as a real stroke/object rather than
            noise (e.g. anti-aliasing fragments).
        composition_grid_size: Size of the N x N grid used to measure
            per-region stroke density for the composition section.
        max_shape_candidates: Maximum number of contours classified by
            shape per analysis, to keep analysis time bounded on dense
            sketches.
    """

    background_threshold: int = _DEFAULT_BACKGROUND_THRESHOLD
    min_component_area: float = _DEFAULT_MIN_COMPONENT_AREA
    composition_grid_size: int = _DEFAULT_COMPOSITION_GRID
    max_shape_candidates: int = _DEFAULT_MAX_SHAPE_CANDIDATES


# --- Component contract -----------------------------------------------


@runtime_checkable
class SketchAnalysisComponent(Protocol):
    """A single, independent unit of sketch analysis.

    Each component inspects the same precomputed inputs - the source
    image and its binary stroke mask - and contributes zero or more
    keys to the final analysis dict. Components never depend on each
    other's output and never mutate their inputs, so they can run in
    any order and be added, removed, or replaced independently.
    """

    def analyze(
        self, image: np.ndarray, mask: np.ndarray, config: SketchAnalyzerConfig
    ) -> Dict[str, Any]:
        """Analyzes the sketch and returns this component's contribution.

        Args:
            image: The source sketch image (BGR or grayscale).
            mask: A binary mask (uint8, values 0 or 255) where non-zero
                pixels are stroke/ink and zero pixels are background,
                already computed once for the whole analysis.
            config: The owning `SketchAnalyzer`'s configuration.

        Returns:
            A dict of zero or more result keys. An empty dict means
            this component found nothing to report (e.g. a blank
            canvas) - never raises for that case.
        """
        ...


# --- SketchAnalyzer -----------------------------------------------------


class SketchAnalyzer:
    """Extracts structured, deterministic information from a sketch.

    Satisfies `apps.reality_painter.ai.manager.SketchAnalyzer`
    structurally, so any instance can be passed directly to
    `AIManager(sketch_analyzer=...)`.

    `sketch` is accepted in either of two forms:
        - A `numpy.ndarray` image (BGR or grayscale), e.g. raw canvas
          pixels.
        - Any object exposing an `.image` attribute that is itself such
          an array (e.g. `engine.vision.frame.Frame`), so a caller can
          pass a `Frame`-like object directly without unwrapping it.
    Anything else - `None`, an unrecognized type, or a malformed array -
    is treated as "nothing to analyze": `analyze()` returns `{}` rather
    than raising.

    Analysis itself is delegated to a list of `SketchAnalysisComponent`
    objects, run in registration order and merged into one result dict.
    The default components cover: drawing bounds, canvas utilization,
    empty space, composition (regional density), stroke statistics,
    relative positions, estimated complexity, dominant geometry/shapes,
    and candidate objects (a local connected-component heuristic,
    intended to be replaced by real object detection later without
    changing this class's public surface).
    """

    def __init__(
        self,
        config: Optional[SketchAnalyzerConfig] = None,
        components: Optional[List[SketchAnalysisComponent]] = None,
    ) -> None:
        """Creates a sketch analyzer.

        Args:
            config: Tuning parameters shared by all components. Defaults
                to `SketchAnalyzerConfig()` if omitted.
            components: Explicit component list to use instead of the
                built-in defaults. Most callers should leave this
                `None` and use `register_component()` to extend the
                defaults instead of replacing them.
        """
        self._config = config if config is not None else SketchAnalyzerConfig()
        self._components: List[SketchAnalysisComponent] = (
            list(components)
            if components is not None
            else [
                _BoundsAnalyzer(),
                _CanvasUtilizationAnalyzer(),
                _EmptySpaceAnalyzer(),
                _CompositionAnalyzer(),
                _StrokeStatisticsAnalyzer(),
                _RelativePositionAnalyzer(),
                _ComplexityAnalyzer(),
                _ShapeAnalyzer(),
                _ObjectAnalyzer(),
            ]
        )

    # --- Extensibility --------------------------------------------------

    def register_component(self, component: SketchAnalysisComponent, position: Optional[int] = None) -> None:
        """Adds an analysis component without touching any existing one.

        This is the sanctioned way to extend `SketchAnalyzer` - a future
        capability (real object detection, shape classification via a
        trained model, OCR, color analysis, semantic segmentation,
        stroke-direction estimation, depth estimation, ...) is added by
        implementing `SketchAnalysisComponent` and registering it here,
        never by editing an existing component.

        Args:
            component: Any object satisfying `SketchAnalysisComponent`.
            position: Index to insert at. Appended to the end (i.e.
                runs last) if omitted.
        """
        if position is None:
            self._components.append(component)
        else:
            self._components.insert(position, component)

    # --- Protocol entry point -------------------------------------------

    def analyze(self, sketch: Any) -> Dict[str, Any]:
        """Analyzes `sketch` and returns structured information about it.

        Never raises: an unusable `sketch` yields `{}`, and a component
        that fails is skipped (logged) rather than aborting the whole
        analysis, so one broken or experimental component can never
        take down sketch analysis for every other component.

        Args:
            sketch: A `numpy.ndarray` image, or an object with an
                `.image` attribute that is one (e.g. `Frame`).

        Returns:
            A dict merging every component's contribution. Keys are
            component-defined (see class docstring for the default
            set); may be empty if `sketch` couldn't be read or no
            component found anything to report.
        """
        image = self._extract_image(sketch)
        if image is None:
            return {}

        mask = self._compute_stroke_mask(image)

        result: Dict[str, Any] = {}
        for component in self._components:
            try:
                contribution = component.analyze(image, mask, self._config)
            except Exception:
                logger.exception("Sketch analysis component %s failed; skipping.", type(component).__name__)
                continue
            if contribution:
                result.update(contribution)

        return result

    # --- Input normalization ---------------------------------------------

    def _extract_image(self, sketch: Any) -> Optional[np.ndarray]:
        """Resolves `sketch` to a usable image array, or `None`.

        Accepts a raw `numpy.ndarray` directly, or unwraps a single
        `.image` attribute (e.g. `Frame.image`). Rejects anything else,
        including arrays with no visible content (zero-sized
        dimensions), without raising.

        Args:
            sketch: The opaque sketch value passed to `analyze()`.

        Returns:
            A 2D or 3D `numpy.ndarray`, or `None` if `sketch` could not
            be resolved to one.
        """
        candidate = sketch
        if not isinstance(candidate, np.ndarray):
            candidate = getattr(sketch, "image", None)

        if not isinstance(candidate, np.ndarray):
            return None
        if candidate.ndim not in (2, 3) or candidate.size == 0:
            return None
        return candidate

    def _compute_stroke_mask(self, image: np.ndarray) -> np.ndarray:
        """Computes a binary stroke mask from `image`.

        Converts to grayscale (if the image has color channels) and
        thresholds against `config.background_threshold`: pixels darker
        than the threshold are considered ink/stroke, everything else
        is background. This assumes a light/white canvas, consistent
        with Reality Painter's default canvas background.

        Args:
            image: The source sketch image.

        Returns:
            A uint8 mask, same height/width as `image`, with stroke
            pixels set to 255 and background pixels set to 0.
        """
        if image.ndim == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] >= 3 else image[:, :, 0]
        else:
            grayscale = image

        _, mask = cv2.threshold(
            grayscale, self._config.background_threshold, 255, cv2.THRESH_BINARY_INV
        )
        return mask


# --- Built-in components ------------------------------------------------
#
# Each component below is intentionally small and single-purpose,
# following the same "one section, one method/class" composition
# pattern already used in `prompt_builder.py`. None of them import or
# reference AI providers, prompts, or anything outside local image
# analysis.


class _BoundsAnalyzer:
    """Reports the bounding box of all stroke pixels ("drawing bounds")."""

    def analyze(self, image: np.ndarray, mask: np.ndarray, config: SketchAnalyzerConfig) -> Dict[str, Any]:
        coordinates = cv2.findNonZero(mask)
        if coordinates is None:
            return {"bounds": None}

        x, y, width, height = cv2.boundingRect(coordinates)
        return {
            "bounds": {
                "x_min": x,
                "y_min": y,
                "x_max": x + width,
                "y_max": y + height,
                "width": width,
                "height": height,
            }
        }


class _CanvasUtilizationAnalyzer:
    """Reports the fraction of canvas pixels that are stroke, not background."""

    def analyze(self, image: np.ndarray, mask: np.ndarray, config: SketchAnalyzerConfig) -> Dict[str, Any]:
        total_pixels = mask.shape[0] * mask.shape[1]
        if total_pixels == 0:
            return {"canvas_utilization": 0.0}

        stroke_pixels = int(np.count_nonzero(mask))
        return {"canvas_utilization": round(stroke_pixels / total_pixels, 4)}


class _EmptySpaceAnalyzer:
    """Reports the fraction of the canvas that remains empty."""

    def analyze(self, image: np.ndarray, mask: np.ndarray, config: SketchAnalyzerConfig) -> Dict[str, Any]:
        total_pixels = mask.shape[0] * mask.shape[1]
        if total_pixels == 0:
            return {"empty_space_ratio": 1.0}

        stroke_pixels = int(np.count_nonzero(mask))
        return {"empty_space_ratio": round(1.0 - (stroke_pixels / total_pixels), 4)}


class _CompositionAnalyzer:
    """Reports stroke density per cell of an N x N grid ("composition").

    Grid size is `config.composition_grid_size`. Cells are labeled by
    row/column index rather than named regions (e.g. "top_left"), so
    this stays correct regardless of grid size.
    """

    def analyze(self, image: np.ndarray, mask: np.ndarray, config: SketchAnalyzerConfig) -> Dict[str, Any]:
        grid_size = max(1, config.composition_grid_size)
        height, width = mask.shape[:2]
        if height == 0 or width == 0:
            return {"composition": {}}

        cell_height = max(1, height // grid_size)
        cell_width = max(1, width // grid_size)

        density: Dict[str, float] = {}
        for row in range(grid_size):
            for col in range(grid_size):
                y1, y2 = row * cell_height, height if row == grid_size - 1 else (row + 1) * cell_height
                x1, x2 = col * cell_width, width if col == grid_size - 1 else (col + 1) * cell_width
                cell = mask[y1:y2, x1:x2]
                cell_area = cell.shape[0] * cell.shape[1]
                cell_density = float(np.count_nonzero(cell)) / cell_area if cell_area else 0.0
                density[f"row{row}_col{col}"] = round(cell_density, 4)

        return {"composition": density}


class _StrokeStatisticsAnalyzer:
    """Reports aggregate statistics about distinct stroke components."""

    def analyze(self, image: np.ndarray, mask: np.ndarray, config: SketchAnalyzerConfig) -> Dict[str, Any]:
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # Label 0 is always the background component; skip it.
        areas = [
            float(stats[label, cv2.CC_STAT_AREA])
            for label in range(1, component_count)
            if stats[label, cv2.CC_STAT_AREA] >= config.min_component_area
        ]

        total_stroke_pixels = int(np.count_nonzero(mask))
        average_area = round(sum(areas) / len(areas), 2) if areas else 0.0

        return {
            "stroke_statistics": {
                "stroke_count": len(areas),
                "total_stroke_pixels": total_stroke_pixels,
                "average_component_area": average_area,
            }
        }


class _RelativePositionAnalyzer:
    """Reports normalized centroid positions of distinct stroke components."""

    def analyze(self, image: np.ndarray, mask: np.ndarray, config: SketchAnalyzerConfig) -> Dict[str, Any]:
        height, width = mask.shape[:2]
        if height == 0 or width == 0:
            return {"relative_positions": []}

        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        positions: List[Dict[str, float]] = []
        for label in range(1, component_count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < config.min_component_area:
                continue
            centroid_x, centroid_y = centroids[label]
            positions.append(
                {
                    "x": round(float(centroid_x) / width, 4),
                    "y": round(float(centroid_y) / height, 4),
                    "area": area,
                }
            )

        return {"relative_positions": positions}


class _ComplexityAnalyzer:
    """Estimates overall drawing complexity from edge density."""

    _LOW_THRESHOLD = 0.02
    _MEDIUM_THRESHOLD = 0.08

    def analyze(self, image: np.ndarray, mask: np.ndarray, config: SketchAnalyzerConfig) -> Dict[str, Any]:
        edges = cv2.Canny(mask, 50, 150)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = float(np.count_nonzero(edges)) / total_pixels if total_pixels else 0.0

        if edge_density < self._LOW_THRESHOLD:
            level = "low"
        elif edge_density < self._MEDIUM_THRESHOLD:
            level = "medium"
        else:
            level = "high"

        return {
            "complexity": {
                "level": level,
                "edge_density": round(edge_density, 4),
            }
        }


class _ShapeAnalyzer:
    """Classifies contours into coarse shape categories ("dominant geometry").

    A local, purely geometric heuristic (vertex count via
    `cv2.approxPolyDP` plus circularity) - not a trained classifier.
    Intended to be replaced or supplemented by a real shape-
    classification model later; nothing outside this component depends
    on the heuristic used here.
    """

    def analyze(self, image: np.ndarray, mask: np.ndarray, config: SketchAnalyzerConfig) -> Dict[str, Any]:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = sorted(contours, key=cv2.contourArea, reverse=True)[: config.max_shape_candidates]

        tally: Dict[str, int] = {}
        for contour in candidates:
            area = cv2.contourArea(contour)
            if area < config.min_component_area:
                continue
            shape = self._classify_contour(contour, area)
            tally[shape] = tally.get(shape, 0) + 1

        dominant_shape = max(tally, key=tally.get) if tally else None

        return {
            "shapes": tally,
            "dominant_geometry": dominant_shape,
        }

    def _classify_contour(self, contour: np.ndarray, area: float) -> str:
        """Classifies a single contour as line/triangle/rectangle/circle/polygon."""
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return "line"

        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        vertex_count = len(approx)

        _, radius = cv2.minEnclosingCircle(contour)
        circularity = area / (np.pi * radius * radius) if radius > 0 else 0.0

        if vertex_count <= 2:
            return "line"
        if circularity > 0.75:
            return "circle"
        if vertex_count == 3:
            return "triangle"
        if vertex_count == 4:
            return "rectangle"
        return "polygon"


class _ObjectAnalyzer:
    """Reports candidate "objects" as bounded, sufficiently large stroke clusters.

    This is a local connected-component heuristic, not semantic object
    detection: each sufficiently large, disconnected cluster of strokes
    is reported as one candidate object with its bounding box and area.
    It exists so `AIManager`/`PromptBuilder` callers have an "objects"
    key to consume today, and is designed to be swapped out for a real
    object-detection component (see `register_component`) without any
    change to the result schema's key name.
    """

    _MIN_OBJECT_AREA_FACTOR = 4.0  # objects must be notably larger than plain noise

    def analyze(self, image: np.ndarray, mask: np.ndarray, config: SketchAnalyzerConfig) -> Dict[str, Any]:
        height, width = mask.shape[:2]
        min_object_area = config.min_component_area * self._MIN_OBJECT_AREA_FACTOR

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        objects: List[Dict[str, Any]] = []
        for label in range(1, component_count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < min_object_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            objects.append(
                {
                    "bounds": {"x": x, "y": y, "width": w, "height": h},
                    "area": area,
                }
            )

        return {"objects": objects}