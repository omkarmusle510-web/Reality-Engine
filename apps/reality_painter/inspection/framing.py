"""Deterministic model normalization for Reality Painter's 3D inspection view.

Computes a loaded mesh's own bounding box and derives a uniform
scale + centering `Transform` so any asset displays at a consistent,
sensible size regardless of its original modeling scale/units. This
never modifies `engine.scene.loader.load_glb`'s raw-loader contract -
it only computes a `Transform` to apply to the resulting `SceneObject`
afterward, at the same integration point
`apps.reality_painter.inspection.controller.InspectionController`
already uses for Block 8's optimizer.

Nothing here is asset-specific: the target size is a single constant
applied identically to any mesh.
"""

from __future__ import annotations

from typing import Any

from engine.core.logger import get_logger
from engine.scene.objects import Transform

logger = get_logger(__name__)

#: Target size, in world units, for a normalized asset's longest
#: bounding-box dimension. Not tuned for any specific asset - the same
#: constant applies to every loaded model.
_TARGET_DISPLAY_SIZE = 2.0


def normalize_transform(mesh: Any, target_size: float = _TARGET_DISPLAY_SIZE) -> Transform:
    """Computes a centering + uniform-scale `Transform` for `mesh`.

    Deterministic: the same mesh geometry always produces the same
    `Transform`. Falls back to the identity `Transform` if bounds
    cannot be determined (e.g. an empty, malformed, or unbounded mesh)
    - never raises, and never fabricates a plausible-looking scale.

    Args:
        mesh: The loaded geometry, as returned by
            `engine.scene.loader.load_glb` (a `trimesh.Trimesh` or
            `trimesh.Scene`) - anything exposing a `.bounds` property
            shaped `(2, 3)` (min corner, max corner).
        target_size: Desired size, in world units, of the mesh's
            longest bounding-box dimension after scaling.

    Returns:
        A `Transform` that centers `mesh` at the origin and scales it
        uniformly so its longest dimension equals `target_size`.
    """
    try:
        bounds = mesh.bounds
        if bounds is None:
            return Transform()

        min_corner, max_corner = bounds[0], bounds[1]
        extents = [float(max_corner[i]) - float(min_corner[i]) for i in range(3)]
        max_dim = max(extents)
        if not max_dim or max_dim <= 0:
            return Transform()

        center = [(float(min_corner[i]) + float(max_corner[i])) / 2.0 for i in range(3)]
        scale_factor = target_size / max_dim
        position = tuple(-c * scale_factor for c in center)

        return Transform(position=position, rotation=(0.0, 0.0, 0.0), scale=(scale_factor,) * 3)
    except Exception:
        logger.warning("Model normalization failed; using identity transform.")
        return Transform()
