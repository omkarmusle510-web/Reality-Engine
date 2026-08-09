"""3D scene object representation for Reality Engine.

`SceneObject` is the minimal representation of a single loaded 3D asset
placed in a `Scene`: geometry plus a transform (position, rotation,
scale). This is the first phase of the engine's 3D capability - no
hierarchy, no materials system, no animation, no physics. A loader
(see `engine/scene/loader.py`) is responsible for producing
`SceneObject` instances from a local model file; this module only
defines the data shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple

Vec3 = Tuple[float, float, float]


@dataclass
class Transform:
    """A minimal position/rotation/scale transform.

    Attributes:
        position: (x, y, z) world position.
        rotation: (x, y, z) Euler rotation in radians (XYZ order).
        scale: (x, y, z) per-axis scale.
    """

    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Vec3 = (0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)


@dataclass
class SceneObject:
    """A single 3D object in a `Scene`: geometry plus a transform.

    Attributes:
        mesh: The loaded geometry, as returned by
            `engine.scene.loader.load_glb` (a `trimesh.Trimesh` or
            `trimesh.Scene`). Opaque to `Scene` - only the loader that
            produced it and the renderer that consumes it need to know
            its shape.
        transform: This object's position/rotation/scale in the scene.
            Defaults to the identity transform (origin, no rotation,
            unit scale), which `Renderer3D`'s default camera placement
            is deliberately chosen to keep in view.
        name: Human-readable identifier, used as this object's key when
            added to a `Scene`.
    """

    mesh: Any
    transform: Transform = field(default_factory=Transform)
    name: str = "scene_object"
