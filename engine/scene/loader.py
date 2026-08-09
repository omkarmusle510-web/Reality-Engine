"""GLB/GLTF loading for Reality Engine's 3D scene system.

Converts a local `.glb`/`.gltf` file into a `SceneObject`. This module
never performs network access or file retrieval - it only operates on
a path to a file that already exists on disk (e.g. produced by
`apps.reality_painter.assets.retriever.AssetRetriever`, Phase 12C).
Retrieval remains that module's responsibility; this module never
imports it, keeping the engine independent of Reality Painter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from engine.core.logger import get_logger
from engine.scene.objects import SceneObject, Transform

logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS = (".glb", ".gltf")


class ModelLoadError(Exception):
    """Raised when a local GLB/GLTF file cannot be loaded into a `SceneObject`."""


def load_glb(path: Union[str, Path], name: str = "scene_object") -> SceneObject:
    """Loads a local `.glb`/`.gltf` file into a `SceneObject`.

    Uses `trimesh` to parse the file. Never downloads or fetches
    anything - `path` must already exist on disk (Phase 12C's
    `AssetRetriever` is what puts it there).

    Args:
        path: Path to an existing local `.glb`/`.gltf` file.
        name: Name to give the resulting `SceneObject`, used as its key
            if added to a `Scene`.

    Returns:
        A `SceneObject` wrapping the loaded geometry, with the default
        identity `Transform` (origin, no rotation, unit scale).

    Raises:
        ModelLoadError: If `path` has an unsupported extension, does
            not exist, or fails to parse as a valid GLB/GLTF model.
    """
    model_path = Path(path)

    if model_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ModelLoadError(f"Unsupported model format {model_path.suffix!r}: {model_path}.")
    if not model_path.is_file():
        raise ModelLoadError(f"Model file not found: {model_path}.")

    import trimesh

    try:
        mesh = trimesh.load(model_path, force="scene")
    except Exception as exc:
        raise ModelLoadError(f"Failed to load model {model_path}: {exc}") from exc

    if mesh is None or (hasattr(mesh, "is_empty") and mesh.is_empty):
        raise ModelLoadError(f"Model file loaded but contained no geometry: {model_path}.")

    logger.info("Loaded 3D model '%s' from %s.", name, model_path)
    return SceneObject(mesh=mesh, transform=Transform(), name=name)
