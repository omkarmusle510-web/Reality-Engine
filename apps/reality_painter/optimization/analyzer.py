"""Offline GLB/GLTF performance analysis for Reality Painter's Asset Optimizer.

`analyze_asset()` is a read-only inspection step: given a path to a
local `.glb`/`.gltf` file, it loads the file with `trimesh` (the same
library dependency already used elsewhere in this repository, e.g.
`engine.scene.loader.load_glb` and
`tests/diagnose_asset_complexity.py`) and reports deterministic
geometry/material/texture metrics as an `AssetAnalysis`.

This module is intentionally standalone:
    - It never imports `engine.scene.loader`, `engine.rendering.renderer`,
      `apps.reality_painter.assets.registry.AssetRegistry`, or
      `apps.reality_painter.assets.retriever.AssetRetriever`.
    - It never downloads, retrieves, or discovers assets - `path` must
      already point at a file on disk.
    - It never mutates, re-saves, re-exports, or otherwise touches the
      source file - only `Path.stat()`/`Path.read_bytes()`-style reads
      and `trimesh.load()` occur.
    - It contains no optimization, LOD, compression, or hardware-
      detection logic - see the package docstring for scope.

All expected failure modes (missing file, malformed/corrupt file,
unreadable geometry) are translated into the typed exceptions below -
no raw `trimesh`/`OSError`/etc. exception is allowed to escape
`analyze_asset()`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Union

_SUPPORTED_EXTENSIONS = (".glb", ".gltf")

# Bytes per texel for an ordinary, uncompressed RGBA8 texture. This is
# an ESTIMATE of raw pixel memory (width * height * 4) - it does not
# model GPU-specific allocation, mipmaps, alignment/padding, or any
# compressed texture format (KTX2, Draco, etc. are out of scope for
# this block - see the module docstring).
_BYTES_PER_RGBA8_TEXEL = 4


class AssetAnalysisError(Exception):
    """Base class for errors raised by the asset analyzer's public API."""


class AssetFileNotFoundError(AssetAnalysisError):
    """Raised when the file at the given path does not exist."""


class UnsupportedAssetFormatError(AssetAnalysisError):
    """Raised when the file's extension is not a supported 3D format."""


class MalformedAssetError(AssetAnalysisError):
    """Raised when the file exists but cannot be parsed as a valid model."""


@dataclass(frozen=True)
class AssetAnalysis:
    """A deterministic performance report for a single local GLB/GLTF file.

    Every field is derived purely from the file's own contents - two
    calls to `analyze_asset()` on the same, unmodified file always
    produce an identical `AssetAnalysis`.

    Attributes:
        file_path: The analyzed file's path, as a string.
        file_size_bytes: Size of the file on disk, in bytes.
        triangle_count: Total triangle count summed across every mesh
            geometry in the file (see module docstring re: correctness
            - every primitive/geometry is counted, not just the first).
        vertex_count: Total vertex count summed across every mesh
            geometry in the file.
        mesh_count: Number of distinct mesh geometries in the file.
        material_count: Number of distinct materials referenced by any
            mesh in the file (deduplicated by object identity).
        texture_count: Number of distinct texture images referenced by
            any material in the file (deduplicated by object identity).
        max_texture_width: Largest texture width found, in pixels, or 0
            if the file has no textures.
        max_texture_height: Largest texture height found, in pixels, or
            0 if the file has no textures.
        estimated_texture_memory_bytes: Estimated *uncompressed* RGBA8
            memory footprint, summed across every distinct texture
            (`width * height * 4` per texture - see module docstring).
            This is explicitly an estimate, not an exact GPU allocation
            figure.
    """

    file_path: str
    file_size_bytes: int
    triangle_count: int
    vertex_count: int
    mesh_count: int
    material_count: int
    texture_count: int
    max_texture_width: int
    max_texture_height: int
    estimated_texture_memory_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        """Returns a plain, JSON-serializable dict of this analysis.

        Field order and values are deterministic for a given input
        file, making this suitable for logging or writing to disk as
        JSON without any further transformation.
        """
        return asdict(self)


def analyze_asset(path: Union[str, Path]) -> AssetAnalysis:
    """Analyzes a local `.glb`/`.gltf` file and returns its performance metrics.

    Read-only end to end: the source file is only ever opened for
    reading (via `Path.stat()` and `trimesh.load()`) and is never
    modified, re-saved, or re-exported.

    Args:
        path: Path to an existing local `.glb`/`.gltf` file.

    Returns:
        A deterministic `AssetAnalysis` describing the file.

    Raises:
        AssetFileNotFoundError: If `path` does not exist.
        UnsupportedAssetFormatError: If `path`'s extension is not
            `.glb`/`.gltf`.
        MalformedAssetError: If the file cannot be parsed as a valid
            model, or contains no usable geometry.
    """
    model_path = Path(path)

    if model_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise UnsupportedAssetFormatError(f"Unsupported asset format {model_path.suffix!r}: {model_path}.")
    if not model_path.is_file():
        raise AssetFileNotFoundError(f"Asset file not found: {model_path}.")

    file_size_bytes = model_path.stat().st_size

    try:
        import trimesh
    except Exception as exc:
        raise AssetAnalysisError(f"trimesh is unavailable: {exc}") from exc

    try:
        loaded = trimesh.load(model_path, force="scene")
    except Exception as exc:
        raise MalformedAssetError(f"Failed to parse asset {model_path}: {exc}") from exc

    if loaded is None:
        raise MalformedAssetError(f"Asset file loaded but produced no scene: {model_path}.")

    geometries = list(loaded.geometry.values()) if hasattr(loaded, "geometry") else [loaded]

    try:
        triangle_count = sum(len(geometry.faces) for geometry in geometries if hasattr(geometry, "faces"))
        vertex_count = sum(len(geometry.vertices) for geometry in geometries if hasattr(geometry, "vertices"))
    except Exception as exc:
        raise MalformedAssetError(f"Failed to inspect geometry for asset {model_path}: {exc}") from exc

    materials, textures = _collect_materials_and_textures(geometries)

    max_width = 0
    max_height = 0
    estimated_texture_memory_bytes = 0
    for texture in textures:
        size = getattr(texture, "size", None)
        if not size or len(size) < 2:
            continue
        width, height = int(size[0]), int(size[1])
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        estimated_texture_memory_bytes += width * height * _BYTES_PER_RGBA8_TEXEL

    return AssetAnalysis(
        file_path=str(model_path),
        file_size_bytes=file_size_bytes,
        triangle_count=triangle_count,
        vertex_count=vertex_count,
        mesh_count=len(geometries),
        material_count=len(materials),
        texture_count=len(textures),
        max_texture_width=max_width,
        max_texture_height=max_height,
        estimated_texture_memory_bytes=estimated_texture_memory_bytes,
    )


def _collect_materials_and_textures(geometries):
    """Extracts unique materials and unique texture images from mesh geometries.

    Deduplicates by object identity (the same pattern already used by
    `tests/diagnose_asset_complexity.py`), so a material or texture
    shared across several meshes is only counted once.

    Args:
        geometries: The scene's mesh geometry objects.

    Returns:
        A `(materials, textures)` tuple of lists, each containing
        unique objects in first-seen order.
    """
    materials = []
    textures = []
    seen_material_ids = set()
    seen_texture_ids = set()

    texture_attrs = (
        "baseColorTexture",
        "metallicRoughnessTexture",
        "normalTexture",
        "occlusionTexture",
        "emissiveTexture",
        "image",  # SimpleMaterial fallback
    )

    for geometry in geometries:
        visual = getattr(geometry, "visual", None)
        material = getattr(visual, "material", None)
        if material is None:
            continue
        if id(material) not in seen_material_ids:
            seen_material_ids.add(id(material))
            materials.append(material)

        for attr in texture_attrs:
            image = getattr(material, attr, None)
            if image is not None and id(image) not in seen_texture_ids:
                seen_texture_ids.add(id(image))
                textures.append(image)

    return materials, textures
