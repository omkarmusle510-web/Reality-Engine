"""TEMPORARY DIAGNOSTIC SCRIPT - not part of the regular test suite.

Read-only inspection of the real, already-cached ABeautifulGame GLB on
disk. Locates the file automatically (no hardcoded, possibly-wrong
path), loads it through the EXISTING `engine.scene.loader.load_glb()`
path, and reports geometry/material/texture/Draco complexity metrics.

This script:
    - never renders anything (no Renderer3D, no pyrender import),
    - never writes, moves, deletes, or modifies any file,
    - never touches `registry.json` or any other production file,
    - never installs or downloads anything.

Run from the repository root:
    python tests/diagnose_asset_complexity.py
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List

from engine.scene.loader import ModelLoadError, load_glb

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIRS = [
    _REPO_ROOT / "apps" / "reality_painter" / "assets" / "cache",
    _REPO_ROOT / "assets_cache",
]
_MODEL_EXTENSIONS = ("*.glb", "*.gltf")
_NAME_HINT = "beautifulgame"  # case/space-insensitive match preference


def _find_candidate_files() -> List[Path]:
    """Searches the known cache directories for local .glb/.gltf files.

    Returns:
        Every matching file found under any existing cache directory.
        Never raises for a missing directory - an absent cache dir is
        simply skipped.
    """
    candidates: List[Path] = []
    for cache_dir in _CACHE_DIRS:
        if not cache_dir.is_dir():
            continue
        for pattern in _MODEL_EXTENSIONS:
            candidates.extend(cache_dir.rglob(pattern))
    return candidates


def _select_target(candidates: List[Path]) -> Path:
    """Picks the best candidate: a name matching ABeautifulGame, else the largest file."""
    for path in candidates:
        normalized = path.stem.lower().replace(" ", "").replace("_", "").replace("-", "")
        if _NAME_HINT in normalized:
            return path
    return max(candidates, key=lambda p: p.stat().st_size)


def _detect_draco(path: Path) -> bool:
    """Best-effort, parse-free Draco detection: scans raw file bytes for the extension marker.

    Works for both .glb (the JSON chunk is stored uncompressed even
    when mesh data is Draco-compressed) and .gltf (plain JSON text) -
    no glTF/JSON parsing is required, so this can never fail on a
    malformed or partially-unsupported file.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return False
    return b"KHR_draco_mesh_compression" in data


def _collect_materials_and_textures(trimesh_scene):
    """Extracts unique materials and unique texture images from a trimesh.Scene.

    Args:
        trimesh_scene: A `trimesh.Scene` (what `load_glb` always
            produces, since it loads via `force="scene"`).

    Returns:
        `(materials, textures)` - lists of unique material and unique
        PIL.Image texture objects referenced by any geometry, deduped
        by object identity.
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

    for geometry in trimesh_scene.geometry.values():
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


def main() -> None:
    print("ASSET COMPLEXITY REPORT")
    print("-" * 24)

    candidates = _find_candidate_files()
    if not candidates:
        print("No cached .glb/.gltf files found under:")
        for cache_dir in _CACHE_DIRS:
            print(f"  - {cache_dir}")
        print("Nothing to inspect. No files were modified.")
        return

    target_path = _select_target(candidates)
    if len(candidates) > 1:
        print(f"Found {len(candidates)} cached model file(s); selected: {target_path.name}")

    file_size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f"1. GLB file path: {target_path}")
    print(f"2. File size: {file_size_mb:.2f} MB")

    warnings_caught: List[str] = []
    load_error: str = ""
    scene_object = None

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            scene_object = load_glb(target_path, name="diagnostic_target")
        except ModelLoadError as exc:
            load_error = str(exc)
        warnings_caught = [str(w.message) for w in captured]

    if scene_object is None:
        print(f"3. SceneObjects returned by load_glb(): 0 (load failed: {load_error})")
        print("Remaining metrics unavailable - load_glb() did not succeed.")
        if warnings_caught:
            print("\n11. Loader warnings/errors:")
            for message in warnings_caught:
                print(f"  - {message}")
        else:
            print(f"\n11. Loader warnings/errors:\n  - {load_error}")
        return

    print("3. SceneObjects returned by load_glb(): 1")

    mesh = scene_object.mesh
    if hasattr(mesh, "geometry"):  # trimesh.Scene (always true for load_glb's force="scene")
        geometries = list(mesh.geometry.values())
    else:  # defensive fallback if a future loader ever returns a bare Trimesh
        geometries = [mesh]

    total_vertices = sum(len(g.vertices) for g in geometries if hasattr(g, "vertices"))
    total_faces = sum(len(g.faces) for g in geometries if hasattr(g, "faces"))
    materials, textures = _collect_materials_and_textures(mesh) if hasattr(mesh, "geometry") else ([], [])

    print(f"4. Total vertices: {total_vertices:,}")
    print(f"5. Total triangles/faces: {total_faces:,}")
    print(f"6. Number of geometries: {len(geometries)}")
    print(f"7. Number of materials: {len(materials)}")
    print(f"8. Number of textures/images: {len(textures)}")

    if textures:
        print("9. Texture dimensions:")
        for index, image in enumerate(textures):
            size = getattr(image, "size", None)
            print(f"  - texture[{index}]: {size if size is not None else 'unknown'}")
    else:
        print("9. Texture dimensions: none found")

    draco_detected = _detect_draco(target_path)
    print(f"10. Draco-related data detected: {'YES' if draco_detected else 'NO'}")

    if warnings_caught:
        print("11. Loader warnings/errors:")
        for message in warnings_caught:
            print(f"  - {message}")
    else:
        print("11. Loader warnings/errors: none")


if __name__ == "__main__":
    main()