"""Offline, deterministic tests for the Asset Optimizer's Block 1 analyzer.

No network access, no GitHub, no NVIDIA, no camera, no pyrender/OpenGL.
Every fixture is a tiny synthetic GLB built locally with `trimesh` and
written to a temporary directory. Only
`apps.reality_painter.optimization.analyzer` is exercised here - the
runtime application, `AssetRegistry`, `AssetRetriever`, `load_glb`, and
`Renderer3D` are never imported or touched.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from apps.reality_painter.optimization.analyzer import (
    AssetAnalysis,
    AssetFileNotFoundError,
    MalformedAssetError,
    UnsupportedAssetFormatError,
    analyze_asset,
)

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {name}")
    else:
        failed += 1
        print(f"FAIL: {name}")


def expect_raises(name, exception_type, func):
    try:
        func()
        check(name, False)
    except exception_type:
        check(name, True)


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    # --- Fixture 1: a single untextured box (1 mesh, no material image) ---
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    plain_path = tmp_path / "plain_box.glb"
    box.export(plain_path, file_type="glb")
    original_bytes = plain_path.read_bytes()

    # 1. Valid simple GLB analyzes successfully.
    result = analyze_asset(plain_path)
    check("analyze_asset returns an AssetAnalysis", isinstance(result, AssetAnalysis))
    check("file_path matches input", result.file_path == str(plain_path))
    check("file_size_bytes matches the file on disk", result.file_size_bytes == plain_path.stat().st_size)

    # 2. Triangle counting (a box has 12 triangles).
    check("triangle_count matches expected box triangle count", result.triangle_count == len(box.faces))
    check("triangle_count is nonzero for real geometry", result.triangle_count > 0)

    # 3. Vertex counting.
    check("vertex_count matches expected box vertex count", result.vertex_count == len(box.vertices))

    # 4. Mesh counting (single-mesh box).
    check("mesh_count is 1 for a single-mesh GLB", result.mesh_count == 1)

    # 5. No-texture asset: material_count/texture_count behavior.
    check("texture_count is 0 for an untextured asset", result.texture_count == 0)
    check("max_texture_width is 0 for an untextured asset", result.max_texture_width == 0)
    check("max_texture_height is 0 for an untextured asset", result.max_texture_height == 0)
    check(
        "estimated_texture_memory_bytes is 0 for an untextured asset",
        result.estimated_texture_memory_bytes == 0,
    )

    # 6. Deterministic, JSON-serializable representation.
    as_dict = result.to_dict()
    check("to_dict() returns a plain dict", isinstance(as_dict, dict))
    check("to_dict() round-trips triangle_count", as_dict["triangle_count"] == result.triangle_count)
    check("to_dict() is deterministic across calls", analyze_asset(plain_path).to_dict() == as_dict)

    # 7. Analyzer never modifies the source file.
    check("source GLB bytes are unchanged after analysis", plain_path.read_bytes() == original_bytes)

    # --- Fixture 2: a textured box (1 material, 1 texture) -----------------
    texture_image = trimesh.visual.color.ColorVisuals()  # placeholder, replaced below
    pil_image = None
    try:
        from PIL import Image

        pil_image = Image.new("RGBA", (64, 32), color=(255, 0, 0, 255))
    except ImportError:
        pil_image = None

    if pil_image is not None:
        material = trimesh.visual.material.SimpleMaterial(image=pil_image)
        textured_box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        uv = np.zeros((len(textured_box.vertices), 2))
        textured_box.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        textured_path = tmp_path / "textured_box.glb"
        textured_box.export(textured_path, file_type="glb")

        textured_result = analyze_asset(textured_path)

        # 8. Textured asset detection.
        check("textured asset reports material_count >= 1", textured_result.material_count >= 1)
        check("textured asset reports texture_count >= 1", textured_result.texture_count >= 1)

        # 9. Texture dimension detection.
        check("max_texture_width matches the fixture texture width", textured_result.max_texture_width == 64)
        check("max_texture_height matches the fixture texture height", textured_result.max_texture_height == 32)

        # 10. Texture memory estimate: width * height * 4 per unique texture.
        expected_bytes = 64 * 32 * 4
        check(
            "estimated_texture_memory_bytes matches width*height*4",
            textured_result.estimated_texture_memory_bytes == expected_bytes,
        )
    else:
        print("NOTE: Pillow unavailable - skipping textured-asset checks (8-10).")

    # --- Fixture 3: multiple meshes / multiple materials --------------------
    mesh_a = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh_b = trimesh.creation.icosphere(radius=0.5)
    multi_scene = trimesh.Scene()
    multi_scene.add_geometry(mesh_a, node_name="box")
    multi_scene.add_geometry(mesh_b, node_name="sphere")
    multi_path = tmp_path / "multi_mesh.glb"
    multi_scene.export(multi_path, file_type="glb")

    multi_result = analyze_asset(multi_path)

    # 11. Multi-mesh counting.
    check("mesh_count is 2 for a two-mesh GLB", multi_result.mesh_count == 2)

    # 12. Triangle count sums across all meshes, not just the first.
    expected_multi_triangles = len(mesh_a.faces) + len(mesh_b.faces)
    check(
        "triangle_count sums triangles across every mesh",
        multi_result.triangle_count == expected_multi_triangles,
    )
    check(
        "vertex_count sums vertices across every mesh",
        multi_result.vertex_count == len(mesh_a.vertices) + len(mesh_b.vertices),
    )

    # --- Missing file --------------------------------------------------------
    # 13. Missing file raises a typed, clean error (never a raw exception).
    expect_raises(
        "missing file raises AssetFileNotFoundError",
        AssetFileNotFoundError,
        lambda: analyze_asset(tmp_path / "does_not_exist.glb"),
    )

    # --- Malformed GLB ---------------------------------------------------
    # 14. Malformed file raises a typed, clean error.
    corrupt_path = tmp_path / "corrupt.glb"
    corrupt_path.write_bytes(b"this is not a real glb file")
    expect_raises(
        "malformed GLB raises MalformedAssetError",
        MalformedAssetError,
        lambda: analyze_asset(corrupt_path),
    )

    # --- Unsupported extension -----------------------------------------
    # 15. Unsupported extension raises a typed, clean error.
    unsupported_path = tmp_path / "model.obj"
    unsupported_path.write_text("dummy")
    expect_raises(
        "unsupported extension raises UnsupportedAssetFormatError",
        UnsupportedAssetFormatError,
        lambda: analyze_asset(unsupported_path),
    )

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
