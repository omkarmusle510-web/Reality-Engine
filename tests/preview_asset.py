"""TEMPORARY GLB asset preview/compatibility diagnostic.

Usage from the repository root:

    python tests\preview_asset.py "C:\path\to\candidate.glb"

This file is intentionally standalone and temporary.

It:
- loads the supplied GLB through the production load_glb()
- reports basic geometry complexity
- uses the production Renderer3D
- uses the production Camera
- composites the model over the live webcam
- rotates the model so its 3D nature can be inspected
- reports FPS and render time

It does NOT:
- modify registry.json
- modify the asset cache
- modify app.py
- download anything
- call NVIDIA
- register the asset
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from engine.rendering.renderer import RenderError, Renderer3D, composite_rgba_onto
from engine.scene.loader import ModelLoadError, load_glb
from engine.scene.objects import Transform
from engine.scene.scene import Scene
from engine.vision.camera import Camera


ROTATION_SPEED = 1.0
WINDOW_NAME = "Temporary GLB Preview - ESC to quit"


def inspect_geometry(scene_object) -> tuple[int, int, int]:
    """Return total vertices, faces and geometry count."""
    mesh = scene_object.mesh

    if hasattr(mesh, "geometry"):
        geometries = list(mesh.geometry.values())
    else:
        geometries = [mesh]

    vertices = sum(
        len(g.vertices)
        for g in geometries
        if hasattr(g, "vertices")
    )

    faces = sum(
        len(g.faces)
        for g in geometries
        if hasattr(g, "faces")
    )

    return vertices, faces, len(geometries)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python tests\\preview_asset.py "
            "\"C:\\path\\to\\candidate.glb\""
        )

    glb_path = Path(sys.argv[1]).resolve()

    if not glb_path.is_file():
        raise SystemExit(f"GLB not found: {glb_path}")

    print("=" * 60)
    print("TEMPORARY REALITY PAINTER GLB PREVIEW")
    print("=" * 60)
    print(f"File: {glb_path}")

    size_mb = glb_path.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

    # ---------------------------------------------------------
    # Load through the REAL production loader.
    # ---------------------------------------------------------

    print("\nLoading with production load_glb()...")

    try:
        scene_object = load_glb(
            glb_path,
            name="temporary_preview",
        )
    except ModelLoadError as exc:
        print(f"LOAD FAILED: {exc}")
        return
    except Exception as exc:
        print(f"LOAD FAILED: {type(exc).__name__}: {exc}")
        return

    vertices, faces, geometry_count = inspect_geometry(scene_object)

    print("LOAD SUCCESS")
    print(f"Vertices:   {vertices:,}")
    print(f"Triangles:  {faces:,}")
    print(f"Geometries: {geometry_count:,}")

    # ---------------------------------------------------------
    # Build production Scene.
    # ---------------------------------------------------------

    scene = Scene()
    scene.add(scene_object)

    try:
        renderer = Renderer3D()
    except RenderError as exc:
        print(f"Renderer initialization failed: {exc}")
        return
    except Exception as exc:
        print(
            f"Renderer initialization failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    camera = Camera()

    try:
        camera.open()
    except Exception as exc:
        renderer.close()
        print(f"Camera initialization failed: {exc}")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("\nPreview started.")
    print("Controls:")
    print("  R     toggle rotation")
    print("  +     increase scale")
    print("  -     decrease scale")
    print("  ESC   exit")
    print()

    rotation_enabled = True
    scale = 1.0

    angle = 0.0
    previous_time = time.perf_counter()

    frame_count = 0
    fps_start = time.perf_counter()
    displayed_fps = 0.0

    try:
        while True:
            frame = camera.read()

            if frame is None:
                continue

            now = time.perf_counter()
            delta = now - previous_time
            previous_time = now

            if rotation_enabled:
                angle += delta * ROTATION_SPEED

            # Keep the model centered and continuously rotating.
            scene_object.transform = Transform(
                position=(0.0, 0.0, 0.0),
                rotation=(0.0, angle, 0.0),
                scale=(scale, scale, scale),
            )

            render_start = time.perf_counter()

            try:
                rendered = renderer.render(scene)
            except Exception as exc:
                print(
                    f"\nRENDER FAILED: "
                    f"{type(exc).__name__}: {exc}"
                )
                break

            render_time_ms = (
                time.perf_counter() - render_start
            ) * 1000.0

            composite_rgba_onto(
                frame.image,
                rendered,
            )

            frame_count += 1

            elapsed = time.perf_counter() - fps_start

            if elapsed >= 1.0:
                displayed_fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.perf_counter()

            cv2.putText(
                frame.image,
                f"FPS: {displayed_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame.image,
                f"Render: {render_time_ms:.1f} ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame.image,
                f"Triangles: {faces:,}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, frame.image)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            elif key in (ord("r"), ord("R")):
                rotation_enabled = not rotation_enabled

            elif key in (ord("+"), ord("=")):
                scale *= 1.1

            elif key in (ord("-"), ord("_")):
                scale /= 1.1

    finally:
        cv2.destroyAllWindows()
        camera.release()
        renderer.close()

    print("\n" + "=" * 60)
    print("PREVIEW FINISHED")
    print("=" * 60)
    print(f"Final measured FPS: {displayed_fps:.1f}")
    print("No production files were modified.")


if __name__ == "__main__":
    main()