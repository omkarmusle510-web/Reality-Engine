"""Minimal offscreen 3D renderer for Reality Engine.

`Renderer3D` renders a `Scene`'s flat collection of `SceneObject`s to
an RGBA image using `pyrender`'s offscreen rendering, so the result can
be composited onto the existing 2D OpenCV camera frame (see
`composite_rgba_onto`) rather than displayed in a separate window. This
is the engine's first 3D rendering implementation: one deterministic
perspective camera, one directional light, no materials system, no
shadow tuning, no scene hierarchy.
"""

from __future__ import annotations

import numpy as np

# `pyrender`'s offscreen backend defaults to a windowed (pyglet) GL
# context, which requires a display server and fails in headless
# environments (e.g. CI, a server with no X/Wayland session). EGL
# provides a display-less GL context and is what makes offscreen
# rendering actually offscreen; this is set before `pyrender` is first
# imported anywhere in the process, and only if the caller hasn't
# already chosen a platform (e.g. "osmesa") themselves

from engine.core.logger import get_logger
from engine.scene.scene import Scene

logger = get_logger(__name__)

_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 480
_DEFAULT_YFOV_RADIANS = 1.0  # ~57 degrees
_DEFAULT_CAMERA_DISTANCE = 3.0
_DEFAULT_LIGHT_INTENSITY = 3.0


class RenderError(Exception):
    """Raised when the 3D backend is unavailable or a render call fails."""


def _flatten_trimesh(mesh):
    """Reduces a `trimesh.Scene` (or a plain `Trimesh`) to one `Trimesh`.

    `engine.scene.loader.load_glb` always loads via `force="scene"`,
    so `mesh` is a `trimesh.Scene` even for a single-mesh GLB;
    `pyrender.Mesh.from_trimesh` expects a single `Trimesh`, so a
    multi-node scene is flattened into one combined mesh.
    """
    if hasattr(mesh, "geometry"):  # trimesh.Scene
        return mesh.dump(concatenate=True)
    return mesh


def _build_object_pose(obj) -> np.ndarray:
    """Builds a 4x4 world transform matrix from a `SceneObject`'s transform."""
    import trimesh

    translation = trimesh.transformations.translation_matrix(obj.transform.position)
    rotation = trimesh.transformations.euler_matrix(*obj.transform.rotation)
    scale = np.eye(4)
    scale[0, 0], scale[1, 1], scale[2, 2] = obj.transform.scale
    return translation @ rotation @ scale


class Renderer3D:
    """Renders a `Scene` to an offscreen RGBA image via `pyrender`.

    Owns a single reusable offscreen rendering context sized to
    `width`/`height`, created once at construction and reused across
    `render()` calls rather than recreated per frame.
    """

    def __init__(
        self,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        camera_distance: float = _DEFAULT_CAMERA_DISTANCE,
    ) -> None:
        """Creates a renderer bound to a fixed output size.

        Args:
            width: Rendered image width, in pixels.
            height: Rendered image height, in pixels.
            camera_distance: Distance along +Z the default camera is
                placed from the world origin, so a `SceneObject` with
                the default identity transform (at the origin) is
                visible in front of it.

        Raises:
            RenderError: If `pyrender` (and its OpenGL backend) is
                unavailable.
        """
        try:
            import pyrender
        except Exception as exc:
            raise RenderError(f"pyrender is unavailable: {exc}") from exc

        self._pyrender = pyrender
        self._width = width
        self._height = height
        self._camera_distance = camera_distance
        self._offscreen_renderer = pyrender.OffscreenRenderer(width, height)

    def render(self, scene: Scene) -> np.ndarray:
        """Renders every object in `scene` to an RGBA image.

        A deterministic camera (looking down -Z, placed at
        `camera_distance` along +Z per `pyrender`'s convention) and a
        single directional light are added every call, so a
        `SceneObject` left at the default identity transform is always
        visible without any caller-side camera setup. An object that
        fails to convert to a renderable mesh is skipped (logged)
        rather than aborting the whole render.

        Args:
            scene: The `Scene` to render.

        Returns:
            An RGBA `uint8` image of shape `(height, width, 4)`.

        Raises:
            RenderError: If the render call itself fails.
        """
        pyrender = self._pyrender
        render_scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])

        for obj in scene.objects():
            try:
                mesh = pyrender.Mesh.from_trimesh(_flatten_trimesh(obj.mesh), smooth=False)
            except Exception as exc:
                logger.warning("Skipping unrenderable scene object '%s': %s", obj.name, exc)
                continue
            render_scene.add(mesh, pose=_build_object_pose(obj))

        camera = pyrender.PerspectiveCamera(yfov=_DEFAULT_YFOV_RADIANS)
        camera_pose = np.eye(4)
        camera_pose[2, 3] = self._camera_distance
        render_scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(intensity=_DEFAULT_LIGHT_INTENSITY)
        render_scene.add(light, pose=camera_pose)

        try:
            color, _depth = self._offscreen_renderer.render(render_scene, flags=pyrender.RenderFlags.RGBA)
        except Exception as exc:
            raise RenderError(f"Rendering failed: {exc}") from exc

        return color

    def close(self) -> None:
        """Releases the underlying offscreen rendering context."""
        self._offscreen_renderer.delete()


def composite_rgba_onto(frame_image: np.ndarray, rendered_rgba: np.ndarray) -> np.ndarray:
    """Alpha-composites a rendered RGBA image onto a BGR camera frame in place.

    Standard "over" compositing driven by the rendered image's own
    alpha channel. Only pixels where the render has any opacity are
    touched, so an empty/transparent render never overwrites the
    camera feed. `rendered_rgba` is resized to match `frame_image`
    first if their dimensions differ.

    Args:
        frame_image: BGR camera frame (uint8), mutated in place.
        rendered_rgba: RGBA image from `Renderer3D.render()`.

    Returns:
        The same `frame_image`, with the rendered model composited
        onto it.
    """
    import cv2

    height, width = frame_image.shape[:2]
    if rendered_rgba.shape[:2] != (height, width):
        rendered_rgba = cv2.resize(rendered_rgba, (width, height), interpolation=cv2.INTER_LINEAR)

    alpha = rendered_rgba[:, :, 3].astype(np.float32) / 255.0
    touched = alpha > 0
    if not np.any(touched):
        return frame_image

    rgb = rendered_rgba[:, :, :3][:, :, ::-1]  # RGB -> BGR

    alpha_3 = alpha[touched][..., None]
    frame_f = frame_image[touched].astype(np.float32)
    render_f = rgb[touched].astype(np.float32)
    blended = render_f * alpha_3 + frame_f * (1.0 - alpha_3)
    frame_image[touched] = blended.astype(np.uint8)

    return frame_image
