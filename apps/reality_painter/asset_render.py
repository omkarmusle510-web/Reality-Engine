"""3D asset compositing stage for Reality Painter.

Bridges the existing engine 3D pipeline (Phase 12D:
`engine.scene.scene.Scene`, `engine.rendering.renderer.Renderer3D`,
`composite_rgba_onto`) into Reality Painter's own pipeline as one more
stage, the same pattern every other stage in
`apps/reality_painter/app.py` already follows. This module owns no
rendering logic of its own - it only calls the existing engine APIs
and composites their result onto the current camera frame.

Asset retrieval (Phase 12C's `AssetRetriever`) and GLB loading (Phase
12D's `engine.scene.loader.load_glb`) both happen once, at application
startup in `app.py` - never inside this stage. This stage only ever
renders whatever `SceneObject`s are already in the `Scene` it was
given; it performs no network access, no file I/O, and no GLB parsing.
"""

from __future__ import annotations

from engine.core.pipeline import PipelineContext, StageFunc
from engine.rendering.renderer import Renderer3D, composite_rgba_onto
from engine.scene.scene import Scene
from engine.vision.frame import Frame


def create_asset_render_stage(scene: Scene, renderer: Renderer3D) -> StageFunc:
    """Builds a pipeline stage that renders `scene` and composites it onto the frame.

    Reads `context["frame"]`. If a frame is present and `scene`
    currently holds at least one object, renders `scene` via the
    existing `Renderer3D.render()` and alpha-composites the result onto
    `frame.image` in place via the existing `composite_rgba_onto()` -
    no new rendering or blending logic is introduced here. A no-op if
    no frame is present yet, or if `scene` is empty (e.g. no asset was
    successfully loaded at startup), so this stage never forces a
    render call when there is nothing to show.

    Args:
        scene: The `Scene` holding whichever `SceneObject`(s) should be
            displayed. Owned by the caller (`app.py`) so assets can be
            added/removed independently of this stage.
        renderer: A `Renderer3D` instance, owned by the caller so its
            offscreen rendering context persists across pipeline
            executions instead of being recreated every frame.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _asset_render_stage(context: PipelineContext) -> PipelineContext:
        frame = context.get("frame")
        if isinstance(frame, Frame) and len(scene) > 0:
            rendered = renderer.render(scene)
            composite_rgba_onto(frame.image, rendered)
        return context

    return _asset_render_stage