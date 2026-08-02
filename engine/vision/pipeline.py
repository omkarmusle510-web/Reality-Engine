"""Vision stage for the engine's single execution pipeline.

This module does not define a pipeline - Reality Engine has exactly one,
owned by engine/core/pipeline.py. It only builds the stage function that
reads a camera frame and places it into the shared pipeline context.
"""

from __future__ import annotations

from engine.core.pipeline import PipelineContext, StageFunc
from engine.interfaces.camera import CameraInterface


def create_vision_stage(camera: CameraInterface) -> StageFunc:
    """Builds a pipeline stage that reads one frame per run and stores it in context.

    The camera must already be open before the pipeline executes; this
    stage does not manage camera lifecycle.

    Args:
        camera: An opened camera instance to read frames from.

    Returns:
        A stage function suitable for `engine.pipeline.register_stage`.
    """

    def _vision_stage(context: PipelineContext) -> PipelineContext:
        frame = camera.read()
        if frame is not None:
            context["frame"] = frame
        return context

    return _vision_stage