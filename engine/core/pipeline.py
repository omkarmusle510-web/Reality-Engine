"""Generic pipeline execution for Reality Engine.

The engine owns exactly one execution pipeline. A `Pipeline` runs an
ordered list of named stages, each a plain callable that takes and returns
a shared context dict. Stages know nothing about the engine or each other.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from .logger import get_logger

logger = get_logger(__name__)

PipelineContext = Dict[str, Any]
StageFunc = Callable[[PipelineContext], PipelineContext]


class Pipeline:
    """An ordered sequence of stages executed over a shared context."""

    def __init__(self, name: str = "pipeline") -> None:
        self.name = name
        self._stages: List[Tuple[str, StageFunc]] = []

    def register_stage(self, name: str, stage: StageFunc) -> None:
        """Appends a named stage to the pipeline.

        Raises:
            ValueError: If a stage with this name is already registered.
        """
        if any(existing == name for existing, _ in self._stages):
            raise ValueError(f"Stage '{name}' is already registered on pipeline '{self.name}'.")
        self._stages.append((name, stage))

    def execute(self, context: Optional[PipelineContext] = None) -> PipelineContext:
        """Runs all stages in order, threading the context through each.

        Args:
            context: Initial shared context. Defaults to an empty dict.

        Returns:
            The final context after all stages have run.

        Raises:
            RuntimeError: If a stage raises, wrapped with the stage's name.
        """
        current_context: PipelineContext = context if context is not None else {}

        for stage_name, stage in self._stages:
            logger.debug("Pipeline '%s' running stage '%s'.", self.name, stage_name)
            try:
                current_context = stage(current_context)
            except Exception as exc:
                raise RuntimeError(f"Stage '{stage_name}' in pipeline '{self.name}' failed.") from exc

        return current_context