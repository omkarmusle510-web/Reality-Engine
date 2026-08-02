"""Engine lifecycle management for Reality Engine.

`Engine` owns configuration, logging, and a `Pipeline`, and enforces a
lifecycle: created -> initialized -> running -> stopped (or error).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from .config import config
from .logger import configure_logging, get_logger
from .pipeline import Pipeline, PipelineContext

class EngineState(str, Enum):
    """Lifecycle states of an `Engine` instance."""

    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class Engine:
    """Lifecycle manager for a Reality Engine application.

    Usage:
        engine = Engine(config)
        engine.initialize()
        engine.pipeline.register_stage(...)
        engine.start()
        engine.shutdown()
    """

    def __init__(self, config: config) -> None:
        self.config = config
        self.state = EngineState.CREATED
        self._logger: Optional[logging.Logger] = None
        self._pipeline: Optional[Pipeline] = None

    @property
    def logger(self) -> logging.Logger:
        """The engine's logger. Raises if accessed before `initialize()`."""
        if self._logger is None:
            raise RuntimeError("Engine logger accessed before initialize() was called.")
        return self._logger

    @property
    def pipeline(self) -> Pipeline:
        """The engine's pipeline. Raises if accessed before `initialize()`."""
        if self._pipeline is None:
            raise RuntimeError("Engine pipeline accessed before initialize() was called.")
        return self._pipeline

    def initialize(self) -> None:
        """Sets up logging and creates the engine's pipeline.

        Raises:
            RuntimeError: If not called from the `CREATED` state.
        """
        if self.state != EngineState.CREATED:
            raise RuntimeError(f"Engine.initialize() called from invalid state '{self.state.value}'.")

        configure_logging(self.config)
        self._logger = get_logger(__name__)
        self._pipeline = Pipeline(name=f"{self.config.app_name}_pipeline")
        self.state = EngineState.INITIALIZED
        self._logger.info("Engine initialized for app '%s'.", self.config.app_name)

    def start(self) -> None:
        """Runs the pipeline continuously until stopped.

        The engine owns execution: applications only call `start()` and
        `shutdown()`, they never call `pipeline.execute()` themselves.

        A single `PipelineContext` is created once and reused across every
        frame, rather than recreated per iteration. This is not just a
        performance choice: several existing stages (e.g. the cursor and
        gesture stages) are deliberately written to be a no-op when their
        input is momentarily missing, deliberately leaving the *previous*
        frame's value in place. That behavior only works if the same
        context object persists across iterations - a fresh dict every
        frame would silently break it.

        The engine has no built-in concept of "why" it should stop. Any
        stage can request a stop generically by setting
        `context["stop_requested"] = True` (for example, a display stage
        detecting ESC or a closed window). This keeps the engine itself
        completely unaware of windows, keys, or any specific input device -
        it only knows one reserved context key. `KeyboardInterrupt` (e.g.
        Ctrl+C) is also treated as a graceful stop signal.

        Raises:
            RuntimeError: If not called from the `INITIALIZED` state, or if
                the pipeline raises (the engine transitions to `ERROR`).
        """
        if self.state != EngineState.INITIALIZED:
            raise RuntimeError(f"Engine.start() called from invalid state '{self.state.value}'.")

        self.state = EngineState.RUNNING
        self.logger.info("Engine starting.")

        context: PipelineContext = {}

        try:
            while self.state == EngineState.RUNNING:
                context = self.pipeline.execute(context)
                if context.get("stop_requested"):
                    self.logger.info("Stop requested by a pipeline stage.")
                    break
        except KeyboardInterrupt:
            self.logger.info("Stop signal received.")
        except Exception:
            self.state = EngineState.ERROR
            self.logger.exception("Engine encountered an error during pipeline execution.")
            raise

    def shutdown(self) -> None:
        """Stops the engine.

        Raises:
            RuntimeError: If the engine was never initialized or is already stopped.
        """
        if self.state in (EngineState.CREATED, EngineState.STOPPED):
            raise RuntimeError(f"Engine.shutdown() called from invalid state '{self.state.value}'.")

        self.logger.info("Engine shutting down.")
        self.state = EngineState.STOPPED