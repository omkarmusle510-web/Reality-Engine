"""Configuration for Reality Engine.

`EngineConfig` is a small, validated configuration object shared by the
engine and its subsystems. It intentionally knows nothing about any specific
application (RealityPainter, etc.) so it stays reusable across the framework.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass
class config:
    """Configuration for an `Engine` instance.

    Attributes:
        app_name: Name of the application using the engine.
        environment: Deployment environment label (e.g. "development").
        log_level: Logging level name (e.g. "INFO").
        log_to_file: Whether the logger should also write to a file.
        log_file_path: Path to the log file, used only if `log_to_file` is True.
    """

    app_name: str
    environment: str = "development"
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: str = "logs/engine.log"

    def __post_init__(self) -> None:
        """Validates fields and normalizes `log_level` to uppercase.

        Raises:
            ValueError: If `app_name` is empty or `log_level` is not a
                recognized logging level.
        """
        if not self.app_name.strip():
            raise ValueError("config.app_name must be a non-empty string.")

        self.log_level = self.log_level.upper()
        if self.log_level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"config.log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got {self.log_level!r}."
            )