"""Centralized logging setup for Reality Engine.

All modules should call `get_logger(__name__)` rather than using `print`.
`configure_logging` wires up the root "reality_engine" logger exactly once
per process, based on the engine's config.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import config as EngineConfig

_ROOT_LOGGER_NAME = "reality_engine"
_configured = False


def configure_logging(config: EngineConfig) -> None:
    """Configures the Reality Engine root logger.

    Idempotent: calling this multiple times will not duplicate handlers.

    Args:
        config: Engine configuration supplying log level and destinations.
    """
    global _configured

    root_logger = logging.getLogger(_ROOT_LOGGER_NAME)

    if _configured:
        root_logger.setLevel(config.log_level)
        return

    root_logger.setLevel(config.log_level)
    root_logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if config.log_to_file:
        log_path = Path(config.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Returns a namespaced logger nested under the Reality Engine root logger.

    Args:
        name: Typically `__name__` of the calling module.
    """
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")