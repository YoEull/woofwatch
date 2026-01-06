"""
Logging utilities for WoofWatch.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.core.config import get_config


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: Optional[bool] = None
) -> logging.Logger:
    """Setup and configure a logger.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR). Uses config if None
        log_file: Log file path. Uses config if None
        console: Enable console output. Uses config if None

    Returns:
        Configured logger instance
    """
    config = get_config()
    log_config = config.logging

    # Use config values if not specified
    level = level or log_config.level
    log_file = log_file or log_config.file
    console = console if console is not None else log_config.console

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_config.format)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up
    if not logger.handlers:
        setup_logger(name)

    return logger
