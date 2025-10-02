import os
import glob
import logging
import threading
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE_BASENAME = "recommender_service.log"
LOG_PATH = LOGS_DIR / LOG_FILE_BASENAME

MAX_BYTES = 5 * 1024 * 1024
BACKUPS_TO_KEEP = 5

def configure_logging(logger_name: str = "recommender") -> logging.Logger:
    """
    Create a module-level logger with:
        - RotatingFileHandler (5 x 5MB) under logs/
        - Console handler
        - Proactive pruning to keep only the last 5 rotated files
    """

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        filename=str(LOG_PATH),
        maxBytes=MAX_BYTES,
        backupCount=BACKUPS_TO_KEEP,
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    _have_file = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
    _have_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in logger.handlers)

    if not _have_file:
        logger.addHandler(file_handler)
    if not _have_console:
        logger.addHandler(console_handler)

    logger.propagate = False

    logger.info(f"Logging initialized. File: {LOG_PATH}")
    return logger

def attach_flask_logger(app, base_logger: logging.Logger | None = None) -> None:
    """
    Attach our configured handlers to Flask's app.logger so both write to the same sinks.
    """
    logger = base_logger or configure_logging()
    app.logger.handlers.clear()
    for h in logger.handlers:
        app.logger.addHandler(h)
    app.logger.setLevel(logger.level)
    app.logger.propagate = False
