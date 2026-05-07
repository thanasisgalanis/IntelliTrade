"""Centralised logging: rotating file + console handlers.

Call ``configure_logging()`` once at process start (e.g. in ``main.py``);
afterwards every module just uses ``logging.getLogger(__name__)``.
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
_BACKUP_COUNT = 5

_configured = False


def configure_logging(
    log_file: str | os.PathLike[str] | None = None,
    level: str | int = "INFO",
) -> logging.Logger:
    """Idempotently configure the root logger and return it."""
    global _configured

    root = logging.getLogger()

    if _configured:
        return root

    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    root.setLevel(level)

    formatter = logging.Formatter(_DEFAULT_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
