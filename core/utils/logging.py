"""Logging helpers for InfoPilot core modules."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s: %(message)s"


@lru_cache(maxsize=128)
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger configured with the shared format."""
    logger = logging.getLogger(name or "infopilot")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
