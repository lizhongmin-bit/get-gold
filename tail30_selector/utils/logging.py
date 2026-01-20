"""Logging utilities."""
from __future__ import annotations

import logging
from logging import Logger


_DEF_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> Logger:
    logging.basicConfig(level=level, format=_DEF_FORMAT)
    return logging.getLogger("tail30_selector")
