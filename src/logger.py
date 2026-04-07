"""
logger.py — Structured logging setup for HybridRAG Bench.

All modules import ``get_logger(__name__)`` rather than calling
``print()`` directly. Log level is controlled via the LOG_LEVEL env var.
"""

import logging
import os
import sys

import colorlog


def get_logger(name: str) -> logging.Logger:
    """
    Return a colorized console logger for the given module name.

    Usage::

        from src.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Index built with %d vectors", n)
    """
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    handler = colorlog.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)-8s] %(name)s:%(reset)s %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
