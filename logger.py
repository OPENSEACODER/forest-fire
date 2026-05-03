"""
logger.py
---------
Centralised logging for VanSuraksha.

Usage in any module:
    from logger import logger
    logger.info("Something happened")
"""

import logging
import sys
from config import LOG_FILE, LOG_LEVEL


def _build_logger(name: str = "vansuraksha") -> logging.Logger:
    log = logging.getLogger(name)

    # Avoid duplicate handlers if module is re-imported
    if log.handlers:
        return log

    log.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console output
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    # Rotating file output
    try:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError:
        log.warning("Could not open log file at %s", LOG_FILE)

    return log


logger = _build_logger()
