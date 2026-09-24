"""Logging setup. Two sinks: a console handler and, for a given run, a file
handler writing to `runs/<exp_id>/run.log`. Text logs are for debugging; the
*structured* record of a run goes through `ror.results`, not here.
"""
from __future__ import annotations

import logging
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "ror") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def add_run_file_handler(logger: logging.Logger, run_dir: str | Path) -> logging.Handler:
    """Attach a file handler for one run; returns it so it can be removed after."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "run.log")
    fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    logger.addHandler(fh)
    return fh


def remove_handler(logger: logging.Logger, handler: logging.Handler) -> None:
    try:
        handler.close()
    finally:
        if handler in logger.handlers:
            logger.removeHandler(handler)
