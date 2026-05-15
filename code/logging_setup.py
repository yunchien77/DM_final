"""
Centralized logging for the DMFP pipeline.

Mirrors output to **stdout** (for live terminal monitoring) and to a
**timestamped log file** under `code/logs/` (audit trail that survives long
training runs).

Usage:
    from logging_setup import get_logger
    log = get_logger(__name__)
    log.info("Training started")

The first call to `get_logger` configures handlers on the root logger; later
calls just return a named logger that inherits those handlers — so it's safe
to call from any module without worrying about ordering.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False
_LOG_PATH: Path | None = None


def _configure_root(label: str = "run"):
    """Idempotent one-time configuration of the root logger."""
    global _CONFIGURED, _LOG_PATH
    if _CONFIGURED:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{label}_{timestamp}.log"
    _LOG_PATH = log_path

    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Replace any preexisting handlers (e.g., notebook environments register one).
    root.handlers = [fh, sh]

    # Silence third-party noise — these libraries dump verbose DEBUG into our file
    # handler otherwise.
    for noisy in ("matplotlib", "PIL", "asyncio", "urllib3", "fontTools", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    root.info(f"Logging to {log_path}")


def get_logger(name: str, label: str = "run") -> logging.Logger:
    """
    Return a configured logger. The first call sets up the timestamped log
    file under code/logs/<label>_<timestamp>.log. `label` is only honored on
    that first call (later calls inherit the original configuration).
    """
    _configure_root(label=label)
    return logging.getLogger(name)


def current_log_path() -> Path | None:
    """Path of the active log file, or None if `get_logger` hasn't run yet."""
    return _LOG_PATH
