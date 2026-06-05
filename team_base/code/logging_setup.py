import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ── Session ID ────────────────────────────────────────────────────────────────
# A single timestamp shared by all loggers in one process run.
# Printed in the startup banner so every log file is traceable to a specific run.
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Formatters ────────────────────────────────────────────────────────────────
# Console: compact time (HH:MM:SS), fixed-width level and module name
_CONSOLE_FMT = logging.Formatter(
    fmt     = "%(asctime)s  %(levelname)-5s  %(name)-16s | %(message)s",
    datefmt = "%H:%M:%S",
)

# File: full date-time for post-run analysis
_FILE_FMT = logging.Formatter(
    fmt     = "%(asctime)s  %(levelname)-5s  %(name)-16s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)


# ── Logger factory ────────────────────────────────────────────────────────────

def get_logger(name: str, label: str = "") -> logging.Logger:
    """Return a configured logger.

    - Console handler: INFO and above (avoids debug noise in the terminal)
    - Rotating file handler: DEBUG and above (full detail for post-run review)
      Each log file grows to 20 MB before rotating; 5 backups are kept.

    The `label` parameter sets the log filename (e.g., label="train" →
    logs/train.log). Falls back to `name` when label is empty.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured; avoid duplicate handlers on re-import

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # prevent messages bubbling up to the root logger

    # Console: INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_CONSOLE_FMT)
    logger.addHandler(console_handler)

    # File: DEBUG and above, rotating to avoid unbounded file growth
    try:
        from config import LOGS_DIR
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        log_file = LOGS_DIR / f"{label or name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            mode        = "a",
            encoding    = "utf-8",
            maxBytes    = 20 * 1024 * 1024,  # 20 MB per file
            backupCount = 5,                  # keep train.log .. train.log.5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_FILE_FMT)
        logger.addHandler(file_handler)
    except Exception:
        pass  # never crash on logging setup failure

    return logger


# ── Startup banner ────────────────────────────────────────────────────────────

def log_startup_banner(logger: logging.Logger, title: str = "") -> None:
    """Log a startup banner with session ID, hardware info, and key config.

    Call once at the top of train.py / predict.py main() so every log file
    begins with a clear record of what machine and configuration was used.
    """
    try:
        from hardware import summary as hw_summary
        hw_line = hw_summary()
    except Exception as exc:
        hw_line = f"(hardware detection failed: {exc})"

    try:
        from config import (
            N_WORKERS, LGBM_JOBS, USE_GPU_LGBM,
            USE_GAP_STRATIFIED, USE_ADVERSARIAL_WEIGHTS,
            USE_CALENDAR_MATCHED_VAL, FEATURE_VERSION,
        )
        cfg_line = (
            f"workers={N_WORKERS}  lgbm_jobs={LGBM_JOBS}  "
            f"gpu={USE_GPU_LGBM}  gap_strat={USE_GAP_STRATIFIED}  "
            f"adversarial={USE_ADVERSARIAL_WEIGHTS}  "
            f"cal_match={USE_CALENDAR_MATCHED_VAL}  feature_v={FEATURE_VERSION}"
        )
    except Exception as exc:
        cfg_line = f"(config summary failed: {exc})"

    width = 72
    logger.info("=" * width)
    if title:
        logger.info(f"  {title}")
    logger.info(f"  Session : {SESSION_ID}")
    logger.info(f"  Hardware: {hw_line}")
    logger.info(f"  Config  : {cfg_line}")
    logger.info("=" * width)
