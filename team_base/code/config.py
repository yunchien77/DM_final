import os
from pathlib import Path

from hardware import CPU_WORKERS, N_GPUS_AVAILABLE, GPU_IDS as _DETECTED_GPU_IDS


# ═════════════════════════════════════════════════════════════════════════════
# Paths
# ═════════════════════════════════════════════════════════════════════════════

ROOT            = Path(__file__).parent.parent
DATA_DIR        = ROOT / "dataset" / "data"
TRAIN_PATH      = DATA_DIR / "train.csv"
TEST_PATH       = DATA_DIR / "test.csv"
SAMPLE_PATH     = ROOT / "dataset" / "sample_submission.csv"
MODELS_DIR      = ROOT / "models"
LOGS_DIR        = ROOT / "logs"
CACHE_DIR       = ROOT / "dataset" / "cache"
SUBMISSION_PATH = ROOT / "submission.csv"
EVAL_DIR        = ROOT / "eval"

# Sub-directories for model artifacts
MODELS_PH_DIR        = MODELS_DIR / "per_horizon"
MODELS_GAP_SHORT_DIR = MODELS_DIR / "gap_short"
MODELS_GAP_LONG_DIR  = MODELS_DIR / "gap_long"

# Artifact file paths
CLIMATOLOGY_PATH   = MODELS_DIR / "climatology.pkl"
PROXY_RIDGE_PATH   = MODELS_DIR / "proxy_ridge.pkl"
MODELS_PKL_PATH    = MODELS_DIR / "lgbm_models.pkl"
CALIBRATOR_PATH    = MODELS_DIR / "calibrator.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
EVAL_REPORT_PATH   = EVAL_DIR   / "eval_report.json"
PREPROC_PATH       = MODELS_DIR / "preprocessing.pkl"


# ── Directory setup ───────────────────────────────────────────────────────────

_ALL_DIRS = [
    MODELS_DIR,
    LOGS_DIR,
    CACHE_DIR,
    EVAL_DIR,
    MODELS_PH_DIR,
    MODELS_GAP_SHORT_DIR,
    MODELS_GAP_LONG_DIR,
]


def ensure_dirs() -> list[Path]:
    """Create all required project directories.

    Returns a list of directories that were newly created (already-existing
    directories are not included). Call this at the start of train/predict
    to get a logged confirmation that the filesystem is ready.
    """
    created = []
    for d in _ALL_DIRS:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(d)
    return created


# Create silently at import time so that other modules that write artifacts
# never fail due to a missing directory.
ensure_dirs()


# ═════════════════════════════════════════════════════════════════════════════
# Meteorological columns
# ═════════════════════════════════════════════════════════════════════════════

# dp_tmp excluded: perfect correlation with wb_tmp (fully redundant per EDA)
# Wind columns excluded: feature gain < 0.43% per EDA Section 6
METEO_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
]
COL_IDX = {c: i for i, c in enumerate(METEO_COLS)}
N_METEO  = len(METEO_COLS)


# ═════════════════════════════════════════════════════════════════════════════
# Feature engineering
# ═════════════════════════════════════════════════════════════════════════════

WINDOW_SIZES             = [7, 14, 21, 42, 91]
TREND_COLS               = ["tmp", "prec", "humidity", "surf_tmp", "tmp_range"]
PROXY_WINDOWS            = [7, 21, 91]
N_PROXY_SIGS             = 7
PROXY_SAMPLES_PER_REGION = 40

# Increment when feature logic changes to invalidate the feature cache.
FEATURE_VERSION = 14

# Score autocorrelation base (ACF ≈ 0.96^lag per EDA Section 2)
SCORE_ACF_BASE = 0.96


# ═════════════════════════════════════════════════════════════════════════════
# General training settings
# ═════════════════════════════════════════════════════════════════════════════

SEED                 = 42
N_PRED               = 5     # forecast horizons (fw = 1..5 weeks)
N_FOLDS              = 5
N_PH_FOLDS           = 4     # walk-forward folds for the main model
N_PH_FOLDS_SECONDARY = 2     # fewer folds for secondary models (speed vs. variance)
MAX_WIN_PER_REGION   = 200   # use more training windows per region
SAMPLE_FRAC          = 1.0   # use all available data
PURGE_GAP_DAYS       = 91    # temporal purge gap to prevent leakage

USE_SAMPLE_WEIGHT = True
WEIGHT_NONZERO    = 1.5      # upweight non-zero drought scores
WEIGHT_SEVERE     = 3.0      # upweight severe drought (score ≥ 3)

USE_CALIBRATION = True

# True  → 5 forecast-horizon models run simultaneously (faster on multi-core)
# False → sequential execution (safer for debugging or low-memory machines)
PARALLEL_FW_TRAINING = True

USE_PER_HORIZON_MODELS = True
USE_CACHE              = True


# ═════════════════════════════════════════════════════════════════════════════
# Parallelism
# ═════════════════════════════════════════════════════════════════════════════

# Workers for multiprocessing.Pool (feature engineering)
N_WORKERS = CPU_WORKERS

# Threads per LightGBM model instance.
# When PARALLEL_FW_TRAINING=True, N_PRED models run at once;
# dividing workers by N_PRED prevents CPU oversubscription.
LGBM_JOBS = max(1, CPU_WORKERS // N_PRED) if PARALLEL_FW_TRAINING else CPU_WORKERS


# ═════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═════════════════════════════════════════════════════════════════════════════

USE_PREPROCESSING = True

# Apply log1p / sqrt to right-skewed columns before feature engineering.
# Rank normalization is intentionally omitted: LightGBM histogram splits are
# invariant to monotonic transforms, so it adds complexity with no MAE benefit.
PREPROC_LOG_FEATURES: dict[str, str] = {
    "prec":     "log1p",
    "surf_pre": "sqrt",
}


# ═════════════════════════════════════════════════════════════════════════════
# Calendar-Matched Validation
# ═════════════════════════════════════════════════════════════════════════════

USE_CALENDAR_MATCHED_VAL     = True
CALENDAR_BANDWIDTH           = 2    # ± months around the test month
CALENDAR_MATCHED_MIN_SAMPLES = 20   # minimum val samples; falls back to all-val if fewer


# ═════════════════════════════════════════════════════════════════════════════
# Calendar Season Weighting (training)
# ═════════════════════════════════════════════════════════════════════════════

# Upweight training samples whose month is close to the region's test season.
# No data is discarded — off-season samples stay but receive weight 1.0.
# In-season samples (dist ≤ SLACK months) receive IN_SEASON_BOOST.
# Severe drought samples (score ≥ SEVERE_THRESHOLD) always receive the boost
# regardless of season, so rare events are never down-weighted.
# Disabled for baseline recovery: previously caused 1.0265 due to interaction
# with double-severity bug (severity² amplification).  The bug is now fixed in
# train.py (adversarial_weights passes pure av-ratio, not pre-multiplied by
# severity).  Re-enable after confirming baseline is restored.
USE_CALENDAR_SEASON_WEIGHTS      = False
CALENDAR_SEASON_SLACK            = 2     # ± months (same as CALENDAR_BANDWIDTH)
IN_SEASON_WEIGHT_BOOST           = 2.0  # multiplier for in-season samples
CALENDAR_SEVERE_THRESHOLD        = 3.0  # score ≥ this always gets the boost


# ═════════════════════════════════════════════════════════════════════════════
# Score Lag Gap Shift
# ═════════════════════════════════════════════════════════════════════════════

USE_SCORE_LAG_SHIFT = True


# ═════════════════════════════════════════════════════════════════════════════
# Gap-Adaptive Proxy Fallback
# ═════════════════════════════════════════════════════════════════════════════

USE_GAP_ADAPTIVE_FALLBACK = True


# ═════════════════════════════════════════════════════════════════════════════
# Kaggle Proxy Validation
# ═════════════════════════════════════════════════════════════════════════════

# For each region per fw, hold out the most recent in-season anchor as a fixed
# early-stopping set that mirrors the Kaggle evaluation split exactly.
# This prevents overfitting to the walk-forward CV validation distribution
# which may not match the test season.
USE_KAGGLE_PROXY_VAL = True


# ═════════════════════════════════════════════════════════════════════════════
# Adversarial Weighting
# ═════════════════════════════════════════════════════════════════════════════

# Disabled: all 2248 train regions == all 2248 test regions, so the binary
# classifier predicts P(test-like)≈0 for all train rows → weights all clip to
# AV_CLIP_LO and normalize to 1.0 (no effect).  Re-enable when train/test
# region sets diverge.
USE_ADVERSARIAL_WEIGHTS = False
AV_CLIP_LO              = 0.25   # clip adversarial weight ratio below this
AV_CLIP_HI              = 4.0    # clip adversarial weight ratio above this


# ═════════════════════════════════════════════════════════════════════════════
# Gap-Stratified Models
# ═════════════════════════════════════════════════════════════════════════════

# EDA shows a bimodal train/test gap distribution:
#   short gap (<13w) → ACF > 0.4, score lag features remain informative
#   long  gap (≥13w) → ACF < 0.4, score lag has decayed; met. features dominate
USE_GAP_STRATIFIED  = True
GAP_SHORT_THRESHOLD = 13    # weeks; below this a region is classified as short-gap
GAP_STRATA_BLEND    = True  # soft-blend short/long predictions based on gap distance


# ═════════════════════════════════════════════════════════════════════════════
# GPU settings
# ═════════════════════════════════════════════════════════════════════════════

# Detected at runtime from nvidia-smi; empty list means no GPUs available.
N_GPUS  = N_GPUS_AVAILABLE
GPU_IDS = _DETECTED_GPU_IDS

# Default: run on CPU.
# Set USE_GPU_LGBM=True and LGBM_DEVICE="gpu" (or "cuda") to enable GPU acceleration.
USE_GPU_LGBM = False
LGBM_DEVICE  = "cpu"


# ═════════════════════════════════════════════════════════════════════════════
# LightGBM hyperparameters
# ═════════════════════════════════════════════════════════════════════════════

LGBM_PARAMS = dict(
    objective         = "regression_l1",
    metric            = "mae",
    n_estimators      = 5000,
    learning_rate     = 0.015,
    num_leaves        = 255,
    min_child_samples = 20,
    feature_fraction  = 0.7,
    bagging_fraction  = 0.8,
    bagging_freq      = 5,
    lambda_l1         = 0.05,
    lambda_l2         = 0.05,
    verbose           = -1,
    n_jobs            = LGBM_JOBS,
    seed              = SEED,
)

EARLY_STOPPING_ROUNDS = 150


# ═════════════════════════════════════════════════════════════════════════════
# Proxy Ridge
# ═════════════════════════════════════════════════════════════════════════════

PROXY_RIDGE_ALPHA = 1.0
