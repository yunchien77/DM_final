"""Phase 7 configuration — focused on the climate-feature revolution.

Stripped of every Phase 5/6 ablation flag (per-cluster training, recent-year
filter, AV weighting, severity weighting, climate-cluster bucketing,
calendar-matched training, time-decay, isotonic, etc). The standing 0.8777
baseline is reproducible from git history. Phase 7 is a clean-slate test of:

  - climate / drought-physics features (climatology-standardized per region × woy)
  - sliding-window temporal stats (replaces the legacy 13-position lag block)
  - score_lag1 as the single Markov anchor
  - empirical drought transition matrix for post-hoc smoothing

Goal: break Kaggle 0.8 (stretch: 0.75).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "dataset" / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = ROOT / "dataset" / "sample_submission.csv"
MODELS_DIR = Path(__file__).parent / "models"
SUBMISSION_PATH = ROOT / "submission.csv"
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data layout
# ---------------------------------------------------------------------------

METEO_FEATURES = [
    "humidity", "tmp", "tmp_range", "tmp_max", "tmp_min",
    "surf_tmp", "surf_pre", "wb_tmp", "prec",
]

# Within-week stats computed in data_pipeline.py stage-1.
STAT_SUFFIXES = ["mean", "std", "min", "max", "sum"]

# Look-back for the residual raw lag block used only by score_lag1 (kept) and
# the cache-key signature. Sliding-window features now carry the temporal signal.
LAG_WINDOW = 12

N_REGIONS = 2248
TRAIN_WEEKS_PER_REGION = 782
TEST_WEEKS_PER_REGION = 13
HORIZONS = [1, 2, 3, 4, 5]

# Legacy fixed-holdout fallback (used only when CALENDAR_MATCHED_VALIDATION=False).
VALID_WEEKS = 26


# ---------------------------------------------------------------------------
# Validation split (calendar-matched — keeps cross-year shift faithful in val)
# ---------------------------------------------------------------------------

CALENDAR_MATCHED_VALIDATION = True
CALENDAR_MATCHED_SLACK_WEEKS = 6
CALENDAR_MATCHED_LAST_YEAR_ONLY = True


# ---------------------------------------------------------------------------
# Daily preprocessing — winsorize, log/sqrt, rank-normalize (kept; not part of
# the feature philosophy change).
# ---------------------------------------------------------------------------

USE_PREPROCESSING = True
USE_RANK_NORMALIZATION = True
RANK_NORMALIZE_FEATURES = ["humidity", "wb_tmp", "tmp_range"]


# ---------------------------------------------------------------------------
# Feature blocks (Phase 7 toggles)
# ---------------------------------------------------------------------------

# Block 1 — score_lag1. Phase 8: DROPPED. The train↔test gap averages ~163
# weeks; score autocorrelation 0.96^163 ≈ 0.001. The lag is predictive on val
# (where the shifted gap is ~14 weeks) but useless on test — a phantom anchor.
# Replaced by Block 1b (proxy score) which has identical train/test semantics.
# Phase 11 considered re-enabling with teammate's per-region gap shift but
# deferred — measured benefit on user's data unclear, focus on independent wins.
USE_SCORE_LAG = False

# Block 1b — Phase 8 proxy score. Ridge regression mapping meteo features
# (climate + windowed) to the anchor week's drought score. The output is one
# new column `proxy_score` consumed by the LGBM. Identical feature space at
# train and test.
USE_PROXY_SCORE = True

# Block 2 — climate / precursor features (pressure variability, EDDI, SPEI,
# composite events, dry-spell, heat-stress, WBD). All climatology-standardized
# per (region, woy). See features_climate.py.
USE_CLIMATE_FEATURES = True

# Block 3 — sliding-window temporal stats. Replaces the legacy 13-position
# lag block. See features_windowed.py.
USE_WINDOWED_FEATURES = True
WINDOWED_CHANNELS = ("prec", "surf_pre", "tmp", "tmp_range", "humidity", "wb_tmp", "tmp_max")
WINDOWED_WINDOWS = (4, 8, 12)

# Block 4 — meteo-feature region clustering. Gated on a diagnostic
# (cluster_diagnostic.py): the cluster id becomes a feature only when the
# diagnostic writes the cluster table. See models/meteo_cluster_table.csv.
USE_METEO_CLUSTER = True   # train/predict pick it up only if the table file exists


# ---------------------------------------------------------------------------
# Drought transition matrix (Phase 7 Step C)
# ---------------------------------------------------------------------------

# Build empirical Markov chain P(score_{w+h} = j | score_w = i) per region cluster.
# Apply post-hoc shrinkage at predict time: pred ← β · model_pred + (1−β) · markov_baseline.
USE_TRANSITION_SMOOTHING = True
TRANSITION_SMOOTHING_BETA = 0.8     # tuned on val ∈ {0.6, 0.7, 0.8, 0.9}
TRANSITION_CLUSTERS = 1             # single global matrix; raise to use per-cluster matrices
TRANSITION_SMOOTHING_ALPHA = 1.0    # Laplace smoothing for transition counts

# Training-time transition weights (Step C3). Phase 11: DISABLED. Stacking
# transition × severity weights amplified the L1-median bias in Phase 10
# (Kaggle 0.8975 with pred mean +0.42 vs baseline). With the lag-shift
# features re-enabled, the anchor pull comes from features instead of weights.
USE_TRANSITION_WEIGHT = False
TRANSITION_WEIGHT_GAMMA = 1.0

# Phase 12 ablation: severity weights DISABLED to test the bias hypothesis.
# Phase 10 (with severity α=0.5 β=3) → Kaggle 0.8975, pred mean 1.23 (+0.42 bias).
# Phase 11 (same severity weights, smaller model) → Kaggle 0.9111, pred mean 1.29 (+0.49).
# Teammate uses identical weights and gets +0.18 bias → 0.8255. The bias
# amplification appears specific to user's woy-standardized climate features.
# This ablation tests whether dropping the weights pulls pred mean back to truth.
USE_SEVERITY_WEIGHT = False
SAMPLE_WEIGHT_ALPHA = 0.5
SAMPLE_WEIGHT_BETA = 3.0


# ---------------------------------------------------------------------------
# Zero-inflated two-stage (Phase 9)
# ---------------------------------------------------------------------------

# Replace the single-stage Tweedie regressor with a two-stage decomposition:
#   classifier:  P(y > 0 | x)   (LGBM binary)
#   regressor:   E(y | y > 0, x) (LGBM tweedie on non-zero subset)
# Final pred = P(y > 0) · E(y | y > 0). Trains 10 boosters total (5 horizons × 2).
#
# Phase 10: disabled per [[feedback_dmfp_loss_choice]] — L1 beats two-stage on
# val (0.4522 vs 0.5212), MAE rewards the median. Teammate's 0.8255 Kaggle
# also runs single-stage L1, not zero-inflated.
USE_ZERO_INFLATED = False


# ---------------------------------------------------------------------------
# Pruning (Phase 7 Step D)
# ---------------------------------------------------------------------------

# Aggressive Pearson + Spearman + VIF collinearity prune. When True, train.py
# computes the pruned list at training time and persists to feature_cols_lean.json;
# predict.py loads it.
USE_PRUNING = False    # initial run: train on the full ~130-col matrix
PRUNING_CORRELATION_THRESHOLD = 0.85
PRUNING_VIF_THRESHOLD = 10.0
LEAN_FEATURE_LIST_PATH = MODELS_DIR / "feature_cols_lean.json"


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

LGBM_PARAMS = dict(
    # Phase 11: rolled back from Phase 10's teammate-style 255-leaf / lr=0.015
    # / 5000-iter setup (regressed to Kaggle 0.8975, +0.42 pred mean bias)
    # toward your previously known-good values. L1 objective retained
    # ([[feedback_dmfp_loss_choice]]). Smaller model + lag-shift features
    # should pull pred mean back toward truth.
    objective="regression_l1",
    metric="mae",
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=63,
    min_child_samples=50,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)
EARLY_STOPPING_ROUNDS = 100


# ---------------------------------------------------------------------------
# Walk-forward CV + isotonic OOF calibration (Phase 10)
# ---------------------------------------------------------------------------

# Replace single calendar-matched train/val split with K-fold walk-forward CV:
# every row gets exactly one OOF prediction → IsotonicRegression fits on OOF.
# At predict time, fold models are mean-averaged (variance reduction).
USE_WALK_FORWARD_CV = True
N_WF_FOLDS = 4              # walk-forward folds per horizon (matches teammate's N_PH_FOLDS)
WF_PURGE_WEEKS = 13         # 91-day purge gap, expressed in weeks (user data is weekly)
USE_ISOTONIC_CALIBRATION = True  # auto-rejected if post-cal MAE > pre-cal MAE


# ---------------------------------------------------------------------------
# Kaggle proxy val + calendar-matched ES (Phase 11 — Path 2 merge)
# ---------------------------------------------------------------------------

# Per region, hold out the most recent in-season anchor as a fixed ES set
# used by EVERY fold. Mirrors Kaggle's "predict 5 weeks after the test window
# end" structure so ES stops where test-season performance peaks instead of
# where global-CV performance peaks.
USE_KAGGLE_PROXY_VAL = True
KAGGLE_PROXY_BANDWIDTH = 2     # ± months around region's test month qualify as in-season

# Within each fold's CV val, filter to samples within ± bandwidth months of
# the region's test month before using as ES set. Falls back to full val if
# too few in-season samples remain.
USE_CAL_MATCHED_ES = True
CAL_MATCHED_ES_BANDWIDTH = 2
CAL_MATCHED_ES_MIN_SAMPLES = 20

# Cap proxy ridge sampling to the N most-recent valid score rows per region.
# Teammate uses 40 → fits on ~90k samples, much less prone to old-climate
# contamination than user's 1.5M-row RidgeCV fit.
PROXY_SAMPLES_PER_REGION = 40


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

RUN_WALK_FORWARD_DIAGNOSTIC = False
WALK_FORWARD_FOLD_BOUNDARIES = [735, 745, 755, 765]


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

N_WORKERS = max(1, os.cpu_count() - 2)
