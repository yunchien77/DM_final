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

# Training-time transition weights (Step C3). Phase 7 v2: enabled to upweight
# rare 1-step score transitions (heavy-tail moves the L1 model under-fits).
# Weight = 1 + γ · (1 − P_c^1[score_lag1, observed_score]). γ=1.0 caps the
# maximum weight at 2× uniform.
# Note: needs `score_lag1` as the "previous score" reference. Phase 8 keeps
# the score_lag1 column computed internally but excludes it from feature_cols
# (it's the phantom-anchor problem; useful here only as a weighting input).
USE_TRANSITION_WEIGHT = True
TRANSITION_WEIGHT_GAMMA = 1.0

# Phase 8 — severity weighting RESTORED at the teammate's settings.
# Per-row weight = 1 + α · 𝟙[y > 0] + β · 𝟙[y ≥ 3], where y = target_w1.
# Multiplied on top of the transition weight before normalization. The
# teammate uses α=0.5 (mild any-nonzero boost) and β=3 (strong severe
# boost) and gets 0.82 Kaggle.
USE_SEVERITY_WEIGHT = True
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
    # Phase 10: L1 (MAE-aligned), matching teammate's 0.8255 Kaggle baseline and
    # the user's own val finding (L1 beats two-stage 0.4522 vs 0.5212, see
    # [[feedback_dmfp_loss_choice]]). MAE-as-metric was already in place; only
    # the objective changes. Tweedie + zero-inflated were Phase 9 experiments
    # that both regressed Kaggle relative to the L1 single-stage baseline.
    objective="regression_l1",
    metric="mae",
    n_estimators=5000,
    learning_rate=0.015,
    num_leaves=255,
    min_child_samples=20,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.05,
    reg_lambda=0.05,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)
EARLY_STOPPING_ROUNDS = 150


# ---------------------------------------------------------------------------
# Walk-forward CV + isotonic OOF calibration (Phase 10)
# ---------------------------------------------------------------------------

# Replace single calendar-matched train/val split with K-fold walk-forward CV:
# every row gets exactly one OOF prediction → IsotonicRegression fits on OOF.
# At predict time, fold models are mean-averaged (variance reduction).
USE_WALK_FORWARD_CV = True
N_WF_FOLDS = 4              # walk-forward folds per horizon (matches teammate's N_PH_FOLDS)
WF_PURGE_WEEKS = 13         # 91-day purge gap, expressed in weeks (user data is weekly)
USE_ISOTONIC_CALIBRATION = True


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

RUN_WALK_FORWARD_DIAGNOSTIC = False
WALK_FORWARD_FOLD_BOUNDARIES = [735, 745, 755, 765]


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

N_WORKERS = max(1, os.cpu_count() - 2)
