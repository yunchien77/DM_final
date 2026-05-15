import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "dataset" / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = ROOT / "dataset" / "sample_submission.csv"
MODELS_DIR = Path(__file__).parent / "models"
SUBMISSION_PATH = ROOT / "submission.csv"

MODELS_DIR.mkdir(exist_ok=True)

METEO_FEATURES = [
    "wind", "wind_min", "wind_max", "wind_range",
    "humidity", "tmp", "tmp_range", "tmp_max", "tmp_min",
    "surf_tmp", "surf_pre", "dp_tmp", "wb_tmp", "prec",
]
# Note: wind_range is the 15th feature but surf_pre ordering matches train.csv column order

STAT_SUFFIXES = ["mean", "std", "min", "max", "sum"]

LAG_WINDOW = 12  # weeks of lookback (matches test's 13 weeks: anchor at week 12, lags 0..12)

N_REGIONS = 2248
TRAIN_WEEKS_PER_REGION = 782
TEST_WEEKS_PER_REGION = 13
VALID_WEEKS = 26  # last N weeks used as validation holdout

HORIZONS = [1, 2, 3, 4, 5]

# Phase 2 toggle: include the woy climatology + anomaly + trend + interaction
# feature blocks from features_extra.py. Used by train.py and predict.py.
USE_EXTRA_FEATURES = True

# Daily-level preprocessing hook (impute → winsorize → log/sqrt).
# Artifacts are fit on training daily rows in train.py and persisted under
# code/models/preprocessing.{json,csv}. predict.py loads and re-applies them.
USE_PREPROCESSING = True

# Lagged-score features: score at week_idx − {1,2,4,8} per region, plus
# weeks_since_last_nonzero and max_score_prev8. Mitigates Section 10's high
# Markov persistence.
USE_SCORE_LAG_FEATURES = True

# Region-cluster archetype features (K-means on per-region score profile).
USE_REGION_CLUSTER_FEATURES = True
N_REGION_CLUSTERS = 8

# Rank-normalize the 6 most KS-drifted daily features against the train∪test
# pooled ECDF. Targets the val/Kaggle gap (Section 9 finding).
USE_RANK_NORMALIZATION = True
RANK_NORMALIZE_FEATURES = [
    "dp_tmp", "humidity", "wb_tmp", "tmp_range", "wind", "wind_max",
]

# Severe-row gradient upweighting for L1 LightGBM. Loss family unchanged; the
# weight on each row scales its absolute-residual contribution.
# weight = 1 + α·𝟙[y>0] + β·𝟙[y≥3].
SAMPLE_WEIGHT_ALPHA = 1.0
SAMPLE_WEIGHT_BETA = 2.0

# Walk-forward CV diagnostic. When True, after the main training+eval,
# train.py runs a second pass with multiple temporal folds (lower n_estimators
# for speed) and reports their average macro-MAE as a more honest signal.
RUN_WALK_FORWARD_DIAGNOSTIC = True
WALK_FORWARD_FOLD_BOUNDARIES = [735, 745, 755, 765]  # weekly anchor cutoffs

LGBM_PARAMS = dict(
    objective="regression_l1",  # L1 = MAE, matches competition metric
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

N_WORKERS = max(1, os.cpu_count() - 4)  # leave 4 cores for system / LightGBM overhead
