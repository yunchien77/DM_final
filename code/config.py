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
    "humidity", "tmp", "tmp_range", "tmp_max", "tmp_min",
    "surf_tmp", "surf_pre", "wb_tmp", "prec",
]
# Wind family (wind/wind_min/wind_max/wind_range) dropped — combined 0.43% gain
# across 152 columns in the prior trained model. dp_tmp dropped — corr(dp_tmp,
# wb_tmp) = 1.000; the model split 0.62% gain between the duplicate pair.

STAT_SUFFIXES = ["mean", "std", "min", "max", "sum", "p95"]
# Phase 2: added "p95" — tail-intensity per anchor week. Cheap (sub-msec per
# anchor) and only applied at lag0 via _build_feature_col_names; ~9 new
# features (one per meteo feature). Targets extreme-event underprediction.

LAG_WINDOW = 12  # weeks of lookback (matches test's 13 weeks: anchor at week 12, lags 0..12)

N_REGIONS = 2248
TRAIN_WEEKS_PER_REGION = 782
TEST_WEEKS_PER_REGION = 13
VALID_WEEKS = 26  # last N weeks used as validation holdout (legacy fixed split)

# Phase 1 — calendar-matched validation. The fixed last-26-weeks holdout puts
# val anchors at end-December for every region; the test windows are mostly
# Jul-Nov per region. Switch to per-region woy-aligned holdout to make val a
# faithful Kaggle proxy. Slack = ±6 weeks around each region's test_end_woy.
CALENDAR_MATCHED_VALIDATION = True
CALENDAR_MATCHED_SLACK_WEEKS = 6
CALENDAR_MATCHED_LAST_YEAR_ONLY = True

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

# Phase 1b — score-lag staleness fix. At test time the model only has scored
# data up to the end of training, then 13 unscored test weeks, then predicts
# 5 future weeks. So test's score_lag1 source is 14 weeks before its first
# prediction target (gap = 14). Training currently uses score_lag1 = scores
# at W−1, giving a 2-week gap to the W+1 target — fresh, very predictive,
# and completely unlike test. To match test's information state, shift the
# train-time lag lookups back by (TEST_WEEKS_PER_REGION − 1) = 12 weeks so
# train's (lag-source → first-target) distance becomes 14, matching test.
SCORE_LAG_TEST_GAP_SHIFT = TEST_WEEKS_PER_REGION - 1  # 12

# Region-cluster archetype features (K-means on per-region score profile).
USE_REGION_CLUSTER_FEATURES = True
N_REGION_CLUSTERS = 8

# Phase 2 — feature audit. After all features are constructed, restrict the
# training matrix to (a) features in feature_cols_pruned.json (top 70% by
# mean cross-horizon gain from the Phase 1b run) plus (b) any new features
# introduced since the prune list was generated (e.g., the *_p95_lag0
# block). Bottom 30% pruned features contribute ~3.3% of total gain.
USE_FEATURE_AUDIT_PRUNE = True
PRUNED_FEATURE_LIST_PATH = MODELS_DIR / "feature_cols_pruned.json"

# Phase 3a — calendar-matched training filter. After the train/val split,
# further restrict training to rows whose woy is within
# CALENDAR_MATCHED_SLACK_WEEKS of that region's test_woy (in-season), plus
# OFF_SEASON_TAIL_FRAC of the remaining off-season rows (preserves rare-event
# diversity: score=5 rows would drop from 31k to 7.7k under a pure filter).
# Net: ~1.69M → ~688k training rows, each structurally test-like.
# Cache-neutral (applied after feature construction).
USE_CALENDAR_MATCHED_TRAINING = True
OFF_SEASON_TAIL_FRAC = 0.2
CALENDAR_MATCHED_TRAINING_SEED = 42

# Phase 3b — adversarial importance weighting. Binary LGBM classifier on
# meteo lag-0 stats predicts P(in-season | x); training rows are weighted
# by P/(1-P), clipped to [LO, HI], normalized to mean=1, and multiplied
# into the severity weight. ESS/N is tracked — if it drops below 0.30, the
# weighting is too aggressive (revert / tighten clip).
USE_ADVERSARIAL_WEIGHTING = True
AV_CLIP_LO = 0.25
AV_CLIP_HI = 4.0
AV_CLASSIFIER_MAX_TRAIN_ROWS = 300_000  # subsample for fitting speed
AV_WEIGHTS_PATH = MODELS_DIR / "av_weights.csv"

# Rank-normalize the 6 most KS-drifted daily features against the train∪test
# pooled ECDF. Targets the val/Kaggle gap (Section 9 finding).
USE_RANK_NORMALIZATION = True
RANK_NORMALIZE_FEATURES = [
    # Original 3 (Phase 1a)
    "humidity", "wb_tmp", "tmp_range",
    # Phase 2: extended to the 5 temperature/dew-point features with KS≥0.18
    # against test. Section 9 of the EDA snapshot shows train/test mean
    # deltas of +4-6.5°C on these. Pooled-ECDF rank-norm aligns their
    # marginal distributions so interaction/anomaly features that consume
    # raw scales aren't biased by the seasonal shift.
    "tmp", "tmp_max", "surf_tmp", "tmp_min", "dp_tmp",
]

# Severe-row gradient upweighting for L1 LightGBM. Loss family unchanged; the
# weight on each row scales its absolute-residual contribution.
# weight = 1 + α·𝟙[y>0] + β·𝟙[y≥3].
SAMPLE_WEIGHT_ALPHA = 1.0
SAMPLE_WEIGHT_BETA = 2.0

# Walk-forward CV diagnostic. When True, after the main training+eval,
# train.py runs a second pass with multiple temporal folds (lower n_estimators
# for speed) and reports their average macro-MAE as a more honest signal.
# Disabled for Phase 1b iteration speed; re-enable for final validation.
RUN_WALK_FORWARD_DIAGNOSTIC = False
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

N_WORKERS = max(1, os.cpu_count() - 3)  # leave 3 cores for system / LightGBM overhead
