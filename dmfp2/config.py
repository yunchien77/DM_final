"""dmfp2 — single source of truth.

Flat, frozen dataclasses of toggles. An ablation is one flag, not a new file.
Cache keys derive from the relevant config's hash, so stale caches are never
silently reused. Deliberately the opposite of the phase-annotated config in
../code/config.py.

Run everything with the DMFP conda env:
    /mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/python -m dmfp2.cli ...
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent          # .../DMFP/dmfp2
ROOT = PKG_DIR.parent                               # .../DMFP
DATA_DIR = ROOT / "dataset" / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

CACHE_DIR = PKG_DIR / "cache"
ARTIFACTS_DIR = PKG_DIR / "artifacts"
REPORTS_DIR = PKG_DIR / "reports"
for _d in (CACHE_DIR, ARTIFACTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SUBMISSION_PATH = ROOT / "submission.csv"
VAL_LB_LOG = REPORTS_DIR / "val_lb_log.csv"

# ---------------------------------------------------------------------------
# Data layout (verified against train.csv / test.csv headers)
# ---------------------------------------------------------------------------
ID_COL = "region_id"
DATE_COL = "date"
SCORE_COL = "score"

# Core meteo channels kept by default. dp_tmp (corr 1.0 with wb_tmp) and the
# wind family (low gain in EDA) are dropped initially; re-enable via DataConfig.
METEO_CORE = (
    "prec", "surf_pre", "humidity", "tmp",
    "wb_tmp", "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
)
METEO_OPTIONAL = ("dp_tmp", "wind", "wind_max", "wind_min", "wind_range")
ALL_METEO = METEO_CORE + METEO_OPTIONAL

# Within-week aggregation stats per channel.
WEEK_STATS = ("mean", "std", "min", "max", "sum")

DAYS_PER_WEEK = 7
N_PRED = 5
HORIZONS = (1, 2, 3, 4, 5)
SUB_COLS = [f"pred_week{h}" for h in HORIZONS]

N_REGIONS = 2248
SEED = 42
N_WORKERS = max(1, (os.cpu_count() or 4) - 2)

# Train LGBM on a per-region random subsample of anchors (climatology transforms still
# fit on the FULL pool). Matches the teammate's ~200-windows/region sampling: faster and
# less in-distribution overfitting. None = use all anchors.
TRAIN_SAMPLE_PER_REGION = 250
# Drop shift-carrying region-baseline features (region anomalies / region-baseline physical),
# which improve in-distribution fit but HURT out-of-support transfer (phase4 0.9644). Lean set wins.
LEAN_FEATURES = True


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    """Controls daily preprocessing + weekly aggregation (the weekly cache key)."""
    channels: tuple = METEO_CORE
    winsorize: bool = True
    winsor_lower_pct: float = 1.0
    winsor_upper_pct: float = 99.0
    log1p_cols: tuple = ("prec",)
    sqrt_cols: tuple = ("surf_pre",)


@dataclass(frozen=True)
class FeatureConfig:
    """Controls the feature matrix (the feature cache key)."""
    include_raw_levels: bool = True          # shift-SENSITIVE rolling levels
    include_local_anomaly: bool = True       # Family A — shift-invariant by construction
    include_climatology_anomaly: bool = False  # Family B — mitigating, leaky if naive
    include_region_anom: bool = True         # current window vs region TRAIN baseline (carries the shift as signal)
    include_physical: bool = True            # VPD, DTR, heat-dd, aridity, precip deficit...
    proxy_scales: tuple = (4, 8, 12)         # multi-scale proxy (current-state anchor)
    include_calendar: bool = True            # sin/cos(doy, month), quarter
    include_region_scalars: bool = True      # train-fold-only history stats
    windows: tuple = (4, 8, 12)              # in WEEKS
    lookback_weeks: int = 12                 # min anchor history required
    use_proxy: bool = True
    proxy_inputs: str = "A_only"             # {"A_only", "A_and_B", "none"}
    proxy_samples_per_region: int = 40


@dataclass(frozen=True)
class ValidationConfig:
    # selection of the held-out anchor per region:
    #   "precip_deficit" — driest in-season week (region-wide DROUGHT state). Reproduces the
    #       test's region_mean<zero property (LB-calibrated: zero-MAE 1.11, rm-MAE 1.04 vs
    #       LB 1.2088/0.9364). The principled default.
    #   "temp_match" — match anchor temp to (region test temp + temp_offset). Matches severity
    #       but NOT region_mean<zero (cherry-picks within-region anomalies). Ablation only.
    selection: str = "precip_deficit"
    temp_offset: float = 4.0                 # used only by temp_match
    temp_channel: str = "tmp_max"            # channel used for temp_match / hotness reporting
    calendar_slack_weeks: int = 6            # ± weeks around region test woy for in-season
    one_anchor_per_region: bool = True       # mirror Kaggle's 2248-row structure
    purge_weeks: int = 13
    n_folds: int = 4                          # walk-forward inner folds on the cool portion


@dataclass(frozen=True)
class TrainSubsetConfig:
    """"Use the data that matters most" — test-likeness reweighting/selection."""
    mode: str = "all"                         # {"all", "weight", "select"}
    strength: float = 1.0                     # exponent on test-likeness for weight mode
    weight_clip: tuple = (0.25, 4.0)
    select_top_frac: float = 0.6              # keep top fraction for select mode


@dataclass(frozen=True)
class ModelConfig:
    horizon_sharing: str = "independent"  # {"independent" (per-horizon + in-season ES), "horizon_feature"}
    # Target-severity sample weighting. dmfp2 originally kept this OFF (phase10 showed +0.42
    # "bias") — but that bias was measured on the COOL in-distribution val; the real test is
    # HOT/droughty (true-mean 1.21) where L1 UNDER-predicts, so the upward push is CORRECTIVE.
    # Lever test: severity is THE high-end-discrimination lever (nails ref droughty-tercile 2.4
    # + frac_low 0.22). Pair with per-horizon (decay) to bring the overshot mean back to ~1.31.
    severity: bool = True
    sev_nonzero: float = 1.5                  # weight for 0 < score < 3
    sev_severe: float = 3.0                   # weight for score >= 3
    monotone_trajectory: bool = True          # smooth w1..w5 path post-hoc
    use_isotonic: bool = True                 # accepted only if it helps the HOT holdout
    use_markov_smoothing: bool = False
    markov_beta: float = 0.8


LGBM_PARAMS = dict(
    objective="regression_l1",
    metric="mae",
    n_estimators=1500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=8,            # capped: the box is shared/oversubscribed (load ~30 on 24 cores);
    random_state=SEED,   # fewer threads = less thrashing, often faster under contention
    verbose=-1,
)
EARLY_STOPPING_ROUNDS = 100


# ---------------------------------------------------------------------------
# Default singletons (import these elsewhere; override in cli for ablations)
# ---------------------------------------------------------------------------
DATA = DataConfig()
FEATURES = FeatureConfig()
VALIDATION = ValidationConfig()
TRAIN_SUBSET = TrainSubsetConfig()
MODEL = ModelConfig()


# ---------------------------------------------------------------------------
# Hashing — cache keys
# ---------------------------------------------------------------------------
def _hash(*objs) -> str:
    h = hashlib.sha1()
    for o in objs:
        h.update(repr(o).encode())
    return h.hexdigest()[:10]


def weekly_hash(data: DataConfig = DATA) -> str:
    """Cache key for the weekly-aggregated parquet/pickle (depends on data prep only)."""
    return _hash("weekly_v1", asdict(data))


def feature_hash(features: FeatureConfig = FEATURES, data: DataConfig = DATA) -> str:
    """Cache key for the feature matrix (depends on features + upstream weekly).

    Bump the version string whenever add_features() logic changes (not just its config).
    """
    return _hash("feat_v3_regionanom_physical_multiproxy", asdict(features), weekly_hash(data))


def config_banner(features=FEATURES, data=DATA, validation=VALIDATION,
                  train_subset=TRAIN_SUBSET, model=MODEL) -> str:
    return (
        f"weekly_hash={weekly_hash(data)} feature_hash={feature_hash(features, data)}\n"
        f"  DATA={asdict(data)}\n"
        f"  FEATURES={asdict(features)}\n"
        f"  VALIDATION={asdict(validation)}\n"
        f"  TRAIN_SUBSET={asdict(train_subset)}\n"
        f"  MODEL={asdict(model)}"
    )
