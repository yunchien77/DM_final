"""
Extra feature blocks layered on top of features.py.

Phase 2 of the enhancement plan: targeted features aimed at the failure modes
identified in the diagnostic (severe underprediction of extreme events,
strong region/season effects).

All training-fold-dependent statistics are computed strictly from the rows
the caller passes in — leakage prevention is the caller's responsibility
(same contract as features.compute_region_stats).

Blocks in this module:
  1. compute_woy_score_climatology / add_woy_score_climatology
     Per (region, week_of_year) historical score stats. Mean / std / max
     of integer scores over the training fold.

  2. compute_woy_meteo_climatology / add_anomaly_features
     Per (region, week_of_year) historical meteo means and stds for a small
     set of high-signal features; current-week values are then standardized
     against them to flag anomalies.

  3. add_trend_features
     Linear-regression slope of `prec_mean_lag*` and `tmp_max_mean_lag*` over
     the last 4 and 8 weekly lags. Captures escalation/de-escalation.

  4. add_interaction_features
     Multiplicative cross-feature combos at lag 0: prec×wind, hot-dry, etc.

  5. EXTRA_FEATURE_COLS_V2
     Updated column list that train.py / predict.py should consume after
     applying features.add_temporal_features + add_region_features + this
     module's adders.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import METEO_FEATURES, MODELS_DIR


# ---------------------------------------------------------------------------
# week-of-year helper
# ---------------------------------------------------------------------------

def _woy_from_doy(doy: pd.Series) -> pd.Series:
    """Map day-of-year (1..365) to a week-of-year bucket 0..52."""
    return ((doy.astype(np.int32) - 1) // 7).clip(0, 52).astype(np.int16)


# ---------------------------------------------------------------------------
# Block 1: per-(region, week_of_year) score climatology
# ---------------------------------------------------------------------------

def compute_woy_score_climatology(weekly_df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Per-region-week-of-year score statistics over the training fold.

    Returns DataFrame with columns:
        region_id, woy, region_woy_score_mean, region_woy_score_std,
        region_woy_score_max, region_woy_score_p90, region_woy_zero_rate
    """
    df = weekly_df_train.copy()
    df["woy"] = _woy_from_doy(df["day_of_year"])

    grouped = df.groupby(["region_id", "woy"])["score"]
    out = pd.DataFrame({
        "region_woy_score_mean": grouped.mean(),
        "region_woy_score_std": grouped.std().fillna(0.0),
        "region_woy_score_max": grouped.max(),
        "region_woy_score_p90": grouped.quantile(0.90),
        "region_woy_zero_rate": grouped.apply(lambda x: (x == 0).mean()),
    }).reset_index()
    for c in out.columns:
        if c not in ("region_id", "woy"):
            out[c] = out[c].astype(np.float32)
    return out


def add_woy_score_climatology(
    features_df: pd.DataFrame,
    climatology: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join climatology onto features by (region_id, woy)."""
    df = features_df.copy()
    df["woy"] = _woy_from_doy(df["day_of_year"])
    df = df.merge(climatology, on=["region_id", "woy"], how="left")

    # For region-woy cells with no training data, fall back to per-region mean,
    # then per-feature global median.
    clim_cols = [c for c in climatology.columns if c not in ("region_id", "woy")]
    for c in clim_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median()).astype(np.float32)
    df = df.drop(columns=["woy"])
    return df


# ---------------------------------------------------------------------------
# Block 2: meteo climatology + anomaly features
# ---------------------------------------------------------------------------

# Subset of meteo to standardize against historical norms.
ANOMALY_FEATURES = ["tmp_max", "tmp_min", "prec", "humidity", "wind_max", "surf_pre"]


def compute_woy_meteo_climatology(weekly_df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Per (region, woy) historical mean/std of selected weekly meteo means.

    Output columns: region_id, woy, then for each f in ANOMALY_FEATURES:
        clim_{f}_mean   (mean of f_mean across training years for that woy)
        clim_{f}_std    (std of  f_mean, floored at 1e-3 to avoid div-by-zero)
    """
    df = weekly_df_train.copy()
    df["woy"] = _woy_from_doy(df["day_of_year"])
    agg_cols = [f"{f}_mean" for f in ANOMALY_FEATURES]
    grouped = df.groupby(["region_id", "woy"])[agg_cols]
    means = grouped.mean()
    stds = grouped.std().fillna(1e-3).clip(lower=1e-3)
    means.columns = [f"clim_{f}_mean" for f in ANOMALY_FEATURES]
    stds.columns = [f"clim_{f}_std" for f in ANOMALY_FEATURES]
    out = pd.concat([means, stds], axis=1).reset_index()
    for c in out.columns:
        if c not in ("region_id", "woy"):
            out[c] = out[c].astype(np.float32)
    return out


def add_anomaly_features(
    features_df: pd.DataFrame,
    meteo_climatology: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each anchor row, add standardized anomalies:
        anom_{f} = ({f}_mean_lag0 - clim_{f}_mean) / clim_{f}_std
    """
    df = features_df.copy()
    df["woy"] = _woy_from_doy(df["day_of_year"])
    df = df.merge(meteo_climatology, on=["region_id", "woy"], how="left")

    for f in ANOMALY_FEATURES:
        clim_mean = df[f"clim_{f}_mean"].fillna(df[f"clim_{f}_mean"].median())
        clim_std = df[f"clim_{f}_std"].fillna(1e-3).clip(lower=1e-3)
        df[f"anom_{f}"] = ((df[f"{f}_mean_lag0"] - clim_mean) / clim_std).astype(np.float32)

    df = df.drop(columns=["woy"])
    return df


# ---------------------------------------------------------------------------
# Block 3: trend / momentum features (slope of last N lags)
# ---------------------------------------------------------------------------

TREND_FEATURES = ["prec_mean", "tmp_max_mean", "tmp_range_mean", "wind_max_mean", "humidity_mean"]


def _slope_over_lags(df: pd.DataFrame, base: str, lags: list[int]) -> pd.Series:
    """
    Slope of base values across the given lags via least-squares.
    Lags are listed from most recent to least; we reverse so x increases over time.
    """
    x = np.arange(len(lags), dtype=np.float32)  # 0,1,2,...
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    cols = [f"{base}_lag{l}" for l in lags]
    Y = df[cols].values.astype(np.float32)
    # Reverse along axis so column-0 = oldest, last column = most recent.
    Y = Y[:, ::-1]
    y_mean = Y.mean(axis=1, keepdims=True)
    slope = ((Y - y_mean) * (x - x_mean)).sum(axis=1) / x_var
    return pd.Series(slope, index=df.index, dtype=np.float32)


def add_trend_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add 4-week and 8-week slopes over the most recent lag values."""
    df = features_df.copy()
    for base in TREND_FEATURES:
        df[f"slope4_{base}"] = _slope_over_lags(df, base, lags=[1, 2, 3, 4])
        df[f"slope8_{base}"] = _slope_over_lags(df, base, lags=[1, 2, 3, 4, 5, 6, 7, 8])
    return df


# ---------------------------------------------------------------------------
# Block 4: interaction features (cross-feature products at lag 0)
# ---------------------------------------------------------------------------

def add_interaction_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Hand-picked interactions for fire-weather / extreme-condition signatures."""
    df = features_df.copy()
    df["int_prec_wind"] = (df["prec_mean_lag0"] * df["wind_max_lag0"]).astype(np.float32)
    df["int_hot_dry"] = (df["tmp_max_mean_lag0"] * (1.0 - df["humidity_mean_lag0"])).astype(np.float32)
    df["int_tmprange_wind"] = (df["tmp_range_mean_lag0"] * df["wind_max_lag0"]).astype(np.float32)
    df["int_prec_humid"] = (df["prec_mean_lag0"] * df["humidity_mean_lag0"]).astype(np.float32)
    df["int_low_pressure"] = (-df["surf_pre_min_lag0"] * df["wind_max_lag0"]).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Final feature column list
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Block 5: lagged-score features (target leakage-safe)
# ---------------------------------------------------------------------------
#
# Section 10's transition matrices show P(score[t+1]=0 | score[t]=0)=0.963 and
# P(score[t+1]≥3 | score[t]≥3)=0.967 — score persistence is the single
# strongest signal in the data and the current feature set ignores it.
#
# For a training anchor `a`, we use scores at week_idx a-1, a-2, a-4, a-8.
# These are FROM THE PAST relative to the anchor, so leakage-safe even though
# we predict scores at week_idx a+1..a+5.
#
# Test rows are the per-region last anchor — for them, "the score at lag 1"
# is the score at the very last scored anchor in train. We persist the last
# 8 anchor scores per region to `region_last_scores.csv` after training.

SCORE_LAG_OFFSETS = [1, 2, 4, 8]
SCORE_LAG_COLS = (
    [f"score_lag{k}" for k in SCORE_LAG_OFFSETS]
    + ["weeks_since_last_nonzero", "max_score_prev8"]
)


def _per_region_score_arrays(weekly_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return {region_id: scores_array sorted by week_idx}."""
    out = {}
    for rid, reg in weekly_df.groupby("region_id", sort=False):
        reg_sorted = reg.sort_values("week_idx")
        out[rid] = reg_sorted["score"].values.astype(np.int16)
    return out


def add_score_lag_features_train(
    features_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add SCORE_LAG_COLS to a training feature matrix by looking up the score at
    (region_id, week_idx − lag) in `weekly_df`. weekly_df must be the same
    object that produced features_df (1 row per region per scored week).
    """
    df = features_df.copy()
    score_by_region = _per_region_score_arrays(weekly_df)

    out_cols = {c: np.zeros(len(df), dtype=np.float32) for c in SCORE_LAG_COLS}
    region_ids = df["region_id"].values
    week_idxs = df["week_idx"].values

    for i in range(len(df)):
        rid = region_ids[i]
        wi = int(week_idxs[i])
        scores = score_by_region[rid]
        for k in SCORE_LAG_OFFSETS:
            idx = wi - k
            out_cols[f"score_lag{k}"][i] = scores[idx] if 0 <= idx < len(scores) else 0.0

        # weeks_since_last_nonzero: scan back from wi-1 until we hit a nonzero
        # score. Cap at 26 weeks; if none found, 26.
        last_nonzero = 26
        for back in range(1, 27):
            idx = wi - back
            if idx < 0:
                break
            if scores[idx] > 0:
                last_nonzero = back
                break
        out_cols["weeks_since_last_nonzero"][i] = float(last_nonzero)

        # max_score_prev8: window of [wi-8, wi-1].
        lo = max(0, wi - 8)
        if lo < wi:
            out_cols["max_score_prev8"][i] = float(scores[lo:wi].max())
        else:
            out_cols["max_score_prev8"][i] = 0.0

    for c, arr in out_cols.items():
        df[c] = arr
    return df


def compute_region_last_score_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each region, derive the score-lag features that a test anchor would see
    (i.e., as if its week_idx = max_train_week_idx + 1). Saved alongside
    training and consumed at inference.

    Output columns: region_id, score_lag1, score_lag2, score_lag4, score_lag8,
    weeks_since_last_nonzero, max_score_prev8.
    """
    score_by_region = _per_region_score_arrays(weekly_df)
    rows = []
    for rid, scores in score_by_region.items():
        n = len(scores)
        row = {"region_id": rid}
        for k in SCORE_LAG_OFFSETS:
            idx = n - k  # last scored week is at index n-1; "lag1" = scores[n-1]
            row[f"score_lag{k}"] = float(scores[idx]) if 0 <= idx < n else 0.0
        last_nonzero = 26
        for back in range(1, 27):
            idx = n - back
            if idx < 0:
                break
            if scores[idx] > 0:
                last_nonzero = back
                break
        row["weeks_since_last_nonzero"] = float(last_nonzero)
        lo = max(0, n - 8)
        row["max_score_prev8"] = float(scores[lo:n].max()) if lo < n else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def add_score_lag_features_test(
    features_df: pd.DataFrame,
    region_last_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-region lagged-score features onto the test feature matrix."""
    df = features_df.merge(region_last_scores, on="region_id", how="left")
    for c in SCORE_LAG_COLS:
        df[c] = df[c].fillna(0.0).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Block 6: region-cluster archetype features
# ---------------------------------------------------------------------------
#
# Section 11 surfaces 8 region archetypes (quiet, mostly-quiet, moderate,
# volatile, high-severity); ~16% of regions sit in the "high-severity" cluster
# and contribute most of the rare-event MAE. Adding cluster identity gives
# LightGBM a coarse categorical handle that complements per-region scalars.

CLUSTER_FEATURE_COLS = [
    "region_cluster_id",
    "cluster_score_mean",
    "cluster_zero_rate",
]
REGION_CLUSTERS_CSV = MODELS_DIR / "region_clusters.csv"


def fit_region_clusters(
    weekly_df_train: pd.DataFrame,
    n_clusters: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    K-means on per-region score profile (zero_rate, mean, std, max, p90).
    Returns DataFrame keyed by region_id with cluster_id and per-cluster
    target-encoded score statistics. Trains on training-fold rows only.
    """
    scored = weekly_df_train[weekly_df_train["score"].notna()]
    feats = scored.groupby("region_id")["score"].agg(
        zero_rate=lambda x: (x == 0).mean(),
        mean_score="mean",
        std_score=lambda x: x.std() if len(x) > 1 else 0.0,
        max_score="max",
        p90_score=lambda x: x.quantile(0.90),
    ).fillna(0.0).reset_index()

    X = StandardScaler().fit_transform(feats[
        ["zero_rate", "mean_score", "std_score", "max_score", "p90_score"]
    ].values)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    feats["region_cluster_id"] = km.fit_predict(X).astype(np.int8)

    cluster_stats = feats.groupby("region_cluster_id").agg(
        cluster_score_mean=("mean_score", "mean"),
        cluster_zero_rate=("zero_rate", "mean"),
    ).reset_index()
    feats = feats.merge(cluster_stats, on="region_cluster_id", how="left")

    out = feats[["region_id"] + CLUSTER_FEATURE_COLS].copy()
    for c in ("cluster_score_mean", "cluster_zero_rate"):
        out[c] = out[c].astype(np.float32)
    return out


def add_region_cluster_features(
    features_df: pd.DataFrame,
    cluster_table: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cluster_id and cluster-level target-encoded stats onto a feature matrix."""
    df = features_df.merge(cluster_table, on="region_id", how="left")
    if df["region_cluster_id"].isnull().any():
        # Fall back to a sentinel cluster for unseen regions.
        df["region_cluster_id"] = df["region_cluster_id"].fillna(-1).astype(np.int8)
        df["cluster_score_mean"] = df["cluster_score_mean"].fillna(
            cluster_table["cluster_score_mean"].mean()
        ).astype(np.float32)
        df["cluster_zero_rate"] = df["cluster_zero_rate"].fillna(
            cluster_table["cluster_zero_rate"].mean()
        ).astype(np.float32)
    return df


# ---------------------------------------------------------------------------


EXTRA_FEATURE_COLS_V2 = (
    # — Block 1: score climatology
    [
        "region_woy_score_mean",
        "region_woy_score_std",
        "region_woy_score_max",
        "region_woy_score_p90",
        "region_woy_zero_rate",
    ]
    # — Block 2: meteo anomalies
    + [f"anom_{f}" for f in ANOMALY_FEATURES]
    # — Block 3: trend / momentum
    + [f"slope4_{b}" for b in TREND_FEATURES]
    + [f"slope8_{b}" for b in TREND_FEATURES]
    # — Block 4: interactions
    + [
        "int_prec_wind",
        "int_hot_dry",
        "int_tmprange_wind",
        "int_prec_humid",
        "int_low_pressure",
    ]
)


def apply_extra_features(
    features_df: pd.DataFrame,
    weekly_df_train: pd.DataFrame,
    score_climatology: pd.DataFrame = None,
    meteo_climatology: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Single entry point that applies all four blocks in order.

    If climatologies are not supplied, they will be computed from
    `weekly_df_train` (must already be filtered to the training fold).
    """
    if score_climatology is None:
        score_climatology = compute_woy_score_climatology(weekly_df_train)
    if meteo_climatology is None:
        meteo_climatology = compute_woy_meteo_climatology(weekly_df_train)

    df = add_woy_score_climatology(features_df, score_climatology)
    df = add_anomaly_features(df, meteo_climatology)
    df = add_trend_features(df)
    df = add_interaction_features(df)
    return df
