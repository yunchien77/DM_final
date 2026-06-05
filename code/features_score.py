"""Phase 7 — score-derived features.

The minimal set we keep alongside the climate / windowed-stats features:

  - score_lag1            : single-step Markov anchor (12-week shift). Currently
                            disabled at the feature level (USE_SCORE_LAG=False)
                            — the train↔test gap of ~163 weeks makes any short
                            shift a phantom anchor. Column still computed for
                            transition-weight code that references it.
  - region history        : region_score_mean / _std / _zero_rate / _max,
                            computed from training-fold rows only.
  - calendar              : month/doy sin/cos.
  - score climatology     : per (region, woy) mean/std/max/p90/zero_rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MODELS_DIR, TEST_WEEKS_PER_REGION

# Test rows predict 5 weeks ahead after a 13-week unscored block. The legacy
# shift matches test's unscored window only (not the full train↔test gap).
SCORE_LAG_SHIFT = TEST_WEEKS_PER_REGION - 1   # 12

REGION_STATS_PATH = MODELS_DIR / "region_stats.csv"
SCORE_CLIMATOLOGY_PATH = MODELS_DIR / "score_climatology.csv"
REGION_LAST_SCORE_PATH = MODELS_DIR / "region_last_score.csv"

CALENDAR_FEATURE_COLS = ["month_sin", "month_cos", "doy_sin", "doy_cos"]
REGION_FEATURE_COLS = ["region_id_int", "region_score_mean", "region_score_std",
                       "region_zero_rate", "region_score_max"]
SCORE_CLIM_FEATURE_COLS = ["region_woy_score_mean", "region_woy_score_std",
                           "region_woy_score_max", "region_woy_score_p90",
                           "region_woy_zero_rate"]
# Phase 8: score_lag1 is the phantom anchor — kept as a column for transition-
# weight computation but excluded from feature_cols when USE_SCORE_LAG=False.
SCORE_LAG_FEATURE_COLS = ["score_lag1"]

# Single static column list — for analysis / cache-key hashing.
SCORE_FEATURE_COLS = (
    SCORE_LAG_FEATURE_COLS
    + REGION_FEATURE_COLS
    + CALENDAR_FEATURE_COLS
    + SCORE_CLIM_FEATURE_COLS
)


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic month + doy embeddings."""
    out = df.copy()
    month = out["month"].to_numpy(np.float32)
    doy = out["day_of_year"].to_numpy(np.float32)
    out["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    out["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365).astype(np.float32)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Region history (train-fold only)
# ---------------------------------------------------------------------------

def compute_region_stats(weekly_df_train: pd.DataFrame) -> pd.DataFrame:
    """Per-region historical score statistics from the TRAINING fold only."""
    scored = weekly_df_train[weekly_df_train["score"].notna()]
    stats = scored.groupby("region_id")["score"].agg(
        region_score_mean="mean",
        region_score_std="std",
        region_score_max="max",
    )
    zero_rate = (
        scored.groupby("region_id")["score"]
              .apply(lambda x: (x == 0).mean())
              .rename("region_zero_rate")
    )
    out = pd.concat([stats, zero_rate], axis=1).reset_index()
    for c in ("region_score_mean", "region_score_std", "region_zero_rate", "region_score_max"):
        out[c] = out[c].fillna(0.0).astype(np.float32)
    return out


def add_region_features(df: pd.DataFrame, region_stats: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["region_id_int"] = (
        out["region_id"].astype(str).str.replace("R", "", regex=False).astype(np.int32)
    )
    out = out.merge(region_stats, on="region_id", how="left")
    for c in ("region_score_mean", "region_score_std", "region_zero_rate", "region_score_max"):
        out[c] = out[c].fillna(0.0).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Per-(region, woy) score climatology
# ---------------------------------------------------------------------------

def _woy_from_doy(doy) -> np.ndarray:
    a = np.asarray(doy, dtype=np.int32)
    return np.clip((a - 1) // 7, 0, 52).astype(np.int16)


def compute_score_climatology(weekly_df_train: pd.DataFrame) -> pd.DataFrame:
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


def add_score_climatology(df: pd.DataFrame, score_clim: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["__woy"] = _woy_from_doy(out["day_of_year"])
    out = out.merge(score_clim, left_on=["region_id", "__woy"],
                    right_on=["region_id", "woy"], how="left")
    clim_cols = [c for c in score_clim.columns if c not in ("region_id", "woy")]
    for c in clim_cols:
        if c in out.columns:
            out[c] = out[c].fillna(out[c].median()).astype(np.float32)
    out = out.drop(columns=["__woy", "woy"], errors="ignore")
    return out


# ---------------------------------------------------------------------------
# Score lag-1 with the test-style staleness shift
# ---------------------------------------------------------------------------

def _per_region_score_arrays(weekly_df: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for rid, reg in weekly_df.groupby("region_id", sort=False):
        out[rid] = reg.sort_values("week_idx")["score"].values.astype(np.int16)
    return out


def add_score_lag1_train(
    features_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    shift: int = SCORE_LAG_SHIFT,
) -> pd.DataFrame:
    """Add score_lag1 to a training feature matrix by looking up the score at
    (region, week_idx − 1 − shift). With shift=12 the (lag-source → first-target)
    distance becomes 14, matching test's information staleness.
    """
    out = features_df.copy()
    score_by_region = _per_region_score_arrays(weekly_df)
    region_ids = out["region_id"].to_numpy()
    week_idxs = out["week_idx"].to_numpy()
    vals = np.zeros(len(out), dtype=np.float32)
    for i in range(len(out)):
        rid = region_ids[i]
        wi = int(week_idxs[i])
        scores = score_by_region.get(rid)
        if scores is None:
            continue
        idx = wi - 1 - shift
        if 0 <= idx < len(scores):
            vals[i] = float(scores[idx])
    out["score_lag1"] = vals
    return out


def compute_region_last_score(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """For each region, the score at the last scored anchor — this is what a
    test row would see at lag 1 (no shift; it's the genuinely most recent score)."""
    score_by_region = _per_region_score_arrays(weekly_df)
    rows = []
    for rid, scores in score_by_region.items():
        n = len(scores)
        rows.append({
            "region_id": rid,
            "score_lag1": float(scores[n - 1]) if n > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def add_score_lag1_test(features_df: pd.DataFrame, region_last_score: pd.DataFrame) -> pd.DataFrame:
    out = features_df.merge(region_last_score, on="region_id", how="left")
    out["score_lag1"] = out["score_lag1"].fillna(0.0).astype(np.float32)
    return out
