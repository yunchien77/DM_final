from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import METEO_COLS, PREPROC_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Transform functions
# ─────────────────────────────────────────────────────────────────────────────

_TRANSFORMS: dict[str, object] = {
    "log1p":        lambda x: np.log1p(np.maximum(x, 0.0)),
    "signed_log1p": lambda x: np.sign(x) * np.log1p(np.abs(x)),
    "sqrt":         lambda x: np.sign(x) * np.sqrt(np.abs(x)),
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Winsorization
# ─────────────────────────────────────────────────────────────────────────────

def compute_winsor_bounds(
    df: pd.DataFrame,
    features: list[str] = METEO_COLS,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
) -> dict[str, tuple[float, float]]:
    """Compute (p_lo, p_hi) clip bounds from the training DataFrame only."""
    bounds: dict[str, tuple[float, float]] = {}
    for f in features:
        if f not in df.columns:
            continue
        col = df[f].dropna()
        if len(col) == 0:
            bounds[f] = (float("-inf"), float("inf"))
        else:
            bounds[f] = (float(col.quantile(q_lo)), float(col.quantile(q_hi)))
    return bounds


def apply_winsorization(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    features: list[str] = METEO_COLS,
) -> pd.DataFrame:
    """Clip each feature to its precomputed bounds; returns a copy."""
    out = df.copy()
    for f in features:
        if f in bounds and f in out.columns:
            lo, hi = bounds[f]
            out[f] = out[f].clip(lo, hi)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Heavy-tail transforms
# ─────────────────────────────────────────────────────────────────────────────

def apply_transforms(
    df: pd.DataFrame,
    log_features: dict[str, str],
) -> pd.DataFrame:
    """Apply named transforms to specified features (e.g. {"prec": "log1p"})."""
    out = df.copy()
    for f, kind in log_features.items():
        if kind not in _TRANSFORMS:
            raise ValueError(f"Unknown transform {kind!r} for feature {f!r}. "
                             f"Valid: {list(_TRANSFORMS)}")
        if f in out.columns:
            out[f] = (_TRANSFORMS[kind](out[f].values.astype(np.float64))
                      .astype(np.float32))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Missing value imputation (per-region median, global fallback)
# ─────────────────────────────────────────────────────────────────────────────

def compute_imputation_table(
    df: pd.DataFrame,
    by: str = "region_id",
    features: list[str] = METEO_COLS,
) -> pd.DataFrame:
    """Build per-region median table with a 'GLOBAL' row as fallback."""
    present = [f for f in features if f in df.columns]
    region_med = df.groupby(by)[present].median()
    global_med = df[present].median()
    region_med.loc["GLOBAL"] = global_med
    return region_med.astype(np.float32)


def apply_imputation(
    df: pd.DataFrame,
    table: pd.DataFrame,
    by: str = "region_id",
    features: list[str] = METEO_COLS,
) -> pd.DataFrame:
    """Fill NaNs with per-region medians; fall back to global median if the region is unknown."""
    out = df.copy()
    if by not in out.columns:
        return out
    no_global = table.drop(index="GLOBAL", errors="ignore")
    merge = out[[by]].merge(no_global, left_on=by, right_index=True, how="left")
    global_row = table.loc["GLOBAL"] if "GLOBAL" in table.index else None
    present = [f for f in features if f in out.columns]
    for f in present:
        fill = merge[f].values
        if global_row is not None:
            fill = np.where(pd.isnull(fill), float(global_row[f]), fill)
        out[f] = out[f].fillna(pd.Series(fill, index=out.index))
        if global_row is not None and out[f].isnull().any():
            out[f] = out[f].fillna(float(global_row[f]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rank normalization (pooled train∪test ECDF)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rank_quantiles(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    n_quantiles: int = 1000,
) -> dict[str, np.ndarray]:
    """Build quantile anchors from pooled train+test data for rank normalization."""
    probs = np.linspace(0.0, 1.0, n_quantiles)
    out: dict[str, np.ndarray] = {}
    for f in features:
        tr_vals = df_train[f].dropna().values if f in df_train.columns else np.array([])
        te_vals = df_test[f].dropna().values  if f in df_test.columns  else np.array([])
        if len(tr_vals) == 0 and len(te_vals) == 0:
            out[f] = np.array([0.0], dtype=np.float32)
        else:
            pooled = np.concatenate([tr_vals, te_vals])
            out[f] = np.quantile(pooled, probs).astype(np.float32)
    return out


def apply_rank_normalization(
    df: pd.DataFrame,
    quantile_table: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Map each feature to [0, 1] using the pooled ECDF quantile anchors."""
    out = df.copy()
    for f, anchors in quantile_table.items():
        if f not in out.columns or len(anchors) <= 1:
            continue
        vals = out[f].values.astype(np.float64)
        ranks = np.searchsorted(anchors, vals, side="left").astype(np.float32)
        out[f] = (ranks / (len(anchors) - 1)).astype(np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline (single call)
# ─────────────────────────────────────────────────────────────────────────────

def apply_pipeline(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    log_features: dict[str, str],
    table: pd.DataFrame,
    quantile_table: dict[str, np.ndarray] | None = None,
    features: list[str] = METEO_COLS,
) -> pd.DataFrame:
    """Apply imputation → winsorization → transform → rank-norm in sequence."""
    df = apply_imputation(df, table, features=features)
    df = apply_winsorization(df, bounds, features=features)
    df = apply_transforms(df, log_features)
    if quantile_table:
        df = apply_rank_normalization(df, quantile_table)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_preprocessing_artifacts(
    bounds: dict,
    log_features: dict[str, str],
    table: pd.DataFrame,
    quantile_table: dict[str, np.ndarray] | None = None,
    path: Path = PREPROC_PATH,
) -> None:
    """Atomically pickle all preprocessing artifacts to a single file."""
    obj = dict(
        bounds=bounds,
        log_features=log_features,
        imputation_table=table,
        quantile_table=quantile_table or {},
    )
    tmp = Path(str(path) + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_preprocessing_artifacts(
    path: Path = PREPROC_PATH,
) -> tuple[dict, dict[str, str], pd.DataFrame, dict[str, np.ndarray]]:
    """Load preprocessing artifacts; returns (bounds, log_features, table, quantile_table)."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return (
        obj["bounds"],
        obj["log_features"],
        obj["imputation_table"],
        obj.get("quantile_table", {}),
    )
