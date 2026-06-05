"""Phase 8 — meteo-only proxy score.

The train↔test gap on this dataset averages ~163 weeks; score autocorrelation
0.96^163 ≈ 0.001. A direct `score_lag1` feature is predictive on val (because
we shift the lag-source by 12 weeks, leaving ~14-week autocorrelation ≈ 0.59)
and useless on Kaggle (where the actual gap is much longer). The result is a
val→Kaggle decoupling that we've fought repeatedly.

The teammate's fix (README): replace the lag with a **proxy score** — a Ridge
regression that maps purely meteorological features to the anchor week's
drought score. Critically, this feature has IDENTICAL semantics at train and
test (the meteo inputs are real both places) and serves the same role as the
score lag in the LGBM (a single scalar that condenses the drought-relevant
signal of the moment).

This module:
- `fit_proxy_ridge(train_features, target, feature_cols)` — fits a RidgeCV
  on the meteo feature subset and persists it as a pickle. Tuned α via
  GroupKFold-style time-aware CV (sklearn's RidgeCV is the simplest path).
- `add_proxy_score(features_df, model, feature_cols)` — applies the model
  to produce the `proxy_score` column.
- `meteo_feature_cols()` — defines the meteo-only feature set for the Ridge:
  CLIMATE_FEATURE_COLS ∪ windowed_feature_cols() ∪ CALENDAR_FEATURE_COLS.

Score climatology and region-history features are excluded from the Ridge
input — those encode the training-year severity distribution and are NOT
identical at test time semantically.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from config import MODELS_DIR, PROXY_SAMPLES_PER_REGION
from features_climate import CLIMATE_FEATURE_COLS
from features_windowed import windowed_feature_cols
from features_score import CALENDAR_FEATURE_COLS

PROXY_RIDGE_PATH = MODELS_DIR / "proxy_ridge.pkl"
PROXY_FEATURE_COL = "proxy_score"

# Wide alpha grid; RidgeCV picks the best via internal LOOCV by default but
# we use the sklearn `cv` param to make it time-honest where possible.
_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


def meteo_feature_cols() -> list[str]:
    """Feature set fed to the Ridge proxy. Pure meteo + calendar — no score
    history, no region history. Identical semantics at train and test."""
    cols = (
        list(CLIMATE_FEATURE_COLS)
        + list(windowed_feature_cols())
        + list(CALENDAR_FEATURE_COLS)
    )
    # Deduplicate while preserving order
    seen = set()
    return [c for c in cols if not (c in seen or seen.add(c))]


def _sample_recent_per_region(
    train_features: pd.DataFrame,
    samples_per_region: int,
) -> pd.DataFrame:
    """Phase 11: cap proxy ridge to the N most-recent valid score rows per
    region. Reduces training-set size from ~1.5M → ~90k and biases the fit
    toward recent climate, which matches test conditions better than fitting
    on the entire training history (which mixes decades of climate regimes).
    """
    if samples_per_region <= 0 or "week_idx" not in train_features.columns:
        return train_features
    # Within each region, take the N rows with the largest week_idx.
    return (
        train_features.sort_values(["region_id", "week_idx"])
        .groupby("region_id", sort=False, group_keys=False)
        .tail(samples_per_region)
        .reset_index(drop=True)
    )


def fit_proxy_ridge(
    train_features: pd.DataFrame,
    target_col: str = "target_w1",
    feature_cols: list[str] | None = None,
    output_path: Path = PROXY_RIDGE_PATH,
    samples_per_region: int | None = None,
) -> tuple[RidgeCV, list[str], float]:
    """Fit a Ridge regressor on meteo features → drought score.

    `target_col` is the anchor week's score reference. We use `target_w1`
    (1-week-ahead truth) rather than the anchor's own score because the
    anchor row's `score` column is the lookup-source for `score_lag1` (and
    the model never sees its own row's score in inference). target_w1 is the
    closest target the proxy can learn against without leakage.

    `samples_per_region` (Phase 11): if set, only fit on the N most-recent
    valid score rows per region. Defaults to `PROXY_SAMPLES_PER_REGION` from
    config (0 disables the cap, preserving Phase 10 behavior).

    Returns (model, used_feature_cols, train_spearman).
    """
    if feature_cols is None:
        feature_cols = meteo_feature_cols()
    if samples_per_region is None:
        samples_per_region = PROXY_SAMPLES_PER_REGION

    fit_df = _sample_recent_per_region(train_features, samples_per_region)

    # Subset to columns actually present in train_features (in case some
    # feature block was disabled).
    cols = [c for c in feature_cols if c in fit_df.columns]
    if not cols:
        raise RuntimeError("No meteo features available for the proxy Ridge.")
    X = fit_df[cols].to_numpy(np.float32)
    y = fit_df[target_col].to_numpy(np.float32)

    # Replace NaN / Inf in X with column-mean (impute, don't drop). Dropping
    # was empirically too aggressive — a single buggy feature column would
    # zero out the whole training set. Ridge handles imputed-mean rows fine.
    bad = ~np.isfinite(X)
    if bad.any():
        col_means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        X = np.where(bad, col_means[None, :], X).astype(np.float32)
    # Drop rows where y is NaN (rare; targets are integer scores 0..5).
    mask_y = np.isfinite(y)
    if not mask_y.all():
        X, y = X[mask_y], y[mask_y]

    # RidgeCV with cv=5 (k-fold without group-awareness). With the per-region
    # sampling cap the fit set is ~90k rows so this is fast.
    model = RidgeCV(alphas=_ALPHA_GRID, cv=5)
    model.fit(X, y)

    # Spearman ρ on the training fold — sanity check (no holdout here).
    from scipy.stats import spearmanr
    sample_n = min(200_000, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=sample_n, replace=False)
    rho, _ = spearmanr(model.predict(X[idx]), y[idx])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": cols, "alpha": float(model.alpha_)}, f)

    return model, cols, float(rho if np.isfinite(rho) else 0.0)


def load_proxy_ridge(path: Path = PROXY_RIDGE_PATH) -> tuple[RidgeCV, list[str]]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["model"], d["feature_cols"]


def add_proxy_score(
    features_df: pd.DataFrame,
    model: RidgeCV,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Add the proxy_score column. Missing input columns are filled with 0
    (defensive; should never trigger if the upstream feature build is correct)."""
    out = features_df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = np.float32(0.0)
    X = out[feature_cols].to_numpy(np.float32)
    # Impute NaN/Inf with 0 (column-mean info isn't available here; 0 is the
    # mean of all z-scored features by construction, so this is reasonable).
    if not np.isfinite(X).all():
        X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    pred = model.predict(X).astype(np.float32)
    # Clip to plausible drought-score range. Outside [0, 5] would only happen
    # for extrapolation edges and is meaningless as a "score" feature.
    pred = np.clip(pred, 0.0, 5.0)
    out[PROXY_FEATURE_COL] = pred
    return out
