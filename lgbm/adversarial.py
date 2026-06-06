"""
Adversarial-validation importance weighting (Phase 3b).

Trains a binary LightGBM classifier to distinguish in-season rows (woy ∈
test_woy ± slack) from off-season rows on meteo lag-0 stats only — no
score-lag, no temporal, no anomaly. Used as a density-ratio estimator: a
training row's predicted P(in-season | x) translates to a sample weight

    w(x) = P(in-season|x) / (1 - P(in-season|x))

then clipped to [LO, HI] and renormalized to mean=1. The resulting weight
upweights training rows that "look like" the in-season distribution and
downweights rows that don't — bridging the residual cross-year covariate
shift not captured by the calendar-matched filter alone.

ESS tracking: if effective sample size (Σw)² / Σw² falls below 30% of N,
the weighting is too aggressive (a small subset of test-like rows would
dominate the gradient). Caller should tighten the clip or revert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


def _circular_woy_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return np.minimum(d, 52 - d)


def make_av_labels(
    features_df: pd.DataFrame,
    test_woy_per_region: dict[str, int],
    slack_weeks: int,
) -> np.ndarray:
    """label=1 if row's woy is within ±slack of its region's test_woy."""
    woys = ((features_df["day_of_year"].astype(np.int32) - 1) // 7).clip(0, 52).astype(np.int16).values
    test_woys = features_df["region_id"].map(test_woy_per_region).astype(np.int16).values
    dist = _circular_woy_distance(woys, test_woys)
    return (dist <= slack_weeks).astype(np.int8)


def fit_av_classifier(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    lag0_cols: list[str],
    max_train: int = 300_000,
    seed: int = 42,
    log_fn=print,
) -> tuple[lgb.LGBMClassifier, float]:
    """Fit a binary LGBM classifier. Returns (model, val_auc).

    Subsamples to `max_train` rows for fitting speed (200-300k is plenty for
    measuring AUC reliably). A 20% holdout is reserved for AUC evaluation.
    """
    n = len(features_df)
    rng = np.random.default_rng(seed)
    if n > max_train:
        idx = rng.choice(n, max_train, replace=False)
    else:
        idx = np.arange(n)
    X_sub = features_df.iloc[idx][lag0_cols].values.astype(np.float32)
    y_sub = labels[idx]

    # 80/20 internal split for AUC measurement
    perm = rng.permutation(len(X_sub))
    split = int(0.8 * len(X_sub))
    tr_idx, vl_idx = perm[:split], perm[split:]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        X_sub[tr_idx], y_sub[tr_idx],
        eval_set=[(X_sub[vl_idx], y_sub[vl_idx])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    val_probs = model.predict_proba(X_sub[vl_idx])[:, 1]
    auc = roc_auc_score(y_sub[vl_idx], val_probs)
    log_fn(f"  [av] fit: positives={int(y_sub.sum()):,}/{len(y_sub):,} "
           f"({y_sub.mean()*100:.1f}%)  AUC={auc:.4f}  best_iter={model.best_iteration_}")
    return model, float(auc)


def compute_av_weights(
    av_model: lgb.LGBMClassifier,
    features_df: pd.DataFrame,
    lag0_cols: list[str],
    clip_lo: float,
    clip_hi: float,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute density-ratio weights per row.

    Returns (weights, probabilities). Weights are clipped to [clip_lo, clip_hi]
    and renormalized so mean(weights) == 1.0.
    """
    X = features_df[lag0_cols].values.astype(np.float32)
    p = av_model.predict_proba(X)[:, 1]
    p = np.clip(p, eps, 1 - eps)
    w = p / (1.0 - p)
    w = np.clip(w, clip_lo, clip_hi)
    w = w / w.mean()
    return w.astype(np.float32), p.astype(np.float32)


def effective_sample_size(w: np.ndarray) -> tuple[float, float]:
    """ESS = (Σw)² / Σ(w²). Returns (ESS, ESS/N)."""
    w = np.asarray(w, dtype=np.float64)
    ess = (w.sum() ** 2) / (w ** 2).sum()
    return float(ess), float(ess / len(w))


def select_lag0_meteo_cols(feature_cols: list[str]) -> list[str]:
    """Pick the meteo lag-0 stat columns from the model's feature list.

    Excludes score-lag (don't end in _lag0), anomaly/cluster/region features,
    and any non-lag0 columns. This is the AV classifier's input feature set.
    """
    out = []
    for c in feature_cols:
        if not c.endswith("_lag0"):
            continue
        if c.startswith(("anom_", "region_", "cluster_", "score_lag", "month_", "doy_", "int_", "slope")):
            continue
        # Keep e.g. prec_mean_lag0, tmp_max_p95_lag0, surf_pre_std_lag0.
        out.append(c)
    return out
