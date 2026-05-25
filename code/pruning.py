"""Phase 7 — aggressive collinearity pruning.

Three-stage filter (replaces the legacy feature_pruning.py + feature_audit.py):

  1. Pairwise collinearity clustering: hierarchical clustering on
     `1 − max(|ρ_pearson|, |ρ_spearman|)`. Cut at distance 0.15
     → features with mutual collinearity ≥ 0.85 collapse into one cluster.

  2. Within-cluster selection: keep the feature with the LOWEST
     adversarial-validation importance (most year-stable). Ties broken by
     highest LightGBM gain on a quick single-horizon model.

  3. Post-cluster VIF check: drop any feature with VIF > 10 iteratively
     (catches multi-feature collinearity that pairwise pruning misses).

The hard-keep list is enforced regardless of cluster membership.

Public:
  prune_features(train_features, target, hard_keep, output_path)
      → list of kept column names, also written to `output_path` (json).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

import lightgbm as lgb

from config import MODELS_DIR

LEAN_FEATURE_LIST_PATH_DEFAULT = MODELS_DIR / "feature_cols_lean.json"
PRUNING_REPORT_PATH = MODELS_DIR.parent / "diagnostics" / "pruning_report.md"

# ---------------------------------------------------------------------------
# AV importance (train-vs-test) — supplied as a function so callers can use
# either a real test feature matrix or a held-out fold from train.
# ---------------------------------------------------------------------------

def fit_av_importance(
    train_features: np.ndarray,   # (N_train, F) float32
    test_features: np.ndarray,    # (N_test, F)  float32
    feature_names: list[str],
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_subsample: int = 200_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, float]:
    """Train a binary classifier (train=0 vs test=1) and return per-feature
    gain importance. Returns (importance_df, val_auc)."""
    rng = np.random.default_rng(seed)
    n_train = train_features.shape[0]
    n_test = test_features.shape[0]
    if n_train > max_subsample:
        idx = rng.choice(n_train, size=max_subsample, replace=False)
        train_features = train_features[idx]
        n_train = max_subsample
    X = np.vstack([train_features, test_features]).astype(np.float32)
    y = np.concatenate([np.zeros(n_train, dtype=np.float32),
                        np.ones(n_test, dtype=np.float32)])
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]
    split = int(0.8 * len(y))
    Xtr, ytr = X[:split], y[:split]
    Xva, yva = X[split:], y[split:]

    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=feature_names, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr, free_raw_data=False)
    params = dict(
        objective="binary", metric="auc",
        learning_rate=learning_rate, num_leaves=31,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
        verbose=-1, n_jobs=-1,
    )
    booster = lgb.train(
        params, dtr, num_boost_round=n_estimators, valid_sets=[dva],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )
    val_auc = float(booster.best_score["val"]["auc"])
    imp = pd.DataFrame({
        "feature": feature_names,
        "av_gain": booster.feature_importance(importance_type="gain"),
    })
    return imp, val_auc


# ---------------------------------------------------------------------------
# Quick per-feature LightGBM gain on the target (for within-cluster tie-break)
# ---------------------------------------------------------------------------

def fit_lgbm_gain(
    train_features: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 200,
    max_subsample: int = 300_000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = train_features.shape[0]
    if n > max_subsample:
        idx = rng.choice(n, size=max_subsample, replace=False)
        train_features = train_features[idx]
        target = target[idx]
    dtrain = lgb.Dataset(train_features, label=target, feature_name=feature_names, free_raw_data=False)
    params = dict(
        objective="regression_l1", metric="mae",
        learning_rate=0.05, num_leaves=31,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
        verbose=-1, n_jobs=-1,
    )
    booster = lgb.train(params, dtrain, num_boost_round=n_estimators, callbacks=[lgb.log_evaluation(0)])
    return pd.DataFrame({
        "feature": feature_names,
        "lgbm_gain": booster.feature_importance(importance_type="gain"),
    })


# ---------------------------------------------------------------------------
# Stage 1: pairwise Pearson + Spearman clustering
# ---------------------------------------------------------------------------

def pairwise_collinearity_clusters(
    features: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.85,
    spearman_sample: int = 50_000,
    seed: int = 42,
) -> list[list[str]]:
    """Hierarchical clustering on max(|ρ_p|, |ρ_s|) ≥ threshold.
    Returns list of clusters (each cluster = list of feature names)."""
    F = len(feature_names)
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    sample_idx = rng.choice(n, size=min(spearman_sample, n), replace=False)
    sample = features[sample_idx].astype(np.float32)

    pearson = np.corrcoef(sample.T)
    pearson = np.nan_to_num(pearson, nan=0.0)
    # Spearman on the same sample (rank then Pearson)
    rank_sample = np.argsort(np.argsort(sample, axis=0), axis=0).astype(np.float32)
    spearman = np.corrcoef(rank_sample.T)
    spearman = np.nan_to_num(spearman, nan=0.0)

    combined = np.maximum(np.abs(pearson), np.abs(spearman))
    np.fill_diagonal(combined, 1.0)
    dist = 1.0 - combined
    dist = np.clip(dist, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=1.0 - threshold, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for f, c in zip(feature_names, cluster_ids):
        clusters.setdefault(int(c), []).append(f)
    return list(clusters.values())


# ---------------------------------------------------------------------------
# Stage 2: within-cluster representative selection (AV-importance first)
# ---------------------------------------------------------------------------

def select_within_clusters(
    clusters: list[list[str]],
    av_importance: pd.DataFrame,
    lgbm_gain: pd.DataFrame,
    hard_keep: set[str],
) -> list[str]:
    av_lookup = dict(zip(av_importance["feature"], av_importance["av_gain"]))
    gain_lookup = dict(zip(lgbm_gain["feature"], lgbm_gain["lgbm_gain"]))
    kept: list[str] = []
    for cluster in clusters:
        hk = [f for f in cluster if f in hard_keep]
        if hk:
            kept.extend(hk)
            # Also keep one non-hard-keep representative if cluster has > 1 hard-keep + others.
            # In practice hard_keep names usually fall in singletons or small clusters; we
            # don't over-prune away their cluster siblings.
        else:
            # Lower AV gain = more year-stable. Tie-break: higher LGBM gain.
            ranked = sorted(
                cluster,
                key=lambda f: (av_lookup.get(f, 0.0), -gain_lookup.get(f, 0.0)),
            )
            kept.append(ranked[0])
    return kept


# ---------------------------------------------------------------------------
# Stage 3: VIF check
# ---------------------------------------------------------------------------

def vif_filter(
    features: np.ndarray,
    feature_names: list[str],
    hard_keep: set[str],
    vif_threshold: float = 10.0,
    max_iter: int = 30,
    sample_n: int = 50_000,
    seed: int = 42,
) -> list[str]:
    """Iteratively drop the feature with the highest VIF until all ≤ vif_threshold."""
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    sample_idx = rng.choice(n, size=min(sample_n, n), replace=False)
    X = features[sample_idx].astype(np.float64)
    names = list(feature_names)
    cols_idx = {name: i for i, name in enumerate(names)}

    for _ in range(max_iter):
        F = len(names)
        if F <= 1:
            break
        # Quick VIF via R² of OLS for each column
        vifs = np.zeros(F, dtype=np.float64)
        idx_map = np.array([cols_idx[n] for n in names])
        mat = X[:, idx_map]
        # mean-center
        mat = mat - mat.mean(axis=0, keepdims=True)
        var = (mat ** 2).sum(axis=0)
        var[var <= 1e-9] = 1e-9
        for i in range(F):
            other = np.delete(mat, i, axis=1)
            y = mat[:, i]
            # OLS via least squares
            try:
                coef, *_ = np.linalg.lstsq(other, y, rcond=None)
                resid = y - other @ coef
                ss_res = (resid ** 2).sum()
                ss_tot = var[i]
                r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
                vifs[i] = 1.0 / max(1.0 - r2, 1e-9)
            except Exception:
                vifs[i] = 1.0
        # Drop the highest-VIF non-hard-keep feature if it exceeds threshold
        order = np.argsort(-vifs)
        dropped = False
        for k in order:
            if names[k] in hard_keep:
                continue
            if vifs[k] > vif_threshold:
                names.pop(k)
                dropped = True
                break
        if not dropped:
            break
    return names


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def prune_features(
    train_features: np.ndarray,
    test_features: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    hard_keep: list[str],
    output_path: Path = LEAN_FEATURE_LIST_PATH_DEFAULT,
    correlation_threshold: float = 0.85,
    vif_threshold: float = 10.0,
) -> tuple[list[str], dict]:
    """End-to-end pruning. Returns (kept_features, report_summary)."""
    hard_keep_set = {f for f in hard_keep if f in feature_names}

    # Stage 1
    clusters = pairwise_collinearity_clusters(
        train_features, feature_names, threshold=correlation_threshold,
    )

    # Stage 2 — need AV importance + LGBM gain
    av_imp, av_auc = fit_av_importance(train_features, test_features, feature_names)
    lgbm_gain = fit_lgbm_gain(train_features, target, feature_names)

    kept_after_clustering = select_within_clusters(clusters, av_imp, lgbm_gain, hard_keep_set)

    # Stage 3 — VIF on survivors
    kept_idx = [feature_names.index(f) for f in kept_after_clustering]
    survivors_arr = train_features[:, kept_idx]
    kept_final = vif_filter(
        survivors_arr, kept_after_clustering, hard_keep_set, vif_threshold=vif_threshold,
    )

    # Persist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(kept_final, f, indent=2)

    report = {
        "input_features": len(feature_names),
        "collinearity_clusters": len(clusters),
        "after_cluster_selection": len(kept_after_clustering),
        "after_vif": len(kept_final),
        "av_classifier_auc": av_auc,
        "correlation_threshold": correlation_threshold,
        "vif_threshold": vif_threshold,
        "hard_keep_present": sorted(hard_keep_set),
    }
    return kept_final, report
