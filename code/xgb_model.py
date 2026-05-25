"""Phase 11 — XGBoost model family (ensemble diversity member).

Trains 5 XGBoost regressors (one per horizon) on the same Phase 8 cached
feature matrix and sample weights. Uses Tweedie loss with variance_power=1.5
to keep the objective parallel to the LightGBM stack. Saves to
models/xgb_h{1..5}.json so the blend script can load them later.

This is a standalone runner — invoked as `python xgb_model.py`. It:
  1. Loads cached train_features (feature_cache_key())
  2. Recomputes the transition+severity sample weight (matching train.py)
  3. Trains 5 XGBoost boosters with calendar-matched val early stopping
  4. Persists models, writes val and test predictions to .npy

Then `blend.py` reads the val/test prediction arrays, blends with the existing
P8 + P9 submissions, and emits a new submission CSV.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from logging_setup import get_logger
log = get_logger("xgb", label="xgb")

from config import (
    MODELS_DIR, HORIZONS, USE_TRANSITION_WEIGHT, USE_TRANSITION_SMOOTHING,
    TRANSITION_CLUSTERS, TRANSITION_SMOOTHING_ALPHA, TRANSITION_WEIGHT_GAMMA,
    USE_SEVERITY_WEIGHT, SAMPLE_WEIGHT_ALPHA, SAMPLE_WEIGHT_BETA,
)
from cache import feature_cache_key, load_from_cache
from transition_matrix import build_transition_matrices, compute_transition_weight
from validate import evaluate_predictions, print_validation_report

XGB_VAL_PREDS_PATH = MODELS_DIR / "xgb_val_preds.npy"
XGB_TEST_PREDS_PATH = MODELS_DIR / "xgb_test_preds.npy"

XGB_PARAMS = dict(
    objective="reg:tweedie",
    tweedie_variance_power=1.5,
    eval_metric="mae",
    learning_rate=0.02,
    max_depth=6,                # depth-wise; ~comparable to LGBM num_leaves=63
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)
N_ESTIMATORS = 3000
EARLY_STOPPING_ROUNDS = 100


def _build_sample_weight(train_split: pd.DataFrame, weekly_df_full: pd.DataFrame) -> np.ndarray | None:
    """Re-derive the same train.py sample weights so XGBoost trains on
    identical examples-with-weights as the Phase 8/9 LGBMs."""
    sample_weight = None
    if USE_TRANSITION_WEIGHT or USE_TRANSITION_SMOOTHING:
        log.info("  building transition matrices...")
        matrices, cluster_assignment = build_transition_matrices(
            weekly_df_full, n_clusters=TRANSITION_CLUSTERS,
            smoothing_alpha=TRANSITION_SMOOTHING_ALPHA,
        )
        if USE_TRANSITION_WEIGHT and "score_lag1" in train_split.columns:
            cluster_ids = np.zeros(len(train_split), dtype=np.int32)
            if cluster_assignment is not None and len(cluster_assignment) > 0:
                cluster_ids = (
                    train_split["region_id"].astype(str)
                    .map(cluster_assignment).fillna(0).astype(int).to_numpy()
                )
            sample_weight = compute_transition_weight(
                score_lag1=train_split["score_lag1"].to_numpy(),
                score=train_split["target_w1"].to_numpy(),
                cluster_ids=cluster_ids,
                matrices=matrices,
                horizon=1,
                horizons=list(HORIZONS),
                score_levels=list(range(6)),
                gamma=TRANSITION_WEIGHT_GAMMA,
            )
            log.info(f"  transition weight: mean={float(sample_weight.mean()):.3f} max={float(sample_weight.max()):.3f}")

    if USE_SEVERITY_WEIGHT:
        y = train_split["target_w1"].to_numpy(np.float32)
        sev = (
            1.0
            + SAMPLE_WEIGHT_ALPHA * (y > 0).astype(np.float32)
            + SAMPLE_WEIGHT_BETA * (y >= 3).astype(np.float32)
        )
        sample_weight = sev if sample_weight is None else sample_weight * sev
        sample_weight = (sample_weight / sample_weight.mean()).astype(np.float32)
        log.info(f"  + severity (α={SAMPLE_WEIGHT_ALPHA}, β={SAMPLE_WEIGHT_BETA}): "
                 f"final mean={float(sample_weight.mean()):.3f} max={float(sample_weight.max()):.3f}")
    return sample_weight


def train_xgb_horizon(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    horizon: int,
    sample_weight: np.ndarray | None = None,
) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, early_stopping_rounds=EARLY_STOPPING_ROUNDS, **XGB_PARAMS)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iter = model.best_iteration
    val_mae = float(model.best_score)
    log.info(f"  H{horizon}: best_iter={best_iter}  val_mae={val_mae:.4f}")
    return model


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Phase 11 — XGBoost training")
    log.info("=" * 70)

    key = feature_cache_key()
    log.info(f"[cache] key={key}")
    cached = load_from_cache("train_features", key)
    if cached is None:
        raise SystemExit("No cached train_features — run train.py first.")
    train_split = cached["train_split"]
    val_split = cached["val_split"]
    feature_cols = cached["feature_cols"]
    weekly_df_full = cached["weekly_df_full"]

    # Match the same trim train.py applies: drop score_lag1 from feature_cols.
    feature_cols = [c for c in feature_cols if c != "score_lag1"]
    # proxy_score should already be in train_split/val_split if Phase 8/9 ran.
    if "proxy_score" not in train_split.columns:
        from proxy_score import fit_proxy_ridge, add_proxy_score, meteo_feature_cols, PROXY_FEATURE_COL
        log.info("  proxy_score not in cache columns — fitting on the fly")
        proxy_in = [c for c in meteo_feature_cols() if c in train_split.columns]
        proxy_model, proxy_used, _ = fit_proxy_ridge(train_split, target_col="target_w1", feature_cols=proxy_in)
        train_split = add_proxy_score(train_split, proxy_model, proxy_used)
        val_split = add_proxy_score(val_split, proxy_model, proxy_used)
        if PROXY_FEATURE_COL not in feature_cols:
            feature_cols = feature_cols + [PROXY_FEATURE_COL]

    log.info(f"  features: {len(feature_cols)}  train: {len(train_split):,}  val: {len(val_split):,}")

    sample_weight = _build_sample_weight(train_split, weekly_df_full)

    X_train = train_split[feature_cols]
    X_val = val_split[feature_cols]

    log.info(f"[stage train] training {len(HORIZONS)} XGBoost regressors  "
             f"(objective={XGB_PARAMS['objective']}  n_estimators={N_ESTIMATORS})")
    models = {}
    val_preds = np.zeros((len(val_split), len(HORIZONS)), dtype=np.float32)
    for i, h in enumerate(HORIZONS):
        log.info(f"--- horizon {h} ---")
        model = train_xgb_horizon(
            X_train, train_split[f"target_w{h}"],
            X_val, val_split[f"target_w{h}"],
            horizon=h, sample_weight=sample_weight,
        )
        models[h] = model
        val_preds[:, i] = np.clip(model.predict(X_val), 0.0, 5.0)
        model_path = MODELS_DIR / f"xgb_h{h}.json"
        model.save_model(str(model_path))
        log.info(f"  saved {model_path}")

    # Val MAE report
    pred_dict = {h: val_preds[:, i] for i, h in enumerate(HORIZONS)}
    macro_mae, horizon_maes = evaluate_predictions(val_split, pred_dict)
    print_validation_report(macro_mae, horizon_maes)

    # Persist val predictions (for the blend tooling)
    np.save(XGB_VAL_PREDS_PATH, val_preds)
    log.info(f"  saved val predictions → {XGB_VAL_PREDS_PATH}")

    log.info(f"[done] total time = {(time.time() - t0) / 60:.2f} min")


if __name__ == "__main__":
    main()
