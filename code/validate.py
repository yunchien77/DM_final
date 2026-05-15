"""
Validation utilities: temporal holdout evaluation and walk-forward CV.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from config import HORIZONS, TRAIN_WEEKS_PER_REGION, VALID_WEEKS


def compute_macro_mae(
    y_true_dict: dict[int, np.ndarray],
    y_pred_dict: dict[int, np.ndarray],
) -> tuple[float, dict[int, float]]:
    """Compute per-horizon MAE and their unweighted mean."""
    horizon_maes = {
        h: mean_absolute_error(y_true_dict[h], y_pred_dict[h])
        for h in HORIZONS
    }
    macro_mae = float(np.mean(list(horizon_maes.values())))
    return macro_mae, horizon_maes


def fixed_holdout_split(
    features_df: pd.DataFrame,
    n_val_weeks: int = VALID_WEEKS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: last n_val_weeks per region go to validation.

    Anchor week_idx runs 0..781. The last valid anchor is week_idx = 776
    (so target_w5 = week 781 exists). Validation anchors: the last n_val_weeks
    of these valid anchors.
    """
    max_anchor = TRAIN_WEEKS_PER_REGION - 6  # 776
    val_threshold = max_anchor - n_val_weeks  # 750 for 26 weeks

    train_mask = features_df["week_idx"] <= val_threshold
    val_mask = features_df["week_idx"] > val_threshold

    train_split = features_df[train_mask].reset_index(drop=True)
    val_split = features_df[val_mask].reset_index(drop=True)

    print(f"Train split: {len(train_split):,} rows "
          f"(week_idx ≤ {val_threshold})")
    print(f"Val   split: {len(val_split):,} rows "
          f"(week_idx > {val_threshold})")
    return train_split, val_split


def evaluate_predictions(
    val_df: pd.DataFrame,
    pred_dict: dict[int, np.ndarray],
) -> tuple[float, dict[int, float]]:
    """Build y_true_dict from val_df and compute macro MAE."""
    y_true_dict = {h: val_df[f"target_w{h}"].values for h in HORIZONS}
    return compute_macro_mae(y_true_dict, pred_dict)


def walk_forward_cv(
    features_df: pd.DataFrame,
    fold_boundaries: list[int],
    train_fn,
    feature_cols: list[str],
) -> list[dict]:
    """
    Walk-forward cross-validation.

    fold_boundaries: list of week_idx values that mark the end of each training fold.
    For each boundary b:
      - Train: week_idx <= b
      - Val:   week_idx in (b, b+5] (5 anchor weeks, each predicting up to b+10)

    train_fn(X_train, y_train_dict, X_val, y_val_dict) -> pred_dict
    """
    results = []
    for b in fold_boundaries:
        train_mask = features_df["week_idx"] <= b
        val_mask = (features_df["week_idx"] > b) & (features_df["week_idx"] <= b + 5)

        fold_train = features_df[train_mask]
        fold_val = features_df[val_mask]

        if len(fold_val) == 0:
            print(f"Fold boundary {b}: no validation rows, skipping")
            continue

        X_train = fold_train[feature_cols]
        X_val = fold_val[feature_cols]
        y_train_dict = {h: fold_train[f"target_w{h}"].values for h in HORIZONS}
        y_val_dict = {h: fold_val[f"target_w{h}"].values for h in HORIZONS}

        pred_dict = train_fn(X_train, y_train_dict, X_val, y_val_dict)
        macro_mae, horizon_maes = compute_macro_mae(y_val_dict, pred_dict)

        print(f"Fold b={b}: macro_MAE={macro_mae:.4f} | "
              + " | ".join(f"w{h}={v:.4f}" for h, v in horizon_maes.items()))
        results.append({
            "boundary": b,
            "macro_mae": macro_mae,
            "horizon_maes": horizon_maes,
        })

    avg_macro = np.mean([r["macro_mae"] for r in results])
    print(f"\nWalk-forward CV avg macro MAE: {avg_macro:.4f}")
    return results


def walk_forward_diagnostic(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    fold_boundaries: list[int],
    train_one_fn,
    sample_weight_fn=None,
    n_estimators_override: int | None = None,
) -> dict:
    """
    Run multi-fold walk-forward diagnostic. For each boundary b:
      - Train on week_idx ≤ b
      - Validate on week_idx ∈ (b, b+5]

    `train_one_fn(X_tr, y_tr, X_val, y_val, h, sample_weight)` mirrors the
    signature used in train.py main(). `sample_weight_fn(y_tr_series)` returns
    a per-row weight vector (or None). `n_estimators_override` lets the caller
    cut tree count to speed up the diagnostic relative to the production run.

    Returns dict with `per_fold`, `avg_macro_mae`, and `last_fold_macro_mae`.
    The last-fold metric is the most relevant proxy for the Kaggle test window.
    """
    from model import LGBM_PARAMS  # avoid circular import at module load
    if n_estimators_override:
        _saved = LGBM_PARAMS.get("n_estimators")
        LGBM_PARAMS["n_estimators"] = n_estimators_override

    per_fold = []
    try:
        for b in fold_boundaries:
            train_mask = features_df["week_idx"] <= b
            val_mask = (features_df["week_idx"] > b) & (features_df["week_idx"] <= b + 5)
            fold_train = features_df[train_mask]
            fold_val = features_df[val_mask]
            if len(fold_val) == 0:
                print(f"  [walk-forward] boundary {b}: empty val, skipping")
                continue

            X_tr = fold_train[feature_cols]
            X_val = fold_val[feature_cols]
            preds = {}
            for h in HORIZONS:
                y_tr = fold_train[f"target_w{h}"]
                y_val = fold_val[f"target_w{h}"]
                w = sample_weight_fn(y_tr) if sample_weight_fn else None
                model = train_one_fn(X_tr, y_tr, X_val, y_val, h, w)
                preds[h] = np.clip(model.predict(X_val), 0.0, 5.0)

            y_val_dict = {h: fold_val[f"target_w{h}"].values for h in HORIZONS}
            macro_mae, horizon_maes = compute_macro_mae(y_val_dict, preds)
            print(f"  [walk-forward] b={b}: macro_MAE={macro_mae:.4f}")
            per_fold.append({"boundary": b, "macro_mae": macro_mae, "horizon_maes": horizon_maes})
    finally:
        if n_estimators_override:
            LGBM_PARAMS["n_estimators"] = _saved

    if not per_fold:
        return {"per_fold": [], "avg_macro_mae": None, "last_fold_macro_mae": None}
    avg = float(np.mean([f["macro_mae"] for f in per_fold]))
    last = per_fold[-1]["macro_mae"]
    return {"per_fold": per_fold, "avg_macro_mae": avg, "last_fold_macro_mae": last}


def print_validation_report(macro_mae: float, horizon_maes: dict[int, float]):
    print("\n" + "=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)
    for h in HORIZONS:
        print(f"  Week {h} MAE: {horizon_maes[h]:.4f}")
    print(f"  Macro MAE:  {macro_mae:.4f}")
    print("=" * 50)
    if macro_mae > 1.0:
        print("WARNING: macro MAE > 1.0 — check for lag alignment bugs or leakage")
