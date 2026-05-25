"""Phase 11 — XGBoost inference. Builds test features (same path as predict.py),
loads xgb_h{1..5}.json, predicts 5 horizons, applies the same transition
smoothing (β=TRANSITION_SMOOTHING_BETA) as the LGBM stack, and writes a
wide-format submission CSV.

Run:
    python xgb_predict.py
    python xgb_predict.py --output /path/to/submission.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from logging_setup import get_logger
log = get_logger("xgb_predict", label="xgb_predict")

from config import (
    MODELS_DIR, HORIZONS,
    USE_TRANSITION_SMOOTHING, TRANSITION_SMOOTHING_BETA,
)
from predict import build_test_features
from transition_matrix import load_transition_matrices, apply_transition_smoothing


def main(output_path: str | Path):
    output_path = Path(output_path)
    t0 = time.time()
    log.info("=" * 70)
    log.info("Phase 11 — XGBoost inference")
    log.info("=" * 70)

    test_features, _ = build_test_features()

    # Match the column order used at training time. xgb_model.py wrote
    # models/feature_cols.json (LGBM did) so we can reuse that.
    import json
    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    for c in feature_cols:
        if c not in test_features.columns:
            log.warning(f"  missing feature in test: {c} → filling with 0")
            test_features[c] = np.float32(0.0)
    X = test_features[feature_cols]
    log.info(f"  test feature matrix: {X.shape}")

    # Predict per-horizon
    pred_matrix = np.zeros((len(X), len(HORIZONS)), dtype=np.float32)
    for i, h in enumerate(HORIZONS):
        path = MODELS_DIR / f"xgb_h{h}.json"
        booster = xgb.XGBRegressor()
        booster.load_model(str(path))
        pred_matrix[:, i] = np.clip(booster.predict(X), 0.0, 5.0)
        log.info(f"  H{h} loaded {path.name}")

    # Transition smoothing (same recipe as predict.py)
    if USE_TRANSITION_SMOOTHING:
        matrices, cluster_assignment, horizons, score_levels = load_transition_matrices()
        if cluster_assignment is not None and len(cluster_assignment) > 0:
            cluster_ids = (
                test_features["region_id"].astype(str).map(cluster_assignment)
                .fillna(0).astype(int).to_numpy()
            )
        else:
            cluster_ids = np.zeros(len(test_features), dtype=np.int32)
        score_lag1 = test_features.get("score_lag1", pd.Series(np.zeros(len(test_features)))).to_numpy()
        smoothed = apply_transition_smoothing(
            pred_matrix, score_lag1, cluster_ids, matrices, horizons, score_levels,
            beta=TRANSITION_SMOOTHING_BETA,
        )
        log.info(f"  transition smoothing (β={TRANSITION_SMOOTHING_BETA})  "
                 f"mean shift: {float(np.mean(smoothed - pred_matrix)):+.4f}")
        pred_matrix = smoothed

    # Wide-format submission, numeric sort
    log.info(f"[write] {output_path}")
    sub_df = pd.DataFrame({
        "region_id": test_features["region_id"].astype(str).to_numpy(),
        **{f"pred_week{h}": pred_matrix[:, i].round(4) for i, h in enumerate(HORIZONS)},
    })
    sub_df["__order"] = sub_df["region_id"].str.replace("R", "", regex=False).astype(int)
    sub_df = sub_df.sort_values("__order").drop(columns=["__order"]).reset_index(drop=True)
    sub_df.to_csv(output_path, index=False)
    vals = pred_matrix.ravel()
    log.info(f"  rows: {len(sub_df):,}  mean={vals.mean():.4f}  std={vals.std():.4f}")
    log.info(f"[done] {(time.time() - t0) / 60:.2f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="/mnt/1stHDD/juiyun/DMFP/submission_phase11_xgb.csv")
    args = parser.parse_args()
    main(output_path=args.output)
