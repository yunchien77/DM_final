"""Phase 7 inference entry point.

Mirrors train.py's feature stack, applies optional drought-transition
smoothing, and writes submission.csv.

Run:
    python predict.py                         # default SUBMISSION_PATH
    python predict.py --output sub_v2.csv     # custom output
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from logging_setup import get_logger
log = get_logger("predict", label="predict")

from config import (
    TEST_PATH, SUBMISSION_PATH, MODELS_DIR, HORIZONS, LAG_WINDOW,
    USE_PREPROCESSING, USE_SCORE_LAG, USE_CLIMATE_FEATURES, USE_WINDOWED_FEATURES,
    USE_PROXY_SCORE, USE_METEO_CLUSTER, USE_ZERO_INFLATED,
    USE_TRANSITION_SMOOTHING, TRANSITION_SMOOTHING_BETA,
    USE_WALK_FORWARD_CV,
    WINDOWED_CHANNELS, WINDOWED_WINDOWS,
)
from data_pipeline import load_and_aggregate_daily_to_weekly, construct_anchor_rows
from features_score import (
    add_calendar_features, add_region_features, add_score_climatology,
    add_score_lag1_test, REGION_STATS_PATH, SCORE_CLIMATOLOGY_PATH, REGION_LAST_SCORE_PATH,
)
from features_climate import (
    add_climate_features,
    PRESSURE_CLIMATOLOGY_PATH, METEO_CLIMATOLOGY_PATH,
    DROUGHT_INDEX_CLIMATOLOGY_PATH, HEAT_CLIMATOLOGY_PATH,
)
from features_windowed import add_windowed_features, WINDOWED_CLIMATOLOGY_PATH
from transition_matrix import (
    load_transition_matrices, apply_transition_smoothing,
)
from proxy_score import load_proxy_ridge, add_proxy_score, PROXY_FEATURE_COL
from model import (
    load_all_models, predict_all_horizons,
    load_fold_models, predict_fold_ensemble,
    load_calibrator, apply_calibration,
)
from zero_inflated import load_zi_models, predict_zi


def build_test_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the test anchor matrix and return (features_df, test_weekly)."""
    preproc_artifacts = None
    if USE_PREPROCESSING:
        from preprocessing import load_preprocessing_artifacts
        preproc_artifacts = load_preprocessing_artifacts()
        log.info(f"  preprocessing artifacts loaded "
                 f"(winsor={len(preproc_artifacts[0])} log/sqrt={len(preproc_artifacts[1])} "
                 f"rank={len(preproc_artifacts[3])})")

    log.info("[stage 1] test daily → weekly")
    test_weekly = load_and_aggregate_daily_to_weekly(
        TEST_PATH, is_train=False, preproc_artifacts=preproc_artifacts,
    )

    log.info("[stage 2] anchor rows")
    test_features = construct_anchor_rows(test_weekly, is_train=False, lag_window=LAG_WINDOW)
    test_features = add_calendar_features(test_features)

    log.info("[stage 3] loading climatologies + score features")
    region_stats = pd.read_csv(REGION_STATS_PATH)
    score_clim = pd.read_csv(SCORE_CLIMATOLOGY_PATH)
    test_features = add_region_features(test_features, region_stats)
    test_features = add_score_climatology(test_features, score_clim)

    # score_lag1 is always populated on test rows (transition smoothing uses
    # it as the "previous score" reference). It's only added to feature_cols
    # when USE_SCORE_LAG=True.
    if REGION_LAST_SCORE_PATH.exists():
        region_last = pd.read_csv(REGION_LAST_SCORE_PATH)
        test_features = add_score_lag1_test(test_features, region_last)

    if USE_CLIMATE_FEATURES:
        climate_clims = {
            "pressure": pd.read_csv(PRESSURE_CLIMATOLOGY_PATH),
            "meteo": pd.read_csv(METEO_CLIMATOLOGY_PATH),
            "drought_index": pd.read_csv(DROUGHT_INDEX_CLIMATOLOGY_PATH),
        }
        if HEAT_CLIMATOLOGY_PATH.exists():
            climate_clims["heat"] = pd.read_csv(HEAT_CLIMATOLOGY_PATH)
        else:
            climate_clims["heat"] = None
        # The lag look-back for climate features (0..7) fits inside the
        # 13-week test window, so we use test_weekly for the lag source.
        test_features, _ = add_climate_features(test_features, test_weekly, climate_clims)

    if USE_WINDOWED_FEATURES:
        windowed_clim = pd.read_csv(WINDOWED_CLIMATOLOGY_PATH)
        test_features, _ = add_windowed_features(
            test_features, test_weekly, windowed_clim,
            channels=WINDOWED_CHANNELS, windows=WINDOWED_WINDOWS,
        )

    # Phase 8 proxy score (fitted at training time).
    if USE_PROXY_SCORE:
        from proxy_score import PROXY_RIDGE_PATH
        if PROXY_RIDGE_PATH.exists():
            proxy_model, proxy_used_cols = load_proxy_ridge()
            test_features = add_proxy_score(test_features, proxy_model, proxy_used_cols)
            log.info(f"  proxy_score applied (alpha={proxy_model.alpha_:.4g}, "
                     f"features={len(proxy_used_cols)})")
        else:
            log.warning("USE_PROXY_SCORE=True but no proxy_ridge.pkl found — skipping")

    # Phase 8 optional meteo cluster.
    if USE_METEO_CLUSTER:
        cluster_path = MODELS_DIR / "meteo_cluster_table.csv"
        if cluster_path.exists():
            cluster_table = pd.read_csv(cluster_path)[["region_id", "meteo_cluster_id"]]
            test_features = test_features.merge(cluster_table, on="region_id", how="left")
            test_features["meteo_cluster_id"] = (
                test_features["meteo_cluster_id"].fillna(-1).astype(np.int32)
            )
            log.info(f"  meteo_cluster_id applied (from {cluster_path})")
        else:
            log.info(f"  USE_METEO_CLUSTER=True but {cluster_path} not found — skipping")

    return test_features, test_weekly


def main(output_path: str | Path | None = None):
    if output_path is None:
        output_path = SUBMISSION_PATH
    output_path = Path(output_path)

    t0 = time.time()
    log.info("=" * 70)
    log.info("Phase 8 inference")
    log.info("=" * 70)

    test_features, _ = build_test_features()

    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    log.info(f"  expected feature columns: {len(feature_cols)}")

    # Align: any missing column is filled with zeros (defensive).
    for c in feature_cols:
        if c not in test_features.columns:
            log.warning(f"  missing feature in test: {c} → filling with 0")
            test_features[c] = np.float32(0.0)

    X_test = test_features[feature_cols]
    log.info(f"  test feature matrix: {X_test.shape}")

    if USE_ZERO_INFLATED:
        log.info("[stage 4] loading zero-inflated models + predicting")
        zi_models = load_zi_models()
        preds = predict_zi(zi_models, X_test)
        pred_matrix = np.stack([np.clip(preds[h], 0.0, 5.0) for h in HORIZONS], axis=1)
    elif USE_WALK_FORWARD_CV:
        log.info("[stage 4] loading walk-forward fold models + ensembling")
        fold_models = load_fold_models()
        preds = predict_fold_ensemble(fold_models, X_test)
        pred_matrix = np.stack([preds[h] for h in HORIZONS], axis=1).astype(np.float32)
        cal = load_calibrator()
        if cal is not None:
            log.info("  applying isotonic calibration")
        pred_matrix = apply_calibration(pred_matrix, cal)
    else:
        log.info("[stage 4] loading models + predicting (legacy single-split)")
        models = load_all_models()
        preds = predict_all_horizons(models, X_test)
        pred_matrix = np.stack([np.clip(preds[h], 0.0, 5.0) for h in HORIZONS], axis=1)

    # Optional drought-transition smoothing
    if USE_TRANSITION_SMOOTHING:
        matrices, cluster_assignment, horizons, score_levels = load_transition_matrices()
        # Build per-row cluster IDs.
        if cluster_assignment is not None and len(cluster_assignment) > 0:
            cluster_ids = test_features["region_id"].astype(str).map(cluster_assignment).fillna(0).astype(int).to_numpy()
        else:
            cluster_ids = np.zeros(len(test_features), dtype=np.int32)
        score_lag1 = test_features.get("score_lag1", pd.Series(np.zeros(len(test_features)))).to_numpy()
        smoothed = apply_transition_smoothing(
            pred_matrix, score_lag1, cluster_ids, matrices, horizons, score_levels,
            beta=TRANSITION_SMOOTHING_BETA,
        )
        log.info(f"  transition smoothing applied (β={TRANSITION_SMOOTHING_BETA})  "
                 f"mean shift: {float(np.mean(smoothed - pred_matrix)):+.4f}")
        pred_matrix = smoothed

    # Write submission in wide format to match the competition layout:
    #   region_id, pred_week1, pred_week2, pred_week3, pred_week4, pred_week5
    # Rows sorted by the numeric portion of region_id (R1, R2, ..., R2248) to
    # match the sample-submission ordering.
    log.info(f"[stage 5] writing submission → {output_path}")
    sub_df = pd.DataFrame({
        "region_id": test_features["region_id"].astype(str).to_numpy(),
        **{f"pred_week{h}": pred_matrix[:, i].round(4)
           for i, h in enumerate(HORIZONS)},
    })
    sub_df["__order"] = (
        sub_df["region_id"].str.replace("R", "", regex=False).astype(int)
    )
    sub_df = sub_df.sort_values("__order").drop(columns=["__order"]).reset_index(drop=True)
    sub_df.to_csv(output_path, index=False)
    all_preds = pred_matrix.ravel()
    log.info(
        f"  rows: {len(sub_df):,}  cols: {list(sub_df.columns)}  "
        f"preds mean={float(all_preds.mean()):.4f}  std={float(all_preds.std()):.4f}"
    )
    log.info(f"[done] total time = {(time.time() - t0) / 60:.2f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()
    main(output_path=args.output)
