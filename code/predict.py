"""
Inference entry point: load saved models → process test data → write submission.csv

Mirrors train.py's feature-engineering steps exactly, then predicts 5 horizons
per region with the saved L1 LightGBM boosters.

Usage:
    python predict.py
"""

import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from logging_setup import get_logger, current_log_path
log = get_logger("predict", label="predict")

from config import (
    TEST_PATH, SUBMISSION_PATH, MODELS_DIR, HORIZONS, N_REGIONS,
    USE_EXTRA_FEATURES, USE_PREPROCESSING,
    USE_SCORE_LAG_FEATURES, USE_REGION_CLUSTER_FEATURES,
)
from data_pipeline import (
    load_and_aggregate_daily_to_weekly,
    construct_lag_features,
    load_feature_columns,
)
from features import add_temporal_features, add_region_features, clip_predictions
from model import load_all_models

if USE_EXTRA_FEATURES:
    from features_extra import (
        add_woy_score_climatology,
        add_anomaly_features,
        add_trend_features,
        add_interaction_features,
    )
if USE_SCORE_LAG_FEATURES:
    from features_extra import add_score_lag_features_test
if USE_REGION_CLUSTER_FEATURES:
    from features_extra import add_region_cluster_features, REGION_CLUSTERS_CSV


def _predict(model, X):
    return np.clip(model.predict(X), 0.0, 5.0)


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("DMFP inference run | model=L1 LightGBM")
    log.info(f"  log file: {current_log_path()}")
    log.info("=" * 70)

    # ------------------------------------------------------------------
    # Load artifacts from training
    # ------------------------------------------------------------------
    log.info("[1/4] Loading models and feature column list...")
    models = load_all_models()
    feature_cols = load_feature_columns()
    region_stats = pd.read_csv(MODELS_DIR / "region_stats.csv")

    preproc_artifacts = None
    if USE_PREPROCESSING:
        from preprocessing import load_preprocessing_artifacts
        preproc_artifacts = load_preprocessing_artifacts()
        log.info(f"  Loaded preprocessing artifacts "
                 f"(winsor={len(preproc_artifacts[0])}, log/sqrt={len(preproc_artifacts[1])}, "
                 f"rank={len(preproc_artifacts[3])})")

    # ------------------------------------------------------------------
    # Process test data
    # ------------------------------------------------------------------
    log.info("[2/4] Loading and aggregating test daily → weekly...")
    test_weekly = load_and_aggregate_daily_to_weekly(
        TEST_PATH, is_train=False, preproc_artifacts=preproc_artifacts,
    )

    log.info("[3/4] Constructing lag features for test...")
    test_features = construct_lag_features(test_weekly, is_train=False)
    test_features = add_temporal_features(test_features)
    test_features = add_region_features(test_features, region_stats)

    if USE_EXTRA_FEATURES:
        score_clim = pd.read_csv(MODELS_DIR / "score_climatology.csv")
        meteo_clim = pd.read_csv(MODELS_DIR / "meteo_climatology.csv")
        test_features = add_woy_score_climatology(test_features, score_clim)
        test_features = add_anomaly_features(test_features, meteo_clim)
        test_features = add_trend_features(test_features)
        test_features = add_interaction_features(test_features)

    if USE_SCORE_LAG_FEATURES:
        region_last_scores = pd.read_csv(MODELS_DIR / "region_last_scores.csv")
        test_features = add_score_lag_features_test(test_features, region_last_scores)

    if USE_REGION_CLUSTER_FEATURES:
        cluster_table = pd.read_csv(REGION_CLUSTERS_CSV)
        test_features = add_region_cluster_features(test_features, cluster_table)

    # ------------------------------------------------------------------
    # Sanity checks: row count and feature alignment
    # ------------------------------------------------------------------
    assert len(test_features) == N_REGIONS, (
        f"Expected {N_REGIONS} test rows, got {len(test_features)}"
    )
    missing_cols = [c for c in feature_cols if c not in test_features.columns]
    assert not missing_cols, f"Missing feature columns: {missing_cols[:5]}"

    X_test = test_features[feature_cols]

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    log.info("[4/4] Predicting 5 horizons...")
    raw_preds = {}
    for h in tqdm(HORIZONS, desc="Horizons", unit="horizon"):
        raw_preds[h] = _predict(models[h], X_test)

    submission = pd.DataFrame({"region_id": test_features["region_id"].values})
    for h in HORIZONS:
        submission[f"pred_week{h}"] = clip_predictions(raw_preds[h])

    assert submission.shape == (N_REGIONS, 6), (
        f"Unexpected submission shape: {submission.shape}"
    )
    pred_cols = [f"pred_week{h}" for h in HORIZONS]
    assert (submission[pred_cols] >= 0).all().all()
    assert (submission[pred_cols] <= 5).all().all()

    submission.to_csv(SUBMISSION_PATH, index=False)
    elapsed = time.time() - t0

    log.info(f"Submission saved to {SUBMISSION_PATH}")
    log.info(f"Shape: {submission.shape}   Elapsed: {elapsed:.1f}s")
    log.info("Prediction summary:\n" + str(submission[pred_cols].describe().round(3)))


if __name__ == "__main__":
    main()
