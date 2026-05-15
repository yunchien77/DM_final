"""
End-to-end training entry point.

Pipeline (all stages logged to code/logs/train_<timestamp>.log + stdout):
  Stage 0 — fit daily-level preprocessing artifacts (USE_PREPROCESSING)
  Stage 1 — daily → weekly aggregation
  Stage 2 — lag feature matrix
  Stage 3 — temporal + region + woy + score-lag + cluster features
  Stage 4 — 5× per-horizon LightGBM L1 with severe-row sample weighting
  Stage 5 — validation eval + persist (lgbm_h{1..5}.txt, val_preds.npz, etc.)

Usage:
    python train.py
"""

import time
import numpy as np
import pandas as pd

from logging_setup import get_logger, current_log_path
log = get_logger("train", label="train")

from config import (
    TRAIN_PATH, TEST_PATH, LAG_WINDOW, MODELS_DIR, HORIZONS,
    USE_EXTRA_FEATURES,
    USE_PREPROCESSING, USE_SCORE_LAG_FEATURES, USE_REGION_CLUSTER_FEATURES,
    USE_RANK_NORMALIZATION, RANK_NORMALIZE_FEATURES, N_REGION_CLUSTERS,
    SAMPLE_WEIGHT_ALPHA, SAMPLE_WEIGHT_BETA,
    RUN_WALK_FORWARD_DIAGNOSTIC, WALK_FORWARD_FOLD_BOUNDARIES,
    METEO_FEATURES,
)
from data_pipeline import (
    load_and_aggregate_daily_to_weekly,
    construct_lag_features,
    get_feature_columns,
    save_feature_columns,
)
from features import (
    add_temporal_features,
    add_region_features,
    compute_region_stats,
    EXTRA_FEATURE_COLS,
)
if USE_EXTRA_FEATURES:
    from features_extra import (
        compute_woy_score_climatology,
        compute_woy_meteo_climatology,
        add_woy_score_climatology,
        add_anomaly_features,
        add_trend_features,
        add_interaction_features,
        EXTRA_FEATURE_COLS_V2,
    )
if USE_SCORE_LAG_FEATURES:
    from features_extra import (
        add_score_lag_features_train,
        compute_region_last_score_features,
        SCORE_LAG_COLS,
    )
if USE_REGION_CLUSTER_FEATURES:
    from features_extra import (
        fit_region_clusters,
        add_region_cluster_features,
        CLUSTER_FEATURE_COLS,
        REGION_CLUSTERS_CSV,
    )
from validate import (
    fixed_holdout_split,
    evaluate_predictions,
    print_validation_report,
    walk_forward_diagnostic,
)


# ---------------------------------------------------------------------------
# Single-model training helpers (L1 LightGBM)
# ---------------------------------------------------------------------------
# Per memory: feedback-dmfp-loss-choice — L1 (median predictor) is the right
# loss for this MAE-evaluated, 67%-zeros target. Two-stage and Tweedie were
# tested and both lost to L1 on validation; their code paths have been removed.

from model import train_single_horizon, save_all_models


def _make_sample_weight(y_train) -> np.ndarray:
    """1 + α·𝟙[y>0] + β·𝟙[y≥3] — upweights severe rows in L1 gradient."""
    y = np.asarray(y_train, dtype=np.float32)
    return (
        1.0
        + SAMPLE_WEIGHT_ALPHA * (y > 0).astype(np.float32)
        + SAMPLE_WEIGHT_BETA * (y >= 3).astype(np.float32)
    )


def _train_one(X_tr, y_tr, X_val, y_val, h, sample_weight=None):
    return train_single_horizon(
        X_tr, y_tr, X_val, y_val, horizon=h, sample_weight=sample_weight,
    )


def _predict(model, X):
    return np.clip(model.predict(X), 0.0, 5.0)


def _save(models):
    save_all_models(models)


def _fit_preprocessing_artifacts():
    """
    Pass 1 of the daily-level pipeline: scan train (and test, for rank-norm
    pooled quantiles) to fit imputation medians, p1/p99 winsor bounds, log/sqrt
    transform spec, and quantile anchors. Save under code/models/ for predict.py.
    Returns the 4-tuple consumed by data_pipeline.load_and_aggregate_daily_to_weekly.
    """
    from preprocessing import (
        compute_winsor_bounds, compute_imputation_table, compute_rank_quantiles,
        save_preprocessing_artifacts,
    )

    log.info("[preproc] Reading train daily for artifact fitting...")
    train_chunks = []
    for chunk in pd.read_csv(TRAIN_PATH, dtype={"region_id": str, "date": str},
                             chunksize=500_000):
        train_chunks.append(chunk)
    train_daily = pd.concat(train_chunks, ignore_index=True)
    log.info(f"[preproc] train_daily: {len(train_daily):,} rows")

    bounds = compute_winsor_bounds(train_daily, q_lo=0.01, q_hi=0.99)
    imputation_table = compute_imputation_table(train_daily)

    # Skew-driven transforms from Section 8 of the EDA snapshot.
    log_features = {"prec": "log1p"}
    for f in ("wind", "wind_min", "wind_range", "surf_pre"):
        log_features[f] = "sqrt"

    quantile_table: dict = {}
    if USE_RANK_NORMALIZATION:
        log.info("[preproc] Reading test daily for pooled rank quantiles...")
        test_daily = pd.read_csv(TEST_PATH, dtype={"region_id": str, "date": str})
        quantile_table = compute_rank_quantiles(
            train_daily, test_daily, RANK_NORMALIZE_FEATURES, n_quantiles=1000,
        )
        del test_daily

    save_preprocessing_artifacts(bounds, log_features, imputation_table, quantile_table)
    log.info(f"[preproc] artifacts saved: winsor={len(bounds)} "
             f"log/sqrt={len(log_features)} imputation_rows={len(imputation_table)} "
             f"quantile_features={len(quantile_table)}")
    del train_daily
    return bounds, log_features, imputation_table, quantile_table


def _log_run_config():
    log.info("=" * 70)
    log.info(f"DMFP training run | model=L1 LightGBM")
    log.info(f"  USE_PREPROCESSING={USE_PREPROCESSING}  USE_EXTRA_FEATURES={USE_EXTRA_FEATURES}")
    log.info(f"  USE_SCORE_LAG_FEATURES={USE_SCORE_LAG_FEATURES}  USE_REGION_CLUSTER_FEATURES={USE_REGION_CLUSTER_FEATURES}")
    log.info(f"  USE_RANK_NORMALIZATION={USE_RANK_NORMALIZATION}  N_REGION_CLUSTERS={N_REGION_CLUSTERS}")
    log.info(f"  SAMPLE_WEIGHT_ALPHA={SAMPLE_WEIGHT_ALPHA}  SAMPLE_WEIGHT_BETA={SAMPLE_WEIGHT_BETA}")
    log.info(f"  RUN_WALK_FORWARD_DIAGNOSTIC={RUN_WALK_FORWARD_DIAGNOSTIC}")
    log.info(f"  log file: {current_log_path()}")
    log.info("=" * 70)


def main():
    t0 = time.time()
    _log_run_config()

    # ------------------------------------------------------------------
    # Stage 0: Fit daily-level preprocessing artifacts (if enabled)
    # ------------------------------------------------------------------
    preproc_artifacts = None
    if USE_PREPROCESSING:
        log.info("[0/5] Fitting daily-level preprocessing artifacts...")
        t_stage = time.time()
        preproc_artifacts = _fit_preprocessing_artifacts()
        log.info(f"[0/5] done in {(time.time() - t_stage) / 60:.2f} min")

    # ------------------------------------------------------------------
    # Stage 1: Load and aggregate
    # ------------------------------------------------------------------
    log.info("[1/5] Loading and aggregating train daily → weekly...")
    t_stage = time.time()
    train_weekly = load_and_aggregate_daily_to_weekly(
        TRAIN_PATH, is_train=True, preproc_artifacts=preproc_artifacts,
    )
    log.info(f"[1/5] train_weekly: {train_weekly.shape}, done in {(time.time() - t_stage) / 60:.2f} min")

    # ------------------------------------------------------------------
    # Stage 2: Construct lag features
    # ------------------------------------------------------------------
    log.info("[2/5] Constructing lag features...")
    t_stage = time.time()
    train_features = construct_lag_features(train_weekly, lag_window=LAG_WINDOW, is_train=True)
    log.info(f"[2/5] train_features: {train_features.shape}, done in {(time.time() - t_stage) / 60:.2f} min")

    # ------------------------------------------------------------------
    # Stage 3: Add temporal + region features
    # ------------------------------------------------------------------
    log.info("[3/5] Adding temporal and region features...")
    t_stage = time.time()
    train_split, val_split = fixed_holdout_split(train_features)
    log.info(f"  fixed_holdout_split → train: {len(train_split):,}  val: {len(val_split):,}")

    max_train_anchor = train_split["week_idx"].max()
    train_weekly_for_stats = train_weekly[train_weekly["week_idx"] <= max_train_anchor]
    region_stats = compute_region_stats(train_weekly_for_stats)

    train_split = add_temporal_features(train_split)
    train_split = add_region_features(train_split, region_stats)
    val_split = add_temporal_features(val_split)
    val_split = add_region_features(val_split, region_stats)

    extra_cols: list = []
    if USE_EXTRA_FEATURES:
        log.info("  Computing woy climatologies on training-fold weekly rows...")
        score_clim = compute_woy_score_climatology(train_weekly_for_stats)
        meteo_clim = compute_woy_meteo_climatology(train_weekly_for_stats)
        score_clim.to_csv(MODELS_DIR / "score_climatology.csv", index=False)
        meteo_clim.to_csv(MODELS_DIR / "meteo_climatology.csv", index=False)
        log.info(f"  Score climatology: {score_clim.shape};  meteo climatology: {meteo_clim.shape}")

        for df_name, df_ in [("train", train_split), ("val", val_split)]:
            df_ = add_woy_score_climatology(df_, score_clim)
            df_ = add_anomaly_features(df_, meteo_clim)
            df_ = add_trend_features(df_)
            df_ = add_interaction_features(df_)
            if df_name == "train":
                train_split = df_
            else:
                val_split = df_
        extra_cols = list(EXTRA_FEATURE_COLS_V2)

    if USE_SCORE_LAG_FEATURES:
        log.info("  Adding lagged-score features (Section 10 mitigation)...")
        train_split = add_score_lag_features_train(train_split, train_weekly)
        val_split = add_score_lag_features_train(val_split, train_weekly)
        region_last_scores = compute_region_last_score_features(train_weekly)
        region_last_scores.to_csv(MODELS_DIR / "region_last_scores.csv", index=False)
        extra_cols = extra_cols + SCORE_LAG_COLS

    if USE_REGION_CLUSTER_FEATURES:
        log.info(f"  Fitting region clusters (k={N_REGION_CLUSTERS})...")
        cluster_table = fit_region_clusters(
            train_weekly_for_stats, n_clusters=N_REGION_CLUSTERS,
        )
        cluster_table.to_csv(REGION_CLUSTERS_CSV, index=False)
        train_split = add_region_cluster_features(train_split, cluster_table)
        val_split = add_region_cluster_features(val_split, cluster_table)
        extra_cols = extra_cols + CLUSTER_FEATURE_COLS

    lag_feature_cols = get_feature_columns()
    all_feature_cols = lag_feature_cols + EXTRA_FEATURE_COLS + extra_cols
    save_feature_columns(all_feature_cols)

    region_stats.to_csv(MODELS_DIR / "region_stats.csv", index=False)
    log.info(f"  Region stats saved → {MODELS_DIR / 'region_stats.csv'}")

    X_train = train_split[all_feature_cols]
    X_val = val_split[all_feature_cols]
    log.info(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")
    log.info(f"[3/5] done in {(time.time() - t_stage) / 60:.2f} min")

    # ------------------------------------------------------------------
    # Stage 4: Train per-horizon models
    # ------------------------------------------------------------------
    log.info(f"[4/5] Training {len(HORIZONS)} L1 LightGBM models (one per horizon)...")
    t_stage = time.time()
    models = {}
    for h in HORIZONS:
        log.info(f"  → starting horizon {h}")
        t_h = time.time()
        y_train = train_split[f"target_w{h}"]
        y_val_h = val_split[f"target_w{h}"]
        sw = _make_sample_weight(y_train)
        models[h] = _train_one(X_train, y_train, X_val, y_val_h, h, sample_weight=sw)
        log.info(f"  → horizon {h} done in {(time.time() - t_h) / 60:.2f} min")
    log.info(f"[4/5] all horizons trained in {(time.time() - t_stage) / 60:.2f} min")

    # ------------------------------------------------------------------
    # Validate + persist
    # ------------------------------------------------------------------
    log.info("[5/5] Evaluating on validation set...")
    val_preds = {h: _predict(models[h], X_val) for h in HORIZONS}
    macro_mae, horizon_maes = evaluate_predictions(val_split, val_preds)
    print_validation_report(macro_mae, horizon_maes)
    log.info(f"  macro val MAE = {macro_mae:.4f}  "
             f"per-horizon = {{{', '.join(f'w{h}={horizon_maes[h]:.4f}' for h in HORIZONS)}}}")

    maes = [horizon_maes[h] for h in HORIZONS]
    if not all(maes[i] <= maes[i + 1] + 0.05 for i in range(len(maes) - 1)):
        log.warning("MAE is not strictly monotone across horizons (may indicate feature issue)")

    # Save validation predictions for ensembling later. One file per method.
    preds_arr = np.stack([val_preds[h] for h in HORIZONS], axis=1).astype(np.float32)
    truth_arr = np.stack(
        [val_split[f"target_w{h}"].values for h in HORIZONS], axis=1
    ).astype(np.float32)
    np.savez(
        MODELS_DIR / "val_preds.npz",
        preds=preds_arr,
        truth=truth_arr,
        region_ids=val_split["region_id"].values.astype(str),
        week_idx=val_split["week_idx"].values.astype(np.int32),
        horizons=np.array(HORIZONS, dtype=np.int32),
    )
    log.info(f"  Validation predictions saved → val_preds.npz")

    _save(models)

    if RUN_WALK_FORWARD_DIAGNOSTIC:
        log.info(f"[diagnostic] Walk-forward CV across "
                 f"{len(WALK_FORWARD_FOLD_BOUNDARIES)} fold boundaries (faster params)...")
        full_features = pd.concat([train_split, val_split], ignore_index=True)
        wf = walk_forward_diagnostic(
            features_df=full_features,
            feature_cols=all_feature_cols,
            fold_boundaries=WALK_FORWARD_FOLD_BOUNDARIES,
            train_one_fn=_train_one,
            sample_weight_fn=_make_sample_weight,
            n_estimators_override=1000,
        )
        log.info(f"[diagnostic] walk-forward avg macro-MAE: {wf['avg_macro_mae']}")
        log.info(f"[diagnostic] walk-forward LAST fold macro-MAE: {wf['last_fold_macro_mae']}")
        log.info("[diagnostic] (last-fold MAE is the closest stationarity-aware proxy for Kaggle.)")

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info(f"Training complete in {elapsed / 60:.2f} min.")
    log.info(f"Models saved to {MODELS_DIR}/  |  log: {current_log_path()}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
