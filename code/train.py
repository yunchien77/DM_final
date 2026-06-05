"""Phase 7 training entry point.

Pipeline:
  Stage 0 — fit daily preprocessing artifacts (USE_PREPROCESSING)
  Stage 1 — daily → weekly aggregation (incl. pressure-derived stats)
  Stage 2 — anchor rows
  Stage 3 — features: score_lag1, region history, calendar, score climatology,
            climate features, sliding-window stats
  Stage 4 — (optional) shift-aware collinearity pruning
  Stage 5 — train 5 LightGBM L1 boosters
  Stage 6 — build drought transition matrices

Run:
    python train.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from logging_setup import get_logger
log = get_logger("train", label="train")

from config import (
    TRAIN_PATH, TEST_PATH, MODELS_DIR, HORIZONS, LAG_WINDOW,
    USE_PREPROCESSING, USE_RANK_NORMALIZATION, RANK_NORMALIZE_FEATURES,
    USE_SCORE_LAG, USE_CLIMATE_FEATURES, USE_WINDOWED_FEATURES,
    USE_PROXY_SCORE, USE_METEO_CLUSTER,
    USE_TRANSITION_SMOOTHING, TRANSITION_CLUSTERS, TRANSITION_SMOOTHING_ALPHA,
    USE_TRANSITION_WEIGHT, TRANSITION_WEIGHT_GAMMA,
    USE_SEVERITY_WEIGHT, SAMPLE_WEIGHT_ALPHA, SAMPLE_WEIGHT_BETA,
    USE_ZERO_INFLATED,
    USE_WALK_FORWARD_CV, N_WF_FOLDS, WF_PURGE_WEEKS, USE_ISOTONIC_CALIBRATION,
    USE_PRUNING, PRUNING_CORRELATION_THRESHOLD, PRUNING_VIF_THRESHOLD,
    LEAN_FEATURE_LIST_PATH, WINDOWED_CHANNELS, WINDOWED_WINDOWS,
    CALENDAR_MATCHED_VALIDATION, CALENDAR_MATCHED_SLACK_WEEKS,
    CALENDAR_MATCHED_LAST_YEAR_ONLY,
    LGBM_PARAMS, EARLY_STOPPING_ROUNDS,
)
from cache import cache_path, feature_cache_key, load_from_cache, save_to_cache
from data_pipeline import load_and_aggregate_daily_to_weekly, construct_anchor_rows
from features_score import (
    add_calendar_features, compute_region_stats, add_region_features,
    compute_score_climatology, add_score_climatology,
    add_score_lag1_train, compute_region_last_score,
    REGION_STATS_PATH, SCORE_CLIMATOLOGY_PATH, REGION_LAST_SCORE_PATH,
    CALENDAR_FEATURE_COLS, REGION_FEATURE_COLS, SCORE_CLIM_FEATURE_COLS,
    SCORE_LAG_FEATURE_COLS,
)
from features_climate import (
    compute_climate_climatologies, add_climate_features, CLIMATE_FEATURE_COLS,
    PRESSURE_CLIMATOLOGY_PATH, METEO_CLIMATOLOGY_PATH,
    DROUGHT_INDEX_CLIMATOLOGY_PATH, HEAT_CLIMATOLOGY_PATH,
)
from features_windowed import (
    compute_windowed_climatology, add_windowed_features, windowed_feature_cols,
    WINDOWED_CLIMATOLOGY_PATH,
)
from transition_matrix import (
    build_transition_matrices, save_transition_matrices, compute_transition_weight,
)
from proxy_score import (
    fit_proxy_ridge, add_proxy_score, meteo_feature_cols, PROXY_FEATURE_COL,
)
from pruning import prune_features
from model import (
    train_single_horizon, save_all_models, predict_all_horizons,
    train_horizon_walk_forward, save_fold_models,
    fit_isotonic_oof, save_calibrator, apply_calibration,
)
from zero_inflated import (
    train_zi_horizon, save_zi_models, predict_zi,
)
from validate import (
    fixed_holdout_split, calendar_matched_split,
    compute_test_anchor_woy_per_region, compute_region_test_months,
    evaluate_predictions, print_validation_report,
)


# ---------------------------------------------------------------------------
# Feature build
# ---------------------------------------------------------------------------

def add_all_features(
    df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    region_stats: pd.DataFrame,
    score_clim: pd.DataFrame,
    climate_clims: dict | None,
    windowed_clim: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add every feature block. Returns (extended_df, full_feature_col_list)."""
    out = add_calendar_features(df)
    out = add_region_features(out, region_stats)
    out = add_score_climatology(out, score_clim)

    feat_cols: list[str] = list(CALENDAR_FEATURE_COLS) + list(REGION_FEATURE_COLS) + list(SCORE_CLIM_FEATURE_COLS)

    # Phase 8: score_lag1 is always computed (the transition-weight code uses
    # it as the "previous score" reference) but is only added to feature_cols
    # when USE_SCORE_LAG=True. With USE_SCORE_LAG=False (the Phase 8 default),
    # the column exists in the DataFrame but the model never sees it.
    out = add_score_lag1_train(out, weekly_df)
    if USE_SCORE_LAG:
        feat_cols += list(SCORE_LAG_FEATURE_COLS)
    if USE_CLIMATE_FEATURES and climate_clims is not None:
        out, climate_added = add_climate_features(out, weekly_df, climate_clims)
        feat_cols += climate_added
    if USE_WINDOWED_FEATURES and windowed_clim is not None:
        out, windowed_added = add_windowed_features(
            out, weekly_df, windowed_clim,
            channels=WINDOWED_CHANNELS, windows=WINDOWED_WINDOWS,
        )
        feat_cols += windowed_added
    # Deduplicate while preserving order
    seen = set()
    feat_cols = [c for c in feat_cols if not (c in seen or seen.add(c))]
    return out, feat_cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Phase 8 training")
    log.info("=" * 70)
    log.info(
        f"  USE_PROXY_SCORE={USE_PROXY_SCORE}  USE_SCORE_LAG={USE_SCORE_LAG}  "
        f"USE_CLIMATE_FEATURES={USE_CLIMATE_FEATURES}  USE_WINDOWED_FEATURES={USE_WINDOWED_FEATURES}"
    )
    log.info(
        f"  USE_METEO_CLUSTER={USE_METEO_CLUSTER}  USE_TRANSITION_SMOOTHING={USE_TRANSITION_SMOOTHING}  "
        f"USE_TRANSITION_WEIGHT={USE_TRANSITION_WEIGHT}  USE_SEVERITY_WEIGHT={USE_SEVERITY_WEIGHT}"
    )
    log.info(
        f"  LGBM objective={LGBM_PARAMS['objective']}  n_estimators={LGBM_PARAMS['n_estimators']}  "
        f"lr={LGBM_PARAMS['learning_rate']}"
    )

    key = feature_cache_key()
    log.info(f"[cache] key={key}  path={cache_path('train_features', key)}")

    cached = load_from_cache("train_features", key)
    if cached is not None:
        log.info("[cache] HIT — using cached train/val features")
        train_split = cached["train_split"]
        val_split = cached["val_split"]
        feature_cols = cached["feature_cols"]
        weekly_df_full = cached["weekly_df_full"]   # for transition matrices
    else:
        log.info("[cache] MISS — rebuilding features")
        # Stage 0: preprocessing artifacts. Prefer the on-disk artifacts if they
        # exist (they're built once and reused — the daily preprocessing isn't
        # part of the Phase 7 feature philosophy change). Otherwise fit them
        # from train+test daily rows.
        preproc_artifacts = None
        if USE_PREPROCESSING:
            from preprocessing import (
                compute_winsor_bounds, compute_imputation_table, compute_rank_quantiles,
                save_preprocessing_artifacts, load_preprocessing_artifacts,
                PREPROC_JSON,
            )
            if PREPROC_JSON.exists():
                log.info("[stage 0] loading existing preprocessing artifacts")
                preproc_artifacts = load_preprocessing_artifacts()
            else:
                log.info("[stage 0] fitting preprocessing artifacts from train+test daily rows")
                from config import METEO_FEATURES
                train_daily = pd.read_csv(
                    TRAIN_PATH, usecols=["region_id"] + list(METEO_FEATURES))
                test_daily = pd.read_csv(
                    TEST_PATH, usecols=["region_id"] + list(METEO_FEATURES))
                bounds = compute_winsor_bounds(train_daily)
                log_features: dict[str, str] = {}
                imp = compute_imputation_table(train_daily)
                qt = (
                    compute_rank_quantiles(train_daily, test_daily, list(RANK_NORMALIZE_FEATURES))
                    if USE_RANK_NORMALIZATION else {}
                )
                save_preprocessing_artifacts(bounds, log_features, imp, qt)
                preproc_artifacts = (bounds, log_features, imp, qt)
                del train_daily, test_daily

        # Stage 1: daily → weekly
        log.info("[stage 1] daily → weekly")
        train_weekly = load_and_aggregate_daily_to_weekly(
            TRAIN_PATH, is_train=True, preproc_artifacts=preproc_artifacts,
        )

        # Stage 2: anchor rows
        log.info("[stage 2] building anchor rows")
        train_features = construct_anchor_rows(train_weekly, is_train=True, lag_window=LAG_WINDOW)

        # Train/val split (calendar-matched preferred)
        if CALENDAR_MATCHED_VALIDATION:
            test_woy_per_region = compute_test_anchor_woy_per_region(TEST_PATH)
            train_split, val_split = calendar_matched_split(
                train_features, test_woy_per_region,
                slack_weeks=CALENDAR_MATCHED_SLACK_WEEKS,
                last_year_only=CALENDAR_MATCHED_LAST_YEAR_ONLY,
            )
        else:
            train_split, val_split = fixed_holdout_split(train_features)
        log.info(f"  train: {len(train_split):,}  val: {len(val_split):,}")

        # Stage 3: features (climatologies fit on train-fold rows only)
        max_train_anchor = int(train_split["week_idx"].max())
        train_weekly_fold = train_weekly[train_weekly["week_idx"] <= max_train_anchor]
        log.info(f"[stage 3] climatologies on training fold ({len(train_weekly_fold):,} weekly rows)")

        region_stats = compute_region_stats(train_weekly_fold)
        score_clim = compute_score_climatology(train_weekly_fold)
        region_stats.to_csv(REGION_STATS_PATH, index=False)
        score_clim.to_csv(SCORE_CLIMATOLOGY_PATH, index=False)

        climate_clims = None
        if USE_CLIMATE_FEATURES:
            climate_clims = compute_climate_climatologies(train_weekly_fold)
            climate_clims["pressure"].to_csv(PRESSURE_CLIMATOLOGY_PATH, index=False)
            climate_clims["meteo"].to_csv(METEO_CLIMATOLOGY_PATH, index=False)
            climate_clims["drought_index"].to_csv(DROUGHT_INDEX_CLIMATOLOGY_PATH, index=False)
            if climate_clims.get("heat") is not None:
                climate_clims["heat"].to_csv(HEAT_CLIMATOLOGY_PATH, index=False)
            log.info(
                f"  climate climatology shapes: pressure {climate_clims['pressure'].shape}, "
                f"meteo {climate_clims['meteo'].shape}, drought_index {climate_clims['drought_index'].shape}"
            )

        windowed_clim = None
        if USE_WINDOWED_FEATURES:
            log.info("  Computing windowed climatology (slow: O(regions × weeks × windows × stats))...")
            windowed_clim = compute_windowed_climatology(
                train_weekly_fold, channels=WINDOWED_CHANNELS, windows=WINDOWED_WINDOWS,
            )
            windowed_clim.to_csv(WINDOWED_CLIMATOLOGY_PATH, index=False)
            log.info(f"  windowed climatology: {windowed_clim.shape}")

        # Region last scores (for predict.py)
        region_last = compute_region_last_score(train_weekly)
        region_last.to_csv(REGION_LAST_SCORE_PATH, index=False)

        # Stage 3b: apply features to both splits
        log.info("[stage 3b] applying features to train/val splits")
        train_split, feature_cols = add_all_features(
            train_split, train_weekly, region_stats, score_clim, climate_clims, windowed_clim,
        )
        val_split, _ = add_all_features(
            val_split, train_weekly, region_stats, score_clim, climate_clims, windowed_clim,
        )
        log.info(f"  feature_cols total: {len(feature_cols)}")

        weekly_df_full = train_weekly
        save_to_cache(
            {"train_split": train_split, "val_split": val_split,
             "feature_cols": feature_cols, "weekly_df_full": weekly_df_full},
            "train_features", key,
        )

    # Stage 3c — Phase 8 proxy score. Fit AFTER feature build / cache load,
    # BEFORE saving feature_cols.json so the order is stable across train and predict.
    if USE_PROXY_SCORE:
        log.info("[stage 3c] fitting proxy Ridge on meteo features → target_w1")
        proxy_input_cols = [c for c in meteo_feature_cols() if c in train_split.columns]
        proxy_model, proxy_used_cols, train_rho = fit_proxy_ridge(
            train_split, target_col="target_w1", feature_cols=proxy_input_cols,
        )
        log.info(
            f"  proxy_ridge alpha={proxy_model.alpha_:.4g}  "
            f"features_used={len(proxy_used_cols)}  "
            f"train Spearman ρ(proxy_score, target_w1)={train_rho:.4f}"
        )
        train_split = add_proxy_score(train_split, proxy_model, proxy_used_cols)
        val_split = add_proxy_score(val_split, proxy_model, proxy_used_cols)
        if PROXY_FEATURE_COL not in feature_cols:
            feature_cols = feature_cols + [PROXY_FEATURE_COL]

    # Phase 8: drop score_lag1 from feature_cols when USE_SCORE_LAG=False.
    # The column is still in train_split for the transition-weight code; only
    # the model never sees it.
    if not USE_SCORE_LAG:
        feature_cols = [c for c in feature_cols if c not in SCORE_LAG_FEATURE_COLS]

    # Optional meteo cluster feature (gated on diagnostic having produced the table).
    if USE_METEO_CLUSTER:
        cluster_path = MODELS_DIR / "meteo_cluster_table.csv"
        if cluster_path.exists():
            cluster_table = pd.read_csv(cluster_path)[["region_id", "meteo_cluster_id"]]
            train_split = train_split.merge(cluster_table, on="region_id", how="left")
            val_split = val_split.merge(cluster_table, on="region_id", how="left")
            train_split["meteo_cluster_id"] = train_split["meteo_cluster_id"].fillna(-1).astype(np.int32)
            val_split["meteo_cluster_id"] = val_split["meteo_cluster_id"].fillna(-1).astype(np.int32)
            if "meteo_cluster_id" not in feature_cols:
                feature_cols = feature_cols + ["meteo_cluster_id"]
            log.info(
                f"  meteo_cluster_id added ({cluster_path}, "
                f"K={cluster_table['meteo_cluster_id'].nunique()})"
            )
        else:
            log.info(f"  USE_METEO_CLUSTER=True but {cluster_path} not found — skipping cluster feature")

    # Stage 4 — optional pruning
    if USE_PRUNING:
        log.info("[stage 4] aggressive collinearity pruning")
        # Use the val split's feature matrix as the "test-like" set for AV importance
        # (real test features aren't available at training time).
        X_train = train_split[feature_cols].to_numpy(np.float32)
        X_val = val_split[feature_cols].to_numpy(np.float32)
        y = train_split["target_w1"].to_numpy(np.float32)
        hard_keep = list(SCORE_LAG_FEATURE_COLS) + list(REGION_FEATURE_COLS) + list(CALENDAR_FEATURE_COLS)
        kept, report = prune_features(
            X_train, X_val, y, feature_cols, hard_keep,
            output_path=LEAN_FEATURE_LIST_PATH,
            correlation_threshold=PRUNING_CORRELATION_THRESHOLD,
            vif_threshold=PRUNING_VIF_THRESHOLD,
        )
        log.info(f"  pruning: {len(feature_cols)} → {len(kept)} features")
        log.info(f"  pruning report: {report}")
        feature_cols = kept

    # Save the final feature_cols.json so predict.py can align columns
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    log.info(f"  feature_cols saved → {MODELS_DIR / 'feature_cols.json'}")

    # Stage 4b — drought transition matrices (built BEFORE LGBM so they can
    # provide sample weights). Always built when USE_TRANSITION_SMOOTHING or
    # USE_TRANSITION_WEIGHT is on.
    matrices = None
    cluster_assignment = None
    if USE_TRANSITION_SMOOTHING or USE_TRANSITION_WEIGHT:
        log.info("[stage 4b] building drought transition matrices")
        matrices, cluster_assignment = build_transition_matrices(
            weekly_df_full, n_clusters=TRANSITION_CLUSTERS,
            smoothing_alpha=TRANSITION_SMOOTHING_ALPHA,
        )
        save_transition_matrices(
            matrices, cluster_assignment,
            horizons=list(HORIZONS), score_levels=list(range(6)),
        )
        log.info(f"  transition matrices saved: shape {matrices.shape}")

    # Stage 4c — compute per-row sample weights over the COMBINED dataset
    # (train_split + val_split). Walk-forward CV sees both, so weights must
    # be aligned to the same row order.
    combined_split = pd.concat([train_split, val_split], ignore_index=True)
    val_mask = np.concatenate([
        np.zeros(len(train_split), dtype=bool),
        np.ones(len(val_split), dtype=bool),
    ])
    log.info(f"  combined dataset: {len(combined_split):,} rows "
             f"(train={len(train_split):,}, val={len(val_split):,})")

    sample_weight = None
    if USE_TRANSITION_WEIGHT and matrices is not None and "score_lag1" in combined_split.columns:
        cluster_ids = np.zeros(len(combined_split), dtype=np.int32)
        if cluster_assignment is not None and len(cluster_assignment) > 0:
            cluster_ids = (
                combined_split["region_id"].astype(str)
                .map(cluster_assignment).fillna(0).astype(int).to_numpy()
            )
        sample_weight = compute_transition_weight(
            score_lag1=combined_split["score_lag1"].to_numpy(),
            score=combined_split["target_w1"].to_numpy(),
            cluster_ids=cluster_ids,
            matrices=matrices,
            horizon=1,
            horizons=list(HORIZONS),
            score_levels=list(range(6)),
            gamma=TRANSITION_WEIGHT_GAMMA,
        )
        log.info(f"  transition sample_weight: "
                 f"mean={float(sample_weight.mean()):.3f}  std={float(sample_weight.std()):.3f}  "
                 f"min={float(sample_weight.min()):.3f}  max={float(sample_weight.max()):.3f}")

    if USE_SEVERITY_WEIGHT:
        y = combined_split["target_w1"].to_numpy(np.float32)
        sev = (
            1.0
            + SAMPLE_WEIGHT_ALPHA * (y > 0).astype(np.float32)
            + SAMPLE_WEIGHT_BETA * (y >= 3).astype(np.float32)
        )
        sample_weight = sev if sample_weight is None else (sample_weight * sev)
        sample_weight = (sample_weight / sample_weight.mean()).astype(np.float32)
        log.info(
            f"  severity weight applied (α={SAMPLE_WEIGHT_ALPHA}, β={SAMPLE_WEIGHT_BETA}).  "
            f"final sample_weight: mean=1.000  std={float(sample_weight.std()):.3f}  "
            f"min={float(sample_weight.min()):.3f}  max={float(sample_weight.max()):.3f}"
        )

    # Stage 5 — walk-forward CV (or legacy single-split training).
    X_combined = combined_split[feature_cols]
    time_keys = combined_split["week_idx"].to_numpy(np.int32)
    # Per-row helpers for Phase 11 Kaggle-proxy ES + calendar-matched ES
    region_ids_arr = combined_split["region_id"].astype(str).to_numpy()
    months_arr = combined_split["month"].to_numpy(np.int32)
    from config import USE_KAGGLE_PROXY_VAL, USE_CAL_MATCHED_ES
    region_test_months = compute_region_test_months(TEST_PATH)
    log.info(f"  region_test_months: {len(region_test_months)} regions  "
             f"USE_KAGGLE_PROXY_VAL={USE_KAGGLE_PROXY_VAL}  "
             f"USE_CAL_MATCHED_ES={USE_CAL_MATCHED_ES}")

    if USE_ZERO_INFLATED:
        # Legacy two-stage path; not used by Phase 10 default config.
        log.info(
            f"[stage 5] training zero-inflated two-stage (legacy)  "
            f"({len(HORIZONS)} × 2 LightGBM models)"
        )
        zi_models = {}
        oof = np.full((len(combined_split), len(HORIZONS)), np.nan, dtype=np.float32)
        X_tr_legacy = train_split[feature_cols]
        X_va_legacy = val_split[feature_cols]
        for i, h in enumerate(HORIZONS):
            log.info(f"--- horizon {h} (zero-inflated) ---")
            clf, reg = train_zi_horizon(
                X_tr_legacy, train_split[f"target_w{h}"],
                X_va_legacy, val_split[f"target_w{h}"],
                horizon=h,
                sample_weight=sample_weight[~val_mask] if sample_weight is not None else None,
            )
            zi_models[h] = (clf, reg)
            preds = predict_zi({h: (clf, reg)}, X_va_legacy)[h]
            oof[val_mask, i] = np.clip(preds, 0.0, 5.0)
        save_zi_models(zi_models)
        save_calibrator(None)  # no calibration in legacy path
    elif USE_WALK_FORWARD_CV:
        log.info(
            f"[stage 5] walk-forward CV  "
            f"folds={N_WF_FOLDS}  purge_weeks={WF_PURGE_WEEKS}  "
            f"objective={LGBM_PARAMS.get('objective')}  "
            f"sample_weight={'on' if sample_weight is not None else 'off'}"
        )
        fold_models_by_horizon: dict[int, list] = {}
        oof = np.full((len(combined_split), len(HORIZONS)), np.nan, dtype=np.float32)
        for i, h in enumerate(HORIZONS):
            log.info(f"--- horizon {h} (walk-forward) ---")
            y_h = combined_split[f"target_w{h}"]
            fold_models, oof_h = train_horizon_walk_forward(
                X_combined, y_h, time_keys,
                horizon=h, sample_weight=sample_weight,
                n_folds=N_WF_FOLDS, purge_weeks=WF_PURGE_WEEKS,
                region_ids=region_ids_arr,
                months=months_arr,
                region_test_months=region_test_months,
                use_kaggle_proxy_val=USE_KAGGLE_PROXY_VAL,
                use_cal_matched_es=USE_CAL_MATCHED_ES,
            )
            fold_models_by_horizon[h] = fold_models
            oof[:, i] = oof_h
        save_fold_models(fold_models_by_horizon)

        # Fit a single global IsotonicRegression on (all-horizons OOF, all-horizons y).
        if USE_ISOTONIC_CALIBRATION:
            y_concat = np.concatenate([
                combined_split[f"target_w{h}"].to_numpy(np.float32) for h in HORIZONS
            ])
            oof_concat = oof.reshape(-1, order="F")
            calibrator = fit_isotonic_oof(oof_concat, y_concat)
            save_calibrator(calibrator)
        else:
            calibrator = None
            save_calibrator(None)
    else:
        log.info(
            f"[stage 5] legacy single-split training  "
            f"({len(HORIZONS)} LightGBM models)"
        )
        models = {}
        oof = np.full((len(combined_split), len(HORIZONS)), np.nan, dtype=np.float32)
        X_tr_legacy = train_split[feature_cols]
        X_va_legacy = val_split[feature_cols]
        for i, h in enumerate(HORIZONS):
            log.info(f"--- horizon {h} ---")
            model = train_single_horizon(
                X_tr_legacy, train_split[f"target_w{h}"],
                X_va_legacy, val_split[f"target_w{h}"],
                horizon=h,
                sample_weight=sample_weight[~val_mask] if sample_weight is not None else None,
            )
            models[h] = model
            oof[val_mask, i] = np.clip(model.predict(X_va_legacy), 0.0, 5.0)
        save_all_models(models)
        save_calibrator(None)

    # ── Reporting: OOF MAE (raw and calibrated), val-slice MAE ───────────────
    from model import load_calibrator
    cal = load_calibrator()
    oof_clip = np.clip(oof, 0.0, 5.0).astype(np.float32)
    oof_cal = apply_calibration(oof_clip, cal)

    horizon_maes_raw: dict[int, float] = {}
    horizon_maes_cal: dict[int, float] = {}
    horizon_maes_val_cal: dict[int, float] = {}
    for i, h in enumerate(HORIZONS):
        y_true = combined_split[f"target_w{h}"].to_numpy(np.float32)
        m = np.isfinite(oof[:, i])
        horizon_maes_raw[h] = float(np.mean(np.abs(oof_clip[m, i] - y_true[m]))) if m.any() else float("nan")
        horizon_maes_cal[h] = float(np.mean(np.abs(oof_cal[m, i] - y_true[m]))) if m.any() else float("nan")
        m_v = m & val_mask
        horizon_maes_val_cal[h] = (
            float(np.mean(np.abs(oof_cal[m_v, i] - y_true[m_v]))) if m_v.any() else float("nan")
        )

    macro_raw = float(np.mean([v for v in horizon_maes_raw.values() if np.isfinite(v)]))
    macro_cal = float(np.mean([v for v in horizon_maes_cal.values() if np.isfinite(v)]))
    macro_val_cal = float(np.mean([v for v in horizon_maes_val_cal.values() if np.isfinite(v)]))

    log.info("=" * 60)
    log.info("OOF metrics (full combined dataset):")
    for h in HORIZONS:
        log.info(f"  h={h}  raw_mae={horizon_maes_raw[h]:.4f}  cal_mae={horizon_maes_cal[h]:.4f}")
    log.info(f"  MACRO  raw={macro_raw:.4f}  cal={macro_cal:.4f}")
    log.info(f"OOF metrics (calendar-matched val slice only, calibrated):")
    for h in HORIZONS:
        log.info(f"  h={h}  cal_mae={horizon_maes_val_cal[h]:.4f}")
    log.info(f"  MACRO  cal={macro_val_cal:.4f}")
    log.info("=" * 60)

    # Persist OOF for downstream diagnostics
    y_matrix = np.stack(
        [combined_split[f"target_w{h}"].to_numpy(np.float32) for h in HORIZONS], axis=1,
    )
    oof_path = MODELS_DIR / "oof_preds.npz"
    np.savez(
        oof_path,
        oof_raw=oof,
        oof_clip=oof_clip,
        oof_cal=oof_cal,
        y=y_matrix,
        val_mask=val_mask,
        week_idx=time_keys,
        region_id=combined_split["region_id"].astype(str).to_numpy(),
        horizons=np.array(list(HORIZONS), dtype=np.int32),
    )
    log.info(f"  OOF predictions saved → {oof_path}")

    log.info(f"[done] total time = {(time.time() - t0) / 60:.2f} min")


if __name__ == "__main__":
    main()
