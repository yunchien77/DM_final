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
    CALENDAR_MATCHED_VALIDATION, CALENDAR_MATCHED_SLACK_WEEKS,
    CALENDAR_MATCHED_LAST_YEAR_ONLY,
    USE_FEATURE_AUDIT_PRUNE, PRUNED_FEATURE_LIST_PATH,
    USE_CALENDAR_MATCHED_TRAINING, OFF_SEASON_TAIL_FRAC,
    CALENDAR_MATCHED_TRAINING_SEED,
    USE_ADVERSARIAL_WEIGHTING, AV_CLIP_LO, AV_CLIP_HI,
    AV_CLASSIFIER_MAX_TRAIN_ROWS, AV_WEIGHTS_PATH,
)
from cache import cache_path, feature_cache_key, load_from_cache, save_to_cache
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
    calendar_matched_split,
    compute_test_anchor_woy_per_region,
    last_anchor_per_region_mask,
    cluster_stratified_report,
)


# ---------------------------------------------------------------------------
# Single-model training helpers (L1 LightGBM)
# ---------------------------------------------------------------------------
# Per memory: feedback-dmfp-loss-choice — L1 (median predictor) is the right
# loss for this MAE-evaluated, 67%-zeros target. Two-stage and Tweedie were
# tested and both lost to L1 on validation; their code paths have been removed.

from model import train_single_horizon, save_all_models


def _make_sample_weight(y_train, av_weights: np.ndarray | None = None) -> np.ndarray:
    """Severity weight × (optional) adversarial-validation weight.

    Severity weight per-row: 1 + α·𝟙[y>0] + β·𝟙[y≥3]. Upweights severe rows
    in the L1 gradient. AV weight (Phase 3b) reweights for covariate shift.
    Final array is renormalized to mean=1 so the effective learning rate
    is unchanged.
    """
    y = np.asarray(y_train, dtype=np.float32)
    w = (
        1.0
        + SAMPLE_WEIGHT_ALPHA * (y > 0).astype(np.float32)
        + SAMPLE_WEIGHT_BETA * (y >= 3).astype(np.float32)
    )
    if av_weights is not None:
        w = w * np.asarray(av_weights, dtype=np.float32)
        w = w / w.mean()
    return w


def _apply_calendar_matched_training_filter(
    train_split: pd.DataFrame,
    test_woy_per_region: dict,
    slack_weeks: int,
    off_season_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Phase 3a — keep all in-season anchors + off_season_frac of off-season
    anchors. Logs the kept counts; returns a re-indexed DataFrame.
    """
    woys = ((train_split["day_of_year"].astype(np.int32) - 1) // 7).clip(0, 52).astype(np.int16).values
    test_woys = train_split["region_id"].map(test_woy_per_region).astype(np.int16).values
    dist = np.abs(woys - test_woys)
    dist = np.minimum(dist, 52 - dist)
    in_season = dist <= slack_weeks

    rng = np.random.default_rng(seed)
    off_season_keep = rng.random(len(train_split)) < off_season_frac
    keep_mask = in_season | (~in_season & off_season_keep)

    kept = train_split[keep_mask].reset_index(drop=True)
    n_in = int((in_season & keep_mask).sum())
    n_off = int(((~in_season) & keep_mask).sum())
    log.info(
        f"  [phase3a] calendar-matched training filter: "
        f"{len(kept):,}/{len(train_split):,} kept  "
        f"(in-season={n_in:,}, off-season tail={n_off:,} @ {off_season_frac*100:.0f}%)"
    )
    return kept


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
    log_features = {"prec": "log1p", "surf_pre": "sqrt"}

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
    log.info(f"  USE_FEATURE_AUDIT_PRUNE={USE_FEATURE_AUDIT_PRUNE}")
    log.info(f"  USE_CALENDAR_MATCHED_TRAINING={USE_CALENDAR_MATCHED_TRAINING} (off_season_tail={OFF_SEASON_TAIL_FRAC})")
    log.info(f"  USE_ADVERSARIAL_WEIGHTING={USE_ADVERSARIAL_WEIGHTING} (clip=[{AV_CLIP_LO}, {AV_CLIP_HI}])")
    log.info(f"  SAMPLE_WEIGHT_ALPHA={SAMPLE_WEIGHT_ALPHA}  SAMPLE_WEIGHT_BETA={SAMPLE_WEIGHT_BETA}")
    log.info(f"  RUN_WALK_FORWARD_DIAGNOSTIC={RUN_WALK_FORWARD_DIAGNOSTIC}")
    log.info(f"  log file: {current_log_path()}")
    log.info("=" * 70)


def _build_train_features():
    """Run stages [0/5]–[3/5]: preprocessing artifacts → weekly aggregation →
    lag features → temporal/region/extra features. Returns
    (train_split, val_split, all_feature_cols).

    Side effects: persists per-region artifacts under MODELS_DIR
    (region_stats.csv, region_clusters.csv, region_last_scores.csv,
    score_climatology.csv, meteo_climatology.csv, feature_cols.json) that
    predict.py consumes.
    """
    # Stage 0: Fit daily-level preprocessing artifacts (if enabled)
    preproc_artifacts = None
    if USE_PREPROCESSING:
        log.info("[0/5] Fitting daily-level preprocessing artifacts...")
        t_stage = time.time()
        preproc_artifacts = _fit_preprocessing_artifacts()
        log.info(f"[0/5] done in {(time.time() - t_stage) / 60:.2f} min")

    # Stage 1: Load and aggregate
    log.info("[1/5] Loading and aggregating train daily → weekly...")
    t_stage = time.time()
    train_weekly = load_and_aggregate_daily_to_weekly(
        TRAIN_PATH, is_train=True, preproc_artifacts=preproc_artifacts,
    )
    log.info(f"[1/5] train_weekly: {train_weekly.shape}, done in {(time.time() - t_stage) / 60:.2f} min")

    # Stage 2: Construct lag features
    log.info("[2/5] Constructing lag features...")
    t_stage = time.time()
    train_features = construct_lag_features(train_weekly, lag_window=LAG_WINDOW, is_train=True)
    log.info(f"[2/5] train_features: {train_features.shape}, done in {(time.time() - t_stage) / 60:.2f} min")

    # Stage 3: Add temporal + region features
    log.info("[3/5] Adding temporal and region features...")
    t_stage = time.time()
    if CALENDAR_MATCHED_VALIDATION:
        test_woy_per_region = compute_test_anchor_woy_per_region(TEST_PATH)
        woy_series = pd.Series(test_woy_per_region.values())
        log.info(
            f"  test_end_woy per region: median={int(woy_series.median())} "
            f"p25={int(woy_series.quantile(0.25))} p75={int(woy_series.quantile(0.75))} "
            f"min={int(woy_series.min())} max={int(woy_series.max())} (n={len(woy_series)})"
        )
        train_split, val_split = calendar_matched_split(
            train_features,
            test_woy_per_region=test_woy_per_region,
            slack_weeks=CALENDAR_MATCHED_SLACK_WEEKS,
            last_year_only=CALENDAR_MATCHED_LAST_YEAR_ONLY,
        )
        log.info(f"  calendar_matched_split → train: {len(train_split):,}  val: {len(val_split):,}")
    else:
        test_woy_per_region = None
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

    # Phase 2 — audit prune. Apply the gain-ranked top-70% allowlist from
    # feature_cols_pruned.json (generated from the Phase 1b models), with
    # an automatic carve-out for new feature blocks that didn't exist when
    # the prune list was built (currently the *_p95_lag0 tail-intensity
    # columns from the Phase 2 STAT_SUFFIXES extension).
    if USE_FEATURE_AUDIT_PRUNE:
        import json as _json
        if PRUNED_FEATURE_LIST_PATH.is_file():
            with open(PRUNED_FEATURE_LIST_PATH) as f:
                pruned_keep = set(_json.load(f))
            new_block = [c for c in all_feature_cols
                         if c not in pruned_keep and "_p95_" in c]
            kept = [c for c in all_feature_cols if c in pruned_keep or c in new_block]
            log.info(
                f"  [audit] {len(kept)}/{len(all_feature_cols)} features kept "
                f"(prune-list ∩ built: {len(kept) - len(new_block)}; new block carve-out: {len(new_block)})"
            )
            all_feature_cols = kept
        else:
            log.warning(
                f"  [audit] USE_FEATURE_AUDIT_PRUNE=True but {PRUNED_FEATURE_LIST_PATH} "
                f"missing — using all {len(all_feature_cols)} features"
            )

    save_feature_columns(all_feature_cols)

    region_stats.to_csv(MODELS_DIR / "region_stats.csv", index=False)
    log.info(f"  Region stats saved → {MODELS_DIR / 'region_stats.csv'}")
    log.info(f"[3/5] done in {(time.time() - t_stage) / 60:.2f} min")

    return train_split, val_split, all_feature_cols


def main():
    t0 = time.time()
    _log_run_config()

    # ------------------------------------------------------------------
    # Cache check: skip stages [0/5]–[3/5] when feature config + raw data
    # are unchanged. On hit we also require feature_cols.json on disk so
    # predict.py has the column ordering to match.
    # ------------------------------------------------------------------
    key = feature_cache_key()
    log.info(f"[cache] key={key}  path={cache_path('train_features', key)}")
    cached = load_from_cache("train_features", key)
    artifacts_present = (MODELS_DIR / "feature_cols.json").is_file()

    if cached is not None and artifacts_present:
        log.info(f"[cache] hit: skipping [0/5]–[3/5] (saved ~4.7 min)")
        train_split = cached["train_split"]
        val_split = cached["val_split"]
        all_feature_cols = cached["all_feature_cols"]
    else:
        if cached is None:
            log.info(f"[cache] miss: rebuilding features")
        else:
            log.info(f"[cache] hit but feature_cols.json missing — rebuilding")
        train_split, val_split, all_feature_cols = _build_train_features()
        save_to_cache(
            {
                "train_split": train_split,
                "val_split": val_split,
                "all_feature_cols": all_feature_cols,
            },
            "train_features", key,
        )
        log.info(f"[cache] saved → {cache_path('train_features', key)}")

    # ------------------------------------------------------------------
    # Phase 3 — covariate-shift correction
    #   3b (first): fit AV classifier on the *unfiltered* train_split so it
    #               sees the full off-season distribution
    #   3a:        apply calendar-matched training filter to train_split
    #   3b (then): compute per-row AV weights aligned to the filtered rows
    # Both knobs are training-only and don't affect the feature cache.
    # ------------------------------------------------------------------
    av_weights = None
    av_model = None
    av_cols: list = []
    if USE_ADVERSARIAL_WEIGHTING or USE_CALENDAR_MATCHED_TRAINING:
        test_woy_per_region = compute_test_anchor_woy_per_region(TEST_PATH)
    else:
        test_woy_per_region = None

    if USE_ADVERSARIAL_WEIGHTING:
        from adversarial import (
            make_av_labels, fit_av_classifier, compute_av_weights,
            effective_sample_size, select_lag0_meteo_cols,
        )
        av_cols = select_lag0_meteo_cols(all_feature_cols)
        log.info(
            f"  [phase3b] fitting AV classifier on {len(train_split):,} pre-filter "
            f"train rows × {len(av_cols)} lag-0 meteo features..."
        )
        av_labels = make_av_labels(
            train_split, test_woy_per_region, CALENDAR_MATCHED_SLACK_WEEKS,
        )
        av_model, av_auc = fit_av_classifier(
            train_split, av_labels, av_cols,
            max_train=AV_CLASSIFIER_MAX_TRAIN_ROWS, seed=42, log_fn=log.info,
        )
        if av_auc < 0.65:
            log.warning(
                f"  [phase3b] AV AUC={av_auc:.3f} < 0.65 — distribution shift on "
                f"meteo lag-0 is weak; weights may not help"
            )

    if USE_CALENDAR_MATCHED_TRAINING:
        train_split = _apply_calendar_matched_training_filter(
            train_split, test_woy_per_region,
            slack_weeks=CALENDAR_MATCHED_SLACK_WEEKS,
            off_season_frac=OFF_SEASON_TAIL_FRAC,
            seed=CALENDAR_MATCHED_TRAINING_SEED,
        )

    if USE_ADVERSARIAL_WEIGHTING and av_model is not None:
        from adversarial import compute_av_weights, effective_sample_size
        av_weights, av_probs = compute_av_weights(
            av_model, train_split, av_cols, AV_CLIP_LO, AV_CLIP_HI,
        )
        ess, ess_frac = effective_sample_size(av_weights)
        log.info(
            f"  [phase3b] AV weights: n={len(av_weights):,}  "
            f"mean={av_weights.mean():.3f}  std={av_weights.std():.3f}  "
            f"min={av_weights.min():.3f}  max={av_weights.max():.3f}  "
            f"ESS/N={ess_frac:.3f}"
        )
        if ess_frac < 0.30:
            log.warning(
                f"  [phase3b] ESS/N={ess_frac:.3f} < 0.30 — weighting too aggressive. "
                f"Tighten AV_CLIP_HI or disable USE_ADVERSARIAL_WEIGHTING."
            )
        pd.DataFrame({
            "region_id": train_split["region_id"].values,
            "week_idx": train_split["week_idx"].values,
            "av_prob": av_probs,
            "av_weight": av_weights,
        }).to_csv(AV_WEIGHTS_PATH, index=False)
        log.info(f"  [phase3b] weights saved → {AV_WEIGHTS_PATH}")

    X_train = train_split[all_feature_cols]
    X_val = val_split[all_feature_cols]
    log.info(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")

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
        sw = _make_sample_weight(y_train, av_weights=av_weights)
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

    # Test-shape val subset: 1 anchor/region. Kaggle scores 1 anchor per
    # region; the broad macro-MAE over many anchors per region is
    # structurally optimistic. When calendar-matched validation is active we
    # pick the val anchor whose woy is closest to that region's test_end_woy
    # — the higher-fidelity Kaggle proxy.
    if CALENDAR_MATCHED_VALIDATION:
        test_woy_per_region = compute_test_anchor_woy_per_region(TEST_PATH)
        la_mask = last_anchor_per_region_mask(val_split, test_woy_per_region)
        la_val = val_split.loc[la_mask].reset_index(drop=True)
        la_preds = {h: val_preds[h][la_mask] for h in HORIZONS}
        la_macro_mae, la_horizon_maes = evaluate_predictions(la_val, la_preds)
        log.info(
            f"  last-anchor-per-region (n={len(la_val):,}) macro MAE = {la_macro_mae:.4f}  "
            f"per-horizon = {{{', '.join(f'w{h}={la_horizon_maes[h]:.4f}' for h in HORIZONS)}}}"
        )
    else:
        last_anchor = int(val_split["week_idx"].max())
        la_mask = val_split["week_idx"].values == last_anchor
        la_val = val_split.loc[la_mask]
        la_preds = {h: val_preds[h][la_mask] for h in HORIZONS}
        la_macro_mae, la_horizon_maes = evaluate_predictions(la_val, la_preds)
        log.info(f"  last-anchor (week={last_anchor}, n={len(la_val):,}) macro MAE = {la_macro_mae:.4f}  "
                 f"per-horizon = {{{', '.join(f'w{h}={la_horizon_maes[h]:.4f}' for h in HORIZONS)}}}")

    # Per-cluster MAE: the high-severity archetype cluster (363 regions)
    # carries most of the macro-MAE. Later phases must improve THAT cluster,
    # not just the easy ones. Reading the cluster table from disk avoids a
    # re-fit on every run.
    cluster_csv = MODELS_DIR / "region_clusters.csv"
    if USE_REGION_CLUSTER_FEATURES and cluster_csv.is_file():
        cluster_table = pd.read_csv(cluster_csv, dtype={"region_id": str})
        cluster_report = cluster_stratified_report(val_split, val_preds, cluster_table)
        log.info("  Per-cluster val MAE (sorted hardest-first):")
        log.info("    " + cluster_report.to_string(index=False).replace("\n", "\n    "))
        cluster_report.to_csv(MODELS_DIR.parent / "diagnostics" / "per_cluster_mae.csv", index=False)

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
