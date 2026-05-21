import argparse
import gc
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from logging_setup import get_logger, log_startup_banner
log = get_logger("train", label="train")

from config import (
    TRAIN_PATH, TEST_PATH, METEO_COLS,
    CLIMATOLOGY_PATH, PROXY_RIDGE_PATH, PREPROC_PATH,
    USE_PREPROCESSING, USE_CACHE, USE_PER_HORIZON_MODELS,
    PREPROC_LOG_FEATURES,
    N_WORKERS, USE_ADVERSARIAL_WEIGHTS,
    USE_GAP_STRATIFIED, N_PH_FOLDS_SECONDARY,
    USE_CALENDAR_SEASON_WEIGHTS, CALENDAR_SEASON_SLACK,
    IN_SEASON_WEIGHT_BOOST, CALENDAR_SEVERE_THRESHOLD,
    SEED,
    ensure_dirs,
)
from climatology import compute_climatology, save_climatology, load_climatology
from proxy import fit_proxy_ridge, save_proxy_ridge, load_proxy_ridge
from data_pipeline import build_training_dataset, make_sample_weights
from features import build_feature_names, save_feature_names
from model import (
    train_lgbm_per_horizon, train_lgbm_cv,
    fit_calibrator, save_models,
    predict_for_test,
)
from evaluation import compute_metrics, print_metrics
from cache import feature_cache_key, load_from_cache, save_to_cache


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    log.info("Loading train.csv ...")
    train = pd.read_csv(
        TRAIN_PATH,
        dtype={c: "float32" for c in METEO_COLS} | {"score": "float32"},
    )
    train = train.sort_values(["region_id", "date"]).reset_index(drop=True)
    train["month"] = train["date"].str.split("-").str[1].astype("int8")
    log.info(f"  train: {train.shape}")
    return train


# ─────────────────────────────────────────────────────────────────────────────
# Region / Gap validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_regions_and_gap(
    train: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float], dict[str, int]]:
    """Return (gap_array, region_gap_dict, region_test_months).

    region_gap_dict:    {region_id: gap_weeks}  for Score Lag Gap Shift
    region_test_months: {region_id: month}      for Calendar-Matched Validation
    """
    log.info("── Region & Gap validation ──")
    test = pd.read_csv(TEST_PATH, usecols=["region_id", "date"])
    test["month"] = test["date"].str.split("-").str[1].astype("int8")

    train_regions = set(train["region_id"].unique())
    test_regions  = set(test["region_id"].unique())
    both      = train_regions & test_regions
    only_test = test_regions - train_regions

    log.info(f"  Train regions: {len(train_regions):,}  Test: {len(test_regions):,}  "
             f"Overlap: {len(both):,}  Only-test: {len(only_test)}")
    if only_test:
        log.warning(f"  {len(only_test)} test regions not seen in train!")

    train_ends        = train.groupby("region_id")["date"].last()
    test_starts       = test.groupby("region_id")["date"].first()
    test_month_series = test.groupby("region_id")["month"].first()

    def _parse_ymd(s: str):
        parts = str(s).strip().split("T")[0].split(" ")[0].split("-")
        return int(parts[0]), int(parts[1]), int(parts[2])

    gaps: list[float] = []
    region_gap_dict:    dict[str, float] = {}
    region_test_months: dict[str, int]   = {}

    for r in list(both):
        try:
            region_test_months[str(r)] = int(test_month_series[r])
        except Exception:
            region_test_months[str(r)] = 6

        te, ts = train_ends[r], test_starts[r]
        try:
            te_y, te_m, te_d = _parse_ymd(te)
            ts_y, ts_m, ts_d = _parse_ymd(ts)
            gap_wk = ((ts_y - te_y) * 365 + (ts_m - te_m) * 30 + (ts_d - te_d)) / 7.0
            gaps.append(gap_wk)
            region_gap_dict[str(r)] = float(gap_wk)
        except Exception:
            region_gap_dict[str(r)] = 52.0

    if gaps:
        arr = np.array(gaps)
        p25, p50, p75 = np.percentile(arr, [25, 50, 75])
        log.info(f"  Gap (weeks): mean={arr.mean():.1f}  median={p50:.1f}  "
                 f"std={arr.std():.1f}  [{arr.min():.1f}, {arr.max():.1f}]  "
                 f"P25={p25:.1f}  P75={p75:.1f}")
        log.info(f"  region_test_months sample: {dict(list(region_test_months.items())[:5])}")

    return (np.array(gaps) if gaps else np.array([])), region_gap_dict, region_test_months


# ─────────────────────────────────────────────────────────────────────────────
# Calendar season weights
# ─────────────────────────────────────────────────────────────────────────────

def _compute_season_weights(
    y:                  np.ndarray,
    groups:             np.ndarray,
    month_keys:         np.ndarray,
    region_test_months: dict[str, int],
) -> np.ndarray:
    """Return a per-sample season weight multiplier (float32, mean ≈ 1.0).

    In-season samples (circular month distance ≤ CALENDAR_SEASON_SLACK) receive
    IN_SEASON_WEIGHT_BOOST.  Off-season samples receive 1.0.  Severe drought
    samples (score ≥ CALENDAR_SEVERE_THRESHOLD) always receive the boost so
    that rare high-severity events are never down-weighted regardless of season.
    The array is normalized to mean = 1.0 so it combines cleanly with other
    sample weight components.
    """
    test_months = np.array(
        [region_test_months.get(str(g), 0) for g in groups], dtype=np.int32
    )
    dist = np.abs(month_keys.astype(np.int32) - test_months)
    dist = np.minimum(dist, 12 - dist)

    is_in_season = (dist <= CALENDAR_SEASON_SLACK) & (test_months > 0)
    is_severe    = y >= CALENDAR_SEVERE_THRESHOLD

    weights = np.where(is_in_season | is_severe,
                       float(IN_SEASON_WEIGHT_BOOST), 1.0).astype(np.float32)
    weights /= weights.mean()

    n_in   = int(is_in_season.sum())
    n_sev  = int((is_severe & ~is_in_season).sum())
    n_off  = int((~is_in_season & ~is_severe).sum())
    log.info(f"  Season weights: in-season={n_in:,}  severe(off-season)={n_sev:,}  "
             f"off-season={n_off:,}  boost={IN_SEASON_WEIGHT_BOOST}×")
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Stages 0–2
# ─────────────────────────────────────────────────────────────────────────────

def stage0_preprocessing(force: bool = False):
    if not USE_PREPROCESSING:
        return None
    if PREPROC_PATH.exists() and not force:
        log.info("[Stage 0] Already exists, loading ...")
        from preprocessing import load_preprocessing_artifacts
        return load_preprocessing_artifacts()

    log.info("[Stage 0] Fitting preprocessing artifacts ...")
    t0 = time.time()
    from preprocessing import (
        compute_winsor_bounds, compute_imputation_table,
        save_preprocessing_artifacts,
    )
    train  = load_data()
    bounds = compute_winsor_bounds(train)
    table  = compute_imputation_table(train)
    save_preprocessing_artifacts(bounds, dict(PREPROC_LOG_FEATURES), table, {})
    del train; gc.collect()
    log.info(f"[Stage 0] done in {(time.time()-t0)/60:.2f} min")
    from preprocessing import load_preprocessing_artifacts
    return load_preprocessing_artifacts()


def stage1_climatology(train: pd.DataFrame, force: bool = False):
    if CLIMATOLOGY_PATH.exists() and not force:
        log.info("[Stage 1] Already exists, loading ...")
        return load_climatology()
    log.info("[Stage 1] Computing climatology ...")
    t0  = time.time()
    clim = compute_climatology(train)
    save_climatology(clim)
    log.info(f"[Stage 1] done in {(time.time()-t0)/60:.2f} min")
    return clim


def stage2_proxy_ridge(train: pd.DataFrame, clim_dict: dict, force: bool = False):
    if PROXY_RIDGE_PATH.exists() and not force:
        log.info("[Stage 2] Already exists, loading ...")
        return load_proxy_ridge()
    log.info("[Stage 2] Fitting Proxy Ridge ...")
    t0          = time.time()
    proxy_ridge = fit_proxy_ridge(train, clim_dict)
    save_proxy_ridge(proxy_ridge)
    log.info(f"[Stage 2] done in {(time.time()-t0)/60:.2f} min")
    return proxy_ridge


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Training feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def stage3_build_features(
    train, clim_dict, month_dict, sstat_dict, smo_dict, proxy_ridge,
    preproc_artifacts, region_gap_dict=None, region_test_months=None, force=False,
):
    key = feature_cache_key()
    log.info(f"[Stage 3] cache key={key}")

    if USE_CACHE and not force:
        cached = load_from_cache("train_features", key)
        if cached is not None:
            log.info("[Stage 3] Cache hit")
            return cached

    log.info("[Stage 3] Building training feature matrix ...")
    t0            = time.time()
    feature_names = build_feature_names()
    save_feature_names(feature_names)
    log.info(f"  Features: {len(feature_names)}")

    result = build_training_dataset(
        train, clim_dict, month_dict, sstat_dict, smo_dict, proxy_ridge,
        preproc_artifacts=preproc_artifacts,
        region_gap_dict=region_gap_dict,
        region_test_months=region_test_months,
    )

    if USE_CACHE:
        p = save_to_cache(result, "train_features", key)
        log.info(f"[Stage 3] Cache saved → {p}")
    log.info(f"[Stage 3] done in {(time.time()-t0)/60:.2f} min")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Model training
# ─────────────────────────────────────────────────────────────────────────────

def stage4_train(
    X, y, groups, time_keys, fw_keys, month_keys=None,
    adversarial_weights=None, region_test_months=None, region_gap_dict=None,
) -> tuple[dict, dict, np.ndarray, dict]:
    """Train all models and return (models, eval_report, oof_preds, region_strata).

    Stage 4a: Per-horizon LightGBM (main model, Calendar-Matched Validation)
    Stage 4b: Gap-Stratified LightGBM (short / long gap strata)
    """
    log.info("[Stage 4] Training started ...")
    t0 = time.time()

    # ── Stage 4a: Main LightGBM ───────────────────────────────────────────────
    log.info(f"[Stage 4a] Per-horizon LightGBM  "
             f"({X.shape[1]} features, {len(X):,} samples) ...")
    models, eval_report, oof_preds = train_lgbm_per_horizon(
        X, y, groups, time_keys, fw_keys,
        month_keys=month_keys,
        adversarial_weights=adversarial_weights,
        region_test_months=region_test_months,
    )

    # ── Stage 4b: Gap-Stratified LightGBM ────────────────────────────────────
    region_strata: dict[str, str] = {}
    if USE_GAP_STRATIFIED and region_gap_dict:
        log.info("[Stage 4b] Gap-Stratified LightGBM (short / long strata) ...")
        t_gs = time.time()
        try:
            from gap_stratified import (
                compute_region_strata, split_dataset_by_stratum, save_stratified_models,
            )
            region_strata          = compute_region_strata(region_gap_dict)
            short_data, long_data  = split_dataset_by_stratum(
                X, y, groups, time_keys, fw_keys, month_keys,
                region_strata, adversarial_weights=adversarial_weights,
            )

            log.info(f"[Stage 4b] Short-gap  ({short_data['X'].shape[0]:,} samples, "
                     f"folds={N_PH_FOLDS_SECONDARY}) ...")
            short_models, _, _ = train_lgbm_per_horizon(
                short_data["X"], short_data["y"],
                short_data["groups"], short_data["time_keys"], short_data["fw_keys"],
                month_keys=short_data["month_keys"],
                adversarial_weights=short_data["adversarial_weights"],
                region_test_months=region_test_months,
                n_ph_folds=N_PH_FOLDS_SECONDARY,
            )

            log.info(f"[Stage 4b] Long-gap  ({long_data['X'].shape[0]:,} samples, "
                     f"folds={N_PH_FOLDS_SECONDARY}) ...")
            long_models, _, _ = train_lgbm_per_horizon(
                long_data["X"], long_data["y"],
                long_data["groups"], long_data["time_keys"], long_data["fw_keys"],
                month_keys=long_data["month_keys"],
                adversarial_weights=long_data["adversarial_weights"],
                region_test_months=region_test_months,
                n_ph_folds=N_PH_FOLDS_SECONDARY,
            )

            save_stratified_models(short_models, long_models)
            log.info(f"[Stage 4b] done in {time.time()-t_gs:.1f}s")
        except Exception as e:
            log.warning(f"[Stage 4b] Gap-Stratified failed ({e}), skipping")

    log.info(f"[Stage 4] done in {(time.time()-t0)/60:.2f} min")
    return models, eval_report, oof_preds, region_strata


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(from_stage: int = 0, force: bool = False):
    t_total = time.time()

    created = ensure_dirs()
    for d in created:
        log.info(f"  Created directory: {d}")

    log_startup_banner(log, "Drought Training Pipeline")
    log.info(f"  from_stage={from_stage}  force={force}")

    # ── Stage 0 ──────────────────────────────────────────────────────────────
    if from_stage <= 0:
        preproc_artifacts = stage0_preprocessing(force=force)
    else:
        if USE_PREPROCESSING and PREPROC_PATH.exists():
            from preprocessing import load_preprocessing_artifacts
            preproc_artifacts = load_preprocessing_artifacts()
        else:
            preproc_artifacts = None

    # ── Load train (only when needed) ────────────────────────────────────────
    need_train = (
        (from_stage <= 1 and not (CLIMATOLOGY_PATH.exists() and not force)) or
        (from_stage <= 2 and not (PROXY_RIDGE_PATH.exists() and not force)) or
        (from_stage <= 3)
    )
    train: pd.DataFrame | None = None
    region_gap_dict:    dict[str, float] = {}
    region_test_months: dict[str, int]   = {}

    if need_train:
        train = load_data()
        _, region_gap_dict, region_test_months = validate_regions_and_gap(train)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if from_stage <= 1:
        clim = stage1_climatology(train if train is not None else load_data(), force=force)
    else:
        clim = load_climatology()

    clim_dict  = clim["clim_dict"]
    month_dict = clim["month_dict"]
    sstat_dict = clim["sstat_dict"]
    smo_dict   = clim["smo_dict"]

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if from_stage <= 2:
        proxy_ridge = stage2_proxy_ridge(
            train if train is not None else load_data(), clim_dict, force=force)
    else:
        proxy_ridge = load_proxy_ridge()

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    if from_stage <= 3:
        if train is None:
            train = load_data()
            if not region_gap_dict:
                _, region_gap_dict, region_test_months = validate_regions_and_gap(train)

        result = stage3_build_features(
            train, clim_dict, month_dict, sstat_dict, smo_dict, proxy_ridge,
            preproc_artifacts=preproc_artifacts,
            region_gap_dict=region_gap_dict,
            region_test_months=region_test_months,
            force=force,
        )
    else:
        key    = feature_cache_key()
        result = load_from_cache("train_features", key)
        if result is None:
            log.error("[Stage 3] Cache not found — rerun from stage 3: --from-stage 3")
            raise RuntimeError("Feature cache not found.")
        log.info("[Stage 3] Cache loaded")
        if not region_gap_dict:
            _t = load_data()
            _, region_gap_dict, region_test_months = validate_regions_and_gap(_t)
            del _t; gc.collect()

    X, y, groups, time_keys, fw_keys, month_keys = result
    log.info(f"  X: {X.shape}")

    # ── Stage 3.5: Adversarial Weights ───────────────────────────────────────
    adversarial_weights = None
    if USE_ADVERSARIAL_WEIGHTS and USE_PER_HORIZON_MODELS:
        log.info("[Stage 3.5] Computing Adversarial Weights ...")
        t_av = time.time()
        try:
            from adversarial import compute_region_adversarial_weights, apply_adversarial_weights
            _test_cols  = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()
            _test_meteo = [c for c in METEO_COLS if c in _test_cols]
            _train_raw  = pd.read_csv(TRAIN_PATH, usecols=["region_id"] + METEO_COLS)
            _test_raw   = pd.read_csv(TEST_PATH,  usecols=["region_id"] + _test_meteo)
            av_dict     = compute_region_adversarial_weights(_train_raw, _test_raw)
            del _train_raw, _test_raw; gc.collect()
            # Pass pure ones so adversarial_weights contains only the av-ratio.
            # model.py then multiplies by severity once: severity × av_ratio.
            # (Previously make_sample_weights was passed here, causing severity².)
            base_sw             = np.ones(len(y), dtype=np.float32)
            adversarial_weights = apply_adversarial_weights(base_sw, groups, av_dict)
            log.info(f"[Stage 3.5] done in {time.time()-t_av:.1f}s")
        except Exception as e:
            log.warning(f"[Stage 3.5] Failed ({e}), skipping")

    # ── Stage 3.6: Calendar Season Weights ───────────────────────────────────
    if USE_CALENDAR_SEASON_WEIGHTS and region_test_months:
        log.info("[Stage 3.6] Computing calendar season weights ...")
        season_w = _compute_season_weights(y, groups, month_keys, region_test_months)
        if adversarial_weights is not None:
            adversarial_weights = adversarial_weights * season_w
            adversarial_weights = (adversarial_weights / adversarial_weights.mean()).astype(np.float32)
        else:
            adversarial_weights = season_w

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    models, eval_report, oof_preds, region_strata = stage4_train(
        X, y, groups, time_keys, fw_keys,
        month_keys=month_keys,
        adversarial_weights=adversarial_weights,
        region_test_months=region_test_months or None,
        region_gap_dict=region_gap_dict,
    )

    # ── Overall OOF MAE ───────────────────────────────────────────────────────
    valid   = ~np.isnan(oof_preds)
    oof_mae = (
        float(mean_absolute_error(y[valid], np.clip(oof_preds[valid], 0, 5)))
        if valid.sum() > 0 else float("nan")
    )
    log.info(f"OOF MAE: {oof_mae:.4f}")

    # ── Stage 5: Calibration + persistence ────────────────────────────────────
    log.info("[Stage 5] Calibration + saving ...")
    calibrator = fit_calibrator(oof_preds, y)
    save_models(models, calibrator, oof_mae)

    elapsed = time.time() - t_total
    log.info("=" * 60)
    log.info(f"Training done  {elapsed/60:.1f} min  |  OOF MAE: {oof_mae:.4f}")
    log.info(f"Features: {X.shape[1]}  |  Gap strata: {len(region_strata)} regions")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="drought training pipeline")
    p.add_argument("--from-stage", type=int, default=0, metavar="N",
                   help="Resume from stage N (0=full run, 3=skip feature engineering)")
    p.add_argument("--force", action="store_true",
                   help="Force rebuild of all artifacts")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(from_stage=args.from_stage, force=args.force)
