import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

from logging_setup import get_logger, log_startup_banner
log = get_logger("predict", label="predict")

from config import (
    TEST_PATH, TRAIN_PATH, SAMPLE_PATH, SUBMISSION_PATH,
    METEO_COLS, N_PRED,
    USE_PREPROCESSING, PREPROC_PATH,
    USE_CACHE, USE_GAP_STRATIFIED,
    ensure_dirs,
)
from climatology import load_climatology
from proxy import load_proxy_ridge
from data_pipeline import build_test_features
from model import load_models, predict_for_test
from cache import feature_cache_key, load_from_cache, save_to_cache


def _load_preproc():
    if not USE_PREPROCESSING or not PREPROC_PATH.exists():
        return None
    from preprocessing import load_preprocessing_artifacts
    pa = load_preprocessing_artifacts()
    log.info(f"  preprocessing loaded (winsor={len(pa[0])} cols)")
    return pa


def _compute_region_test_months(test: pd.DataFrame) -> dict[str, int]:
    first_rows = test.groupby("region_id").first().reset_index()
    return {str(r): int(m) for r, m in zip(first_rows["region_id"], first_rows["month"])}


def _load_region_gap_dict() -> dict[str, float]:
    """Reconstruct region_gap_dict from train/test dates for Gap-Stratified inference."""
    try:
        test  = pd.read_csv(TEST_PATH,  usecols=["region_id", "date"])
        train = pd.read_csv(TRAIN_PATH, usecols=["region_id", "date"])
        train_ends  = train.groupby("region_id")["date"].last()
        test_starts = test.groupby("region_id")["date"].first()

        def _parse_ymd(s: str):
            parts = str(s).strip().split("T")[0].split(" ")[0].split("-")
            return int(parts[0]), int(parts[1]), int(parts[2])

        region_gap_dict = {}
        for r in test["region_id"].unique():
            if r not in train_ends:
                region_gap_dict[str(r)] = 52.0
                continue
            try:
                te_y, te_m, te_d = _parse_ymd(train_ends[r])
                ts_y, ts_m, ts_d = _parse_ymd(test_starts[r])
                gap_wk = ((ts_y - te_y) * 365 + (ts_m - te_m) * 30 + (ts_d - te_d)) / 7.0
                region_gap_dict[str(r)] = float(gap_wk)
            except Exception:
                region_gap_dict[str(r)] = 52.0
        return region_gap_dict
    except Exception as e:
        log.warning(f"  region_gap_dict build failed ({e}), using empty dict")
        return {}


def _build_or_load_test_features(
    test, clim_dict, month_dict, sstat_dict, smo_dict, proxy_ridge,
    preproc_artifacts, region_test_months=None, force_rebuild=False,
):
    key = feature_cache_key()
    log.info(f"[cache] test key={key}")

    if USE_CACHE and not force_rebuild:
        cached = load_from_cache("test_features", key)
        if cached is not None:
            log.info("[cache] Test feature cache hit")
            return cached

    log.info("[cache] Cache miss — building test features ...")
    test_features = build_test_features(
        test, clim_dict, month_dict, sstat_dict, smo_dict, proxy_ridge,
        preproc_artifacts=preproc_artifacts,
        region_test_months=region_test_months,
    )
    if USE_CACHE:
        p = save_to_cache(test_features, "test_features", key)
        log.info(f"[cache] Cache saved → {p}")
    return test_features


def main(output_path=None, force_rebuild=False):
    if output_path is None:
        output_path = SUBMISSION_PATH
    output_path = Path(output_path)

    t0 = time.time()

    created = ensure_dirs()
    for d in created:
        log.info(f"  Created directory: {d}")

    log_startup_banner(log, "Drought Inference Pipeline")
    log.info(f"  output: {output_path}")

    # ── Load test.csv ──────────────────────────────────────────────────────────
    test = pd.read_csv(TEST_PATH, dtype={c: "float32" for c in METEO_COLS})
    test = test.sort_values(["region_id", "date"]).reset_index(drop=True)
    test["month"] = test["date"].str.split("-").str[1].astype("int8")
    log.info(f"  test: {test.shape}  regions: {test['region_id'].nunique()}")

    region_test_months = _compute_region_test_months(test)

    # ── Load artifacts ─────────────────────────────────────────────────────────
    clim                       = load_climatology()
    proxy_ridge                = load_proxy_ridge()
    lgbm_models, calibrator, _ = load_models()
    preproc_artifacts          = _load_preproc()

    clim_dict  = clim["clim_dict"]
    month_dict = clim["month_dict"]
    sstat_dict = clim["sstat_dict"]
    smo_dict   = clim["smo_dict"]

    # ── Build test features ────────────────────────────────────────────────────
    test_features = _build_or_load_test_features(
        test, clim_dict, month_dict, sstat_dict, smo_dict, proxy_ridge,
        preproc_artifacts=preproc_artifacts,
        region_test_months=region_test_months,
        force_rebuild=force_rebuild,
    )
    log.info(f"  Test features: {len(test_features)} regions")

    # ── Fallback score (train mean, used for unseen regions) ──────────────────
    try:
        fallback = float(
            pd.read_csv(TRAIN_PATH, usecols=["score"], dtype={"score": "float32"})
            ["score"].dropna().mean()
        )
    except Exception:
        fallback = 0.5
    log.info(f"  fallback score: {fallback:.4f}")

    # ── Predictions ───────────────────────────────────────────────────────────
    # Use Gap-Stratified LightGBM if available (trained short/long strata models).
    # Falls back to the main LightGBM + calibrator when not available.
    final_preds: dict[str, list[float]] = {}

    if USE_GAP_STRATIFIED:
        try:
            from gap_stratified import (
                load_stratified_models, predict_stratified, compute_region_strata,
            )
            region_gap_dict        = _load_region_gap_dict()
            short_models, long_models = load_stratified_models()
            if short_models and long_models:
                region_strata = compute_region_strata(region_gap_dict)
                final_preds   = predict_stratified(
                    test_features, short_models, long_models,
                    region_strata, region_gap_dict, fallback=fallback,
                )
                log.info("  Using Gap-Stratified LightGBM")
        except Exception as e:
            log.warning(f"  Gap-Stratified load failed ({e})")

    if not final_preds:
        final_preds = predict_for_test(
            test_features, lgbm_models, calibrator, fallback=fallback,
        )
        log.info("  Using main LightGBM")

    # ── Write submission ───────────────────────────────────────────────────────
    sample    = pd.read_csv(SAMPLE_PATH)
    pred_cols = [f"pred_week{i}" for i in range(1, N_PRED + 1)]
    rows = []
    for _, row in sample.iterrows():
        rid = row["region_id"]
        p   = final_preds.get(rid, [fallback] * N_PRED)
        rows.append({
            "region_id":  rid,
            "pred_week1": round(float(p[0]), 4),
            "pred_week2": round(float(p[1]), 4),
            "pred_week3": round(float(p[2]), 4),
            "pred_week4": round(float(p[3]), 4),
            "pred_week5": round(float(p[4]), 4),
        })

    sub = pd.DataFrame(rows)
    assert (sub[pred_cols] >= 0).all().all(), "Negative predictions found"
    assert (sub[pred_cols] <= 5).all().all(), "Predictions exceed 5"
    assert len(sub) == len(sample), f"Row count mismatch: {len(sub)} vs {len(sample)}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    log.info(f"Submission saved → {output_path}  ({sub.shape})")
    log.info(f"\n{sub[pred_cols].describe().round(4).to_string()}")
    log.info(f"Inference done in {time.time()-t0:.1f}s")


def _parse_args():
    p = argparse.ArgumentParser(description="drought inference pipeline")
    p.add_argument("-o", "--output",  type=str, default=None)
    p.add_argument("--force-rebuild", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(output_path=args.output, force_rebuild=args.force_rebuild)
