"""
Smoke test: extract 5 regions, run the full pipeline end-to-end, assert outputs.

Usage:
    cd code && python smoke_test.py
"""

import sys
import time
import tempfile
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# ── patch config before importing anything else ──────────────────────────────
import config
from config import TEST_PATH, TRAIN_PATH

N_SMOKE = 5
# Discover actual region IDs from the (small) test file instead of guessing
_test_ids = pd.read_csv(TEST_PATH, usecols=["region_id"])["region_id"].unique()
SMOKE_REGIONS = sorted(_test_ids, key=lambda x: int(x[1:]))[:N_SMOKE]
print(f"Smoke regions: {SMOKE_REGIONS}")

_orig_n = config.N_REGIONS
_orig_train_weeks = config.TRAIN_WEEKS_PER_REGION
_orig_test_weeks = config.TEST_WEEKS_PER_REGION

config.N_REGIONS = N_SMOKE
# keep TRAIN_WEEKS_PER_REGION and TEST_WEEKS_PER_REGION as-is;
# assertions in pipeline are overridden below

# ── fast LightGBM params for smoke ──────────────────────────────────────────
config.LGBM_PARAMS = {**config.LGBM_PARAMS,
                      "n_estimators": 50,
                      "learning_rate": 0.1,
                      "num_leaves": 15}
config.EARLY_STOPPING_ROUNDS = 5

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
    clip_predictions,
    EXTRA_FEATURE_COLS,
)
from model import train_single_horizon, save_all_models, load_all_models
from validate import fixed_holdout_split, evaluate_predictions, compute_macro_mae
from config import MODELS_DIR, HORIZONS, LAG_WINDOW


# ── helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(label, cond, detail=""):
    status = PASS if cond else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return cond


def extract_regions(src_path: Path, regions: list[str], tmp_path: Path, is_train: bool):
    """Stream-read only the rows for `regions` and write to tmp_path."""
    target_set = set(regions)
    dtype_map = {feat: "float32" for feat in [
        "wind","wind_min","wind_max","wind_range","humidity","tmp","tmp_range",
        "tmp_max","tmp_min","surf_tmp","surf_pre","dp_tmp","wb_tmp","prec"]}
    dtype_map["region_id"] = str
    dtype_map["date"] = str
    if is_train:
        dtype_map["score"] = "Int8"

    first = True
    print(f"  Extracting {len(regions)} regions from {src_path.name}...")
    for chunk in pd.read_csv(src_path, dtype=dtype_map, chunksize=300_000):
        sub = chunk[chunk["region_id"].isin(target_set)]
        if len(sub):
            sub.to_csv(tmp_path, mode="w" if first else "a",
                       header=first, index=False)
            first = False
    rows = sum(1 for _ in open(tmp_path)) - 1
    print(f"  → {rows:,} rows written to {tmp_path.name}")
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    all_ok = True

    print("\n" + "=" * 55)
    print("  SMOKE TEST — 5 regions, fast LightGBM params")
    print("=" * 55)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        smoke_train = tmpdir / "smoke_train.csv"
        smoke_test  = tmpdir / "smoke_test.csv"
        smoke_models = tmpdir / "models"
        smoke_models.mkdir()

        # Override model dir so smoke doesn't clobber real models
        config.MODELS_DIR = smoke_models

        # ── Stage 0: data extraction ─────────────────────────────────────
        print("\n[0] Extracting smoke data")
        train_rows = extract_regions(TRAIN_PATH, SMOKE_REGIONS, smoke_train, is_train=True)
        test_rows  = extract_regions(TEST_PATH,  SMOKE_REGIONS, smoke_test,  is_train=False)

        exp_train_daily = len(SMOKE_REGIONS) * 5480  # 5480 daily rows per region
        exp_test_daily  = len(SMOKE_REGIONS) * 91    # 91 daily rows per region
        all_ok &= check("Train rows (daily)",
                         train_rows == exp_train_daily,
                         f"{train_rows} == {exp_train_daily}")
        all_ok &= check("Test rows (daily)",
                         test_rows == exp_test_daily,
                         f"{test_rows} == {exp_test_daily}")

        # ── Stage 1: daily → weekly aggregation ─────────────────────────
        print("\n[1] Daily → weekly aggregation")
        train_weekly = load_and_aggregate_daily_to_weekly(smoke_train, is_train=True)
        test_weekly  = load_and_aggregate_daily_to_weekly(smoke_test,  is_train=False)

        exp_train_w = N_SMOKE * config.TRAIN_WEEKS_PER_REGION
        exp_test_w  = N_SMOKE * config.TEST_WEEKS_PER_REGION
        all_ok &= check("Train weekly rows",  len(train_weekly) == exp_train_w,
                         f"{len(train_weekly)} == {exp_train_w}")
        all_ok &= check("Test weekly rows",   len(test_weekly)  == exp_test_w,
                         f"{len(test_weekly)} == {exp_test_w}")
        all_ok &= check("Score range 0-5",
                         train_weekly["score"].between(0, 5).all())
        all_ok &= check("No NaN in meteo cols",
                         not train_weekly[[c for c in train_weekly.columns
                                           if c not in ("region_id","score","week_idx","month","day_of_year")]
                                         ].isnull().any().any())

        # ── Stage 2: lag feature construction ────────────────────────────
        print("\n[2] Lag feature construction")
        train_feat = construct_lag_features(train_weekly, lag_window=LAG_WINDOW, is_train=True)
        test_feat  = construct_lag_features(test_weekly,  lag_window=LAG_WINDOW, is_train=False)

        expected_train_feat = N_SMOKE * (config.TRAIN_WEEKS_PER_REGION - LAG_WINDOW - 5)
        all_ok &= check("Train feature rows",
                         len(train_feat) == expected_train_feat,
                         f"{len(train_feat)} == {expected_train_feat}")
        all_ok &= check("Test feature rows",
                         len(test_feat) == N_SMOKE,
                         f"{len(test_feat)} == {N_SMOKE}")
        lag_cols = get_feature_columns()
        all_ok &= check("Lag columns present",
                         all(c in train_feat.columns for c in lag_cols))
        all_ok &= check("No NaN in lag features",
                         not train_feat[lag_cols].isnull().any().any())
        all_ok &= check("Targets present",
                         all(f"target_w{h}" in train_feat.columns for h in HORIZONS))

        # ── Stage 3: feature engineering ─────────────────────────────────
        print("\n[3] Feature engineering")
        train_split, val_split = fixed_holdout_split(train_feat, n_val_weeks=26)

        max_anchor = config.TRAIN_WEEKS_PER_REGION - 6
        train_weekly_for_stats = train_weekly[
            train_weekly["week_idx"] <= train_split["week_idx"].max()
        ]
        region_stats = compute_region_stats(train_weekly_for_stats)

        train_split = add_temporal_features(train_split)
        train_split = add_region_features(train_split, region_stats)
        val_split   = add_temporal_features(val_split)
        val_split   = add_region_features(val_split, region_stats)
        test_feat   = add_temporal_features(test_feat)
        test_feat   = add_region_features(test_feat, region_stats)

        all_feat_cols = lag_cols + EXTRA_FEATURE_COLS
        save_feature_columns(all_feat_cols)

        all_ok &= check("Temporal features added",
                         all(c in train_split.columns
                             for c in ["month_sin","month_cos","doy_sin","doy_cos"]))
        all_ok &= check("Region features added",
                         all(c in train_split.columns
                             for c in ["region_id_int","region_score_mean","region_zero_rate"]))
        all_ok &= check("No NaN after feature eng",
                         not train_split[all_feat_cols].isnull().any().any())

        # ── Stage 4: training ─────────────────────────────────────────────
        print("\n[4] Model training (50 trees, fast)")
        X_train = train_split[all_feat_cols]
        X_val   = val_split[all_feat_cols]

        models = {}
        for h in HORIZONS:
            models[h] = train_single_horizon(
                X_train, train_split[f"target_w{h}"],
                X_val,   val_split[f"target_w{h}"],
                horizon=h,
            )

        all_ok &= check("All 5 models trained", len(models) == 5)
        all_ok &= check("Models are LGBMRegressor",
                         all(isinstance(m, lgb.LGBMRegressor) for m in models.values()))

        # ── Stage 5: validation metric ────────────────────────────────────
        print("\n[5] Validation metric")
        val_preds = {h: models[h].predict(X_val) for h in HORIZONS}
        macro_mae, horizon_maes = evaluate_predictions(val_split, val_preds)

        for h in HORIZONS:
            all_ok &= check(f"Horizon {h} MAE finite",
                             np.isfinite(horizon_maes[h]), f"{horizon_maes[h]:.4f}")
        all_ok &= check("Macro MAE < 5.0 (sanity bound)", macro_mae < 5.0,
                         f"{macro_mae:.4f}")
        print(f"  Macro MAE = {macro_mae:.4f}")

        # ── Stage 6: save + reload models ────────────────────────────────
        print("\n[6] Model save / reload")
        save_all_models(models, models_dir=smoke_models)
        loaded = load_all_models(models_dir=smoke_models)

        for h in HORIZONS:
            orig = models[h].predict(X_val[:10])
            reloaded = loaded[h].predict(X_val[:10])
            all_ok &= check(f"Horizon {h} reload identical",
                             np.allclose(orig, reloaded, atol=1e-4))

        # ── Stage 7: inference ────────────────────────────────────────────
        print("\n[7] Inference on test data")
        X_test = test_feat[all_feat_cols]
        raw_preds = {h: loaded[h].predict(X_test) for h in HORIZONS}

        submission = pd.DataFrame({"region_id": test_feat["region_id"].values})
        for h in HORIZONS:
            submission[f"pred_week{h}"] = clip_predictions(raw_preds[h])

        all_ok &= check("Submission shape",
                         submission.shape == (N_SMOKE, 6),
                         str(submission.shape))
        all_ok &= check("All regions present",
                         set(submission["region_id"]) == set(SMOKE_REGIONS))
        pred_cols = [f"pred_week{h}" for h in HORIZONS]
        all_ok &= check("Predictions in [0, 5]",
                         (submission[pred_cols] >= 0).all().all() and
                         (submission[pred_cols] <= 5).all().all())
        all_ok &= check("No NaN in submission",
                         not submission[pred_cols].isnull().any().any())

        print("\n  Submission preview:")
        print(submission.to_string(index=False))

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 55)
    if all_ok:
        print(f"  ALL CHECKS PASSED  ({elapsed:.1f}s)")
    else:
        print(f"  SOME CHECKS FAILED ({elapsed:.1f}s)")
        sys.exit(1)
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
