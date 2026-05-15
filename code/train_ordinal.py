"""
Gate 1 experiment — train.py with the L1 regression head replaced by a
6-class multiclass head over {0, 1, 2, 3, 4, 5}; predictions are the
expected value E[Y|X] = sum_{k=0..5} k * P(y=k|x).

Only Stage 4 (model training) and Stage 5 (evaluation) differ from train.py.
Stages 0-3 (preprocessing, daily->weekly, lag features, feature engineering)
are run identically so the comparison isolates the head.

What we are testing
-------------------
Claim: catastrophic underprediction of extreme rows (true score 5 -> mean
prediction 1.39 with L1) is driven by L1's collapse-to-median behavior on a
60%-zero target. A multiclass head with expected-value output has sharper
gradients for rare classes and parameterizes the full conditional
distribution, so should lift score>=3 predictions toward truth.

Test: train the same LightGBM with the same features and sample weights;
swap the head from `regression_l1` to `multiclass` (num_class=6); predict
expected value; compute val MAE stratified by true score. Compare to the
baseline `val_preds.npz` produced by train.py.

Outputs
-------
- code/models/lgbm_h{1..5}_ordinal.txt  -- 5 multiclass boosters
- code/models/val_preds_ordinal.npz    -- val predictions + probabilities

Usage
-----
    python train_ordinal.py
"""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

from logging_setup import get_logger, current_log_path
log = get_logger("train_ordinal", label="train_ordinal")

from config import (
    TRAIN_PATH, TEST_PATH, LAG_WINDOW, MODELS_DIR, HORIZONS,
    USE_EXTRA_FEATURES,
    USE_PREPROCESSING, USE_SCORE_LAG_FEATURES, USE_REGION_CLUSTER_FEATURES,
    USE_RANK_NORMALIZATION, RANK_NORMALIZE_FEATURES, N_REGION_CLUSTERS,
    SAMPLE_WEIGHT_ALPHA, SAMPLE_WEIGHT_BETA,
    LGBM_PARAMS, EARLY_STOPPING_ROUNDS,
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
from validate import fixed_holdout_split

NUM_CLASSES = 6  # scores 0..5
CLASS_VALUES = np.arange(NUM_CLASSES, dtype=np.float32)  # [0,1,2,3,4,5]


# ---------------------------------------------------------------------------
# Multiclass head: training + expected-value prediction
# ---------------------------------------------------------------------------

def _make_sample_weight(y_train) -> np.ndarray:
    """Same severe-row upweighting as the L1 baseline."""
    y = np.asarray(y_train, dtype=np.float32)
    return (
        1.0
        + SAMPLE_WEIGHT_ALPHA * (y > 0).astype(np.float32)
        + SAMPLE_WEIGHT_BETA * (y >= 3).astype(np.float32)
    )


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax; z shape (n_samples, n_classes)."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _expected_value_eval_factory():
    """
    Custom LightGBM eval metric for multiclass training: compute expected
    value E[Y|X] from softmax probabilities and return MAE vs y_true.

    LightGBM's sklearn-API feval signature in multiclass mode:
        y_pred shape is (n_samples * n_classes,) flattened by class (column-major).
    We reshape to (n_samples, n_classes) accordingly.

    Used for early stopping so we pick the iteration that minimizes the
    metric we actually care about, not multi_logloss.
    """
    def feval(y_true, y_pred):
        n_samples = len(y_true)
        # LightGBM flattens by class: rows = classes, cols = samples
        z = y_pred.reshape(NUM_CLASSES, n_samples).T  # (n_samples, n_classes)
        proba = _softmax_rows(z)
        expected = proba @ CLASS_VALUES
        mae = float(np.mean(np.abs(expected - y_true)))
        return "expected_value_mae", mae, False  # is_higher_better=False
    return feval


class _TqdmCallback:
    """LightGBM callback that drives a per-iteration tqdm bar."""
    order = 10
    before_iteration = False

    def __init__(self, total: int, desc: str):
        self.pbar = tqdm(total=total, desc=desc, unit="iter", leave=False)

    def __call__(self, env):
        self.pbar.update(1)
        if env.evaluation_result_list:
            _, eval_name, result = env.evaluation_result_list[-1][:3]
            self.pbar.set_postfix_str(f"{eval_name}={result:.4f}")

    def close(self):
        self.pbar.close()


def train_single_horizon_ordinal(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    horizon: int,
    sample_weight: np.ndarray | None = None,
) -> lgb.LGBMClassifier:
    """Train one multiclass LightGBM for a single horizon, ev_mae early-stop."""
    params = {**LGBM_PARAMS}
    params["objective"] = "multiclass"
    params["num_class"] = NUM_CLASSES
    # Remove regression-specific defaults if present
    params.pop("metric", None)

    n_estimators = params["n_estimators"]
    pbar_cb = _TqdmCallback(total=n_estimators, desc=f"Horizon {horizon} ord")
    model = lgb.LGBMClassifier(**params)
    try:
        model.fit(
            X_train,
            y_train.astype(np.int32),
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val.astype(np.int32))],
            eval_metric=_expected_value_eval_factory(),
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False, first_metric_only=True),
                pbar_cb,
            ],
        )
    finally:
        pbar_cb.close()

    best = model.best_iteration_
    val_metric = model.best_score_.get("valid_0", {}).get("expected_value_mae")
    extra = f", val_ev_mae={val_metric:.4f}" if isinstance(val_metric, (int, float)) else ""
    log.info(f"Horizon {horizon} (ordinal): best iteration = {best}{extra}")
    return model


def predict_expected_value(model: lgb.LGBMClassifier, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (expected_value, full_proba) for a multiclass classifier."""
    proba = model.predict_proba(X, num_iteration=model.best_iteration_)  # (n, 6)
    expected = proba @ CLASS_VALUES
    return np.clip(expected, 0.0, 5.0), proba.astype(np.float32)


# ---------------------------------------------------------------------------
# Stages 0-3 (identical to train.py main): preprocessing + features
# ---------------------------------------------------------------------------

def _fit_preprocessing_artifacts():
    from preprocessing import (
        compute_winsor_bounds, compute_imputation_table, compute_rank_quantiles,
        save_preprocessing_artifacts,
    )

    log.info("[preproc] Reading train daily for artifact fitting...")
    chunks = []
    for chunk in pd.read_csv(TRAIN_PATH, dtype={"region_id": str, "date": str},
                             chunksize=500_000):
        chunks.append(chunk)
    train_daily = pd.concat(chunks, ignore_index=True)
    log.info(f"[preproc] train_daily: {len(train_daily):,} rows")

    bounds = compute_winsor_bounds(train_daily, q_lo=0.01, q_hi=0.99)
    imputation_table = compute_imputation_table(train_daily)

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
    del train_daily
    return bounds, log_features, imputation_table, quantile_table


def _log_run_config():
    log.info("=" * 70)
    log.info("DMFP GATE-1 (ordinal head) | objective=multiclass num_class=6")
    log.info(f"  USE_PREPROCESSING={USE_PREPROCESSING}  USE_EXTRA_FEATURES={USE_EXTRA_FEATURES}")
    log.info(f"  USE_SCORE_LAG_FEATURES={USE_SCORE_LAG_FEATURES}  USE_REGION_CLUSTER_FEATURES={USE_REGION_CLUSTER_FEATURES}")
    log.info(f"  USE_RANK_NORMALIZATION={USE_RANK_NORMALIZATION}")
    log.info(f"  SAMPLE_WEIGHT_ALPHA={SAMPLE_WEIGHT_ALPHA}  SAMPLE_WEIGHT_BETA={SAMPLE_WEIGHT_BETA}")
    log.info(f"  log file: {current_log_path()}")
    log.info("=" * 70)


def main():
    t0 = time.time()
    _log_run_config()

    # ------------------------------------------------------------------
    # Stage 0: preprocessing artifacts
    # ------------------------------------------------------------------
    preproc_artifacts = None
    if USE_PREPROCESSING:
        log.info("[0/5] Fitting daily-level preprocessing artifacts...")
        t_stage = time.time()
        preproc_artifacts = _fit_preprocessing_artifacts()
        log.info(f"[0/5] done in {(time.time() - t_stage) / 60:.2f} min")

    # ------------------------------------------------------------------
    # Stage 1: daily -> weekly
    # ------------------------------------------------------------------
    log.info("[1/5] Loading and aggregating train daily -> weekly...")
    t_stage = time.time()
    train_weekly = load_and_aggregate_daily_to_weekly(
        TRAIN_PATH, is_train=True, preproc_artifacts=preproc_artifacts,
    )
    log.info(f"[1/5] train_weekly: {train_weekly.shape}, done in {(time.time() - t_stage) / 60:.2f} min")

    # ------------------------------------------------------------------
    # Stage 2: lag features
    # ------------------------------------------------------------------
    log.info("[2/5] Constructing lag features...")
    t_stage = time.time()
    train_features = construct_lag_features(train_weekly, lag_window=LAG_WINDOW, is_train=True)
    log.info(f"[2/5] train_features: {train_features.shape}, done in {(time.time() - t_stage) / 60:.2f} min")

    # ------------------------------------------------------------------
    # Stage 3: feature engineering (temporal + region + extras)
    # ------------------------------------------------------------------
    log.info("[3/5] Adding temporal and region features...")
    t_stage = time.time()
    train_split, val_split = fixed_holdout_split(train_features)
    log.info(f"  fixed_holdout_split -> train: {len(train_split):,}  val: {len(val_split):,}")

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
        log.info("  Adding lagged-score features...")
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

    X_train = train_split[all_feature_cols]
    X_val = val_split[all_feature_cols]
    log.info(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")
    log.info(f"[3/5] done in {(time.time() - t_stage) / 60:.2f} min")

    # ------------------------------------------------------------------
    # Stage 4: multiclass training (the only architectural difference)
    # ------------------------------------------------------------------
    log.info(f"[4/5] Training {len(HORIZONS)} multiclass LightGBM models...")
    t_stage = time.time()
    models = {}
    for h in HORIZONS:
        log.info(f"  -> starting horizon {h}")
        t_h = time.time()
        y_train = train_split[f"target_w{h}"]
        y_val_h = val_split[f"target_w{h}"]
        sw = _make_sample_weight(y_train)
        models[h] = train_single_horizon_ordinal(
            X_train, y_train, X_val, y_val_h, h, sample_weight=sw,
        )
        # save immediately so a crash later doesn't lose work
        models[h].booster_.save_model(str(MODELS_DIR / f"lgbm_h{h}_ordinal.txt"))
        log.info(f"  -> horizon {h} done in {(time.time() - t_h) / 60:.2f} min, saved.")
    log.info(f"[4/5] all horizons trained in {(time.time() - t_stage) / 60:.2f} min")

    # ------------------------------------------------------------------
    # Stage 5: evaluate expected-value MAE + breakdown by true score
    # ------------------------------------------------------------------
    log.info("[5/5] Evaluating on validation set...")
    ev_preds = {}
    probas = {}
    for h in HORIZONS:
        ev, pr = predict_expected_value(models[h], X_val)
        ev_preds[h] = ev
        probas[h] = pr

    # Per-horizon MAE + macro
    horizon_maes = {
        h: float(np.mean(np.abs(ev_preds[h] - val_split[f"target_w{h}"].values)))
        for h in HORIZONS
    }
    macro_mae = float(np.mean(list(horizon_maes.values())))

    log.info("=" * 60)
    log.info("ORDINAL HEAD VALIDATION (Gate 1)")
    log.info("=" * 60)
    log.info(f"  macro val MAE = {macro_mae:.4f}  "
             f"per-horizon = {{{', '.join(f'w{h}={horizon_maes[h]:.4f}' for h in HORIZONS)}}}")

    # Per-true-score MAE breakdown for horizon 1 (most directly comparable
    # to the diagnose.py output remembered in project memory).
    log.info("")
    log.info("Horizon 1 — MAE stratified by true score:")
    log.info(f"  {'true':>5} {'n':>10} {'pct':>6} {'mae':>8} {'mean_pred':>10}")
    y_true_h1 = val_split["target_w1"].values
    pred_h1 = ev_preds[1]
    total_n = len(y_true_h1)
    for s in range(NUM_CLASSES):
        mask = y_true_h1 == s
        n = int(mask.sum())
        if n == 0:
            continue
        mae_s = float(np.mean(np.abs(pred_h1[mask] - s)))
        mean_pred_s = float(np.mean(pred_h1[mask]))
        pct = 100.0 * n / total_n
        log.info(f"  {s:>5} {n:>10,} {pct:>5.1f}% {mae_s:>8.4f} {mean_pred_s:>10.4f}")
    log.info("=" * 60)

    # Compare to baseline val_preds.npz (which train.py wrote) if available
    baseline_path = MODELS_DIR / "val_preds.npz"
    if baseline_path.exists():
        log.info("")
        log.info("Direct head-to-head vs L1 baseline (val_preds.npz):")
        baseline = np.load(baseline_path, allow_pickle=True)
        b_preds = baseline["preds"]    # (n_val, 5)
        b_truth = baseline["truth"]    # (n_val, 5)
        if b_preds.shape[0] == len(val_split):
            b_macro = float(np.mean(np.abs(b_preds - b_truth)))
            log.info(f"  L1 baseline   macro MAE: {b_macro:.4f}")
            log.info(f"  ordinal head  macro MAE: {macro_mae:.4f}")
            log.info(f"  delta (ord - L1):        {macro_mae - b_macro:+.4f}")
            # Per-horizon
            for i, h in enumerate(HORIZONS):
                b_h = float(np.mean(np.abs(b_preds[:, i] - b_truth[:, i])))
                log.info(f"    w{h}: L1={b_h:.4f}  ord={horizon_maes[h]:.4f}  delta={horizon_maes[h] - b_h:+.4f}")
            # Score-stratified comparison on horizon 1
            log.info("")
            log.info("Horizon 1 — score=5 row test (the key claim):")
            b_h1_truth = b_truth[:, 0]
            b_h1_pred = b_preds[:, 0]
            for s in (3, 4, 5):
                m = b_h1_truth == s
                if not m.any():
                    continue
                b_mae = float(np.mean(np.abs(b_h1_pred[m] - s)))
                b_mean = float(np.mean(b_h1_pred[m]))
                ord_mae = float(np.mean(np.abs(pred_h1[m] - s)))
                ord_mean = float(np.mean(pred_h1[m]))
                log.info(
                    f"  true={s}: L1 mae={b_mae:.3f} mean_pred={b_mean:.3f}  | "
                    f"ord mae={ord_mae:.3f} mean_pred={ord_mean:.3f}"
                )
        else:
            log.warning(
                f"Baseline val_preds row count ({b_preds.shape[0]}) does not "
                f"match current val_split ({len(val_split)}); skipping head-to-head."
            )
    else:
        log.info("No baseline val_preds.npz found — run train.py first for direct comparison.")

    # Save artifacts
    preds_arr = np.stack([ev_preds[h] for h in HORIZONS], axis=1).astype(np.float32)
    proba_arr = np.stack([probas[h] for h in HORIZONS], axis=1).astype(np.float32)  # (n, 5, 6)
    truth_arr = np.stack(
        [val_split[f"target_w{h}"].values for h in HORIZONS], axis=1
    ).astype(np.float32)
    np.savez(
        MODELS_DIR / "val_preds_ordinal.npz",
        preds=preds_arr,
        probas=proba_arr,
        truth=truth_arr,
        region_ids=val_split["region_id"].values.astype(str),
        week_idx=val_split["week_idx"].values.astype(np.int32),
        horizons=np.array(HORIZONS, dtype=np.int32),
    )
    log.info(f"Saved val_preds_ordinal.npz to {MODELS_DIR}")

    elapsed = time.time() - t0
    log.info(f"Gate 1 (ordinal) complete in {elapsed / 60:.2f} min.")


if __name__ == "__main__":
    main()
