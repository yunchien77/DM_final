import os
import time
import pickle
import contextlib
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.isotonic import IsotonicRegression


@contextlib.contextmanager
def _suppress_fd_stderr():
    """Redirect file-descriptor-level stderr to suppress LightGBM GPU OpenCL
    kernel compilation warnings that bypass Python's logging system."""
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
    except Exception:
        yield


from config import (
    LGBM_PARAMS, EARLY_STOPPING_ROUNDS,
    N_FOLDS, N_PH_FOLDS, SEED, N_PRED,
    USE_SAMPLE_WEIGHT, USE_CALIBRATION,
    USE_PER_HORIZON_MODELS,
    PURGE_GAP_DAYS,
    MODELS_PKL_PATH, CALIBRATOR_PATH, EVAL_REPORT_PATH,
    MODELS_PH_DIR,
    USE_CALENDAR_MATCHED_VAL, CALENDAR_BANDWIDTH, CALENDAR_MATCHED_MIN_SAMPLES,
    USE_GPU_LGBM, LGBM_DEVICE, N_GPUS,
    PARALLEL_FW_TRAINING,
    USE_KAGGLE_PROXY_VAL,
)
from temporal_cv import TemporalKFold
from data_pipeline import make_sample_weights
from evaluation import compute_metrics, print_metrics, per_region_mae, fold_summary, save_eval_report
from logging_setup import get_logger

log = get_logger("model")


# ─────────────────────────────────────────────────────────────────────────────
# Calendar-Matched Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _month_dist(m1: int, m2: int) -> int:
    """Circular month distance (0 = same month, 6 = opposite)."""
    d = abs(int(m1) - int(m2))
    return min(d, 12 - d)


def _calendar_matched_val(
    va_pos:             np.ndarray,
    groups_fw:          np.ndarray,
    month_keys_fw:      np.ndarray,
    region_test_months: dict[str, int],
    bandwidth:          int = CALENDAR_BANDWIDTH,
    min_samples:        int = CALENDAR_MATCHED_MIN_SAMPLES,
) -> np.ndarray:
    """Filter validation indices to samples that fall within the test season.

    For each validation sample, compute the circular distance between its month
    and the region's test start month.  Keep samples within `bandwidth` months.
    Fall back to the full validation set if too few samples remain.
    """
    val_months = month_keys_fw[va_pos]
    val_groups = groups_fw[va_pos]

    mask = np.array([
        _month_dist(
            val_months[i],
            region_test_months.get(str(val_groups[i]), val_months[i])
        ) <= bandwidth
        for i in range(len(va_pos))
    ], dtype=bool)

    if mask.sum() >= min_samples:
        return va_pos[mask]
    return va_pos


def _extract_kaggle_val_mask(
    grp_fw:             np.ndarray,
    tk_fw:              np.ndarray,
    mo_fw:              np.ndarray,
    region_test_months: dict[str, int],
    bandwidth:          int = CALENDAR_BANDWIDTH,
) -> np.ndarray:
    """Build a holdout mask that mirrors the Kaggle evaluation split.

    For each region in the fw-slice, select the **most recent in-season
    anchor** (circular month distance ≤ bandwidth) as a fixed early-stopping
    proxy.  The mask has exactly one True per region that has an in-season
    sample; regions without any in-season data are skipped.

    The holdout samples are excluded from the walk-forward CV entirely and
    used as a fixed eval set for early stopping in every fold.  This prevents
    the model from overfitting to the CV val distribution (which covers the
    full training timeline) and instead stops when test-season performance
    peaks — replicating how Kaggle scores the submission.
    """
    kv_mask = np.zeros(len(grp_fw), dtype=bool)
    if mo_fw is None or region_test_months is None:
        return kv_mask

    for region in np.unique(grp_fw):
        r_mask   = grp_fw == region
        test_m   = region_test_months.get(str(region), 0)
        if test_m == 0:
            continue
        r_months = mo_fw[r_mask].astype(int)
        r_times  = tk_fw[r_mask]
        r_idx    = np.where(r_mask)[0]

        dist      = np.abs(r_months - test_m)
        dist      = np.minimum(dist, 12 - dist)
        in_season = dist <= bandwidth

        if not in_season.any():
            continue

        # Pick the most recent in-season sample as holdout
        in_idx = r_idx[in_season]
        best   = in_idx[np.argmax(r_times[in_season])]
        kv_mask[best] = True

    return kv_mask


# ─────────────────────────────────────────────────────────────────────────────
# Single LightGBM fit
# ─────────────────────────────────────────────────────────────────────────────

def _fit_lgbm(
    X_tr:    np.ndarray,
    y_tr:    np.ndarray,
    X_va:    np.ndarray,
    y_va:    np.ndarray,
    X_es:    np.ndarray | None = None,  # calendar-matched early stopping set
    y_es:    np.ndarray | None = None,
    sw_tr:   np.ndarray | None = None,
    label:   str = "",
    params_override: dict | None = None,
    device_id: int | None = None,
) -> lgb.LGBMRegressor:
    """Train a single LGBMRegressor with optional GPU assignment and calendar-matched early stopping."""
    params = dict(LGBM_PARAMS)
    if params_override:
        params.update(params_override)

    if USE_GPU_LGBM and device_id is not None:
        params["device_type"]   = LGBM_DEVICE
        params["gpu_device_id"] = device_id % max(1, N_GPUS)
    elif USE_GPU_LGBM:
        params["device_type"] = LGBM_DEVICE

    es_set = (X_es, y_es) if X_es is not None else (X_va, y_va)

    def _do_fit(p):
        m = lgb.LGBMRegressor(**p)
        using_gpu = p.get("device_type", "cpu") != "cpu"
        ctx = _suppress_fd_stderr() if using_gpu else contextlib.nullcontext()
        with ctx:
            m.fit(
                X_tr, y_tr,
                sample_weight=sw_tr,
                eval_set=[es_set],
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(period=500),
                ],
            )
        return m

    try:
        model = _do_fit(params)
    except lgb.basic.LightGBMError as e:
        if any(k in str(e).upper() for k in ("CUDA", "GPU", "DEVICE")):
            log.warning(f"  LightGBM GPU unavailable ({e}), falling back to CPU")
            params.pop("device_type", None)
            params.pop("device", None)
            params.pop("gpu_device_id", None)
            model = _do_fit(params)
        else:
            raise

    if label:
        log.info(f"  [{label}] best_iter={model.best_iteration_}  "
                 f"es_mae≈{model.best_score_.get('valid_0', {}).get('l1', 'N/A')}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Parallel fw training worker
# ─────────────────────────────────────────────────────────────────────────────

def _train_one_fw(args: tuple) -> tuple:
    """ProcessPoolExecutor worker: train all folds for one forecast horizon.

    Returns (fw, fw_models, oof_slice_dict, fold_maes).
    oof_slice_dict maps original row indices to OOF predictions.

    args[18] = kv_mask_fw (bool array | None) — Kaggle proxy val holdout mask.
    When provided, those samples are excluded from walk-forward CV and used as
    a fixed early-stopping set that approximates the Kaggle evaluation split.
    """
    (fw, X_fw, y_fw, sw_fw, tk_fw, grp_fw, mo_fw,
     region_test_months, use_cal, gpu_id,
     n_ph_folds, purge_gap_days, early_stopping_rounds,
     lgbm_params, use_gpu_lgbm, lgbm_device, n_gpus,
     orig_idx_fw, kv_mask_fw) = args

    import numpy as np
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error
    from temporal_cv import TemporalKFold

    def _month_dist_local(m1, m2):
        d = abs(int(m1) - int(m2))
        return min(d, 12 - d)

    # ── Extract Kaggle proxy val holdout ──────────────────────────────────────
    X_kv, y_kv = None, None
    if kv_mask_fw is not None and kv_mask_fw.sum() >= 5:
        X_kv = X_fw[kv_mask_fw]
        y_kv = y_fw[kv_mask_fw]
        keep_mask = ~kv_mask_fw
    else:
        keep_mask = np.ones(len(X_fw), dtype=bool)

    # CV uses only the non-holdout slice
    X_cv      = X_fw[keep_mask]
    y_cv      = y_fw[keep_mask]
    tk_cv     = tk_fw[keep_mask]
    sw_cv     = sw_fw[keep_mask] if sw_fw is not None else None
    grp_cv    = grp_fw[keep_mask]
    mo_cv     = mo_fw[keep_mask] if mo_fw is not None else None
    orig_cv   = orig_idx_fw[keep_mask]

    tkf = TemporalKFold(n_splits=n_ph_folds, gap_days=purge_gap_days)
    fw_models, fold_maes = [], []
    oof_slice = {}

    for fold_idx, (tr_pos, va_pos) in enumerate(tkf.split(tk_cv)):
        if len(tr_pos) < 20 or len(va_pos) < 10:
            continue

        X_tr, y_tr = X_cv[tr_pos], y_cv[tr_pos]
        X_va, y_va = X_cv[va_pos], y_cv[va_pos]
        sw_tr = sw_cv[tr_pos] if sw_cv is not None else None

        # ── Early-stopping set priority ───────────────────────────────────────
        # 1. Kaggle proxy val (best — mirrors Kaggle evaluation split)
        # 2. Calendar-matched val subset (fallback when no proxy val)
        # 3. Full validation fold (last resort)
        if X_kv is not None:
            es_set = (X_kv, y_kv)
        elif use_cal and mo_cv is not None and region_test_months is not None:
            mask = np.array([
                _month_dist_local(mo_cv[va_pos[i]],
                                  region_test_months.get(str(grp_cv[va_pos[i]]),
                                                         mo_cv[va_pos[i]])) <= 2
                for i in range(len(va_pos))
            ], dtype=bool)
            if mask.sum() >= 20:
                es_set = (X_cv[va_pos[mask]], y_cv[va_pos[mask]])
            else:
                es_set = (X_va, y_va)
        else:
            es_set = (X_va, y_va)

        params = dict(lgbm_params)
        if use_gpu_lgbm:
            params["device_type"]   = lgbm_device
            params["gpu_device_id"] = gpu_id % max(1, n_gpus)

        def _do_fit(p):
            m = lgb.LGBMRegressor(**p)
            m.fit(
                X_tr, y_tr,
                sample_weight=sw_tr,
                eval_set=[es_set],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            return m

        try:
            model = _do_fit(params)
        except lgb.basic.LightGBMError as e:
            if any(k in str(e).upper() for k in ("CUDA", "GPU", "DEVICE")):
                params.pop("device_type", None)
                params.pop("device", None)
                params.pop("gpu_device_id", None)
                model = _do_fit(params)
            else:
                raise

        fw_models.append(model)
        val_pred = model.predict(X_va).astype(np.float32)
        for i, pos in enumerate(va_pos):
            oof_slice[int(orig_cv[pos])] = float(val_pred[i])
        fold_maes.append(float(mean_absolute_error(y_va, np.clip(val_pred, 0, 5))))

    if not fw_models:
        # Fallback: single 80/20 split on the CV slice when no fold had enough data
        split = int(len(X_cv) * 0.8)
        if split >= 20 and len(X_cv) - split >= 10:
            params = dict(lgbm_params)
            if use_gpu_lgbm:
                params["device_type"]   = lgbm_device
                params["gpu_device_id"] = gpu_id % max(1, n_gpus)
            es_fb = (X_kv, y_kv) if X_kv is not None else (X_cv[split:], y_cv[split:])
            try:
                m = lgb.LGBMRegressor(**params)
                m.fit(X_cv[:split], y_cv[:split],
                      sample_weight=sw_cv[:split] if sw_cv is not None else None,
                      eval_set=[es_fb],
                      callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                                 lgb.log_evaluation(period=-1)])
                fw_models = [m]
            except Exception:
                pass
            preds = fw_models[0].predict(X_cv[split:]).astype(np.float32) if fw_models else np.array([])
            for i, pos in enumerate(range(split, len(X_cv))):
                if i < len(preds):
                    oof_slice[int(orig_cv[pos])] = float(preds[i])

    return fw, fw_models, oof_slice, fold_maes


# ─────────────────────────────────────────────────────────────────────────────
# Per-horizon training
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm_per_horizon(
    X:                   np.ndarray,
    y:                   np.ndarray,
    groups:              np.ndarray,
    time_keys:           np.ndarray,
    fw_keys:             np.ndarray,
    month_keys:          np.ndarray | None = None,
    adversarial_weights: np.ndarray | None = None,
    region_test_months:  dict[str, int] | None = None,
    n_ph_folds:          int | None = None,
) -> tuple[dict[int, list[lgb.LGBMRegressor]], dict, np.ndarray]:
    """Train one LightGBM model per forecast horizon using walk-forward CV."""
    use_cal  = USE_CALENDAR_MATCHED_VAL and month_keys is not None and region_test_months is not None
    _n_folds = n_ph_folds if n_ph_folds is not None else N_PH_FOLDS

    log.info(f"Per-horizon training (fw=1..{N_PRED}, folds={_n_folds}, gap={PURGE_GAP_DAYS}d, "
             f"cal_match={use_cal}, features={X.shape[1]}, "
             f"parallel={PARALLEL_FW_TRAINING}, "
             f"adversarial={'on' if adversarial_weights is not None else 'off'}) ...")

    # ── Combine severity × adversarial weights ────────────────────────────────
    if USE_SAMPLE_WEIGHT:
        severity_w = make_sample_weights(y)
        if adversarial_weights is not None:
            combined = severity_w * adversarial_weights.astype(np.float32)
            sample_weight_all = (combined / combined.mean()).astype(np.float32)
        else:
            sample_weight_all = severity_w
    else:
        sample_weight_all = None

    models: dict[int, list] = {}
    fw_reports = {}
    oof_preds = np.full(len(X), np.nan, dtype=np.float32)

    # ── Pre-split data per fw (workers only receive their own slice) ──────────
    fw_args_list = []
    for fw in range(1, N_PRED + 1):
        fw_mask  = fw_keys == fw
        X_fw     = X[fw_mask]
        y_fw     = y[fw_mask]
        tk_fw    = time_keys[fw_mask]
        sw_fw    = sample_weight_all[fw_mask] if sample_weight_all is not None else None
        grp_fw   = groups[fw_mask]
        mo_fw    = month_keys[fw_mask] if month_keys is not None else None
        orig_idx = np.where(fw_mask)[0]
        if len(X_fw) < 50:
            log.warning(f"  fw={fw}: too few samples ({len(X_fw)}), skipping")
            continue

        # ── Kaggle proxy val holdout mask ─────────────────────────────────────
        kv_mask_fw = None
        if USE_KAGGLE_PROXY_VAL and mo_fw is not None and region_test_months is not None:
            kv_mask_fw = _extract_kaggle_val_mask(
                grp_fw, tk_fw, mo_fw, region_test_months, bandwidth=CALENDAR_BANDWIDTH
            )
            log.info(f"  fw={fw}: Kaggle proxy val holdout={kv_mask_fw.sum()} samples "
                     f"({len(np.unique(grp_fw))} regions)")

        fw_args_list.append((
            fw, X_fw, y_fw, sw_fw, tk_fw, grp_fw, mo_fw,
            region_test_months, use_cal, (fw - 1) % max(1, N_GPUS),
            _n_folds, PURGE_GAP_DAYS, EARLY_STOPPING_ROUNDS,
            dict(LGBM_PARAMS), USE_GPU_LGBM, LGBM_DEVICE, N_GPUS,
            orig_idx, kv_mask_fw,
        ))

    if PARALLEL_FW_TRAINING and len(fw_args_list) > 1:
        # ── Parallel mode: fw run in parallel processes ───────────────────────
        import concurrent.futures, multiprocessing
        n_workers = min(len(fw_args_list), N_GPUS if USE_GPU_LGBM else N_PRED)
        log.info(f"  Parallel training: {len(fw_args_list)} fw on {n_workers} workers ...")
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx
        ) as executor:
            futures = {executor.submit(_train_one_fw, args): args[0]
                       for args in fw_args_list}
            pbar = tqdm(concurrent.futures.as_completed(futures),
                        total=len(futures), desc="Parallel fw", unit="fw", ncols=80)
            for future in pbar:
                fw_done = futures[future]
                try:
                    fw_r, fw_models, oof_slice, fold_maes = future.result()
                    models[fw_r] = fw_models
                    for idx, val in oof_slice.items():
                        oof_preds[idx] = val
                    fw_mask_r = fw_keys == fw_r
                    fw_val_mask = fw_mask_r & ~np.isnan(oof_preds)
                    if fw_val_mask.sum() > 0:
                        fw_metrics = compute_metrics(y[fw_val_mask], oof_preds[fw_val_mask],
                                                     label=f"fw={fw_r} OOF")
                        print_metrics(fw_metrics)
                        fw_metrics["fold_maes"] = fold_maes
                        fw_reports[fw_r] = fw_metrics
                    else:
                        fw_reports[fw_r] = {"mae": float("nan"), "fold_maes": fold_maes}
                    pbar.set_postfix({"fw": fw_done,
                                      "MAE": f"{fw_reports[fw_done].get('mae', float('nan')):.4f}"})
                    log.info(f"  fw={fw_done} done: {len(fw_models)} models, "
                             f"fold MAEs={[f'{m:.4f}' for m in fold_maes]}")
                except Exception as e:
                    log.error(f"  fw={fw_done} training failed: {e}")
                    fw_reports[fw_done] = {"mae": float("nan"), "fold_maes": []}
    else:
        # ── Sequential mode (PARALLEL_FW_TRAINING=False, for debugging) ──────
        log.info(f"  Sequential training: {len(fw_args_list)} fw ...")
        fw_bar = tqdm(fw_args_list, desc="Horizon", unit="fw", ncols=80, leave=True)
        for args in fw_bar:
            fw = args[0]
            t_fw = time.time()
            fw_bar.set_description(f"fw={fw}/{N_PRED}")
            log.info(f"  fw={fw}: n={args[1].shape[0]:,}  "
                     f"GPU={args[9] if USE_GPU_LGBM else 'CPU'}")
            fw_r, fw_models, oof_slice, fold_maes = _train_one_fw(args)
            models[fw_r] = fw_models
            for idx, val in oof_slice.items():
                oof_preds[idx] = val
            fw_mask_r   = fw_keys == fw_r
            fw_val_mask = fw_mask_r & ~np.isnan(oof_preds)
            if fw_val_mask.sum() > 0:
                fw_metrics = compute_metrics(y[fw_val_mask], oof_preds[fw_val_mask],
                                             label=f"fw={fw_r} OOF")
                print_metrics(fw_metrics)
                fw_metrics["fold_maes"] = fold_maes
                fw_reports[fw_r] = fw_metrics
            else:
                fw_reports[fw_r] = {"mae": float("nan"), "fold_maes": fold_maes}
            elapsed_fw = time.time() - t_fw
            fw_bar.set_postfix({
                "OOF_MAE": f"{fw_reports[fw_r].get('mae', float('nan')):.4f}",
                "elapsed": f"{elapsed_fw:.0f}s",
            })

    fw_maes = {fw: fw_reports[fw]["mae"] for fw in fw_reports}
    valid_maes = [m for m in fw_maes.values() if np.isfinite(m)]
    macro_mae = float(np.mean(valid_maes)) if valid_maes else float("nan")
    log.info(f"Per-horizon OOF MAE: { {fw: f'{mae:.4f}' for fw, mae in fw_maes.items()} }")
    log.info(f"Macro OOF MAE: {macro_mae:.4f}")

    eval_report = dict(mode="per_horizon", fw_reports=fw_reports,
                       fw_maes=fw_maes, macro_mae=macro_mae)
    save_eval_report(eval_report, EVAL_REPORT_PATH)
    return models, eval_report, oof_preds


# ─────────────────────────────────────────────────────────────────────────────
# TemporalKFold training (non per-horizon fallback)
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm_cv(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, time_keys: np.ndarray,
) -> tuple[list, np.ndarray, dict]:
    log.info(f"LightGBM TemporalKFold training (folds={N_FOLDS-1}, gap={PURGE_GAP_DAYS}d) ...")
    tkf       = TemporalKFold(n_splits=N_FOLDS, gap_days=PURGE_GAP_DAYS)
    models    = []
    oof_preds = np.full(len(X), np.nan, dtype=np.float32)
    sw        = make_sample_weights(y) if USE_SAMPLE_WEIGHT else None
    fold_maes = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tkf.split(time_keys)):
        fold_num = fold_idx + 1
        log.info(f"  Fold {fold_num}  train={len(tr_idx):,}  val={len(va_idx):,}")
        model = _fit_lgbm(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
                          sw_tr=sw[tr_idx] if sw is not None else None,
                          label=f"Fold {fold_num}")
        fold_pred = model.predict(X[va_idx]).astype(np.float32)
        oof_preds[va_idx] = fold_pred
        fm = compute_metrics(y[va_idx], fold_pred, label=f"Fold {fold_num}")
        print_metrics(fm)
        fold_maes.append(fm["mae"])
        models.append(model)

    valid = ~np.isnan(oof_preds)
    overall = compute_metrics(y[valid], oof_preds[valid], label="Overall OOF")
    print_metrics(overall)
    fs = fold_summary(fold_maes)
    eval_report = dict(mode="temporal_cv", overall_metrics=overall, fold_summary=fs)
    save_eval_report(eval_report, EVAL_REPORT_PATH)
    return models, oof_preds, eval_report


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def fit_calibrator(oof_preds: np.ndarray, y: np.ndarray):
    if not USE_CALIBRATION:
        return None
    valid = ~np.isnan(oof_preds)
    if valid.sum() < 50:
        log.warning("Insufficient OOF samples for calibration, skipping")
        return None
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof_preds[valid], y[valid])
    cal_preds = np.clip(cal.predict(oof_preds[valid]), 0, 5)
    print_metrics(compute_metrics(y[valid], cal_preds, label="Calibrated OOF"))
    return cal


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_for_test(
    test_features: dict[str, np.ndarray],
    models,
    calibrator,
    fallback: float = 0.5,
) -> dict[str, list[float]]:
    preds: dict[str, list[float]] = {}

    if isinstance(models, dict):
        for region, feat_arr in test_features.items():
            region_preds = []
            for fw in range(1, N_PRED + 1):
                fv = feat_arr[fw - 1].reshape(1, -1)
                raw = float(np.mean([m.predict(fv)[0] for m in models[fw]])) if fw in models else fallback
                if calibrator is not None:
                    raw = float(calibrator.predict([raw])[0])
                region_preds.append(float(np.clip(raw, 0, 5)))
            preds[region] = region_preds
    else:
        for region, feat_arr in test_features.items():
            region_preds = []
            for fw_idx in range(N_PRED):
                fv = feat_arr[fw_idx].reshape(1, -1)
                raw = float(np.mean([m.predict(fv)[0] for m in models]))
                if calibrator is not None:
                    raw = float(calibrator.predict([raw])[0])
                region_preds.append(float(np.clip(raw, 0, 5)))
            preds[region] = region_preds

    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_models(models, calibrator, oof_mae: float):
    if isinstance(models, dict):
        for fw, fw_models in models.items():
            path = MODELS_PH_DIR / f"lgbm_fw{fw}.pkl"
            with open(path, "wb") as f:
                pickle.dump(fw_models, f, protocol=4)
        summary_path = MODELS_PH_DIR / "summary.pkl"
        with open(summary_path, "wb") as f:
            pickle.dump({"oof_mae": oof_mae, "mode": "per_horizon"}, f, protocol=4)
        log.info(f"LGBM models (per-horizon) saved → {MODELS_PH_DIR}")
    else:
        with open(MODELS_PKL_PATH, "wb") as f:
            pickle.dump({"models": models, "oof_mae": oof_mae}, f, protocol=4)
        log.info(f"LGBM models saved → {MODELS_PKL_PATH}")

    if calibrator is not None:
        with open(CALIBRATOR_PATH, "wb") as f:
            pickle.dump(calibrator, f, protocol=4)
        log.info(f"calibrator saved → {CALIBRATOR_PATH}")


def load_models():
    calibrator = None
    if CALIBRATOR_PATH.exists():
        with open(CALIBRATOR_PATH, "rb") as f:
            calibrator = pickle.load(f)

    ph_files = [MODELS_PH_DIR / f"lgbm_fw{fw}.pkl" for fw in range(1, N_PRED + 1)]
    if all(p.exists() for p in ph_files):
        models = {}
        for fw in range(1, N_PRED + 1):
            with open(MODELS_PH_DIR / f"lgbm_fw{fw}.pkl", "rb") as f:
                loaded = pickle.load(f)
                models[fw] = loaded if isinstance(loaded, list) else [loaded]
        log.info("LGBM per-horizon models loaded")
        return models, calibrator, "per_horizon"

    with open(MODELS_PKL_PATH, "rb") as f:
        d = pickle.load(f)
    return d["models"], calibrator, "temporal_cv"
