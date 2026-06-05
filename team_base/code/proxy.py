import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from pathlib import Path

from config import (
    METEO_COLS, COL_IDX, N_PROXY_SIGS, PROXY_WINDOWS,
    PROXY_SAMPLES_PER_REGION, PROXY_RIDGE_ALPHA, PROXY_RIDGE_PATH, SEED,
)
from logging_setup import get_logger

log = get_logger("proxy")


# ─────────────────────────────────────────────────────────────────────────────
# Internal: single window → 7 physical drought signals
# ─────────────────────────────────────────────────────────────────────────────

def _proxy_signals_window(arr: np.ndarray, rb: dict) -> np.ndarray:
    """
    Compute 7 physical drought signals from a (N, N_METEO) weather window.

    Returns (N_PROXY_SIGS,) float32.
    """
    if len(arr) == 0:
        return np.zeros(N_PROXY_SIGS, dtype=np.float32)

    ci    = COL_IDX
    tmp_  = arr[:, ci["tmp"]]
    wb_   = arr[:, ci["wb_tmp"]]
    prec_ = arr[:, ci["prec"]]
    hum   = arr[:, ci["humidity"]]
    tmax  = arr[:, ci["tmp_max"]]
    tmin  = arr[:, ci["tmp_min"]]

    sigs = np.empty(N_PROXY_SIGS, dtype=np.float32)

    # 1. VPD anomaly (temp - wet-bulb temp vs long-term baseline)
    vpd_cur = float(np.nanmean(tmp_ - wb_))
    vpd_mu  = rb.get("tmp_mean", 20.0) - rb.get("wb_tmp_mean", 10.0)
    vpd_sig = max(rb.get("tmp_std", 5.0) + rb.get("wb_tmp_std", 3.0), 0.5)
    sigs[0] = (vpd_cur - vpd_mu) / vpd_sig

    # 2. Precipitation deficit z-score (positive = water deficit)
    prec_mu  = rb.get("prec_mean", 1.0)
    prec_sig = max(rb.get("prec_std", 1.0), 0.01)
    expected = prec_mu * len(arr)
    actual   = float(np.nansum(prec_))
    sigs[1]  = (expected - actual) / max(prec_sig * np.sqrt(len(arr)), 0.1)

    # 3. Dry day fraction (prec < 0.1 mm)
    sigs[2] = float(np.nanmean(prec_ < 0.1))

    # 4. Normalized heat degree days (relative to region long-term mean)
    hdd_cur = float(np.nanmean(np.maximum(tmp_ - 10.0, 0.0)))
    hdd_ref = max(rb.get("tmp_mean", 20.0) - 10.0, 0.5)
    sigs[3] = hdd_cur / hdd_ref

    # 5. Humidity anomaly
    hum_mu  = rb.get("humidity_mean", 60.0)
    hum_sig = max(rb.get("humidity_std", 15.0), 0.5)
    sigs[4] = (float(np.nanmean(hum)) - hum_mu) / hum_sig

    # 6. Temperature anomaly
    tmp_mu  = rb.get("tmp_mean", 20.0)
    tmp_sig = max(rb.get("tmp_std", 5.0), 0.5)
    sigs[5] = (float(np.nanmean(tmp_)) - tmp_mu) / tmp_sig

    # 7. DTR anomaly (diurnal temperature range)
    dtr_cur = float(np.nanmean(tmax - tmin))
    dtr_mu  = rb.get("tmp_range_mean", 10.0)
    dtr_sig = max(rb.get("tmp_range_std", 3.0), 0.5)
    sigs[6] = (dtr_cur - dtr_mu) / dtr_sig

    return np.nan_to_num(sigs, nan=0.0, posinf=5.0, neginf=-5.0)


def compute_proxy_signals(window_arr: np.ndarray, rb: dict) -> np.ndarray:
    """Compute 7 signals at 3 scales (7d, 21d, 91d) and concatenate to (21,)."""
    out = []
    for w in PROXY_WINDOWS:
        tail = window_arr[-w:] if len(window_arr) >= w else window_arr
        out.append(_proxy_signals_window(tail, rb))
    return np.concatenate(out)


# ─────────────────────────────────────────────────────────────────────────────
# Public: inference
# ─────────────────────────────────────────────────────────────────────────────

def get_proxy_scores(
    window_arr: np.ndarray,
    rb: dict,
    proxy_ridge: Ridge,
) -> tuple[float, float, float, float]:
    """Return (p7, p21, p91, p_main) proxy drought scores.

    Each scale uses only its own segment of the Ridge coefficients so that
    short-term and long-term proxies can differ.
    """
    coef      = proxy_ridge.coef_
    intercept = proxy_ridge.intercept_

    sig7  = _proxy_signals_window(
        window_arr[-7:]  if len(window_arr) >= 7  else window_arr, rb)
    sig21 = _proxy_signals_window(
        window_arr[-21:] if len(window_arr) >= 21 else window_arr, rb)
    sig91 = _proxy_signals_window(window_arr, rb)

    p7   = float(np.clip(np.dot(sig7,  coef[:7])   + intercept / 3, 0, 5))
    p21  = float(np.clip(np.dot(sig21, coef[7:14]) + intercept / 3, 0, 5))
    p91  = float(np.clip(np.dot(sig91, coef[14:])  + intercept / 3, 0, 5))

    all_sigs = np.concatenate([sig7, sig21, sig91])
    p_main   = float(np.clip(
        proxy_ridge.predict(all_sigs.reshape(1, -1))[0], 0, 5))

    return p7, p21, p91, p_main


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _worker_proxy_region(args):
    """Per-region worker function (module-level for pickling)."""
    region, meteo, scores, rb, n_samples = args
    valid_idxs = np.where(~np.isnan(scores))[0]
    if len(valid_idxs) == 0:
        return [], []

    if len(valid_idxs) > n_samples:
        valid_idxs = valid_idxs[-n_samples:]

    X_reg, y_reg = [], []
    for vi in valid_idxs:
        if vi < 7:
            continue
        win  = meteo[max(0, vi - 91): vi]
        sigs = compute_proxy_signals(win, rb)
        X_reg.append(sigs)
        y_reg.append(float(scores[vi]))
    return X_reg, y_reg


def fit_proxy_ridge(
    train: pd.DataFrame,
    clim_dict: dict,
) -> Ridge:
    """Fit a Ridge proxy model on scored training samples using parallel per-region sampling."""
    import multiprocessing as mp
    from config import N_WORKERS

    log.info("Fitting Proxy Ridge ...")
    meteo_arr = train[METEO_COLS].values.astype(np.float32)
    score_arr = train["score"].values.astype(np.float32)
    reg_arr   = train["region_id"].values

    region_args = []
    for region in train["region_id"].unique():
        mask   = reg_arr == region
        idxs   = np.where(mask)[0]
        region_args.append((
            region,
            meteo_arr[idxs],
            score_arr[idxs],
            clim_dict.get(region, {}),
            PROXY_SAMPLES_PER_REGION,
        ))

    log.info(f"  Collecting proxy samples in parallel (workers={N_WORKERS}) ...")
    with mp.Pool(processes=N_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(_worker_proxy_region, region_args, chunksize=16),
            total=len(region_args),
            desc="Proxy samples",
        ))

    px_X, px_y = [], []
    for X_reg, y_reg in results:
        px_X.extend(X_reg)
        px_y.extend(y_reg)

    px_X = np.array(px_X, dtype=np.float32)
    px_y = np.array(px_y, dtype=np.float32)

    ok = np.isfinite(px_X).all(axis=1) & np.isfinite(px_y)
    px_X, px_y = px_X[ok], px_y[ok]
    log.info(f"  Proxy samples: {len(px_y):,}")

    ridge = Ridge(alpha=PROXY_RIDGE_ALPHA, fit_intercept=True)
    ridge.fit(px_X, px_y)

    px_pred = np.clip(ridge.predict(px_X), 0, 5)
    train_mae = mean_absolute_error(px_y, px_pred)
    corr = float(np.corrcoef(px_y, px_pred)[0, 1])
    log.info(f"  Proxy Ridge train MAE={train_mae:.4f}  corr={corr:.4f}")

    return ridge


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_proxy_ridge(ridge: Ridge, path: Path = PROXY_RIDGE_PATH):
    with open(path, "wb") as f:
        pickle.dump(ridge, f, protocol=4)
    log.info(f"proxy_ridge saved → {path}")


def load_proxy_ridge(path: Path = PROXY_RIDGE_PATH) -> Ridge:
    with open(path, "rb") as f:
        ridge = pickle.load(f)
    log.info(f"proxy_ridge loaded ← {path}")
    return ridge
