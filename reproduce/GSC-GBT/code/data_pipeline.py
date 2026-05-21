import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    METEO_COLS, N_METEO, N_PRED, SEED,
    MAX_WIN_PER_REGION, SAMPLE_FRAC, N_WORKERS,
    USE_SCORE_LAG_SHIFT, USE_GAP_ADAPTIVE_FALLBACK,
)
from features import make_features
from logging_setup import get_logger

log = get_logger("data_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Date string → ordinal integer (supports year > 9999)
# ─────────────────────────────────────────────────────────────────────────────

_MDAYS = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _ymd_to_ordinal(y: int, m: int, d: int) -> int:
    is_leap = (y % 4 == 0) and (y % 100 != 0 or y % 400 == 0)
    doy = sum(_MDAYS[1:m]) + (1 if m > 2 and is_leap else 0) + d
    y1 = y - 1
    return y1 * 365 + y1 // 4 - y1 // 100 + y1 // 400 + doy


def _date_str_to_ordinal(s) -> int:
    try:
        s_str = str(s).strip(" \t\r\n\x00").split("T")[0].split(" ")[0]
        parts = s_str.replace("/", "-").replace(".", "-").split("-")
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        return _ymd_to_ordinal(y, m, d)
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Training worker
# ─────────────────────────────────────────────────────────────────────────────

def _worker_train_region(args: tuple):
    """Process one region and return (X, y, regions, time_keys, fw_keys, month_keys)."""
    (region, meteo, months, dates, scores,
     rb, rm_df, ss, smo, proxy_ridge,
     sample_frac, max_win, seed_offset,
     region_gap_wk, test_month) = args

    score_idxs = np.where(~np.isnan(scores))[0]
    if len(score_idxs) < 10:
        return None, None, None, None, None, None

    n_take = max(1, int(len(score_idxs) * sample_frac))
    n_take = min(n_take, max_win)

    rng = np.random.RandomState(SEED + seed_offset)
    if n_take < len(score_idxs):
        # Oversample recent data (last 40%) to improve relevance
        split    = int(len(score_idxs) * 0.6)
        recent   = score_idxs[split:]
        early    = score_idxs[:split]
        n_recent = min(n_take, len(recent))
        n_early  = n_take - n_recent
        chosen   = list(rng.choice(recent, n_recent, replace=False))
        if n_early > 0 and len(early) > 0:
            chosen += list(rng.choice(early, min(n_early, len(early)), replace=False))
    else:
        chosen = score_idxs.tolist()

    date_ords = np.array([_date_str_to_ordinal(d) for d in dates], dtype=np.int64)

    X_list, y_list, g_list, t_list, fw_list, mo_list = [], [], [], [], [], []

    for target_idx in chosen:
        for fw in range(1, N_PRED + 1):
            w_end   = max(7, target_idx - (fw - 1) * 7)
            w_start = max(0, w_end - 91)
            win_fw  = meteo[w_start:w_end]
            if len(win_fw) < 7:
                continue
            wmo_fw = months[w_start:w_end]

            # ── Score Lag Gap Shift ───────────────────────────────────────────
            # Shift the score lookup backwards by region_gap_wk to simulate
            # the information state at test time.
            if USE_SCORE_LAG_SHIFT and region_gap_wk > 0:
                shift_days = int(region_gap_wk * 7)
                lookup_end = max(0, w_end - shift_days)
            else:
                lookup_end = w_end

            scores_before = scores[:lookup_end]
            valid_before  = np.where(~np.isnan(scores_before))[0]

            if len(valid_before) > 0:
                last_sc_idx = valid_before[-1]
                last_sc     = float(scores[last_sc_idx])
                target_ord  = date_ords[target_idx] if date_ords[target_idx] > 0 else date_ords[w_end - 1]
                last_sc_ord = date_ords[last_sc_idx]
                gap_wk      = max(0.0, (target_ord - last_sc_ord) / 7.0) if last_sc_ord > 0 else 52.0
            else:
                # ── Gap-Adaptive Proxy Fallback ───────────────────────────────
                # No score available → estimate using proxy rather than region mean.
                if USE_GAP_ADAPTIVE_FALLBACK and proxy_ridge is not None:
                    try:
                        from proxy import compute_proxy_signals
                        sigs = compute_proxy_signals(win_fw, rb)
                        last_sc = float(np.clip(
                            proxy_ridge.predict(sigs.reshape(1, -1))[0], 0, 5
                        ))
                    except Exception:
                        last_sc = ss.get("reg_mean", 0.5)
                else:
                    last_sc = ss.get("reg_mean", 0.5)
                gap_wk = float(region_gap_wk) if region_gap_wk > 0 else 52.0

            fv = make_features(
                win_fw, wmo_fw, rb, rm_df, ss, smo, fw, proxy_ridge,
                last_known_score=last_sc,
                gap_weeks=gap_wk,
                test_month=test_month,
            )
            X_list.append(fv)
            y_list.append(float(scores[target_idx]))
            g_list.append(region)

            t_key = int(date_ords[target_idx])
            if t_key == 0:
                valid = date_ords[max(0, target_idx - 14): target_idx + 1]
                nonzero = valid[valid > 0]
                t_key = int(nonzero[-1]) if len(nonzero) > 0 else 0
            t_list.append(t_key)
            fw_list.append(fw)
            mo_list.append(int(months[target_idx]))

    if not X_list:
        return None, None, None, None, None, None

    return (
        np.array(X_list,  dtype=np.float32),
        np.array(y_list,  dtype=np.float32),
        g_list,
        t_list,
        fw_list,
        mo_list,
    )


def _worker_test_region(args: tuple):
    """Build N_PRED feature vectors for one test region."""
    region, meteo, months, dates, rb, rm_df, ss, smo, proxy_ridge, test_month = args

    last_sc  = ss.get("last_score", ss.get("reg_mean", 0.5))
    last_ord = ss.get("last_date_ordinal", 0)

    test_dates_ord = [_date_str_to_ordinal(d) for d in dates]
    test_end_ord   = max(test_dates_ord) if test_dates_ord else 0

    rows = []
    for fw in range(1, N_PRED + 1):
        target_ord = test_end_ord + fw * 7 if test_end_ord > 0 else 0
        if last_ord > 0 and target_ord > 0:
            gap_wk = max(0.0, (target_ord - last_ord) / 7.0)
        else:
            gap_wk = 52.0
        fv = make_features(
            meteo, months, rb, rm_df, ss, smo, fw, proxy_ridge,
            last_known_score=last_sc,
            gap_weeks=gap_wk,
            test_month=test_month,
        )
        rows.append(fv)
    return region, np.array(rows, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing application
# ─────────────────────────────────────────────────────────────────────────────

def _apply_preproc(train: pd.DataFrame, preproc_artifacts: tuple | None) -> pd.DataFrame:
    if preproc_artifacts is None:
        return train
    from preprocessing import apply_pipeline
    bounds, log_features, table, quantile_table = preproc_artifacts
    log.info(f"  Applying preprocessing pipeline "
             f"(winsor={len(bounds)} log/sqrt={len(log_features)} "
             f"rank={len(quantile_table)}) ...")
    return apply_pipeline(train, bounds, log_features, table, quantile_table)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_training_dataset(
    train:              pd.DataFrame,
    clim_dict:          dict,
    month_dict:         dict,
    sstat_dict:         dict,
    smo_dict:           dict,
    proxy_ridge,
    sample_frac:        float        = SAMPLE_FRAC,
    max_win:            int          = MAX_WIN_PER_REGION,
    n_workers:          int          = N_WORKERS,
    preproc_artifacts:  tuple | None = None,
    region_gap_dict:    dict | None  = None,
    region_test_months: dict | None  = None,
) -> tuple[np.ndarray, ...]:
    """Parallel construction of the training feature matrix.

    Returns (X, y, groups, time_keys, fw_keys, month_keys).
    month_keys holds the target month for each sample (used by Calendar-Matched Validation).
    """
    log.info(f"Building training dataset (workers={n_workers}, lag_shift={USE_SCORE_LAG_SHIFT}) ...")

    train = _apply_preproc(train, preproc_artifacts)

    meteo_arr = train[METEO_COLS].values.astype(np.float32)
    month_arr = train["month"].values
    date_arr  = train["date"].values
    score_arr = train["score"].values.astype(np.float32)
    reg_arr   = train["region_id"].values

    log.info(
        f"  Date samples: {train['date'].head(3).tolist()}  "
        f"NaN count={pd.isnull(train['date']).sum()}"
    )

    region_args = []
    for region in train["region_id"].unique():
        mask     = reg_arr == region
        idxs     = np.where(mask)[0]
        gap_wk   = float(region_gap_dict.get(str(region), 52.0)) if region_gap_dict else 52.0
        t_month  = int(region_test_months.get(str(region), 0)) if region_test_months else 0
        region_args.append((
            region,
            meteo_arr[idxs],
            month_arr[idxs],
            date_arr[idxs],
            score_arr[idxs],
            clim_dict.get(region, {}),
            month_dict.get(region, None),
            sstat_dict.get(region, {}),
            smo_dict.get(region, {}),
            proxy_ridge,
            sample_frac,
            max_win,
            abs(hash(region)) % 100_000,
            gap_wk,
            t_month,
        ))

    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(_worker_train_region, region_args, chunksize=8),
            total=len(region_args),
            desc="Train regions",
        ))

    X_b, y_b, g_b, t_b, fw_b, mo_b = [], [], [], [], [], []
    for X_r, y_r, g_r, t_r, fw_r, mo_r in results:
        if X_r is None:
            continue
        X_b.append(X_r); y_b.append(y_r)
        g_b.extend(g_r); t_b.extend(t_r)
        fw_b.extend(fw_r); mo_b.extend(mo_r)

    X          = np.vstack(X_b).astype(np.float32)
    y          = np.concatenate(y_b).astype(np.float32)
    groups     = np.array(g_b)
    time_keys  = np.array(t_b, dtype=np.int64)
    fw_keys    = np.array(fw_b, dtype=np.int8)
    month_keys = np.array(mo_b, dtype=np.int8)

    log.info(f"  X: {X.shape},  y mean={y.mean():.3f},  regions={len(np.unique(groups))}")
    n_zero = int((time_keys == 0).sum())
    log.info(
        f"  time_keys: ordinal {time_keys.min()} ~ {time_keys.max()}  "
        f"(zero={n_zero}, {n_zero/max(len(time_keys),1)*100:.1f}%)"
    )
    log.info(f"  fw_keys dist: {dict(zip(*np.unique(fw_keys, return_counts=True)))}")
    log.info(f"  month_keys dist: {dict(zip(*np.unique(month_keys, return_counts=True)))}")
    return X, y, groups, time_keys, fw_keys, month_keys


def build_test_features(
    test:               pd.DataFrame,
    clim_dict:          dict,
    month_dict:         dict,
    sstat_dict:         dict,
    smo_dict:           dict,
    proxy_ridge,
    n_workers:          int          = N_WORKERS,
    preproc_artifacts:  tuple | None = None,
    region_test_months: dict | None  = None,
) -> dict[str, np.ndarray]:
    """Parallel construction of test features; returns {region: (N_PRED, N_FEATURES)}."""
    log.info(f"Building test features (workers={n_workers}) ...")

    test = _apply_preproc(test, preproc_artifacts)

    region_args = []
    for region, grp in test.groupby("region_id"):
        grp = grp.reset_index(drop=True)
        t_month = int(region_test_months.get(str(region), 0)) if region_test_months else 0
        region_args.append((
            region,
            grp[METEO_COLS].values.astype(np.float32),
            grp["month"].values,
            grp["date"].values,
            clim_dict.get(region, {}),
            month_dict.get(region, None),
            sstat_dict.get(region, {}),
            smo_dict.get(region, {}),
            proxy_ridge,
            t_month,
        ))

    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(_worker_test_region, region_args, chunksize=8),
            total=len(region_args),
            desc="Test regions",
        ))

    return {region: feat_arr for region, feat_arr in results}


def make_sample_weights(y: np.ndarray) -> np.ndarray:
    from config import WEIGHT_NONZERO, WEIGHT_SEVERE
    w = (
        1.0
        + (WEIGHT_NONZERO - 1.0) * (y > 0).astype(np.float32)
        + WEIGHT_SEVERE * (y >= 3).astype(np.float32)
    )
    return (w / w.mean()).astype(np.float32)
