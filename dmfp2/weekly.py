"""Daily → weekly aggregation (the weekly cache layer).

A "week" is the 7-day window ending at an anchor day:
  - TRAIN anchors = the scored days (scores fall exactly every 7th day).
  - TEST anchors  = the last day and every 7th day back (13 weeks over the 91-day window);
    the last week is the forecast origin.

Per channel we aggregate mean/std/min/max/sum over the 7 days. Daily preprocessing
(winsorize + log1p/sqrt) is fit on TRAIN ONLY and applied identically to both.
Cached per `weekly_hash(data_cfg)`.
"""
from __future__ import annotations

import multiprocessing as mp
import pickle

import numpy as np
import pandas as pd

from . import config as C
from . import io_data as io

_OFFSETS = np.arange(-(C.DAYS_PER_WEEK - 1), 1)  # [-6,...,0]


# ---------------------------------------------------------------------------
# Daily preprocessing
# ---------------------------------------------------------------------------
def fit_preprocess(train_daily: pd.DataFrame, data: C.DataConfig) -> dict:
    bounds = {}
    if data.winsorize:
        for ch in data.channels:
            lo = np.nanpercentile(train_daily[ch].to_numpy(), data.winsor_lower_pct)
            hi = np.nanpercentile(train_daily[ch].to_numpy(), data.winsor_upper_pct)
            bounds[ch] = (float(lo), float(hi))
    return {"winsor": bounds}


def apply_preprocess(daily: pd.DataFrame, prep: dict, data: C.DataConfig) -> pd.DataFrame:
    daily = daily.copy()
    if data.winsorize:
        for ch, (lo, hi) in prep["winsor"].items():
            daily[ch] = daily[ch].clip(lo, hi)
    for ch in data.log1p_cols:
        if ch in daily.columns:
            daily[ch] = np.log1p(daily[ch].clip(lower=0))
    for ch in data.sqrt_cols:
        if ch in daily.columns:
            daily[ch] = np.sqrt(daily[ch].clip(lower=0))
    return daily


# ---------------------------------------------------------------------------
# Per-region aggregation worker
# ---------------------------------------------------------------------------
def _aggregate_region(args):
    (region_idx, region_id, meteo, ordi, doy, month, woy, score, is_train, channels) = args
    n = len(meteo)
    if n < C.DAYS_PER_WEEK:
        return None

    if is_train:
        anchor_pos = np.where(~np.isnan(score))[0]
    else:
        L = n - 1
        anchor_pos = np.arange(L, -1, -C.DAYS_PER_WEEK)[::-1]
    anchor_pos = anchor_pos[anchor_pos >= C.DAYS_PER_WEEK - 1]
    if anchor_pos.size == 0:
        return None

    idx = anchor_pos[:, None] + _OFFSETS[None, :]      # (A, 7)
    win = meteo[idx]                                    # (A, 7, Ch)

    agg = {
        "mean": win.mean(1), "std": win.std(1),
        "min": win.min(1), "max": win.max(1), "sum": win.sum(1),
    }
    A = anchor_pos.size
    out = {
        "region_idx": np.full(A, region_idx, np.int64),
        "week_pos": np.arange(A, dtype=np.int64),
        "anchor_ord": ordi[anchor_pos],
        "doy": doy[anchor_pos],
        "month": month[anchor_pos],
        "woy": woy[anchor_pos],
        "score": (score[anchor_pos] if is_train else np.full(A, np.nan, np.float32)),
    }
    for ci, ch in enumerate(channels):
        for stat in C.WEEK_STATS:
            out[f"{ch}_{stat}"] = agg[stat][:, ci].astype(np.float32)
    return region_id, out


def _build_one(daily: pd.DataFrame, is_train: bool, channels: tuple) -> pd.DataFrame:
    tasks = []
    for region_idx, g in daily.groupby("region_idx", sort=True):
        meteo = g[list(channels)].to_numpy(np.float32)
        score = (g[C.SCORE_COL].to_numpy(np.float32) if is_train
                 else np.full(len(g), np.nan, np.float32))
        tasks.append((
            int(region_idx), g[C.ID_COL].iloc[0], meteo,
            g["ordinal"].to_numpy(np.int64), g["doy"].to_numpy(np.int64),
            g["month"].to_numpy(np.int64), g["woy"].to_numpy(np.int64),
            score, is_train, channels,
        ))

    with mp.Pool(C.N_WORKERS) as pool:
        results = pool.map(_aggregate_region, tasks, chunksize=16)

    cols = None
    parts, region_ids = [], []
    for r in results:
        if r is None:
            continue
        region_id, out = r
        if cols is None:
            cols = list(out.keys())
        parts.append(out)
        region_ids.append((out["region_idx"][0], region_id))

    frame = {c: np.concatenate([p[c] for p in parts]) for c in cols}
    df = pd.DataFrame(frame)
    id_map = dict(region_ids)
    df[C.ID_COL] = df["region_idx"].map(id_map)
    return df


# ---------------------------------------------------------------------------
# Public entry (cached)
# ---------------------------------------------------------------------------
def get_weekly(data: C.DataConfig = C.DATA, rebuild: bool = False):
    """Return (train_weekly, test_weekly), building + caching if needed."""
    h = C.weekly_hash(data)
    cache = C.CACHE_DIR / f"weekly_{h}.pkl"
    if cache.exists() and not rebuild:
        with open(cache, "rb") as f:
            return pickle.load(f)

    print(f"[weekly] building weekly aggregates (hash={h}) ...")
    train_daily = io.load_raw(C.TRAIN_PATH, data.channels, has_score=True)
    test_daily = io.load_raw(C.TEST_PATH, data.channels, has_score=False)

    prep = fit_preprocess(train_daily, data)
    train_daily = apply_preprocess(train_daily, prep, data)
    test_daily = apply_preprocess(test_daily, prep, data)
    with open(C.ARTIFACTS_DIR / f"preprocess_{h}.pkl", "wb") as f:
        pickle.dump(prep, f)

    train_weekly = _build_one(train_daily, is_train=True, channels=data.channels)
    test_weekly = _build_one(test_daily, is_train=False, channels=data.channels)
    print(f"[weekly] train weekly rows={len(train_weekly):,} | test weekly rows={len(test_weekly):,}")

    with open(cache, "wb") as f:
        pickle.dump((train_weekly, test_weekly), f)
    return train_weekly, test_weekly
