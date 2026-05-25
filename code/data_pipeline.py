"""Two-stage pipeline (Phase 7 refactor):

  Stage 1: daily CSV → weekly aggregated stats. Per-region parallel; builds the
           usual mean/std/min/max/sum block plus the daily-derived pressure
           variability stats (drop / climb / std).

  Stage 2: anchor rows. One row per scored anchor week (train) or per region's
           last week (test). Includes target_w1..target_w5 for train. The
           feature modules (features_score, features_climate, features_windowed)
           do the heavy lifting by looking up weekly_df at (region, week_idx − lag).

Phase 7 removed the legacy 13-position lag block (382 cols) — sliding-window
features carry the temporal signal and let LGBM split on a much smaller, less
correlated set.
"""
from __future__ import annotations

import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    TRAIN_PATH, TEST_PATH, METEO_FEATURES, STAT_SUFFIXES,
    N_REGIONS, TRAIN_WEEKS_PER_REGION, TEST_WEEKS_PER_REGION,
    LAG_WINDOW, N_WORKERS,
)

WEEK_DAYS = 7
N_STATS = len(STAT_SUFFIXES)
_N_STAT_COLS = len(METEO_FEATURES) * N_STATS

_ALL_STAT_COLS = [f"{feat}_{stat}" for feat in METEO_FEATURES for stat in STAT_SUFFIXES]
_SURF_PRE_IDX = METEO_FEATURES.index("surf_pre")

# Phase 7 — daily-derived pressure stats. Computed within each weekly block
# from the raw daily series.
PRESSURE_DERIVED_COLS = (
    "surf_pre_drop_3d_max",
    "surf_pre_climb_3d_max",
    "surf_pre_daily_std",
)
_N_PRESSURE_DERIVED = len(PRESSURE_DERIVED_COLS)

# Cumulative days before each month (non-leap year).
_MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_DOY_OFFSETS = [0] + [sum(_MONTH_DAYS[:i]) for i in range(1, 12)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _month_doy_from_str(date_str: str) -> tuple[int, int]:
    _, mm, dd = date_str.split("-")
    month = int(mm)
    doy = _DOY_OFFSETS[month - 1] + int(dd)
    return month, doy


def _pressure_derived_for_block(block_pressure: np.ndarray) -> tuple[float, float, float]:
    sp = block_pressure
    n = len(sp)
    if n < 4:
        return 0.0, 0.0, float(sp.std()) if n >= 2 else 0.0
    diffs_3d = sp[3:] - sp[:-3]
    min_diff = float(diffs_3d.min())
    max_diff = float(diffs_3d.max())
    drop = -min_diff if min_diff < 0.0 else 0.0
    climb = max_diff if max_diff > 0.0 else 0.0
    return drop, climb, float(sp.std())


# ---------------------------------------------------------------------------
# Stage 1 workers
# ---------------------------------------------------------------------------

def _worker_weekly_train(args: tuple):
    region_id, feat_matrix, date_strs, scores = args
    anchor_indices = np.where(scores >= 0)[0]
    n_weeks = len(anchor_indices)

    meta = np.zeros((n_weeks, 4), dtype=np.int32)
    stats = np.zeros((n_weeks, _N_STAT_COLS), dtype=np.float32)
    pressure_derived = np.zeros((n_weeks, _N_PRESSURE_DERIVED), dtype=np.float32)

    for week_idx, anchor_i in enumerate(anchor_indices):
        start_i = max(0, anchor_i - WEEK_DAYS + 1)
        block = feat_matrix[start_i: anchor_i + 1]
        month, doy = _month_doy_from_str(date_strs[anchor_i])
        meta[week_idx] = (week_idx, month, doy, int(scores[anchor_i]))
        stats[week_idx, 0::N_STATS] = np.nanmean(block, axis=0)
        stats[week_idx, 1::N_STATS] = np.nanstd(block, axis=0)
        stats[week_idx, 2::N_STATS] = np.nanmin(block, axis=0)
        stats[week_idx, 3::N_STATS] = np.nanmax(block, axis=0)
        stats[week_idx, 4::N_STATS] = np.nansum(block, axis=0)
        drop, climb, std = _pressure_derived_for_block(block[:, _SURF_PRE_IDX])
        pressure_derived[week_idx] = (drop, climb, std)
    return region_id, meta, stats, pressure_derived


def _worker_weekly_test(args: tuple):
    region_id, feat_matrix, date_strs = args
    n_weeks = len(date_strs) // WEEK_DAYS

    meta = np.zeros((n_weeks, 3), dtype=np.int32)
    stats = np.zeros((n_weeks, _N_STAT_COLS), dtype=np.float32)
    pressure_derived = np.zeros((n_weeks, _N_PRESSURE_DERIVED), dtype=np.float32)

    for w in range(n_weeks):
        block = feat_matrix[w * WEEK_DAYS: (w + 1) * WEEK_DAYS]
        anchor_date_str = date_strs[(w + 1) * WEEK_DAYS - 1]
        month, doy = _month_doy_from_str(anchor_date_str)
        meta[w] = (w, month, doy)
        stats[w, 0::N_STATS] = np.nanmean(block, axis=0)
        stats[w, 1::N_STATS] = np.nanstd(block, axis=0)
        stats[w, 2::N_STATS] = np.nanmin(block, axis=0)
        stats[w, 3::N_STATS] = np.nanmax(block, axis=0)
        stats[w, 4::N_STATS] = np.nansum(block, axis=0)
        drop, climb, std = _pressure_derived_for_block(block[:, _SURF_PRE_IDX])
        pressure_derived[w] = (drop, climb, std)
    return region_id, meta, stats, pressure_derived


# ---------------------------------------------------------------------------
# Stage 1 entry point
# ---------------------------------------------------------------------------

def load_and_aggregate_daily_to_weekly(
    filepath=None,
    is_train: bool = True,
    chunksize: int = 500_000,
    n_workers: int = N_WORKERS,
    preproc_artifacts: tuple | None = None,
) -> pd.DataFrame:
    """Load CSV, aggregate daily → weekly. Returns DataFrame with
    region_id, week_idx, month, day_of_year, {feat}_{stat} × 9 × 5,
    + 3 pressure-derived cols, + score (train only)."""
    if filepath is None:
        filepath = TRAIN_PATH if is_train else TEST_PATH

    dtype_map = {feat: np.float32 for feat in METEO_FEATURES}
    dtype_map["region_id"] = str
    dtype_map["date"] = str
    if is_train:
        dtype_map["score"] = "Int8"

    chunks = []
    print(f"Reading {'train' if is_train else 'test'} CSV in chunks...")
    for chunk in tqdm(pd.read_csv(filepath, dtype=dtype_map, chunksize=chunksize), desc="Loading"):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {len(df):,} rows")

    if preproc_artifacts is not None:
        from preprocessing import apply_pipeline
        bounds, log_features, imputation_table, quantile_table = preproc_artifacts
        print(f"  Applying preprocessing pipeline (winsor={len(bounds)} "
              f"log/sqrt={len(log_features)} rank={len(quantile_table)})...")
        df = apply_pipeline(df, bounds, log_features, imputation_table, quantile_table)

    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)

    print(f"Aggregating to weekly stats per region (workers={n_workers})...")
    region_args = []
    for region_id, region_df in df.groupby("region_id", sort=False):
        region_df = region_df.reset_index(drop=True)
        feat_matrix = region_df[METEO_FEATURES].values.astype(np.float32)
        date_strs = region_df["date"].tolist()
        if is_train:
            scores = region_df["score"].fillna(-128).astype(np.int8).values
            region_args.append((region_id, feat_matrix, date_strs, scores))
        else:
            region_args.append((region_id, feat_matrix, date_strs))

    worker_fn = _worker_weekly_train if is_train else _worker_weekly_test

    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(worker_fn, region_args, chunksize=8),
                            total=len(region_args), desc="Regions"))

    region_ids_per_row = []
    meta_blocks = []
    stat_blocks = []
    pressure_blocks = []
    for rid, meta, stats, pressure_derived in results:
        n = meta.shape[0]
        region_ids_per_row.extend([rid] * n)
        meta_blocks.append(meta)
        stat_blocks.append(stats)
        pressure_blocks.append(pressure_derived)

    meta_all = np.vstack(meta_blocks)
    stat_all = np.vstack(stat_blocks)
    pressure_all = np.vstack(pressure_blocks)

    weekly_df = pd.DataFrame(stat_all, columns=_ALL_STAT_COLS, copy=False)
    weekly_df.insert(0, "region_id", region_ids_per_row)
    weekly_df.insert(1, "week_idx", meta_all[:, 0])
    weekly_df.insert(2, "month", meta_all[:, 1].astype(np.int8))
    weekly_df.insert(3, "day_of_year", meta_all[:, 2].astype(np.int16))
    if is_train:
        weekly_df["score"] = meta_all[:, 3].astype(np.int8)
    for i, col in enumerate(PRESSURE_DERIVED_COLS):
        weekly_df[col] = pressure_all[:, i]

    n_actual = weekly_df["region_id"].nunique()
    if is_train:
        assert len(weekly_df) == n_actual * TRAIN_WEEKS_PER_REGION
        assert weekly_df["score"].between(0, 5).all()
    else:
        assert len(weekly_df) == n_actual * TEST_WEEKS_PER_REGION
    print(f"  Weekly DataFrame: {weekly_df.shape}")
    return weekly_df


# ---------------------------------------------------------------------------
# Stage 2: anchor rows (replaces the legacy 13-position lag matrix)
# ---------------------------------------------------------------------------

def construct_anchor_rows(
    weekly_df: pd.DataFrame,
    is_train: bool = True,
    lag_window: int = LAG_WINDOW,
) -> pd.DataFrame:
    """Build the anchor matrix:
      train: one row per (region, week_idx) where lag_window ≤ week_idx ≤ n-6
             (need lag look-back + 5 future weeks for targets).
      test:  the last week per region (week_idx = n-1).

    Columns: region_id, week_idx, month, day_of_year [, target_w1..target_w5].
    The feature modules then look up weekly_df to add their features.
    """
    weekly_df = weekly_df.sort_values(["region_id", "week_idx"]).reset_index(drop=True)
    rows = []
    targets = []

    for rid, sub in weekly_df.groupby("region_id", sort=False):
        sub = sub.sort_values("week_idx").reset_index(drop=True)
        n = len(sub)
        if is_train:
            scores = sub["score"].to_numpy(np.int32)
            anchors = np.arange(lag_window, n - 5)
            if len(anchors) == 0:
                continue
            t = np.zeros((len(anchors), 5), dtype=np.float32)
            for i, a in enumerate(anchors):
                t[i] = [scores[a + h] for h in range(1, 6)]
            targets.append(t)
        else:
            anchors = np.array([n - 1], dtype=np.int32)
        anchor_meta = sub.iloc[anchors][["week_idx", "month", "day_of_year"]].copy()
        anchor_meta.insert(0, "region_id", rid)
        rows.append(anchor_meta)

    out = pd.concat(rows, ignore_index=True)
    if is_train:
        target_all = np.vstack(targets)
        for h in range(1, 6):
            out[f"target_w{h}"] = target_all[:, h - 1]
    out["week_idx"] = out["week_idx"].astype(np.int32)
    out["month"] = out["month"].astype(np.int8)
    out["day_of_year"] = out["day_of_year"].astype(np.int16)
    print(f"  Anchor rows: {out.shape}")
    return out
