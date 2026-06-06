"""
Two-stage pipeline:
  Stage 1: load_and_aggregate_daily_to_weekly  — daily rows → weekly aggregated rows
  Stage 2: construct_lag_features              — weekly rows → model-ready feature matrix

Both stages parallelize over regions using multiprocessing.Pool.
Workers return numpy arrays (not list[dict]) so the main process can assemble
the final DataFrame via np.vstack + a single typed column construction —
avoiding ~1B boxed Python float objects on the full train run.
"""

import json
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    TRAIN_PATH, TEST_PATH, METEO_FEATURES, STAT_SUFFIXES,
    LAG_WINDOW, N_REGIONS, TRAIN_WEEKS_PER_REGION, TEST_WEEKS_PER_REGION,
    MODELS_DIR, N_WORKERS,
)

WEEK_DAYS = 7
N_FEATURES = len(METEO_FEATURES)
N_STATS = len(STAT_SUFFIXES)  # 5

# Cumulative days before each month (non-leap-year).
_MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_DOY_OFFSETS = [0] + [sum(_MONTH_DAYS[:i]) for i in range(1, 12)]

# Stat column layout: feat outer, stat inner — wind_mean, wind_std, wind_min, wind_max, wind_sum, wind_min_mean, ...
_ALL_STAT_COLS = [f"{feat}_{stat}" for feat in METEO_FEATURES for stat in STAT_SUFFIXES]
_STAT_IDX = {name: i for i, name in enumerate(_ALL_STAT_COLS)}
_N_STAT_COLS = len(_ALL_STAT_COLS)  # 75

# Precomputed column-index vectors for fancy-indexing inside the lag worker.
# Each indexes into a (n_weeks, _N_STAT_COLS) stat matrix.
_LAG0_IDX = np.array(
    [_STAT_IDX[f"{feat}_{stat}"] for feat in METEO_FEATURES for stat in STAT_SUFFIXES],
    dtype=np.int32,
)  # 75
_LAG_NEAR_IDX = np.array(
    [_STAT_IDX[f"{feat}_{stat}"] for feat in METEO_FEATURES for stat in ("mean", "max", "sum")],
    dtype=np.int32,
)  # 45
_LAG_FAR_IDX = np.array(
    [_STAT_IDX[f"{feat}_{stat}"] for feat in METEO_FEATURES for stat in ("mean", "max")],
    dtype=np.int32,
)  # 30
_MEAN_IDX = np.array(
    [_STAT_IDX[f"{feat}_mean"] for feat in METEO_FEATURES],
    dtype=np.int32,
)  # 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _month_doy_from_str(date_str: str) -> tuple[int, int]:
    """Return (month, day_of_year) from 'YYYY-MM-DD'. Avoids pd.to_datetime overflow."""
    _, mm, dd = date_str.split("-")
    month = int(mm)
    doy = _DOY_OFFSETS[month - 1] + int(dd)
    return month, doy


# ---------------------------------------------------------------------------
# Stage 1 workers (top-level so they are picklable).
# Return numpy arrays instead of list[dict].
# ---------------------------------------------------------------------------

def _worker_weekly_train(args: tuple):
    """
    Aggregate one region's daily rows → weekly stats.
    Returns: (region_id, meta_array, stat_array)
      meta_array  : (n_weeks, 4) int32  — [week_idx, month, day_of_year, score]
      stat_array  : (n_weeks, _N_STAT_COLS) float32 in _ALL_STAT_COLS order
    """
    region_id, feat_matrix, date_strs, scores = args
    anchor_indices = np.where(scores >= 0)[0]
    n_weeks = len(anchor_indices)

    meta = np.zeros((n_weeks, 4), dtype=np.int32)
    stats = np.zeros((n_weeks, _N_STAT_COLS), dtype=np.float32)

    for week_idx, anchor_i in enumerate(anchor_indices):
        start_i = max(0, anchor_i - WEEK_DAYS + 1)
        block = feat_matrix[start_i : anchor_i + 1]  # (≤7, N_FEATURES)
        month, doy = _month_doy_from_str(date_strs[anchor_i])
        meta[week_idx, 0] = week_idx
        meta[week_idx, 1] = month
        meta[week_idx, 2] = doy
        meta[week_idx, 3] = int(scores[anchor_i])

        # Layout: stats[w, fi*N_STATS + 0..N_STATS-1] = (mean, std, min, max, sum, p95).
        stats[week_idx, 0::N_STATS] = np.nanmean(block, axis=0)
        stats[week_idx, 1::N_STATS] = np.nanstd(block, axis=0)
        stats[week_idx, 2::N_STATS] = np.nanmin(block, axis=0)
        stats[week_idx, 3::N_STATS] = np.nanmax(block, axis=0)
        stats[week_idx, 4::N_STATS] = np.nansum(block, axis=0)
        # Phase 2: p95 captures tail intensity that min/max/std flatten —
        # a single extreme day in a week of 6 normal ones is invisible to
        # mean/std but jumps p95.
        stats[week_idx, 5::N_STATS] = np.nanpercentile(block, 95, axis=0)

    return region_id, meta, stats


def _worker_weekly_test(args: tuple):
    """
    Aggregate one region's daily rows → weekly stats (test mode: fixed 7-day blocks).
    Returns: (region_id, meta_array, stat_array)
      meta_array  : (n_weeks, 3) int32  — [week_idx, month, day_of_year]
      stat_array  : (n_weeks, _N_STAT_COLS) float32
    """
    region_id, feat_matrix, date_strs = args
    n_rows = len(date_strs)
    n_weeks = n_rows // WEEK_DAYS

    meta = np.zeros((n_weeks, 3), dtype=np.int32)
    stats = np.zeros((n_weeks, _N_STAT_COLS), dtype=np.float32)

    for w in range(n_weeks):
        block = feat_matrix[w * WEEK_DAYS : (w + 1) * WEEK_DAYS]
        anchor_date_str = date_strs[(w + 1) * WEEK_DAYS - 1]
        month, doy = _month_doy_from_str(anchor_date_str)
        meta[w, 0] = w
        meta[w, 1] = month
        meta[w, 2] = doy

        stats[w, 0::N_STATS] = np.nanmean(block, axis=0)
        stats[w, 1::N_STATS] = np.nanstd(block, axis=0)
        stats[w, 2::N_STATS] = np.nanmin(block, axis=0)
        stats[w, 3::N_STATS] = np.nanmax(block, axis=0)
        stats[w, 4::N_STATS] = np.nansum(block, axis=0)
        stats[w, 5::N_STATS] = np.nanpercentile(block, 95, axis=0)

    return region_id, meta, stats


# ---------------------------------------------------------------------------
# Stage 1: daily → weekly
# ---------------------------------------------------------------------------

def load_and_aggregate_daily_to_weekly(
    filepath=None,
    is_train: bool = True,
    chunksize: int = 500_000,
    n_workers: int = N_WORKERS,
    preproc_artifacts: tuple | None = None,
) -> pd.DataFrame:
    """
    Load a CSV and aggregate daily rows to weekly summary statistics.

    Train: scored rows define week anchors; aggregate 7 preceding days per feature.
    Test:  fixed 7-day blocks (91 rows → 13 weeks).

    If `preproc_artifacts` is supplied it must be the 4-tuple
    (bounds, log_features, imputation_table, quantile_table) returned by
    `preprocessing.load_preprocessing_artifacts`; daily rows are run through
    `preprocessing.apply_pipeline` (impute → winsorize → log/sqrt → rank-norm)
    before the per-region groupby. Passing None skips preprocessing.

    Returns DataFrame: region_id, week_idx, month, day_of_year,
      {feat}_{stat} × 15 feats × 5 stats, score (train only).
    """
    if filepath is None:
        filepath = TRAIN_PATH if is_train else TEST_PATH

    dtype_map = {feat: np.float32 for feat in METEO_FEATURES}
    dtype_map["region_id"] = str
    dtype_map["date"] = str
    if is_train:
        dtype_map["score"] = "Int8"

    chunks = []
    print(f"Reading {'train' if is_train else 'test'} CSV in chunks...")
    for chunk in tqdm(
        pd.read_csv(filepath, dtype=dtype_map, chunksize=chunksize),
        desc="Loading",
    ):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {len(df):,} rows")

    if preproc_artifacts is not None:
        # Lazy import to avoid a hard dependency when USE_PREPROCESSING=False.
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
        results = list(tqdm(
            pool.imap(worker_fn, region_args, chunksize=8),
            total=len(region_args),
            desc="Regions",
        ))

    # ── Assemble result with one vstack + typed columns (no dict overhead) ──
    print("  Concatenating region arrays...")
    region_ids_per_row = []
    meta_blocks = []
    stat_blocks = []
    for rid, meta, stats in results:
        n = meta.shape[0]
        region_ids_per_row.extend([rid] * n)
        meta_blocks.append(meta)
        stat_blocks.append(stats)

    meta_all = np.vstack(meta_blocks)
    stat_all = np.vstack(stat_blocks)

    print("  Building weekly DataFrame...")
    weekly_df = pd.DataFrame(stat_all, columns=_ALL_STAT_COLS, copy=False)
    weekly_df.insert(0, "region_id", region_ids_per_row)
    weekly_df.insert(1, "week_idx", meta_all[:, 0])
    weekly_df.insert(2, "month", meta_all[:, 1].astype(np.int8))
    weekly_df.insert(3, "day_of_year", meta_all[:, 2].astype(np.int16))
    if is_train:
        weekly_df["score"] = meta_all[:, 3].astype(np.int8)

    n_actual_regions = weekly_df["region_id"].nunique()
    if is_train:
        assert len(weekly_df) == n_actual_regions * TRAIN_WEEKS_PER_REGION, (
            f"Uneven weekly rows: {len(weekly_df)} for {n_actual_regions} regions"
        )
        assert weekly_df["score"].between(0, 5).all(), "Unexpected score values"
    else:
        assert len(weekly_df) == n_actual_regions * TEST_WEEKS_PER_REGION, (
            f"Uneven weekly rows: {len(weekly_df)} for {n_actual_regions} regions"
        )

    print(f"  Weekly DataFrame: {weekly_df.shape}")
    return weekly_df


# ---------------------------------------------------------------------------
# Stage 2 worker. Returns numpy arrays instead of list[dict].
# ---------------------------------------------------------------------------

def _worker_lag(args: tuple):
    """
    Build lag features for one region.
    Returns: (region_id, meta_array, feat_array, target_array_or_None)
      meta_array   : (n_anchors, 3) int32  — [week_idx, month, day_of_year]
      feat_array   : (n_anchors, n_feat_cols) float32  — in _build_feature_col_names() order
      target_array : (n_anchors, 5) float32 or None
    """
    region_id, stat_matrix, week_indices, months, doys, scores, lag_window, is_train = args
    n_weeks = stat_matrix.shape[0]

    if is_train:
        max_anchor = n_weeks - 6
        anchors = np.arange(lag_window, max_anchor + 1, dtype=np.int32)
    else:
        anchors = np.array([n_weeks - 1], dtype=np.int32)

    n_anchors = len(anchors)
    n_lag0 = _LAG0_IDX.shape[0]      # 75
    n_near = _LAG_NEAR_IDX.shape[0]  # 45
    n_far = _LAG_FAR_IDX.shape[0]    # 30
    n_mean = _MEAN_IDX.shape[0]      # 15

    n_feat_cols = (
        n_lag0
        + 4 * n_near                  # lag 1-4
        + (lag_window - 4) * n_far    # lag 5..lag_window
        + n_mean * 3                  # roll4 mean/max/std
        + n_mean * 2                  # roll8 mean/max
    )

    meta_array = np.zeros((n_anchors, 3), dtype=np.int32)
    feat_array = np.zeros((n_anchors, n_feat_cols), dtype=np.float32)
    target_array = np.zeros((n_anchors, 5), dtype=np.float32) if is_train else None

    for ai, anchor in enumerate(anchors):
        anchor = int(anchor)
        meta_array[ai, 0] = week_indices[anchor]
        meta_array[ai, 1] = months[anchor]
        meta_array[ai, 2] = doys[anchor]

        col = 0

        # Lag 0: all 5 stats
        feat_array[ai, col : col + n_lag0] = stat_matrix[anchor, _LAG0_IDX]
        col += n_lag0

        # Lag 1-4: mean, max, sum
        for lag in range(1, 5):
            idx = anchor - lag
            if idx >= 0:
                feat_array[ai, col : col + n_near] = stat_matrix[idx, _LAG_NEAR_IDX]
            col += n_near

        # Lag 5..lag_window: mean, max
        for lag in range(5, lag_window + 1):
            idx = anchor - lag
            if idx >= 0:
                feat_array[ai, col : col + n_far] = stat_matrix[idx, _LAG_FAR_IDX]
            col += n_far

        # Rolling 4-week (lags 1..4) — mean, max, std of each feature's weekly mean.
        roll4_start = max(0, anchor - 4)
        if roll4_start < anchor:
            roll4_vals = stat_matrix[roll4_start:anchor][:, _MEAN_IDX]  # (n_lags, n_feats)
        else:
            roll4_vals = np.zeros((1, n_mean), dtype=np.float32)
        m = np.nanmean(roll4_vals, axis=0)
        mx = np.nanmax(roll4_vals, axis=0)
        sd = np.nanstd(roll4_vals, axis=0)
        # Interleave per-feature to match _build_feature_col_names: feat outer, stat inner.
        feat_array[ai, col : col + n_mean * 3] = np.stack([m, mx, sd], axis=1).ravel()
        col += n_mean * 3

        # Rolling 8-week (lags 1..8) — mean, max.
        roll8_start = max(0, anchor - 8)
        if roll8_start < anchor:
            roll8_vals = stat_matrix[roll8_start:anchor][:, _MEAN_IDX]
        else:
            roll8_vals = np.zeros((1, n_mean), dtype=np.float32)
        m = np.nanmean(roll8_vals, axis=0)
        mx = np.nanmax(roll8_vals, axis=0)
        feat_array[ai, col : col + n_mean * 2] = np.stack([m, mx], axis=1).ravel()
        col += n_mean * 2

        if is_train:
            for h in range(1, 6):
                target_array[ai, h - 1] = scores[anchor + h]

    return region_id, meta_array, feat_array, target_array


# ---------------------------------------------------------------------------
# Stage 2: weekly → lag feature matrix
# ---------------------------------------------------------------------------

def _build_feature_col_names() -> list[str]:
    cols = []
    for feat in METEO_FEATURES:
        for stat in STAT_SUFFIXES:
            cols.append(f"{feat}_{stat}_lag0")
    for lag in range(1, 5):
        for feat in METEO_FEATURES:
            for stat in ["mean", "max", "sum"]:
                cols.append(f"{feat}_{stat}_lag{lag}")
    for lag in range(5, LAG_WINDOW + 1):
        for feat in METEO_FEATURES:
            for stat in ["mean", "max"]:
                cols.append(f"{feat}_{stat}_lag{lag}")
    for feat in METEO_FEATURES:
        for stat in ["mean", "max", "std"]:
            cols.append(f"{feat}_{stat}_roll4")
    for feat in METEO_FEATURES:
        for stat in ["mean", "max"]:
            cols.append(f"{feat}_{stat}_roll8")
    return cols


def get_feature_columns() -> list[str]:
    return _build_feature_col_names()


def construct_lag_features(
    weekly_df: pd.DataFrame,
    lag_window: int = LAG_WINDOW,
    is_train: bool = True,
    n_workers: int = N_WORKERS,
) -> pd.DataFrame:
    """
    Build model-ready lag feature matrix from weekly aggregated data.
    Parallelized over regions; assembled via np.vstack to avoid dict-of-floats overhead.
    """
    weekly_df = weekly_df.sort_values(["region_id", "week_idx"]).reset_index(drop=True)

    print(f"Building lag features ({'train' if is_train else 'test'}, workers={n_workers})...")
    region_args = []
    for region_id, reg_df in weekly_df.groupby("region_id", sort=False):
        reg_df = reg_df.reset_index(drop=True)
        stat_matrix = reg_df[_ALL_STAT_COLS].values.astype(np.float32)
        week_indices = reg_df["week_idx"].values.astype(np.int32)
        months = reg_df["month"].values.astype(np.int32)
        doys = reg_df["day_of_year"].values.astype(np.int32)
        scores = reg_df["score"].values.astype(np.float32) if is_train else None
        region_args.append((
            region_id, stat_matrix, week_indices, months, doys,
            scores, lag_window, is_train,
        ))

    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(_worker_lag, region_args, chunksize=8),
            total=len(region_args),
            desc="Regions",
        ))

    # ── Assemble: one vstack + typed columns; no dict materialization ──
    print("  Concatenating region arrays...")
    region_ids_per_row = []
    meta_blocks = []
    feat_blocks = []
    target_blocks = [] if is_train else None
    for rid, meta, feats, tgt in results:
        n = meta.shape[0]
        region_ids_per_row.extend([rid] * n)
        meta_blocks.append(meta)
        feat_blocks.append(feats)
        if is_train:
            target_blocks.append(tgt)

    meta_all = np.vstack(meta_blocks)
    feat_all = np.vstack(feat_blocks)

    feat_cols = get_feature_columns()
    assert feat_all.shape[1] == len(feat_cols), (
        f"Feature column count mismatch: array={feat_all.shape[1]}, names={len(feat_cols)}"
    )

    print("  Building feature DataFrame...")
    result = pd.DataFrame(feat_all, columns=feat_cols, copy=False)
    result.insert(0, "region_id", region_ids_per_row)
    result.insert(1, "week_idx", meta_all[:, 0])
    result.insert(2, "month", meta_all[:, 1].astype(np.int8))
    result.insert(3, "day_of_year", meta_all[:, 2].astype(np.int16))

    if is_train:
        target_all = np.vstack(target_blocks)
        for h in range(1, 6):
            result[f"target_w{h}"] = target_all[:, h - 1]

    n_actual = result["region_id"].nunique()
    if is_train:
        expected_rows = n_actual * (TRAIN_WEEKS_PER_REGION - lag_window - 5)
        assert len(result) == expected_rows, (
            f"Expected {expected_rows} train feature rows, got {len(result)}"
        )
        assert result[[f"target_w{h}" for h in range(1, 6)]].notna().all().all(), \
            "NaN targets found — check anchor range"
    else:
        assert len(result) == n_actual, (
            f"Expected {n_actual} test rows, got {len(result)}"
        )

    # NaN can appear when np.nanmean is called on all-NaN slices; fill with 0.
    result[feat_cols] = result[feat_cols].fillna(0)
    print(f"  Feature matrix: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_feature_columns(feature_cols: list[str]):
    path = MODELS_DIR / "feature_cols.json"
    with open(path, "w") as f:
        json.dump(feature_cols, f)
    print(f"Feature columns saved to {path}")


def load_feature_columns() -> list[str]:
    path = MODELS_DIR / "feature_cols.json"
    with open(path) as f:
        return json.load(f)
