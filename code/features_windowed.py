"""Phase 7 — sliding-window temporal stats.

Replaces the legacy 13-position weekly lag block with multi-window temporal
summaries. For each meteo channel and each window N ∈ {4, 8, 12} weeks, five
stats over the rolling window ending at the anchor:

  mean              — smoothed level (rolling mean)
  std               — within-window variability
  trend             — OLS slope across the window (positive = rising)
  ewm               — exponentially weighted mean (α=0.5; recent-heavy)
  recent_minus_past — (mean of last 2 weeks) − (mean of remaining weeks)

Each stat is climatology-standardized per (region, week-of-year) so the
feature encodes an anomaly rather than an absolute level. Implementation uses
1-D convolutions per (channel, window) — fully vectorized; the full build over
1.7M training rows runs in seconds, not hours.

Public:
  compute_windowed_climatology(weekly_df_train, channels, windows)
      → long DataFrame with one row per (region_id, woy, channel, stat, window)
        and columns clim_mean / clim_std.

  add_windowed_features(features_df, weekly_df, climatology, channels, windows)
      → (extended_df, added_cols) — adds z_<channel>_<stat>_w<N>.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MODELS_DIR

WINDOWED_CHANNELS_DEFAULT = ("prec", "surf_pre", "tmp", "tmp_range", "humidity", "wb_tmp", "tmp_max")
WINDOWED_WINDOWS_DEFAULT = (4, 8, 12)
WINDOWED_STATS = ("mean", "std", "trend", "ewm", "recent_minus_past")
WINDOWED_CLIMATOLOGY_PATH = MODELS_DIR / "windowed_climatology.csv"
_EWM_ALPHA = 0.5


def _woy_from_doy(doy) -> np.ndarray:
    a = np.asarray(doy, dtype=np.int32)
    return np.clip((a - 1) // 7, 0, 52).astype(np.int16)


# ---------------------------------------------------------------------------
# Vectorized per-stat convolutions
# ---------------------------------------------------------------------------

def _causal_rolling_mean(series: np.ndarray, N: int) -> np.ndarray:
    """Rolling mean of last N values ending at each position. Same length as input."""
    if N <= 1:
        return series.copy()
    return pd.Series(series).rolling(N, min_periods=1).mean().to_numpy(np.float32)


def _causal_rolling_std(series: np.ndarray, N: int) -> np.ndarray:
    if N <= 1:
        return np.zeros_like(series, dtype=np.float32)
    return pd.Series(series).rolling(N, min_periods=2).std().fillna(0.0).to_numpy(np.float32)


def _trend_weights(N: int) -> np.ndarray:
    """OLS slope weights so that slope_t = sum_i w[i] · y[t-N+1+i].
    Closed form: w_i = (i − (N−1)/2) / sum_j ((j − (N−1)/2)^2)."""
    if N < 2:
        return np.zeros(max(1, N), dtype=np.float32)
    i = np.arange(N, dtype=np.float32)
    centered = i - (N - 1) / 2.0
    denom = (centered ** 2).sum()
    return (centered / max(denom, 1e-9)).astype(np.float32)


def _ewm_weights(N: int, alpha: float) -> np.ndarray:
    """Exponentially weighted mean: more weight on most-recent values within the
    window. Returns weights of length N such that w[N-1] is the largest."""
    if N < 1:
        return np.array([1.0], dtype=np.float32)
    i = np.arange(N, dtype=np.float32)
    raw = alpha * (1.0 - alpha) ** (N - 1 - i)
    return (raw / raw.sum()).astype(np.float32)


def _recent_minus_past_weights(N: int) -> np.ndarray:
    """(mean of last 2 weeks) − (mean of remaining N−2 weeks)."""
    if N < 3:
        # Degenerate: just diff between the 2 most recent values.
        w = np.zeros(max(2, N), dtype=np.float32)
        w[-1] = 0.5
        w[-2] = 0.5
        return w
    w = np.full(N, -1.0 / (N - 2), dtype=np.float32)
    w[-1] = 0.5
    w[-2] = 0.5
    return w


def _causal_conv1d(series: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Causal 1D convolution: out[t] = sum_i weights[i] · series[t - N + 1 + i].
    For t < N-1 (insufficient history), fall back to a partial-window value
    using only the available samples (re-weighted to sum to 1 in magnitude).
    Returns array of same length as series.
    """
    N = len(weights)
    n = len(series)
    out = np.zeros(n, dtype=np.float32)
    if N == 0 or n == 0:
        return out
    # Full-window region: use np.convolve for vectorized speed.
    # np.convolve(a, w[::-1], mode='valid') gives the correlation w[i]·a[t-N+1+i] for t = N-1 .. n-1.
    full = np.convolve(series.astype(np.float32), weights[::-1], mode="valid")
    if N - 1 < n:
        out[N - 1:] = full
    # Partial-window region: for t in 0..N-2, use the truncated window.
    # Re-weight the available weights so they still represent a comparable stat.
    for t in range(min(N - 1, n)):
        avail = t + 1
        sub_w = weights[-avail:]
        sub_s = series[: avail]
        # For mean-like stats, re-normalize. For trend/ewm/rmp, use as-is (slope
        # at the start of the series is undefined; this is a graceful fallback).
        s = float(sub_w.sum())
        if abs(s) > 1e-6:
            out[t] = float((sub_w * sub_s).sum() / s)
        else:
            out[t] = float((sub_w * sub_s).sum())
    return out


def compute_window_stats_for_series(
    series_matrix: np.ndarray,        # (n_weeks, n_channels)
    windows: tuple[int, ...],
) -> dict[tuple[int, str], np.ndarray]:
    """For a single region's chronologically-sorted weekly series, compute every
    (window, stat) tensor of shape (n_weeks, n_channels)."""
    n_weeks, n_ch = series_matrix.shape
    out: dict[tuple[int, str], np.ndarray] = {}
    for N in windows:
        mean_mat = np.zeros((n_weeks, n_ch), dtype=np.float32)
        std_mat = np.zeros((n_weeks, n_ch), dtype=np.float32)
        trend_mat = np.zeros((n_weeks, n_ch), dtype=np.float32)
        ewm_mat = np.zeros((n_weeks, n_ch), dtype=np.float32)
        rmp_mat = np.zeros((n_weeks, n_ch), dtype=np.float32)
        trend_w = _trend_weights(N)
        ewm_w = _ewm_weights(N, _EWM_ALPHA)
        rmp_w = _recent_minus_past_weights(N)
        for ci in range(n_ch):
            col = series_matrix[:, ci].astype(np.float32, copy=False)
            mean_mat[:, ci] = _causal_rolling_mean(col, N)
            std_mat[:, ci] = _causal_rolling_std(col, N)
            trend_mat[:, ci] = _causal_conv1d(col, trend_w)
            ewm_mat[:, ci] = _causal_conv1d(col, ewm_w)
            rmp_mat[:, ci] = _causal_conv1d(col, rmp_w)
        out[(N, "mean")] = mean_mat
        out[(N, "std")] = std_mat
        out[(N, "trend")] = trend_mat
        out[(N, "ewm")] = ewm_mat
        out[(N, "recent_minus_past")] = rmp_mat
    return out


# ---------------------------------------------------------------------------
# Packed-matrix helpers
# ---------------------------------------------------------------------------

def _packed_columns(
    channels: tuple[str, ...], windows: tuple[int, ...]
) -> list[tuple[str, str, int]]:
    """Order = channel outer, stat middle, window inner. Same ordering used by
    `windowed_feature_cols()` and the packed matrix."""
    return [(ch, stat, int(N)) for ch in channels for stat in WINDOWED_STATS for N in windows]


def windowed_feature_cols(
    channels: tuple[str, ...] = WINDOWED_CHANNELS_DEFAULT,
    windows: tuple[int, ...] = WINDOWED_WINDOWS_DEFAULT,
) -> list[str]:
    return [f"z_{ch}_{stat}_w{N}" for ch, stat, N in _packed_columns(channels, windows)]


def _pack_region(
    sub_sorted: pd.DataFrame,
    channels: tuple[str, ...],
    windows: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """For one region's chronologically sorted weekly rows, return:
       packed : (n_weeks, n_packed_cols) where columns follow _packed_columns().
       woys   : (n_weeks,) week-of-year per row.
       wi_to_row : dict mapping week_idx → row position.
    """
    needed = [f"{ch}_mean" for ch in channels]
    series = sub_sorted[needed].to_numpy(np.float32)
    n_weeks = series.shape[0]
    n_ch = len(channels)
    stats_by_key = compute_window_stats_for_series(series, windows)

    packed_cols = _packed_columns(channels, windows)
    packed = np.zeros((n_weeks, len(packed_cols)), dtype=np.float32)
    for j, (ch, stat, N) in enumerate(packed_cols):
        ci = channels.index(ch)
        packed[:, j] = stats_by_key[(N, stat)][:, ci]

    woys = _woy_from_doy(sub_sorted["day_of_year"].to_numpy())
    wi_to_row = {int(w): int(i) for i, w in enumerate(sub_sorted["week_idx"].to_numpy())}
    return packed, woys, wi_to_row


# ---------------------------------------------------------------------------
# Climatology computation
# ---------------------------------------------------------------------------

def compute_windowed_climatology(
    weekly_df_train: pd.DataFrame,
    channels: tuple[str, ...] = WINDOWED_CHANNELS_DEFAULT,
    windows: tuple[int, ...] = WINDOWED_WINDOWS_DEFAULT,
) -> pd.DataFrame:
    """For every (region, woy, channel, stat, window) bucket, mean and std of
    the windowed stat across the training fold."""
    packed_cols = _packed_columns(channels, windows)
    n_packed = len(packed_cols)
    weekly_df_sorted = weekly_df_train.sort_values(["region_id", "week_idx"]).reset_index(drop=True)

    # Stream per region into long-form (region, woy, packed[:]) → groupby.
    long_chunks = []
    for rid, sub in weekly_df_sorted.groupby("region_id", sort=False):
        sub = sub.sort_values("week_idx").reset_index(drop=True)
        packed, woys, _ = _pack_region(sub, channels, windows)
        df = pd.DataFrame(packed, columns=[f"_p{j}" for j in range(n_packed)])
        df.insert(0, "woy", woys.astype(np.int16))
        df.insert(0, "region_id", rid)
        long_chunks.append(df)
    long_df = pd.concat(long_chunks, ignore_index=True)

    # GroupBy → per-bucket mean/std across all packed columns
    agg = long_df.groupby(["region_id", "woy"]).agg(["mean", "std"])
    # Reshape: result has columns MultiIndex (_p{j}, mean) / (_p{j}, std). Flatten.
    out_rows = []
    means = agg.xs("mean", axis=1, level=1)
    stds = agg.xs("std", axis=1, level=1).fillna(1e-3).clip(lower=1e-3)
    for j, (ch, stat, N) in enumerate(packed_cols):
        col = f"_p{j}"
        sub = pd.DataFrame({
            "region_id": means.index.get_level_values(0),
            "woy": means.index.get_level_values(1),
            "channel": ch,
            "stat": stat,
            "window": int(N),
            "clim_mean": means[col].to_numpy(np.float32),
            "clim_std": stds[col].to_numpy(np.float32),
        })
        out_rows.append(sub)
    return pd.concat(out_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Apply features to anchor rows
# ---------------------------------------------------------------------------

def add_windowed_features(
    features_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    climatology: pd.DataFrame,
    channels: tuple[str, ...] = WINDOWED_CHANNELS_DEFAULT,
    windows: tuple[int, ...] = WINDOWED_WINDOWS_DEFAULT,
) -> tuple[pd.DataFrame, list[str]]:
    """Add z-scored windowed-stat features to anchor rows."""
    packed_cols = _packed_columns(channels, windows)
    n_packed = len(packed_cols)
    out = features_df.copy()

    weekly_df_sorted = weekly_df.sort_values(["region_id", "week_idx"]).reset_index(drop=True)

    # 1) Compute per-region packed matrices ONCE, in a dict.
    region_packed: dict[str, np.ndarray] = {}
    region_wi_to_row: dict[str, dict[int, int]] = {}
    for rid, sub in weekly_df_sorted.groupby("region_id", sort=False):
        sub = sub.sort_values("week_idx").reset_index(drop=True)
        packed, _, wi_to_row = _pack_region(sub, channels, windows)
        region_packed[rid] = packed
        region_wi_to_row[rid] = wi_to_row

    # 2) For every anchor row, gather all 105 windowed values.
    region_str = out["region_id"].astype(str).to_numpy()
    week_idx = out["week_idx"].astype(int).to_numpy()
    woy_anchor = _woy_from_doy(out["day_of_year"]).astype(np.int32)
    n_rows = len(out)
    raw_values = np.zeros((n_rows, n_packed), dtype=np.float32)
    # Group rows by region_id for batched lookup.
    df_idx = pd.DataFrame({"row_id": np.arange(n_rows), "rid": region_str, "wi": week_idx})
    for rid, sub in df_idx.groupby("rid", sort=False):
        if rid not in region_packed:
            continue
        packed = region_packed[rid]
        wi_to_row = region_wi_to_row[rid]
        rows_in_sub = sub["row_id"].to_numpy()
        wi_in_sub = sub["wi"].to_numpy()
        # Map each anchor's week_idx to the corresponding row in `packed`. Default 0.
        idxs = np.array([wi_to_row.get(int(w), -1) for w in wi_in_sub], dtype=np.int32)
        valid = idxs >= 0
        if valid.any():
            raw_values[rows_in_sub[valid]] = packed[idxs[valid]]

    # 3) Vectorized z-score using the climatology DataFrame. Collect new
    #    columns into a dict and concat once to avoid DataFrame fragmentation.
    clim_indexed = climatology.set_index(["region_id", "woy", "channel", "stat", "window"])
    new_cols: dict[str, np.ndarray] = {}
    added: list[str] = []
    for j, (ch, stat, N) in enumerate(packed_cols):
        idx = pd.MultiIndex.from_arrays([
            region_str,
            woy_anchor.astype(int),
            np.full(n_rows, ch),
            np.full(n_rows, stat),
            np.full(n_rows, int(N), dtype=int),
        ])
        clim_rows = clim_indexed.reindex(idx)
        gm = float(clim_rows["clim_mean"].dropna().mean()) if clim_rows["clim_mean"].notna().any() else 0.0
        gs = float(clim_rows["clim_std"].dropna().mean()) if clim_rows["clim_std"].notna().any() else 1.0
        mean = clim_rows["clim_mean"].fillna(gm).to_numpy(np.float32)
        std = clim_rows["clim_std"].fillna(gs).clip(lower=1e-3).to_numpy(np.float32)
        z = ((raw_values[:, j] - mean) / std).astype(np.float32)
        name = f"z_{ch}_{stat}_w{N}"
        new_cols[name] = z
        added.append(name)
    out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    return out, added
