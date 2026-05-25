"""Phase 7 — climate / drought-physics features.

This module consolidates every "physical drought indicator" the model uses.
Each feature is climatology-standardized per (region, week-of-year) so it
encodes an *anomaly* (year-stable) rather than an absolute level
(year-volatile). That's the core of the Phase 7 bet: the val→Kaggle gap on
this dataset comes from cross-year covariate shift, and z-scoring against
per-(region, woy) climatology is the cheapest principled way to neutralize
it without per-instance normalization (which RevIN tried and failed).

Blocks:
  A1. Pressure variability — within-week 3-day pressure drops/climbs/std
      (computed in data_pipeline.py stage-1; this module just z-scores).
  A2. EDDI — standardized PET anomaly over {4, 8, 12} weeks.
  A3. SPEI — standardized water-balance anomaly over {4, 8, 12} weeks.
  A4. Composite events — co-occurrence indicators across surf_pre + humidity + tmp_max.
  A5. Dry-spell counts — consecutive dry weeks, dry-week counts.
  A6. Heat-stress counts — hot weeks above per-(region, woy) p90.
  A7. Wet-bulb depression — single lag-0 feature.

Public entry points:
  compute_climate_climatologies(weekly_df_train)   → dict of climatology DataFrames
  add_climate_features(features_df, weekly_df, climatologies) → features_df + ~32 new cols
  CLIMATE_FEATURE_COLS                              — list of all column names this module emits
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MODELS_DIR
from data_pipeline import PRESSURE_DERIVED_COLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRESSURE_LAG_RANGE = list(range(5))                 # lag 0..4
DROUGHT_INDEX_WINDOWS = (4, 8, 12)                  # EDDI / SPEI / dry-spell windows
COMPOSITE_LAG_RANGE = list(range(8))                # lag 0..7 for event aggregation
COMPOSITE_WINDOWS = (4, 8)                          # count/divergence windows
COMPOSITE_CHANNELS = ("surf_pre", "humidity", "tmp_max")

# Rule thresholds (z-score units) for the composite cross-channel event.
_THRESH_SURF_PRE = 1.0
_THRESH_HUMIDITY = -1.0
_THRESH_TMP_MAX = 0.5

# Dry-week threshold: weekly precipitation below this counts as a dry week.
_DRY_WEEK_PREC_MM = 7.0

# Water-balance constant — prec [mm/week] − k · PET_proxy. Scale-invariant for
# trees; only the ranking matters.
_WB_K = 0.16

PRESSURE_CLIMATOLOGY_PATH = MODELS_DIR / "climate_pressure_climatology.csv"
METEO_CLIMATOLOGY_PATH = MODELS_DIR / "climate_meteo_climatology.csv"
DROUGHT_INDEX_CLIMATOLOGY_PATH = MODELS_DIR / "climate_drought_index_climatology.csv"
HEAT_CLIMATOLOGY_PATH = MODELS_DIR / "climate_heat_climatology.csv"

# Per-(region, woy) z-scored meteo channels we need at multiple lags (for
# composite events) — the same set that goes into the per-channel climatology.
_METEO_CLIM_CHANNELS = ("surf_pre", "humidity", "tmp_max", "tmp", "tmp_range", "prec", "wb_tmp")


def woy_from_doy(doy: pd.Series | np.ndarray) -> np.ndarray:
    """Map day-of-year (1..365) to week-of-year (0..52)."""
    a = np.asarray(doy, dtype=np.int32)
    return np.clip((a - 1) // 7, 0, 52).astype(np.int16)


# ---------------------------------------------------------------------------
# Climatology computation (train-fold only)
# ---------------------------------------------------------------------------

def _pet_proxy(tmp_mean: np.ndarray, tmp_range_mean: np.ndarray) -> np.ndarray:
    """Modified Hargreaves PET (no latitude). Returns per-day PET proxy."""
    return (tmp_mean + 17.8) * np.sqrt(np.clip(tmp_range_mean, 0.0, None))


def compute_climate_climatologies(weekly_df_train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all climatology tables from the training-fold weekly rows.

    Returns:
      {
        'pressure': per (region, woy) mean/std of each PRESSURE_DERIVED_COL,
        'meteo':    per (region, woy) mean/std of each _METEO_CLIM_CHANNELS column (using _mean variant),
        'drought_index': per (region, woy, window) mean/std for SPI (precip) and SPEI (precip − k·PET),
        'heat':     per (region, woy) p90 of tmp_max (weekly mean of daily max),
      }
    """
    df = weekly_df_train.copy()
    df["woy"] = woy_from_doy(df["day_of_year"])

    # --- Pressure climatology (the daily-derived stats pass through from
    # data_pipeline.py stage-1).
    pressure_parts = []
    grouped = df.groupby(["region_id", "woy"])
    for col in PRESSURE_DERIVED_COLS:
        mean = grouped[col].mean().rename(f"{col}_clim_mean")
        std = grouped[col].std().fillna(1e-3).clip(lower=1e-3).rename(f"{col}_clim_std")
        pressure_parts.append(pd.concat([mean, std], axis=1))
    pressure_clim = pd.concat(pressure_parts, axis=1).reset_index()
    for c in pressure_clim.columns:
        if c not in ("region_id", "woy"):
            pressure_clim[c] = pressure_clim[c].astype(np.float32)

    # --- Meteo climatology (weekly means of the channels we z-score).
    meteo_parts = []
    for ch in _METEO_CLIM_CHANNELS:
        src = f"{ch}_mean"
        if src not in df.columns:
            continue
        mean = grouped[src].mean().rename(f"clim_{ch}_mean")
        std = grouped[src].std().fillna(1e-3).clip(lower=1e-3).rename(f"clim_{ch}_std")
        meteo_parts.append(pd.concat([mean, std], axis=1))
    meteo_clim = pd.concat(meteo_parts, axis=1).reset_index()
    for c in meteo_clim.columns:
        if c not in ("region_id", "woy"):
            meteo_clim[c] = meteo_clim[c].astype(np.float32)

    # --- Drought-index climatology (SPI on precip totals, SPEI on water-balance).
    # The N-week rolling sums are built per region from the chronological weekly
    # series of (prec_sum, pet_proxy), then grouped by anchor woy.
    df_sorted = df.sort_values(["region_id", "week_idx"]).reset_index(drop=True)
    if "prec_sum" not in df_sorted.columns:
        raise KeyError("weekly_df_train must contain prec_sum")
    if "tmp_mean" not in df_sorted.columns or "tmp_range_mean" not in df_sorted.columns:
        raise KeyError("weekly_df_train must contain tmp_mean and tmp_range_mean")

    pet_daily = _pet_proxy(
        df_sorted["tmp_mean"].to_numpy(np.float32),
        df_sorted["tmp_range_mean"].to_numpy(np.float32),
    )
    df_sorted["pet_weekly"] = pet_daily * 7.0
    df_sorted["wb_weekly"] = (
        df_sorted["prec_sum"].to_numpy(np.float32) - _WB_K * df_sorted["pet_weekly"].to_numpy(np.float32)
    )

    # Vectorized: groupby region, compute rolling sum per channel (pet, wb),
    # then groupby (region, woy) → mean/std. EDDI standardizes PET anomalies;
    # SPEI standardizes (precip − k·PET) water-balance anomalies. Both are
    # consumed by `_add_drought_index_features` via the `index` key.
    index_parts = []
    for series_col, ind_name in (("pet_weekly", "eddi"), ("wb_weekly", "spei")):
        grouped = df_sorted.groupby("region_id", sort=False)[series_col]
        for N in DROUGHT_INDEX_WINDOWS:
            roll_vals = (
                grouped.rolling(window=N, min_periods=N).sum()
                       .reset_index(level=0, drop=True)
                       .to_numpy(np.float32)
            )
            tmp = pd.DataFrame({
                "region_id": df_sorted["region_id"].to_numpy(),
                "woy": df_sorted["woy"].to_numpy(np.int16),
                "v": roll_vals,
            }).dropna(subset=["v"])
            agg = tmp.groupby(["region_id", "woy"])["v"].agg(["mean", "std"]).reset_index()
            agg["window"] = int(N)
            agg["index"] = ind_name
            agg = agg.rename(columns={"mean": "clim_mean", "std": "clim_std"})
            agg["clim_std"] = agg["clim_std"].fillna(1e-3).clip(lower=1e-3)
            index_parts.append(agg[["region_id", "woy", "window", "index", "clim_mean", "clim_std"]])
    drought_index_clim = pd.concat(index_parts, ignore_index=True) if index_parts else pd.DataFrame()
    for c in ("clim_mean", "clim_std"):
        if c in drought_index_clim.columns:
            drought_index_clim[c] = drought_index_clim[c].astype(np.float32)

    # --- Heat climatology (per-(region, woy) p90 of tmp_max weekly max).
    heat_clim = None
    if "tmp_max_max" in df.columns:
        heat_clim = (
            df.groupby(["region_id", "woy"])["tmp_max_max"]
              .quantile(0.90).reset_index()
              .rename(columns={"tmp_max_max": "tmp_max_p90"})
        )
        heat_clim["tmp_max_p90"] = heat_clim["tmp_max_p90"].astype(np.float32)

    return {
        "pressure": pressure_clim,
        "meteo": meteo_clim,
        "drought_index": drought_index_clim,
        "heat": heat_clim,
    }


# ---------------------------------------------------------------------------
# Lag-aware z-score lookups
# ---------------------------------------------------------------------------

def _lookup_weekly(
    weekly_indexed: pd.DataFrame,
    region_str: np.ndarray,
    week_idx: np.ndarray,
    lag: int,
    col: str,
    fallback: float,
) -> np.ndarray:
    """Return weekly_df[col] at (region, week_idx − lag), or fallback for missing."""
    if col not in weekly_indexed.columns:
        return np.full(len(region_str), fallback, dtype=np.float32)
    lookup = pd.MultiIndex.from_arrays([region_str, week_idx - lag])
    vals = weekly_indexed[col].reindex(lookup).to_numpy(np.float32)
    return np.where(np.isnan(vals), fallback, vals)


def _lookup_climatology(
    clim_indexed: pd.DataFrame,
    region_str: np.ndarray,
    woy_anchor: np.ndarray,
    lag: int,
    mean_col: str,
    std_col: str,
    fallback_mean: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (clim_mean, clim_std) at (region, (woy_anchor − lag) mod 53)."""
    woy_at_lag = (woy_anchor - lag) % 53
    lookup = pd.MultiIndex.from_arrays([region_str, woy_at_lag])
    mean = clim_indexed[mean_col].reindex(lookup).to_numpy(np.float32)
    std = clim_indexed[std_col].reindex(lookup).to_numpy(np.float32)
    mean = np.where(np.isnan(mean), fallback_mean, mean)
    std = np.where(np.isnan(std) | (std < 1e-3), 1.0, std)
    return mean, std


# ---------------------------------------------------------------------------
# Feature-block builders
# ---------------------------------------------------------------------------

def _add_pressure_features(
    new_cols: dict[str, np.ndarray],
    weekly_indexed: pd.DataFrame, clim_indexed: pd.DataFrame,
    region_str: np.ndarray, week_idx: np.ndarray, woy_anchor: np.ndarray,
) -> list[str]:
    """Block A1 — 3 pressure-derived stats × 5 lags = 15 z-scored features."""
    added = []
    for col in PRESSURE_DERIVED_COLS:
        global_mean = float(weekly_indexed[col].mean()) if col in weekly_indexed.columns else 0.0
        for lag in PRESSURE_LAG_RANGE:
            raw = _lookup_weekly(weekly_indexed, region_str, week_idx, lag, col, global_mean)
            mean, std = _lookup_climatology(
                clim_indexed, region_str, woy_anchor, lag,
                f"{col}_clim_mean", f"{col}_clim_std", global_mean,
            )
            name = f"z_{col}_lag{lag}"
            new_cols[name] = ((raw - mean) / std).astype(np.float32)
            added.append(name)
    return added


def _add_drought_index_features(
    new_cols: dict[str, np.ndarray],
    week_idx: np.ndarray,
    weekly_indexed: pd.DataFrame, drought_index_clim: pd.DataFrame,
    region_str: np.ndarray, woy_anchor: np.ndarray,
) -> list[str]:
    """Blocks A2 (EDDI) + A3 (SPEI). Computes N-week PET / water-balance sums
    and z-scores them per (region, anchor woy)."""
    added = []
    max_n = max(DROUGHT_INDEX_WINDOWS)
    n_rows = len(region_str)

    def _channel_at_lag(col: str, lag: int) -> np.ndarray:
        return _lookup_weekly(weekly_indexed, region_str, week_idx, lag, col, 0.0)

    pet_by_lag = np.zeros((n_rows, max_n), dtype=np.float32)
    wb_by_lag = np.zeros((n_rows, max_n), dtype=np.float32)
    for lag in range(max_n):
        tmp = _channel_at_lag("tmp_mean", lag)
        tmp_range = _channel_at_lag("tmp_range_mean", lag)
        prec = _channel_at_lag("prec_sum", lag)
        pet_weekly = _pet_proxy(tmp, tmp_range) * 7.0
        pet_by_lag[:, lag] = pet_weekly
        wb_by_lag[:, lag] = prec - _WB_K * pet_weekly

    clim_indexed = (
        drought_index_clim
            .assign(_r=lambda d: d["region_id"].astype(str),
                    _w=lambda d: d["woy"].astype(int))
    )

    for N in DROUGHT_INDEX_WINDOWS:
        eddi_sum = pet_by_lag[:, :N].sum(axis=1)
        spei_sum = wb_by_lag[:, :N].sum(axis=1)
        for ind_name, current in (("eddi", eddi_sum), ("spei", spei_sum)):
            sub = clim_indexed[(clim_indexed["index"] == ind_name) & (clim_indexed["window"] == N)]
            ci = sub.set_index(["_r", "_w"])[["clim_mean", "clim_std"]]
            lookup = pd.MultiIndex.from_arrays([region_str, woy_anchor.astype(int)])
            vals = ci.reindex(lookup)
            gm = float(sub["clim_mean"].mean())
            gs = float(max(sub["clim_std"].mean(), 1e-3))
            mean = vals["clim_mean"].fillna(gm).to_numpy(np.float32)
            std = vals["clim_std"].fillna(gs).clip(lower=1e-3).to_numpy(np.float32)
            name = f"{ind_name}_w{N}"
            new_cols[name] = ((current - mean) / std).astype(np.float32)
            added.append(name)
    return added


def _add_composite_event_features(
    new_cols: dict[str, np.ndarray],
    n_rows: int,
    weekly_indexed: pd.DataFrame, meteo_clim_indexed: pd.DataFrame,
    region_str: np.ndarray, week_idx: np.ndarray, woy_anchor: np.ndarray,
) -> list[str]:
    """Block A4 — cross-channel dry-anomaly events. For each lag k in 0..7,
    decide if the rule fires (z_surf_pre>1 ∧ z_humidity<−1 ∧ z_tmp_max>0.5)
    and compute z_surf_pre − z_humidity. Aggregate over {4w, 8w}."""
    added = []
    event_per_lag = np.zeros((n_rows, len(COMPOSITE_LAG_RANGE)), dtype=np.float32)
    div_per_lag = np.zeros((n_rows, len(COMPOSITE_LAG_RANGE)), dtype=np.float32)

    for li, lag in enumerate(COMPOSITE_LAG_RANGE):
        z_pre = _z_meteo(weekly_indexed, meteo_clim_indexed, region_str, week_idx, woy_anchor,
                         lag, "surf_pre")
        z_hum = _z_meteo(weekly_indexed, meteo_clim_indexed, region_str, week_idx, woy_anchor,
                         lag, "humidity")
        z_tmax = _z_meteo(weekly_indexed, meteo_clim_indexed, region_str, week_idx, woy_anchor,
                          lag, "tmp_max")
        event_per_lag[:, li] = (
            (z_pre > _THRESH_SURF_PRE)
            & (z_hum < _THRESH_HUMIDITY)
            & (z_tmax > _THRESH_TMP_MAX)
        ).astype(np.float32)
        div_per_lag[:, li] = (z_pre - z_hum).astype(np.float32)

    for W in COMPOSITE_WINDOWS:
        new_cols[f"dry_anomaly_event_weeks_{W}w"] = event_per_lag[:, :W].sum(axis=1).astype(np.float32)
        new_cols[f"pressure_humidity_divergence_{W}w"] = div_per_lag[:, :W].mean(axis=1).astype(np.float32)
        added.extend([f"dry_anomaly_event_weeks_{W}w", f"pressure_humidity_divergence_{W}w"])
    return added


def _z_meteo(weekly_indexed, meteo_clim_indexed, region_str, week_idx, woy_anchor, lag, channel):
    """Helper: z-score of `<channel>_mean` at (region, week_idx − lag)
    against climatology at (region, (woy_anchor − lag) mod 53)."""
    raw_col = f"{channel}_mean"
    gm = float(weekly_indexed[raw_col].mean()) if raw_col in weekly_indexed.columns else 0.0
    raw = _lookup_weekly(weekly_indexed, region_str, week_idx, lag, raw_col, gm)
    mean, std = _lookup_climatology(
        meteo_clim_indexed, region_str, woy_anchor, lag,
        f"clim_{channel}_mean", f"clim_{channel}_std", gm,
    )
    return (raw - mean) / std


def _add_dry_spell_features(
    new_cols: dict[str, np.ndarray],
    n_rows: int,
    weekly_indexed: pd.DataFrame,
    region_str: np.ndarray, week_idx: np.ndarray,
) -> list[str]:
    """Block A5 — climatology-free dry-spell counts."""
    added = []
    lags = list(range(1, max(DROUGHT_INDEX_WINDOWS) + 1))
    dry_mat = np.zeros((n_rows, len(lags)), dtype=np.float32)
    for i, lag in enumerate(lags):
        prec = _lookup_weekly(weekly_indexed, region_str, week_idx, lag, "prec_sum", 0.0)
        dry_mat[:, i] = (prec < _DRY_WEEK_PREC_MM).astype(np.float32)

    for W in DROUGHT_INDEX_WINDOWS:
        col = f"dry_week_count_past_{W}w"
        new_cols[col] = dry_mat[:, :W].sum(axis=1).astype(np.float32)
        added.append(col)

    longest = np.zeros(n_rows, dtype=np.float32)
    current = np.zeros(n_rows, dtype=np.float32)
    for i in range(dry_mat.shape[1]):
        current = (current + 1.0) * dry_mat[:, i]
        longest = np.maximum(longest, current)
    new_cols["consec_dry_weeks_past_12w"] = longest
    added.append("consec_dry_weeks_past_12w")
    return added


def _add_heat_features(
    new_cols: dict[str, np.ndarray],
    n_rows: int,
    weekly_indexed: pd.DataFrame, heat_clim: pd.DataFrame | None,
    region_str: np.ndarray, week_idx: np.ndarray, woy_anchor: np.ndarray,
) -> list[str]:
    """Block A6 — count of weeks above per-(region, woy) tmp_max p90."""
    if heat_clim is None or "tmp_max_max" not in weekly_indexed.columns:
        return []
    added = []
    clim_idx = (
        heat_clim
            .assign(_r=lambda d: d["region_id"].astype(str), _w=lambda d: d["woy"].astype(int))
            .set_index(["_r", "_w"])["tmp_max_p90"]
    )
    lookup = pd.MultiIndex.from_arrays([region_str, woy_anchor.astype(int)])
    p90 = clim_idx.reindex(lookup).fillna(heat_clim["tmp_max_p90"].mean()).to_numpy(np.float32)

    for W in (4, 8):
        count = np.zeros(n_rows, dtype=np.float32)
        for lag in range(1, W + 1):
            tmax = _lookup_weekly(weekly_indexed, region_str, week_idx, lag, "tmp_max_max", -1e9)
            count += (tmax > p90).astype(np.float32)
        col = f"hot_week_count_past_{W}w"
        new_cols[col] = count
        added.append(col)
    return added


def _add_wbd_lag0(
    new_cols: dict[str, np.ndarray],
    weekly_indexed: pd.DataFrame,
    region_str: np.ndarray, week_idx: np.ndarray,
) -> list[str]:
    """Block A7 — wet-bulb depression at lag 0 only."""
    if "tmp_mean" not in weekly_indexed.columns or "wb_tmp_mean" not in weekly_indexed.columns:
        return []
    tmp = _lookup_weekly(weekly_indexed, region_str, week_idx, 0, "tmp_mean", 0.0)
    wb = _lookup_weekly(weekly_indexed, region_str, week_idx, 0, "wb_tmp_mean", 0.0)
    new_cols["wbd_lag0"] = (tmp - wb).astype(np.float32)
    return ["wbd_lag0"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def add_climate_features(
    features_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    climatologies: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """Apply every climate-feature block. Helpers populate one shared dict;
    a single pd.concat at the end appends everything to features_df without
    fragmenting the underlying block manager."""
    out = features_df.copy()
    n_rows = len(out)
    region_str = out["region_id"].astype(str).to_numpy()
    week_idx = out["week_idx"].astype(int).to_numpy()
    woy_anchor = woy_from_doy(out["day_of_year"]).astype(np.int32)

    weekly_indexed = (
        weekly_df
            .assign(_r=lambda d: d["region_id"].astype(str),
                    _w=lambda d: d["week_idx"].astype(int))
            .set_index(["_r", "_w"])
    )
    pressure_clim_indexed = (
        climatologies["pressure"]
            .assign(_r=lambda d: d["region_id"].astype(str),
                    _w=lambda d: d["woy"].astype(int))
            .set_index(["_r", "_w"])
    )
    meteo_clim_indexed = (
        climatologies["meteo"]
            .assign(_r=lambda d: d["region_id"].astype(str),
                    _w=lambda d: d["woy"].astype(int))
            .set_index(["_r", "_w"])
    )

    new_cols: dict[str, np.ndarray] = {}
    added: list[str] = []
    added += _add_pressure_features(new_cols, weekly_indexed, pressure_clim_indexed,
                                    region_str, week_idx, woy_anchor)
    added += _add_drought_index_features(new_cols, week_idx, weekly_indexed,
                                         climatologies["drought_index"],
                                         region_str, woy_anchor)
    added += _add_composite_event_features(new_cols, n_rows, weekly_indexed, meteo_clim_indexed,
                                           region_str, week_idx, woy_anchor)
    added += _add_dry_spell_features(new_cols, n_rows, weekly_indexed, region_str, week_idx)
    added += _add_heat_features(new_cols, n_rows, weekly_indexed, climatologies.get("heat"),
                                region_str, week_idx, woy_anchor)
    added += _add_wbd_lag0(new_cols, weekly_indexed, region_str, week_idx)

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    return out, added


# Static column list — for cache-key hashing and analysis.
CLIMATE_FEATURE_COLS: list[str] = (
    [f"z_{col}_lag{lag}" for col in PRESSURE_DERIVED_COLS for lag in PRESSURE_LAG_RANGE]
    + [f"eddi_w{N}" for N in DROUGHT_INDEX_WINDOWS]
    + [f"spei_w{N}" for N in DROUGHT_INDEX_WINDOWS]
    + [f"dry_anomaly_event_weeks_{W}w" for W in COMPOSITE_WINDOWS]
    + [f"pressure_humidity_divergence_{W}w" for W in COMPOSITE_WINDOWS]
    + [f"dry_week_count_past_{W}w" for W in DROUGHT_INDEX_WINDOWS]
    + ["consec_dry_weeks_past_12w"]
    + [f"hot_week_count_past_{W}w" for W in (4, 8)]
    + ["wbd_lag0"]
)
# 15 + 3 + 3 + 4 + 3 + 1 + 2 + 1 = 32 climate features
