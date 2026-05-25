"""Dataset and windowing for the PatchTST track — daily-resolution refactor.

Phase 4 redesign:
  - Per-region daily feature matrices (n_days, n_channels) instead of weekly
    stat matrices (n_weeks, 70).
  - Anchor at the last day of each scored week in train; one anchor per
    region at the last day of test in inference.
  - Score-lag side inputs computed per anchor with the Phase 1b staleness
    shift (offsets = base_offset + 12 weeks). Test anchors reuse the LGBM
    track's region_last_scores.csv for parity.
  - Calendar-matched train/val split: hold out per region the anchors whose
    woy is within ±CALENDAR_SLACK_WEEKS of that region's test_end_woy,
    restricted to the last training year.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))  # let `import config` resolve
from config import (
    TRAIN_PATH, TEST_PATH, METEO_FEATURES, MODELS_DIR,
    TRAIN_WEEKS_PER_REGION, TEST_WEEKS_PER_REGION,
)
from validate import compute_test_anchor_woy_per_region

from patchtst.config_pt import (
    LOOKBACK_DAYS, PATCH_LEN_DAYS, PATCH_STRIDE_DAYS, N_CHANNELS,
    HORIZONS, SCORE_LAG_OFFSETS, SCORE_LAG_TEST_GAP_SHIFT,
    CALENDAR_SLACK_WEEKS, CALENDAR_LAST_YEAR_ONLY,
    USE_CALENDAR_MATCHED_VAL,
    MAJORITY_KEEP_FRAC, SAMPLE_WEIGHT_ALPHA, SAMPLE_WEIGHT_BETA,
    PT_NORM_PATH, PT_REGION_MAP_PATH, PT_TRAIN_PKL,
)

WEEK_DAYS = 7
N_SCORE_LAG_FEATURES = len(SCORE_LAG_OFFSETS)
N_EXTRA_SCORE_FEATURES = 2  # weeks_since_last_nonzero, max_score_prev8
N_CALENDAR_FEATURES = 4     # month_sin, month_cos, doy_sin, doy_cos


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_MONTH_CUM = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def _month_doy_from_str(date_str: str) -> tuple[int, int]:
    """'YYYY-MM-DD' -> (month, doy). Avoids pd.to_datetime overflow on
    synthetic future-year dates like 39181-10-08."""
    _, mm, dd = date_str.split("-")
    m = int(mm)
    return m, _MONTH_CUM[m - 1] + int(dd)


def _woy_from_doy(doy: int) -> int:
    return min(52, max(0, (int(doy) - 1) // 7))


# ---------------------------------------------------------------------------
# Per-region daily arrays
# ---------------------------------------------------------------------------

def build_region_daily_arrays(
    daily_df: pd.DataFrame,
    is_train: bool,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Phase 13: build per-region WEEKLY aggregated float32 arrays.

    Variable names retained as `daily_*` for code-compat — but each row is
    now a 7-day mean. Each region's series is trimmed to whole weeks; the
    anchor doy stored is the doy of each week's LAST day.

    Returns:
        region_ids:    list[str] in encounter order
        daily_feats:   list[np.ndarray (n_weeks, N_CHANNELS)] float32 — weekly mean meteo
        daily_doys:    list[np.ndarray (n_weeks,)] int16 — doy of each week's LAST day
        weekly_scores: list[np.ndarray (n_weeks,)] int8 — per-region scored-week scores
                       (only filled in train; test returns empty arrays)
    """
    daily_df = daily_df.sort_values(["region_id", "date"]).reset_index(drop=True)
    region_ids: list[str] = []
    daily_feats: list[np.ndarray] = []
    daily_doys: list[np.ndarray] = []
    weekly_scores: list[np.ndarray] = []

    surf_pre_idx = METEO_FEATURES.index("surf_pre")
    prec_idx = METEO_FEATURES.index("prec")
    tmp_max_idx = METEO_FEATURES.index("tmp_max")
    PREC_DRY_THRESH = 1.0   # mm/day; matches LGBM dry-day logic

    for rid, sub in daily_df.groupby("region_id", sort=False):
        sub = sub.reset_index(drop=True)
        n_days = len(sub)
        n_weeks = n_days // WEEK_DAYS
        if n_weeks == 0:
            continue
        sub = sub.iloc[: n_weeks * WEEK_DAYS]
        region_ids.append(rid)

        # Daily channels reshaped to (n_weeks, 7 days, 9 meteo cols)
        feats_daily = sub[METEO_FEATURES].values.astype(np.float32)
        feats_dw = feats_daily.reshape(n_weeks, WEEK_DAYS, -1)

        # Channels 0..8: weekly mean of each meteo channel
        weekly_mean = feats_dw.mean(axis=1)                     # (n_weeks, 9)

        # Surf_pre within-week dynamics (channels 9..11)
        sp = feats_dw[:, :, surf_pre_idx]                       # (n_weeks, 7)
        sp_diffs_3d = sp[:, 3:] - sp[:, :-3]                    # (n_weeks, 4)
        surf_pre_drop = np.maximum(-sp_diffs_3d.min(axis=1), 0).astype(np.float32)
        surf_pre_climb = np.maximum(sp_diffs_3d.max(axis=1), 0).astype(np.float32)
        surf_pre_std = sp.std(axis=1).astype(np.float32)

        # Prec extras (channels 12..13)
        prec = feats_dw[:, :, prec_idx]
        prec_max = prec.max(axis=1).astype(np.float32)
        prec_dry_count = (prec < PREC_DRY_THRESH).sum(axis=1).astype(np.float32)

        # Tmp_max peak (channel 14)
        tmp_max_peak = feats_dw[:, :, tmp_max_idx].max(axis=1).astype(np.float32)

        feats_weekly = np.column_stack([
            weekly_mean,                # 9
            surf_pre_drop[:, None],     # +1
            surf_pre_climb[:, None],    # +1
            surf_pre_std[:, None],      # +1
            prec_max[:, None],          # +1
            prec_dry_count[:, None],    # +1
            tmp_max_peak[:, None],      # +1
        ]).astype(np.float32)           # (n_weeks, 15)
        daily_feats.append(feats_weekly)

        # Anchor doy: doy of the LAST day of each week
        doys_daily = np.array(
            [_month_doy_from_str(d)[1] for d in sub["date"].values[WEEK_DAYS - 1::WEEK_DAYS]],
            dtype=np.int16,
        )
        daily_doys.append(doys_daily)

        if is_train:
            sc = sub["score"].values
            mask = ~pd.isna(sc)
            weekly_scores.append(sc[mask].astype(np.int8))
        else:
            weekly_scores.append(np.zeros(0, dtype=np.int8))

    return region_ids, daily_feats, daily_doys, weekly_scores


# ---------------------------------------------------------------------------
# Anchor day enumeration (training only)
# ---------------------------------------------------------------------------

def enumerate_anchor_days(
    daily_df_per_region_sizes: list[int],
    daily_doys: list[np.ndarray],
    is_train: bool = True,
) -> list[np.ndarray]:
    """Phase 13: anchors are now WEEK indices (one per element of the weekly
    aggregated series). Train: every week index 0..n_weeks-1. Test: the last
    week index per region.

    `daily_df_per_region_sizes` is now the WEEKLY length per region.
    """
    out = []
    for n_weeks, _doys in zip(daily_df_per_region_sizes, daily_doys):
        if is_train:
            anchors = np.arange(n_weeks, dtype=np.int32)
        else:
            anchors = np.array([n_weeks - 1], dtype=np.int32)
        out.append(anchors)
    return out


# ---------------------------------------------------------------------------
# Score-lag side inputs (staleness-shifted, Phase 1b)
# ---------------------------------------------------------------------------

def compute_score_lag_side_inputs_train(
    weekly_scores: list[np.ndarray],
    shift_weeks: int = SCORE_LAG_TEST_GAP_SHIFT,
) -> list[np.ndarray]:
    """For each region, return (n_anchors, N_SCORE_LAG + 2) float32 array of
    side inputs: [score_lag_offsets..., weeks_since_last_nonzero, max_score_prev8].

    All lookups use index `w - k - shift` to match test's information
    staleness: test's score_lag1 source is the last training week, 14 weeks
    before its first prediction target. Train's shift=12 (= TEST_WEEKS_PER_REGION-1)
    makes train's (lag-source → target_w1) distance also 14.

    Phase 12: when USE_SCORE_LAG_SIDE_INPUTS=False, return zero arrays of the
    same shape — keeps the model architecture unchanged while neutralizing
    the phantom-anchor feature (the train↔test gap of ~163 weeks makes the
    lag autocorrelation effectively zero at test time).
    """
    from patchtst.config_pt import USE_SCORE_LAG_SIDE_INPUTS
    out = []
    for scores in weekly_scores:
        n = len(scores)
        feats = np.zeros((n, N_SCORE_LAG_FEATURES + N_EXTRA_SCORE_FEATURES),
                         dtype=np.float32)
        if not USE_SCORE_LAG_SIDE_INPUTS:
            out.append(feats)
            continue
        for w in range(n):
            for j, k in enumerate(SCORE_LAG_OFFSETS):
                idx = w - k - shift_weeks
                feats[w, j] = scores[idx] if 0 <= idx < n else 0.0

            # weeks_since_last_nonzero: scan back from w-1-shift
            last_nonzero = 26
            for back in range(1, 27):
                idx = w - back - shift_weeks
                if idx < 0:
                    break
                if scores[idx] > 0:
                    last_nonzero = back
                    break
            feats[w, N_SCORE_LAG_FEATURES] = float(last_nonzero)

            # max_score_prev8: window [w-8-shift, w-1-shift]
            lo = max(0, w - 8 - shift_weeks)
            hi = max(0, w - shift_weeks)
            feats[w, N_SCORE_LAG_FEATURES + 1] = (
                float(scores[lo:hi].max()) if lo < hi else 0.0
            )
        out.append(feats)
    return out


def load_score_lag_side_inputs_test() -> dict[str, np.ndarray]:
    """Test anchors reuse the LGBM track's pre-saved region_last_scores.csv,
    which contains the same score_lag features (no staleness shift needed —
    test's information state is naturally at lag(last_train_week)=14-weeks-before-target).

    Returns {region_id: (N_SCORE_LAG + 2,) float32 array}.

    Phase 12: when USE_SCORE_LAG_SIDE_INPUTS=False, return zero arrays so the
    model architecture is unchanged but the phantom-anchor signal is neutralized.
    """
    from patchtst.config_pt import USE_SCORE_LAG_SIDE_INPUTS
    zero_vec = np.zeros(N_SCORE_LAG_FEATURES + N_EXTRA_SCORE_FEATURES, dtype=np.float32)
    if not USE_SCORE_LAG_SIDE_INPUTS:
        # Read region_id list from anywhere available; the dataset.csv covers all regions.
        import pandas as pd
        path = MODELS_DIR / "region_last_scores.csv"
        if path.is_file():
            df = pd.read_csv(path, dtype={"region_id": str})
            return {row["region_id"]: zero_vec.copy() for _, row in df.iterrows()}
        # Last-ditch fallback: empty dict; caller will use .get() with default
        return {}
    path = MODELS_DIR / "region_last_scores.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run LGBM train.py first to generate it, OR set "
            "USE_SCORE_LAG_SIDE_INPUTS=False in patchtst/config_pt.py."
        )
    df = pd.read_csv(path, dtype={"region_id": str})
    lag_cols = [f"score_lag{k}" for k in SCORE_LAG_OFFSETS]
    extra_cols = ["weeks_since_last_nonzero", "max_score_prev8"]
    expected_cols = lag_cols + extra_cols
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"region_last_scores.csv missing cols: {missing}")
    out = {}
    for _, row in df.iterrows():
        rid = row["region_id"]
        out[rid] = row[expected_cols].values.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Calendar side inputs per anchor
# ---------------------------------------------------------------------------

def compute_calendar_side_inputs(anchor_doys: np.ndarray) -> np.ndarray:
    """(n_anchors,) doys -> (n_anchors, 4) [month_sin, month_cos, doy_sin, doy_cos]."""
    n = len(anchor_doys)
    feats = np.zeros((n, N_CALENDAR_FEATURES), dtype=np.float32)
    # Approximate month from doy (1..365)
    for i, doy in enumerate(anchor_doys):
        # Find month: largest m where _MONTH_CUM[m] < doy
        m = 1
        while m < 12 and _MONTH_CUM[m] < int(doy):
            m += 1
        feats[i, 0] = np.sin(2 * np.pi * m / 12)
        feats[i, 1] = np.cos(2 * np.pi * m / 12)
        feats[i, 2] = np.sin(2 * np.pi * int(doy) / 365)
        feats[i, 3] = np.cos(2 * np.pi * int(doy) / 365)
    return feats


# ---------------------------------------------------------------------------
# Calendar-matched train/val split
# ---------------------------------------------------------------------------

def split_train_val_anchors(
    region_ids: list[str],
    daily_doys: list[np.ndarray],
    anchor_days: list[np.ndarray],
    test_woy_per_region: dict[str, int],
    slack_weeks: int = CALENDAR_SLACK_WEEKS,
    last_year_only: bool = CALENDAR_LAST_YEAR_ONLY,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (region_idx, anchor_day_idx) arrays for train and val.

    Val per region: anchors whose woy is within ±slack_weeks of the region's
    test_end_woy AND (optionally) in the last 52 weeks of training.
    """
    train_anchors: list[tuple[int, int]] = []
    val_anchors: list[tuple[int, int]] = []

    for r_idx, (rid, doys, anchors) in enumerate(zip(region_ids, daily_doys, anchor_days)):
        test_woy = test_woy_per_region.get(rid, -1)
        if test_woy < 0:
            # Unknown test_woy: put all into train as fallback
            for a in anchors:
                train_anchors.append((r_idx, int(a)))
            continue

        n_weeks = len(anchors)
        last_year_cutoff = max(0, n_weeks - 52)

        for w_idx, a in enumerate(anchors):
            anchor_doy = int(doys[a])
            woy = _woy_from_doy(anchor_doy)
            dist = abs(woy - test_woy)
            dist = min(dist, 52 - dist)
            in_val = dist <= slack_weeks
            if last_year_only:
                in_val = in_val and (w_idx >= last_year_cutoff)
            if in_val:
                val_anchors.append((r_idx, int(a)))
            else:
                train_anchors.append((r_idx, int(a)))

    return (
        np.asarray(train_anchors, dtype=np.int32),
        np.asarray(val_anchors, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Channel normalization (training fold only)
# ---------------------------------------------------------------------------

def fit_channel_norm(
    daily_feats: list[np.ndarray],
    train_anchor_max_day_per_region: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std on daily values up to each region's training-fold
    cutoff. The cutoff is the last day used by any *training* anchor (so val
    rows don't leak into normalization stats)."""
    blocks = []
    for mat, last_day in zip(daily_feats, train_anchor_max_day_per_region):
        last_day = min(last_day, mat.shape[0] - 1)
        if last_day >= 0:
            blocks.append(mat[: last_day + 1])
    pooled = np.vstack(blocks)
    mean = pooled.mean(axis=0).astype(np.float32)
    std = pooled.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def save_channel_norm(mean: np.ndarray, std: np.ndarray, path=PT_NORM_PATH):
    np.savez(path, mean=mean, std=std)


def load_channel_norm(path=PT_NORM_PATH) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["mean"].astype(np.float32), z["std"].astype(np.float32)


# ---------------------------------------------------------------------------
# Region id <-> int map (region embedding)
# ---------------------------------------------------------------------------

def save_region_map(region_ids: list[str], path=PT_REGION_MAP_PATH):
    with open(path, "w") as f:
        json.dump({rid: i for i, rid in enumerate(region_ids)}, f)


def load_region_map(path=PT_REGION_MAP_PATH) -> dict[str, int]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Targets (next-5-weeks scores per anchor)
# ---------------------------------------------------------------------------

def compute_anchor_targets(
    weekly_scores: list[np.ndarray],
    n_horizons: int = HORIZONS,
) -> list[np.ndarray]:
    """For each region, per-anchor target vector of length n_horizons.
    Returns (n_anchors, n_horizons) float32 arrays; anchors with insufficient
    future scores are dropped at the caller's level (via valid_anchors mask)."""
    out = []
    for scores in weekly_scores:
        n = len(scores)
        targets = np.zeros((n, n_horizons), dtype=np.float32)
        for w in range(n):
            for h in range(1, n_horizons + 1):
                idx = w + h
                targets[w, h - 1] = scores[idx] if 0 <= idx < n else np.nan
        out.append(targets)
    return out


def severity_sample_weights(targets_per_anchor: np.ndarray) -> np.ndarray:
    """(n_anchors, n_horizons) targets -> (n_anchors,) weight = 1 + α·𝟙[max_y>0] + β·𝟙[max_y≥3]."""
    mx = targets_per_anchor.max(axis=1)
    w = np.ones(len(mx), dtype=np.float32)
    w += SAMPLE_WEIGHT_ALPHA * (mx > 0).astype(np.float32)
    w += SAMPLE_WEIGHT_BETA * (mx >= 3).astype(np.float32)
    return w


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------

class DailyWindowDataset(Dataset):
    """Slices (N_CHANNELS, LOOKBACK_DAYS) windows + side inputs lazily."""

    def __init__(
        self,
        daily_feats: list[np.ndarray],          # per-region (n_days, N_CHANNELS)
        daily_doys: list[np.ndarray],            # per-region (n_days,)
        score_lag_inputs: list[np.ndarray],      # per-region (n_anchors, N_SCORE_LAG + 2)
        targets: list[np.ndarray],               # per-region (n_anchors, n_horizons)
        anchors: np.ndarray,                     # (N, 2) int32 — (region_idx, anchor_day_idx)
        mean: np.ndarray,
        std: np.ndarray,
        lookback_days: int = LOOKBACK_DAYS,
        return_targets: bool = True,
        apply_global_norm: bool = True,
    ):
        # apply_global_norm=False is used by the RevIN model, which does
        # per-instance normalization itself and reads raw daily values so its
        # (μ, σ) side inputs carry the unmuted cross-year shift signal.
        self.daily_feats = daily_feats
        self.daily_doys = daily_doys
        self.score_lag_inputs = score_lag_inputs
        self.targets = targets
        self.anchors = anchors
        self.mean = mean.reshape(1, -1)
        self.std = std.reshape(1, -1)
        self.lookback_days = lookback_days
        self.return_targets = return_targets
        self.apply_global_norm = apply_global_norm

        # Phase 13: anchors are week indices into the weekly aggregated arrays.
        # No day→week conversion needed.
        n = len(anchors)
        self._anchor_week_idx = anchors[:, 1].astype(np.int32)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx: int):
        r_idx = int(self.anchors[idx, 0])
        d_idx = int(self.anchors[idx, 1])
        w_idx = int(self._anchor_week_idx[idx])

        # Phase 13: window is `lookback_days` (=weeks) of weekly meteo ending
        # at the anchor week (inclusive). For early anchors with <lookback
        # history, pad with the first row.
        mat = self.daily_feats[r_idx]
        start = d_idx - self.lookback_days + 1
        if start < 0:
            pad_rows = -start
            window = np.vstack([
                np.repeat(mat[0:1], pad_rows, axis=0),
                mat[: d_idx + 1],
            ])
        else:
            window = mat[start : d_idx + 1]
        assert window.shape[0] == self.lookback_days, (
            f"window shape {window.shape}, expected ({self.lookback_days}, M)"
        )
        if self.apply_global_norm:
            window = (window - self.mean) / self.std
        x_daily = window.T.astype(np.float32)  # (M, L)

        # Score-lag side inputs
        side = self.score_lag_inputs[r_idx][w_idx]   # (N_SCORE_LAG + 2,)

        # Calendar side inputs for this anchor: weekly doys are doys of each
        # week's last day. Index w_idx maps directly into the doy array.
        doy = int(self.daily_doys[r_idx][w_idx])
        # Approximate month
        m = 1
        while m < 12 and _MONTH_CUM[m] < doy:
            m += 1
        cal = np.array([
            np.sin(2 * np.pi * m / 12),
            np.cos(2 * np.pi * m / 12),
            np.sin(2 * np.pi * doy / 365),
            np.cos(2 * np.pi * doy / 365),
        ], dtype=np.float32)

        if self.return_targets:
            y = self.targets[r_idx][w_idx].astype(np.float32)
            return (
                torch.from_numpy(x_daily),
                torch.from_numpy(side),
                torch.from_numpy(cal),
                torch.from_numpy(y),
                r_idx,
            )
        return (
            torch.from_numpy(x_daily),
            torch.from_numpy(side),
            torch.from_numpy(cal),
            r_idx,
        )
