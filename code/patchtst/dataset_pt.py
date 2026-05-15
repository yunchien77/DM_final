"""Dataset and windowing for the PatchTST track.

Consumes the weekly stat DataFrame produced by
data_pipeline.load_and_aggregate_daily_to_weekly and turns it into
fixed-length (M, L) input windows + 5-week target vectors per (region, anchor)
pair. Handles:

  - temporal train/val split (last VALID_WEEKS per region for val)
  - majority-class subsampling (anchors with all 5 future scores == 0)
  - per-channel z-score normalization (stats fit on train fold only)
  - sample weighting (1 + alpha*1[max_y>0] + beta*1[max_y>=3])

The same WindowBuilder is used at inference time on the train+test weekly
concatenation to produce one prediction window per region.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # let `import config` resolve
from config import METEO_FEATURES, STAT_SUFFIXES, TRAIN_WEEKS_PER_REGION
from data_pipeline import _ALL_STAT_COLS  # 14*5 = 70 stat columns

from patchtst.config_pt import (
    LOOKBACK_WEEKS, HORIZONS, VALID_WEEKS,
    MAJORITY_KEEP_FRAC, SAMPLE_WEIGHT_ALPHA, SAMPLE_WEIGHT_BETA,
    PT_NORM_PATH, PT_REGION_MAP_PATH,
)

N_CHANNELS = len(_ALL_STAT_COLS)   # 70

MAX_ANCHOR = TRAIN_WEEKS_PER_REGION - HORIZONS - 1   # 776
TRAIN_ANCHOR_MAX = MAX_ANCHOR - VALID_WEEKS          # 750
MIN_ANCHOR = LOOKBACK_WEEKS - 1                      # 51 — needs L weeks of past data inclusive


# ---------------------------------------------------------------------------
# Building per-region weekly stat / score matrices
# ---------------------------------------------------------------------------

def build_region_arrays(weekly_df: pd.DataFrame, is_train: bool = True):
    """Group the weekly DataFrame into per-region float32 arrays.

    Returns:
        region_ids: list[str] in encounter order
        stat_mats:  list[np.ndarray (n_weeks, N_CHANNELS)] float32
        score_arrs: list[np.ndarray (n_weeks,)] float32 — full of -1 for test
    """
    weekly_df = weekly_df.sort_values(["region_id", "week_idx"]).reset_index(drop=True)
    region_ids = []
    stat_mats = []
    score_arrs = []
    for rid, sub in weekly_df.groupby("region_id", sort=False):
        sub = sub.reset_index(drop=True)
        region_ids.append(rid)
        stat_mats.append(sub[_ALL_STAT_COLS].values.astype(np.float32))
        if is_train:
            score_arrs.append(sub["score"].values.astype(np.float32))
        else:
            score_arrs.append(np.full(len(sub), -1.0, dtype=np.float32))
    return region_ids, stat_mats, score_arrs


# ---------------------------------------------------------------------------
# Channel normalization (z-score with training-fold pooled stats)
# ---------------------------------------------------------------------------

def fit_channel_norm(stat_mats: list[np.ndarray], train_anchor_max: int = TRAIN_ANCHOR_MAX):
    """Compute per-channel mean/std using only weeks that the training fold can see.

    A training anchor at week w pulls in weeks [w-L+1, w]. The largest training
    anchor is TRAIN_ANCHOR_MAX, so the latest training-visible week index is
    TRAIN_ANCHOR_MAX. Anything beyond is val/test-only.
    """
    blocks = []
    for mat in stat_mats:
        last = min(mat.shape[0] - 1, train_anchor_max)
        if last >= 0:
            blocks.append(mat[: last + 1])
    pooled = np.vstack(blocks)
    mean = pooled.mean(axis=0).astype(np.float32)
    std = pooled.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def save_channel_norm(mean: np.ndarray, std: np.ndarray, path=PT_NORM_PATH):
    np.savez(path, mean=mean, std=std)


def load_channel_norm(path=PT_NORM_PATH):
    z = np.load(path)
    return z["mean"].astype(np.float32), z["std"].astype(np.float32)


# ---------------------------------------------------------------------------
# Region id <-> int map (for the region embedding)
# ---------------------------------------------------------------------------

def save_region_map(region_ids: list[str], path=PT_REGION_MAP_PATH):
    with open(path, "w") as f:
        json.dump({rid: i for i, rid in enumerate(region_ids)}, f)


def load_region_map(path=PT_REGION_MAP_PATH) -> dict[str, int]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Anchor enumeration (training+val) and subsampling
# ---------------------------------------------------------------------------

def enumerate_anchors(
    stat_mats: list[np.ndarray],
    score_arrs: list[np.ndarray],
    min_anchor: int = MIN_ANCHOR,
    train_anchor_max: int = TRAIN_ANCHOR_MAX,
    max_anchor: int = MAX_ANCHOR,
    keep_majority_frac: float = MAJORITY_KEEP_FRAC,
    seed: int = 42,
):
    """Yield (region_idx, anchor_week, split) tuples.

    split is "train" for anchor <= train_anchor_max, "val" otherwise.
    Majority-class anchors (all 5 horizon scores == 0) in the training split
    are subsampled to `keep_majority_frac` of their count. Validation anchors
    are kept in full so val MAE remains comparable to the LGBM track.
    """
    rng = np.random.default_rng(seed)
    train_anchors = []   # (region_idx, week)
    val_anchors = []
    train_zero_mask = []
    train_targets = []   # for downstream weighting

    for r_idx, (mat, scores) in enumerate(zip(stat_mats, score_arrs)):
        n_weeks = mat.shape[0]
        last_anchor = min(max_anchor, n_weeks - HORIZONS - 1)
        for w in range(min_anchor, last_anchor + 1):
            future = scores[w + 1 : w + 1 + HORIZONS]
            if w <= train_anchor_max:
                is_zero = bool((future == 0).all())
                train_anchors.append((r_idx, w))
                train_zero_mask.append(is_zero)
                train_targets.append(future.copy())
            else:
                val_anchors.append((r_idx, w))

    train_anchors = np.asarray(train_anchors, dtype=np.int32)
    train_zero_mask = np.asarray(train_zero_mask, dtype=bool)
    val_anchors = np.asarray(val_anchors, dtype=np.int32)

    # Subsample all-zero training anchors.
    if 0.0 < keep_majority_frac < 1.0:
        zero_idx = np.where(train_zero_mask)[0]
        n_keep = int(round(len(zero_idx) * keep_majority_frac))
        keep_zero = rng.choice(zero_idx, size=n_keep, replace=False)
        nonzero_idx = np.where(~train_zero_mask)[0]
        keep = np.concatenate([keep_zero, nonzero_idx])
        keep.sort()
        train_anchors = train_anchors[keep]
        train_targets = [train_targets[i] for i in keep]
    # else: keep_majority_frac >= 1.0 -> keep all

    return train_anchors, val_anchors, train_targets


def sample_weights(targets: list[np.ndarray]) -> np.ndarray:
    """1 + alpha*1[max_y>0] + beta*1[max_y>=3] per training anchor."""
    w = np.ones(len(targets), dtype=np.float32)
    for i, t in enumerate(targets):
        mx = float(t.max()) if len(t) else 0.0
        if mx > 0:
            w[i] += SAMPLE_WEIGHT_ALPHA
        if mx >= 3:
            w[i] += SAMPLE_WEIGHT_BETA
    return w


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------

class WeeklyWindowDataset(Dataset):
    """Slices (M, L) windows lazily from in-memory region matrices."""

    def __init__(
        self,
        stat_mats: list[np.ndarray],
        score_arrs: list[np.ndarray],
        anchors: np.ndarray,        # (N, 2) int32 — (region_idx, week)
        mean: np.ndarray,
        std: np.ndarray,
        lookback: int = LOOKBACK_WEEKS,
        return_targets: bool = True,
    ):
        self.stat_mats = stat_mats
        self.score_arrs = score_arrs
        self.anchors = anchors
        self.mean = mean.reshape(1, -1)        # (1, M)
        self.std = std.reshape(1, -1)
        self.lookback = lookback
        self.return_targets = return_targets

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx: int):
        r_idx, w = int(self.anchors[idx, 0]), int(self.anchors[idx, 1])
        mat = self.stat_mats[r_idx]
        start = w - self.lookback + 1
        window = mat[start : w + 1]                          # (L, M)
        window = (window - self.mean) / self.std             # z-score
        x = window.T.astype(np.float32)                      # (M, L)
        if self.return_targets:
            scores = self.score_arrs[r_idx]
            y = scores[w + 1 : w + 1 + HORIZONS].astype(np.float32)
            return torch.from_numpy(x), torch.from_numpy(y), r_idx
        return torch.from_numpy(x), r_idx
