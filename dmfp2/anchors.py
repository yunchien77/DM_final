"""Anchor rows + ragged multi-horizon targets.

An anchor is a scored week `t`. The FIXED input window ends at the anchor; the
targets are the scores at weeks `t+1..t+5` (looked up by ordinal so week gaps are
handled correctly). Per-horizon models train only on anchors whose `t+h` exists
(ragged), keeping more data at short horizons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _attach_targets(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add target_w1..w5 by ordinal lookup within each region."""
    base = weekly.copy()
    lookup = weekly[["region_idx", "anchor_ord", "score"]].rename(
        columns={"anchor_ord": "k", "score": "tgt"})
    for h in C.HORIZONS:
        key = base[["region_idx"]].copy()
        key["k"] = base["anchor_ord"] + C.DAYS_PER_WEEK * h
        merged = key.merge(lookup, on=["region_idx", "k"], how="left")
        base[f"target_w{h}"] = merged["tgt"].to_numpy()
    return base


def build_train_anchors(train_weekly: pd.DataFrame,
                        features: C.FeatureConfig = C.FEATURES) -> pd.DataFrame:
    """Scored anchors with enough history and at least the 1-week target present."""
    df = _attach_targets(train_weekly)
    df = df[df["score"].notna()]
    df = df[df["week_pos"] >= features.lookback_weeks]
    tgt_cols = [f"target_w{h}" for h in C.HORIZONS]
    df = df[df[tgt_cols].notna().any(axis=1)].reset_index(drop=True)
    return df


def build_test_anchors(test_weekly: pd.DataFrame) -> pd.DataFrame:
    """One anchor per region = its last (latest) week."""
    last = (test_weekly.sort_values(["region_idx", "week_pos"])
            .groupby("region_idx", sort=True).tail(1).reset_index(drop=True))
    return last
