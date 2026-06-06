"""Split-DEPENDENT features: fit on the train pool only, applied to val/test.

Region score climatology (each region's historical drought level) is a strong,
shift-robust signal — the real test rewards it (region_mean beats zero on the LB).
Fit strictly on the train pool to avoid leakage into the held-out val.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


class RegionScalars:
    """Per-region historical drought statistics from the origin score."""

    COLS = ["region_score_mean", "region_score_std", "region_zero_rate", "region_score_p90"]

    def fit(self, pool: pd.DataFrame):
        g = pool.groupby("region_idx")["score"]
        tbl = pd.DataFrame({
            "region_score_mean": g.mean(),
            "region_score_std": g.std().fillna(0.0),
            "region_zero_rate": g.apply(lambda s: float((s == 0).mean())),
            "region_score_p90": g.quantile(0.90),
        })
        self.table_ = tbl
        self.global_ = {
            "region_score_mean": float(pool["score"].mean()),
            "region_score_std": float(pool["score"].std()),
            "region_zero_rate": float((pool["score"] == 0).mean()),
            "region_score_p90": float(pool["score"].quantile(0.90)),
        }
        return self

    def transform(self, anchors: pd.DataFrame) -> pd.DataFrame:
        joined = anchors[["region_idx"]].join(self.table_, on="region_idx")
        out = {}
        for c in self.COLS:
            out[c] = joined[c].fillna(self.global_[c]).to_numpy(np.float32)
        return pd.DataFrame(out, index=anchors.index)


class SeasonalScoreClim:
    """Per-(region, month) drought-score climatology — the seasonal swing region_mean misses.

    This is the largest transferable signal the teammate's repo has and ours lacked: the
    region's typical score for THIS month. The seasonal pattern (which months are droughty)
    is stable across the +6 °C shift, so it transfers; only the absolute level lifts (handled
    by calibration). Fit on the train pool only.
    """

    COLS = ["region_month_score_mean", "region_month_score_std"]

    def fit(self, pool: pd.DataFrame):
        g = pool.groupby(["region_idx", "month"])["score"]
        self.mean_ = g.mean()
        self.std_ = g.std().fillna(0.0)
        rg = pool.groupby("region_idx")["score"]
        self.region_mean_ = rg.mean()
        self.region_std_ = rg.std().fillna(0.0)
        self.global_mean_ = float(pool["score"].mean())
        self.global_std_ = float(pool["score"].std())
        return self

    def transform(self, anchors: pd.DataFrame) -> pd.DataFrame:
        idx = pd.MultiIndex.from_arrays([anchors["region_idx"], anchors["month"]])
        rmean = anchors["region_idx"].map(self.region_mean_).fillna(self.global_mean_)
        rstd = anchors["region_idx"].map(self.region_std_).fillna(self.global_std_)
        mo_mean = self.mean_.reindex(idx).to_numpy()
        mo_std = self.std_.reindex(idx).to_numpy()
        mo_mean = np.where(np.isnan(mo_mean), rmean.to_numpy(), mo_mean)
        mo_std = np.where(np.isnan(mo_std), rstd.to_numpy(), mo_std)
        return pd.DataFrame(
            {"region_month_score_mean": mo_mean.astype(np.float32),
             "region_month_score_std": mo_std.astype(np.float32)},
            index=anchors.index)
