"""Multi-scale proxy score: a Ridge per window scale mapping SHIFT-INVARIANT meteo
features → the 1-week-ahead score, plus their cross-scale differences.

This is the teammate's current-drought-state anchor (proxy_p7/p21/p91 + diffs). A strong
multi-scale proxy is what lets near-term predictions sit high and decay across horizons,
instead of the flat, too-low predictions our weak single proxy produced. Inputs are
restricted to shift-invariant columns (region/local anomalies, SPI, deficit, VPD, DTR,
aridity, calendar) — never raw temperature levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from . import config as C


def _scale_columns(df: pd.DataFrame, w: int) -> list:
    cols = [f"{ch}_ranom{w}" for ch in C.DATA.channels]
    cols += [f"spi_{w}", f"prec_deficit{w}", f"heat_dd{w}", f"aridity{w}", f"vpd_roll{w}", f"dtr_roll{w}"]
    cols += [c for c in df.columns if c.endswith("_anom_sl") or c.endswith("_anom_z")]
    cols += [c for c in ("doy_sin", "doy_cos", "month_sin", "month_cos") if c in df.columns]
    return [c for c in dict.fromkeys(cols) if c in df.columns]   # dedup, keep present


class ProxyScore:
    def __init__(self, features: C.FeatureConfig = C.FEATURES):
        self.features = features
        self.scales = features.proxy_scales

    def fit(self, pool: pd.DataFrame):
        n = self.features.proxy_samples_per_region
        recent = (pool.sort_values(["region_idx", "anchor_ord"])
                  .groupby("region_idx", sort=False).tail(n))
        recent = recent[recent["target_w1"].notna()]
        y = recent["target_w1"].to_numpy(np.float32)
        self.cols_, self.models_ = {}, {}
        for w in self.scales:
            cols = _scale_columns(pool, w)
            self.cols_[w] = cols
            mdl = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            mdl.fit(recent[cols].to_numpy(np.float32), y)
            self.models_[w] = mdl
        return self

    def transform(self, anchors: pd.DataFrame) -> pd.DataFrame:
        p = {}
        for w in self.scales:
            p[w] = np.clip(self.models_[w].predict(anchors[self.cols_[w]].to_numpy(np.float32)), 0, 5)
        out = {f"proxy_p{w}": p[w].astype(np.float32) for w in self.scales}
        ws = list(self.scales)
        out["proxy_main"] = np.mean([p[w] for w in ws], axis=0).astype(np.float32)
        out[f"proxy_{ws[0]}_{ws[-1]}"] = (p[ws[0]] - p[ws[-1]]).astype(np.float32)
        if len(ws) >= 3:
            out[f"proxy_{ws[1]}_{ws[-1]}"] = (p[ws[1]] - p[ws[-1]]).astype(np.float32)
            out[f"proxy_{ws[0]}_{ws[1]}"] = (p[ws[0]] - p[ws[1]]).astype(np.float32)
        return pd.DataFrame(out, index=anchors.index)

    @property
    def out_cols(self):
        ws = list(self.scales)
        cols = [f"proxy_p{w}" for w in ws] + ["proxy_main", f"proxy_{ws[0]}_{ws[-1]}"]
        if len(ws) >= 3:
            cols += [f"proxy_{ws[1]}_{ws[-1]}", f"proxy_{ws[0]}_{ws[1]}"]
        return cols
