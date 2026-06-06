"""Split-independent feature layer (computed on the weekly frame, cached).

These features use only causal, within-region meteo (available identically at train
and test), so they can be precomputed before the train/val split:
  - RAW rolling levels (mean/std over {4,8,12} weeks of each channel's weekly mean)
    — shift-SENSITIVE.
  - LOCAL anomalies (Family A): deviations of the anchor week / short window from the
    series' own longer trailing window — shift-INVARIANT by construction.
  - CALENDAR: sin/cos of day-of-year and month.

Split-DEPENDENT features (region score climatology, proxy score) are fit on the train
pool only — see transforms.py / proxy.py.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from . import config as C
from . import weekly as W

EPS = 1e-6
# columns that are never features
_NON_FEATURE = {
    C.ID_COL, "region_idx", "week_pos", "anchor_ord", "doy", "month", "woy", "score",
    "test_woy", "woy_dist",
} | {f"target_w{h}" for h in C.HORIZONS}


def compute_region_base(train_weekly: pd.DataFrame) -> pd.DataFrame:
    """Per-region long-term meteo climatology (mean/std) from TRAIN — applied to train AND test."""
    base_cols = [f"{ch}_mean" for ch in C.DATA.channels] + ["prec_sum"]
    g = train_weekly.groupby("region_idx")[base_cols]
    out = pd.DataFrame(index=g.mean().index)
    m, s = g.mean(), g.std()
    for c in base_cols:
        out[f"{c}_bmean"] = m[c]
        out[f"{c}_bstd"] = s[c].fillna(0.0)
    return out


def add_features(weekly: pd.DataFrame, features: C.FeatureConfig = C.FEATURES,
                 region_base: pd.DataFrame = None) -> pd.DataFrame:
    """Return the weekly frame enriched with rolling, anomaly, region-anomaly, physical,
    and calendar features. `region_base` (from TRAIN) enables region-baseline features."""
    df = weekly.sort_values(["region_idx", "week_pos"], kind="stable").reset_index(drop=True)
    mean_cols = [f"{ch}_mean" for ch in C.DATA.channels]
    g = df.groupby("region_idx", sort=False)

    roll = {}  # (W, stat) -> DataFrame of rolling values per mean col
    for w in features.windows:
        roll[(w, "mean")] = g[mean_cols].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        roll[(w, "std")] = g[mean_cols].rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
    # precip-sum rolling (for deficit / SPI)
    gp = df.groupby("region_idx", sort=False)["prec_sum"]
    prec_roll = {w: gp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).to_numpy()
                 for w in features.windows}

    def _base(col):
        return (df["region_idx"].map(region_base[f"{col}_bmean"]).to_numpy(),
                df["region_idx"].map(region_base[f"{col}_bstd"]).to_numpy() + EPS)

    new = {}
    if features.include_raw_levels:
        for w in features.windows:
            for ch in C.DATA.channels:
                new[f"{ch}_roll{w}_mean"] = roll[(w, "mean")][f"{ch}_mean"].to_numpy()
                new[f"{ch}_roll{w}_std"] = roll[(w, "std")][f"{ch}_mean"].to_numpy()

    if features.include_local_anomaly:
        w_short, w_long = features.windows[0], features.windows[-1]
        rl_mean = roll[(w_long, "mean")]
        rl_std = roll[(w_long, "std")]
        rs_mean = roll[(w_short, "mean")]
        for ch in C.DATA.channels:
            m = f"{ch}_mean"
            long_mean = rl_mean[m].to_numpy()
            long_std = rl_std[m].to_numpy()
            # short-vs-long window deviation (uniform offset cancels)
            new[f"{ch}_anom_sl"] = rs_mean[m].to_numpy() - long_mean
            # current week standardized vs its own trailing long window
            new[f"{ch}_anom_z"] = (df[m].to_numpy() - long_mean) / (long_std + EPS)

    # Region-baseline ANOMALIES: current window vs the region's TRAIN climatology. Unlike
    # local (trailing-window) anomalies, these do NOT cancel the sustained +6C heat — they
    # carry it as a large positive signal the model maps to "more drought" (the teammate's
    # anom_z block). Requires region_base.
    if features.include_region_anom and region_base is not None:
        for ch in C.DATA.channels:
            bm, bs = _base(f"{ch}_mean")
            for w in features.windows:
                rw = roll[(w, "mean")][f"{ch}_mean"].to_numpy()
                new[f"{ch}_ranom{w}"] = (rw - bm) / bs

    if features.include_physical:
        tmp_, wb = df["tmp_mean"].to_numpy(), df["wb_tmp_mean"].to_numpy()
        tmax, tmin = df["tmp_max_mean"].to_numpy(), df["tmp_min_mean"].to_numpy()
        hum = df["humidity_mean"].to_numpy()
        vpd = (tmp_ - wb).astype(np.float32)          # temp DIFFERENCE → offset cancels
        dtr = (tmax - tmin).astype(np.float32)         # diurnal range → offset cancels
        new["vpd"] = vpd
        new["dtr"] = dtr
        for s, nm in [(vpd, "vpd"), (dtr, "dtr")]:
            tmp_s = pd.DataFrame({"s": s, "region_idx": df["region_idx"].to_numpy()})
            gs = tmp_s.groupby("region_idx", sort=False)["s"]
            for w in features.windows:
                new[f"{nm}_roll{w}"] = gs.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).to_numpy()
        # region-baseline physical anomalies (shift-carrying drought-stress signals)
        if region_base is not None:
            pbm, _ = _base("prec_sum")
            tbm, tbs = _base("tmp_mean")
            hbm, hbs = _base("humidity_mean")
            for w in features.windows:
                new[f"prec_deficit{w}"] = pbm - prec_roll[w]                 # >0 = drier than normal
                rw_t = roll[(w, "mean")]["tmp_mean"].to_numpy()
                new[f"heat_dd{w}"] = np.maximum(rw_t - tbm, 0.0)             # heat above region norm
                rw_h = roll[(w, "mean")]["humidity_mean"].to_numpy()
                new[f"aridity{w}"] = (hbm - rw_h) / hbs                      # drier-air anomaly
        # causal SPI: precip standardized vs the series' OWN trailing long window (shift-invariant)
        pl = df.groupby("region_idx", sort=False)["prec_sum"]
        long = features.windows[-1]
        plm = pl.rolling(long, min_periods=2).mean().reset_index(level=0, drop=True).to_numpy()
        pls = pl.rolling(long, min_periods=2).std().reset_index(level=0, drop=True).to_numpy()
        for w in features.windows:
            new[f"spi_{w}"] = (prec_roll[w] - plm) / (pls + EPS)

    if features.include_calendar:
        doy = df["doy"].to_numpy()
        mon = df["month"].to_numpy()
        new["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
        new["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
        new["month_sin"] = np.sin(2 * np.pi * mon / 12.0)
        new["month_cos"] = np.cos(2 * np.pi * mon / 12.0)

    out = pd.concat([df, pd.DataFrame(new, index=df.index).astype(np.float32)], axis=1)
    return out


def feature_columns(df: pd.DataFrame) -> list:
    """All usable feature columns present in df (excludes keys/targets/score)."""
    return [c for c in df.columns if c not in _NON_FEATURE and df[c].dtype != object]


def get_features(data: C.DataConfig = C.DATA, features: C.FeatureConfig = C.FEATURES,
                 rebuild: bool = False):
    """Return (train_feat_weekly, test_feat_weekly), cached by feature_hash."""
    h = C.feature_hash(features, data)
    cache = C.CACHE_DIR / f"features_{h}.pkl"
    if cache.exists() and not rebuild:
        with open(cache, "rb") as f:
            return pickle.load(f)

    print(f"[features] building feature matrices (hash={h}) ...")
    tr_w, te_w = W.get_weekly(data)
    region_base = compute_region_base(tr_w)   # TRAIN climatology, applied to both
    tr_f = add_features(tr_w, features, region_base)
    te_f = add_features(te_w, features, region_base)
    n_feat = len(feature_columns(tr_f))
    print(f"[features] {n_feat} split-independent feature cols | train {len(tr_f):,} test {len(te_f):,}")
    with open(cache, "wb") as f:
        pickle.dump((tr_f, te_f), f)
    return tr_f, te_f
