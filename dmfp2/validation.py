"""Shift-aware validation — the spine.

The recurring failure of this project: local val (~0.45) did not track Kaggle
(~0.88). The fix is a validation whose held-out set is a structural twin of the
Kaggle test: hot + in the test season + one anchor per region. We deliberately
hold out the HOTTEST in-season anchor per region so the train→val gap mimics the
+6 °C train→test gap, then train on everything else (minus a purge).

Also provides: baselines, bias diagnostics (the number that predicts Kaggle bias),
inner walk-forward folds for OOF, and the val↔LB tracking log.
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from . import config as C

WOY_MAX = 53
TGT_COLS = [f"target_w{h}" for h in C.HORIZONS]


# ---------------------------------------------------------------------------
# Hotness + season distance
# ---------------------------------------------------------------------------
def woy_distance(woy_a: np.ndarray, woy_b: np.ndarray) -> np.ndarray:
    """Circular week-of-year distance (0..~26)."""
    d = np.abs(woy_a - woy_b)
    return np.minimum(d, WOY_MAX - d)


def attach_region_meta(anchors: pd.DataFrame, test_anchors: pd.DataFrame) -> pd.DataFrame:
    """Attach each region's test woy + woy distance of every anchor to it."""
    meta = test_anchors.set_index("region_idx")["woy"].rename("test_woy")
    out = anchors.copy()
    out["test_woy"] = out["region_idx"].map(meta).to_numpy()
    out["woy_dist"] = woy_distance(out["woy"].to_numpy(), out["test_woy"].to_numpy())
    return out


# ---------------------------------------------------------------------------
# Hot-holdout split
# ---------------------------------------------------------------------------
def make_hot_holdout(anchors: pd.DataFrame, test_anchors: pd.DataFrame,
                     val: C.ValidationConfig = C.VALIDATION) -> dict:
    """Select, per region, one in-season held-out anchor + a purged train pool.

    Default selection="precip_deficit": the driest in-season week (region-mean
    prec_sum minus the anchor's prec_sum). This picks the region's active, region-WIDE
    drought state, which reproduces the real test's defining property (region_mean
    BEATS zero — climatology is informative). Calibrated against the two Kaggle
    baselines (zero-MAE 1.2088, region_mean-MAE 0.9364): drought-state selection gives
    zero-MAE≈1.11, rm-MAE≈1.04, rm<zero; the residual severity undershoot vs 1.21 is the
    irreducible out-of-support heat gap (training droughts occur at training temps).

    selection="temp_match" matches anchor temperature to (test temp + temp_offset); it
    matches severity but NOT the region_mean<zero structure (ablation only).
    `anchors` must carry weekly stat columns + keys.
    """
    a = attach_region_meta(anchors, test_anchors).reset_index(drop=True)
    hot_col = f"{val.temp_channel}_mean"
    if hot_col not in a.columns:
        hot_col = val.temp_channel
    in_season = (a["woy_dist"] <= val.calendar_slack_weeks).to_numpy()

    if val.selection == "precip_deficit":
        region_mean_prec = a.groupby("region_idx")["prec_sum"].transform("mean")
        score = (region_mean_prec - a["prec_sum"]).to_numpy()   # higher = drier = more drought
    elif val.selection == "temp_match":
        test_temp = test_anchors.set_index("region_idx")[hot_col]
        tt = a["region_idx"].map(test_temp).to_numpy()
        score = -np.abs(a[hot_col].to_numpy() - (tt + val.temp_offset))
    else:
        raise ValueError(f"unknown selection={val.selection!r}")

    # in-season anchors always preferred; among them pick the max-score (driest / best-matched)
    a["_pref"] = score + in_season.astype(float) * 1e9
    val_rows = a.groupby("region_idx")["_pref"].idxmax().to_numpy()
    val_mask = np.zeros(len(a), bool)
    val_mask[val_rows] = True

    # purge: drop train-pool anchors within purge_weeks of their region's val anchor
    val_ord = a.loc[val_mask].set_index("region_idx")["anchor_ord"]
    a["_val_ord"] = a["region_idx"].map(val_ord).to_numpy()
    near = np.abs(a["anchor_ord"].to_numpy() - a["_val_ord"].to_numpy()) < val.purge_weeks * C.DAYS_PER_WEEK
    train_mask = (~val_mask) & (~near)

    tgt_cols = [c for c in TGT_COLS if c in a.columns]
    return {
        "val_mask": val_mask,
        "train_mask": train_mask,
        "n_val": int(val_mask.sum()),
        "n_train": int(train_mask.sum()),
        "selection": val.selection,
        "hot_col": hot_col,
        "val_temp_mean": float(a.loc[val_mask, hot_col].mean()),
        "test_temp_mean": float(test_anchors[hot_col].mean()),
        "train_temp_mean": float(a.loc[train_mask, hot_col].mean()),
        "val_true_mean": float(np.nanmean(a.loc[val_mask, tgt_cols].to_numpy())) if tgt_cols else float("nan"),
    }


def in_season_es_mask(anchors: pd.DataFrame, test_anchors: pd.DataFrame,
                      val: C.ValidationConfig = C.VALIDATION) -> np.ndarray:
    """Kaggle-proxy early-stopping set: the most-recent IN-SEASON anchor per region.

    Selected by time+season only (not by a model feature), it mimics the test's calendar
    position so early-stopping halts where TEST-SEASON performance peaks instead of where
    the in-distribution tail keeps improving. Returns a boolean mask over `anchors`.
    """
    a = attach_region_meta(anchors, test_anchors).reset_index(drop=True)
    in_season = (a["woy_dist"] <= val.calendar_slack_weeks).to_numpy()
    a["_pref"] = np.where(in_season, a["anchor_ord"].to_numpy(), -1)
    rows = a.groupby("region_idx")["_pref"].idxmax().to_numpy()
    mask = np.zeros(len(a), bool)
    mask[rows] = True
    return mask


def inner_folds(anchor_ord: np.ndarray, val: C.ValidationConfig = C.VALIDATION):
    """Walk-forward folds over the train pool (by ordinal) with a purge gap.

    Yields (train_idx, oof_idx) into the given anchor_ord array.
    """
    n = len(anchor_ord)
    order = np.argsort(anchor_ord, kind="stable")
    times = anchor_ord[order]
    block = n // val.n_folds
    gap = val.purge_weeks * C.DAYS_PER_WEEK
    for fold in range(1, val.n_folds):
        vs = fold * block
        ve = (fold + 1) * block if fold < val.n_folds - 1 else n
        cutoff = times[vs] - gap
        te = int(np.searchsorted(times, cutoff, side="right"))
        if te <= 0:
            continue
        yield order[:te], order[vs:ve]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def baseline_preds(kind: str, train_anchors: pd.DataFrame,
                   eval_anchors: pd.DataFrame) -> np.ndarray:
    """(n_eval, 5) prediction matrix for a simple baseline."""
    n = len(eval_anchors)
    if kind == "zero":
        return np.zeros((n, C.N_PRED), np.float32)
    if kind == "persistence":
        cur = eval_anchors["score"].to_numpy(np.float32)
        return np.repeat(cur[:, None], C.N_PRED, axis=1)
    if kind == "region_mean":
        rm = train_anchors.groupby("region_idx")["score"].mean()
        glob = float(train_anchors["score"].mean())
        cur = eval_anchors["region_idx"].map(rm).fillna(glob).to_numpy(np.float32)
        return np.repeat(cur[:, None], C.N_PRED, axis=1)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Metrics / bias diagnostics
# ---------------------------------------------------------------------------
def evaluate(eval_anchors: pd.DataFrame, pred: np.ndarray) -> dict:
    """MAE + bias diagnostics for a (n, 5) prediction matrix vs ragged targets."""
    y = eval_anchors[TGT_COLS].to_numpy(np.float32)        # (n,5), may contain NaN
    pred = np.clip(pred, 0, 5)
    mask = ~np.isnan(y)

    per_h_mae, per_h_pm, per_h_tm = [], [], []
    for j in range(C.N_PRED):
        m = mask[:, j]
        if m.sum() == 0:
            per_h_mae.append(np.nan); per_h_pm.append(np.nan); per_h_tm.append(np.nan)
            continue
        per_h_mae.append(float(np.mean(np.abs(pred[m, j] - y[m, j]))))
        per_h_pm.append(float(pred[m, j].mean()))
        per_h_tm.append(float(y[m, j].mean()))

    all_p = pred[mask]; all_y = y[mask]
    bucket = {}
    for name, lo, hi in [("0", -0.5, 0.5), ("1-2", 0.5, 2.5), (">=3", 2.5, 5.5)]:
        bm = (all_y >= lo) & (all_y < hi)
        bucket[name] = float(np.mean(np.abs(all_p[bm] - all_y[bm]))) if bm.sum() else np.nan

    return {
        "macro_mae": float(np.nanmean(per_h_mae)),
        "per_horizon_mae": [round(x, 4) for x in per_h_mae],
        "pred_mean": float(all_p.mean()),
        "true_mean": float(all_y.mean()),
        "mean_bias": float(all_p.mean() - all_y.mean()),
        "per_horizon_bias": [round(p - t, 4) for p, t in zip(per_h_pm, per_h_tm)],
        "mae_by_bucket": {k: (round(v, 4) if v == v else None) for k, v in bucket.items()},
        "frac_pred_low": float(np.mean(all_p <= 0.5)),
        "frac_true_zero": float(np.mean(all_y <= 0.5)),
        "n_eval_rows": int(len(eval_anchors)),
    }


def log_val_lb(config_hash: str, tag: str, diag: dict, kaggle_lb=None):
    """Append a row to reports/val_lb_log.csv for val↔LB tracking."""
    row = {
        "tag": tag, "config_hash": config_hash,
        "hot_holdout_mae": round(diag["macro_mae"], 4),
        "pred_mean": round(diag["pred_mean"], 4),
        "true_mean": round(diag["true_mean"], 4),
        "mean_bias": round(diag["mean_bias"], 4),
        "kaggle_lb": kaggle_lb if kaggle_lb is not None else "",
    }
    new = not C.VAL_LB_LOG.exists()
    with open(C.VAL_LB_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
