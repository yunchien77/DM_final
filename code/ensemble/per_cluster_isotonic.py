"""Per-(cluster, horizon) isotonic calibration.

Loads a val_preds.npz to learn one IsotonicRegression(0, 5) per (cluster,
horizon). Applies the learned mapping to a submission CSV and writes a new
calibrated submission. Reports val MAE before/after.

Usage:
    python -m ensemble.per_cluster_isotonic \\
        --val-preds code/models/val_preds_phase1b_v2.npz \\
        --in-submission submission_phase1b_v2.csv \\
        --out-submission submission_phase1b_v2_iso_cluster.csv

Defaults to phase1b_v2 paths if no args are given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def fit_per_cluster_isotonic(
    preds_val: np.ndarray,        # (N, H)
    truth_val: np.ndarray,        # (N, H)
    cluster_val: np.ndarray,      # (N,)
    y_min: float = 0.0,
    y_max: float = 5.0,
) -> dict[tuple[int, int], IsotonicRegression]:
    """One IsotonicRegression per (cluster_id, horizon). Returns dict keyed by
    (cluster_id, horizon_index)."""
    models: dict[tuple[int, int], IsotonicRegression] = {}
    horizons = preds_val.shape[1]
    for cid in np.unique(cluster_val):
        m_c = cluster_val == cid
        if not m_c.any():
            continue
        for h in range(horizons):
            ir = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds="clip")
            ir.fit(preds_val[m_c, h], truth_val[m_c, h])
            models[(int(cid), h)] = ir
    return models


def apply_per_cluster_isotonic(
    models: dict[tuple[int, int], IsotonicRegression],
    preds: np.ndarray,        # (N, H)
    cluster: np.ndarray,      # (N,)
    fallback_models: list[IsotonicRegression] | None = None,
    y_min: float = 0.0,
    y_max: float = 5.0,
) -> np.ndarray:
    """Apply (cluster, horizon)-specific isotonic. Rows whose cluster has no
    fitted model fall back to a global per-horizon isotonic if supplied; else
    pass through unchanged."""
    out = np.zeros_like(preds, dtype=np.float32)
    horizons = preds.shape[1]
    for cid in np.unique(cluster):
        m = cluster == cid
        for h in range(horizons):
            key = (int(cid), h)
            if key in models:
                out[m, h] = models[key].transform(preds[m, h])
            elif fallback_models is not None and h < len(fallback_models):
                out[m, h] = fallback_models[h].transform(preds[m, h])
            else:
                out[m, h] = preds[m, h]
    return np.clip(out, y_min, y_max).astype(np.float32)


def macro_mae(preds: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([np.mean(np.abs(preds[:, h] - truth[:, h]))
                          for h in range(preds.shape[1])]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val-preds", default=str(MODELS_DIR / "val_preds_phase1b_v2.npz"))
    p.add_argument("--in-submission", default=str(ROOT / "submission_phase1b_v2.csv"))
    p.add_argument("--out-submission", default=str(ROOT / "submission_phase1b_v2_iso_cluster.csv"))
    p.add_argument("--cluster-csv", default=str(MODELS_DIR / "region_clusters.csv"))
    args = p.parse_args()

    print(f"[load] val preds:   {args.val_preds}")
    d = np.load(args.val_preds, allow_pickle=True)
    preds_val = d["preds"]            # (N, H)
    truth_val = d["truth"]            # (N, H)
    region_ids_val = d["region_ids"].astype(str)
    horizons = d["horizons"]

    print(f"[load] clusters:    {args.cluster_csv}")
    clusters = pd.read_csv(args.cluster_csv)[["region_id", "region_cluster_id"]]
    cluster_map = dict(zip(clusters["region_id"].astype(str), clusters["region_cluster_id"].astype(int)))
    cluster_val = np.array([cluster_map.get(r, -1) for r in region_ids_val], dtype=np.int32)
    if (cluster_val == -1).any():
        print(f"  WARNING: {(cluster_val == -1).sum()} val rows have no cluster — passed through")

    pre_macro = macro_mae(preds_val, truth_val)
    print(f"[val] macro MAE pre-calibration:  {pre_macro:.4f}")

    # Per-cluster per-horizon MAE pre
    print("\n  per-cluster val MAE pre-calibration:")
    for cid in sorted(np.unique(cluster_val[cluster_val >= 0])):
        m = cluster_val == cid
        cluster_mae = float(np.mean([np.mean(np.abs(preds_val[m, h] - truth_val[m, h]))
                                     for h in range(preds_val.shape[1])]))
        n = int(m.sum())
        print(f"    cluster {int(cid)}: n={n:,}  macro_mae={cluster_mae:.4f}")

    print("\n[fit] per-(cluster, horizon) isotonic on val...")
    models = fit_per_cluster_isotonic(preds_val, truth_val, cluster_val)
    print(f"  fitted {len(models)} (cluster,horizon) isotonic models")

    cal_val = apply_per_cluster_isotonic(models, preds_val, cluster_val)
    post_macro = macro_mae(cal_val, truth_val)
    print(f"[val] macro MAE post-calibration: {post_macro:.4f}  (delta = {pre_macro - post_macro:+.4f})")

    print("\n  per-cluster val MAE post-calibration:")
    for cid in sorted(np.unique(cluster_val[cluster_val >= 0])):
        m = cluster_val == cid
        cluster_mae = float(np.mean([np.mean(np.abs(cal_val[m, h] - truth_val[m, h]))
                                     for h in range(preds_val.shape[1])]))
        print(f"    cluster {int(cid)}: macro_mae={cluster_mae:.4f}")

    # Apply to submission CSV
    print(f"\n[apply] reading {args.in_submission}")
    sub = pd.read_csv(args.in_submission)
    horizon_cols = [f"pred_week{int(h)}" for h in range(1, len(horizons) + 1)]
    preds_test = sub[horizon_cols].to_numpy(np.float32)
    cluster_test = np.array(
        [cluster_map.get(str(r), -1) for r in sub["region_id"]], dtype=np.int32
    )
    if (cluster_test == -1).any():
        print(f"  WARNING: {(cluster_test == -1).sum()} test rows have no cluster — passed through")

    cal_test = apply_per_cluster_isotonic(models, preds_test, cluster_test)
    sub[horizon_cols] = cal_test
    sub.to_csv(args.out_submission, index=False)
    print(f"[done] wrote {args.out_submission}")

    print("\n[summary] prediction-shape delta (pre -> post calibration):")
    pre_means = preds_test.mean(axis=0)
    post_means = cal_test.mean(axis=0)
    for h in range(preds_test.shape[1]):
        print(
            f"  week {h+1}: pre_mean={pre_means[h]:.3f} -> post_mean={post_means[h]:.3f}  "
            f"delta={post_means[h] - pre_means[h]:+.3f}"
        )


if __name__ == "__main__":
    main()
