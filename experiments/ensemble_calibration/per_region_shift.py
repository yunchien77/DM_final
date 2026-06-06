"""Per-region recent-season post-hoc shift.

For each region, compute `r_region = mean(pred - truth)` on its calendar-matched
val rows (the closest available proxy for "last 26 weeks of training that look
test-like"). Subtract `r_region` from each region's test predictions, then clip
to [0, 5]. Cap |r_region| ≤ 0.5 and require ≥ 8 non-zero val rows per region.

Why: the Step 0 diagnostic showed LGBM Phase 1b v2 systematically over-predicts
high-severity clusters by 0.07-0.22. Per-region shift captures within-cluster
variation that per-cluster isotonic averages over.

Usage:
    python -m ensemble.per_region_shift \\
        --val-preds code/models/val_preds_phase1b_v2.npz \\
        --in-submission submission_phase1b_v2.csv \\
        --out-submission submission_phase1b_v2_region_shift.csv

Defaults to phase1b_v2 paths.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

MIN_NONZERO_OBS = 8
MAX_ABS_SHIFT = 0.5


def compute_per_region_shift(
    preds_val: np.ndarray,        # (N, H)
    truth_val: np.ndarray,        # (N, H)
    region_ids_val: np.ndarray,   # (N,) str
    min_nonzero_obs: int = MIN_NONZERO_OBS,
    max_abs_shift: float = MAX_ABS_SHIFT,
) -> dict[str, float]:
    """For each region, compute mean residual (pred - truth) across all val rows
    and horizons. Skip regions with fewer than `min_nonzero_obs` non-zero truth
    rows (avoids over-correction on all-zero regions). Cap the absolute shift."""
    shifts: dict[str, float] = {}
    unique_regions = np.unique(region_ids_val)
    for rid in unique_regions:
        m = region_ids_val == rid
        sub_truth = truth_val[m]
        sub_pred = preds_val[m]
        # Count non-zero anchors (rows where ANY horizon truth > 0)
        nonzero_anchor = (sub_truth > 0).any(axis=1)
        n_nonzero = int(nonzero_anchor.sum())
        if n_nonzero < min_nonzero_obs:
            shifts[str(rid)] = 0.0
            continue
        residual = sub_pred - sub_truth   # (n_rows, H)
        r = float(residual.mean())
        r = max(-max_abs_shift, min(max_abs_shift, r))
        shifts[str(rid)] = r
    return shifts


def apply_per_region_shift(
    preds: np.ndarray,              # (N_regions, H)
    region_ids: np.ndarray,         # (N_regions,) str
    shifts: dict[str, float],
    y_min: float = 0.0,
    y_max: float = 5.0,
) -> np.ndarray:
    out = preds.copy().astype(np.float32)
    n_applied = 0
    for i, rid in enumerate(region_ids):
        r = shifts.get(str(rid), 0.0)
        if r != 0.0:
            out[i] = out[i] - r
            n_applied += 1
    print(f"  applied per-region shift to {n_applied}/{len(region_ids)} regions")
    return np.clip(out, y_min, y_max)


def macro_mae(preds: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([np.mean(np.abs(preds[:, h] - truth[:, h]))
                          for h in range(preds.shape[1])]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val-preds", default=str(MODELS_DIR / "val_preds_phase1b_v2.npz"))
    p.add_argument("--in-submission", default=str(ROOT / "submission_phase1b_v2.csv"))
    p.add_argument("--out-submission", default=str(ROOT / "submission_phase1b_v2_region_shift.csv"))
    p.add_argument("--min-nonzero-obs", type=int, default=MIN_NONZERO_OBS)
    p.add_argument("--max-abs-shift", type=float, default=MAX_ABS_SHIFT)
    args = p.parse_args()

    print(f"[load] val preds:   {args.val_preds}")
    d = np.load(args.val_preds, allow_pickle=True)
    preds_val = d["preds"]
    truth_val = d["truth"]
    region_ids_val = d["region_ids"].astype(str)
    horizons = d["horizons"]

    pre_macro = macro_mae(preds_val, truth_val)
    print(f"[val] macro MAE pre-shift:  {pre_macro:.4f}")

    print(f"\n[fit] computing per-region shifts (min_nonzero={args.min_nonzero_obs}, cap={args.max_abs_shift})...")
    shifts = compute_per_region_shift(
        preds_val, truth_val, region_ids_val,
        min_nonzero_obs=args.min_nonzero_obs,
        max_abs_shift=args.max_abs_shift,
    )
    nonzero_shifts = [s for s in shifts.values() if s != 0.0]
    n_capped = sum(1 for s in shifts.values() if abs(s) >= args.max_abs_shift * 0.999)
    print(f"  {len(nonzero_shifts):,}/{len(shifts):,} regions get a non-zero shift")
    if nonzero_shifts:
        print(
            f"  shift stats: mean={np.mean(nonzero_shifts):+.4f}  "
            f"median={np.median(nonzero_shifts):+.4f}  "
            f"min={np.min(nonzero_shifts):+.4f}  max={np.max(nonzero_shifts):+.4f}  "
            f"capped={n_capped}"
        )

    # Apply to val (using the SAME shift for evaluation — overfits slightly but
    # tells us if the recipe is structurally correct)
    val_idx_to_region = region_ids_val
    cal_val = preds_val.copy()
    for i in range(len(cal_val)):
        cal_val[i] = cal_val[i] - shifts.get(str(val_idx_to_region[i]), 0.0)
    cal_val = np.clip(cal_val, 0.0, 5.0)
    post_macro = macro_mae(cal_val, truth_val)
    print(f"[val] macro MAE post-shift: {post_macro:.4f}  (delta = {pre_macro - post_macro:+.4f})")
    print("  NB: val delta is in-sample. The test-side gain is typically smaller.")

    # Apply to submission CSV
    print(f"\n[apply] reading {args.in_submission}")
    sub = pd.read_csv(args.in_submission)
    horizon_cols = [f"pred_week{int(h)}" for h in range(1, len(horizons) + 1)]
    preds_test = sub[horizon_cols].to_numpy(np.float32)
    region_ids_test = sub["region_id"].astype(str).to_numpy()

    cal_test = apply_per_region_shift(preds_test, region_ids_test, shifts)
    sub[horizon_cols] = cal_test
    sub.to_csv(args.out_submission, index=False)
    print(f"[done] wrote {args.out_submission}")

    print("\n[summary] prediction-shape delta (pre -> post shift):")
    pre_means = preds_test.mean(axis=0)
    post_means = cal_test.mean(axis=0)
    for h in range(preds_test.shape[1]):
        print(
            f"  week {h+1}: pre_mean={pre_means[h]:.3f} -> post_mean={post_means[h]:.3f}  "
            f"delta={post_means[h] - pre_means[h]:+.3f}"
        )


if __name__ == "__main__":
    main()
