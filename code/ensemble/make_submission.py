"""Build the Phase 5 Day-2 submission.

Strategy (decided 2026-05-19 after 3-member val analysis):
  - 33/33/33 blend of LGBM Phase 1b + CatBoost + PatchTST_v2
  - Use Phase 1b LGBM (Kaggle 0.8798) as the LGBM test contribution since it's
    our best-on-Kaggle single, even though val_preds.npz came from Phase 3*.
  - Fit isotonic calibration on the 3-way val blend; apply to test only if it
    improves val cluster_mae.

Outputs:
  submission_phase5_3way.csv          (raw 33/33/33 blend)
  submission_phase5_3way_iso.csv      (iso-calibrated; only if iso helped val)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from ensemble.hillclimb import (
    load_member, align_members, macro_mae, cluster_averaged_macro_mae,
    blend_submissions,
)
from ensemble.calibrate import fit_isotonic_per_horizon, apply_isotonic


REPO_ROOT = Path(__file__).parent.parent.parent  # /mnt/1stHDD/juiyun/DMFP

# Test-side submission inputs. LGBM = Phase 1b (best known). PatchTST = v2.
# CatBoost = freshly trained.
TEST_SUBS = {
    "lgbm":     REPO_ROOT / "submission_phase1b.csv",
    "catboost": REPO_ROOT / "submission_catboost.csv",
    "patchtst": REPO_ROOT / "submission_patchtst_v2.csv",
}

VAL_PREDS = {
    "lgbm":     "models/val_preds.npz",                       # Phase 3* — closest aligned val we have for LGBM family
    "catboost": "models/val_preds_catboost.npz",
    "patchtst": "patchtst/models_pt/val_preds_patchtst.npz",
}

WEIGHTS = {"lgbm": 1/3, "catboost": 1/3, "patchtst": 1/3}

NAMES = ["lgbm", "catboost", "patchtst"]  # consistent ordering across both dicts

RAW_OUT = REPO_ROOT / "submission_phase5_3way.csv"
ISO_OUT = REPO_ROOT / "submission_phase5_3way_iso.csv"


def main():
    # Sanity-check input files
    for tag, p in TEST_SUBS.items():
        assert p.is_file(), f"missing test submission for {tag}: {p}"
    for tag, p in VAL_PREDS.items():
        assert Path(p).is_file(), f"missing val_preds for {tag}: {p}"

    # ----- Val analysis -----
    print("=== Val analysis ===")
    members = [load_member(VAL_PREDS[n]) for n in NAMES]
    aligned = align_members(members)
    preds = aligned["preds"]   # (M, N, H)
    truth = aligned["truth"]
    cl = aligned["cluster_ids"]

    w = np.array([WEIGHTS[n] for n in NAMES], dtype=np.float32)
    val_blend = (preds * w[:, None, None]).sum(axis=0)
    pre_cluster = cluster_averaged_macro_mae(val_blend, truth, cl)
    pre_macro = macro_mae(val_blend, truth)
    print(f"  weights: " + ", ".join(f"{n}={WEIGHTS[n]:.3f}" for n in NAMES))
    print(f"  val blend cluster_mae = {pre_cluster:.4f}")
    print(f"  val blend macro_mae   = {pre_macro:.4f}")
    print("  individual val cluster_mae:")
    for i, n in enumerate(NAMES):
        print(f"    {n:>10s}: {cluster_averaged_macro_mae(preds[i], truth, cl):.4f}")

    # ----- Isotonic calibration on val (per horizon) -----
    print("\n=== Isotonic calibration (val) ===")
    iso_models = fit_isotonic_per_horizon(val_blend, truth)
    val_iso = apply_isotonic(iso_models, val_blend)
    post_cluster = cluster_averaged_macro_mae(val_iso, truth, cl)
    post_macro = macro_mae(val_iso, truth)
    print(f"  post-iso cluster_mae  = {post_cluster:.4f}  (Δ = {pre_cluster - post_cluster:+.4f})")
    print(f"  post-iso macro_mae    = {post_macro:.4f}  (Δ = {pre_macro - post_macro:+.4f})")
    iso_helps = post_cluster < pre_cluster - 1e-4

    # ----- Build raw blend submission -----
    print("\n=== Test-time blend ===")
    submission_paths = [TEST_SUBS[n] for n in NAMES]
    weights_arr = np.array([WEIGHTS[n] for n in NAMES], dtype=np.float32)
    raw_sub = blend_submissions(submission_paths, weights_arr, RAW_OUT)

    # ----- Apply isotonic to test blend if it helped on val -----
    if iso_helps:
        print("\n=== Test-time isotonic (applied per horizon) ===")
        pred_cols = [f"pred_week{h}" for h in range(1, 6)]
        raw_preds = raw_sub[pred_cols].values.astype(np.float32)
        iso_preds = apply_isotonic(iso_models, raw_preds)
        iso_sub = raw_sub[["region_id"]].copy()
        for i, c in enumerate(pred_cols):
            iso_sub[c] = iso_preds[:, i]
        iso_sub.to_csv(ISO_OUT, index=False)
        print(f"Wrote {ISO_OUT}  shape={iso_sub.shape}")
        print(iso_sub.head().to_string(index=False))
        print(f"\nRECOMMEND: submit {ISO_OUT} (iso lowered val cluster_mae by "
              f"{pre_cluster - post_cluster:.4f})")
    else:
        print("\nIsotonic did NOT improve val cluster_mae — skipping iso submission.")
        print(f"RECOMMEND: submit {RAW_OUT}")

    # ----- Compare against existing submissions -----
    print("\n=== Distribution check ===")
    for n, p in [("phase1b", REPO_ROOT / "submission_phase1b.csv"),
                 ("catboost", REPO_ROOT / "submission_catboost.csv"),
                 ("patchtst_v2", REPO_ROOT / "submission_patchtst_v2.csv"),
                 ("phase5_3way", RAW_OUT)]:
        if p.is_file():
            df = pd.read_csv(p)
            pred_cols = [f"pred_week{h}" for h in range(1, 6)]
            arr = df[pred_cols].values
            print(f"  {n:>15s}: mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
                  f"%>=3={(arr >= 3).mean()*100:.2f}%")


if __name__ == "__main__":
    main()
