"""Diagnostic: per-model val stats + pairwise correlations + grid-search blend.

Helps pick a single blend by exposing the val-side trade-off between bias
(macro MAE) and diversity (pairwise correlation) of the candidate members.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from ensemble.hillclimb import load_member, align_members, macro_mae, cluster_averaged_macro_mae


def main():
    members = [
        ("lgbm",     load_member("models/val_preds.npz")),
        ("patchtst", load_member("patchtst/models_pt/val_preds_patchtst.npz")),
        ("catboost", load_member("models/val_preds_catboost.npz")),
    ]
    names = [n for n, _ in members]

    aligned = align_members([m for _, m in members])
    preds = aligned["preds"]   # (M, N, H)
    truth = aligned["truth"]
    cl = aligned["cluster_ids"]
    print(f"\nN_common = {preds.shape[1]}, H = {preds.shape[2]}")

    print("\n=== Per-model val MAE ===")
    print(f"{'model':>10s}  {'macro_mae':>10s}  {'cluster_mae':>11s}  {'mean':>6s}  {'median':>6s}  {'%>=3':>5s}")
    for i, n in enumerate(names):
        p = preds[i]
        print(f"{n:>10s}  {macro_mae(p, truth):10.4f}  "
              f"{cluster_averaged_macro_mae(p, truth, cl):11.4f}  "
              f"{p.mean():6.3f}  {np.median(p):6.3f}  "
              f"{(p >= 3).mean()*100:5.2f}%")

    print("\n=== Per-horizon MAE per model ===")
    for h in range(preds.shape[2]):
        row = " | ".join(f"{n}={np.mean(np.abs(preds[i, :, h] - truth[:, h])):.4f}"
                          for i, n in enumerate(names))
        print(f"  h{h+1}: {row}")

    print("\n=== Pairwise prediction correlation (averaged over horizons) ===")
    M = len(names)
    corr = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            cs = [np.corrcoef(preds[i, :, h], preds[j, :, h])[0, 1]
                  for h in range(preds.shape[2])]
            corr[i, j] = float(np.mean(cs))
    print("       " + "  ".join(f"{n:>8s}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:>5s}  " + "  ".join(f"{corr[i,j]:8.3f}" for j in range(M)))

    print("\n=== Per-cluster MAE per model ===")
    uniq = sorted([c for c in np.unique(cl) if c >= 0])
    print(f"  cluster  n_rows  " + "  ".join(f"{n:>10s}" for n in names))
    for c in uniq:
        mask = cl == c
        row = "  ".join(f"{macro_mae(preds[i, mask], truth[mask]):10.4f}"
                        for i in range(M))
        print(f"  c{c:<6d}  {mask.sum():>6d}  {row}")

    print("\n=== Grid search (cluster_mae objective) ===")
    grid = np.arange(0.0, 1.01, 0.05)
    rows = []
    for wL in grid:
        for wC in grid:
            wP = 1.0 - wL - wC
            if wP < -1e-6 or wP > 1.0 + 1e-6:
                continue
            wP = max(0.0, wP)
            blend = wL * preds[0] + wP * preds[1] + wC * preds[2]
            rows.append({
                "wL": round(wL, 2), "wP": round(wP, 2), "wC": round(wC, 2),
                "cluster_mae": cluster_averaged_macro_mae(blend, truth, cl),
                "macro_mae": macro_mae(blend, truth),
            })
    df = pd.DataFrame(rows).sort_values("cluster_mae").reset_index(drop=True)
    print("Top 15 by cluster_mae:")
    print(df.head(15).to_string(index=False))

    print("\nTop 15 by macro_mae:")
    df2 = df.sort_values("macro_mae").reset_index(drop=True)
    print(df2.head(15).to_string(index=False))

    print("\n=== Highlighted candidate blends ===")
    candidates = [
        ("100% lgbm",                {"lgbm": 1.0, "patchtst": 0.0, "catboost": 0.0}),
        ("100% catboost",            {"lgbm": 0.0, "patchtst": 0.0, "catboost": 1.0}),
        ("100% patchtst",            {"lgbm": 0.0, "patchtst": 1.0, "catboost": 0.0}),
        ("50/50 lgbm+catboost",      {"lgbm": 0.5, "patchtst": 0.0, "catboost": 0.5}),
        ("60/40 lgbm+catboost",      {"lgbm": 0.6, "patchtst": 0.0, "catboost": 0.4}),
        ("45/45/10 L+C+P",           {"lgbm": 0.45, "patchtst": 0.10, "catboost": 0.45}),
        ("40/40/20 L+C+P",           {"lgbm": 0.40, "patchtst": 0.20, "catboost": 0.40}),
        ("33/33/33 L+C+P",           {"lgbm": 0.333, "patchtst": 0.334, "catboost": 0.333}),
        ("50/30/20 L+C+P",           {"lgbm": 0.50, "patchtst": 0.20, "catboost": 0.30}),
    ]
    print(f"{'blend':30s}  {'cluster_mae':>11s}  {'macro_mae':>10s}  "
          f"{'+gap est Kaggle':>16s}")
    # Per-member gap estimates: LGBM (Phase 3*) 0.46, PatchTST_v2 0.62, CatBoost ~0.46 (assumed)
    gaps = {"lgbm": 0.46, "patchtst": 0.62, "catboost": 0.46}
    for label, ws in candidates:
        blend = sum(ws[n] * preds[i] for i, n in enumerate(names))
        cm = cluster_averaged_macro_mae(blend, truth, cl)
        mm = macro_mae(blend, truth)
        gap_est = sum(ws[n] * gaps[n] for n in names)
        est_kaggle = cm + gap_est
        print(f"  {label:30s}  {cm:11.4f}  {mm:10.4f}  {est_kaggle:16.4f}")


if __name__ == "__main__":
    main()
