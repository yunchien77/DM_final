"""Phase 11 — n-way blend of wide-format submission CSVs.

Usage:
  python blend.py --inputs sub1.csv sub2.csv sub3.csv --weights 1 1 1 \\
                  --output sub_blend.csv [--compare-teammate path]

Default: equal-weight average of all --inputs. If --weights are given they
are normalized to sum 1.

Prints a distribution-vs-teammate diagnostic so we can pre-screen the blend
before submitting to Kaggle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PRED_COLS = [f"pred_week{h}" for h in (1, 2, 3, 4, 5)]


def load_sub(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "region_id" not in df.columns or not all(c in df.columns for c in PRED_COLS):
        raise ValueError(f"{path}: expected wide format with region_id + pred_week1..5")
    df["__order"] = df["region_id"].str.replace("R", "", regex=False).astype(int)
    return df.sort_values("__order").drop(columns=["__order"]).reset_index(drop=True)


def blend(inputs: list[str], weights: list[float], output: str, compare: str | None) -> None:
    subs = [load_sub(p) for p in inputs]
    n = len(subs)
    # Align on region_id
    rid = subs[0]["region_id"].to_numpy()
    for i, df in enumerate(subs[1:], 1):
        if (df["region_id"].to_numpy() != rid).any():
            raise ValueError(f"{inputs[i]} region_ids don't match {inputs[0]}")
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    print(f"Blending {n} submissions with weights {list(w.round(4))}:")
    for p, wi in zip(inputs, w):
        print(f"  {wi:.4f}  {p}")

    matrices = np.stack([df[PRED_COLS].to_numpy(np.float32) for df in subs], axis=0)  # (n, R, 5)
    blended = (w[:, None, None] * matrices).sum(axis=0)  # (R, 5)

    out = pd.DataFrame({
        "region_id": rid,
        **{c: blended[:, i].round(4) for i, c in enumerate(PRED_COLS)},
    })
    out.to_csv(output, index=False)
    vals = blended.ravel()
    print(f"\nWrote {output}")
    print(f"  rows: {len(out):,}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
          f"median={np.median(vals):.4f}  %<0.5={(vals<0.5).mean()*100:.1f}%  "
          f"%>=2={(vals>=2.0).mean()*100:.1f}%")

    # Per-input distribution comparison
    print("\nPer-input distribution:")
    for p, df in zip(inputs, subs):
        v = df[PRED_COLS].to_numpy().ravel()
        print(f"  {Path(p).name:35s}  mean={v.mean():.3f}  std={v.std():.3f}  "
              f"%<0.5={(v<0.5).mean()*100:.1f}%  %>=2={(v>=2.0).mean()*100:.1f}%")

    if compare:
        team = load_sub(compare)
        team_long = team.melt(id_vars="region_id", value_vars=PRED_COLS, value_name="team", var_name="wk")
        team_long["horizon"] = team_long["wk"].str.replace("pred_week", "", regex=False).astype(int)

        def metric(name: str, df: pd.DataFrame):
            long_df = df.melt(id_vars="region_id", value_vars=PRED_COLS, value_name="ours", var_name="wk")
            long_df["horizon"] = long_df["wk"].str.replace("pred_week", "", regex=False).astype(int)
            m = long_df.merge(team_long, on=["region_id", "horizon"])
            rho_s, _ = spearmanr(m["ours"], m["team"])
            rho_p = np.corrcoef(m["ours"], m["team"])[0, 1]
            diff = m["ours"] - m["team"]
            print(f"  {name:35s}  Spearman={rho_s:.4f}  Pearson={rho_p:.4f}  "
                  f"mean(ours-team)={diff.mean():+.4f}  MAE={diff.abs().mean():.4f}")

        print(f"\nPer-row comparison vs {Path(compare).name}:")
        for p, df in zip(inputs, subs):
            metric(Path(p).name, df)
        metric(f"BLEND → {Path(output).name}", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--compare-teammate", default="/mnt/1stHDD/juiyun/DMFP/submission_teammate_082.csv")
    args = p.parse_args()
    weights = args.weights or [1.0] * len(args.inputs)
    if len(weights) != len(args.inputs):
        p.error("--weights must have same length as --inputs")
    blend(args.inputs, weights, args.output, args.compare_teammate)


if __name__ == "__main__":
    main()
