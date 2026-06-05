"""dmfp2 command-line entry.

    python -m dmfp2.cli validate            # baselines on the hot-holdout + diagnostics
    python -m dmfp2.cli baseline --kind region_mean   # write a submission for a baseline
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C
from . import weekly as W
from . import anchors as A
from . import validation as V
from . import pipeline as P
from .pipeline import write_submission


def _load_anchors(rebuild=False):
    tr_w, te_w = W.get_weekly(rebuild=rebuild)
    tr_anchors = A.build_train_anchors(tr_w)
    te_anchors = A.build_test_anchors(te_w)
    return tr_anchors, te_anchors


def _print_diag(tag: str, diag: dict):
    print(f"\n=== {tag} ===")
    print(f"  macro_MAE      = {diag['macro_mae']:.4f}   (per-horizon {diag['per_horizon_mae']})")
    print(f"  pred_mean      = {diag['pred_mean']:.4f}   true_mean = {diag['true_mean']:.4f}   "
          f"mean_bias = {diag['mean_bias']:+.4f}")
    print(f"  MAE by bucket  = {diag['mae_by_bucket']}")
    print(f"  frac_pred_low  = {diag['frac_pred_low']:.3f}   frac_true_zero = {diag['frac_true_zero']:.3f}")


def cmd_validate(args):
    print(C.config_banner())
    tr_anchors, te_anchors = _load_anchors(rebuild=args.rebuild)
    split = V.make_hot_holdout(tr_anchors, te_anchors)
    val_anchors = tr_anchors[split["val_mask"]].reset_index(drop=True)
    pool_anchors = tr_anchors[split["train_mask"]].reset_index(drop=True)

    print(f"\n[split] selection={split['selection']}  n_val={split['n_val']:,} (one/region)  "
          f"n_train_pool={split['n_train']:,}")
    print(f"[split] val true-mean = {split['val_true_mean']:.3f}  (test true-mean from LB = 1.2088)")
    print(f"[split] {split['hot_col']}: val = {split['val_temp_mean']:.2f}  test = {split['test_temp_mean']:.2f}  "
          f"train-pool = {split['train_temp_mean']:.2f}")

    # known Kaggle public-LB anchors for the submittable baselines (val↔LB tracking)
    KAGGLE_LB = {"zero": 1.2088, "region_mean": 0.9364}
    results = {}
    for kind in ["zero", "region_mean", "persistence"]:
        pred = V.baseline_preds(kind, pool_anchors, val_anchors)
        diag = V.evaluate(val_anchors, pred)
        lb = KAGGLE_LB.get(kind)
        suffix = f"   [Kaggle LB {lb} | val−LB {diag['macro_mae']-lb:+.4f}]" if lb else ""
        _print_diag(f"baseline: {kind}{suffix}", diag)
        V.log_val_lb(C.weekly_hash(), f"baseline_{kind}", diag, kaggle_lb=lb)
        results[kind] = diag

    out = C.REPORTS_DIR / "phase0_baselines.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] wrote {out}")


def cmd_baseline(args):
    tr_anchors, te_anchors = _load_anchors(rebuild=args.rebuild)
    pred = V.baseline_preds(args.kind, tr_anchors, te_anchors)
    out = args.out if args.out else C.SUBMISSION_PATH
    write_submission(te_anchors[C.ID_COL].to_numpy(), pred, out)


def cmd_train(args):
    P.run_train(submit=args.submit, tag=args.tag)


def main():
    p = argparse.ArgumentParser(prog="dmfp2")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("validate", help="baselines on the hot-holdout + diagnostics")
    pv.add_argument("--rebuild", action="store_true")
    pv.set_defaults(func=cmd_validate)

    pb = sub.add_parser("baseline", help="write a submission for a baseline")
    pb.add_argument("--kind", choices=["zero", "region_mean"], default="region_mean")
    pb.add_argument("--out", default=None, help="output csv path (default: submission.csv)")
    pb.add_argument("--rebuild", action="store_true")
    pb.set_defaults(func=cmd_baseline)

    pt = sub.add_parser("train", help="train Phase 1 LGBM + eval on hot-holdout")
    pt.add_argument("--submit", action="store_true", help="also write submission.csv from test preds")
    pt.add_argument("--tag", default="phase1")
    pt.set_defaults(func=cmd_train)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
