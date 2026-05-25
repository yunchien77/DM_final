"""Hill-climbing weight search for ensembling DMFP val predictions.

Given a list of (preds, truth, region_ids, week_idx) blocks from different
models, intersect on (region_id, week_idx), then greedily build a positive-
weight blend that minimizes per-cluster-averaged macro MAE on the overlap.

Per-cluster averaging — rather than raw row-mean macro MAE — keeps cluster 1
(180 high-severity regions, ~25% of Kaggle error) from being drowned out by
the much larger cluster 0 / cluster 2 row counts.

Usage as a library:

    from ensemble.hillclimb import (
        load_member, align_members, hill_climb,
    )

    members = [load_member(p) for p in ["code/models/val_preds.npz",
                                        "code/patchtst/models_pt/val_preds_patchtst.npz"]]
    names = ["lgbm", "patchtst"]
    aligned = align_members(members)
    weights, history = hill_climb(aligned["preds"], aligned["truth"],
                                  cluster_ids=aligned["cluster_ids"],
                                  names=names)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR  # for region_clusters.csv path


# ---------------------------------------------------------------------------
# IO + alignment
# ---------------------------------------------------------------------------

def load_member(path: str | Path) -> dict:
    """Return {preds (N,H), truth (N,H), region_ids (N,), week_idx (N,)}."""
    z = np.load(path, allow_pickle=True)
    return {
        "preds": z["preds"].astype(np.float32),
        "truth": z["truth"].astype(np.float32),
        "region_ids": z["region_ids"].astype(str),
        "week_idx": z["week_idx"].astype(np.int32),
    }


def _row_key(region_ids: np.ndarray, week_idx: np.ndarray) -> np.ndarray:
    """Compose a 1-D structured key array for fast set-intersection."""
    return np.array([f"{r}|{int(w)}" for r, w in zip(region_ids, week_idx)],
                    dtype=object)


def align_members(members: Iterable[dict]) -> dict:
    """Intersect all members on (region_id, week_idx). Returns:
        preds:       (M, N_common, H) stacked per-model preds
        truth:       (N_common, H)
        region_ids:  (N_common,)
        week_idx:    (N_common,)
        cluster_ids: (N_common,) int — from region_clusters.csv (or -1 if missing)
    """
    members = list(members)
    keys = [_row_key(m["region_ids"], m["week_idx"]) for m in members]
    common = set(keys[0])
    for k in keys[1:]:
        common = common.intersection(k)
    common_arr = np.array(sorted(common), dtype=object)

    # Index per member
    aligned_preds = []
    aligned_truth = None
    aligned_rid = None
    aligned_widx = None
    for k, m in zip(keys, members):
        order = pd.Series(np.arange(len(k)), index=k).reindex(common_arr).values
        if np.any(pd.isna(order)):
            raise RuntimeError("alignment failed — sanity check pd.Series.reindex result")
        order = order.astype(np.int64)
        aligned_preds.append(m["preds"][order])
        if aligned_truth is None:
            aligned_truth = m["truth"][order]
            aligned_rid = m["region_ids"][order]
            aligned_widx = m["week_idx"][order]
        else:
            # Verify truth matches across members (sanity)
            diff = np.max(np.abs(aligned_truth - m["truth"][order]))
            if diff > 1e-3:
                print(f"  WARNING: truth mismatch across members: max|Δ|={diff:.4f}")

    preds_stack = np.stack(aligned_preds, axis=0).astype(np.float32)  # (M, N, H)

    # Cluster ids
    cluster_csv = MODELS_DIR / "region_clusters.csv"
    if cluster_csv.is_file():
        tab = pd.read_csv(cluster_csv, dtype={"region_id": str})
        rid_to_cid = dict(zip(tab["region_id"].astype(str),
                              tab["region_cluster_id"].astype(int)))
        cluster_ids = np.array(
            [rid_to_cid.get(str(r), -1) for r in aligned_rid], dtype=np.int16,
        )
    else:
        cluster_ids = np.full(len(aligned_rid), -1, dtype=np.int16)

    print(f"[align] N_common={len(common_arr)} "
          f"(member sizes: {[m['preds'].shape[0] for m in members]})")
    return {
        "preds": preds_stack,
        "truth": aligned_truth,
        "region_ids": aligned_rid,
        "week_idx": aligned_widx,
        "cluster_ids": cluster_ids,
    }


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def macro_mae(blend: np.ndarray, truth: np.ndarray) -> float:
    """Mean of per-horizon MAE."""
    return float(np.mean([np.mean(np.abs(blend[:, h] - truth[:, h]))
                          for h in range(blend.shape[1])]))


def cluster_averaged_macro_mae(
    blend: np.ndarray, truth: np.ndarray, cluster_ids: np.ndarray,
) -> float:
    """Cluster-stratified objective: mean over clusters of (their macro MAE).
    Clusters with cid=-1 are ignored. Falls back to macro_mae when no cluster
    information is present."""
    uniq = [c for c in np.unique(cluster_ids) if c >= 0]
    if not uniq:
        return macro_mae(blend, truth)
    vals = []
    for c in uniq:
        m = cluster_ids == c
        if m.sum() == 0:
            continue
        vals.append(macro_mae(blend[m], truth[m]))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Hill climbing (Caruana 2004 ensemble selection)
# ---------------------------------------------------------------------------

def hill_climb(
    preds: np.ndarray,           # (M, N, H)
    truth: np.ndarray,           # (N, H)
    cluster_ids: np.ndarray | None = None,
    names: list[str] | None = None,
    n_iter: int = 200,
    step: float = 0.02,
    objective: str = "cluster",  # "cluster" or "macro"
) -> tuple[np.ndarray, list[dict]]:
    """Greedy weight search: at each step, find the (member, +step) move that
    most improves the objective. Stop when no move helps. Step=0.02 → 200 iters
    can give weights at 0.02 granularity that sum to ≤ 4.0.

    Returns (weights (M,), history). Weights are renormalized to sum to 1.
    """
    M, N, H = preds.shape
    if names is None:
        names = [f"m{i}" for i in range(M)]
    if cluster_ids is None or objective == "macro":
        score_fn = lambda b: macro_mae(b, truth)
    else:
        score_fn = lambda b: cluster_averaged_macro_mae(b, truth, cluster_ids)

    weights = np.zeros(M, dtype=np.float32)
    blend = np.zeros_like(truth, dtype=np.float32)
    cumulative = np.zeros((N, H), dtype=np.float32)
    total_w = 0.0

    history: list[dict] = []
    # Seed with single-model baselines
    for i in range(M):
        m_score = score_fn(preds[i])
        print(f"  baseline {names[i]:>15s}: {objective}_mae = {m_score:.4f}")

    for it in range(n_iter):
        best_i = -1
        best_score = score_fn(blend) if total_w > 0 else float("inf")
        for i in range(M):
            # Try adding step weight to member i:
            #   new_blend = (cumulative + step * preds[i]) / (total_w + step)
            new_cum = cumulative + step * preds[i]
            new_total = total_w + step
            cand = new_cum / new_total
            s = score_fn(cand)
            if s < best_score - 1e-6:
                best_score = s
                best_i = i

        if best_i == -1:
            print(f"  iter {it:3d}: no improving move; stopping")
            break
        cumulative = cumulative + step * preds[best_i]
        total_w += step
        weights[best_i] += step
        blend = cumulative / total_w
        history.append({"iter": it, "added": names[best_i], "score": best_score,
                        "weights": weights.copy().tolist()})
        if it < 20 or it % 10 == 0:
            wstr = " ".join(f"{n}={w/total_w:.3f}" for n, w in zip(names, weights))
            print(f"  iter {it:3d}: +{names[best_i]}  score={best_score:.5f}  "
                  f"weights={wstr}")

    final = weights / weights.sum() if weights.sum() > 0 else weights
    print(f"\n[hillclimb] final weights: " +
          ", ".join(f"{n}={w:.3f}" for n, w in zip(names, final)))
    print(f"[hillclimb] final {objective}_mae = {score_fn((preds * final[:, None, None]).sum(0)):.5f}")
    macro_baseline = macro_mae((preds * final[:, None, None]).sum(0), truth)
    print(f"[hillclimb] final row-macro_mae = {macro_baseline:.5f}")
    return final, history


# ---------------------------------------------------------------------------
# Submission blending (test-side)
# ---------------------------------------------------------------------------

def blend_submissions(
    submission_paths: list[str | Path],
    weights: np.ndarray,
    output_path: str | Path,
    clip_lo: float = 0.0,
    clip_hi: float = 5.0,
) -> pd.DataFrame:
    """Combine test-time submission CSVs with the given weights. CSVs must
    share region_id ordering and pred_week1..pred_week5 columns."""
    weights = np.asarray(weights, dtype=np.float32)
    assert abs(weights.sum() - 1.0) < 1e-4, f"weights must sum to 1, got {weights.sum()}"
    assert len(weights) == len(submission_paths)

    frames = [pd.read_csv(p, dtype={"region_id": str}) for p in submission_paths]
    base = frames[0][["region_id"]].copy()
    pred_cols = [f"pred_week{h}" for h in range(1, 6)]
    out_preds = np.zeros((len(base), len(pred_cols)), dtype=np.float32)
    for w, f in zip(weights, frames):
        # Align by region_id
        f = f.set_index("region_id").reindex(base["region_id"]).reset_index()
        arr = f[pred_cols].values.astype(np.float32)
        out_preds += w * arr
    out_preds = np.clip(out_preds, clip_lo, clip_hi)
    for i, c in enumerate(pred_cols):
        base[c] = out_preds[:, i]
    base.to_csv(output_path, index=False)
    print(f"Wrote {output_path}  shape={base.shape}")
    print(base.head().to_string(index=False))
    return base


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Hill-climb ensemble over val_preds.npz files.")
    p.add_argument("--members", nargs="+", required=True,
                   help="Pairs of name=path, e.g. lgbm=code/models/val_preds.npz")
    p.add_argument("--objective", choices=["cluster", "macro"], default="cluster")
    p.add_argument("--step", type=float, default=0.02)
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--save-weights", type=str, default=None,
                   help="If set, write {names, weights, objective, score} to this JSON.")
    args = p.parse_args()

    names, paths = [], []
    for entry in args.members:
        if "=" not in entry:
            raise SystemExit(f"--members expects name=path, got {entry}")
        n, pth = entry.split("=", 1)
        names.append(n)
        paths.append(pth)

    print(f"Loading {len(paths)} members...")
    members = [load_member(p) for p in paths]
    for n, m in zip(names, members):
        print(f"  {n:>15s}: N={len(m['preds'])} H={m['preds'].shape[1]}")
    aligned = align_members(members)
    weights, history = hill_climb(
        aligned["preds"], aligned["truth"],
        cluster_ids=aligned["cluster_ids"],
        names=names, n_iter=args.n_iter, step=args.step,
        objective=args.objective,
    )

    if args.save_weights:
        Path(args.save_weights).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_weights, "w") as f:
            json.dump({
                "names": names,
                "paths": paths,
                "weights": weights.tolist(),
                "objective": args.objective,
                "step": args.step,
                "n_common_val_rows": int(aligned["preds"].shape[1]),
            }, f, indent=2)
        print(f"Saved weights -> {args.save_weights}")


if __name__ == "__main__":
    main()
