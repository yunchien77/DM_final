"""Phase 7 — drought transition matrix.

Empirical Markov chain on weekly drought scores: encodes the per-region
"stickiness" pattern directly so we can (a) shrink noisy LGBM predictions
toward the empirical conditional, and (b) optionally upweight training rows
with rare 1-step transitions.

The conditional `P(score_{w+h} = j | score_w = i)` is more transferable across
years than the marginal `P(score)` — that's the assumption that distinguishes
this from iso_then_shift (a global level adjustment) which failed.

Public:
  build_transition_matrices(weekly_df, score_levels=range(6), horizons=range(1,6),
                            cluster_assignment=None)
      → np.ndarray of shape (n_clusters, n_horizons, n_levels, n_levels)
        plus the cluster_id_by_region dict (or None if not stratified).

  apply_transition_smoothing(pred_matrix, score_lag1, cluster_assignment,
                             transition_matrices, beta)
      → smoothed_pred_matrix; pred_matrix is (N, n_horizons).

  compute_transition_weight(score_lag1, score, cluster_ids, transition_matrices,
                            horizon=1, gamma=1.0)
      → per-row sample weight (1 + γ · surprise), where surprise = 1 − P(observed | prev).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MODELS_DIR

TRANSITION_MATRICES_PATH = MODELS_DIR / "transition_matrices.npz"


def build_transition_matrices(
    weekly_df: pd.DataFrame,
    score_levels: range = range(6),
    horizons: range = range(1, 6),
    cluster_assignment: dict[str, int] | None = None,
    n_clusters: int = 1,
    smoothing_alpha: float = 1.0,
) -> tuple[np.ndarray, dict[str, int] | None]:
    """Build per-(cluster, horizon) empirical transition matrices from training data.

    For each region's scored sequence, accumulate counts of (score[w], score[w+h])
    pairs into a cluster-specific count matrix. Apply Laplace smoothing
    (`smoothing_alpha`) and normalize rows to sum to 1.

    `cluster_assignment` maps region_id → cluster_id ∈ [0, n_clusters). If None,
    n_clusters defaults to 1 (single global matrix).

    Returns (matrices, cluster_assignment_used):
      matrices shape: (n_clusters, len(horizons), len(score_levels), len(score_levels))
        matrices[c, h_idx, i, j] = P(score[w+h] = j | score[w] = i)  in cluster c at horizon h
    """
    n_lv = len(score_levels)
    horizons_list = list(horizons)
    n_h = len(horizons_list)

    counts = np.full((n_clusters, n_h, n_lv, n_lv), smoothing_alpha, dtype=np.float64)

    for rid, sub in weekly_df.groupby("region_id", sort=False):
        sub_sorted = sub.sort_values("week_idx").reset_index(drop=True)
        scores = sub_sorted["score"].to_numpy()
        valid = ~pd.isna(scores)
        scores = np.where(valid, scores, -1).astype(np.int32)

        c = 0
        if cluster_assignment is not None:
            c = int(cluster_assignment.get(rid, 0))
            c = max(0, min(c, n_clusters - 1))

        n = len(scores)
        for h_idx, h in enumerate(horizons_list):
            if n <= h:
                continue
            src = scores[:n - h]
            tgt = scores[h:]
            mask = (src >= 0) & (tgt >= 0) & (src < n_lv) & (tgt < n_lv)
            src_v = src[mask]
            tgt_v = tgt[mask]
            np.add.at(counts[c, h_idx], (src_v, tgt_v), 1.0)

    # Normalize rows to probability distributions
    row_sums = counts.sum(axis=-1, keepdims=True)
    matrices = (counts / row_sums).astype(np.float32)
    return matrices, cluster_assignment


def save_transition_matrices(
    matrices: np.ndarray,
    cluster_assignment: dict[str, int] | None,
    horizons: list[int],
    score_levels: list[int],
) -> None:
    if cluster_assignment is None:
        rid_arr = np.array([], dtype=object)
        cid_arr = np.array([], dtype=np.int32)
    else:
        rid_arr = np.array(list(cluster_assignment.keys()), dtype=object)
        cid_arr = np.array(list(cluster_assignment.values()), dtype=np.int32)
    np.savez(
        TRANSITION_MATRICES_PATH,
        matrices=matrices,
        cluster_region_ids=rid_arr,
        cluster_ids=cid_arr,
        horizons=np.array(horizons, dtype=np.int32),
        score_levels=np.array(score_levels, dtype=np.int32),
    )


def load_transition_matrices() -> tuple[np.ndarray, dict[str, int] | None, list[int], list[int]]:
    data = np.load(TRANSITION_MATRICES_PATH, allow_pickle=True)
    matrices = data["matrices"]
    rid_arr = data["cluster_region_ids"]
    cid_arr = data["cluster_ids"]
    horizons = list(data["horizons"])
    score_levels = list(data["score_levels"])
    cluster_assignment = None
    if len(rid_arr) > 0:
        cluster_assignment = {str(r): int(c) for r, c in zip(rid_arr, cid_arr)}
    return matrices, cluster_assignment, horizons, score_levels


def markov_baseline(
    score_lag1: np.ndarray,
    cluster_ids: np.ndarray,
    matrices: np.ndarray,
    horizons: list[int],
    score_levels: list[int],
) -> np.ndarray:
    """Expected score under the empirical Markov chain, per (row, horizon).
    Returns shape (N, len(horizons))."""
    n_h = len(horizons)
    n_lv = len(score_levels)
    levels = np.asarray(score_levels, dtype=np.float32)
    s_prev = np.clip(np.asarray(score_lag1, dtype=np.int32), 0, n_lv - 1)
    c = np.asarray(cluster_ids, dtype=np.int32)
    c = np.clip(c, 0, matrices.shape[0] - 1)
    out = np.zeros((len(s_prev), n_h), dtype=np.float32)
    for h_idx in range(n_h):
        # matrices[c, h_idx, s_prev, :] is shape (N, n_lv)
        probs = matrices[c, h_idx, s_prev, :]
        out[:, h_idx] = (probs * levels[None, :]).sum(axis=1)
    return out


def apply_transition_smoothing(
    pred_matrix: np.ndarray,   # (N, n_horizons)
    score_lag1: np.ndarray,    # (N,)
    cluster_ids: np.ndarray,   # (N,)
    matrices: np.ndarray,      # (n_clusters, n_horizons, n_lv, n_lv)
    horizons: list[int],
    score_levels: list[int],
    beta: float = 0.8,
) -> np.ndarray:
    """final = β · pred + (1 − β) · markov_baseline, clipped to [0, 5]."""
    baseline = markov_baseline(score_lag1, cluster_ids, matrices, horizons, score_levels)
    out = beta * pred_matrix + (1.0 - beta) * baseline
    return np.clip(out, 0.0, 5.0).astype(np.float32)


def compute_transition_weight(
    score_lag1: np.ndarray,
    score: np.ndarray,
    cluster_ids: np.ndarray,
    matrices: np.ndarray,
    horizon: int = 1,
    horizons: list[int] | None = None,
    score_levels: list[int] | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """`1 + γ · surprise` where surprise = 1 − P(score | score_lag1) at horizon h.

    Rare transitions get higher weight — useful when the model under-fits
    transition rows (sudden score jumps).
    """
    if horizons is None:
        horizons = list(range(1, matrices.shape[1] + 1))
    if score_levels is None:
        score_levels = list(range(matrices.shape[2]))
    n_lv = matrices.shape[2]
    s_prev = np.clip(np.asarray(score_lag1, dtype=np.int32), 0, n_lv - 1)
    s_obs = np.clip(np.asarray(score, dtype=np.int32), 0, n_lv - 1)
    c = np.clip(np.asarray(cluster_ids, dtype=np.int32), 0, matrices.shape[0] - 1)
    h_idx = horizons.index(horizon)
    probs = matrices[c, h_idx, s_prev, s_obs]
    surprise = (1.0 - probs).clip(0.0, 1.0).astype(np.float32)
    return (1.0 + gamma * surprise).astype(np.float32)
