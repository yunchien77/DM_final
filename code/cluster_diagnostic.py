"""Phase 8 — region clustering diagnostic.

Standalone script (run AFTER train.py builds the feature cache + proxy_score):

  python cluster_diagnostic.py

Computes per-region "meteo signature" (mean of every climate + windowed
feature + proxy_score over the region's training rows), runs K-means for
K ∈ {2..8}, plots silhouette curve + PCA-2D scatter colored by
region_score_mean, and applies a binary decision rule:

  silhouette ≥ 0.25 AND severe-vs-mild cluster mean(region_score_mean)
  ratio ≥ 1.5× AND no cluster has < 5% of regions.

If all three hold, the chosen K's cluster assignment is written to
models/meteo_cluster_table.csv. train.py + predict.py auto-pick it up via
USE_METEO_CLUSTER when the file exists.

Output:
  code/diagnostics/region_clustering.md
  code/diagnostics/region_clustering_k.png
  models/meteo_cluster_table.csv  (only if decision rule passes)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache import load_from_cache, feature_cache_key
from config import MODELS_DIR
from features_climate import CLIMATE_FEATURE_COLS
from features_windowed import windowed_feature_cols
from proxy_score import PROXY_FEATURE_COL

OUT_DIR = Path(__file__).resolve().parent / "diagnostics"
MD_PATH = OUT_DIR / "region_clustering.md"
PNG_PATH = OUT_DIR / "region_clustering_k.png"
CLUSTER_TABLE_PATH = MODELS_DIR / "meteo_cluster_table.csv"

K_RANGE = list(range(2, 9))
SILHOUETTE_SAMPLE = 1500
SILHOUETTE_THRESHOLD = 0.25
SEVERITY_RATIO_THRESHOLD = 1.5
MIN_CLUSTER_FRAC = 0.05
SEED = 42


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    cached = load_from_cache("train_features", feature_cache_key())
    if cached is None:
        print("No cached train_features — run train.py first.")
        sys.exit(1)
    train_split: pd.DataFrame = cached["train_split"]

    # Feature set: climate + windowed + proxy
    signature_cols = (
        [c for c in CLIMATE_FEATURE_COLS if c in train_split.columns]
        + [c for c in windowed_feature_cols() if c in train_split.columns]
    )
    if PROXY_FEATURE_COL in train_split.columns:
        signature_cols.append(PROXY_FEATURE_COL)
    print(f"Signature dims: {len(signature_cols)} columns")
    if not signature_cols:
        print("ERROR: no climate/windowed columns in cache.")
        sys.exit(1)

    # Per-region signature: mean of every feature over the region's rows.
    print(f"Aggregating per-region signatures over {len(train_split):,} rows...")
    per_region = (
        train_split.groupby("region_id", sort=False)[signature_cols].mean().reset_index()
    )
    # Region-mean score (severity reference for coloring)
    region_score_mean = (
        train_split.groupby("region_id", sort=False)["target_w1"].mean().rename("region_score_mean")
    )
    per_region = per_region.merge(region_score_mean, on="region_id", how="left")
    print(f"Per-region signature: {per_region.shape}")

    X = StandardScaler().fit_transform(per_region[signature_cols].to_numpy(np.float32))
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)

    # K-means sweep
    metrics_rows = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
        labels = km.labels_
        labels_by_k[k] = labels
        sil = silhouette_score(X[sample_idx], labels[sample_idx])
        sizes = np.bincount(labels)
        # Per-cluster severity = mean of region_score_mean within cluster
        sev_by_cluster = (
            per_region.assign(__c=labels).groupby("__c")["region_score_mean"].mean().to_numpy()
        )
        sev_ratio = (
            float(sev_by_cluster.max() / max(sev_by_cluster.min(), 1e-6)) if len(sev_by_cluster) else 1.0
        )
        min_frac = sizes.min() / len(X)
        metrics_rows.append({
            "k": k,
            "silhouette": float(sil),
            "min_cluster_size": int(sizes.min()),
            "max_cluster_size": int(sizes.max()),
            "min_cluster_frac": float(min_frac),
            "severity_ratio": sev_ratio,
            "sev_per_cluster": [round(float(s), 3) for s in sev_by_cluster.tolist()],
        })
        print(f"  k={k}: silhouette={sil:.3f}  sizes={sizes.tolist()}  "
              f"sev_per_cluster={[round(float(s), 3) for s in sev_by_cluster.tolist()]}")
    metrics = pd.DataFrame(metrics_rows)

    # Decision: choose K with highest silhouette satisfying all three rules.
    eligible = metrics[
        (metrics["silhouette"] >= SILHOUETTE_THRESHOLD)
        & (metrics["severity_ratio"] >= SEVERITY_RATIO_THRESHOLD)
        & (metrics["min_cluster_frac"] >= MIN_CLUSTER_FRAC)
    ]
    if not eligible.empty:
        best_k = int(eligible.sort_values("silhouette", ascending=False).iloc[0]["k"])
        chosen = labels_by_k[best_k]
        # Order clusters by severity ascending so cluster_id == 0 is the
        # mildest, cluster_id == K-1 is the most severe — interpretable.
        sev_by_cluster = (
            per_region.assign(__c=chosen)
                      .groupby("__c")["region_score_mean"].mean()
                      .sort_values()
        )
        remap = {old_c: new_c for new_c, old_c in enumerate(sev_by_cluster.index.to_numpy())}
        new_labels = np.array([remap[int(c)] for c in chosen], dtype=np.int32)
        cluster_table = pd.DataFrame({
            "region_id": per_region["region_id"].astype(str).to_numpy(),
            "meteo_cluster_id": new_labels,
        })
        CLUSTER_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cluster_table.to_csv(CLUSTER_TABLE_PATH, index=False)
        decision_msg = (
            f"**Decision:** K={best_k}. Cluster table written to {CLUSTER_TABLE_PATH}. "
            f"train.py + predict.py will pick this up on the next run via USE_METEO_CLUSTER=True."
        )
    else:
        best_k = None
        cluster_table = None
        decision_msg = (
            "**Decision:** SKIP. No K satisfied all three rules "
            f"(silhouette ≥ {SILHOUETTE_THRESHOLD}, severity ratio ≥ {SEVERITY_RATIO_THRESHOLD}×, "
            f"min cluster frac ≥ {MIN_CLUSTER_FRAC*100:.0f}%). No cluster table written. "
            f"Delete an existing models/meteo_cluster_table.csv if you want predictions to ignore prior runs."
        )

    # PCA-2D scatter for selected K values (or all if small)
    pca = PCA(n_components=2).fit(X)
    coords = pca.transform(X)
    var = pca.explained_variance_ratio_

    panels = [2, 3, 5, 8] if 8 in K_RANGE else K_RANGE
    panels = [k for k in panels if k in labels_by_k]
    n_pan = len(panels)
    fig, axes = plt.subplots(2, max(n_pan, 1), figsize=(4 * max(n_pan, 1), 9), dpi=120)
    if n_pan == 1:
        axes = axes.reshape(2, 1)

    # Top row: cluster colors
    for i, k in enumerate(panels):
        ax = axes[0, i]
        labels = labels_by_k[k]
        for c in range(k):
            mask = labels == c
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.7, label=f"c{c} n={mask.sum()}")
        ax.set_title(f"K={k}  silhouette={metrics[metrics['k']==k]['silhouette'].iloc[0]:.3f}")
        if i == 0:
            ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
        if k <= 5:
            ax.legend(fontsize=7, loc="best")

    # Bottom row: same panels but colored by region_score_mean (severity)
    sev_arr = per_region["region_score_mean"].to_numpy(np.float32)
    for i, k in enumerate(panels):
        ax = axes[1, i]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=sev_arr, s=8, alpha=0.7, cmap="viridis")
        ax.set_title(f"K={k} — colored by region_score_mean")
        if i == 0:
            ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Phase 8 meteo-feature region clustering  "
        f"(n_regions={len(per_region)}, signature_dims={len(signature_cols)})",
        fontsize=13, y=0.995,
    )
    fig.savefig(PNG_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PNG_PATH}")

    # Markdown report
    lines = [
        "# Phase 8 region clustering diagnostic",
        "",
        f"- Per-region signature dims: **{len(signature_cols)}**  (climate + windowed{' + proxy' if PROXY_FEATURE_COL in signature_cols else ''})",
        f"- Regions: **{len(per_region):,}**",
        f"- Decision rule: silhouette ≥ {SILHOUETTE_THRESHOLD} AND "
        f"severity ratio ≥ {SEVERITY_RATIO_THRESHOLD}× AND min_cluster_frac ≥ {MIN_CLUSTER_FRAC*100:.0f}%",
        "",
        "## K sweep",
        "",
        metrics.round(4).to_string(index=False),
        "",
        decision_msg,
        "",
        f"![K-means PCA scatter]({PNG_PATH.name})",
    ]
    MD_PATH.write_text("\n".join(lines))
    print(f"Saved {MD_PATH}")
    print(decision_msg)


if __name__ == "__main__":
    main()
