"""Phase 7 — feature analysis (replaces the legacy feature_analysis.py).

Load the cached training matrix and report per-feature diagnostics:
  - distribution (mean/std/min/max/% zero)
  - Spearman ρ with target_w1, target_w3, target_w5 (signal strength)
  - year_mean_range_z (year stability — high = drifts across years → Kaggle risk)
  - block grouping (score / climate / windowed)

Output: stdout + code/diagnostics/feature_analysis.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache import load_from_cache, feature_cache_key
from features_score import (
    SCORE_LAG_FEATURE_COLS, REGION_FEATURE_COLS, CALENDAR_FEATURE_COLS,
    SCORE_CLIM_FEATURE_COLS,
)
from features_climate import CLIMATE_FEATURE_COLS
from features_windowed import windowed_feature_cols

WINDOWED_COLS = windowed_feature_cols()

OUT_DIR = Path(__file__).resolve().parent / "diagnostics"
OUT_FILE = OUT_DIR / "feature_analysis.md"
SAMPLE_FOR_SPEARMAN = 100_000

TARGETS = ("target_w1", "target_w3", "target_w5")


BLOCKS: dict[str, list[str]] = {
    "score_lag": list(SCORE_LAG_FEATURE_COLS),
    "region_history": list(REGION_FEATURE_COLS),
    "calendar": list(CALENDAR_FEATURE_COLS),
    "score_climatology": list(SCORE_CLIM_FEATURE_COLS),
    "climate": list(CLIMATE_FEATURE_COLS),
    "windowed": list(WINDOWED_COLS),
}


def _block_of(feat: str) -> str:
    for name, cols in BLOCKS.items():
        if feat in cols:
            return name
    return "other"


def _year_idx(week_idx: np.ndarray) -> np.ndarray:
    return (week_idx // 52).astype(np.int32)


def summarize(name: str, vals: np.ndarray, targets: dict, year_idx: np.ndarray, rng) -> dict:
    n = len(vals)
    finite = np.isfinite(vals)
    row = {
        "feature": name,
        "n": n,
        "mean": float(np.nanmean(vals)) if finite.any() else 0.0,
        "std": float(np.nanstd(vals)) if finite.any() else 0.0,
        "min": float(np.nanmin(vals)) if finite.any() else 0.0,
        "max": float(np.nanmax(vals)) if finite.any() else 0.0,
        "pct_zero": float(np.mean(vals == 0.0) * 100),
        "pct_nan": float(np.mean(~finite) * 100),
    }
    sample_n = min(SAMPLE_FOR_SPEARMAN, n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    vs = vals[sample_idx]
    for tn, tarr in targets.items():
        ts = tarr[sample_idx]
        mask = np.isfinite(vs) & np.isfinite(ts) & (np.std(vs) > 1e-9) & (np.std(ts) > 1e-9)
        if mask.sum() > 100:
            try:
                rho, _ = spearmanr(vs[mask], ts[mask])
                row[f"rho_{tn}"] = float(rho) if np.isfinite(rho) else 0.0
            except Exception:
                row[f"rho_{tn}"] = 0.0
        else:
            row[f"rho_{tn}"] = 0.0
    yr_means = pd.DataFrame({"v": vals, "y": year_idx}).groupby("y")["v"].mean()
    row["year_mean_range"] = float(yr_means.max() - yr_means.min())
    row["year_mean_range_z"] = float(row["year_mean_range"] / max(row["std"], 1e-6))
    return row


def main() -> None:
    cached = load_from_cache("train_features", feature_cache_key())
    if cached is None:
        print("ERROR: no cached feature matrix found. Run `python train.py` first.")
        sys.exit(1)
    train_split = cached["train_split"]
    feature_cols = cached["feature_cols"]
    print(f"Loaded train_split: {train_split.shape}, features: {len(feature_cols)}")

    missing_targets = [t for t in TARGETS if t not in train_split.columns]
    if missing_targets:
        print(f"ERROR: targets missing from cache: {missing_targets}")
        sys.exit(1)

    targets = {t: train_split[t].to_numpy(np.float32) for t in TARGETS}
    year_idx = _year_idx(train_split["week_idx"].to_numpy(np.int64))
    rng = np.random.default_rng(42)

    rows = []
    for c in feature_cols:
        vals = train_split[c].to_numpy(np.float32)
        row = summarize(c, vals, targets, year_idx, rng)
        row["block"] = _block_of(c)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("rho_target_w1", key=lambda s: s.abs(), ascending=False)

    OUT_DIR.mkdir(exist_ok=True)
    # Fall back to to_string when tabulate isn't installed.
    try:
        table_top = df.head(30).round(4).to_markdown(index=False)
        table_full = df.round(4).to_markdown(index=False)
    except ImportError:
        table_top = df.head(30).round(4).to_string(index=False)
        table_full = df.round(4).to_string(index=False)
    OUT_FILE.write_text("\n".join([
        "# Phase 7 — feature analysis",
        "",
        f"Cache: train_split shape = {train_split.shape}",
        f"Features: {len(feature_cols)}",
        "",
        "## By block",
        "",
        *[f"- **{name}**: {len(cols)} features in spec" for name, cols in BLOCKS.items()],
        "",
        "## Top 30 by |Spearman ρ_target_w1|",
        "",
        "```",
        table_top,
        "```",
        "",
        "## Full table",
        "",
        "```",
        table_full,
        "```",
    ]))
    print(f"Saved {OUT_FILE}")
    print("\nTop 15 by |ρ_target_w1|:")
    print(df.head(15)[["block", "feature", "rho_target_w1", "rho_target_w3", "rho_target_w5",
                       "pct_zero", "year_mean_range_z"]].to_string(index=False))


if __name__ == "__main__":
    main()
