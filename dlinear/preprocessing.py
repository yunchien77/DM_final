"""
Daily-level data preprocessing for DMFP.

Three transformations, all leakage-safe (artifacts are fit on a training-fold
DataFrame supplied by the caller — same contract as features.compute_region_stats):

  1. Winsorization at conservative p1/p99 per feature (Phase B-1)
  2. Log1p / signed-log1p / sqrt for heavily skewed features (Phase B-2)
  3. Robust missing-value imputation: per-region median fill, global median fallback
     (replaces the current `fillna(0)` blanket strategy in data_pipeline.py)

The module is standalone in this iteration: it is NOT wired into train.py or
predict.py. The Phase C toggle `USE_PREPROCESSING` in config.py governs
future integration — the intended insertion point is inside
`data_pipeline.load_and_aggregate_daily_to_weekly`, right after the chunked
read+concat and before the `groupby('region_id')` aggregation:

    df = preprocessing.apply_pipeline(df, artifacts)   # ← future hook

Saved artifacts live under code/models/preprocessing.{json,csv} so test-time
inference can apply the same fit.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import METEO_FEATURES, MODELS_DIR


PREPROC_JSON = MODELS_DIR / "preprocessing.json"
PREPROC_IMPUTATION_CSV = MODELS_DIR / "preprocessing_imputation.csv"
PREPROC_QUANTILES_NPZ = MODELS_DIR / "preprocessing_quantiles.npz"


# ---------------------------------------------------------------------------
# 1. Winsorization (conservative p1/p99)
# ---------------------------------------------------------------------------

def compute_winsor_bounds(
    df_train: pd.DataFrame,
    features: list[str] = METEO_FEATURES,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
) -> dict[str, tuple[float, float]]:
    """
    Compute per-feature (low, high) clip bounds from the training fold only.

    df_train: daily-rows DataFrame containing METEO_FEATURES columns.
    Returns: {feature_name: (low, high)} dict.
    """
    bounds = {}
    for f in features:
        col = df_train[f].dropna()
        if len(col) == 0:
            bounds[f] = (float("-inf"), float("inf"))
            continue
        lo = float(col.quantile(q_lo))
        hi = float(col.quantile(q_hi))
        bounds[f] = (lo, hi)
    return bounds


def apply_winsorization(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    features: list[str] = METEO_FEATURES,
) -> pd.DataFrame:
    """Clip each feature to its bounds; returns a copy (does not mutate input)."""
    out = df.copy()
    for f in features:
        if f not in bounds or f not in out.columns:
            continue
        lo, hi = bounds[f]
        out[f] = out[f].clip(lo, hi)
    return out


# ---------------------------------------------------------------------------
# 2. Heavy-tail transforms (log1p / signed-log1p / sqrt)
# ---------------------------------------------------------------------------

def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


_TRANSFORMS = {
    "log1p":        lambda x: np.log1p(np.maximum(x, 0.0)),
    "signed_log1p": _signed_log1p,
    "sqrt":         lambda x: np.sign(x) * np.sqrt(np.abs(x)),
}


def apply_transforms(
    df: pd.DataFrame,
    log_features: dict[str, str],
) -> pd.DataFrame:
    """
    Apply a named transform per feature, e.g.
        {"prec": "log1p", "surf_pre": "signed_log1p"}
    Returns a copy.
    """
    out = df.copy()
    for f, kind in log_features.items():
        if kind not in _TRANSFORMS:
            raise ValueError(f"Unknown transform {kind!r} for {f!r}")
        if f in out.columns:
            out[f] = _TRANSFORMS[kind](out[f].values.astype(np.float64)).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 3. Robust NaN imputation: per-region median, fallback to global median
# ---------------------------------------------------------------------------

def compute_imputation_table(
    df_train: pd.DataFrame,
    by: str = "region_id",
    features: list[str] = METEO_FEATURES,
) -> pd.DataFrame:
    """
    Build a per-region median table. Returns a DataFrame indexed by region_id
    with one column per feature plus a synthetic 'GLOBAL' row holding the
    pooled global median (used as fallback when a region has no data).
    """
    region_medians = df_train.groupby(by)[features].median()
    global_medians = df_train[features].median()
    region_medians.loc["GLOBAL"] = global_medians
    return region_medians.astype(np.float32)


def apply_imputation(
    df: pd.DataFrame,
    imputation_table: pd.DataFrame,
    by: str = "region_id",
    features: list[str] = METEO_FEATURES,
) -> pd.DataFrame:
    """
    Replace NaN in each feature with the per-region median; if a region is not
    in the table, fall back to the global median row.
    """
    out = df.copy()
    if by not in out.columns:
        return out  # nothing we can join on
    # Merge per-region medians as suffix columns, fill, then drop.
    medians_no_global = imputation_table.drop(index="GLOBAL", errors="ignore")
    merge = out[[by]].merge(
        medians_no_global, left_on=by, right_index=True, how="left"
    )
    global_row = imputation_table.loc["GLOBAL"] if "GLOBAL" in imputation_table.index else None
    for f in features:
        if f not in out.columns:
            continue
        fill = merge[f].values
        if global_row is not None:
            # If a region had no data → its row in `merge` is NaN; fill with global.
            fill = np.where(pd.isnull(fill), float(global_row[f]), fill)
        out[f] = out[f].fillna(pd.Series(fill, index=out.index))
        # Any cell that's still NaN (e.g., region matched but its median is NaN)
        # falls back to global median.
        if global_row is not None and out[f].isnull().any():
            out[f] = out[f].fillna(float(global_row[f]))
    return out


# ---------------------------------------------------------------------------
# 4. Rank / quantile normalization across train ∪ test
# ---------------------------------------------------------------------------

def compute_rank_quantiles(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    n_quantiles: int = 1000,
) -> dict[str, np.ndarray]:
    """
    Build a shared ECDF reference (per feature) from the train ∪ test pooled
    daily values. Returns a dict {feature: quantile_anchors} where
    `quantile_anchors` is a length-`n_quantiles` array of sorted values.

    Anchors are the empirical quantiles at uniform probability levels — i.e.,
    np.quantile(pool, np.linspace(0, 1, n_quantiles)). Apply with
    `apply_rank_normalization`, which maps each value to its rank-quantile via
    np.searchsorted.
    """
    out = {}
    probs = np.linspace(0.0, 1.0, n_quantiles)
    for f in features:
        tr = df_train[f].dropna().values
        te = df_test[f].dropna().values
        if len(tr) == 0 and len(te) == 0:
            out[f] = np.array([0.0], dtype=np.float32)
            continue
        pooled = np.concatenate([tr, te])
        out[f] = np.quantile(pooled, probs).astype(np.float32)
    return out


def apply_rank_normalization(
    df: pd.DataFrame,
    quantile_table: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Map each value to its pooled rank-quantile in [0, 1] via searchsorted."""
    out = df.copy()
    for f, anchors in quantile_table.items():
        if f not in out.columns or len(anchors) <= 1:
            continue
        vals = out[f].values.astype(np.float64)
        ranks = np.searchsorted(anchors, vals, side="left").astype(np.float32)
        out[f] = (ranks / (len(anchors) - 1)).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_preprocessing_artifacts(
    bounds: dict,
    log_features: dict[str, str],
    imputation_table: pd.DataFrame,
    quantile_table: dict[str, np.ndarray] | None = None,
    json_path: Path = PREPROC_JSON,
    csv_path: Path = PREPROC_IMPUTATION_CSV,
    npz_path: Path = PREPROC_QUANTILES_NPZ,
):
    payload = {
        "bounds": {f: [float(lo), float(hi)] for f, (lo, hi) in bounds.items()},
        "log_features": log_features,
        "rank_features": list(quantile_table.keys()) if quantile_table else [],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    imputation_table.to_csv(csv_path, index_label="region_id")
    print(f"[preproc] Saved bounds + log_features → {json_path}")
    print(f"[preproc] Saved imputation table → {csv_path}")
    if quantile_table:
        np.savez(npz_path, **{f: a for f, a in quantile_table.items()})
        print(f"[preproc] Saved {len(quantile_table)} quantile tables → {npz_path}")


def load_preprocessing_artifacts(
    json_path: Path = PREPROC_JSON,
    csv_path: Path = PREPROC_IMPUTATION_CSV,
    npz_path: Path = PREPROC_QUANTILES_NPZ,
) -> tuple[dict, dict, pd.DataFrame, dict[str, np.ndarray]]:
    with open(json_path) as fh:
        payload = json.load(fh)
    bounds = {f: (float(lo), float(hi)) for f, (lo, hi) in payload["bounds"].items()}
    log_features = payload.get("log_features", {})
    imputation_table = pd.read_csv(csv_path).set_index("region_id")
    quantile_table: dict[str, np.ndarray] = {}
    if npz_path.exists():
        with np.load(npz_path) as data:
            quantile_table = {k: data[k] for k in data.files}
    return bounds, log_features, imputation_table, quantile_table


# ---------------------------------------------------------------------------
# Convenience: full pipeline in one call (intended for future train.py hook)
# ---------------------------------------------------------------------------

def apply_pipeline(
    df: pd.DataFrame,
    bounds: dict,
    log_features: dict[str, str],
    imputation_table: pd.DataFrame,
    quantile_table: dict[str, np.ndarray] | None = None,
    features: list[str] = METEO_FEATURES,
) -> pd.DataFrame:
    """Impute → winsorize → transform → rank-normalize (if supplied).

    Imputation first so subsequent quantile / log operations see clean inputs.
    Rank-normalization runs last because it overwrites the original feature
    values with their pooled-ECDF ranks (in [0, 1]).
    """
    df = apply_imputation(df, imputation_table, features=features)
    df = apply_winsorization(df, bounds, features=features)
    df = apply_transforms(df, log_features)
    if quantile_table:
        df = apply_rank_normalization(df, quantile_table)
    return df


# ---------------------------------------------------------------------------
# Self-test (round-trip + assertions on synthetic data)
# ---------------------------------------------------------------------------

def _self_test():
    print("[preproc] Self-test…")
    rng = np.random.default_rng(0)
    n_rows = 5000
    regions = np.array([f"R{r}" for r in rng.integers(1, 11, n_rows)])
    df = pd.DataFrame({
        "region_id": regions,
        **{f: rng.normal(loc=10.0, scale=2.0, size=n_rows).astype(np.float32)
           for f in METEO_FEATURES},
    })
    # Inject 1% NaN per feature.
    for f in METEO_FEATURES:
        mask = rng.random(n_rows) < 0.01
        df.loc[mask, f] = np.nan

    bounds = compute_winsor_bounds(df, q_lo=0.01, q_hi=0.99)
    clipped = apply_winsorization(df, bounds)
    for f in METEO_FEATURES:
        lo, hi = bounds[f]
        col = clipped[f].dropna()
        assert col.min() >= lo - 1e-6 and col.max() <= hi + 1e-6, f"winsor bounds fail for {f}"

    log_features = {METEO_FEATURES[0]: "log1p", METEO_FEATURES[1]: "signed_log1p", METEO_FEATURES[2]: "sqrt"}
    transformed = apply_transforms(clipped, log_features)
    assert transformed[METEO_FEATURES[0]].isnull().sum() == clipped[METEO_FEATURES[0]].isnull().sum()

    tbl = compute_imputation_table(df)
    assert "GLOBAL" in tbl.index, "imputation table missing GLOBAL row"
    imputed = apply_imputation(df, tbl)
    for f in METEO_FEATURES:
        assert imputed[f].isnull().sum() == 0, f"NaN still present in {f} after imputation"

    # Persistence round-trip (to /tmp so we don't pollute the real artifacts dir)
    tmp_dir = Path("/tmp/dmfp_preproc_selftest")
    tmp_dir.mkdir(exist_ok=True)
    save_preprocessing_artifacts(
        bounds, log_features, tbl,
        json_path=tmp_dir / "p.json", csv_path=tmp_dir / "p.csv",
    )
    b2, lf2, tbl2 = load_preprocessing_artifacts(tmp_dir / "p.json", tmp_dir / "p.csv")
    for f in METEO_FEATURES:
        assert abs(b2[f][0] - bounds[f][0]) < 1e-3, f"bound round-trip failed for {f}"
    assert lf2 == log_features
    print(f"  bounds: {len(bounds)} features")
    print(f"  log_features: {log_features}")
    print(f"  imputation_table: {tbl.shape}")
    print("[preproc] All self-test assertions passed ✓")


if __name__ == "__main__":
    _self_test()
