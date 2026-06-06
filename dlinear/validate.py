"""
Validation utilities: temporal holdout evaluation and walk-forward CV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from config import HORIZONS, TRAIN_WEEKS_PER_REGION, VALID_WEEKS


# ---------------------------------------------------------------------------
# Calendar-matched validation: hold out, per region, the last training-year
# anchors whose woy aligns with that region's test_end_woy. The test set is
# 91 days per region whose calendar window varies (woy 0–52 across regions,
# concentrated around 40–46). The fixed-26-weeks holdout puts val anchors at
# the end of December, which doesn't match the test window's calendar position
# for most regions — the resulting val MAE is structurally optimistic.
# ---------------------------------------------------------------------------

# Cumulative days before each month (non-leap-year). Mirrors data_pipeline.py.
_MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_DOY_OFFSETS = [0] + [sum(_MONTH_DAYS[:i]) for i in range(1, 12)]


def _doy_from_md(month: int, day: int) -> int:
    return _DOY_OFFSETS[month - 1] + day


def _woy_from_doy(doy: int) -> int:
    """0-indexed week-of-year in [0, 52]. Matches features_extra._woy_from_doy."""
    return min(52, max(0, (int(doy) - 1) // 7))


def compute_test_anchor_woy_per_region(test_path: str | Path) -> dict[str, int]:
    """Return {region_id: woy_of_last_test_date}. The test set's last day is
    the anchor; predictions are for the 5 weeks after it."""
    test = pd.read_csv(test_path, usecols=["region_id", "date"], dtype={"region_id": str, "date": str})
    last = test.groupby("region_id")["date"].last()
    parts = last.str.split("-", expand=True)
    months = parts[1].astype(int).values
    days = parts[2].astype(int).values
    woys = np.array([_woy_from_doy(_doy_from_md(m, d)) for m, d in zip(months, days)],
                    dtype=np.int16)
    return dict(zip(last.index, woys.tolist()))


def compute_region_test_months(test_path: str | Path) -> dict[str, int]:
    """Return {region_id: month_of_first_test_date}. Used by the Kaggle-proxy
    val holdout (Phase 11) — each region's "test season" is anchored to the
    month its 13-week test window begins."""
    test = pd.read_csv(test_path, usecols=["region_id", "date"], dtype={"region_id": str, "date": str})
    first = test.groupby("region_id")["date"].first()
    months = first.str.split("-", expand=True)[1].astype(int).values
    return dict(zip(first.index, months.tolist()))


# ---------------------------------------------------------------------------
# Region-gap helper (Phase 11) — weeks between each region's train end and
# test start. Used by the lag-shift feature (`USE_SCORE_LAG_SHIFT=True`) so
# the (lag-source → target) distance at training matches the test gap.
# ---------------------------------------------------------------------------

def _ymd_to_ordinal(y: int, m: int, d: int) -> int:
    """Gregorian ordinal that works for years > 9999 (dataset uses synthetic years)."""
    mdays = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    is_leap = (y % 4 == 0) and (y % 100 != 0 or y % 400 == 0)
    doy = sum(mdays[1:m]) + (1 if m > 2 and is_leap else 0) + d
    y1 = y - 1
    return y1 * 365 + y1 // 4 - y1 // 100 + y1 // 400 + doy


def _parse_date_ymd(s: str) -> tuple[int, int, int]:
    parts = str(s).strip().split("T")[0].split(" ")[0].replace("/", "-").replace(".", "-").split("-")
    return int(parts[0]), int(parts[1]), int(parts[2])


def compute_region_gap_dict(
    train_path: str | Path,
    test_path: str | Path,
    default_gap_weeks: float = 52.0,
) -> dict[str, float]:
    """Per region, weeks between last train date and first test date.

    Mirrors teammate's `validate_regions_and_gap`. Returns gap in weeks (float)
    keyed by region_id (string). Regions missing from one side get
    `default_gap_weeks`.
    """
    train = pd.read_csv(train_path, usecols=["region_id", "date"], dtype={"region_id": str, "date": str})
    test = pd.read_csv(test_path, usecols=["region_id", "date"], dtype={"region_id": str, "date": str})

    train_ends = train.groupby("region_id")["date"].last()
    test_starts = test.groupby("region_id")["date"].first()

    region_gap: dict[str, float] = {}
    for r in test_starts.index:
        if r not in train_ends.index:
            region_gap[str(r)] = default_gap_weeks
            continue
        try:
            te_y, te_m, te_d = _parse_date_ymd(train_ends[r])
            ts_y, ts_m, ts_d = _parse_date_ymd(test_starts[r])
            ord_te = _ymd_to_ordinal(te_y, te_m, te_d)
            ord_ts = _ymd_to_ordinal(ts_y, ts_m, ts_d)
            gap_days = max(0, ord_ts - ord_te)
            region_gap[str(r)] = float(gap_days) / 7.0
        except Exception:
            region_gap[str(r)] = default_gap_weeks
    return region_gap


def _circular_woy_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wrap-around distance in [0, 26]."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return np.minimum(d, 52 - d)


def calendar_matched_split(
    features_df: pd.DataFrame,
    test_woy_per_region: dict[str, int],
    slack_weeks: int = 6,
    last_year_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-region calendar-matched holdout.

    For each region with `test_end_woy = w`, val anchors are rows whose
    week-of-year is within ±`slack_weeks` of `w`. If `last_year_only=True`
    we additionally restrict to the most recent 52 weeks of training anchors
    (so val ≈ "the last summer-ish window per region").

    Returns (train_split, val_split). Both reset_index'd. Mutual exclusion via
    masks; lag-feature columns and targets unchanged.
    """
    df = features_df.reset_index(drop=True).copy()
    woys = ((df["day_of_year"].astype(np.int32) - 1) // 7).clip(0, 52).astype(np.int16).values
    test_woys = df["region_id"].map(test_woy_per_region).astype(np.int16).values
    dist = _circular_woy_distance(woys, test_woys)

    val_mask = dist <= slack_weeks
    if last_year_only:
        max_week_per_region = df.groupby("region_id")["week_idx"].transform("max")
        recent = df["week_idx"].values > (max_week_per_region.values - 52)
        val_mask = val_mask & recent

    train_mask = ~val_mask
    train_split = df[train_mask].reset_index(drop=True)
    val_split = df[val_mask].reset_index(drop=True)

    print(f"Calendar-matched split: train={len(train_split):,}  val={len(val_split):,}  "
          f"(slack=±{slack_weeks} weeks, last_year_only={last_year_only})")
    return train_split, val_split


def last_anchor_per_region_mask(
    val_df: pd.DataFrame,
    test_woy_per_region: dict[str, int],
) -> np.ndarray:
    """Boolean mask selecting one row per region — the val anchor whose woy is
    closest to that region's test_end_woy, breaking ties by highest week_idx.

    Mirrors Kaggle's 1-anchor-per-region structure. Use to compute the
    last-anchor headline MAE."""
    df = val_df.reset_index(drop=True).copy()
    woys = ((df["day_of_year"].astype(np.int32) - 1) // 7).clip(0, 52).astype(np.int16).values
    test_woys = df["region_id"].map(test_woy_per_region).astype(np.int16).values
    df["_woy_dist"] = _circular_woy_distance(woys, test_woys)
    df["_neg_week_idx"] = -df["week_idx"].astype(np.int32)
    df_sorted = df.sort_values(["region_id", "_woy_dist", "_neg_week_idx"])
    picked = df_sorted.groupby("region_id").head(1).index
    mask = np.zeros(len(df), dtype=bool)
    mask[picked] = True
    return mask


def cluster_stratified_report(
    val_df: pd.DataFrame,
    pred_dict: dict[int, np.ndarray],
    cluster_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-cluster macro-MAE breakdown. `cluster_table` must have columns
    region_id, region_cluster_id (matches features_extra.fit_region_clusters)."""
    cl = cluster_table[["region_id", "region_cluster_id"]].drop_duplicates()
    merged = val_df[["region_id"] + [f"target_w{h}" for h in HORIZONS]].copy()
    merged = merged.merge(cl, on="region_id", how="left")
    merged["region_cluster_id"] = merged["region_cluster_id"].fillna(-1).astype(int)
    for h in HORIZONS:
        merged[f"_abs_err_h{h}"] = np.abs(
            merged[f"target_w{h}"].values - pred_dict[h]
        )
    rows = []
    for cid, grp in merged.groupby("region_cluster_id"):
        h_maes = [grp[f"_abs_err_h{h}"].mean() for h in HORIZONS]
        rows.append({
            "cluster_id": int(cid),
            "n_rows": len(grp),
            "n_regions": grp["region_id"].nunique(),
            **{f"mae_w{h}": h_maes[i] for i, h in enumerate(HORIZONS)},
            "macro_mae": float(np.mean(h_maes)),
        })
    return pd.DataFrame(rows).sort_values("macro_mae", ascending=False).reset_index(drop=True)


def compute_macro_mae(
    y_true_dict: dict[int, np.ndarray],
    y_pred_dict: dict[int, np.ndarray],
) -> tuple[float, dict[int, float]]:
    """Compute per-horizon MAE and their unweighted mean."""
    horizon_maes = {
        h: mean_absolute_error(y_true_dict[h], y_pred_dict[h])
        for h in HORIZONS
    }
    macro_mae = float(np.mean(list(horizon_maes.values())))
    return macro_mae, horizon_maes


def fixed_holdout_split(
    features_df: pd.DataFrame,
    n_val_weeks: int = VALID_WEEKS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: last n_val_weeks per region go to validation.

    Anchor week_idx runs 0..781. The last valid anchor is week_idx = 776
    (so target_w5 = week 781 exists). Validation anchors: the last n_val_weeks
    of these valid anchors.
    """
    max_anchor = TRAIN_WEEKS_PER_REGION - 6  # 776
    val_threshold = max_anchor - n_val_weeks  # 750 for 26 weeks

    train_mask = features_df["week_idx"] <= val_threshold
    val_mask = features_df["week_idx"] > val_threshold

    train_split = features_df[train_mask].reset_index(drop=True)
    val_split = features_df[val_mask].reset_index(drop=True)

    print(f"Train split: {len(train_split):,} rows "
          f"(week_idx ≤ {val_threshold})")
    print(f"Val   split: {len(val_split):,} rows "
          f"(week_idx > {val_threshold})")
    return train_split, val_split


def evaluate_predictions(
    val_df: pd.DataFrame,
    pred_dict: dict[int, np.ndarray],
) -> tuple[float, dict[int, float]]:
    """Build y_true_dict from val_df and compute macro MAE."""
    y_true_dict = {h: val_df[f"target_w{h}"].values for h in HORIZONS}
    return compute_macro_mae(y_true_dict, pred_dict)


def walk_forward_cv(
    features_df: pd.DataFrame,
    fold_boundaries: list[int],
    train_fn,
    feature_cols: list[str],
) -> list[dict]:
    """
    Walk-forward cross-validation.

    fold_boundaries: list of week_idx values that mark the end of each training fold.
    For each boundary b:
      - Train: week_idx <= b
      - Val:   week_idx in (b, b+5] (5 anchor weeks, each predicting up to b+10)

    train_fn(X_train, y_train_dict, X_val, y_val_dict) -> pred_dict
    """
    results = []
    for b in fold_boundaries:
        train_mask = features_df["week_idx"] <= b
        val_mask = (features_df["week_idx"] > b) & (features_df["week_idx"] <= b + 5)

        fold_train = features_df[train_mask]
        fold_val = features_df[val_mask]

        if len(fold_val) == 0:
            print(f"Fold boundary {b}: no validation rows, skipping")
            continue

        X_train = fold_train[feature_cols]
        X_val = fold_val[feature_cols]
        y_train_dict = {h: fold_train[f"target_w{h}"].values for h in HORIZONS}
        y_val_dict = {h: fold_val[f"target_w{h}"].values for h in HORIZONS}

        pred_dict = train_fn(X_train, y_train_dict, X_val, y_val_dict)
        macro_mae, horizon_maes = compute_macro_mae(y_val_dict, pred_dict)

        print(f"Fold b={b}: macro_MAE={macro_mae:.4f} | "
              + " | ".join(f"w{h}={v:.4f}" for h, v in horizon_maes.items()))
        results.append({
            "boundary": b,
            "macro_mae": macro_mae,
            "horizon_maes": horizon_maes,
        })

    avg_macro = np.mean([r["macro_mae"] for r in results])
    print(f"\nWalk-forward CV avg macro MAE: {avg_macro:.4f}")
    return results


def walk_forward_diagnostic(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    fold_boundaries: list[int],
    train_one_fn,
    sample_weight_fn=None,
    n_estimators_override: int | None = None,
) -> dict:
    """
    Run multi-fold walk-forward diagnostic. For each boundary b:
      - Train on week_idx ≤ b
      - Validate on week_idx ∈ (b, b+5]

    `train_one_fn(X_tr, y_tr, X_val, y_val, h, sample_weight)` mirrors the
    signature used in train.py main(). `sample_weight_fn(y_tr_series)` returns
    a per-row weight vector (or None). `n_estimators_override` lets the caller
    cut tree count to speed up the diagnostic relative to the production run.

    Returns dict with `per_fold`, `avg_macro_mae`, and `last_fold_macro_mae`.
    The last-fold metric is the most relevant proxy for the Kaggle test window.
    """
    from model import LGBM_PARAMS  # avoid circular import at module load
    if n_estimators_override:
        _saved = LGBM_PARAMS.get("n_estimators")
        LGBM_PARAMS["n_estimators"] = n_estimators_override

    per_fold = []
    try:
        for b in fold_boundaries:
            train_mask = features_df["week_idx"] <= b
            val_mask = (features_df["week_idx"] > b) & (features_df["week_idx"] <= b + 5)
            fold_train = features_df[train_mask]
            fold_val = features_df[val_mask]
            if len(fold_val) == 0:
                print(f"  [walk-forward] boundary {b}: empty val, skipping")
                continue

            X_tr = fold_train[feature_cols]
            X_val = fold_val[feature_cols]
            preds = {}
            for h in HORIZONS:
                y_tr = fold_train[f"target_w{h}"]
                y_val = fold_val[f"target_w{h}"]
                w = sample_weight_fn(y_tr) if sample_weight_fn else None
                model = train_one_fn(X_tr, y_tr, X_val, y_val, h, w)
                preds[h] = np.clip(model.predict(X_val), 0.0, 5.0)

            y_val_dict = {h: fold_val[f"target_w{h}"].values for h in HORIZONS}
            macro_mae, horizon_maes = compute_macro_mae(y_val_dict, preds)
            print(f"  [walk-forward] b={b}: macro_MAE={macro_mae:.4f}")
            per_fold.append({"boundary": b, "macro_mae": macro_mae, "horizon_maes": horizon_maes})
    finally:
        if n_estimators_override:
            LGBM_PARAMS["n_estimators"] = _saved

    if not per_fold:
        return {"per_fold": [], "avg_macro_mae": None, "last_fold_macro_mae": None}
    avg = float(np.mean([f["macro_mae"] for f in per_fold]))
    last = per_fold[-1]["macro_mae"]
    return {"per_fold": per_fold, "avg_macro_mae": avg, "last_fold_macro_mae": last}


def print_validation_report(macro_mae: float, horizon_maes: dict[int, float]):
    print("\n" + "=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)
    for h in HORIZONS:
        print(f"  Week {h} MAE: {horizon_maes[h]:.4f}")
    print(f"  Macro MAE:  {macro_mae:.4f}")
    print("=" * 50)
    if macro_mae > 1.0:
        print("WARNING: macro MAE > 1.0 — check for lag alignment bugs or leakage")
