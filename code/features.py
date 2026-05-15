"""
Stateless feature transformations applied after the lag feature matrix is built.
"""

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic and raw temporal features from month and day_of_year columns."""
    df = df.copy()
    month = df["month"].values
    doy = df["day_of_year"].values

    df["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365).astype(np.float32)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365).astype(np.float32)
    # keep raw month as a feature for tree-based splits
    df["month_raw"] = month.astype(np.int8)
    return df


def compute_region_stats(train_weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-region historical score statistics from the TRAINING fold only.

    Returns a DataFrame indexed by region_id with columns:
      region_score_mean, region_score_std, region_zero_rate
    """
    scored = train_weekly_df[train_weekly_df["score"].notna()]
    stats = scored.groupby("region_id")["score"].agg(
        region_score_mean="mean",
        region_score_std="std",
    )
    zero_rate = (scored.groupby("region_id")["score"]
                 .apply(lambda x: (x == 0).mean())
                 .rename("region_zero_rate"))
    return pd.concat([stats, zero_rate], axis=1).reset_index()


def add_region_features(
    df: pd.DataFrame,
    region_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge per-region statistics into the feature matrix.

    region_stats must be computed from the training fold only to prevent leakage.
    Also adds region_id_int (integer region index for LightGBM).
    """
    df = df.copy()
    df["region_id_int"] = df["region_id"].str.replace("R", "", regex=False).astype(np.int32)
    df = df.merge(region_stats, on="region_id", how="left")
    df["region_score_mean"] = df["region_score_mean"].astype(np.float32)
    df["region_score_std"] = df["region_score_std"].fillna(0).astype(np.float32)
    df["region_zero_rate"] = df["region_zero_rate"].astype(np.float32)
    return df


def clip_predictions(arr: np.ndarray, round_to_int: bool = False) -> np.ndarray:
    """Clip predictions to valid score range [0, 5] and optionally round."""
    clipped = np.clip(arr, 0.0, 5.0)
    if round_to_int:
        clipped = np.round(clipped)
    return clipped.astype(np.float32)


EXTRA_FEATURE_COLS = [
    "month_sin", "month_cos", "doy_sin", "doy_cos", "month_raw",
    "region_id_int", "region_score_mean", "region_score_std", "region_zero_rate",
]
