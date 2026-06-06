"""Raw CSV loading, synthetic-date parsing, and per-region metadata.

Dates use scrambled synthetic years (e.g. 3004 .. 58061) that overflow
pandas/`datetime`, so all date math is integer arithmetic, fully vectorized.
Within a region, dates are monotone and day-of-year carries the real seasonal
signal — the absolute year is discarded after ordering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# The data uses a SYNTHETIC calendar: Feb 29 appears in some data-years and not
# others, NOT by real Gregorian leap rules (e.g. year 3005 has 366 days though
# 3005 % 4 != 0). Real-leap-rule doy gives collisions (02-29 and 03-01 both → 60).
# Fix: a FIXED 366-day calendar (Feb always 29 days). Every date maps to a unique,
# strictly-increasing (within region) ordinal; non-leap data-years simply lack the
# 02-29 row (a harmless 1-day gap at doy 60). Consistent across train & test.
_DAYS_IN_YEAR = 366
# cumulative days before month start, Feb=29 always: index by month (1..12)
_CUM_MDAYS = np.array([0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335], dtype=np.int64)


def parse_dates(date_str: pd.Series) -> pd.DataFrame:
    """Vectorized parse of 'YYYY-MM-DD' → ordinal day, day-of-year, month, week-of-year.

    Uses the data's synthetic fixed-366 calendar so ordinals are collision-free and
    strictly monotone within a region (good for ordering & gap math), and doy/woy are
    consistent between train and test (good for calendar matching).
    """
    parts = date_str.str.split("-", expand=True)
    y = parts[0].astype(np.int64).to_numpy()
    m = parts[1].astype(np.int64).to_numpy()
    d = parts[2].astype(np.int64).to_numpy()

    doy = _CUM_MDAYS[m] + d                       # 1..366, Feb 29 = 60
    ordinal = (y - 1) * _DAYS_IN_YEAR + doy
    woy = ((doy - 1) // C.DAYS_PER_WEEK + 1).astype(np.int64)  # 1..53

    return pd.DataFrame(
        {"ordinal": ordinal, "doy": doy.astype(np.int64),
         "month": m, "woy": woy},
        index=date_str.index,
    )


def region_to_int(region_id: pd.Series) -> pd.Series:
    """'R123' -> 123 (numeric region index for LGBM + sorting)."""
    return region_id.str.slice(1).astype(np.int64)


def load_raw(path, channels: tuple, has_score: bool) -> pd.DataFrame:
    """Load a raw CSV, parse dates, attach region_idx; keep only needed columns."""
    usecols = [C.ID_COL, C.DATE_COL] + list(channels) + ([C.SCORE_COL] if has_score else [])
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    dt = parse_dates(df[C.DATE_COL])
    df = pd.concat([df.drop(columns=[C.DATE_COL]), dt], axis=1)
    df["region_idx"] = region_to_int(df[C.ID_COL])
    df = df.sort_values(["region_idx", "ordinal"], kind="stable").reset_index(drop=True)
    return df


def region_metadata(test_df: pd.DataFrame) -> pd.DataFrame:
    """Per-region test anchor (last test day) — its ordinal, woy, month.

    The forecast origin at test time is each region's last test day; we predict
    weeks +1..+5 from there. Returned indexed by region_idx.
    """
    last = (test_df.sort_values("ordinal")
            .groupby("region_idx", sort=True)
            .tail(1)
            .set_index("region_idx"))
    return last[[C.ID_COL, "ordinal", "woy", "month", "doy"]].rename(
        columns={"ordinal": "test_anchor_ord", "woy": "test_anchor_woy",
                 "month": "test_anchor_month", "doy": "test_anchor_doy"})


def attach_gap_weeks(train_df: pd.DataFrame, region_meta: pd.DataFrame) -> pd.DataFrame:
    """Add per-region train_end_ord and gap_weeks (train end → test anchor)."""
    train_end = train_df.groupby("region_idx")["ordinal"].max().rename("train_end_ord")
    meta = region_meta.join(train_end, how="left")
    meta["gap_weeks"] = (meta["test_anchor_ord"] - meta["train_end_ord"]) / C.DAYS_PER_WEEK
    return meta
