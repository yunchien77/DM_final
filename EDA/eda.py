"""
EDA functions for the natural disaster severity prediction dataset.
Each section_* function returns (figs, findings):
  - figs:     list of matplotlib Figures
  - findings: dict with structured analysis output (numbers + interpretation)
The function also prints a "Key findings" block to stdout so the notebook
shows interpretation alongside the figures.

Call them from eda.ipynb.

NOTE: dates in this dataset use synthetic years (e.g., 3005, 8074) that overflow
pandas' nanosecond datetime range. We keep `date` as a string and parse month /
day-of-year / week-of-year directly from the 'YYYY-MM-DD' string.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "code"))

TRAIN_PATH = ROOT / "dataset" / "data" / "train.csv"
TEST_PATH = ROOT / "dataset" / "data" / "test.csv"
SAMPLE_PATH = ROOT / "dataset" / "sample_submission.csv"

METEO_FEATURES = [
    "wind", "wind_min", "wind_max", "wind_range",
    "humidity", "tmp", "tmp_range", "tmp_max", "tmp_min",
    "surf_tmp", "surf_pre", "dp_tmp", "wb_tmp", "prec",
]

SAMPLE_REGIONS = ["R1", "R500", "R1000", "R1500", "R2000"]


# ---------------------------------------------------------------------------
# Date parsing — avoids pd.to_datetime nanosecond overflow on synthetic years.
# ---------------------------------------------------------------------------

# Cumulative non-leap-year day counts before each month start.
_MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_DOY_OFFSETS = [0] + [sum(_MONTH_DAYS[:i]) for i in range(1, 12)]


def _parse_month(date_series: pd.Series) -> pd.Series:
    """Extract 1..12 month from 'YYYY-MM-DD' / 'YYYYY-MM-DD' strings, vectorized.

    Year width varies (the dataset has 4- and 5-digit synthetic years), so index
    from the right end where MM-DD is always in fixed positions.
    """
    return date_series.str.slice(-5, -3).astype(np.int16)


def _parse_day_of_month(date_series: pd.Series) -> pd.Series:
    """Last two characters of 'YYYY[Y]-MM-DD' → day of month, 1..31."""
    return date_series.str.slice(-2).astype(np.int16)


def _parse_day_of_year(date_series: pd.Series) -> pd.Series:
    """1..365 day-of-year from 'YYYY-MM-DD' strings (non-leap-year convention)."""
    month = _parse_month(date_series)
    dom = _parse_day_of_month(date_series)
    offsets = month.map(lambda m: _DOY_OFFSETS[m - 1])
    return (offsets + dom).astype(np.int16)


def _parse_week_of_year(date_series: pd.Series) -> pd.Series:
    """1..53 week bucket derived from day_of_year: ((doy - 1) // 7) + 1."""
    return (((_parse_day_of_year(date_series) - 1) // 7) + 1).astype(np.int16)


def _add_calendar_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Attach month / day_of_year / week_of_year derived from the string date."""
    df = df.copy()
    df["month"] = _parse_month(df["date"])
    df["day_of_year"] = _parse_day_of_year(df["date"])
    df["week_of_year"] = _parse_week_of_year(df["date"])
    return df


def _add_day_index(df: pd.DataFrame, group_col: str = "region_id") -> pd.DataFrame:
    """
    Sequential within-region day index (0, 1, 2, …) for x-axis plotting.
    The dataset has uniform daily granularity per region, so this maps 1:1 to time.
    """
    df = df.sort_values([group_col, "date"]).reset_index(drop=True)
    df["day_idx"] = df.groupby(group_col).cumcount()
    return df


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_train_sample(n_regions=50, seed=42):
    """Load a random subset of regions from train for fast EDA."""
    rng = np.random.default_rng(seed)
    all_ids = [f"R{i}" for i in range(1, 2249)]
    chosen = rng.choice(all_ids, size=n_regions, replace=False).tolist()
    chosen_set = set(chosen)
    chunks = []
    for chunk in pd.read_csv(TRAIN_PATH, dtype={"date": str}, chunksize=500_000):
        sub = chunk[chunk["region_id"].isin(chosen_set)]
        if len(sub):
            chunks.append(sub)
    df = pd.concat(chunks, ignore_index=True)
    return _add_calendar_cols(df)


def _load_full_train_scored():
    """Load only the scored rows from train (score is non-null)."""
    chunks = []
    for chunk in pd.read_csv(TRAIN_PATH, dtype={"date": str}, chunksize=500_000):
        sub = chunk[chunk["score"].notna()]
        chunks.append(sub)
    df = pd.concat(chunks, ignore_index=True)
    df["score"] = df["score"].astype(int)
    return _add_calendar_cols(df)


def _load_test():
    df = pd.read_csv(TEST_PATH, dtype={"date": str})
    return _add_calendar_cols(df)


# ---------------------------------------------------------------------------
# Section 1: Data Overview
# ---------------------------------------------------------------------------

def section_data_overview():
    """Print data shapes, dtypes, null counts, rows/region uniformity,
    head-previews, and per-column missing rates.
    Returns (figs=[], findings)."""
    print("=" * 60)
    print("SECTION 1: DATA OVERVIEW")
    print("=" * 60)

    train_sample = _load_train_sample(n_regions=50)
    test = _load_test()
    sub = pd.read_csv(SAMPLE_PATH)

    print(f"\nShape — train sample: {train_sample.shape}   test: {test.shape}")
    print(f"\nTrain columns : {list(train_sample.columns)}")
    print(f"Test  columns : {list(test.columns)}")
    print(f"Sub   columns : {list(sub.columns)}")

    print(f"\nTrain sample dtypes:\n{train_sample.dtypes}")

    # ── DataFrame head previews (all columns wide) ───────────────────────
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n--- Train sample preview (first 20 rows, all ~20 columns) ---")
        print(train_sample.head(20))

        print("\n--- Test preview (first 20 rows, all ~19 columns) ---")
        print(test.head(20))

    # ── Per-column missing-rate tables (every column, not just non-zero) ─
    def _missing_table(df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
        n_missing = df.isnull().sum()
        pct = (n_missing / total_rows * 100.0).round(3)
        return (
            pd.DataFrame({"n_missing": n_missing.astype(int), "missing_pct": pct})
            .sort_values("missing_pct", ascending=False)
        )

    train_missing = _missing_table(train_sample, len(train_sample))
    test_missing = _missing_table(test, len(test))

    print(f"\n--- Missing rate per column (train sample, n={len(train_sample):,}) ---")
    print(train_missing.to_string())
    print(f"\n--- Missing rate per column (test, n={len(test):,}) ---")
    print(test_missing.to_string())

    null_counts = train_sample.isnull().sum()
    region_counts = test.groupby("region_id").size()
    print(f"\nTest rows/region: min={region_counts.min()}, max={region_counts.max()}, "
          f"unique={region_counts.nunique()} distinct values")

    scored = train_sample[train_sample["score"].notna()]
    total = len(train_sample)
    print(f"\nTrain sample: {total:,} rows, {len(scored):,} scored rows "
          f"({100*len(scored)/total:.1f}%)")

    print(f"\nSample submission shape: {sub.shape}")
    print(sub.head())

    # ── Findings ─────────────────────────────────────────────────────────
    # Drop `score` from train missing — it is expected to be ~85% missing
    # (one score per week × 7 days = 1/7 ≈ 14% scored).
    train_missing_no_score = train_missing.drop("score", errors="ignore")
    findings = {
        "section": 1,
        "shape_train_sample": list(train_sample.shape),
        "shape_test": list(test.shape),
        "n_train_columns": len(train_sample.columns),
        "n_test_columns": len(test.columns),
        "test_rows_per_region_uniform": int(region_counts.nunique()) == 1,
        "test_rows_per_region": int(region_counts.iloc[0]),
        "train_missing_pct": train_missing["missing_pct"].to_dict(),
        "test_missing_pct": test_missing["missing_pct"].to_dict(),
        "train_max_missing_pct_excl_score": float(train_missing_no_score["missing_pct"].max())
            if len(train_missing_no_score) else 0.0,
        "test_max_missing_pct": float(test_missing["missing_pct"].max())
            if len(test_missing) else 0.0,
        "train_sample_scored_frac": float(len(scored) / total) if total else 0.0,
        "submission_shape": list(sub.shape),
    }

    def _top_missing(df, exclude=()):
        df = df[~df.index.isin(exclude)]
        df = df[df["missing_pct"] > 0]
        return df.head(3)

    train_top = _top_missing(train_missing, exclude=("score",))
    test_top = _top_missing(test_missing)
    _print_findings(1, "Data Overview", [
        f"Train sample shape: {findings['shape_train_sample']}; "
        f"test shape: {findings['shape_test']}",
        f"Test rows/region uniform = {findings['test_rows_per_region_uniform']} "
        f"({findings['test_rows_per_region']} rows/region)",
        f"Train scored-row fraction: {100*findings['train_sample_scored_frac']:.2f}% "
        "(weekly scoring → 1/7 ≈ 14.3%)",
        "Missing data (train, excluding score): "
        + (", ".join(f"{c}({train_top.loc[c, 'missing_pct']:.2f}%)" for c in train_top.index)
           if len(train_top) else "0 missing in any feature column"),
        "Missing data (test): "
        + (", ".join(f"{c}({test_top.loc[c, 'missing_pct']:.2f}%)" for c in test_top.index)
           if len(test_top) else "0 missing in any column"),
        f"Sample submission shape: {findings['submission_shape']}",
    ])
    return [], findings


# ---------------------------------------------------------------------------
# Section 2: Score Distribution
# ---------------------------------------------------------------------------

def section_score_distribution(scored_df=None):
    """
    Returns list of Figures:
      [0] Overall score value-count bar chart
      [1] Per-region mean score histogram
      [2] ACF/PACF of weekly score for R1
    """
    if scored_df is None:
        print("Loading scored rows from train (this may take a moment)...")
        scored_df = _load_full_train_scored()

    figs = []

    # --- Fig 0: overall distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = scored_df["score"].value_counts().sort_index()
    pcts = 100 * counts / counts.sum()
    bars = ax.bar(counts.index, counts.values, color="steelblue", edgecolor="white")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5000,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Severity Score")
    ax.set_ylabel("Count")
    ax.set_title("Overall Score Distribution (all regions, all weeks)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    figs.append(fig)

    # --- Fig 1: per-region mean score ---
    region_means = scored_df.groupby("region_id")["score"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(region_means, bins=40, color="coral", edgecolor="white")
    ax.set_xlabel("Mean Weekly Score per Region")
    ax.set_ylabel("Number of Regions")
    ax.set_title("Distribution of Per-Region Mean Severity Score")
    ax.axvline(region_means.mean(), color="black", linestyle="--",
               label=f"Global mean = {region_means.mean():.3f}")
    ax.legend()
    fig.tight_layout()
    figs.append(fig)

    # --- Fig 2: ACF/PACF for one region ---
    target_region = "R1" if "R1" in scored_df["region_id"].values else scored_df["region_id"].iloc[0]
    series = scored_df[scored_df["region_id"] == target_region].sort_values("date")["score"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series, lags=40, ax=axes[0], title=f"ACF — Region {target_region} Weekly Score")
    plot_pacf(series, lags=40, ax=axes[1], title=f"PACF — Region {target_region} Weekly Score")
    fig.tight_layout()
    figs.append(fig)

    # ── Findings ─────────────────────────────────────────────────────────
    zero_rate = float((scored_df["score"] == 0).mean())
    nonzero = scored_df.loc[scored_df["score"] > 0, "score"]
    mean_nonzero = float(nonzero.mean()) if len(nonzero) else 0.0
    score_freq = counts.to_dict()
    score_pct = (pcts.round(2)).to_dict()
    # First significant trough in ACF (rough heuristic)
    try:
        from statsmodels.tsa.stattools import acf
        acf_vals = acf(series, nlags=40, fft=True)
        below_zero = np.where(acf_vals < 0.05)[0]
        first_low_lag = int(below_zero[0]) if len(below_zero) > 1 else None
    except Exception:
        first_low_lag = None

    findings = {
        "section": 2,
        "zero_rate": round(zero_rate, 4),
        "mean_nonzero_score": round(mean_nonzero, 4),
        "score_counts": {int(k): int(v) for k, v in score_freq.items()},
        "score_pct": {int(k): float(v) for k, v in score_pct.items()},
        "global_mean_per_region_score": round(float(region_means.mean()), 4),
        "acf_first_lag_below_0.05": first_low_lag,
        "acf_region": target_region,
    }
    _print_findings(2, "Score Distribution", [
        f"Zero rate (all regions, all weeks): {100*zero_rate:.2f}%",
        f"Mean score (non-zero rows only): {mean_nonzero:.3f}",
        f"Score frequency (% of weeks): "
        + ", ".join(f"s={s}: {findings['score_pct'].get(s, 0):.2f}%" for s in range(6)),
        f"Per-region mean score: global mean = {findings['global_mean_per_region_score']:.3f}",
        (f"ACF for {target_region} first drops below 0.05 at lag={first_low_lag}"
         if first_low_lag is not None else
         f"ACF for {target_region} remains > 0.05 for all 40 lags shown"),
    ])
    return figs, findings


# ---------------------------------------------------------------------------
# Section 3: Feature Distributions
# ---------------------------------------------------------------------------

def section_feature_distributions(train_sample=None):
    """
    Returns list of Figures:
      [0] Histograms for each meteorological feature
      [1] Correlation heatmap
      [2] Feature-score correlation bar chart
    """
    if train_sample is None:
        print("Loading train sample...")
        train_sample = _load_train_sample(n_regions=100)

    figs = []

    # --- Fig 0: feature histograms ---
    n_feat = len(METEO_FEATURES)
    ncols = 3
    nrows = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3))
    axes_flat = axes.flatten()
    for i, feat in enumerate(METEO_FEATURES):
        col = train_sample[feat].dropna()
        axes_flat[i].hist(col, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
        axes_flat[i].set_title(feat, fontsize=10)
        axes_flat[i].set_ylabel("Count")
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Meteorological Feature Distributions (sample)", y=1.01)
    fig.tight_layout()
    figs.append(fig)

    # --- Fig 1: correlation heatmap ---
    corr = train_sample[METEO_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.4, ax=ax, annot_kws={"size": 7})
    ax.set_title("Pearson Correlation — Meteorological Features")
    fig.tight_layout()
    figs.append(fig)

    # --- Fig 2: feature-score correlation ---
    scored = train_sample[train_sample["score"].notna()]
    feat_score_corr = scored[METEO_FEATURES + ["score"]].corr()["score"].drop("score").sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["coral" if v > 0 else "steelblue" for v in feat_score_corr]
    ax.barh(feat_score_corr.index, feat_score_corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Score")
    ax.set_title("Feature–Score Correlation (sample, scored rows only)")
    fig.tight_layout()
    figs.append(fig)

    # ── Findings ─────────────────────────────────────────────────────────
    abs_fs = feat_score_corr.abs().sort_values(ascending=False)
    top_score_corr = abs_fs.head(3)
    # Top correlated feature pairs (multicollinearity)
    corr_mat = corr.copy()
    np.fill_diagonal(corr_mat.values, 0)
    abs_pairs = corr_mat.abs().unstack().sort_values(ascending=False)
    seen = set(); pair_rows = []
    for (a, b), v in abs_pairs.items():
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        pair_rows.append((a, b, float(corr_mat.loc[a, b])))
        if len(pair_rows) >= 5:
            break
    findings = {
        "section": 3,
        "top_feature_score_corr": {f: float(feat_score_corr[f]) for f in top_score_corr.index},
        "top_collinear_pairs": [{"a": a, "b": b, "corr": round(v, 3)} for a, b, v in pair_rows],
    }
    _print_findings(3, "Feature Distributions", [
        "Top features by |corr(feature, score)|: " + ", ".join(
            f"{f}({feat_score_corr[f]:+.3f})" for f in top_score_corr.index),
        "Top collinear pairs (|r| highest, may inflate variance in linear models): "
        + ", ".join(f"{a}↔{b}({v:+.2f})" for a, b, v in pair_rows[:5]),
    ])
    return figs, findings


# ---------------------------------------------------------------------------
# Section 4: Time-Series Visualization
# ---------------------------------------------------------------------------

def section_timeseries_viz(train_sample=None):
    """
    Returns list of Figures — one per sample region showing:
      - Daily tmp, prec, wind, humidity vs. day index (since real dates overflow
        pandas datetime, we plot a within-region sequential day index)
      - Overlaid weekly score markers
    Plus one Figure showing average score by calendar month.
    """
    if train_sample is None:
        print("Loading train sample for 5 specific regions...")
        chosen = {"R1", "R500", "R1000", "R1500", "R2000"}
        chunks = []
        for chunk in pd.read_csv(TRAIN_PATH, dtype={"date": str}, chunksize=500_000):
            sub = chunk[chunk["region_id"].isin(chosen)]
            if len(sub):
                chunks.append(sub)
        train_sample = pd.concat(chunks, ignore_index=True)
        train_sample = _add_calendar_cols(train_sample)

    train_sample = _add_day_index(train_sample)

    figs = []
    regions_available = [r for r in SAMPLE_REGIONS if r in train_sample["region_id"].unique()]

    for reg in regions_available:
        df_r = train_sample[train_sample["region_id"] == reg].sort_values("day_idx")
        scored_r = df_r[df_r["score"].notna()]

        fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
        plot_cols = ["tmp", "prec", "wind", "humidity"]
        colors = ["tomato", "steelblue", "green", "purple"]
        for ax, col, color in zip(axes, plot_cols, colors):
            ax.plot(df_r["day_idx"], df_r[col], linewidth=0.4, color=color, alpha=0.8)
            ax.set_ylabel(col, fontsize=9)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        # overlay score on tmp panel
        sc = axes[0].scatter(scored_r["day_idx"], scored_r["tmp"],
                             c=scored_r["score"], cmap="Reds", s=8, zorder=5,
                             vmin=0, vmax=5)
        fig.colorbar(sc, ax=axes[0], label="score")
        axes[-1].set_xlabel("Day index since start of region's record")
        fig.suptitle(f"Region {reg} — Daily Meteorological Features + Weekly Score", fontsize=11)
        fig.tight_layout()
        figs.append(fig)

    # Average score by calendar month (parsed from the string date)
    scored_all = train_sample[train_sample["score"].notna()]
    monthly = scored_all.groupby("month")["score"].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(monthly.index, monthly.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Calendar Month")
    ax.set_ylabel("Mean Score")
    ax.set_title("Average Severity Score by Calendar Month (sample regions)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"])
    fig.tight_layout()
    figs.append(fig)

    # ── Findings ─────────────────────────────────────────────────────────
    peak_month = int(monthly.idxmax())
    trough_month = int(monthly.idxmin())
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    findings = {
        "section": 4,
        "n_regions_plotted": len(regions_available),
        "mean_score_by_month": {int(k): float(v) for k, v in monthly.round(3).items()},
        "peak_month": peak_month,
        "trough_month": trough_month,
        "peak_minus_trough": float(monthly.max() - monthly.min()),
    }
    _print_findings(4, "Time-Series Viz", [
        f"Per-region time-series rendered for: {regions_available}",
        f"Peak severity month: {month_names[peak_month-1]} (mean score = {monthly[peak_month]:.3f})",
        f"Trough severity month: {month_names[trough_month-1]} (mean score = {monthly[trough_month]:.3f})",
        f"Seasonal amplitude (peak − trough) = {findings['peak_minus_trough']:.3f}",
    ])
    return figs, findings


# ---------------------------------------------------------------------------
# Section 5: Weekly Aggregation Patterns
# ---------------------------------------------------------------------------

def section_weekly_patterns(scored_df=None):
    """
    Returns list of Figures:
      [0] Feature–score correlation after weekly aggregation
      [1] Mean score by week-of-year
      [2] Score distribution by season
    """
    if scored_df is None:
        print("Loading scored rows...")
        scored_df = _load_full_train_scored()

    scored_df = scored_df.copy()  # don't mutate caller's frame

    def _season(m):
        if m in (12, 1, 2):
            return "DJF"
        elif m in (3, 4, 5):
            return "MAM"
        elif m in (6, 7, 8):
            return "JJA"
        return "SON"

    scored_df["season"] = scored_df["month"].map(_season)

    figs = []

    # --- Fig 0: feature-score corr on weekly scored rows ---
    corr_weekly = (scored_df[METEO_FEATURES + ["score"]]
                   .corr()["score"].drop("score").sort_values())
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["coral" if v > 0 else "steelblue" for v in corr_weekly]
    ax.barh(corr_weekly.index, corr_weekly.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Score")
    ax.set_title("Feature–Score Correlation on Weekly Scored Rows (all regions)")
    fig.tight_layout()
    figs.append(fig)

    # --- Fig 1: score by week-of-year ---
    woy = scored_df.groupby("week_of_year")["score"].mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(woy.index, woy.values, marker="o", markersize=3, linewidth=1, color="steelblue")
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("Mean Score")
    ax.set_title("Average Severity Score by Week-of-Year (all regions, all years)")
    fig.tight_layout()
    figs.append(fig)

    # --- Fig 2: score distribution by season ---
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    season_mean = {}
    for ax, season in zip(axes, ["DJF", "MAM", "JJA", "SON"]):
        sub = scored_df[scored_df["season"] == season]["score"]
        season_mean[season] = float(sub.mean()) if len(sub) else float("nan")
        vc = sub.value_counts().sort_index()
        ax.bar(vc.index, vc.values, color="steelblue", edgecolor="white")
        ax.set_title(season)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count" if ax == axes[0] else "")
    fig.suptitle("Score Distribution by Season")
    fig.tight_layout()
    figs.append(fig)

    # ── Findings ─────────────────────────────────────────────────────────
    top_pos = corr_weekly.tail(3)
    top_neg = corr_weekly.head(3)
    peak_woy = int(woy.idxmax())
    trough_woy = int(woy.idxmin())
    findings = {
        "section": 5,
        "top_positive_feature_score_corr": {f: float(corr_weekly[f]) for f in top_pos.index},
        "top_negative_feature_score_corr": {f: float(corr_weekly[f]) for f in top_neg.index},
        "peak_week_of_year": peak_woy,
        "trough_week_of_year": trough_woy,
        "season_mean_score": {k: round(v, 4) for k, v in season_mean.items()},
        "highest_severity_season": max(season_mean, key=season_mean.get),
    }
    _print_findings(5, "Weekly Patterns", [
        "Top + feature↔score corr: " + ", ".join(f"{f}({corr_weekly[f]:+.3f})" for f in top_pos.index),
        "Top − feature↔score corr: " + ", ".join(f"{f}({corr_weekly[f]:+.3f})" for f in top_neg.index),
        f"Peak severity week-of-year: {peak_woy} (mean={woy[peak_woy]:.3f}); "
        f"trough: {trough_woy} (mean={woy[trough_woy]:.3f})",
        f"Highest-severity season: {findings['highest_severity_season']} "
        f"(mean={season_mean[findings['highest_severity_season']]:.3f})",
    ])
    return figs, findings


# ---------------------------------------------------------------------------
# Section 6: Test Window Analysis
# ---------------------------------------------------------------------------

def section_test_analysis(scored_df=None):
    """
    Returns list of Figures:
      [0] Test feature distribution vs. train distribution (box plots)
    Also prints region overlap stats.
    """
    if scored_df is None:
        print("Loading scored rows for train feature stats...")
        scored_df = _load_full_train_scored()

    test = _load_test()

    # Region overlap
    train_regions = set(scored_df["region_id"].unique())
    test_regions = set(test["region_id"].unique())
    missing_in_train = test_regions - train_regions
    print(f"Test regions: {len(test_regions)}")
    print(f"Train regions: {len(train_regions)}")
    print(f"Test regions NOT in train: {len(missing_in_train)}")
    if missing_in_train:
        print(f"  Missing: {sorted(missing_in_train)[:10]}")

    # Feature distribution shift
    figs = []
    n_feat = len(METEO_FEATURES)
    ncols = 3
    nrows = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3))
    axes_flat = axes.flatten()

    median_offsets = {}
    for i, feat in enumerate(METEO_FEATURES):
        ax = axes_flat[i]
        train_vals = scored_df[feat].dropna().sample(min(50000, len(scored_df)), random_state=0)
        test_vals = test[feat].dropna()
        median_offsets[feat] = float(test_vals.median() - train_vals.median())
        ax.boxplot([train_vals, test_vals], labels=["Train", "Test"], widths=0.5,
                   medianprops=dict(color="red", linewidth=2))
        ax.set_title(feat, fontsize=10)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Train vs. Test Feature Distributions (box plots)", y=1.01)
    fig.tight_layout()
    figs.append(fig)

    # ── Findings ─────────────────────────────────────────────────────────
    abs_offsets = pd.Series(median_offsets).abs().sort_values(ascending=False)
    top_offset = abs_offsets.head(3)
    findings = {
        "section": 6,
        "n_test_regions": len(test_regions),
        "n_train_regions": len(train_regions),
        "n_missing_in_train": len(missing_in_train),
        "top_train_test_median_offsets": {f: float(median_offsets[f]) for f in top_offset.index},
    }
    _print_findings(6, "Test Window Analysis", [
        f"Test regions: {len(test_regions)}; train regions: {len(train_regions)}; "
        f"test regions missing from train: {len(missing_in_train)}",
        "Top features by |test_median − train_median|: " + ", ".join(
            f"{f}({median_offsets[f]:+.3f})" for f in top_offset.index),
    ])
    return figs, findings


# ---------------------------------------------------------------------------
# Helper for printing "Key findings" blocks
# ---------------------------------------------------------------------------

def _print_findings(section_num: int, title: str, key_lines: list):
    print()
    print("=" * 70)
    print(f"  Section {section_num} key findings — {title}")
    print("=" * 70)
    for line in key_lines:
        print(f"  • {line}")
    print()


# ---------------------------------------------------------------------------
# Section 7: Outlier Analysis (p1/p99 winsorization candidates)
# ---------------------------------------------------------------------------

def section_outlier_analysis(train_sample=None, q_lo=0.01, q_hi=0.99):
    """
    Per-feature p1/p99 thresholds, outlier rates, top-5 most extreme rows.
    Returns (figs, findings).
    """
    if train_sample is None:
        print("Loading train sample for outlier analysis...")
        train_sample = _load_train_sample(n_regions=100)

    rows = []
    extremes = {}
    for feat in METEO_FEATURES:
        col = train_sample[feat]
        valid = col.dropna()
        if len(valid) == 0:
            continue
        lo = float(valid.quantile(q_lo))
        hi = float(valid.quantile(q_hi))
        n = len(valid)
        n_below = int((valid < lo).sum())
        n_above = int((valid > hi).sum())
        rows.append({
            "feature": feat,
            "p1": lo, "p99": hi,
            "min": float(valid.min()), "max": float(valid.max()),
            "n_below_p1": n_below, "n_above_p99": n_above,
            "pct_below_p1": 100.0 * n_below / n,
            "pct_above_p99": 100.0 * n_above / n,
            "n_nan": int(col.isnull().sum()),
        })
        sub = train_sample[[feat, "region_id", "date"]].dropna(subset=[feat])
        extremes[feat] = pd.concat([
            sub.nlargest(3, feat).assign(side="above"),
            sub.nsmallest(2, feat).assign(side="below"),
        ]).reset_index(drop=True)

    outlier_table = pd.DataFrame(rows).set_index("feature")
    top_above = outlier_table.nlargest(3, "pct_above_p99")[["pct_above_p99", "p99", "max"]]
    top_below = outlier_table.nlargest(3, "pct_below_p1")[["pct_below_p1", "p1", "min"]]

    # Figure: per-feature histogram with p1/p99 marked
    n_feat = len(METEO_FEATURES)
    ncols = 3
    nrows = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3))
    axes_flat = axes.flatten()
    for i, feat in enumerate(METEO_FEATURES):
        valid = train_sample[feat].dropna()
        ax = axes_flat[i]
        ax.hist(valid, bins=80, color="steelblue", alpha=0.75)
        if feat in outlier_table.index:
            lo, hi = outlier_table.loc[feat, "p1"], outlier_table.loc[feat, "p99"]
            ax.axvline(lo, color="red", linestyle="--", linewidth=1, label=f"p1={lo:.2f}")
            ax.axvline(hi, color="red", linestyle="--", linewidth=1, label=f"p99={hi:.2f}")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_title(feat, fontsize=10)
        ax.set_yscale("log")
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Feature distributions with p1/p99 thresholds (log-y)", y=1.01)
    fig.tight_layout()

    findings = {
        "section": 7,
        "outlier_table": outlier_table.round(4).to_dict(orient="index"),
        "top_above_p99": top_above.round(4).to_dict(orient="index"),
        "top_below_p1": top_below.round(4).to_dict(orient="index"),
        "extreme_rows_top1_feature": extremes.get(top_above.index[0], pd.DataFrame()).to_dict(orient="records"),
        "total_nans": int(outlier_table["n_nan"].sum()),
    }
    _print_findings(7, "Outlier Analysis", [
        "Top features by % above p99: " + ", ".join(
            f"{f} ({top_above.loc[f, 'pct_above_p99']:.2f}%, p99={top_above.loc[f, 'p99']:.2f}, max={top_above.loc[f, 'max']:.2f})"
            for f in top_above.index),
        "Top features by % below p1: " + ", ".join(
            f"{f} ({top_below.loc[f, 'pct_below_p1']:.2f}%, p1={top_below.loc[f, 'p1']:.2f}, min={top_below.loc[f, 'min']:.2f})"
            for f in top_below.index),
        f"Total NaN cells across all features (sample): {findings['total_nans']:,}",
        f"5 most extreme rows for '{top_above.index[0]}': "
        + ", ".join(f"{r['region_id']}@{r['date'][:10]}={r[top_above.index[0]]:.2f}"
                    for r in findings["extreme_rows_top1_feature"]),
    ])
    return [fig], findings


# ---------------------------------------------------------------------------
# Section 8: Distribution Shape (skewness / kurtosis → transform candidates)
# ---------------------------------------------------------------------------

def section_distribution_shape(train_sample=None, skew_threshold=2.0):
    """
    Per-feature skewness & kurtosis. Suggests log1p / signed-log / sqrt transforms
    for heavily skewed features.
    Returns (figs, findings).
    """
    if train_sample is None:
        print("Loading train sample for distribution shape analysis...")
        train_sample = _load_train_sample(n_regions=100)

    rows = []
    for feat in METEO_FEATURES:
        col = train_sample[feat].dropna()
        if len(col) == 0:
            continue
        skew = float(col.skew())
        kurt = float(col.kurt())
        rec = "none"
        if abs(skew) > skew_threshold:
            if col.min() >= 0:
                rec = "log1p"
            else:
                rec = "signed_log1p"
        elif abs(skew) > skew_threshold / 2:
            rec = "sqrt"
        rows.append({
            "feature": feat, "skew": skew, "kurt_excess": kurt,
            "min": float(col.min()), "max": float(col.max()),
            "recommended_transform": rec,
        })
    shape_table = pd.DataFrame(rows).set_index("feature")

    # Figures: 2 panels — skewness bar + kurtosis bar, sorted by |skew|.
    sorted_skew = shape_table["skew"].reindex(shape_table["skew"].abs().sort_values(ascending=False).index)
    sorted_kurt = shape_table["kurt_excess"].reindex(sorted_skew.index)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors_s = ["crimson" if abs(v) > skew_threshold else
                "tomato" if abs(v) > skew_threshold / 2 else "steelblue"
                for v in sorted_skew.values]
    axes[0].barh(sorted_skew.index, sorted_skew.values, color=colors_s)
    axes[0].axvline(0, color="black", linewidth=0.6)
    axes[0].axvline(skew_threshold, color="red", linestyle="--", linewidth=0.7)
    axes[0].axvline(-skew_threshold, color="red", linestyle="--", linewidth=0.7)
    axes[0].set_title(f"Skewness  (|skew| > {skew_threshold} flagged for log-transform)")
    axes[0].invert_yaxis()

    axes[1].barh(sorted_kurt.index, sorted_kurt.values, color="steelblue")
    axes[1].axvline(0, color="black", linewidth=0.6)
    axes[1].set_title("Excess Kurtosis  (0 = normal; high → heavy tails)")
    axes[1].invert_yaxis()
    fig.tight_layout()

    log_candidates = shape_table[shape_table["recommended_transform"].isin(["log1p", "signed_log1p"])]
    sqrt_candidates = shape_table[shape_table["recommended_transform"] == "sqrt"]

    findings = {
        "section": 8,
        "shape_table": shape_table.round(4).to_dict(orient="index"),
        "log_transform_candidates": log_candidates.round(4).to_dict(orient="index"),
        "sqrt_transform_candidates": sqrt_candidates.round(4).to_dict(orient="index"),
    }
    _print_findings(8, "Distribution Shape", [
        f"Log-transform candidates (|skew| > {skew_threshold}): "
        + (", ".join(f"{f}(skew={shape_table.loc[f, 'skew']:.2f} → {shape_table.loc[f, 'recommended_transform']})"
                     for f in log_candidates.index) or "none"),
        f"Sqrt-transform candidates (|skew| ∈ ({skew_threshold/2}, {skew_threshold}]): "
        + (", ".join(f"{f}(skew={shape_table.loc[f, 'skew']:.2f})" for f in sqrt_candidates.index) or "none"),
        "Highest excess kurtosis (heavy-tailed): " + ", ".join(
            f"{f}(kurt={shape_table.loc[f, 'kurt_excess']:.1f})"
            for f in shape_table.nlargest(3, "kurt_excess").index),
    ])
    return [fig], findings


# ---------------------------------------------------------------------------
# Section 9: Train vs. Test Distribution Shift (Kolmogorov-Smirnov)
# ---------------------------------------------------------------------------

def section_train_vs_test_shift(train_sample=None, test_df=None, ks_alert=0.1):
    """
    KS-test per feature between train sample and full test set. Quantifies the
    distribution shift that drives most of the val→Kaggle gap.
    Returns (figs, findings).
    """
    if train_sample is None:
        print("Loading train sample...")
        train_sample = _load_train_sample(n_regions=100)
    if test_df is None:
        print("Loading test...")
        test_df = _load_test()

    rows = []
    rng = np.random.default_rng(0)
    n_max = 50_000
    for feat in METEO_FEATURES:
        t = train_sample[feat].dropna().values
        s = test_df[feat].dropna().values
        if len(t) == 0 or len(s) == 0:
            continue
        if len(t) > n_max:
            t = rng.choice(t, n_max, replace=False)
        if len(s) > n_max:
            s = rng.choice(s, n_max, replace=False)
        ks_stat, ks_pval = scipy_stats.ks_2samp(t, s)
        rows.append({
            "feature": feat,
            "ks_distance": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "train_mean": float(np.mean(t)),
            "test_mean": float(np.mean(s)),
            "train_median": float(np.median(t)),
            "test_median": float(np.median(s)),
            "mean_delta": float(np.mean(s) - np.mean(t)),
        })
    ks_table = pd.DataFrame(rows).set_index("feature").sort_values("ks_distance", ascending=False)

    meaningful = ks_table[ks_table["ks_distance"] > ks_alert]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["crimson" if v > ks_alert else "tomato" if v > ks_alert / 2 else "steelblue"
              for v in ks_table["ks_distance"]]
    ax.barh(ks_table.index, ks_table["ks_distance"], color=colors)
    ax.axvline(ks_alert, color="red", linestyle="--", linewidth=0.8, label=f"KS={ks_alert}")
    ax.set_xlabel("Kolmogorov-Smirnov distance (0=identical, 1=disjoint)")
    ax.set_title("Train vs. Test distribution shift per feature")
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()

    findings = {
        "section": 9,
        "ks_table": ks_table.round(4).to_dict(orient="index"),
        "meaningful_shifts": meaningful.round(4).to_dict(orient="index"),
    }
    if len(meaningful) > 0:
        interp_lines = []
        for f in meaningful.index:
            d = ks_table.loc[f, "mean_delta"]
            direction = "higher" if d > 0 else "lower"
            interp_lines.append(
                f"{f}: KS={ks_table.loc[f, 'ks_distance']:.3f}, test mean is {abs(d):.3f} {direction} than train"
            )
    else:
        interp_lines = [f"No feature with KS > {ks_alert} — distributions are largely overlapping"]
    _print_findings(9, "Train vs. Test Distribution Shift", [
        f"Features with KS > {ks_alert}: " + (", ".join(meaningful.index) or "none"),
        *interp_lines,
        "All KS p-values are 0.0 below ~50k samples — KS is too sensitive to be a "
        "significance test here; rely on the distance value for ranking.",
    ])
    return [fig], findings


# ---------------------------------------------------------------------------
# Section 10: Score Transitions (Markov-style P(s[t+k] | s[t]))
# ---------------------------------------------------------------------------

def section_score_transitions(scored_df=None, horizons=(1, 2, 3, 4, 5)):
    """
    Aggregate transition matrices P(score[t+k] | score[t]) for k=1..5 weeks.
    Reveals persistence of severe events and zero stretches — critical for
    deciding how much "carry-over" information matters at each horizon.
    Returns (figs, findings).
    """
    if scored_df is None:
        print("Loading scored rows...")
        scored_df = _load_full_train_scored()

    scored_df = scored_df.sort_values(["region_id", "date"])
    n_states = 6  # scores 0..5

    transitions = {k: np.zeros((n_states, n_states), dtype=np.int64) for k in horizons}
    for _, region_df in scored_df.groupby("region_id", sort=False):
        scores = region_df["score"].values.astype(np.int8)
        for k in horizons:
            if len(scores) <= k:
                continue
            src = scores[:-k]
            dst = scores[k:]
            for s in range(n_states):
                mask = (src == s)
                if not mask.any():
                    continue
                d_counts = np.bincount(dst[mask], minlength=n_states)
                transitions[k][s] += d_counts

    # Normalize rows → P(dst | src)
    probs = {}
    for k, mat in transitions.items():
        row_sums = mat.sum(axis=1, keepdims=True)
        p = np.where(row_sums > 0, mat / np.maximum(row_sums, 1), 0)
        probs[k] = p

    # Figure: 5 heatmaps in one row
    fig, axes = plt.subplots(1, len(horizons), figsize=(4 * len(horizons), 4))
    if len(horizons) == 1:
        axes = [axes]
    for ax, k in zip(axes, horizons):
        sns.heatmap(probs[k], annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    vmin=0, vmax=1, cbar=False,
                    xticklabels=list(range(n_states)), yticklabels=list(range(n_states)))
        ax.set_title(f"P(score[t+{k}] | score[t])")
        ax.set_xlabel("score[t+k]")
        ax.set_ylabel("score[t]" if k == horizons[0] else "")
    fig.suptitle("Score transition probabilities — aggregated across regions", y=1.02)
    fig.tight_layout()

    # Derived metrics
    persistence_zero = {k: float(probs[k][0, 0]) for k in horizons}
    persistence_severe = {
        k: float(probs[k][3:, 3:].sum(axis=1).mean()) for k in horizons
    }  # mean over src∈{3,4,5} of P(dst≥3 | src)
    decay_from_5 = {k: float(probs[k][5, 3:].sum()) for k in horizons}

    findings = {
        "section": 10,
        "matrices": {k: probs[k].round(4).tolist() for k in horizons},
        "persistence_P_zero_given_zero": persistence_zero,
        "persistence_P_severe_given_severe": persistence_severe,
        "decay_P_severe_given_score5": decay_from_5,
    }
    _print_findings(10, "Score Transitions", [
        "P(score[t+k]=0 | score[t]=0): "
        + ", ".join(f"k={k}: {persistence_zero[k]:.3f}" for k in horizons)
        + "  → zeros are highly persistent (model can confidently predict 0 for low-risk regions)",
        "P(score[t+k]≥3 | score[t]≥3) (avg over score 3-5): "
        + ", ".join(f"k={k}: {persistence_severe[k]:.3f}" for k in horizons)
        + "  → severe events DO carry over",
        "P(score[t+k]≥3 | score[t]=5): "
        + ", ".join(f"k={k}: {decay_from_5[k]:.3f}" for k in horizons),
    ])
    return [fig], findings


# ---------------------------------------------------------------------------
# Section 11: Region Clustering (per-region score-pattern archetypes)
# ---------------------------------------------------------------------------

def section_region_clustering(scored_df=None, n_clusters=8, random_state=42):
    """
    Cluster regions by their historical score distribution shape. The plan's
    region-stats features capture mean/std/zero_rate; this section asks 'do
    regions sort into a handful of archetypes?' — informing whether a single
    global model is over-stretched or per-cluster modeling could help later.
    Returns (figs, findings).
    """
    if scored_df is None:
        print("Loading scored rows...")
        scored_df = _load_full_train_scored()

    # Build per-region feature table.
    region_feats = (
        scored_df.groupby("region_id")["score"]
        .agg(
            zero_rate=lambda x: float((x == 0).mean()),
            mean_score="mean",
            std_score="std",
            max_score="max",
            p90_score=lambda x: float(np.quantile(x, 0.90)),
        )
        .fillna(0)
        .astype(np.float32)
    )

    # Standardize features before k-means.
    z = (region_feats - region_feats.mean()) / region_feats.std().replace(0, 1)
    z = z.fillna(0)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    region_feats["cluster"] = km.fit_predict(z.values)

    # Cluster summary.
    centroids = region_feats.groupby("cluster").mean().round(3)
    sizes = region_feats.groupby("cluster").size().rename("n_regions")
    cluster_summary = centroids.join(sizes).sort_values("mean_score", ascending=False)

    # PCA-2D for the scatter.
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(z.values)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    palette = sns.color_palette("tab10", n_clusters)
    for c in range(n_clusters):
        mask = region_feats["cluster"].values == c
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=8,
                        color=palette[c], alpha=0.6,
                        label=f"C{c} (n={int(sizes.loc[c])})")
    axes[0].set_xlabel(f"PCA-1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PCA-2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].set_title(f"Regions in PCA-2D, colored by k-means cluster (k={n_clusters})")
    axes[0].legend(fontsize=7, ncol=2)

    # Heatmap of centroids.
    sns.heatmap(cluster_summary.drop(columns=["n_regions"]),
                annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1])
    axes[1].set_title("Cluster centroids (rows sorted by mean_score)")
    fig.tight_layout()

    # Auto-named archetypes.
    archetypes = {}
    for c, row in cluster_summary.iterrows():
        if row["zero_rate"] > 0.95 and row["max_score"] <= 2:
            label = "quiet"
        elif row["mean_score"] > 1.5:
            label = "high-severity"
        elif row["zero_rate"] > 0.85:
            label = "mostly-quiet"
        elif row["std_score"] > 0.9:
            label = "volatile"
        else:
            label = "moderate"
        archetypes[int(c)] = {
            "label": label,
            "n_regions": int(row["n_regions"]),
            "zero_rate": float(row["zero_rate"]),
            "mean_score": float(row["mean_score"]),
            "max_score": float(row["max_score"]),
        }

    high_severity = [c for c, a in archetypes.items() if a["label"] == "high-severity"]

    findings = {
        "section": 11,
        "cluster_summary": cluster_summary.to_dict(orient="index"),
        "archetypes": archetypes,
        "high_severity_clusters": high_severity,
        "n_high_severity_regions": sum(archetypes[c]["n_regions"] for c in high_severity),
    }
    _print_findings(11, "Region Clustering", [
        f"Cluster sizes (sorted by mean_score desc): "
        + ", ".join(f"C{c}={archetypes[c]['label']}({archetypes[c]['n_regions']})"
                    for c in cluster_summary.index),
        f"High-severity clusters: {high_severity}  →  "
        f"{findings['n_high_severity_regions']} regions (~{100*findings['n_high_severity_regions']/len(region_feats):.1f}%) "
        "drive most of the rare-event MAE",
        f"PCA-2D explains {sum(pca.explained_variance_ratio_)*100:.1f}% of region-feature variance",
    ])
    return [fig], findings


# ---------------------------------------------------------------------------
# Section 12: Lag Cross-Correlation (feature vs. score over k = 0..12 weeks)
# ---------------------------------------------------------------------------

def section_lag_cross_correlation(scored_df=None, max_lag=12):
    """
    For each meteo feature × each lag k, compute mean (across regions) of
    Pearson corr(feature[t-k], score[t]). The heatmap reveals which features
    predict score and at what horizon — a sanity check on LAG_WINDOW=12 and
    a hint at which lags carry the strongest signal.
    Returns (figs, findings).
    """
    if scored_df is None:
        print("Loading scored rows...")
        scored_df = _load_full_train_scored()

    scored_df = scored_df.sort_values(["region_id", "date"]).reset_index(drop=True)

    # Per-region correlation, then average across regions.
    region_corrs = []
    for _, reg_df in scored_df.groupby("region_id", sort=False):
        score = reg_df["score"].values.astype(np.float32)
        if len(score) < max_lag + 5:
            continue
        if score.std() < 1e-6:
            continue  # all-constant regions can't contribute
        per_lag = np.zeros((len(METEO_FEATURES), max_lag + 1), dtype=np.float32)
        for fi, feat in enumerate(METEO_FEATURES):
            feat_arr = reg_df[feat].values.astype(np.float32)
            for k in range(max_lag + 1):
                if k == 0:
                    a = feat_arr
                    b = score
                else:
                    a = feat_arr[:-k]
                    b = score[k:]
                if a.std() < 1e-6 or b.std() < 1e-6:
                    per_lag[fi, k] = 0.0
                else:
                    per_lag[fi, k] = float(np.corrcoef(a, b)[0, 1])
        region_corrs.append(per_lag)

    region_corrs = np.stack(region_corrs, axis=0)  # (n_regions, n_feats, n_lags+1)
    mean_corr = region_corrs.mean(axis=0)         # (n_feats, n_lags+1)
    corr_df = pd.DataFrame(mean_corr, index=METEO_FEATURES,
                           columns=[f"lag{k}" for k in range(max_lag + 1)])

    # Best-lag per feature (by absolute correlation).
    abs_corr = np.abs(mean_corr)
    best_lag_idx = abs_corr.argmax(axis=1)
    best_lag = {METEO_FEATURES[i]: (int(best_lag_idx[i]), float(mean_corr[i, best_lag_idx[i]]))
                for i in range(len(METEO_FEATURES))}
    sorted_by_strength = sorted(best_lag.items(), key=lambda kv: abs(kv[1][1]), reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-0.3, vmax=0.3, ax=ax, annot_kws={"size": 7})
    ax.set_title(f"Mean Pearson corr(feature[t-k], score[t])  —  averaged across regions")
    ax.set_xlabel("Lag in weeks")
    fig.tight_layout()

    findings = {
        "section": 12,
        "mean_corr": corr_df.round(4).to_dict(orient="index"),
        "best_lag_per_feature": best_lag,
    }
    _print_findings(12, "Lag Cross-Correlation", [
        "Top features at their BEST lag: " + ", ".join(
            f"{name}(lag {lag}, r={r:+.3f})"
            for name, (lag, r) in sorted_by_strength[:5]),
        "Features whose best lag is 0 (immediate weather → severity): " + ", ".join(
            name for name, (lag, _) in best_lag.items() if lag == 0) or "none",
        "Features whose best lag is > 4 weeks (deeper memory matters): " + (", ".join(
            f"{name}(lag {lag})" for name, (lag, _) in best_lag.items() if lag > 4) or "none"),
        f"LAG_WINDOW=12 is currently used — best lags observed: "
        f"min={min(v[0] for v in best_lag.values())}, max={max(v[0] for v in best_lag.values())}",
    ])
    return [fig], findings


# ---------------------------------------------------------------------------
# Aggregated takeaway helper for the notebook's final cell
# ---------------------------------------------------------------------------

def print_aggregated_findings(findings_dicts: list):
    """
    Pretty-print the union of all sections' findings dicts at the end of the
    notebook. Used by the final "Auto-aggregated EDA Findings" cell.
    """
    print()
    print("=" * 70)
    print("  AGGREGATED EDA FINDINGS")
    print("=" * 70)
    for f in findings_dicts:
        if not f:
            continue
        s = f.get("section", "?")
        print(f"\n  Section {s}:")
        for k, v in f.items():
            if k == "section":
                continue
            # Top-level scalars are most useful in the summary; skip large nested dicts.
            if isinstance(v, (int, float, str)):
                print(f"    {k}: {v}")
            elif isinstance(v, list) and len(v) <= 10 and all(isinstance(x, (int, float, str)) for x in v):
                print(f"    {k}: {v}")
            else:
                if isinstance(v, dict):
                    print(f"    {k}: <{len(v)}-entry dict, see section output>")
                else:
                    print(f"    {k}: <complex value, see section output>")
    print()
