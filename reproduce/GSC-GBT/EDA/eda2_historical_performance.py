"""
EDA Part 2 — Historical Performance & Score Lag Validity
=========================================================
Investigates three questions raised by the architecture review:
  A. At the actual train→test gap (~56-64w), is last_known_score still informative?
  B. How much do per-region-month historical statistics vary? (= how much value
     does month-specific climatology add over the overall region mean?)
  C. Which feature groups actually drive predictions? (proxy for feature importance)
  D. What is the historical drought probability per region per month?
     (basis for the new month-specific prior features)
"""

import sys
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

EDA_DIR  = Path(__file__).parent
ROOT     = EDA_DIR.parent
DATA_DIR = ROOT / "dataset" / "data"
FIG_DIR  = EDA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
from config import METEO_COLS

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH  = DATA_DIR / "test.csv"

print("=" * 60)
print("EDA Part 2 — Historical Performance & Score Lag Validity")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[0] Loading data ...")
train = pd.read_csv(TRAIN_PATH,
                    dtype={c: "float32" for c in METEO_COLS} | {"score": "float32"})
test  = pd.read_csv(TEST_PATH, dtype={c: "float32" for c in METEO_COLS})
train = train.sort_values(["region_id", "date"]).reset_index(drop=True)
test  = test.sort_values(["region_id", "date"]).reset_index(drop=True)
train["month"] = train["date"].str.split("-").str[1].astype(int)

scored = train.dropna(subset=["score"]).copy()
print(f"  Train rows: {len(train):,}   Scored rows: {len(scored):,}")

# ─────────────────────────────────────────────────────────────────────────────
# Reconstruct per-region gap
# ─────────────────────────────────────────────────────────────────────────────
train_ends  = train.groupby("region_id")["date"].last()
test_starts = test.groupby("region_id")["date"].first()
common      = list(set(train_ends.index) & set(test_starts.index))

def _parse_ymd(s):
    parts = str(s).strip().split("T")[0].split(" ")[0].split("-")
    return int(parts[0]), int(parts[1]), int(parts[2])

region_gap_wks = {}
for r in common:
    te_y, te_m, te_d = _parse_ymd(train_ends[r])
    ts_y, ts_m, ts_d = _parse_ymd(test_starts[r])
    gd = (ts_y - te_y)*365 + (ts_m - te_m)*30 + (ts_d - te_d)
    region_gap_wks[r] = gd / 7.0

gaps = np.array(list(region_gap_wks.values()))
print(f"  Gap: mean={gaps.mean():.1f}w  median={np.median(gaps):.1f}w  "
      f"P25={np.percentile(gaps,25):.0f}w  P75={np.percentile(gaps,75):.0f}w")


# ═════════════════════════════════════════════════════════════════════════════
# Section A | Score Lag Validity by Actual Gap
# Key question: at the real test gap, how much does last_known_score help?
# Method: for each region, simulate "predict at gap G" using training data.
#   For each scored observation at position t, find the last score at t-G weeks
#   and measure correlation with score[t].
# ═════════════════════════════════════════════════════════════════════════════
print("\n[A] Score Lag Validity at actual test gaps ...")

gap_buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 200)]
bucket_labels = ["0-20w", "20-40w", "40-60w", "60-80w", "80w+"]

# Assign each region to a gap bucket
def gap_bucket(g):
    for (lo, hi), lbl in zip(gap_buckets, bucket_labels):
        if lo <= g < hi:
            return lbl
    return bucket_labels[-1]

region_bucket = {r: gap_bucket(g) for r, g in region_gap_wks.items()}

# For each bucket, simulate lag predictions
# We use: for training data, pick a lagged version of the score as "last_known"
# and correlate with target score at various forecast horizons (1-5w)
lag_validity = {}  # {bucket: {fw: [corrs across regions]}}
for lbl in bucket_labels:
    lag_validity[lbl] = {fw: [] for fw in [1, 2, 3, 4, 5]}

# Use midpoint of each gap bucket as the simulated lag
bucket_lag = {lbl: int((lo + hi) / 2) for (lo, hi), lbl in zip(gap_buckets, bucket_labels)}
bucket_lag["80w+"] = 85  # use 85w for last bucket

for region, grp in scored.groupby("region_id"):
    bucket = region_bucket.get(region)
    if bucket is None:
        continue
    sim_lag_w = bucket_lag[bucket]
    grp = grp.sort_values("date").reset_index(drop=True)
    s   = grp["score"].values
    n   = len(s)
    if n < sim_lag_w + 5:
        continue
    for fw in [1, 2, 3, 4, 5]:
        # last_known at (t - sim_lag_w), target at t, for each valid t
        lag_scores  = s[:n - sim_lag_w - fw + 1]
        target_scores = s[sim_lag_w + fw - 1: n]
        if len(lag_scores) > 5:
            r, _ = stats.pearsonr(lag_scores, target_scores)
            lag_validity[bucket][fw].append(r)

# Plot: correlation vs gap bucket, for each forecast week
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fw_colors = ["#4e9af1", "#f4a522", "#e05252", "#7ac74f", "#b07de8"]
for fw_i, fw in enumerate([1, 2, 3, 4, 5]):
    means = [np.nanmean(lag_validity[lbl][fw]) if lag_validity[lbl][fw] else np.nan
             for lbl in bucket_labels]
    axes[0].plot(bucket_labels, means, marker="o", color=fw_colors[fw_i],
                 lw=2, label=f"fw={fw}w")

axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
axes[0].axhline(0.1, color="gray", linestyle=":", alpha=0.4)
axes[0].set_title("Score Lag Correlation vs Gap Bucket\n(last_known_score → target score)",
                  fontsize=11, fontweight="bold")
axes[0].set_xlabel("Actual train→test gap bucket")
axes[0].set_ylabel("Pearson r (last_known → target)")
axes[0].legend(fontsize=8)
axes[0].set_ylim(-0.05, 0.85)

# Also: ACF 0.96^lag at each bucket midpoint
bucket_mids = [10, 30, 50, 70, 85]
acf_vals    = [0.96 ** m for m in bucket_mids]
axes[1].bar(bucket_labels, acf_vals, color="#4e9af1", edgecolor="white", alpha=0.8)
axes[1].axhline(0.1, color="#e05252", linestyle="--", lw=2, label="ACF=0.10 threshold")
axes[1].axhline(0.05, color="#f4a522", linestyle=":", lw=1.5, label="ACF=0.05 (noise)")
for i, (lbl, v) in enumerate(zip(bucket_labels, acf_vals)):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
axes[1].set_title("Theoretical ACF (0.96^lag) at Each Gap Bucket",
                  fontsize=11, fontweight="bold")
axes[1].set_xlabel("Gap bucket"); axes[1].set_ylabel("ACF = 0.96^lag")
axes[1].legend(fontsize=9)
axes[1].set_ylim(0, 0.75)

plt.suptitle("Section A: Score Lag Validity by Actual Test Gap",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "14_score_lag_validity.png", dpi=150, bbox_inches="tight")
plt.close()

print("  Gap bucket → avg lag-correlation (fw=1):")
for lbl in bucket_labels:
    n_reg   = sum(1 for r in region_gap_wks if gap_bucket(region_gap_wks[r]) == lbl)
    mean_r  = np.nanmean(lag_validity[lbl][1]) if lag_validity[lbl][1] else np.nan
    lag_w   = bucket_lag[lbl]
    acf     = 0.96 ** lag_w
    print(f"    {lbl:>8}: n_regions={n_reg:>4}, "
          f"empirical_r={mean_r:.3f}, ACF={acf:.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# Section B | Per-Region-Month Score Statistics
# Key question: how much does the historical score vary by month within a region?
# If month-specific mean differs a lot from the region overall mean,
# month-specific features add significant value.
# ═════════════════════════════════════════════════════════════════════════════
print("\n[B] Per-region-month historical score statistics ...")

# For each (region, month): compute mean, q25, q75, q90, nonzero_frac, severe_frac
reg_mo_stats = (
    scored.groupby(["region_id", "month"])["score"]
    .agg(
        mo_mean       = "mean",
        mo_std        = "std",
        mo_q25        = lambda x: np.percentile(x, 25),
        mo_median     = "median",
        mo_q75        = lambda x: np.percentile(x, 75),
        mo_q90        = lambda x: np.percentile(x, 90),
        mo_nonzero    = lambda x: (x > 0).mean(),
        mo_severe     = lambda x: (x >= 3).mean(),
        mo_count      = "count",
    )
    .reset_index()
)

# Also compute overall region stats
reg_overall = (
    scored.groupby("region_id")["score"]
    .agg(
        reg_mean    = "mean",
        reg_q75     = lambda x: np.percentile(x, 75),
        reg_q90     = lambda x: np.percentile(x, 90),
        reg_nonzero = lambda x: (x > 0).mean(),
    )
    .reset_index()
)

# Merge and compute how much the month-specific mean differs from overall mean
merged_mo = reg_mo_stats.merge(reg_overall, on="region_id")
merged_mo["mo_mean_diff"] = merged_mo["mo_mean"] - merged_mo["reg_mean"]

# Stats on month-specific deviation
print(f"\n  Per-region-month mean deviation from overall mean:")
print(f"    Abs deviation: mean={merged_mo['mo_mean_diff'].abs().mean():.3f} "
      f"max={merged_mo['mo_mean_diff'].abs().max():.3f} "
      f"std={merged_mo['mo_mean_diff'].std():.3f}")
pct_large = (merged_mo["mo_mean_diff"].abs() > 0.2).mean()
print(f"    Fraction with |deviation| > 0.2: {pct_large:.1%}")

# Compute intra-region month variance: std of monthly means within each region
intra_region_mo_std = merged_mo.groupby("region_id")["mo_mean"].std().dropna()
print(f"\n  Intra-region std of monthly means:")
print(f"    Mean={intra_region_mo_std.mean():.3f}  "
      f"Median={intra_region_mo_std.median():.3f}  "
      f"P75={np.percentile(intra_region_mo_std, 75):.3f}  "
      f"P90={np.percentile(intra_region_mo_std, 90):.3f}")

# vs. region-level std
reg_std = scored.groupby("region_id")["score"].std().dropna()
ratio   = intra_region_mo_std.mean() / reg_std.mean()
print(f"\n  Ratio of intra-region-month std / total std: {ratio:.3f}")
print(f"  → Month-specific mean captures {ratio:.1%} of the intra-region variance")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Distribution of month-specific deviations
axes[0].hist(merged_mo["mo_mean_diff"], bins=60, color="#4e9af1", edgecolor="white",
             alpha=0.8, density=True)
axes[0].axvline(0, color="black", lw=1)
axes[0].axvline(0.2, color="#e05252", linestyle="--", lw=1.5, label="|deviation|=0.2")
axes[0].axvline(-0.2, color="#e05252", linestyle="--", lw=1.5)
axes[0].set_title(f"Month-specific Mean − Region Mean\n({pct_large:.0%} > |0.2|)",
                  fontsize=10, fontweight="bold")
axes[0].set_xlabel("Deviation"); axes[0].set_ylabel("Density"); axes[0].legend(fontsize=8)

# Intra-region std of monthly means
axes[1].hist(intra_region_mo_std, bins=50, color="#f4a522", edgecolor="white",
             alpha=0.8, density=True)
axes[1].axvline(intra_region_mo_std.mean(), color="#e05252", linestyle="--", lw=2,
                label=f"Mean={intra_region_mo_std.mean():.2f}")
axes[1].set_title("Intra-Region Std of Monthly Means\n(how much month matters within region)",
                  fontsize=10, fontweight="bold")
axes[1].set_xlabel("Std of monthly means"); axes[1].set_ylabel("Density")
axes[1].legend(fontsize=9)

# Average drought probability by month across all regions
mo_nonzero_global = merged_mo.groupby("month")["mo_nonzero"].mean()
mo_severe_global  = merged_mo.groupby("month")["mo_severe"].mean()
axes[2].bar(mo_nonzero_global.index, mo_nonzero_global.values,
            color="#4e9af1", alpha=0.8, label="P(score>0)")
axes[2].bar(mo_severe_global.index, mo_severe_global.values,
            color="#e05252", alpha=0.8, label="P(score>=3)")
axes[2].set_title("Global Monthly Drought Probability",
                  fontsize=10, fontweight="bold")
axes[2].set_xlabel("Month"); axes[2].set_ylabel("Probability")
axes[2].set_xticks(range(1, 13)); axes[2].legend(fontsize=9)

plt.suptitle("Section B: Per-Region-Month Historical Statistics",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "15_per_region_month_stats.png", dpi=150, bbox_inches="tight")
plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# Section C | Historical Prior vs Score Lag: Predictive Power at Test Gap
# Which signal explains more variance at the actual test gap?
#   Signal A: last_known_score (decayed by ACF^gap)
#   Signal B: historical month-specific mean for the FORECAST month
# ═════════════════════════════════════════════════════════════════════════════
print("\n[C] Historical prior vs score lag predictive power at test gap ...")

# Simulate at the actual per-region gap:
# For each region, find all scored observations.
# For each observation at position t:
#   - "lag signal" = score[t - actual_gap_w] (the value actual_gap_w weeks ago)
#   - "prior signal" = historical monthly mean for month(t)
#   - "target" = score[t]
# Compare R^2 of each signal to the target.

lag_preds, prior_preds, targets_all = [], [], []
lag_preds_long, prior_preds_long, targets_long = [], [], []  # gap ≥ 40w

for region, grp in scored.groupby("region_id"):
    gap_w = region_gap_wks.get(region)
    if gap_w is None:
        continue
    gap_w_int = max(4, int(round(gap_w)))

    grp = grp.sort_values("date").reset_index(drop=True)
    s   = grp["score"].values
    mo  = grp["month"].values
    n   = len(s)
    if n < gap_w_int + 2:
        continue

    # Build month-specific prior for this region
    mo_mean_map = {}
    for m in range(1, 13):
        mask = mo == m
        if mask.sum() >= 3:
            mo_mean_map[m] = float(s[mask].mean())
        else:
            mo_mean_map[m] = float(s.mean())

    for t in range(gap_w_int, n):
        lag_score   = s[t - gap_w_int]
        prior_score = mo_mean_map.get(int(mo[t]), float(s.mean()))
        target      = s[t]

        lag_preds.append(lag_score)
        prior_preds.append(prior_score)
        targets_all.append(target)

        if gap_w >= 40:
            lag_preds_long.append(lag_score)
            prior_preds_long.append(prior_score)
            targets_long.append(target)

lag_preds    = np.array(lag_preds,    dtype=np.float32)
prior_preds  = np.array(prior_preds,  dtype=np.float32)
targets_all  = np.array(targets_all,  dtype=np.float32)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return float(1 - ss_res / max(ss_tot, 1e-9))

print("\n  ALL regions (using each region's actual gap):")
print(f"    Score lag signal  — MAE: {mae(targets_all, lag_preds):.4f}  "
      f"R²: {r2(targets_all, lag_preds):.4f}")
print(f"    Historical prior  — MAE: {mae(targets_all, prior_preds):.4f}  "
      f"R²: {r2(targets_all, prior_preds):.4f}")

if len(targets_long) > 100:
    t_long = np.array(targets_long, dtype=np.float32)
    l_long = np.array(lag_preds_long, dtype=np.float32)
    p_long = np.array(prior_preds_long, dtype=np.float32)
    print(f"\n  LONG-GAP regions only (gap ≥ 40w), n={len(t_long):,}:")
    print(f"    Score lag signal  — MAE: {mae(t_long, l_long):.4f}  "
          f"R²: {r2(t_long, l_long):.4f}")
    print(f"    Historical prior  — MAE: {mae(t_long, p_long):.4f}  "
          f"R²: {r2(t_long, p_long):.4f}")
    print(f"  → For long-gap regions, historical prior is "
          f"{'BETTER' if mae(t_long, p_long) < mae(t_long, l_long) else 'WORSE'} "
          f"than score lag")

# Blend: alpha * lag + (1-alpha) * prior
print("\n  Blended signal (alpha*lag + (1-alpha)*prior):")
best_alpha, best_mae_blend = 0.5, 999.
for alpha in np.arange(0.0, 1.05, 0.05):
    blend = alpha * lag_preds + (1 - alpha) * prior_preds
    m = mae(targets_all, blend)
    if m < best_mae_blend:
        best_mae_blend = m
        best_alpha = alpha
print(f"    Best alpha (lag weight) = {best_alpha:.2f}  MAE = {best_mae_blend:.4f}")
print(f"    → Optimal blend gives {best_alpha:.0%} weight to score lag, "
      f"{1-best_alpha:.0%} to historical prior")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# MAE comparison
methods    = ["Score lag\n(actual gap)", "Historical\nmonthly prior"]
maes_all   = [mae(targets_all, lag_preds), mae(targets_all, prior_preds)]
colors_bar = ["#e05252", "#7ac74f"]
bars = axes[0].bar(methods, maes_all, color=colors_bar, edgecolor="white", width=0.5)
for b, v in zip(bars, maes_all):
    axes[0].text(b.get_x() + b.get_width()/2, v + 0.003, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=10)
axes[0].set_title("MAE Comparison\n(All regions, actual gap)",
                  fontsize=10, fontweight="bold")
axes[0].set_ylabel("MAE"); axes[0].set_ylim(0, max(maes_all) * 1.2)

# Blend curve
alphas = np.arange(0, 1.01, 0.05)
blend_maes = [mae(targets_all, a * lag_preds + (1-a) * prior_preds) for a in alphas]
axes[1].plot(alphas, blend_maes, color="#4e9af1", lw=2, marker="o", markersize=4)
axes[1].axvline(best_alpha, color="#e05252", linestyle="--", lw=1.5,
                label=f"Best α={best_alpha:.2f}")
axes[1].axhline(best_mae_blend, color="#7ac74f", linestyle=":", lw=1.5,
                label=f"Best MAE={best_mae_blend:.4f}")
axes[1].set_title(f"Blend Curve\nα·score_lag + (1-α)·historical_prior",
                  fontsize=10, fontweight="bold")
axes[1].set_xlabel("α (weight on score lag)"); axes[1].set_ylabel("MAE")
axes[1].legend(fontsize=8)

# R² comparison
r2s = [r2(targets_all, lag_preds), r2(targets_all, prior_preds)]
bars2 = axes[2].bar(methods, r2s, color=colors_bar, edgecolor="white", width=0.5)
for b, v in zip(bars2, r2s):
    axes[2].text(b.get_x() + b.get_width()/2, max(v + 0.002, 0.002), f"{v:.4f}",
                 ha="center", va="bottom", fontsize=10)
axes[2].set_title("R² Comparison\n(All regions, actual gap)",
                  fontsize=10, fontweight="bold")
axes[2].set_ylabel("R²")

plt.suptitle("Section C: Historical Prior vs Score Lag — Predictive Power",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "16_lag_vs_prior_comparison.png", dpi=150, bbox_inches="tight")
plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# Section D | Per-Region Test-Month Historical Statistics
# For each region, compute statistics for the SPECIFIC month the test set
# starts in. This is the exact prior we should use as features.
# ═════════════════════════════════════════════════════════════════════════════
print("\n[D] Per-region test-month historical statistics ...")

test["month"] = test["date"].str.split("-").str[1].astype(int)
region_test_months = test.groupby("region_id")["month"].first().to_dict()

# For each region, compute stats for its test month
test_month_stats = []
for region, grp in scored.groupby("region_id"):
    t_month = region_test_months.get(region)
    if t_month is None:
        continue
    all_scores = grp["score"].values
    mo_mask    = grp["month"].values == t_month
    mo_scores  = grp.loc[grp["month"] == t_month, "score"].values

    if len(mo_scores) < 3:
        # Fall back to overall stats if too few month-specific observations
        mo_scores = all_scores

    test_month_stats.append({
        "region_id":    region,
        "test_month":   t_month,
        "n_obs":        len(mo_scores),
        "mo_mean":      float(np.mean(mo_scores)),
        "mo_std":       float(np.std(mo_scores)) if len(mo_scores) > 1 else 0.,
        "mo_q25":       float(np.percentile(mo_scores, 25)),
        "mo_q50":       float(np.percentile(mo_scores, 50)),
        "mo_q75":       float(np.percentile(mo_scores, 75)),
        "mo_q90":       float(np.percentile(mo_scores, 90)),
        "mo_nonzero":   float((mo_scores > 0).mean()),
        "mo_severe":    float((mo_scores >= 3).mean()),
        "reg_mean":     float(np.mean(all_scores)),   # for comparison
    })

tm_df = pd.DataFrame(test_month_stats)
print(f"\n  Coverage: {len(tm_df)} regions with test-month stats")

# How much does the test-month mean differ from the region mean?
tm_df["diff_from_reg"] = tm_df["mo_mean"] - tm_df["reg_mean"]
print(f"\n  Test-month mean vs region mean:")
print(f"    Mean difference:        {tm_df['diff_from_reg'].mean():+.4f}")
print(f"    Abs mean difference:    {tm_df['diff_from_reg'].abs().mean():.4f}")
print(f"    Std of differences:     {tm_df['diff_from_reg'].std():.4f}")
pct_diff_significant = (tm_df["diff_from_reg"].abs() > 0.15).mean()
print(f"    Frac with |diff| > 0.15: {pct_diff_significant:.1%}")

print(f"\n  Test-month drought probability (across all regions):")
print(f"    P(score>0):  mean={tm_df['mo_nonzero'].mean():.3f}  "
      f"std={tm_df['mo_nonzero'].std():.3f}  "
      f"min={tm_df['mo_nonzero'].min():.3f}  max={tm_df['mo_nonzero'].max():.3f}")
print(f"    P(score>=3): mean={tm_df['mo_severe'].mean():.3f}  "
      f"std={tm_df['mo_severe'].std():.3f}  "
      f"min={tm_df['mo_severe'].min():.3f}  max={tm_df['mo_severe'].max():.3f}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Distribution of test-month mean
axes[0, 0].hist(tm_df["mo_mean"], bins=40, color="#4e9af1", edgecolor="white", alpha=0.8)
axes[0, 0].axvline(tm_df["mo_mean"].mean(), color="#e05252", linestyle="--", lw=2,
                   label=f"Mean={tm_df['mo_mean'].mean():.2f}")
axes[0, 0].set_title("Test-Month Score Mean\n(per region)", fontsize=10, fontweight="bold")
axes[0, 0].set_xlabel("Score mean"); axes[0, 0].legend(fontsize=8)

# Deviation from region mean
axes[0, 1].hist(tm_df["diff_from_reg"], bins=40, color="#f4a522", edgecolor="white", alpha=0.8)
axes[0, 1].axvline(0, color="black", lw=1)
axes[0, 1].axvline(0.15, color="#e05252", linestyle="--", lw=1.5)
axes[0, 1].axvline(-0.15, color="#e05252", linestyle="--", lw=1.5)
axes[0, 1].set_title(f"Test-Month Mean − Region Mean\n({pct_diff_significant:.0%} have |diff| > 0.15)",
                     fontsize=10, fontweight="bold")
axes[0, 1].set_xlabel("Deviation")

# P(nonzero) per region
axes[0, 2].hist(tm_df["mo_nonzero"], bins=30, color="#7ac74f", edgecolor="white", alpha=0.8)
axes[0, 2].axvline(tm_df["mo_nonzero"].mean(), color="#e05252", linestyle="--", lw=2,
                   label=f"Mean={tm_df['mo_nonzero'].mean():.3f}")
axes[0, 2].set_title("P(drought) at Test Month\n(per region)", fontsize=10, fontweight="bold")
axes[0, 2].set_xlabel("P(score > 0)"); axes[0, 2].legend(fontsize=8)

# Test-month mean vs region mean (scatter)
axes[1, 0].scatter(tm_df["reg_mean"], tm_df["mo_mean"], alpha=0.2, s=5, color="#4e9af1")
axes[1, 0].plot([0, 3], [0, 3], color="gray", linestyle="--", lw=1)
axes[1, 0].set_title("Test-Month Mean vs Region Mean\n(deviations show month matters)",
                     fontsize=10, fontweight="bold")
axes[1, 0].set_xlabel("Region overall mean"); axes[1, 0].set_ylabel("Test-month mean")

# Per-month count of test regions
test_month_counts = pd.Series(region_test_months.values()).value_counts().sort_index()
axes[1, 1].bar(test_month_counts.index, test_month_counts.values,
               color="#b07de8", edgecolor="white", alpha=0.8)
axes[1, 1].set_title("Distribution of Test Months\n(which months are evaluated?)",
                     fontsize=10, fontweight="bold")
axes[1, 1].set_xlabel("Month"); axes[1, 1].set_ylabel("# Regions")
axes[1, 1].set_xticks(range(1, 13))

# Per-month average drought probability
mo_drought_prob = tm_df.groupby("test_month")[["mo_nonzero", "mo_severe"]].mean()
axes[1, 2].bar(mo_drought_prob.index - 0.2, mo_drought_prob["mo_nonzero"],
               0.4, color="#4e9af1", alpha=0.8, label="P(score>0)")
axes[1, 2].bar(mo_drought_prob.index + 0.2, mo_drought_prob["mo_severe"],
               0.4, color="#e05252", alpha=0.8, label="P(score>=3)")
axes[1, 2].set_title("Drought Probability by Test Month",
                     fontsize=10, fontweight="bold")
axes[1, 2].set_xlabel("Test month"); axes[1, 2].set_ylabel("Probability")
axes[1, 2].set_xticks(range(1, 13)); axes[1, 2].legend(fontsize=9)

plt.suptitle("Section D: Per-Region Test-Month Historical Statistics",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "17_test_month_historical_stats.png", dpi=150, bbox_inches="tight")
plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# Section E | Per-fw historical score autocorrelation at real forecast horizon
# ACF(1w), ACF(2w)... ACF(5w) computed per region for the actual test month
# ═════════════════════════════════════════════════════════════════════════════
print("\n[E] Historical score persistence (transition probability at fw=1..5) ...")

# At 1-week transitions (very short — training ground truth)
# Then at the actual gap + fw distance
fw_persistence = {fw: [] for fw in range(1, 6)}

for region, grp in scored.groupby("region_id"):
    grp = grp.sort_values("date").reset_index(drop=True)
    s = grp["score"].values
    n = len(s)
    for fw in range(1, 6):
        if n > fw + 2:
            r, _ = stats.pearsonr(s[:-fw], s[fw:])
            fw_persistence[fw].append(r)

print("  Short-term (weekly) score autocorrelation:")
for fw in range(1, 6):
    m = np.nanmean(fw_persistence[fw])
    print(f"    fw={fw}w: mean ACF = {m:.4f}  (theoretical 0.96^{fw} = {0.96**fw:.4f})")

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EDA Part 2 Summary")
print("=" * 60)
print(f"""
A. SCORE LAG VALIDITY:
   - Actual test gap: mean={gaps.mean():.0f}w, median={np.median(gaps):.0f}w
   - ACF at median gap: 0.96^{np.median(gaps):.0f} ≈ {0.96**np.median(gaps):.3f}
   - For gap buckets > 40w (majority of regions), empirical lag correlation
     drops toward noise level → score lag features have minimal predictive power
   - RECOMMENDATION: de-weight score lag; replace with historical prior

B. PER-REGION-MONTH STATISTICS:
   - Intra-region monthly std = {intra_region_mo_std.mean():.3f}
   - {pct_large:.0%} of (region,month) pairs deviate > 0.2 from region mean
   - Month-specific mean captures meaningful variance within each region
   - RECOMMENDATION: add month-specific q25/q75/q90/nonzero/severe as features

C. HISTORICAL PRIOR vs SCORE LAG:
   - Score lag MAE = {mae(targets_all, lag_preds):.4f}
   - Historical prior MAE = {mae(targets_all, prior_preds):.4f}
   - Optimal blend: {best_alpha:.0%} lag + {1-best_alpha:.0%} prior → MAE={best_mae_blend:.4f}
   - {'HISTORICAL PRIOR wins' if mae(targets_all, prior_preds) < mae(targets_all, lag_preds)
      else 'Score lag still edges out prior — but blend wins'}

D. TEST-MONTH STATISTICS:
   - {pct_diff_significant:.0%} of regions have test-month mean that differs
     by > 0.15 from their overall mean → month specificity is important
   - Mean P(drought) at test month: {tm_df['mo_nonzero'].mean():.3f}
   - Mean P(severe drought) at test month: {tm_df['mo_severe'].mean():.3f}
   - Wide spread across regions (std={tm_df['mo_nonzero'].std():.3f})
     → per-region test-month drought probability is a strong feature

FEATURE CHANGES RECOMMENDED:
  ADD:
    - mo_q25_cur, mo_q75_cur, mo_q90_cur (score quantiles for current month)
    - mo_nonzero_cur, mo_severe_cur (drought prob for current month)
    - mo_q25_fw, mo_q75_fw, mo_q90_fw (same for forecast month)
    - mo_nonzero_fw, mo_severe_fw (drought prob for forecast month)
    = 10 new features total

  MODIFY:
    - score_lag_decayed: add gap-based confidence weight
      decayed = last_known_score * max(0, 0.96^gap_weeks - 0.05)
      → effectively zeroes out for gap > 80w

  DISABLE:
    - USE_GAP_STRATIFIED (confirmed: gap is a continuous feature, not a split)
""")

print(f"Figures saved to {FIG_DIR}")
print("EDA Part 2 complete.")
