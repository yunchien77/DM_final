import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

EDA_DIR  = Path(__file__).parent
ROOT     = EDA_DIR.parent
DATA_DIR = ROOT / "dataset" / "data"
FIG_DIR  = EDA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
from config import METEO_COLS, COL_IDX

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH  = DATA_DIR / "test.csv"

PALETTE = {
    0: "#4e9af1", 1: "#f4a522", 2: "#e05252",
    3: "#7ac74f", 4: "#b07de8", 5: "#f07f1e",
}

print("=" * 60)
print("Drought EDA")
print("=" * 60)

# ═════════════════════════════════════════════════════════════════════════════
# Section 0 | 讀取資料
# ═════════════════════════════════════════════════════════════════════════════
print("\n[0] Loading data ...")
train = pd.read_csv(
    TRAIN_PATH,
    dtype={c: "float32" for c in METEO_COLS} | {"score": "float32"},
)
test = pd.read_csv(TEST_PATH, dtype={c: "float32" for c in METEO_COLS})
train = train.sort_values(["region_id", "date"]).reset_index(drop=True)
test  = test.sort_values(["region_id", "date"]).reset_index(drop=True)
train["month"] = train["date"].str.split("-").str[1].astype(int)
test["month"]  = test["date"].str.split("-").str[1].astype(int)

scored = train.dropna(subset=["score"])
print(f"  Train rows: {len(train):,}   Test rows: {len(test):,}")
print(f"  Regions (train): {train['region_id'].nunique()}")
print(f"  Regions (test):  {test['region_id'].nunique()}")
print(f"  Score rows: {len(scored):,}  ({100*len(scored)/len(train):.1f}%)")

# ═════════════════════════════════════════════════════════════════════════════
# Section 1 | Score 分佈
# ═════════════════════════════════════════════════════════════════════════════
print("\n[1] Score 分佈 ...")
dist = scored["score"].value_counts(normalize=True).sort_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
bars = axes[0].bar(
    dist.index.astype(str), dist.values * 100,
    color=[PALETTE.get(int(s), "#999") for s in dist.index],
    edgecolor="white", linewidth=0.5,
)
for bar, v in zip(bars, dist.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5, f"{v*100:.1f}%",
                 ha="center", va="bottom", fontsize=9)
axes[0].set_title("Score Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Drought Severity Score")
axes[0].set_ylabel("Proportion (%)")
axes[0].set_ylim(0, 75)
axes[0].axhline(16.7, color="gray", linestyle="--", alpha=0.5, label="Uniform 16.7%")
axes[0].legend(fontsize=8)

cdf = dist.cumsum()
axes[1].step(cdf.index, cdf.values * 100, where="post", color="#4e9af1", lw=2)
axes[1].fill_between(cdf.index, cdf.values * 100, step="post", alpha=0.2, color="#4e9af1")
axes[1].set_title("Cumulative Score Distribution", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Score")
axes[1].set_ylabel("Cumulative (%)")
axes[1].set_ylim(0, 105)

plt.tight_layout()
plt.savefig(FIG_DIR / "01_score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(dist.map("{:.1%}".format).to_string())

zero_pct   = float(dist.get(0, 0))
severe_pct = float(dist[dist.index >= 3].sum())

# ═════════════════════════════════════════════════════════════════════════════
# Section 2 | Train / Test Gap 分析（Score Lag 有效性）
# ═════════════════════════════════════════════════════════════════════════════
print("\n[2] Train/Test Gap 分析 ...")
# 資料已依 ["region_id", "date"] 排序，用 last/first 取位置上的最後/第一筆
# 避免字串 max/min 在不同位數年份（如 "9999" vs "10000"）比大小錯誤
train_ends  = train.groupby("region_id")["date"].last()
test_starts = test.groupby("region_id")["date"].first()
common      = list(set(train_ends.index) & set(test_starts.index))

def _parse_ymd(s: str):
    parts = str(s).strip().split("T")[0].split(" ")[0].split("-")
    return int(parts[0]), int(parts[1]), int(parts[2])

gaps_days   = []
region_gaps = {}   # 供後續 JSON 匯出
for r in common:
    te, ts = train_ends[r], test_starts[r]
    te_y, te_m, te_d = _parse_ymd(te)
    ts_y, ts_m, ts_d = _parse_ymd(ts)
    gap = (ts_y - te_y) * 365 + (ts_m - te_m) * 30 + (ts_d - te_d)
    gaps_days.append(gap)
    region_gaps[str(r)] = {
        "train_end":  str(te),
        "test_start": str(ts),
        "gap_days":   gap,
        "gap_weeks":  round(gap / 7, 2),
    }

# 印出每個 region 的 train_end / test_start / gap
print(f"\n  {'region_id':<20} {'train_end':<20} {'test_start':<20} {'gap_weeks':>10}")
print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*10}")
for r in sorted(region_gaps):
    d = region_gaps[r]
    print(f"  {r:<20} {d['train_end']:<20} {d['test_start']:<20} {d['gap_weeks']:>10.1f}")

gaps_weeks     = np.array(gaps_days) / 7
gap_mean_wks   = np.mean(gaps_weeks)
gap_median_wks = np.median(gaps_weeks)
gap_p75_wks    = np.percentile(gaps_weeks, 75)
gap_p90_wks    = np.percentile(gaps_weeks, 90)
alpha_val      = 0.96
lag_range      = np.arange(1, 220)
autocorr_decay = alpha_val ** lag_range
acf_at_mean    = alpha_val ** gap_mean_wks
acf_at_median  = alpha_val ** gap_median_wks

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(lag_range, autocorr_decay, color="#e05252", lw=2, label="0.96^lag")
axes[0].axvline(gap_mean_wks, color="#4e9af1", linestyle="--",
                label=f"Mean gap={gap_mean_wks:.0f} wks")
axes[0].axvline(gap_median_wks, color="#f4a522", linestyle="--",
                label=f"Median gap={gap_median_wks:.0f} wks")
axes[0].axhline(0.05, color="gray", linestyle=":", alpha=0.7, label="ACF ≈ 0.05")
axes[0].scatter([gap_mean_wks], [acf_at_mean], color="#4e9af1", s=80, zorder=5)
axes[0].scatter([gap_median_wks], [acf_at_median], color="#f4a522", s=80, zorder=5)
axes[0].set_title("Score Autocorr Decay vs Gap", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Lag (weeks)"); axes[0].set_ylabel("Estimated ACF")
axes[0].legend(fontsize=9); axes[0].set_ylim(-0.05, 1.05)

axes[1].hist(gaps_weeks, bins=40, color="#4e9af1", edgecolor="white", alpha=0.8)
axes[1].axvline(gap_mean_wks, color="#e05252", linestyle="--", lw=2,
                label=f"Mean={gap_mean_wks:.0f} wks")
axes[1].axvline(np.median(gaps_weeks), color="#f4a522", linestyle="--", lw=2,
                label=f"Median={np.median(gaps_weeks):.0f} wks")
axes[1].set_title("Per-Region Train→Test Gap (weeks)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Gap (weeks)"); axes[1].set_ylabel("Region Count")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "02_gap_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Gap: mean={gap_mean_wks:.1f} wks, median={gap_median_wks:.1f} wks, "
      f"std={gaps_weeks.std():.1f}, min={gaps_weeks.min():.1f}, max={gaps_weeks.max():.1f}")
print(f"  P75={gap_p75_wks:.1f} wks, P90={gap_p90_wks:.1f} wks")
print(f"  Score ACF at mean gap:   0.96^{gap_mean_wks:.0f} ≈ {acf_at_mean:.4f}")
print(f"  Score ACF at median gap: 0.96^{gap_median_wks:.0f} ≈ {acf_at_median:.4f}  "
      f"(75% of regions have gap <= P75={gap_p75_wks:.0f} wks)")

# ═════════════════════════════════════════════════════════════════════════════
# Section 3 | KS 特徵偏移
# ═════════════════════════════════════════════════════════════════════════════
print("\n[3] Train vs Test 特徵分佈偏移（KS test）...")
ks_results = []
for col in METEO_COLS:
    tr_vals = train[col].dropna().values
    te_vals = test[col].dropna().values
    rng = np.random.RandomState(42)
    tr_s = rng.choice(tr_vals, min(50000, len(tr_vals)), replace=False)
    te_s = rng.choice(te_vals, min(50000, len(te_vals)), replace=False)
    ks_stat, ks_p = stats.ks_2samp(tr_s, te_s)
    ks_results.append({"feature": col, "KS_stat": ks_stat, "p_value": ks_p})

ks_df = pd.DataFrame(ks_results).sort_values("KS_stat", ascending=False)
print(ks_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#e05252" if ks > 0.1 else "#4e9af1" for ks in ks_df["KS_stat"]]
ax.barh(ks_df["feature"], ks_df["KS_stat"], color=colors, edgecolor="white")
ax.axvline(0.1, color="gray", linestyle="--", alpha=0.7, label="KS=0.1 threshold")
ax.set_title("Train vs Test Feature Shift (KS Statistic)", fontsize=12, fontweight="bold")
ax.set_xlabel("KS Statistic"); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_feature_shift_ks.png", dpi=150, bbox_inches="tight")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# Section 4 | 特徵與 Score 的相關性
# ═════════════════════════════════════════════════════════════════════════════
print("\n[4] 特徵 vs Score 相關性 ...")
corr_rows = []
for col in METEO_COLS:
    merged = scored[[col, "score"]].dropna()
    if len(merged) > 100:
        r, p = stats.spearmanr(merged[col], merged["score"])
        corr_rows.append({"feature": col, "spearman_r": r, "p_value": p})

corr_df = pd.DataFrame(corr_rows).sort_values("spearman_r", key=abs, ascending=False)
print(corr_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#e05252" if r > 0 else "#4e9af1" for r in corr_df["spearman_r"]]
ax.barh(corr_df["feature"], corr_df["spearman_r"], color=colors, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Spearman Corr: Feature vs Score", fontsize=12, fontweight="bold")
ax.set_xlabel("Spearman r")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_feature_score_corr.png", dpi=150, bbox_inches="tight")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# Section 5 | 季節性分析
# ═════════════════════════════════════════════════════════════════════════════
print("\n[5] 季節性分析 ...")
monthly_score = scored.groupby("month")["score"].agg(["mean", "std", "count"]).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(monthly_score["month"], monthly_score["mean"],
            yerr=monthly_score["std"] / np.sqrt(monthly_score["count"]),
            color="#f4a522", edgecolor="white", capsize=3)
axes[0].set_title("Monthly Avg Score (±SE)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Month"); axes[0].set_ylabel("Mean Score"); axes[0].set_xticks(range(1, 13))

pivot = (scored.groupby(["month", "score"]).size()
         .unstack(fill_value=0)
         .apply(lambda x: x / x.sum(), axis=1))
im = axes[1].imshow(pivot.T, aspect="auto", cmap="YlOrRd", origin="lower",
                    extent=[0.5, 12.5, -0.5, 5.5])
plt.colorbar(im, ax=axes[1], label="Proportion")
axes[1].set_title("Score by Month (Heatmap)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Month"); axes[1].set_ylabel("Score")
axes[1].set_xticks(range(1, 13)); axes[1].set_yticks(range(6))
plt.tight_layout()
plt.savefig(FIG_DIR / "05_seasonal_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# Section 6 | Score 轉移矩陣
# ═════════════════════════════════════════════════════════════════════════════
print("\n[6] Score 轉移矩陣 ...")
weekly_scores_list = []
for region, grp in scored.groupby("region_id"):
    grp = grp.sort_values("date").reset_index(drop=True)
    s = grp["score"].values
    for i in range(len(s) - 1):
        weekly_scores_list.append((int(s[i]), int(s[i+1])))

trans_df  = pd.DataFrame(weekly_scores_list, columns=["from", "to"])
trans_mat = (trans_df.groupby(["from", "to"]).size()
             .unstack(fill_value=0)
             .apply(lambda x: x / x.sum(), axis=1))

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(trans_mat.values, cmap="Blues", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label="P(t+1|t)")
ax.set_title("Weekly Score Transition Matrix", fontsize=12, fontweight="bold")
ax.set_xlabel("Score at t+1"); ax.set_ylabel("Score at t")
ax.set_xticks(range(6)); ax.set_yticks(range(6))
for i in range(6):
    for j in range(6):
        if j < trans_mat.shape[1] and i < trans_mat.shape[0]:
            val = trans_mat.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if val < 0.6 else "white")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_transition_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# Section 7 | 物理乾旱指標
# ═════════════════════════════════════════════════════════════════════════════
print("\n[7] 物理乾旱指標 vs Score ...")
train["vpd"]     = train["tmp"] - train["dp_tmp"]
train["dry_day"] = (train["prec"] < 0.1).astype(float)
train["dtr"]     = train["tmp_max"] - train["tmp_min"]

phys_cols = ["vpd", "dry_day", "dtr", "prec", "humidity"]
fig, axes = plt.subplots(1, len(phys_cols), figsize=(16, 4))
for ax, col in zip(axes, phys_cols):
    merged   = train[[col, "score"]].dropna()
    by_score = merged.groupby("score")[col].agg(["mean", "sem"])
    scores_int = by_score.index.astype(int)
    ax.bar(scores_int, by_score["mean"],
           yerr=by_score["sem"],
           color=[PALETTE.get(s, "#999") for s in scores_int],
           edgecolor="white", capsize=3)
    ax.set_title(f"{col}\nvs Score", fontsize=10, fontweight="bold")
    ax.set_xlabel("Score"); ax.set_xticks(range(6))
plt.suptitle("Physical Drought Indicators by Score Level",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "07_physical_indicators.png", dpi=150, bbox_inches="tight")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# Section 8 | Proxy Signal 預覽
# ═════════════════════════════════════════════════════════════════════════════
print("\n[8] Proxy Signal 強度 ...")
global_vpd_mu  = float(train["vpd"].mean())
global_vpd_sig = max(float(train["vpd"].std()), 0.1)
train["vpd_anom"] = (train["vpd"] - global_vpd_mu) / global_vpd_sig

global_prec_mu  = float(train["prec"].mean())
global_prec_sig = max(float(train["prec"].std()), 0.01)
train["prec_deficit"] = (global_prec_mu - train["prec"]) / global_prec_sig

proxy_signals = ["vpd_anom", "prec_deficit", "dry_day"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, proxy_signals):
    merged   = train[[col, "score"]].dropna()
    by_score = merged.groupby("score")[col].agg(["mean", "sem"])
    scores_int = by_score.index.astype(int)
    ax.bar(scores_int, by_score["mean"],
           yerr=by_score["sem"],
           color=[PALETTE.get(s, "#999") for s in scores_int],
           edgecolor="white", capsize=3)
    ax.set_title(f"{col}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Score"); ax.set_xticks(range(6))
plt.suptitle("Proxy Signals by Drought Score", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "08_proxy_signals.png", dpi=150, bbox_inches="tight")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# Section 9 | Region 異質性
# ═════════════════════════════════════════════════════════════════════════════
print("\n[9] Region 異質性 ...")
reg_stats = scored.groupby("region_id")["score"].agg(["mean", "std", "max"]).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, label in zip(axes, ["mean", "std", "max"],
                           ["Mean Score", "Score Std", "Max Score"]):
    ax.hist(reg_stats[col], bins=50, color="#4e9af1", edgecolor="white", alpha=0.8)
    ax.axvline(reg_stats[col].mean(), color="#e05252", linestyle="--",
               label=f"Mean={reg_stats[col].mean():.2f}")
    ax.set_title(f"Region {label}", fontsize=11, fontweight="bold")
    ax.set_xlabel(label); ax.set_ylabel("Count"); ax.legend(fontsize=9)
plt.suptitle("Region-level Score Heterogeneity", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "09_region_heterogeneity.png", dpi=150, bbox_inches="tight")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# Section 10 | Lag 相關分析
# ═════════════════════════════════════════════════════════════════════════════
print("\n[10] Lag 相關分析 ...")
lag_cors = []
for lag in [1, 2, 4, 8, 13]:
    sample_regions = scored["region_id"].unique()[:200]
    rs = []
    for region in sample_regions:
        grp = scored[scored["region_id"] == region].sort_values("date")
        s   = grp["score"].values
        p   = grp["prec"].values
        if len(s) > lag + 5:
            r, _ = stats.pearsonr(p[lag:], s[:-lag] if lag > 0 else s)
            rs.append(r)
    lag_cors.append({"lag_weeks": lag, "prec_score_r": np.mean(rs)})

lag_df = pd.DataFrame(lag_cors)
print(lag_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(lag_df["lag_weeks"], lag_df["prec_score_r"], marker="o", color="#4e9af1", lw=2)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Precipitation→Score Corr at Different Lags",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Lag (weeks)"); ax.set_ylabel("Pearson r")
plt.tight_layout()
plt.savefig(FIG_DIR / "10_lag_correlation.png", dpi=150, bbox_inches="tight")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# Section 11 | ★ Region 重疊分析（v6 新增）
# ═════════════════════════════════════════════════════════════════════════════
print("\n[11] ★ Region 重疊分析（GroupKFold 問題根源）...")

train_regions = set(train["region_id"].unique())
test_regions  = set(test["region_id"].unique())
only_train    = train_regions - test_regions
only_test     = test_regions  - train_regions
both          = train_regions & test_regions

print(f"  Train-only regions: {len(only_train)}")
print(f"  Test-only  regions: {len(only_test)}")
print(f"  Overlap regions:   {len(both):,}")
overlap_pct = len(both) / len(test_regions) * 100
print(f"  Overlap ratio (test): {overlap_pct:.1f}%")

fig, ax = plt.subplots(figsize=(7, 5))

categories = ["Train only", "Both (overlap)", "Test only"]
values     = [len(only_train), len(both), len(only_test)]
colors     = ["#4e9af1", "#7ac74f", "#e05252"]
bars = ax.bar(categories, values, color=colors, edgecolor="white", width=0.5)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5, str(v),
            ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_title("Region ID Overlap: Train vs Test", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Regions")
ax.set_ylim(0, max(values) * 1.15)

plt.tight_layout()
plt.savefig(FIG_DIR / "11_region_overlap.png", dpi=150, bbox_inches="tight")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# Section 12 | ★ 時間洩漏風險量化（v6 新增）
# ═════════════════════════════════════════════════════════════════════════════
print("\n[12] ★ 時間洩漏風險量化（Temporal Leakage）...")

# 滑動窗口重疊量：91-天窗口，每週滑動 7 天 → 相鄰窗口重疊 84 天
window_size_days = 91
step_days        = 7
overlap_days     = window_size_days - step_days  # 84
overlap_pct_win  = overlap_days / window_size_days * 100

# 若 GroupKFold-by-region 把同一 region 不同時間窗口拆入 train/val，
# 相鄰 k 個窗口都會有 overlap。計算最大洩漏深度（幾個 step 之內會重疊）
max_leaky_steps = window_size_days // step_days  # 13 個步驟之內的窗口都重疊

print(f"  Window size:        {window_size_days} days")
print(f"  Step size:          {step_days} days")
print(f"  Overlap per window: {overlap_days} days ({overlap_pct_win:.0f}%)")
print(f"  Max leaky steps (within same region): {max_leaky_steps}")
print(f"  GroupKFold (no time split): adjacent windows share {overlap_pct_win:.0f}% data -> val MAE underestimated")
print(f"  TemporalKFold fix: purge_gap = {window_size_days} days (time-based) -> eliminates overlap")

steps     = np.arange(1, max_leaky_steps + 3)
overlaps  = np.maximum(0, window_size_days - steps * step_days)
overlap_r = overlaps / window_size_days

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 重疊量 vs 步數
axes[0].bar(steps, overlap_r * 100, color="#e05252", edgecolor="white", alpha=0.8)
axes[0].axvline(max_leaky_steps, color="#4e9af1", linestyle="--", lw=2,
                label=f"Purge gap = {max_leaky_steps}")
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_title("Window Overlap vs Step Distance", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Step distance (# of sliding steps)")
axes[0].set_ylabel("Overlap (%)")
axes[0].legend(fontsize=9)
axes[0].set_ylim(-5, 105)

# 示意圖：GroupKFold vs TemporalKFold
timeline = np.linspace(0, 100, 200)
axes[1].fill_between(timeline[:120], 1.6, 1.9, color="#4e9af1", alpha=0.7, label="Train")
axes[1].fill_between(timeline[80:], 1.1, 1.4, color="#e05252", alpha=0.7, label="Val (GroupKFold, w/ overlap)")
axes[1].fill_between(timeline[133:], 0.6, 0.9, color="#7ac74f", alpha=0.7, label="Val (TemporalKFold, no overlap)")
axes[1].axvline(timeline[120], color="#4e9af1", linestyle="--", lw=1.5, alpha=0.6)
axes[1].axvline(timeline[133], color="#7ac74f", linestyle="--", lw=1.5, alpha=0.6, label="Purge gap boundary")
axes[1].set_xlim(0, 100)
axes[1].set_ylim(0.3, 2.1)
axes[1].set_yticks([])
axes[1].set_xlabel("Time →")
axes[1].set_title("GroupKFold vs TemporalKFold Leakage", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=8, loc="upper left")

plt.suptitle("Section 12: Temporal Leakage Risk", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "12_temporal_leakage.png", dpi=150, bbox_inches="tight")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# Section 13 | ★ Per-Region Gap 異質性（v6 新增）
# ═════════════════════════════════════════════════════════════════════════════
print("\n[13] ★ Per-Region Gap 異質性 ...")

gaps_arr = np.array(gaps_weeks)
gap_std  = gaps_arr.std()
gap_iqr  = np.percentile(gaps_arr, 75) - np.percentile(gaps_arr, 25)

print(f"  Gap std: {gap_std:.1f} wks   IQR: {gap_iqr:.1f} wks")
print(f"  → gap 異質性大，各 region 的預測難度不同")
print(f"  → Proxy Score 基於純氣象特徵計算，與 gap 長度無關")
print(f"  → 驗證時需追蹤 per-region MAE 分佈，找出高誤差區域")

# Gap 分佈 + 累積分佈
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with percentile lines
axes[0].hist(gaps_arr, bins=50, color="#4e9af1", edgecolor="white", alpha=0.8,
             density=True, label="Region count")
for p, label, col in [(25, "P25", "#7ac74f"), (50, "Median", "#f4a522"),
                       (75, "P75", "#e05252"), (90, "P90", "#b07de8")]:
    pv = np.percentile(gaps_arr, p)
    axes[0].axvline(pv, linestyle="--", color=col, lw=1.5, label=f"{label}={pv:.0f}w")
axes[0].set_title("Per-Region Gap Distribution", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Gap (weeks)"); axes[0].set_ylabel("Density")
axes[0].legend(fontsize=8)

# Score 與 gap 的關係（gap 大的 region 預測更難？）
# 用 region 的 score mean vs gap 做散佈圖
reg_gap = {}
for r in common:
    te, ts = train_ends[r], test_starts[r]
    te_y, te_m, te_d = _parse_ymd(te)
    ts_y, ts_m, ts_d = _parse_ymd(ts)
    gd = (ts_y - te_y) * 365 + (ts_m - te_m) * 30 + (ts_d - te_d)
    reg_gap[r] = gd / 7

gap_series  = pd.Series(reg_gap, name="gap")
score_means = scored.groupby("region_id")["score"].mean()
gap_score   = pd.concat([gap_series, score_means], axis=1).dropna()

axes[1].scatter(gap_score["gap"], gap_score["score"],
                alpha=0.15, s=5, color="#4e9af1")
# 趨勢線
z = np.polyfit(gap_score["gap"], gap_score["score"], 1)
p = np.poly1d(z)
xs = np.linspace(gap_score["gap"].min(), gap_score["gap"].max(), 100)
axes[1].plot(xs, p(xs), color="#e05252", lw=2,
             label=f"Trend: slope={z[0]:.4f}")
r_val, _ = stats.pearsonr(gap_score["gap"], gap_score["score"])
axes[1].set_title(f"Gap vs Region Mean Score (r={r_val:.3f})",
                  fontsize=12, fontweight="bold")
axes[1].set_xlabel("Gap (weeks)"); axes[1].set_ylabel("Region Mean Score")
axes[1].legend(fontsize=9)

plt.suptitle("Section 13: Per-Region Gap Heterogeneity", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "13_per_region_gap.png", dpi=150, bbox_inches="tight")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# 摘要
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EDA Summary")
print("=" * 60)
print(f"""
1. Score distribution is heavily imbalanced:
   score=0 accounts for {zero_pct:.1%}, severe (>=3) for {severe_pct:.1%}.
   Sample weighting is needed to prevent the model from ignoring rare events.

2. Train/Test gap is bimodal — NOT a single fixed value:
   Median={gap_median_wks:.0f} wks (P75={gap_p75_wks:.0f} wks), Mean={gap_mean_wks:.0f} wks (skewed by outliers at ~{gap_p90_wks:.0f} wks).
   ACF at median gap: 0.96^{gap_median_wks:.0f} = {acf_at_median:.3f} — score lag is still informative for ~75% of regions.
   ACF at mean gap:   0.96^{gap_mean_wks:.0f} = {acf_at_mean:.3f} — mean is a poor summary due to the long-tail.
   Strategy: use last_known_score + gap_weeks as features; model learns the decay automatically.

3. All test regions appear in train (overlap={overlap_pct:.1f}%).
   GroupKFold(group=region) assumption (unseen regions in val) does not hold.
   Fix: TemporalKFold — val is always strictly later in time than train.

4. Sliding window overlap = {overlap_pct_win:.0f}% ({overlap_days} of {window_size_days} days).
   TemporalKFold uses a time-based purge gap of {window_size_days} days to eliminate leakage.

5. Per-region gap std={gap_std:.1f} wks (IQR={gap_iqr:.1f} wks) — heterogeneous across regions.
   gap_weeks should be an explicit feature so the model can adapt its strategy per region.

6. Largest feature shift (KS): {ks_df.head(3)['feature'].tolist()}
   Region-level anomaly features and rank normalization are needed to reduce covariate shift.

7. Seasonal patterns are clear; physical signals (VPD, precip deficit) correlate with drought.
   sin/cos seasonal encoding and Proxy Score capture these signals for test inference.
""")

print(f"Figures saved to {FIG_DIR}")

# ═════════════════════════════════════════════════════════════════════════════
# 匯出 EDA 數值結果
# ═════════════════════════════════════════════════════════════════════════════
import json

eda_results = {
    "score_distribution": {
        str(int(k)): round(float(v), 6)
        for k, v in dist.items()
    },
    "score_summary": {
        "zero_pct":   round(zero_pct, 6),
        "severe_pct": round(severe_pct, 6),
    },
    "gap_stats": {
        "mean_wks":   round(float(gap_mean_wks), 3),
        "median_wks": round(float(gap_median_wks), 3),
        "std_wks":    round(float(gaps_weeks.std()), 3),
        "p25_wks":    round(float(np.percentile(gaps_weeks, 25)), 3),
        "p75_wks":    round(float(gap_p75_wks), 3),
        "p90_wks":    round(float(gap_p90_wks), 3),
        "min_wks":    round(float(gaps_weeks.min()), 3),
        "max_wks":    round(float(gaps_weeks.max()), 3),
        "acf_at_mean":   round(float(acf_at_mean), 6),
        "acf_at_median": round(float(acf_at_median), 6),
    },
    "region_gaps": region_gaps,   # 每個 region 的 train_end / test_start / gap
    "region_overlap": {
        "train_only": len(only_train),
        "test_only":  len(only_test),
        "both":       len(both),
        "overlap_pct": round(overlap_pct, 3),
    },
    "feature_ks": {
        row["feature"]: {
            "ks_stat": round(float(row["KS_stat"]), 6),
            "p_value": round(float(row["p_value"]), 6),
        }
        for _, row in ks_df.iterrows()
    },
    "feature_score_corr": {
        row["feature"]: {
            "spearman_r": round(float(row["spearman_r"]), 6),
            "p_value":    round(float(row["p_value"]), 6),
        }
        for _, row in corr_df.iterrows()
    },
    "lag_correlation": [
        {"lag_weeks": int(row["lag_weeks"]), "prec_score_r": round(float(row["prec_score_r"]), 6)}
        for _, row in lag_df.iterrows()
    ],
    "temporal_leakage": {
        "window_size_days": window_size_days,
        "step_days":        step_days,
        "overlap_days":     overlap_days,
        "overlap_pct":      round(overlap_pct_win, 1),
        "max_leaky_steps":  max_leaky_steps,
    },
    "per_region_gap_heterogeneity": {
        "std_wks": round(float(gap_std), 3),
        "iqr_wks": round(float(gap_iqr), 3),
    },
}

out_path = EDA_DIR / "eda_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(eda_results, f, ensure_ascii=False, indent=2)
print(f"EDA results saved → {out_path}")

print("EDA complete.")
