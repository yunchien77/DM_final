"""Calibrate the hot-holdout selection criterion against the two Kaggle LB points.

LB targets: zero-MAE = 1.2088 (test true-mean), region_mean-MAE = 0.9364, and
region_mean must BEAT zero (the test is a region-wide/systematic drought regime).
Tests drought-state selection: precip-deficit, and "hot AND dry" combinations.
"""
import sys
sys.path.insert(0, "/mnt/1stHDD/juiyun/DMFP")
import numpy as np
import pandas as pd
from dmfp2 import config as C, weekly as W, anchors as A, validation as V

LB_ZERO, LB_RM = 1.2088, 0.9364

tr_w, te_w = W.get_weekly()
tra = A.build_train_anchors(tr_w)
tea = A.build_test_anchors(te_w)
tra = V.attach_region_meta(tra, tea)
SLACK = C.VALIDATION.calendar_slack_weeks

# --- per-region standardized features for the selection score ---
g = tra.groupby("region_idx")
# precip deficit: region-mean prec_sum minus this week's prec_sum (higher = drier)
tra["_prec_def"] = g["prec_sum"].transform("mean") - tra["prec_sum"]
def zreg(col):
    m = g[col].transform("mean"); s = g[col].transform("std").replace(0, 1.0)
    return (tra[col] - m) / s
tra["_z_def"] = (tra["_prec_def"] - g["_prec_def"].transform("mean")) / g["_prec_def"].transform("std").replace(0, 1.0)
tra["_z_tmax"] = zreg("tmp_max_mean")
in_season = (tra["woy_dist"] <= SLACK).to_numpy()


def metrics(val_df):
    z = V.baseline_preds("zero", tra, val_df)
    rm = V.baseline_preds("region_mean", tra, val_df)
    dz = V.evaluate(val_df, z); drm = V.evaluate(val_df, rm)
    zmae, rmae = dz["macro_mae"], drm["macro_mae"]
    dist = ((zmae - LB_ZERO) ** 2 + (rmae - LB_RM) ** 2) ** 0.5
    return dict(n=len(val_df), zero=round(zmae, 4), rm=round(rmae, 4),
                rm_beats=rmae < zmae, rm_minus_z=round(rmae - zmae, 4),
                dist=round(dist, 4), frac0=round(dz["frac_true_zero"], 3),
                true_mean=round(dz["true_mean"], 4))


def select_argmax(score_col):
    a = tra.copy()
    a["_s"] = np.where(in_season, a[score_col], -1e18)
    rows = a.groupby("region_idx")["_s"].idxmax().to_numpy()
    return tra.loc[rows]


print(f"{'strategy':<34}{'n':>6}{'zero':>8}{'rm':>8}{'rmbeats':>9}{'dist':>8}{'frac0':>8}")
results = {}
# precip-deficit only
tra["_score_def"] = tra["_z_def"]
v = select_argmax("_score_def"); results["precip_deficit"] = metrics(v)
# hot AND dry, sweep temp weight
for w in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    tra["_score_w"] = tra["_z_def"] + w * tra["_z_tmax"]
    v = select_argmax("_score_w"); results[f"drought_stress_w{w}"] = metrics(v)
# temp only (reference) + current temp-match style would need test temp; skip
tra["_score_t"] = tra["_z_tmax"]
v = select_argmax("_score_t"); results["temp_only"] = metrics(v)

for name, m in results.items():
    print(f"{name:<34}{m['n']:>6}{m['zero']:>8.4f}{m['rm']:>8.4f}{str(m['rm_beats']):>9}{m['dist']:>8.4f}{m['frac0']:>8.3f}")

best = min((m for m in results.values() if m["rm_beats"]),
           key=lambda m: m["dist"], default=None)
best_name = [n for n, m in results.items() if m is best][0] if best else None
print(f"\nBEST (rm_beats_zero & min dist): {best_name} -> {best}")
print(f"LB targets: zero={LB_ZERO}, rm={LB_RM}")
