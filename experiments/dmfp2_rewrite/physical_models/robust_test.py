"""Test whether dropping out-of-support TEMPERATURE features recovers transferable signal.

Diagnosis: the model ties region_mean on Kaggle because temperature is +6C out-of-support;
the learned meteo->drought mapping breaks. Robust features = pressure / precip / humidity /
variability + region climatology + a precip SPI index. Compare feature subsets by their edge
over region_mean and prediction bias, on a leak-free (time+season selected) holdout AND the
precip-deficit (severity-matched) holdout.
"""
import sys
sys.path.insert(0, "/mnt/1stHDD/juiyun/DMFP")
import numpy as np
import pandas as pd
import lightgbm as lgb
from dmfp2 import config as C, anchors as A, validation as V, features as F
from dmfp2.transforms import RegionScalars

tr_f, te_f = F.get_features()
tra = A.build_train_anchors(tr_f)
tea = A.build_test_anchors(te_f)
feat_cols = F.feature_columns(tra)
tra = V.attach_region_meta(tra, tea)

# SPI-like: precip_sum standardized per (region, woy). (quick-test climatology uses all anchors)
gw = tra.groupby(["region_idx", "woy"])["prec_sum"]
tra["spi"] = ((tra["prec_sum"] - gw.transform("mean")) / gw.transform("std").replace(0, 1.0)).astype(np.float32)

TEMP = ("tmp", "tmp_max", "tmp_min", "tmp_range", "surf_tmp", "wb_tmp", "dp_tmp")
def is_temp(c): return any(c == v or c.startswith(v + "_") for v in TEMP)

subsets = {
    "all": feat_cols,
    "no_temp": [c for c in feat_cols if not is_temp(c)],
    "no_temp_keep_std": [c for c in feat_cols if (not is_temp(c)) or c.endswith("_std")],
    "no_temp+spi": [c for c in feat_cols if not is_temp(c)] + ["spi"],
}


def leakfree_split(a, slack=6, purge=13):
    a = a.reset_index(drop=True)
    insea = (a["woy_dist"] <= slack).to_numpy()
    a = a.assign(_p=np.where(insea, a["anchor_ord"].to_numpy(), -1))
    vr = a.groupby("region_idx")["_p"].idxmax().to_numpy()
    vm = np.zeros(len(a), bool); vm[vr] = True
    vo = a.loc[vm].set_index("region_idx")["anchor_ord"]
    near = np.abs(a["anchor_ord"].to_numpy() - a["region_idx"].map(vo).to_numpy()) < purge * 7
    return vm, (~vm) & (~near)


def make_Xy(anchors, Xdf):
    base = Xdf.to_numpy(np.float32)
    Xs, ys = [], []
    for fw in [1, 2, 3, 4, 5]:
        y = anchors[f"target_w{fw}"].to_numpy(np.float32); m = ~np.isnan(y)
        Xs.append(np.column_stack([base[m], np.full(int(m.sum()), fw, np.float32)])); ys.append(y[m])
    return np.concatenate(Xs), np.concatenate(ys)


def pred_mtx(model, Xdf):
    base = Xdf.to_numpy(np.float32); out = np.zeros((len(base), 5), np.float32)
    for j, fw in enumerate([1, 2, 3, 4, 5]):
        out[:, j] = model.predict(np.column_stack([base, np.full(len(base), fw, np.float32)]))
    return np.clip(out, 0, 5)


def run(name, split_fn):
    if split_fn == "leakfree":
        vm, tm = leakfree_split(tra)
    else:
        s = V.make_hot_holdout(tra, tea); vm, tm = s["val_mask"], s["train_mask"]
    val = tra[vm].reset_index(drop=True); pool = tra[tm].reset_index(drop=True)
    rs = RegionScalars().fit(pool)
    rs_val, rs_pool = rs.transform(val), rs.transform(pool)
    rm = V.baseline_preds("region_mean", pool, val)
    rm_mae = V.evaluate(val, rm)["macro_mae"]
    tm_mean = V.evaluate(val, np.zeros((len(val), 5)))["true_mean"]
    print(f"\n--- {name} holdout: n={len(val)} true_mean={tm_mean:.3f} region_mean_mae={rm_mae:.4f} ---")
    print(f"{'subset':<20}{'ncols':>6}{'model_mae':>11}{'edge_vs_rm':>11}{'bias':>9}")
    for sub, cols in subsets.items():
        Xpool = pd.concat([pool[cols].reset_index(drop=True), rs_pool.reset_index(drop=True)], axis=1)
        Xval = pd.concat([val[cols].reset_index(drop=True), rs_val.reset_index(drop=True)], axis=1)
        Xtr, ytr = make_Xy(pool, Xpool)
        m = lgb.LGBMRegressor(objective="regression_l1", n_estimators=400, learning_rate=0.05,
                              num_leaves=63, n_jobs=8, verbose=-1).fit(Xtr, ytr)
        d = V.evaluate(val, pred_mtx(m, Xval))
        print(f"{sub:<20}{len(cols)+4:>6}{d['macro_mae']:>11.4f}{rm_mae-d['macro_mae']:>+11.4f}{d['mean_bias']:>+9.4f}")


run("LEAK-FREE (time+season)", "leakfree")
run("PRECIP-DEFICIT (severity-matched)", "precipdef")
print("\nNote: edge_vs_rm = region_mean_mae - model_mae (higher=model better). On Kaggle the full model's "
      "edge is ~0. If 'no_temp' keeps a similar internal edge to 'all', the temp features were adding "
      "in-distribution-only (non-transferable) signal; dropping them should transfer better. Submit to confirm.")
