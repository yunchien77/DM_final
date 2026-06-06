"""Physics-based SPEI drought model — a mathematically different, decorrelated track.

SPEI = standardized (precip - PET). PET from temperature (Hargreaves, no-Ra form;
the per-region standardization absorbs the seasonal radiation cycle). Standardized
per region using TRAIN statistics, so the +6C test (higher PET -> lower water balance)
shows up as more-negative SPEI -> more drought BY CONSTRUCTION (no learned meteo->score
mapping that could fail to extrapolate). A per-horizon isotonic map SPEI->score adds the
only data-dependent step (monotone, so physically constrained) and captures persistence+decay.

Goal: a model decorrelated from the LGBM/blend (corr<~0.6) that's still decent, so it can
push the 0.8027 blend lower. Evaluated by profile + corr vs the 0.8027 blend + MAE vs the
0.8316 reference (pseudo-truth). Submission written for Kaggle.
"""
import sys
sys.path.insert(0, "/mnt/1stHDD/juiyun/DMFP")
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from dmfp2 import config as C, weekly as W, anchors as A

SCALES = [12, 26, 52]
H = list(range(1, 6))
RAW = C.DataConfig(log1p_cols=(), sqrt_cols=(), winsorize=False)   # raw precip (mm) for water balance


def compute_D(w, pet_method):
    w = w.sort_values(["region_idx", "week_pos"], kind="stable").reset_index(drop=True)
    tmean = w["tmp_mean"].to_numpy(np.float64)
    trange = np.clip((w["tmp_max_mean"] - w["tmp_min_mean"]).to_numpy(np.float64), 0, None)
    vpd = np.clip((w["tmp_mean"] - w["wb_tmp_mean"]).to_numpy(np.float64), 0, None)
    if pet_method == "hargreaves":
        pet = (tmean + 17.8) * np.sqrt(trange) * 7.0       # weekly PET proxy (no Ra; abs by std)
    elif pet_method == "vpd":
        pet = vpd * 7.0
    else:
        pet = np.zeros(len(w))                              # SPI (precip only)
    w["_D"] = w["prec_sum"].to_numpy(np.float64) - pet
    return w


def train_spei(trw):
    """Per-anchor SPEI on train + per-region cum-D mean/std per scale (for test standardization)."""
    trw = trw.sort_values(["region_idx", "week_pos"], kind="stable").reset_index(drop=True)
    g = trw.groupby("region_idx", sort=False)["_D"]
    region_stats = {}
    zcols = []
    for sc in SCALES:
        cum = g.rolling(sc, min_periods=max(4, sc // 2)).sum().reset_index(level=0, drop=True)
        trw[f"_cum{sc}"] = cum.to_numpy()
        m = trw.groupby("region_idx")[f"_cum{sc}"].transform("mean")
        s = trw.groupby("region_idx")[f"_cum{sc}"].transform("std").replace(0, 1.0)
        trw[f"_z{sc}"] = ((trw[f"_cum{sc}"] - m) / s).to_numpy()
        region_stats[sc] = trw.groupby("region_idx")[f"_cum{sc}"].agg(["mean", "std"])
        zcols.append(f"_z{sc}")
    trw["spei"] = trw[zcols].mean(axis=1)   # avg multi-scale; negative = drought
    return trw, region_stats


def test_spei(trw, tew, region_stats):
    trg = {r: g["_D"].to_numpy(np.float64) for r, g in
           trw.sort_values(["region_idx", "week_pos"]).groupby("region_idx", sort=False)}
    teg = {r: g["_D"].to_numpy(np.float64) for r, g in
           tew.sort_values(["region_idx", "week_pos"]).groupby("region_idx", sort=False)}
    out = {}
    for r, td in teg.items():
        full = np.concatenate([trg.get(r, np.zeros(0)), td])
        zs = []
        for sc in SCALES:
            seg = full[-sc:] if len(full) >= sc else full
            cum = seg.sum()
            st = region_stats[sc].loc[r] if r in region_stats[sc].index else None
            m = float(st["mean"]) if st is not None else 0.0
            s = float(st["std"]) if st is not None and st["std"] else 1.0
            zs.append((cum - m) / (s if s else 1.0))
        out[r] = float(np.mean(zs))
    return out


def run(pet_method):
    tr_w, te_w = W.get_weekly(RAW)
    trw = compute_D(tr_w, pet_method)
    tew = compute_D(te_w, pet_method)
    trw, region_stats = train_spei(trw)
    # train anchors: attach SPEI + targets
    tra = A.build_train_anchors(trw)   # carries 'spei' (build keeps all weekly cols) + score + target_w1..5
    tea = A.build_test_anchors(tew)
    sp_test = test_spei(trw, tew, region_stats)
    tea["spei"] = tea["region_idx"].map(sp_test)

    # per-horizon isotonic map: SPEI -> target_wh (decreasing: lower SPEI = more drought)
    x_tr = tra["spei"].to_numpy(np.float64)
    preds = np.zeros((len(tea), 5), np.float32)
    x_te = tea["spei"].to_numpy(np.float64)
    glob_mean = float(np.nanmean(x_tr))
    x_tr = np.nan_to_num(x_tr, nan=glob_mean); x_te = np.nan_to_num(x_te, nan=glob_mean)
    for j, h in enumerate(H):
        y = tra[f"target_w{h}"].to_numpy(np.float64)
        m = ~np.isnan(y)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(x_tr[m], y[m])
        preds[:, j] = np.clip(iso.predict(x_te), 0, 5)
    return tea["region_id"].to_numpy(), preds


def profile(v, nm, extra=""):
    print(f"{nm:24s} mean={v.mean():.3f} wk1={v[:,0].mean():.3f} wk5={v[:,4].mean():.3f} "
          f"decay={v[:,0].mean()-v[:,4].mean():+.3f} frac<0.5={np.mean(v<0.5):.3f} {extra}")


def main():
    cols = [f"pred_week{j}" for j in range(1, 6)]
    B = pd.read_csv("/mnt/1stHDD/juiyun/DMFP/submission_blend_t065_u035.csv").sort_values("region_id").reset_index(drop=True)  # 0.8027
    R = pd.read_csv("/mnt/1stHDD/juiyun/DMFP/submission_teamv2_baseline.csv").sort_values("region_id").reset_index(drop=True)  # 0.8316 ref
    print("=== SPEI physics model variants ===")
    profile(B[cols].values, "blend 0.8027(target div)")
    profile(R[cols].values, "reference 0.8316")
    for pet in ["hargreaves", "vpd", "none"]:
        ids, preds = run(pet)
        sub = pd.DataFrame({"region_id": ids})
        for j in range(5):
            sub[f"pred_week{j+1}"] = preds[:, j]
        sub = sub.sort_values("region_id").reset_index(drop=True)
        # corr + MAE vs reference (pseudo-truth) + vs blend
        m = sub.merge(R, on="region_id", suffixes=("_s", "_r"))
        A_ = m[[f"pred_week{j}_s" for j in range(1,6)]].values.ravel()
        Rv = m[[f"pred_week{j}_r" for j in range(1,6)]].values.ravel()
        corr_ref = np.corrcoef(A_, Rv)[0,1]; mae_ref = np.abs(A_-Rv).mean()
        mb = sub.merge(B, on="region_id", suffixes=("_s","_b"))
        Bv = mb[[f"pred_week{j}_b" for j in range(1,6)]].values.ravel()
        As = mb[[f"pred_week{j}_s" for j in range(1,6)]].values.ravel()
        corr_blend = np.corrcoef(As, Bv)[0,1]
        profile(preds, f"SPEI[{pet}]", f"corr_blend={corr_blend:.3f} corr_ref={corr_ref:.3f} MAE_vs_ref={mae_ref:.3f}")
        sub.to_csv(f"/mnt/1stHDD/juiyun/DMFP/submission_spei_{pet}.csv", index=False)
    print("\nwrote submission_spei_{hargreaves,vpd,none}.csv")


if __name__ == "__main__":
    main()
