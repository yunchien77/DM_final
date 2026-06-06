"""Linear-on-physics model — the workflow's #1 'last jump' lever (odds 75).

Stacks the ONLY two mechanisms that have transferred to the +6C out-of-support test:
  (1) LINEAR map  -> extrapolates the shift (what made DLinear 0.846 work; trees clip, deep NNs mis-extrapolate)
  (2) SHIFT-INVARIANT inputs -> the +6C never enters as an out-of-support level (climatology-normalized).
Features (all per-region z-scored on TRAIN stats, so in-support on test): multi-scale SPI (precip),
SPEI (precip-PET), EDDI (evaporative demand=PET), cross-scale drought-trend diffs, region score-climatology
(targets the chronic-dry high-tail the residual analysis flagged), calendar. Per-horizon Ridge.

This is the DISCIPLINED version of spei_model.py (which failed standalone at 0.96 by mapping ONE averaged
index via isotonic) — here we fit linear weights over a RICH invariant feature set against the actual target.
Goal: good (~0.85) AND decorrelated (corr<~0.6 vs the LGBM cluster AND vs raw-input DLinear) -> blend < 0.7862.
"""
import sys
sys.path.insert(0, "/mnt/1stHDD/juiyun/DMFP")
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from dmfp2 import config as C, weekly as W, anchors as A

SCALES = [4, 12, 26, 52]
H = list(range(1, 6))
RAW = C.DataConfig(log1p_cols=(), sqrt_cols=(), winsorize=False)
SERIES = ["_P", "_D", "_PET"]   # SPI, SPEI, EDDI sources


def build_series(w):
    w = w.sort_values(["region_idx", "week_pos"], kind="stable").reset_index(drop=True)
    P = w["prec_sum"].to_numpy(np.float64)
    vpd = np.clip((w["tmp_mean"] - w["wb_tmp_mean"]).to_numpy(np.float64), 0, None)
    pet = vpd * 7.0
    w["_P"] = P; w["_PET"] = pet; w["_D"] = P - pet
    return w


def train_features(trw):
    """Per-anchor multi-scale z-features on train + per-region (col,scale) cum mean/std for test."""
    trw = trw.sort_values(["region_idx", "week_pos"], kind="stable").reset_index(drop=True)
    stats = {}
    fcols = []
    for col in SERIES:
        g = trw.groupby("region_idx", sort=False)[col]
        for s in SCALES:
            cum = g.rolling(s, min_periods=max(2, s // 2)).sum().reset_index(level=0, drop=True)
            cc = f"_cum_{col}_{s}"; trw[cc] = cum.to_numpy()
            m = trw.groupby("region_idx")[cc].transform("mean")
            sd = trw.groupby("region_idx")[cc].transform("std").replace(0, 1.0)
            zc = f"z{col}{s}"; trw[zc] = ((trw[cc] - m) / sd).to_numpy()
            stats[(col, s)] = trw.groupby("region_idx")[cc].agg(["mean", "std"])
            fcols.append(zc)
    # cross-scale drought-trend diffs (short minus long) for SPEI(_D) and SPI(_P)
    for col in ["_D", "_P"]:
        trw[f"ztrend{col}"] = trw[f"z{col}4"] - trw[f"z{col}52"]; fcols.append(f"ztrend{col}")
    # SHIFT-CARRYING region-baseline anomalies (recent 4wk mean - region TRAIN mean, NOT std-normalized):
    # on the +6C test these go large-positive -> StandardScaler maps out-of-range -> the LINEAR model
    # extrapolates UP (the DLinear mechanism), fixing the under-prediction of pure shift-invariant inputs.
    rbstats = {}
    for col in ["_PET", "_D", "_P"]:
        rec = trw.groupby("region_idx", sort=False)[col].rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
        regmean = trw.groupby("region_idx")[col].transform("mean")
        trw[f"rb{col}"] = (rec.to_numpy() - regmean.to_numpy())
        rbstats[col] = trw.groupby("region_idx")[col].mean()
        fcols.append(f"rb{col}")
    return trw, (stats, rbstats), fcols


def test_features(trw, tew, stats_pair, fcols):
    stats, rbstats = stats_pair
    trg = {r: g for r, g in trw.sort_values(["region_idx", "week_pos"]).groupby("region_idx", sort=False)}
    teg = {r: g for r, g in tew.sort_values(["region_idx", "week_pos"]).groupby("region_idx", sort=False)}
    rows = []
    for r, te in teg.items():
        full = {c: np.concatenate([trg[r][c].to_numpy(np.float64), te[c].to_numpy(np.float64)]) if r in trg
                else te[c].to_numpy(np.float64) for c in SERIES}
        feat = {"region_idx": r}
        for col in SERIES:
            for s in SCALES:
                seg = full[col][-s:] if len(full[col]) >= s else full[col]
                cum = seg.sum()
                st = stats[(col, s)].loc[r] if r in stats[(col, s)].index else None
                m = float(st["mean"]) if st is not None else 0.0
                sd = float(st["std"]) if st is not None and st["std"] else 1.0
                feat[f"z{col}{s}"] = (cum - m) / (sd if sd else 1.0)
        for col in ["_D", "_P"]:
            feat[f"ztrend{col}"] = feat[f"z{col}4"] - feat[f"z{col}52"]
        for col in ["_PET", "_D", "_P"]:
            rec = full[col][-4:].mean() if len(full[col]) >= 1 else 0.0
            rm = float(rbstats[col].loc[r]) if r in rbstats[col].index else 0.0
            feat[f"rb{col}"] = rec - rm
        rows.append(feat)
    return pd.DataFrame(rows)


def add_region_scalars(trw, tr_anchors, te_feat):
    """region score climatology (shift-invariant region identity) — targets chronic-dry high tail."""
    rs = trw.groupby("region_idx")["score"].agg(score_mean="mean", score_std="std",
                                                 zero_rate=lambda x: float((x == 0).mean())).reset_index()
    rs["score_std"] = rs["score_std"].fillna(0.0)
    tr_anchors = tr_anchors.merge(rs, on="region_idx", how="left")
    te_feat = te_feat.merge(rs, on="region_idx", how="left")
    return tr_anchors, te_feat, ["score_mean", "score_std", "zero_rate"]


def add_calendar(df):
    woy = df["woy"].to_numpy(np.float64) if "woy" in df.columns else np.zeros(len(df))
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.0); df["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    return df, ["woy_sin", "woy_cos"]


def main():
    tr_w, te_w = W.get_weekly(RAW)
    trw = build_series(tr_w); tew = build_series(te_w)
    trw, stats, zcols = train_features(trw)
    tra = A.build_train_anchors(trw)              # carries z-features + score + target_w1..5 + woy
    tea = A.build_test_anchors(tew)
    te_feat = test_features(trw, tew, stats, zcols)
    tea = tea[["region_id", "region_idx"] + (["woy"] if "woy" in tea.columns else [])].merge(te_feat, on="region_idx", how="left")

    tra, tea, rscols = add_region_scalars(trw, tra, tea)
    tra, calcols = add_calendar(tra); tea, _ = add_calendar(tea)
    feat_cols = zcols + rscols + calcols
    print(f"[features] {len(feat_cols)}: {feat_cols}")

    Xtr = tra[feat_cols].to_numpy(np.float64)
    Xte = tea[feat_cols].to_numpy(np.float64)
    Xtr = np.nan_to_num(Xtr, nan=0.0); Xte = np.nan_to_num(Xte, nan=0.0)
    preds = np.zeros((len(tea), 5), np.float32)
    for j, h in enumerate(H):
        y = tra[f"target_w{h}"].to_numpy(np.float64)
        m = ~np.isnan(y)
        mdl = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        mdl.fit(Xtr[m], y[m])
        preds[:, j] = np.clip(mdl.predict(Xte), 0, 5)

    cols = [f"pred_week{j}" for j in range(1, 6)]
    sub = tea[["region_id"]].copy()
    for j in range(5): sub[f"pred_week{j+1}"] = preds[:, j]
    sub = sub.sort_values("region_id").reset_index(drop=True)
    # diagnostics
    B = pd.read_csv("/mnt/1stHDD/juiyun/DMFP/submission_blend_ENS_354025.csv").sort_values("region_id").reset_index(drop=True)
    R = pd.read_csv("/mnt/1stHDD/juiyun/DMFP/submission_teamv2_baseline.csv").sort_values("region_id").reset_index(drop=True)
    D = pd.read_csv("/mnt/1stHDD/juiyun/DMFP/submission_dlinear.csv").sort_values("region_id").reset_index(drop=True)
    T = pd.read_csv("/mnt/1stHDD/juiyun/DMFP/submission_teammate_082.csv").sort_values("region_id").reset_index(drop=True)
    v = sub[cols].values
    cb = np.corrcoef(v.ravel(), B[cols].values.ravel())[0, 1]
    cd = np.corrcoef(v.ravel(), D[cols].values.ravel())[0, 1]
    ct = np.corrcoef(v.ravel(), T[cols].values.ravel())[0, 1]
    print(f"\nLINPHYS mean={v.mean():.3f} wk1={v[:,0].mean():.3f} wk5={v[:,4].mean():.3f} "
          f"decay={v[:,0].mean()-v[:,4].mean():+.3f} frac<0.5={np.mean(v<0.5):.3f}")
    print(f"  MAE_vs_ref={np.abs(v-R[cols].values).mean():.3f}  corr_vs_blend={cb:.3f} corr_vs_DLinear={cd:.3f} corr_vs_team={ct:.3f}")
    sub.to_csv("/mnt/1stHDD/juiyun/DMFP/submission_linphys.csv", index=False)
    print("wrote submission_linphys.csv")


if __name__ == "__main__":
    main()
