"""Orchestrator: assemble features → split → fit transforms → train → evaluate → submit.

Phase 1 baseline. Judge by the hot-holdout MAE + prediction-mean bias (NOT inner OOF).
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from . import config as C
from .features import get_features, feature_columns
from .anchors import build_train_anchors, build_test_anchors
from .validation import make_hot_holdout, in_season_es_mask, evaluate, log_val_lb
from .transforms import RegionScalars, SeasonalScoreClim
from .proxy import ProxyScore
from .model import Phase1Model, PerHorizonModel

KAGGLE_LB = {"zero": 1.2088, "region_mean": 0.9364}


def write_submission(region_ids, pred_matrix: np.ndarray, path=C.SUBMISSION_PATH):
    pred_matrix = np.clip(pred_matrix, 0, 5)
    df = pd.DataFrame({C.ID_COL: region_ids})
    for j, col in enumerate(C.SUB_COLS):
        df[col] = pred_matrix[:, j]
    df["_i"] = df[C.ID_COL].str.slice(1).astype(int)
    df = df.sort_values("_i").drop(columns="_i").reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"[submission] wrote {path}  ({len(df)} rows)")
    return df


def _assemble(anchors, rs, px, ssc, base_cols: list) -> pd.DataFrame:
    base = anchors[base_cols]
    return pd.concat([base, rs.transform(anchors), ssc.transform(anchors),
                      px.transform(anchors)], axis=1)


def run_train(data=C.DATA, features=C.FEATURES, val_cfg=C.VALIDATION,
              model_cfg=C.MODEL, submit=False, tag="phase1"):
    print(C.config_banner(features, data, val_cfg, C.TRAIN_SUBSET, model_cfg))
    tr_f, te_f = get_features(data, features)
    tr_anchors = build_train_anchors(tr_f, features)
    te_anchors = build_test_anchors(te_f)

    split = make_hot_holdout(tr_anchors, te_anchors, val_cfg)
    val_anchors = tr_anchors[split["val_mask"]].reset_index(drop=True)
    pool_anchors = tr_anchors[split["train_mask"]].reset_index(drop=True)
    print(f"\n[split] selection={split['selection']}  n_val={split['n_val']:,}  "
          f"n_pool={split['n_train']:,}  val_true_mean={split['val_true_mean']:.3f}")

    # transforms (climatology) fit on the FULL pool for max data
    rs = RegionScalars().fit(pool_anchors)
    ssc = SeasonalScoreClim().fit(pool_anchors)
    px = ProxyScore(features).fit(pool_anchors)
    base_cols = feature_columns(tr_anchors)
    if getattr(C, "LEAN_FEATURES", False):
        def _shift_carrying(c):
            return ("_ranom" in c) or c.startswith("prec_deficit") or c.startswith("heat_dd") or c.startswith("aridity")
        dropped = [c for c in base_cols if _shift_carrying(c)]
        base_cols = [c for c in base_cols if not _shift_carrying(c)]
        print(f"[lean] dropped {len(dropped)} shift-carrying features; kept {len(base_cols)} base cols")
    full_cols = base_cols + RegionScalars.COLS + SeasonalScoreClim.COLS + px.out_cols

    # carve a Kaggle-proxy in-season early-stopping set from the pool (per-horizon mode)
    es_mask = in_season_es_mask(pool_anchors, te_anchors, val_cfg)
    es_anchors = pool_anchors[es_mask].reset_index(drop=True)
    rest = pool_anchors[~es_mask]

    # LGBM trains on a per-region subsample of the rest (faster, less in-distribution overfit)
    if C.TRAIN_SAMPLE_PER_REGION:
        train_anchors = (rest.sample(frac=1.0, random_state=C.SEED)
                         .groupby("region_idx", sort=False)
                         .head(C.TRAIN_SAMPLE_PER_REGION).reset_index(drop=True))
    else:
        train_anchors = rest.reset_index(drop=True)
    print(f"[train] LGBM rows: {len(train_anchors):,} anchors  |  in-season ES set: {len(es_anchors):,}  "
          f"(from pool {len(pool_anchors):,})")

    X_train = _assemble(train_anchors, rs, px, ssc, base_cols)
    X_es = _assemble(es_anchors, rs, px, ssc, base_cols)
    X_val = _assemble(val_anchors, rs, px, ssc, base_cols)

    if model_cfg.horizon_sharing == "independent":
        model = PerHorizonModel(full_cols, model_cfg).fit(train_anchors, X_train, es_anchors, X_es)
        print(f"[model] mode=per-horizon  best_iters={model.best_iters_}  n_features={len(full_cols)}")
    else:
        model = Phase1Model(full_cols, model_cfg).fit(train_anchors, X_train)
        print(f"[model] mode=horizon_feature  best_iter={model.best_iter_}  n_features={len(full_cols)+1}")

    val_pred = model.predict(X_val)
    diag = evaluate(val_anchors, val_pred)

    print(f"\n=== HOT-HOLDOUT (the Kaggle twin) ===")
    print(f"  macro_MAE  = {diag['macro_mae']:.4f}   per-horizon {diag['per_horizon_mae']}")
    print(f"  pred_mean  = {diag['pred_mean']:.4f}   true_mean = {diag['true_mean']:.4f}   "
          f"mean_bias = {diag['mean_bias']:+.4f}")
    print(f"  MAE bucket = {diag['mae_by_bucket']}   per-horizon bias {diag['per_horizon_bias']}")
    print(f"  baselines on this holdout: zero=1.110 region_mean=1.062  |  Kaggle LB: zero=1.2088 region_mean=0.9364")
    print(f"\n[top features]")
    for name, imp in model.feature_importance(20):
        print(f"    {name:<28}{imp}")

    log_val_lb(C.feature_hash(features, data), tag, diag)
    # persist for predict
    with open(C.ARTIFACTS_DIR / f"model_{tag}.pkl", "wb") as f:
        pickle.dump({"model": model, "rs": rs, "px": px, "ssc": ssc, "base_cols": base_cols}, f)

    if submit:
        X_te = _assemble(te_anchors, rs, px, ssc, base_cols)
        te_pred = model.predict(X_te)
        out_path = C.ROOT / f"submission_{tag}.csv"   # uniquely named per run (no overwrite)
        write_submission(te_anchors[C.ID_COL].to_numpy(), te_pred, out_path)
        print(f"[submit] test pred mean = {te_pred.mean():.4f}  (val true_mean {diag['true_mean']:.3f}, "
              f"test true_mean from LB 1.2088)")
    return diag
