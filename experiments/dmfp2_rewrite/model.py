"""Per-horizon LightGBM via horizon-as-feature (fixed input window, varying target).

Each anchor is exploded into up to 5 training rows sharing the SAME feature vector,
differing only in `forecast_week` (1..5) and the target (score at anchor+fw). This is
the cheap form of horizon information-sharing without a neural net, and it matches the
test (one input window, predict +1..+5 from its end). severity_weight stays 0 by default
(target-severity weighting caused the documented +0.42 prediction-mean bias).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

from . import config as C

# Benign: we fit/predict on NumPy arrays built from an identical, ordered column list,
# so feature alignment is guaranteed; sklearn just can't verify names match.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _tqdm_callback(total: int, desc: str):
    """LightGBM callback driving a tqdm bar (live ETA + current eval MAE).

    Returns (callback, bar); close the bar after fit(). mininterval keeps tee'd logs sane.
    """
    # mininterval throttles writes; do NOT force refresh each iter (floods tee'd logs)
    bar = tqdm(total=total, desc=desc, mininterval=15.0, miniters=50, dynamic_ncols=True)

    def _cb(env):
        if env.evaluation_result_list:
            bar.set_postfix_str(f"l1={env.evaluation_result_list[0][2]:.4f}", refresh=False)
        bar.update((env.iteration + 1) - bar.n)   # tqdm rate-limits the actual display

    _cb.order = 10
    return _cb, bar


def _sev_weights(y: np.ndarray, cfg) -> np.ndarray:
    """Target-severity sample weights: nonzero→sev_nonzero, severe(>=3)→sev_severe, else 1."""
    if not getattr(cfg, "severity", False):
        return None
    w = np.ones(len(y), np.float32)
    w[y > 0] = cfg.sev_nonzero
    w[y >= 3] = cfg.sev_severe
    return w


def _explode(anchors: pd.DataFrame, base: np.ndarray):
    """Stack (features + forecast_week) → rows, dropping NaN (ragged) targets."""
    ords = anchors["anchor_ord"].to_numpy()
    Xs, ys, os = [], [], []
    for fw in C.HORIZONS:
        y = anchors[f"target_w{fw}"].to_numpy(np.float32)
        m = ~np.isnan(y)
        fw_col = np.full(int(m.sum()), fw, np.float32)
        Xs.append(np.column_stack([base[m], fw_col]))
        ys.append(y[m]); os.append(ords[m])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(os)


class Phase1Model:
    def __init__(self, feat_cols: list, model_cfg: C.ModelConfig = C.MODEL):
        self.feat_cols = list(feat_cols)
        self.cfg = model_cfg

    def _matrix(self, X_df: pd.DataFrame) -> np.ndarray:
        return X_df[self.feat_cols].to_numpy(np.float32)

    def fit(self, pool_anchors: pd.DataFrame, X_pool: pd.DataFrame, es_frac: float = 0.05):
        base = self._matrix(X_pool)
        X, y, ords = _explode(pool_anchors, base)
        # temporal early-stopping split: most-recent es_frac of rows (by anchor_ord).
        # Subsample the ES set so the per-iteration metric eval stays cheap (the full
        # tail is ~400k rows; ES never actually fires here, it's just for logging).
        cutoff = np.quantile(ords, 1 - es_frac)
        es = ords >= cutoff
        es_idx = np.where(es)[0]
        if es_idx.size > 100_000:
            rng = np.random.RandomState(C.SEED)
            es_idx = rng.choice(es_idx, 100_000, replace=False)
        self.model_ = lgb.LGBMRegressor(**C.LGBM_PARAMS)
        cb, bar = _tqdm_callback(C.LGBM_PARAMS["n_estimators"], "horizon_feature")
        sw = _sev_weights(y, self.cfg)
        self.model_.fit(
            X[~es], y[~es], sample_weight=(sw[~es] if sw is not None else None),
            eval_set=[(X[es_idx], y[es_idx])], eval_metric="l1",
            callbacks=[lgb.early_stopping(C.EARLY_STOPPING_ROUNDS, verbose=False), cb],
        )
        bar.close()
        self.best_iter_ = self.model_.best_iteration_
        return self

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        base = self._matrix(X_df)
        out = np.zeros((len(base), C.N_PRED), np.float32)
        for j, fw in enumerate(C.HORIZONS):
            Xf = np.column_stack([base, np.full(len(base), fw, np.float32)])
            out[:, j] = self.model_.predict(Xf, num_iteration=self.best_iter_)
        return np.clip(out, 0, 5)

    def feature_importance(self, top: int = 25):
        imp = self.model_.feature_importances_
        names = self.feat_cols + ["forecast_week"]
        order = np.argsort(imp)[::-1][:top]
        return [(names[i], int(imp[i])) for i in order]


class PerHorizonModel:
    """One LGBM per horizon, early-stopped on a Kaggle-proxy in-season holdout.

    Each horizon-h model trains on the anchors with target_w{h}, and stops where its MAE
    on the in-season ES set (test-like calendar position) peaks — the teammate's setup.
    """

    def __init__(self, feat_cols: list, model_cfg: C.ModelConfig = C.MODEL):
        self.feat_cols = list(feat_cols)
        self.cfg = model_cfg

    def _m(self, X_df):
        return X_df[self.feat_cols].to_numpy(np.float32)

    def fit(self, train_anchors, X_train, es_anchors, X_es):
        Xtr_all, Xes_all = self._m(X_train), self._m(X_es)
        self.models_, self.best_iters_ = {}, {}
        for i, h in enumerate(C.HORIZONS, 1):
            ytr = train_anchors[f"target_w{h}"].to_numpy(np.float32); mtr = ~np.isnan(ytr)
            yes = es_anchors[f"target_w{h}"].to_numpy(np.float32); mes = ~np.isnan(yes)
            mdl = lgb.LGBMRegressor(**C.LGBM_PARAMS)
            cb, bar = _tqdm_callback(C.LGBM_PARAMS["n_estimators"], f"h={h} [{i}/{C.N_PRED}]")
            sw = _sev_weights(ytr[mtr], self.cfg)
            mdl.fit(Xtr_all[mtr], ytr[mtr], sample_weight=sw,
                    eval_set=[(Xes_all[mes], yes[mes])], eval_metric="l1",
                    callbacks=[lgb.early_stopping(C.EARLY_STOPPING_ROUNDS, verbose=False), cb])
            bar.close()
            self.models_[h] = mdl
            self.best_iters_[h] = mdl.best_iteration_
            print(f"  [h={h} done] best_iter={mdl.best_iteration_}  in-season ES l1={mdl.best_score_['valid_0']['l1']:.4f}",
                  flush=True)
        self.best_iter_ = int(np.mean(list(self.best_iters_.values())))
        return self

    def predict(self, X_df):
        X = self._m(X_df)
        out = np.zeros((len(X), C.N_PRED), np.float32)
        for j, h in enumerate(C.HORIZONS):
            out[:, j] = self.models_[h].predict(X, num_iteration=self.best_iters_[h])
        return np.clip(out, 0, 5)

    def feature_importance(self, top: int = 25):
        imp = np.mean([self.models_[h].feature_importances_ for h in C.HORIZONS], axis=0)
        order = np.argsort(imp)[::-1][:top]
        return [(self.feat_cols[i], int(imp[i])) for i in order]
