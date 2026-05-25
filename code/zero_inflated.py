"""Phase 9 — zero-inflated two-stage model.

~67% of training scores are exactly zero. A single regression objective
(L1, L2, or Tweedie) has to model both the bursty "drought present"
distribution *and* the dense zero-mass simultaneously — fundamentally
mismatched. The teammate's pipeline (Kaggle 0.82) explicitly decouples
these via a two-stage model:

  Stage 1 (classifier):  P(y > 0 | x)        — LightGBM binary
  Stage 2 (regressor):   E(y | y > 0, x)     — LightGBM (Tweedie or L1),
                                                trained on the non-zero
                                                subset only.

  Final:  pred = P(y > 0 | x) · E(y | y > 0, x)

Each horizon trains its own (clf, reg) pair → 10 LightGBM models total.

Module entrypoints:
  train_zi_horizon(X_train, y_train, X_val, y_val, horizon, sample_weight, ...)
      → (clf, reg)

  save_zi_models(models_by_h, models_dir, prefix='lgbm_zi')
  load_zi_models(models_dir, prefix='lgbm_zi') → {h: (clf, reg)}
  predict_zi(models_by_h, X)                    → {h: array[N]}

Sample weights compose as follows:
  - The classifier sees the WHOLE training set; weight = `sample_weight` is
    passed through. (Class imbalance is the same in train and val.)
  - The regressor sees only the non-zero subset; weight = the corresponding
    slice of `sample_weight`. The severity-weight contribution (which scales
    rows with y ≥ 3) keeps its effect on the regressor.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import LGBM_PARAMS, EARLY_STOPPING_ROUNDS, HORIZONS, MODELS_DIR
from logging_setup import get_logger

log = get_logger("zi")


# ---------------------------------------------------------------------------
# Per-stage default params
# ---------------------------------------------------------------------------

def _classifier_params() -> dict:
    """Binary classifier params for stage-1. Inherits most LGBM_PARAMS but
    swaps objective/metric and drops the Tweedie-specific knob."""
    p = {**LGBM_PARAMS}
    p["objective"] = "binary"
    p["metric"] = "binary_logloss"
    p.pop("tweedie_variance_power", None)
    return p


def _regressor_params() -> dict:
    """Stage-2 regressor on the non-zero subset. Keeps the existing
    Tweedie-on-tail objective (variance_power=1.5 acts as a near-Gamma on
    the positive subset). Same metric ('mae') for early stopping."""
    return {**LGBM_PARAMS}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class _TqdmCallback:
    """Reused per-stage progress bar (mirrors model._TqdmCallback)."""

    order = 10
    before_iteration = False

    def __init__(self, total: int, desc: str):
        self.pbar = tqdm(total=total, desc=desc, unit="iter", leave=False)

    def __call__(self, env):
        self.pbar.update(1)
        if env.evaluation_result_list:
            _, eval_name, result = env.evaluation_result_list[0][:3]
            self.pbar.set_postfix_str(f"{eval_name}={result:.4f}")

    def close(self):
        self.pbar.close()


def train_zi_horizon(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    horizon: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[lgb.LGBMClassifier, lgb.LGBMRegressor]:
    """Train one (classifier, regressor) pair for a single horizon.

    Classifier: predicts P(y > 0 | x) on the full train+val.
    Regressor: predicts E(y | y > 0, x) on the non-zero subset only.
    """
    y_train_arr = np.asarray(y_train, dtype=np.float32)
    y_val_arr = np.asarray(y_val, dtype=np.float32)
    y_train_bin = (y_train_arr > 0).astype(np.int8)
    y_val_bin = (y_val_arr > 0).astype(np.int8)

    # ---- Stage 1: classifier ----
    clf_params = _classifier_params()
    n_iter = clf_params.get("n_estimators", 3000)
    clf = lgb.LGBMClassifier(**clf_params)
    pbar = _TqdmCallback(total=n_iter, desc=f"H{horizon} clf")
    try:
        clf.fit(
            X_train, y_train_bin,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val_bin)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), pbar],
        )
    finally:
        pbar.close()
    val_clf = (clf.best_score_ or {}).get("valid_0", {})
    clf_metric = val_clf.get("binary_logloss")
    log.info(
        f"H{horizon} clf: best_iter={clf.best_iteration_}  "
        f"val_logloss={clf_metric:.4f}" if isinstance(clf_metric, float) else f"H{horizon} clf done"
    )

    # ---- Stage 2: regressor on non-zero subset ----
    nz_train = y_train_arr > 0
    nz_val = y_val_arr > 0
    n_nz_tr = int(nz_train.sum())
    n_nz_va = int(nz_val.sum())
    log.info(
        f"H{horizon} reg: train non-zero rows {n_nz_tr:,}/{len(y_train_arr):,} "
        f"({n_nz_tr / len(y_train_arr) * 100:.1f}%)  "
        f"val non-zero {n_nz_va:,}/{len(y_val_arr):,}"
    )
    if n_nz_tr == 0:
        raise RuntimeError(f"H{horizon} regressor: no non-zero training rows.")
    if n_nz_va == 0:
        raise RuntimeError(f"H{horizon} regressor: no non-zero val rows.")

    X_train_nz = X_train.iloc[nz_train.nonzero()[0]]
    y_train_nz = y_train_arr[nz_train]
    X_val_nz = X_val.iloc[nz_val.nonzero()[0]]
    y_val_nz = y_val_arr[nz_val]
    sw_train_nz = sample_weight[nz_train] if sample_weight is not None else None

    reg_params = _regressor_params()
    reg = lgb.LGBMRegressor(**reg_params)
    pbar = _TqdmCallback(total=reg_params.get("n_estimators", 3000), desc=f"H{horizon} reg")
    try:
        reg.fit(
            X_train_nz, y_train_nz,
            sample_weight=sw_train_nz,
            eval_set=[(X_val_nz, y_val_nz)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), pbar],
        )
    finally:
        pbar.close()
    val_reg = (reg.best_score_ or {}).get("valid_0", {})
    reg_metric = val_reg.get("l1") or val_reg.get("mae")
    log.info(
        f"H{horizon} reg: best_iter={reg.best_iteration_}  "
        f"val_mae={reg_metric:.4f}" if isinstance(reg_metric, float) else f"H{horizon} reg done"
    )

    return clf, reg


# ---------------------------------------------------------------------------
# Persistence + inference
# ---------------------------------------------------------------------------

def save_zi_models(
    models_by_h: dict[int, tuple[lgb.LGBMClassifier, lgb.LGBMRegressor]],
    models_dir: Path = MODELS_DIR,
    prefix: str = "lgbm_zi",
) -> None:
    for h, (clf, reg) in models_by_h.items():
        clf_path = models_dir / f"{prefix}_clf_h{h}.txt"
        reg_path = models_dir / f"{prefix}_reg_h{h}.txt"
        clf.booster_.save_model(str(clf_path))
        reg.booster_.save_model(str(reg_path))
        log.info(f"Saved H{h} → {clf_path.name}, {reg_path.name}")


def load_zi_models(
    models_dir: Path = MODELS_DIR,
    prefix: str = "lgbm_zi",
) -> dict[int, tuple[lgb.Booster, lgb.Booster]]:
    out: dict[int, tuple[lgb.Booster, lgb.Booster]] = {}
    for h in HORIZONS:
        clf_path = models_dir / f"{prefix}_clf_h{h}.txt"
        reg_path = models_dir / f"{prefix}_reg_h{h}.txt"
        clf = lgb.Booster(model_file=str(clf_path))
        reg = lgb.Booster(model_file=str(reg_path))
        out[h] = (clf, reg)
        log.info(f"Loaded H{h} → {clf_path.name}, {reg_path.name}")
    return out


def predict_zi(
    models_by_h: dict[int, tuple],
    X: pd.DataFrame,
) -> dict[int, np.ndarray]:
    """Return {h: pred} where pred[i] = P(y>0|x_i) · E(y|y>0, x_i), clipped [0, 5]."""
    preds: dict[int, np.ndarray] = {}
    for h, (clf, reg) in models_by_h.items():
        # Boosters from raw save: predict() returns log-odds for binary; sklearn
        # LGBMClassifier's booster outputs the raw score, so we apply sigmoid.
        # LGBMRegressor's booster outputs the regression target directly.
        if isinstance(clf, lgb.Booster):
            raw = clf.predict(X)
            p = 1.0 / (1.0 + np.exp(-raw))
        else:
            p = clf.predict_proba(X)[:, 1]
        if isinstance(reg, lgb.Booster):
            mu = reg.predict(X)
        else:
            mu = reg.predict(X)
        preds[h] = np.clip(p * mu, 0.0, 5.0).astype(np.float32)
    return preds
