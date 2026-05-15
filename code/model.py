"""
LightGBM model wrappers: build, train, save, and load.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import LGBM_PARAMS, EARLY_STOPPING_ROUNDS, HORIZONS, MODELS_DIR
from logging_setup import get_logger

log = get_logger("model")


def build_lgbm_model(**overrides) -> lgb.LGBMRegressor:
    params = {**LGBM_PARAMS, **overrides}
    return lgb.LGBMRegressor(**params)


class _TqdmCallback:
    """LightGBM callback that drives a per-iteration tqdm bar and shows val MAE."""

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


def train_single_horizon(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    horizon: int,
    sample_weight: np.ndarray | None = None,
    **param_overrides,
) -> lgb.LGBMRegressor:
    """Train one LightGBM model for a single prediction horizon."""
    model = build_lgbm_model(**param_overrides)
    n_estimators = {**LGBM_PARAMS, **param_overrides}["n_estimators"]
    pbar_cb = _TqdmCallback(total=n_estimators, desc=f"Horizon {horizon}")
    try:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                pbar_cb,
            ],
        )
    finally:
        pbar_cb.close()
    best = model.best_iteration_
    val_l1 = model.best_score_.get("valid_0", {}).get("l1") if model.best_score_ else None
    extra = f", val_l1={val_l1:.4f}" if isinstance(val_l1, (int, float)) else ""
    log.info(f"Horizon {horizon}: best iteration = {best}{extra}")
    return model


def train_all_horizons(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[int, lgb.LGBMRegressor]:
    """Train 5 independent models, one per prediction horizon."""
    models = {}
    for h in HORIZONS:
        log.info(f"--- Training horizon {h} ---")
        y_train = train_df[f"target_w{h}"]
        y_val = val_df[f"target_w{h}"]
        models[h] = train_single_horizon(X_train, y_train, X_val, y_val, horizon=h)
    return models


def save_all_models(models: dict[int, lgb.LGBMRegressor], models_dir: Path = MODELS_DIR):
    for h, model in models.items():
        path = models_dir / f"lgbm_h{h}.txt"
        model.booster_.save_model(str(path))
        log.info(f"Saved horizon {h} model → {path}")


def load_all_models(models_dir: Path = MODELS_DIR) -> dict[int, lgb.Booster]:
    models = {}
    for h in HORIZONS:
        path = models_dir / f"lgbm_h{h}.txt"
        models[h] = lgb.Booster(model_file=str(path))
        log.info(f"Loaded horizon {h} model from {path}")
    return models


def predict_all_horizons(
    models: dict,
    X_test: pd.DataFrame,
) -> dict[int, np.ndarray]:
    """Run inference for all horizons. Works with both LGBMRegressor and Booster."""
    preds = {}
    for h in HORIZONS:
        m = models[h]
        if isinstance(m, lgb.Booster):
            preds[h] = m.predict(X_test)
        else:
            preds[h] = m.predict(X_test)
    return preds
