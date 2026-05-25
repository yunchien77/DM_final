"""
LightGBM model wrappers: build, train, save, and load.
"""

import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

from config import LGBM_PARAMS, EARLY_STOPPING_ROUNDS, HORIZONS, MODELS_DIR
from logging_setup import get_logger

log = get_logger("model")

# Phase 10: walk-forward CV stores a list of fold models per horizon (averaged
# at predict time). A single global IsotonicRegression is fit on OOF preds.
FOLD_MODELS_DIR = MODELS_DIR / "wf_folds"
FOLD_MODELS_DIR.mkdir(exist_ok=True)
CALIBRATOR_PATH = MODELS_DIR / "isotonic_calibrator.pkl"


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
    # LightGBM uses either "l1" or "mae" as the metric key depending on objective.
    bs = model.best_score_.get("valid_0", {}) if model.best_score_ else {}
    val_metric = bs.get("l1") if bs else None
    if val_metric is None and bs:
        val_metric = bs.get("mae")
    extra = f", val_mae={val_metric:.4f}" if isinstance(val_metric, (int, float)) else ""
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


def save_all_models(
    models: dict[int, lgb.LGBMRegressor],
    models_dir: Path = MODELS_DIR,
    prefix: str = "lgbm",
):
    """Save 5 horizon models as `{prefix}_h{1..5}.txt`. For per-cluster training,
    use bucket-prefixed names (e.g. prefix='lgbm_high', 'lgbm_mid', 'lgbm_low')."""
    for h, model in models.items():
        path = models_dir / f"{prefix}_h{h}.txt"
        model.booster_.save_model(str(path))
        log.info(f"Saved horizon {h} model → {path}")


def load_all_models(
    models_dir: Path = MODELS_DIR,
    prefix: str = "lgbm",
) -> dict[int, lgb.Booster]:
    models = {}
    for h in HORIZONS:
        path = models_dir / f"{prefix}_h{h}.txt"
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


# ---------------------------------------------------------------------------
# Walk-forward CV (Phase 10)
# ---------------------------------------------------------------------------

def walk_forward_splits(
    time_keys: np.ndarray,
    n_folds: int,
    purge_weeks: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) walk-forward splits sorted by time_keys.

    Mirrors teammate's TemporalKFold: contiguous time blocks for val, all earlier
    samples for train, with a purge gap that drops train rows whose time_key
    falls within `purge_weeks` of the val block's start. Produces n_folds-1
    yielded folds (the first block is train-only).
    """
    n = len(time_keys)
    sorted_pos = np.argsort(time_keys, kind="stable")
    sorted_times = time_keys[sorted_pos]
    block = n // n_folds

    splits = []
    for fold in range(1, n_folds):
        val_start = fold * block
        val_end = (fold + 1) * block if fold < n_folds - 1 else n
        val_start_time = sorted_times[val_start]
        cutoff_time = val_start_time - purge_weeks
        train_end = int(np.searchsorted(sorted_times, cutoff_time, side="right"))
        if train_end <= 0:
            continue
        tr_idx = sorted_pos[:train_end]
        va_idx = sorted_pos[val_start:val_end]
        actual_gap = int(val_start_time - sorted_times[train_end - 1])
        log.info(
            f"  WF fold {fold}/{n_folds - 1}: train={len(tr_idx):,}  val={len(va_idx):,}  "
            f"purge_gap={actual_gap}w (target={purge_weeks}w)"
        )
        splits.append((tr_idx, va_idx))
    return splits


def train_horizon_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    time_keys: np.ndarray,
    horizon: int,
    sample_weight: np.ndarray | None,
    n_folds: int,
    purge_weeks: int,
    **param_overrides,
) -> tuple[list[lgb.LGBMRegressor], np.ndarray]:
    """Train n_folds-1 LightGBM models via walk-forward CV; return (models, oof).

    Each row's OOF prediction is the val-fold prediction from the model trained
    on earlier blocks (with purge gap). Rows in the first time block never
    appear in any val fold and keep nan as their OOF; isotonic calibration
    fits only on the finite entries.
    """
    n = len(X)
    X_np = X.to_numpy(np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, np.float32)
    y_np = y.to_numpy(np.float32) if isinstance(y, pd.Series) else np.asarray(y, np.float32)

    oof = np.full(n, np.nan, dtype=np.float32)
    fold_models: list[lgb.LGBMRegressor] = []
    fold_maes: list[float] = []

    splits = walk_forward_splits(time_keys, n_folds=n_folds, purge_weeks=purge_weeks)
    if not splits:
        log.warning(f"  horizon {horizon}: no walk-forward splits produced; "
                    f"falling back to single 80/20 split.")
        sp = int(n * 0.8)
        splits = [(np.arange(sp), np.arange(sp, n))]

    for fold_i, (tr_idx, va_idx) in enumerate(splits, start=1):
        X_tr, y_tr = X_np[tr_idx], y_np[tr_idx]
        X_va, y_va = X_np[va_idx], y_np[va_idx]
        sw_tr = sample_weight[tr_idx] if sample_weight is not None else None

        model = build_lgbm_model(**param_overrides)
        n_estimators = {**LGBM_PARAMS, **param_overrides}["n_estimators"]
        pbar_cb = _TqdmCallback(total=n_estimators, desc=f"h{horizon} fold{fold_i}")
        try:
            model.fit(
                X_tr, y_tr,
                sample_weight=sw_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    pbar_cb,
                ],
            )
        finally:
            pbar_cb.close()

        val_pred = np.clip(model.predict(X_va).astype(np.float32), 0.0, 5.0)
        oof[va_idx] = val_pred
        fold_mae = float(np.mean(np.abs(val_pred - y_va)))
        fold_maes.append(fold_mae)
        fold_models.append(model)
        log.info(f"  h{horizon} fold{fold_i}: best_iter={model.best_iteration_}  "
                 f"fold_mae={fold_mae:.4f}")

    log.info(f"  horizon {horizon}: {len(fold_models)} fold models, "
             f"fold MAEs={[f'{m:.4f}' for m in fold_maes]}  "
             f"oof coverage={int(np.isfinite(oof).sum()):,}/{n:,}")
    return fold_models, oof


def save_fold_models(fold_models_by_horizon: dict[int, list[lgb.LGBMRegressor]]):
    """Persist the per-horizon list of fold-trained LGBM boosters."""
    FOLD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for h, models in fold_models_by_horizon.items():
        for fi, m in enumerate(models, start=1):
            path = FOLD_MODELS_DIR / f"lgbm_h{h}_fold{fi}.txt"
            m.booster_.save_model(str(path))
        log.info(f"  saved {len(models)} fold models for horizon {h} → {FOLD_MODELS_DIR}")


def load_fold_models() -> dict[int, list[lgb.Booster]]:
    """Load the per-horizon list of fold-trained boosters from disk."""
    out: dict[int, list[lgb.Booster]] = {}
    for h in HORIZONS:
        fold_files = sorted(FOLD_MODELS_DIR.glob(f"lgbm_h{h}_fold*.txt"))
        if not fold_files:
            raise FileNotFoundError(
                f"No fold models found for horizon {h} in {FOLD_MODELS_DIR}. "
                f"Run train.py to (re)build them."
            )
        out[h] = [lgb.Booster(model_file=str(p)) for p in fold_files]
        log.info(f"  loaded {len(out[h])} fold models for horizon {h}")
    return out


def predict_fold_ensemble(
    fold_models_by_horizon: dict[int, list],
    X_test: pd.DataFrame,
) -> dict[int, np.ndarray]:
    """Mean-average fold predictions per horizon. Each model in the list can be
    an LGBMRegressor or a Booster; both expose .predict(X) → ndarray."""
    preds: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        stacked = np.stack(
            [m.predict(X_test).astype(np.float32) for m in fold_models_by_horizon[h]],
            axis=0,
        )
        preds[h] = stacked.mean(axis=0)
    return preds


def fit_isotonic_oof(oof_preds: np.ndarray, y_true: np.ndarray) -> IsotonicRegression | None:
    """Fit a single global IsotonicRegression on OOF→y, ignoring NaN OOFs.
    Returns None if too few finite samples remain (caller should skip calibration)."""
    mask = np.isfinite(oof_preds) & np.isfinite(y_true)
    if mask.sum() < 50:
        log.warning(f"  isotonic skipped: only {int(mask.sum())} finite OOF samples")
        return None
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=5.0)
    cal.fit(oof_preds[mask], y_true[mask])
    pre = float(np.mean(np.abs(oof_preds[mask] - y_true[mask])))
    cal_pred = np.clip(cal.predict(oof_preds[mask]), 0.0, 5.0)
    post = float(np.mean(np.abs(cal_pred - y_true[mask])))
    log.info(f"  isotonic fit: pre-cal MAE={pre:.4f}  post-cal MAE={post:.4f}  "
             f"(n={int(mask.sum()):,})")
    return cal


def save_calibrator(cal: IsotonicRegression | None):
    if cal is None:
        if CALIBRATOR_PATH.exists():
            CALIBRATOR_PATH.unlink()
        return
    with open(CALIBRATOR_PATH, "wb") as f:
        pickle.dump(cal, f, protocol=4)
    log.info(f"  isotonic calibrator saved → {CALIBRATOR_PATH}")


def load_calibrator() -> IsotonicRegression | None:
    if not CALIBRATOR_PATH.exists():
        return None
    with open(CALIBRATOR_PATH, "rb") as f:
        return pickle.load(f)


def apply_calibration(
    pred_matrix: np.ndarray,
    cal: IsotonicRegression | None,
) -> np.ndarray:
    """Apply isotonic calibration column-by-column then clip to [0, 5]."""
    if cal is None:
        return np.clip(pred_matrix, 0.0, 5.0).astype(np.float32)
    out = np.empty_like(pred_matrix, dtype=np.float32)
    for j in range(pred_matrix.shape[1]):
        out[:, j] = np.clip(cal.predict(pred_matrix[:, j]), 0.0, 5.0)
    return out
