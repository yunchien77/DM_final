import pickle
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import Optional

from config import (
    GAP_SHORT_THRESHOLD, GAP_STRATA_BLEND,
    MODELS_GAP_SHORT_DIR, MODELS_GAP_LONG_DIR,
    N_PRED,
)
from logging_setup import get_logger

log = get_logger("gap_stratified")


# ─────────────────────────────────────────────────────────────────────────────
# Region stratification
# ─────────────────────────────────────────────────────────────────────────────

def compute_region_strata(region_gap_dict: dict[str, float]) -> dict[str, str]:
    """Classify each region as 'short' or 'long' based on train/test gap.

    Short gap (<13w): ACF > 0.4, score lag features are still informative.
    Long gap (≥13w): ACF < 0.4, model relies primarily on meteorological features.
    """
    strata = {}
    n_short = n_long = 0
    for region, gap_wk in region_gap_dict.items():
        if gap_wk < GAP_SHORT_THRESHOLD:
            strata[str(region)] = "short"
            n_short += 1
        else:
            strata[str(region)] = "long"
            n_long += 1
    log.info(f"Gap strata: short (<{GAP_SHORT_THRESHOLD}w) = {n_short}, "
             f"long (≥{GAP_SHORT_THRESHOLD}w) = {n_long}")
    return strata


def split_dataset_by_stratum(
    X:          np.ndarray,
    y:          np.ndarray,
    groups:     np.ndarray,
    time_keys:  np.ndarray,
    fw_keys:    np.ndarray,
    month_keys: np.ndarray,
    region_strata: dict[str, str],
    adversarial_weights: Optional[np.ndarray] = None,
) -> tuple[dict, dict]:
    """Split training data into short-gap and long-gap subsets.

    Returns (short_data_dict, long_data_dict), each containing
    X, y, groups, time_keys, fw_keys, month_keys, adversarial_weights.
    """
    short_mask = np.array([region_strata.get(str(g), "long") == "short" for g in groups])
    long_mask  = ~short_mask

    def _subset(mask):
        d = dict(
            X          = X[mask],
            y          = y[mask],
            groups     = groups[mask],
            time_keys  = time_keys[mask],
            fw_keys    = fw_keys[mask],
            month_keys = month_keys[mask],
        )
        d["adversarial_weights"] = adversarial_weights[mask] if adversarial_weights is not None else None
        return d

    short_data = _subset(short_mask)
    long_data  = _subset(long_mask)

    log.info(f"  Short-gap samples: {short_mask.sum():,}  "
             f"Long-gap samples: {long_mask.sum():,}")
    return short_data, long_data


# ─────────────────────────────────────────────────────────────────────────────
# Stratified prediction blending
# ─────────────────────────────────────────────────────────────────────────────

def blend_stratified_predictions(
    short_preds:   dict[str, list[float]],
    long_preds:    dict[str, list[float]],
    region_strata: dict[str, str],
    region_gap_dict: dict[str, float],
) -> dict[str, list[float]]:
    """Blend short/long stratum predictions.

    If GAP_STRATA_BLEND: soft blend proportional to gap distance from the threshold.
    Otherwise: hard routing based on the region's assigned stratum.
    """
    blended = {}
    thresh = GAP_SHORT_THRESHOLD
    transition_width = 4.0

    for region in set(list(short_preds.keys()) + list(long_preds.keys())):
        sp = short_preds.get(region)
        lp = long_preds.get(region)

        if sp is None and lp is not None:
            blended[region] = lp; continue
        if lp is None and sp is not None:
            blended[region] = sp; continue
        if sp is None and lp is None:
            blended[region] = [0.5] * N_PRED; continue

        if GAP_STRATA_BLEND:
            gap_wk  = region_gap_dict.get(str(region), thresh)
            long_w  = float(np.clip((gap_wk - thresh) / transition_width, 0.0, 1.0))
            short_w = 1.0 - long_w
            blended[region] = [
                float(np.clip(short_w * s + long_w * l, 0, 5))
                for s, l in zip(sp, lp)
            ]
        else:
            if region_strata.get(str(region), "long") == "short":
                blended[region] = sp
            else:
                blended[region] = lp

    return blended


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_stratified_models(short_models: dict, long_models: dict):
    for fw, fw_models in short_models.items():
        p = MODELS_GAP_SHORT_DIR / f"lgbm_fw{fw}.pkl"
        with open(p, "wb") as f:
            pickle.dump(fw_models, f, protocol=4)
    for fw, fw_models in long_models.items():
        p = MODELS_GAP_LONG_DIR / f"lgbm_fw{fw}.pkl"
        with open(p, "wb") as f:
            pickle.dump(fw_models, f, protocol=4)
    log.info(f"Gap-stratified models saved: "
             f"short→{MODELS_GAP_SHORT_DIR}  long→{MODELS_GAP_LONG_DIR}")


def load_stratified_models() -> tuple[dict[int, list], dict[int, list]]:
    """Load short/long stratum models; returns (short_models, long_models)."""
    def _load_dir(d: Path) -> dict[int, list]:
        models = {}
        for fw in range(1, N_PRED + 1):
            p = d / f"lgbm_fw{fw}.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    loaded = pickle.load(f)
                    models[fw] = loaded if isinstance(loaded, list) else [loaded]
        return models

    short_models = _load_dir(MODELS_GAP_SHORT_DIR)
    long_models  = _load_dir(MODELS_GAP_LONG_DIR)

    if short_models or long_models:
        log.info(f"Gap-stratified models loaded: "
                 f"short fw={sorted(short_models.keys())}  "
                 f"long fw={sorted(long_models.keys())}")
    return short_models, long_models


def predict_stratified(
    test_features:  dict[str, np.ndarray],
    short_models:   dict[int, list],
    long_models:    dict[int, list],
    region_strata:  dict[str, str],
    region_gap_dict: dict[str, float],
    fallback:       float = 0.5,
) -> dict[str, list[float]]:
    """Run short/long stratum models separately, then blend the results."""
    short_preds: dict[str, list[float]] = {}
    long_preds:  dict[str, list[float]] = {}

    for region, feat_arr in test_features.items():
        sp_list, lp_list = [], []
        for fw in range(1, N_PRED + 1):
            fv = feat_arr[fw - 1].reshape(1, -1)

            sp = float(np.mean([m.predict(fv)[0] for m in short_models[fw]])) if fw in short_models else fallback
            lp = float(np.mean([m.predict(fv)[0] for m in long_models[fw]])) if fw in long_models else fallback
            sp_list.append(float(np.clip(sp, 0, 5)))
            lp_list.append(float(np.clip(lp, 0, 5)))

        short_preds[region] = sp_list
        long_preds[region]  = lp_list

    return blend_stratified_predictions(
        short_preds, long_preds, region_strata, region_gap_dict
    )
