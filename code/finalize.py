"""One-shot recovery: load cached features + saved boosters, compute val MAE,
build & save transition matrices. Use when train.py crashes after LGBM saves
but before the post-training stages.

Run:
    python finalize.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from logging_setup import get_logger
log = get_logger("finalize", label="finalize")

from config import (
    MODELS_DIR, HORIZONS,
    USE_TRANSITION_SMOOTHING, TRANSITION_CLUSTERS, TRANSITION_SMOOTHING_ALPHA,
)
from cache import feature_cache_key, load_from_cache
from model import load_all_models, predict_all_horizons
from validate import evaluate_predictions, print_validation_report
from transition_matrix import build_transition_matrices, save_transition_matrices


def main():
    key = feature_cache_key()
    log.info(f"[cache] key={key}")
    cached = load_from_cache("train_features", key)
    if cached is None:
        raise SystemExit("No cached features. Run train.py first.")
    train_split = cached["train_split"]
    val_split = cached["val_split"]
    weekly_df_full = cached["weekly_df_full"]

    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    log.info(f"  features: {len(feature_cols)}  val rows: {len(val_split):,}")

    log.info("[1/3] loading boosters + predicting val")
    models = load_all_models()
    X_val = val_split[feature_cols]
    preds = predict_all_horizons(models, X_val)
    pred_dict = {h: np.clip(preds[h], 0.0, 5.0) for h in HORIZONS}

    log.info("[2/3] computing validation MAE")
    macro_mae, horizon_maes = evaluate_predictions(val_split, pred_dict)
    print_validation_report(macro_mae, horizon_maes)

    if USE_TRANSITION_SMOOTHING:
        log.info("[3/3] building drought transition matrices")
        matrices, cluster_assignment = build_transition_matrices(
            weekly_df_full, n_clusters=TRANSITION_CLUSTERS,
            smoothing_alpha=TRANSITION_SMOOTHING_ALPHA,
        )
        save_transition_matrices(
            matrices, cluster_assignment,
            horizons=list(HORIZONS), score_levels=list(range(6)),
        )
        log.info(f"  transition matrices saved: shape {matrices.shape}")

    log.info("[done]")


if __name__ == "__main__":
    main()
