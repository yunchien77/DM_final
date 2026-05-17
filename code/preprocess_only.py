"""Build the train-features cache without running LGBM training.

Useful for:
  - Inspecting the preprocessed dataframe via EDA/eda_preprocessed.ipynb
  - Warming the cache so subsequent train.py runs skip stages [0/5]–[3/5]

Usage:
    python -m preprocess_only           # build + cache the train features

Cache lands at dataset/cache/train_features_<key>.pkl where <key> is the
8-char hash from cache.feature_cache_key().
"""
from __future__ import annotations

import time

from logging_setup import get_logger
log = get_logger("preprocess_only", label="train")

from cache import cache_path, feature_cache_key, load_from_cache, save_to_cache
from train import _build_train_features


def main():
    key = feature_cache_key()
    path = cache_path("train_features", key)
    log.info(f"[cache] key={key}  path={path}")

    if load_from_cache("train_features", key) is not None:
        log.info("[cache] hit: features already cached. Nothing to do.")
        log.info(f"  → inspect via EDA/eda_preprocessed.ipynb")
        return

    log.info("[cache] miss: running stages [0/5]–[3/5]...")
    t0 = time.time()
    train_split, val_split, all_feature_cols = _build_train_features()
    save_to_cache(
        {
            "train_split": train_split,
            "val_split": val_split,
            "all_feature_cols": all_feature_cols,
        },
        "train_features", key,
    )
    log.info(f"[cache] saved → {path}")
    log.info(f"  train_split: {train_split.shape}")
    log.info(f"  val_split  : {val_split.shape}")
    log.info(f"  feature cols: {len(all_feature_cols)}")
    log.info(f"Done in {(time.time() - t0) / 60:.2f} min.")


if __name__ == "__main__":
    main()
