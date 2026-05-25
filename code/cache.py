"""Disk cache for the post-feature-engineering DataFrames.

Cache key = SHA1 of (sorted config items, file (size, mtime) for each input
path). Any change to the feature pipeline config invalidates the cache;
flags that only affect training (sample weights, LightGBM params, walk-forward)
are deliberately excluded.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional


_CACHE_DIR = Path(__file__).resolve().parent.parent / "dataset" / "cache"


def cache_key(config: dict, *file_paths: os.PathLike) -> str:
    payload = {
        "config": _canonicalize(config),
        "files": [_file_fingerprint(p) for p in file_paths],
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def feature_cache_key() -> str:
    """Shared cache key for train and predict. Hashes every config setting that
    affects feature columns or values, plus the (size, mtime) of train.csv and
    test.csv."""
    from config import (
        TRAIN_PATH, TEST_PATH, METEO_FEATURES, STAT_SUFFIXES, LAG_WINDOW,
        USE_PREPROCESSING, USE_CLIMATE_FEATURES, USE_WINDOWED_FEATURES,
        USE_SCORE_LAG, USE_PROXY_SCORE, USE_METEO_CLUSTER,
        USE_RANK_NORMALIZATION, RANK_NORMALIZE_FEATURES,
        CALENDAR_MATCHED_VALIDATION, CALENDAR_MATCHED_SLACK_WEEKS,
        CALENDAR_MATCHED_LAST_YEAR_ONLY,
        WINDOWED_CHANNELS, WINDOWED_WINDOWS,
    )
    hash_input = {
        "METEO_FEATURES": METEO_FEATURES,
        "STAT_SUFFIXES": STAT_SUFFIXES,
        "LAG_WINDOW": LAG_WINDOW,
        "USE_PREPROCESSING": USE_PREPROCESSING,
        "USE_CLIMATE_FEATURES": USE_CLIMATE_FEATURES,
        "USE_WINDOWED_FEATURES": USE_WINDOWED_FEATURES,
        "USE_SCORE_LAG": USE_SCORE_LAG,
        "USE_PROXY_SCORE": USE_PROXY_SCORE,
        "USE_METEO_CLUSTER": USE_METEO_CLUSTER,
        "USE_RANK_NORMALIZATION": USE_RANK_NORMALIZATION,
        "RANK_NORMALIZE_FEATURES": list(RANK_NORMALIZE_FEATURES),
        "CALENDAR_MATCHED_VALIDATION": CALENDAR_MATCHED_VALIDATION,
        "CALENDAR_MATCHED_SLACK_WEEKS": CALENDAR_MATCHED_SLACK_WEEKS,
        "CALENDAR_MATCHED_LAST_YEAR_ONLY": CALENDAR_MATCHED_LAST_YEAR_ONLY,
        "WINDOWED_CHANNELS": list(WINDOWED_CHANNELS),
        "WINDOWED_WINDOWS": list(WINDOWED_WINDOWS),
    }
    if USE_CLIMATE_FEATURES:
        from features_climate import CLIMATE_FEATURE_COLS
        hash_input["CLIMATE_FEATURE_COLS"] = CLIMATE_FEATURE_COLS
    if USE_WINDOWED_FEATURES:
        from features_windowed import windowed_feature_cols
        hash_input["WINDOWED_FEATURE_COLS"] = windowed_feature_cols()
    return cache_key(hash_input, TRAIN_PATH, TEST_PATH)


def cache_path(name: str, key: str) -> Path:
    return _CACHE_DIR / f"{name}_{key}.pkl"


def load_from_cache(name: str, key: str) -> Optional[Any]:
    p = cache_path(name, key)
    if not p.is_file():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def save_to_cache(obj: Any, name: str, key: str) -> Path:
    p = cache_path(name, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)
    return p


def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _file_fingerprint(p: os.PathLike) -> dict:
    sp = Path(p)
    st = sp.stat()
    return {"path": str(sp), "size": st.st_size, "mtime": int(st.st_mtime)}
