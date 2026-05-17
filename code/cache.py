"""Disk cache for the post-feature-engineering dataframes.

The train and predict pipelines spend ~4.7 min rebuilding the 1.7M-row feature
matrix on every run. This module skips that rebuild when nothing meaningful has
changed.

Cache key = SHA1 of (sorted config items, file (size, mtime) for each input
path). Any change to the feature pipeline config invalidates the cache; flags
that only affect training (sample weights, LightGBM params, walk-forward) are
deliberately excluded by the caller.

Files land under <repo>/dataset/cache/<name>_<key>.pkl. Format is pickle for
parity with the PatchTST track (parquet has environment issues in this env).
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
    """Return an 8-char hex key derived from config + input-file fingerprints."""
    payload = {
        "config": _canonicalize(config),
        "files": [_file_fingerprint(p) for p in file_paths],
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def feature_cache_key() -> str:
    """Shared cache key for train and predict. Hashes every config setting that
    affects feature columns or values, plus the (size, mtime) of train.csv and
    test.csv. Training-only knobs (sample weights, LGBM params, walk-forward)
    are deliberately excluded so they don't invalidate the cache."""
    from config import (
        TRAIN_PATH, TEST_PATH, METEO_FEATURES, STAT_SUFFIXES, LAG_WINDOW,
        USE_PREPROCESSING, USE_EXTRA_FEATURES, USE_SCORE_LAG_FEATURES,
        USE_REGION_CLUSTER_FEATURES, N_REGION_CLUSTERS,
        USE_RANK_NORMALIZATION, RANK_NORMALIZE_FEATURES,
        CALENDAR_MATCHED_VALIDATION, CALENDAR_MATCHED_SLACK_WEEKS,
        CALENDAR_MATCHED_LAST_YEAR_ONLY,
    )
    hash_input = {
        "METEO_FEATURES": METEO_FEATURES,
        "STAT_SUFFIXES": STAT_SUFFIXES,
        "LAG_WINDOW": LAG_WINDOW,
        "USE_PREPROCESSING": USE_PREPROCESSING,
        "USE_EXTRA_FEATURES": USE_EXTRA_FEATURES,
        "USE_SCORE_LAG_FEATURES": USE_SCORE_LAG_FEATURES,
        "USE_REGION_CLUSTER_FEATURES": USE_REGION_CLUSTER_FEATURES,
        "N_REGION_CLUSTERS": N_REGION_CLUSTERS,
        "USE_RANK_NORMALIZATION": USE_RANK_NORMALIZATION,
        "RANK_NORMALIZE_FEATURES": RANK_NORMALIZE_FEATURES,
        # Split choice changes which rows feed region_stats / climatologies,
        # so it must invalidate the feature cache.
        "CALENDAR_MATCHED_VALIDATION": CALENDAR_MATCHED_VALIDATION,
        "CALENDAR_MATCHED_SLACK_WEEKS": CALENDAR_MATCHED_SLACK_WEEKS,
        "CALENDAR_MATCHED_LAST_YEAR_ONLY": CALENDAR_MATCHED_LAST_YEAR_ONLY,
    }
    if USE_SCORE_LAG_FEATURES:
        from features_extra import SCORE_LAG_OFFSETS
        from config import SCORE_LAG_TEST_GAP_SHIFT
        hash_input["SCORE_LAG_OFFSETS"] = SCORE_LAG_OFFSETS
        hash_input["SCORE_LAG_TEST_GAP_SHIFT"] = SCORE_LAG_TEST_GAP_SHIFT
    if USE_EXTRA_FEATURES:
        from features_extra import ANOMALY_FEATURES, TREND_FEATURES, EXTRA_FEATURE_COLS_V2
        hash_input["ANOMALY_FEATURES"] = ANOMALY_FEATURES
        hash_input["TREND_FEATURES"] = TREND_FEATURES
        # Tracks interaction-column membership and any future block additions.
        hash_input["EXTRA_FEATURE_COLS_V2"] = EXTRA_FEATURE_COLS_V2
    return cache_key(hash_input, TRAIN_PATH, TEST_PATH)


def cache_path(name: str, key: str) -> Path:
    return _CACHE_DIR / f"{name}_{key}.pkl"


def load_from_cache(name: str, key: str) -> Optional[Any]:
    """Return cached object or None if not present."""
    p = cache_path(name, key)
    if not p.is_file():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def save_to_cache(obj: Any, name: str, key: str) -> Path:
    """Pickle `obj` to the cache and return the resolved path."""
    p = cache_path(name, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)
    return p


def _canonicalize(obj: Any) -> Any:
    """Make list/dict/Path/etc. order- and type-deterministic for hashing."""
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
