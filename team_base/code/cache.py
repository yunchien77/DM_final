from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional


_CACHE_DIR = Path(__file__).resolve().parent.parent / "dataset" / "cache"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def cache_key(config: dict, *file_paths) -> str:
    """Return an 8-character SHA1 hex derived from a config dict and file fingerprints."""
    payload = {
        "config": _canonicalize(config),
        "files":  [_file_fingerprint(p) for p in file_paths],
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def feature_cache_key() -> str:
    """Cache key covering all config settings that affect the feature matrix.

    Training hyperparameters (LGBM params, sample weights) are intentionally
    excluded so that tuning them does not invalidate the expensive feature cache.
    """
    from config import (
        TRAIN_PATH, TEST_PATH,
        METEO_COLS, WINDOW_SIZES, TREND_COLS, PROXY_WINDOWS,
        SAMPLE_FRAC, MAX_WIN_PER_REGION, N_PRED, SEED,
        USE_PREPROCESSING, PREPROC_LOG_FEATURES,
        FEATURE_VERSION,
    )

    cfg = {
        "FEATURE_VERSION":      FEATURE_VERSION,
        "METEO_COLS":           METEO_COLS,
        "WINDOW_SIZES":         WINDOW_SIZES,
        "TREND_COLS":           TREND_COLS,
        "PROXY_WINDOWS":        PROXY_WINDOWS,
        "SAMPLE_FRAC":          SAMPLE_FRAC,
        "MAX_WIN_PER_REGION":   MAX_WIN_PER_REGION,
        "N_PRED":               N_PRED,
        "SEED":                 SEED,
        "USE_PREPROCESSING":    USE_PREPROCESSING,
        "PREPROC_LOG_FEATURES": PREPROC_LOG_FEATURES,
    }
    return cache_key(cfg, TRAIN_PATH, TEST_PATH)


def cache_path(name: str, key: str) -> Path:
    """Return the cache file path, creating parent directories if needed."""
    p = _CACHE_DIR / f"{name}_{key}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_from_cache(name: str, key: str) -> Optional[Any]:
    """Load a cached object; return None if the cache file does not exist."""
    p = cache_path(name, key)
    if not p.is_file():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def save_to_cache(obj: Any, name: str, key: str) -> Path:
    """Atomically pickle obj to the cache directory and return the path."""
    p = cache_path(name, key)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)
    return p


def clear_cache(name_prefix: str | None = None):
    """Remove all (or prefix-matched) cache files and return the count removed."""
    pattern = f"{name_prefix}_*.pkl" if name_prefix else "*.pkl"
    removed = 0
    for p in _CACHE_DIR.glob(pattern):
        p.unlink(missing_ok=True)
        removed += 1
    return removed


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _canonicalize(obj: Any) -> Any:
    """Recursively sort dicts and stringify Paths for deterministic JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _file_fingerprint(p) -> dict:
    """Return (path, size, mtime) fingerprint; sentinel values if file is missing."""
    sp = Path(p)
    try:
        st = sp.stat()
        return {"path": str(sp), "size": st.st_size, "mtime": int(st.st_mtime)}
    except FileNotFoundError:
        return {"path": str(sp), "size": -1, "mtime": -1}
