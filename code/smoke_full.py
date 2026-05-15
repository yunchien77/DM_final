"""
Smoke test for the FULL pipeline (every USE_* feature block on), restricted to
5 regions so it runs in ~30s. Verifies the new score-lag / cluster / preproc
wiring without touching real saved models.
"""

import sys
import time
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import config

N_SMOKE = 5
_test_ids = pd.read_csv(config.TEST_PATH, usecols=["region_id"])["region_id"].unique()
SMOKE_REGIONS = sorted(_test_ids, key=lambda x: int(x[1:]))[:N_SMOKE]
print(f"Smoke regions: {SMOKE_REGIONS}")

# Faster LightGBM for the smoke run.
config.LGBM_PARAMS = {**config.LGBM_PARAMS,
                     "n_estimators": 50, "learning_rate": 0.1, "num_leaves": 15}
config.EARLY_STOPPING_ROUNDS = 5
config.N_REGION_CLUSTERS = 3       # k=8 needs ≥8 regions; we have 5
config.RUN_WALK_FORWARD_DIAGNOSTIC = False
config.N_REGIONS = N_SMOKE

# Patch dataset paths to point at the smoke extracts.
tmp_root = Path(tempfile.mkdtemp(prefix="dmfp_smoke_"))
smoke_train_csv = tmp_root / "train.csv"
smoke_test_csv = tmp_root / "test.csv"
smoke_models = tmp_root / "models"
smoke_models.mkdir()
print(f"Smoke tmp dir: {tmp_root}")


def _extract(src_path, dst_path, regions, is_train):
    target = set(regions)
    dtype_map = {f: "float32" for f in config.METEO_FEATURES}
    dtype_map["region_id"] = str
    dtype_map["date"] = str
    if is_train:
        dtype_map["score"] = "Int8"
    first = True
    for chunk in pd.read_csv(src_path, dtype=dtype_map, chunksize=300_000):
        sub = chunk[chunk["region_id"].isin(target)]
        if len(sub):
            sub.to_csv(dst_path, mode="w" if first else "a",
                       header=first, index=False)
            first = False


_extract(config.TRAIN_PATH, smoke_train_csv, SMOKE_REGIONS, is_train=True)
_extract(config.TEST_PATH, smoke_test_csv, SMOKE_REGIONS, is_train=False)

config.TRAIN_PATH = smoke_train_csv
config.TEST_PATH = smoke_test_csv
config.MODELS_DIR = smoke_models
config.SUBMISSION_PATH = tmp_root / "submission.csv"

# Re-import train/predict so they see the patched module-level constants.
import importlib
import train as _train_module
import predict as _predict_module
import preprocessing as _preproc_module
import features_extra as _fx_module
# config attrs flow through at function-call time (they look them up by attribute)
# but the modules also captured them at import time. Re-import to refresh.
importlib.reload(_preproc_module)
importlib.reload(_fx_module)
importlib.reload(_train_module)
importlib.reload(_predict_module)

t0 = time.time()
print("\n=== TRAIN ===")
_train_module.main()
print(f"\n=== PREDICT ===")
_predict_module.main()
print(f"\nTotal elapsed: {time.time() - t0:.1f}s")

# Verify outputs.
sub = pd.read_csv(config.SUBMISSION_PATH)
assert sub.shape == (N_SMOKE, 6), f"submission shape {sub.shape}"
pred_cols = [f"pred_week{h}" for h in config.HORIZONS]
assert (sub[pred_cols] >= 0).all().all() and (sub[pred_cols] <= 5).all().all()
print(f"\nSubmission preview:\n{sub.to_string(index=False)}")

# Verify all expected artifacts were saved.
expected_artifacts = [
    "feature_cols.json",
    "region_stats.csv",
    "preprocessing.json",
    "preprocessing_imputation.csv",
    "preprocessing_quantiles.npz",
    "score_climatology.csv",
    "meteo_climatology.csv",
    "region_last_scores.csv",
    "region_clusters.csv",
]
for name in expected_artifacts:
    p = smoke_models / name
    assert p.exists(), f"missing artifact: {p}"
    print(f"  [OK] {name}")

shutil.rmtree(tmp_root)
print("\n[OK] FULL smoke passed; tmp dir cleaned up")
