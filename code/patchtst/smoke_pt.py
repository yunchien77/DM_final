"""End-to-end smoke test for the PatchTST track on 5 regions.

Extracts a 5-region slice of train.csv and test.csv into a temp dir, points
the config at the smoke paths + a smoke models dir, runs train_pt.main() for
2 epochs and predict_pt.main(), and asserts the artifacts exist + the
submission is well-formed.

Usage (from code/):
    /mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/python -m patchtst.smoke_pt
"""
from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from config import TRAIN_PATH, TEST_PATH
from patchtst import config_pt

N_SMOKE = 5
_test_ids = pd.read_csv(TEST_PATH, usecols=["region_id"])["region_id"].unique()
SMOKE_REGIONS = sorted(_test_ids, key=lambda x: int(x[1:]))[:N_SMOKE]
print(f"Smoke regions: {SMOKE_REGIONS}")


def extract_regions(src_path: Path, regions: list[str], tmp_path: Path, is_train: bool):
    target_set = set(regions)
    dtype_map = {feat: "float32" for feat in [
        "wind","wind_min","wind_max","wind_range","humidity","tmp","tmp_range",
        "tmp_max","tmp_min","surf_tmp","surf_pre","dp_tmp","wb_tmp","prec"]}
    dtype_map["region_id"] = str
    dtype_map["date"] = str
    if is_train:
        dtype_map["score"] = "Int8"

    first = True
    print(f"  Extracting {len(regions)} regions from {src_path.name}...")
    for chunk in pd.read_csv(src_path, dtype=dtype_map, chunksize=300_000):
        sub = chunk[chunk["region_id"].isin(target_set)]
        if len(sub):
            sub.to_csv(tmp_path, mode="w" if first else "a",
                       header=first, index=False)
            first = False
    rows = sum(1 for _ in open(tmp_path)) - 1
    print(f"  -> {rows:,} rows written to {tmp_path.name}")
    return rows


def main():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        smoke_train = tmpdir / "smoke_train.csv"
        smoke_test = tmpdir / "smoke_test.csv"
        smoke_models = tmpdir / "models_pt"
        smoke_models.mkdir()
        smoke_sub = tmpdir / "submission_patchtst.csv"

        extract_regions(TRAIN_PATH, SMOKE_REGIONS, smoke_train, is_train=True)
        extract_regions(TEST_PATH, SMOKE_REGIONS, smoke_test, is_train=False)

        # Override paths
        config.TRAIN_PATH = smoke_train
        config.TEST_PATH = smoke_test
        config_pt.PT_MODELS_DIR = smoke_models
        config_pt.PT_MODEL_PATH = smoke_models / "patchtst.pt"
        config_pt.PT_NORM_PATH = smoke_models / "channel_norm.npz"
        config_pt.PT_REGION_MAP_PATH = smoke_models / "region_id_map.json"
        config_pt.PT_METRICS_PATH = smoke_models / "metrics.json"
        config_pt.PT_SUBMISSION_PATH = smoke_sub

        # Shrink for speed
        config_pt.EPOCHS = 2
        config_pt.WARMUP_EPOCHS = 0
        config_pt.BATCH_SIZE = 64
        config_pt.NUM_WORKERS_LOADER = 0
        config_pt.EARLY_STOP_PATIENCE = 5
        config_pt.D_MODEL = 32
        config_pt.D_FF = 64
        config_pt.N_LAYERS = 2
        # USE_PREPROCESSING in LGBM config may try to load artifacts that
        # don't exist in the smoke models dir; disable to avoid noise.
        config.USE_PREPROCESSING = False

        # Re-import the dataset / model modules so they pick up overridden
        # config constants captured at module import time.
        for m in list(sys.modules):
            if m.startswith("patchtst.") and m not in ("patchtst", "patchtst.config_pt"):
                del sys.modules[m]
        from patchtst import train_pt, predict_pt

        print("\n[smoke] training (2 epochs, tiny model)")
        train_pt.main()
        assert config_pt.PT_MODEL_PATH.exists(), "train_pt did not save model"
        assert config_pt.PT_NORM_PATH.exists(), "train_pt did not save channel norm"

        print("\n[smoke] predicting")
        predict_pt.main()
        assert smoke_sub.exists(), f"predict_pt did not write {smoke_sub}"
        sub = pd.read_csv(smoke_sub)
        assert len(sub) >= N_SMOKE, f"submission too short: {len(sub)} < {N_SMOKE}"
        cols = ["region_id"] + [f"pred_week{h}" for h in range(1, 6)]
        assert list(sub.columns) == cols, f"unexpected columns: {sub.columns.tolist()}"
        smoke_rows = sub[sub["region_id"].isin(SMOKE_REGIONS)]
        assert smoke_rows.notna().all().all(), "NaNs in smoke region predictions"
        preds = smoke_rows[[f"pred_week{h}" for h in range(1, 6)]].values
        assert ((preds >= 0) & (preds <= 5)).all(), "predictions out of [0, 5]"

        print(f"\n[smoke] PASS in {time.time() - t0:.1f}s")
        print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
