"""Inference entry point for the PatchTST track.

Reads weekly stats from train.csv + test.csv (the last LOOKBACK_WEEKS - 13
training weeks + 13 test weeks form the input window per region), loads the
trained PatchTSTRegressor, and writes submission_patchtst.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import USE_PREPROCESSING, SAMPLE_SUB_PATH
from data_pipeline import load_and_aggregate_daily_to_weekly

from patchtst.config_pt import (
    LOOKBACK_WEEKS, HORIZONS, BATCH_SIZE, NUM_WORKERS_LOADER,
    PT_MODEL_PATH, PT_SUBMISSION_PATH,
)
from patchtst.dataset_pt import (
    WeeklyWindowDataset,
    build_region_arrays, load_channel_norm, load_region_map,
)
from patchtst.model_pt import build_model_from_config


def _last_anchor_per_region(stat_mats):
    """One anchor per region at the last week. Format matches WeeklyWindowDataset."""
    anchors = []
    for r_idx, mat in enumerate(stat_mats):
        anchors.append((r_idx, mat.shape[0] - 1))
    return np.asarray(anchors, dtype=np.int32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    preproc_artifacts = None
    if USE_PREPROCESSING:
        from preprocessing import load_preprocessing_artifacts
        try:
            preproc_artifacts = load_preprocessing_artifacts()
            print("[Stage 0] Loaded preprocessing artifacts from LGBM track.")
        except FileNotFoundError:
            print("[Stage 0] No preprocessing artifacts found — raw daily input.")

    # -- Aggregate train and test, then concatenate per region ------------
    t0 = time.time()
    print("[Stage 1a] aggregating train -> weekly")
    train_weekly = load_and_aggregate_daily_to_weekly(
        is_train=True, preproc_artifacts=preproc_artifacts,
    )
    print(f"  train weekly: {train_weekly.shape}  ({time.time() - t0:.1f}s)")

    t0 = time.time()
    print("[Stage 1b] aggregating test -> weekly")
    test_weekly = load_and_aggregate_daily_to_weekly(
        is_train=False, preproc_artifacts=preproc_artifacts,
    )
    print(f"  test  weekly: {test_weekly.shape}  ({time.time() - t0:.1f}s)")

    # Make a unified per-region stat matrix by concatenating train + test
    # weeks. The score column is unknown for test weeks; we set it to -1 for
    # safety though predict-time WeeklyWindowDataset will not consume it.
    train_weekly = train_weekly.sort_values(["region_id", "week_idx"]).reset_index(drop=True)
    test_weekly = test_weekly.sort_values(["region_id", "week_idx"]).reset_index(drop=True)

    region_ids_train, stat_mats_train, _ = build_region_arrays(train_weekly, is_train=True)
    region_ids_test, stat_mats_test, _ = build_region_arrays(test_weekly, is_train=False)
    assert region_ids_train == region_ids_test, \
        "Region ordering differs between train and test aggregation — investigate."

    n_train_weeks = stat_mats_train[0].shape[0]
    n_test_weeks = stat_mats_test[0].shape[0]
    need_train_weeks = LOOKBACK_WEEKS - n_test_weeks
    if need_train_weeks <= 0:
        print(f"  test alone provides >= lookback weeks ({n_test_weeks} >= {LOOKBACK_WEEKS}) "
              f"— using last {LOOKBACK_WEEKS} test weeks only.")
        stat_mats = [mt[-LOOKBACK_WEEKS:] for mt in stat_mats_test]
    else:
        print(f"  concatenating last {need_train_weeks} train weeks + {n_test_weeks} test weeks "
              f"= {LOOKBACK_WEEKS} weeks/region")
        stat_mats = [
            np.vstack([mtr[-need_train_weeks:], mte])
            for mtr, mte in zip(stat_mats_train, stat_mats_test)
        ]

    region_ids = region_ids_train

    # Sanity: every concatenated matrix has exactly LOOKBACK_WEEKS rows.
    for r_idx, mat in enumerate(stat_mats):
        assert mat.shape[0] == LOOKBACK_WEEKS, (
            f"region {region_ids[r_idx]}: {mat.shape[0]} rows, expected {LOOKBACK_WEEKS}"
        )

    # -- Load artifacts and model ---------------------------------------
    mean, std = load_channel_norm()
    region_map = load_region_map()
    score_arrs = [np.full(LOOKBACK_WEEKS, -1.0, dtype=np.float32) for _ in stat_mats]

    anchors = _last_anchor_per_region(stat_mats)
    test_ds = WeeklyWindowDataset(
        stat_mats, score_arrs, anchors, mean, std, return_targets=False,
    )

    # WeeklyWindowDataset with return_targets=False yields (x, r_idx). We
    # remap r_idx (positional) to the trained-model region embedding index.
    class TestWrap:
        def __init__(self, base, region_ids, region_map):
            self.base = base
            self.region_emb_idx = np.asarray(
                [region_map[rid] for rid in region_ids], dtype=np.int64,
            )
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            x, r_pos = self.base[i]
            return x, int(self.region_emb_idx[r_pos])

    loader = DataLoader(
        TestWrap(test_ds, region_ids, region_map),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS_LOADER, pin_memory=True,
    )

    model = build_model_from_config(n_regions=len(region_map)).to(device)
    ckpt = torch.load(PT_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Stage 2] loaded model from {PT_MODEL_PATH}  "
          f"(epoch {ckpt['epoch']}, val_macro_MAE={ckpt['val_macro_mae']:.4f})")

    # -- Inference ------------------------------------------------------
    all_preds = []
    with torch.no_grad():
        for x, r in tqdm(loader, desc="predict"):
            x = x.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True).long()
            out = model(x, r).clamp(0.0, 5.0).cpu().numpy()
            all_preds.append(out)
    preds = np.vstack(all_preds)
    assert preds.shape == (len(region_ids), HORIZONS), preds.shape

    sub = pd.DataFrame({"region_id": region_ids})
    for h in range(HORIZONS):
        sub[f"pred_week{h + 1}"] = preds[:, h]

    # Match the row order of sample_submission.csv for deterministic output.
    # If we ran on a subset of regions (e.g. smoke), keep only that subset
    # rather than emit NaN rows for regions we didn't predict.
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    have = set(sub["region_id"])
    sample_subset = sample[sample["region_id"].isin(have)][["region_id"]]
    sub = sample_subset.merge(sub, on="region_id", how="left")
    assert not sub.isna().any().any(), "missing region predictions among the ones we predicted"

    sub.to_csv(PT_SUBMISSION_PATH, index=False)
    print(f"Wrote {PT_SUBMISSION_PATH}  shape={sub.shape}")
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
