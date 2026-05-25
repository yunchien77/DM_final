"""Dump per-anchor PatchTST val predictions for ensembling.

Reuses train_pt.py's val-split logic but skips training:
  1. load daily train (or reuse cached daily_train.pkl)
  2. rebuild per-region daily arrays, anchors, score-lag side inputs, targets
  3. apply the same calendar-matched val split
  4. drop anchors with NaN targets (last HORIZONS weeks)
  5. load saved model + channel norm; run inference on val_loader
  6. save (preds, truth, region_ids, week_idx) to val_preds_patchtst.npz

Output schema matches LGBM's val_preds.npz so the hill-climb can intersect on
(region_id, week_idx).

    /mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/python -m patchtst.dump_val_preds
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

from config import USE_PREPROCESSING, TRAIN_PATH, TEST_PATH, METEO_FEATURES
from validate import compute_test_anchor_woy_per_region

from patchtst.config_pt import (
    HORIZONS, BATCH_SIZE, NUM_WORKERS_LOADER,
    PT_MODEL_PATH, PT_MODELS_DIR, PT_TRAIN_PKL,
    CALENDAR_SLACK_WEEKS, CALENDAR_LAST_YEAR_ONLY,
)
from patchtst.dataset_pt import (
    DailyWindowDataset,
    build_region_daily_arrays,
    enumerate_anchor_days,
    compute_score_lag_side_inputs_train,
    compute_anchor_targets,
    load_channel_norm,
    save_region_map,
    split_train_val_anchors,
)
from patchtst.model_pt import build_model_from_config

OUT_PATH = PT_MODELS_DIR / "val_preds_patchtst.npz"


def _load_daily_train(preproc_artifacts) -> pd.DataFrame:
    if PT_TRAIN_PKL.is_file():
        print(f"[Stage 1] reusing cached daily train -> {PT_TRAIN_PKL}")
        return pd.read_pickle(PT_TRAIN_PKL)
    print("[Stage 1] cache miss; loading daily train.csv from scratch")
    dtype = {"region_id": str, "date": str}
    for f in METEO_FEATURES:
        dtype[f] = np.float32
    dtype["score"] = "Int8"
    chunks = []
    for chunk in pd.read_csv(TRAIN_PATH, dtype=dtype, chunksize=500_000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    if preproc_artifacts is not None:
        from preprocessing import apply_pipeline
        bounds, log_features, imputation_table, quantile_table = preproc_artifacts
        df = apply_pipeline(df, bounds, log_features, imputation_table, quantile_table)
    return df


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    preproc_artifacts = None
    if USE_PREPROCESSING:
        from preprocessing import load_preprocessing_artifacts
        try:
            preproc_artifacts = load_preprocessing_artifacts()
            print("[Stage 0] loaded preprocessing artifacts.")
        except FileNotFoundError:
            print("[Stage 0] no preprocessing artifacts — raw daily input.")

    t0 = time.time()
    train_daily = _load_daily_train(preproc_artifacts)
    print(f"  daily train: {train_daily.shape}  ({time.time()-t0:.1f}s)")

    print("[Stage 2] building per-region daily arrays")
    region_ids, daily_feats, daily_doys, weekly_scores = build_region_daily_arrays(
        train_daily, is_train=True,
    )
    del train_daily
    save_region_map(region_ids)  # regenerate for safety; identical mapping
    print(f"  regions={len(region_ids)}  days/region={daily_feats[0].shape[0]}")

    region_sizes = [m.shape[0] for m in daily_feats]
    anchor_days = enumerate_anchor_days(region_sizes, daily_doys, is_train=True)
    score_lag_inputs = compute_score_lag_side_inputs_train(weekly_scores)
    targets = compute_anchor_targets(weekly_scores)

    print("[Stage 3] calendar-matched val split")
    test_woy_per_region = compute_test_anchor_woy_per_region(TEST_PATH)
    _, val_anchors = split_train_val_anchors(
        region_ids, daily_doys, anchor_days, test_woy_per_region,
        slack_weeks=CALENDAR_SLACK_WEEKS, last_year_only=CALENDAR_LAST_YEAR_ONLY,
    )
    print(f"  raw val anchors: {len(val_anchors):,}")

    # NaN-target filter (matches train_pt.py)
    valid = np.zeros(len(val_anchors), dtype=bool)
    for i, (r_idx, d_idx) in enumerate(val_anchors):
        w_idx = int(d_idx) // 7
        t = targets[int(r_idx)][w_idx]
        valid[i] = not np.isnan(t).any()
    val_anchors = val_anchors[valid]
    print(f"  after NaN-target filter: {len(val_anchors):,}")

    truths = np.stack([
        targets[int(r)][int(d) // 7] for r, d in val_anchors
    ]).astype(np.float32)

    # Map back to (region_id, week_idx) for cross-model join
    val_region_ids = np.array(
        [region_ids[int(r)] for r in val_anchors[:, 0]], dtype=object,
    )
    val_week_idx = (val_anchors[:, 1].astype(np.int32) // 7).astype(np.int32)

    print("[Stage 4] loading channel norm + model checkpoint")
    mean, std = load_channel_norm()

    val_ds = DailyWindowDataset(
        daily_feats, daily_doys, score_lag_inputs, targets,
        val_anchors, mean, std,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS_LOADER, pin_memory=True,
    )

    model = build_model_from_config(n_regions=len(region_ids)).to(device)
    ckpt = torch.load(PT_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  loaded {PT_MODEL_PATH}  "
          f"(epoch {ckpt['epoch']}, val_macro_MAE={ckpt['val_macro_mae']:.4f})")

    print("[Stage 5] inference on val anchors")
    preds_chunks = []
    with torch.no_grad():
        for batch in tqdm(val_loader, ncols=100):
            x_daily, side, cal, _y, r = batch
            x_daily = x_daily.to(device, non_blocking=True)
            side = side.to(device, non_blocking=True)
            cal = cal.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True).long()
            out = model(x_daily, side, cal, r).clamp(0.0, 5.0).cpu().numpy()
            preds_chunks.append(out)
    preds = np.vstack(preds_chunks).astype(np.float32)
    assert preds.shape == (len(val_anchors), HORIZONS), preds.shape

    macro = float(np.mean([np.mean(np.abs(preds[:, h] - truths[:, h])) for h in range(HORIZONS)]))
    print(f"  reproduced val macro MAE = {macro:.4f}")

    np.savez(
        OUT_PATH,
        preds=preds,
        truth=truths,
        region_ids=val_region_ids.astype(str),
        week_idx=val_week_idx,
        horizons=np.arange(1, HORIZONS + 1, dtype=np.int32),
    )
    print(f"Wrote {OUT_PATH}  N={len(preds)}  macro_mae={macro:.4f}")


if __name__ == "__main__":
    main()
