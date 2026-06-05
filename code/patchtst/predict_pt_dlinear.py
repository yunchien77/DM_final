"""DLinear inference -> submission. Mirrors predict_pt.py (concat train+test history ->
last 260 weeks, global norm) but loads the DLinear model.

Usage: /mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/python -u -m patchtst.predict_pt_dlinear -o /abs/out.csv
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import USE_PREPROCESSING, TEST_PATH, SAMPLE_SUB_PATH, METEO_FEATURES
from patchtst.config_pt import LOOKBACK_DAYS, HORIZONS, BATCH_SIZE, NUM_WORKERS_LOADER, PT_LOGS_DIR, PT_TRAIN_PKL
from patchtst.dataset_pt import build_region_daily_arrays, load_channel_norm, load_region_map, load_score_lag_side_inputs_test
from patchtst.predict_pt import TestDailyDataset, _load_test_daily   # reuse the working test dataset + loader
from patchtst.dlinear import build_dlinear_model_from_config
from patchtst.train_pt_dlinear import PT_DLINEAR_MODEL_PATH, DLIN_NORM_PATH, DLIN_REGMAP_PATH

DLINEAR_SUBMISSION_PATH = Path(__file__).parent.parent.parent / "submission_dlinear.csv"


def main(output_path=None):
    output_path = Path(output_path or DLINEAR_SUBMISSION_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pf = open(PT_LOGS_DIR / f"predict_dlinear_{_ts}.log", "w")
    print(f"Device: {device}  Output: {output_path}", flush=True)

    preproc = None
    if USE_PREPROCESSING:
        from preprocessing import load_preprocessing_artifacts
        try:
            preproc = load_preprocessing_artifacts()
        except FileNotFoundError:
            pass

    test_daily = _load_test_daily(preproc)
    train_daily = pd.read_pickle(PT_TRAIN_PKL)
    tr_ids, tr_feats, _, _ = build_region_daily_arrays(train_daily, is_train=False)
    tr_map = dict(zip(tr_ids, tr_feats)); del train_daily
    te_ids, te_feats, te_doys, _ = build_region_daily_arrays(test_daily, is_train=False)

    region_ids, daily_feats, daily_doys = [], [], []
    for rid, tf, td in zip(te_ids, te_feats, te_doys):
        feats = np.vstack([tr_map[rid], tf]) if rid in tr_map else tf
        region_ids.append(rid); daily_feats.append(feats); daily_doys.append(td)
    print(f"[concat] weekly rows/region={daily_feats[0].shape[0]} (need {LOOKBACK_DAYS})", flush=True)

    mean, std = load_channel_norm(path=DLIN_NORM_PATH)
    region_map = load_region_map(path=DLIN_REGMAP_PATH)
    region_emb_idx = np.asarray([region_map[r] for r in region_ids], dtype=np.int64)
    lag_table = load_score_lag_side_inputs_test()
    lag_per_region = [lag_table[r] for r in region_ids]

    model = build_dlinear_model_from_config(n_regions=len(region_map)).to(device)
    ckpt = torch.load(PT_DLINEAR_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    print(f"[model] loaded {PT_DLINEAR_MODEL_PATH} (epoch {ckpt['epoch']}, val {ckpt['val_macro_mae']:.4f})", flush=True)

    ds = TestDailyDataset(daily_feats, daily_doys, lag_per_region, region_emb_idx, mean, std)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_LOADER, pin_memory=True)
    preds = []
    with torch.no_grad():
        for x, side, cal, r in tqdm(loader, desc="predict", file=pf, mininterval=1.0, ncols=100):
            out = model(x.to(device), side.to(device), cal.to(device), r.to(device).long()).clamp(0, 5)
            preds.append(out.cpu().numpy())
    preds = np.vstack(preds)
    assert preds.shape == (len(region_ids), HORIZONS), preds.shape

    sub = pd.DataFrame({"region_id": region_ids})
    for h in range(HORIZONS):
        sub[f"pred_week{h+1}"] = preds[:, h]
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    sub = sample[sample["region_id"].isin(set(sub["region_id"]))][["region_id"]].merge(sub, on="region_id", how="left")
    assert not sub.isna().any().any()
    sub.to_csv(output_path, index=False)
    print(f"Wrote {output_path}  shape={sub.shape}  pred_mean={preds.mean():.3f}", flush=True)
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("-o", "--output", default=None)
    main(p.parse_args().output)
