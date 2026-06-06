"""Train the DLinear regressor (reuses the PatchTST data pipeline).

Differences from train_pt_revin.py:
  - DLinear model (linear decomposition) instead of RevIN PatchTST.
  - apply_global_norm=True (DLinear takes the globally-normed series; the +6C shift then
    appears as a positive offset its linear trend term can extrapolate).
  - REPRESENTATIVE validation: random 90/10 split over ALL valid anchors (mean ~train base
    rate ~1.15), NOT the calendar-matched 0.58-mean slice that made PatchTST under-predict
    (the early-stop objective then rewards the right level).
  - Single-GPU (DLinear is tiny / fast).

Usage: /mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/python -u -m patchtst.train_pt_dlinear
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset as _Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import USE_PREPROCESSING, TRAIN_PATH, METEO_FEATURES
from patchtst.config_pt import (
    HORIZONS, BATCH_SIZE, NUM_WORKERS_LOADER, LR, WEIGHT_DECAY, EPOCHS,
    WARMUP_EPOCHS, GRAD_CLIP, EARLY_STOP_PATIENCE, SEED,
    PT_MODELS_DIR, PT_LOGS_DIR, PT_TRAIN_PKL, PT_NORM_PATH, PT_REGION_MAP_PATH,
)
from patchtst.dataset_pt import (
    DailyWindowDataset, build_region_daily_arrays, enumerate_anchor_days,
    compute_score_lag_side_inputs_train, compute_anchor_targets,
    severity_sample_weights, fit_channel_norm, save_channel_norm, save_region_map,
)
from patchtst.dlinear import build_dlinear_model_from_config

import os
SEED = int(os.environ.get("DLIN_SEED", SEED))          # allow seed-ensemble override
# DLIN_SUF isolates a run's artifacts (model+norm+regmap) so variants can train in PARALLEL safely
_SUF = os.environ.get("DLIN_SUF") or (f"_s{SEED}" if os.environ.get("DLIN_SEED") else "")
PT_DLINEAR_MODEL_PATH = PT_MODELS_DIR / f"patchtst_dlinear{_SUF}.pt"
PT_DLINEAR_METRICS_PATH = PT_MODELS_DIR / f"metrics_dlinear{_SUF}.json"
DLIN_NORM_PATH = PT_NORM_PATH.with_name(PT_NORM_PATH.stem + _SUF + PT_NORM_PATH.suffix)
DLIN_REGMAP_PATH = PT_REGION_MAP_PATH.with_name(PT_REGION_MAP_PATH.stem + _SUF + PT_REGION_MAP_PATH.suffix)
VAL_FRAC = 0.10


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def macro_mae(preds, truth):
    per = [float(np.mean(np.abs(preds[:, h] - truth[:, h]))) for h in range(HORIZONS)]
    return float(np.mean(per)), per


@torch.no_grad()
def evaluate(model, loader, device, pf):
    model.eval(); P, T = [], []
    for x, side, cal, y, r in tqdm(loader, desc="val", leave=False, file=pf, mininterval=2.0, ncols=100):
        out = model(x.to(device), side.to(device), cal.to(device), r.to(device).long()).clamp(0, 5)
        P.append(out.cpu().numpy()); T.append(y.numpy())
    return np.vstack(P), np.vstack(T)


def _load_daily():
    return pd.read_pickle(PT_TRAIN_PKL)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pf = open(PT_LOGS_DIR / f"progress_dlinear_{_ts}.log", "w")
    print(f"Device: {device}  [DLinear]", flush=True)

    if not PT_TRAIN_PKL.is_file():
        raise FileNotFoundError(f"need cached {PT_TRAIN_PKL} (run a patchtst train once)")
    t0 = time.time(); train_daily = _load_daily()
    print(f"[1] daily train {train_daily.shape} ({time.time()-t0:.0f}s)", flush=True)

    region_ids, daily_feats, daily_doys, weekly_scores = build_region_daily_arrays(train_daily, is_train=True)
    del train_daily
    save_region_map(region_ids, path=DLIN_REGMAP_PATH)
    region_sizes = [m.shape[0] for m in daily_feats]
    anchor_days = enumerate_anchor_days(region_sizes, daily_doys, is_train=True)
    score_lag_inputs = compute_score_lag_side_inputs_train(weekly_scores)
    targets = compute_anchor_targets(weekly_scores)

    # all (region, week) anchors flattened
    all_anchors = np.array([(r, int(a)) for r, anc in enumerate(anchor_days) for a in anc], dtype=np.int32)
    # valid = all 5 targets present (week index used directly — Phase-13 anchors ARE week indices)
    valid = np.array([not np.isnan(targets[int(r)][int(d)]).any() for r, d in all_anchors])
    all_anchors = all_anchors[valid]
    # REPRESENTATIVE random split (mean ~ train base rate, not the mild calendar slice)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_anchors))
    n_val = int(len(all_anchors) * VAL_FRAC)
    val_anchors = all_anchors[perm[:n_val]]
    train_anchors = all_anchors[perm[n_val:]]
    tgt_flat = np.stack([targets[int(r)][int(d)] for r, d in train_anchors]).astype(np.float32)
    weights = severity_sample_weights(tgt_flat)
    print(f"[2] train={len(train_anchors):,} val={len(val_anchors):,}  "
          f"train target-mean={tgt_flat.mean():.3f} (representative)", flush=True)

    last_day = [0] * len(region_ids)
    for r, d in train_anchors:
        last_day[int(r)] = max(last_day[int(r)], int(d))
    mean, std = fit_channel_norm(daily_feats, last_day)
    save_channel_norm(mean, std, path=DLIN_NORM_PATH)

    mk = lambda anc: DailyWindowDataset(daily_feats, daily_doys, score_lag_inputs, targets,
                                        anc, mean, std, apply_global_norm=True)
    train_ds, val_ds = mk(train_anchors), mk(val_anchors)

    class WithIdx(_Dataset):
        def __init__(s, b): s.b = b
        def __len__(s): return len(s.b)
        def __getitem__(s, i): return (*s.b[i], i)
    weights_t = torch.from_numpy(weights)
    train_loader = DataLoader(WithIdx(train_ds), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS_LOADER, pin_memory=True, drop_last=True,
                              persistent_workers=NUM_WORKERS_LOADER > 0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS_LOADER, pin_memory=True,
                            persistent_workers=NUM_WORKERS_LOADER > 0)

    model = build_dlinear_model_from_config(n_regions=len(region_ids)).to(device)
    print(f"[3] DLinear params={sum(p.numel() for p in model.parameters())/1e6:.3f}M", flush=True)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    spe = max(1, len(train_loader))
    def lr_l(step):
        ws = spe * WARMUP_EPOCHS
        if step < ws: return (step + 1) / max(1, ws)
        p = (step - ws) / max(1, spe * EPOCHS - ws); return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_l)

    best, best_ep, patience, hist = float("inf"), -1, 0, []
    for ep in range(EPOCHS):
        model.train(); rl = rn = 0
        for x, side, cal, y, r, idx in tqdm(train_loader, desc=f"ep{ep+1}/{EPOCHS}", file=pf, mininterval=3.0, ncols=100):
            x, side, cal, y, r = x.to(device), side.to(device), cal.to(device), y.to(device), r.to(device).long()
            w = weights_t[idx].to(device)
            pred = model(x, side, cal, r)
            loss = ((pred - y).abs().mean(dim=1) * w).sum() / w.sum()
            optim.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optim.step(); sched.step()
            rl += loss.item() * x.size(0); rn += x.size(0)
        P, T = evaluate(model, val_loader, device, pf)
        macro, per = macro_mae(P, T)
        print(f"  ep{ep+1}: train_loss={rl/max(1,rn):.4f} val_macro={macro:.4f} "
              f"val_pred_mean={P.mean():.3f} per_h={[round(p,3) for p in per]}", flush=True)
        hist.append({"epoch": ep + 1, "val_macro": macro, "val_pred_mean": float(P.mean())})
        if macro < best - 1e-4:
            best, best_ep, patience = macro, ep + 1, 0
            torch.save({"model_state": model.state_dict(), "epoch": ep + 1, "val_macro_mae": macro}, PT_DLINEAR_MODEL_PATH)
            print(f"    new best -> {PT_DLINEAR_MODEL_PATH}", flush=True)
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"  early stop ep{ep+1}; best={best_ep}", flush=True); break
    print(f"\nBest val macro MAE: {best:.4f} (epoch {best_ep})", flush=True)
    json.dump({"best_val_macro_mae": best, "best_epoch": best_ep, "history": hist},
              open(PT_DLINEAR_METRICS_PATH, "w"), indent=2)


if __name__ == "__main__":
    main()
