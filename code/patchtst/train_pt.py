"""Training entry point for the PatchTST track.

Single-GPU run:
    /mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/python -m patchtst.train_pt

Multi-GPU run (4x via torchrun + DDP):
    torchrun --standalone --nproc_per_node=4 -m patchtst.train_pt

When launched under torchrun, the LOCAL_RANK env var is set on each worker
and we enable DistributedDataParallel + DistributedSampler. Validation runs
only on rank 0 (small set; saves all-gather complexity). Artifacts + logs
are written only by rank 0.

Stages:
  0. (optional) load preprocessing artifacts if USE_PREPROCESSING is on
  1. daily -> weekly aggregation (reuses data_pipeline.load_and_aggregate_daily_to_weekly)
  2. enumerate (region, anchor) pairs, subsample all-zero majority class
  3. fit per-channel mean/std on the training fold only
  4. train PatchTSTRegressor with L1 loss, weighted sample loss, AdamW + cosine LR
  5. report val macro MAE per horizon and save model + artifacts
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))   # allow `import config`, `import data_pipeline`

from config import USE_PREPROCESSING, HORIZONS as LGBM_HORIZONS
from data_pipeline import load_and_aggregate_daily_to_weekly

from patchtst.config_pt import (
    LOOKBACK_WEEKS, HORIZONS, BATCH_SIZE, NUM_WORKERS_LOADER,
    LR, WEIGHT_DECAY, EPOCHS, WARMUP_EPOCHS, GRAD_CLIP,
    EARLY_STOP_PATIENCE, SEED,
    PT_MODEL_PATH, PT_METRICS_PATH,
)
from patchtst.dataset_pt import (
    WeeklyWindowDataset,
    build_region_arrays, fit_channel_norm, save_channel_norm,
    save_region_map, enumerate_anchors, sample_weights,
)
from patchtst.model_pt import build_model_from_config


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize torch.distributed if launched under torchrun.

    Returns: (local_rank, rank, world_size, device).
    When LOCAL_RANK is not set, runs single-process on cuda:0 (or CPU).
    """
    if "LOCAL_RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 0, 1, device
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, rank, world_size, device


def is_main_rank() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def build_lr_schedule(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def macro_mae_per_horizon(preds: np.ndarray, truth: np.ndarray) -> tuple[float, list[float]]:
    per = [float(np.mean(np.abs(preds[:, h] - truth[:, h]))) for h in range(HORIZONS)]
    return float(np.mean(per)), per


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, list[float]]:
    model.eval()
    preds, truths = [], []
    for x, y, r in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True).long()
        out = model(x, r).clamp(0.0, 5.0).cpu().numpy()
        preds.append(out)
        truths.append(y.numpy())
    preds = np.vstack(preds)
    truths = np.vstack(truths)
    return macro_mae_per_horizon(preds, truths)


def main():
    set_seed(SEED)
    local_rank, rank, world_size, device = init_distributed()
    is_main = is_main_rank()
    distributed = world_size > 1

    def log(*args, **kwargs):
        if is_main:
            print(*args, **kwargs, flush=True)

    log(f"Device: {device}  rank={rank}/{world_size}  distributed={distributed}")

    # -- Stage 0 + 1: load + weekly aggregate -----------------------------
    # Only rank 0 reads + aggregates the CSV; other ranks wait at a barrier
    # and read the cached parquet. 4 ranks * 20 mp workers would otherwise
    # spawn 80 processes on a 24-core box.
    preproc_artifacts = None
    if USE_PREPROCESSING:
        from preprocessing import load_preprocessing_artifacts
        try:
            preproc_artifacts = load_preprocessing_artifacts()
            log("[Stage 0] Loaded preprocessing artifacts from LGBM track.")
        except FileNotFoundError:
            log("[Stage 0] No preprocessing artifacts found — running raw daily input.")

    from patchtst.config_pt import PT_MODELS_DIR
    weekly_cache = PT_MODELS_DIR / "weekly_train.pkl"
    t0 = time.time()
    if is_main:
        log("[Stage 1] daily -> weekly aggregation (rank 0)")
        weekly_df = load_and_aggregate_daily_to_weekly(
            is_train=True, preproc_artifacts=preproc_artifacts,
        )
        weekly_df.to_pickle(weekly_cache)
        log(f"  cached -> {weekly_cache}  ({time.time() - t0:.1f}s)")
    if distributed:
        dist.barrier()
    if not is_main:
        import pandas as pd
        weekly_df = pd.read_pickle(weekly_cache)
    log(f"  weekly_df: {weekly_df.shape}")

    # -- Stage 2: per-region arrays + anchor enumeration ------------------
    region_ids, stat_mats, score_arrs = build_region_arrays(weekly_df, is_train=True)
    log(f"[Stage 2] regions={len(region_ids)}  weeks/region={stat_mats[0].shape[0]}")
    if is_main:
        save_region_map(region_ids)

    train_anchors, val_anchors, train_targets = enumerate_anchors(
        stat_mats, score_arrs, seed=SEED,
    )
    log(f"  train anchors after subsampling: {len(train_anchors):,}")
    log(f"  val   anchors                  : {len(val_anchors):,}")

    # -- Stage 3: channel norm on training fold only ----------------------
    mean, std = fit_channel_norm(stat_mats)
    if is_main:
        save_channel_norm(mean, std)
    log(f"[Stage 3] channel norm fit on training weeks: mean[0]={mean[0]:.3f} std[0]={std[0]:.3f}")

    # -- Datasets / loaders ----------------------------------------------
    train_ds = WeeklyWindowDataset(stat_mats, score_arrs, train_anchors, mean, std)
    val_ds   = WeeklyWindowDataset(stat_mats, score_arrs, val_anchors,   mean, std)

    # Per-sample weights -> looked up by index inside the training step.
    weights = sample_weights(train_targets)
    weights_t = torch.from_numpy(weights)
    log(f"  sample weights: mean={weights.mean():.3f} max={weights.max():.1f}")

    # Carry the anchor index through the loader so we can look up its
    # per-sample weight at loss time.
    from torch.utils.data import Dataset as _Dataset
    class WithIndex(_Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            x, y, r = self.base[i]
            return x, y, r, i

    train_wrapped = WithIndex(train_ds)
    if distributed:
        train_sampler = DistributedSampler(
            train_wrapped, num_replicas=world_size, rank=rank,
            shuffle=True, seed=SEED, drop_last=True,
        )
        train_loader = DataLoader(
            train_wrapped, batch_size=BATCH_SIZE, sampler=train_sampler,
            num_workers=NUM_WORKERS_LOADER, pin_memory=True, drop_last=True,
            persistent_workers=NUM_WORKERS_LOADER > 0,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_wrapped, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS_LOADER, pin_memory=True, drop_last=True,
            persistent_workers=NUM_WORKERS_LOADER > 0,
        )

    # Validation runs only on rank 0 — set is small, all-gather not worth it.
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS_LOADER, pin_memory=True,
        persistent_workers=NUM_WORKERS_LOADER > 0,
    ) if is_main else None

    # -- Stage 4: model + optim -----------------------------------------
    base_model = build_model_from_config(n_regions=len(region_ids)).to(device)
    n_params = sum(p.numel() for p in base_model.parameters())
    log(f"[Stage 4] PatchTSTRegressor  params={n_params/1e6:.2f}M  "
        f"(global batch={BATCH_SIZE * world_size})")
    model = DDP(base_model, device_ids=[local_rank]) if distributed else base_model

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    sched = build_lr_schedule(optim, steps_per_epoch * EPOCHS, steps_per_epoch * WARMUP_EPOCHS)

    best_macro = float("inf")
    best_epoch = -1
    patience = 0
    history = []

    for epoch in range(EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}", disable=not is_main)
        for batch in pbar:
            x, y, r, idx = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True).long()
            w = weights_t[idx].to(device, non_blocking=True)

            preds = model(x, r)
            per_sample_l1 = (preds - y).abs().mean(dim=1)         # (B,)
            loss = (per_sample_l1 * w).sum() / w.sum()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            sched.step()

            running_loss += loss.item() * x.size(0)
            running_n += x.size(0)
            if is_main and running_n % (BATCH_SIZE * 20) == 0:
                pbar.set_postfix(loss=running_loss / running_n,
                                 lr=optim.param_groups[0]["lr"])

        train_loss = running_loss / max(1, running_n)

        # Validate on rank 0; broadcast best/stop decisions to other ranks.
        if is_main:
            macro, per = evaluate(model.module if distributed else model,
                                  val_loader, device)
            log(f"  epoch {epoch + 1}: train_loss={train_loss:.4f}  "
                f"val_macro_MAE={macro:.4f}  "
                f"per_h=[{', '.join(f'{p:.3f}' for p in per)}]")
            history.append({"epoch": epoch + 1, "train_loss": train_loss,
                            "val_macro_mae": macro, "val_per_horizon": per})

            if macro < best_macro - 1e-4:
                best_macro = macro
                best_epoch = epoch + 1
                patience = 0
                state = (model.module if distributed else model).state_dict()
                torch.save({
                    "model_state": state,
                    "epoch": epoch + 1,
                    "val_macro_mae": macro,
                    "val_per_horizon": per,
                }, PT_MODEL_PATH)
                log(f"    new best -> saved to {PT_MODEL_PATH}")
                stop_signal = 0
            else:
                patience += 1
                stop_signal = 1 if patience >= EARLY_STOP_PATIENCE else 0
        else:
            stop_signal = 0

        # Sync the early-stop decision across ranks.
        if distributed:
            stop_t = torch.tensor([stop_signal], device=device)
            dist.broadcast(stop_t, src=0)
            stop_signal = int(stop_t.item())
        if stop_signal:
            log(f"  early stopping at epoch {epoch + 1}; best epoch={best_epoch}")
            break

    log(f"\nBest val macro MAE: {best_macro:.4f}  (epoch {best_epoch})")
    if is_main:
        with open(PT_METRICS_PATH, "w") as f:
            json.dump({"best_val_macro_mae": best_macro, "best_epoch": best_epoch,
                       "world_size": world_size, "history": history}, f, indent=2)
        log(f"Wrote metrics -> {PT_METRICS_PATH}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
