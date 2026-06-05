"""Training entry point for the RevIN-equipped PatchTST.

Mirrors train_pt.py exactly except:
  - Uses `build_revin_model_from_config` instead of the base PatchTST
  - DailyWindowDataset is configured with `apply_global_norm=False`
    (RevIN does its own in-model per-instance normalization)
  - Artifacts land at *_revin.pt, *_revin.npz, metrics_revin.json — does not
    overwrite the existing patchtst_v2 checkpoint

Usage:
    /mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/python -m patchtst.train_pt_revin
    torchrun --standalone --nproc_per_node=4 -m patchtst.train_pt_revin
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset as _Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    USE_PREPROCESSING, TRAIN_PATH, TEST_PATH, METEO_FEATURES, MODELS_DIR,
)
try:
    from config import USE_REGION_CLUSTER_FEATURES
except ImportError:
    USE_REGION_CLUSTER_FEATURES = False   # optional per-cluster val logging only
from validate import compute_test_anchor_woy_per_region

from patchtst.config_pt import (
    LOOKBACK_DAYS, HORIZONS, BATCH_SIZE, NUM_WORKERS_LOADER,
    LR, WEIGHT_DECAY, EPOCHS, WARMUP_EPOCHS, GRAD_CLIP,
    EARLY_STOP_PATIENCE, SEED,
    MAJORITY_KEEP_FRAC,
    PT_MODELS_DIR, PT_LOGS_DIR, PT_TRAIN_PKL,
    USE_CALENDAR_MATCHED_VAL,
    CALENDAR_SLACK_WEEKS, CALENDAR_LAST_YEAR_ONLY,
)
from patchtst.dataset_pt import (
    DailyWindowDataset,
    build_region_daily_arrays,
    enumerate_anchor_days,
    compute_score_lag_side_inputs_train,
    compute_anchor_targets,
    severity_sample_weights,
    fit_channel_norm, save_channel_norm,
    save_region_map,
    split_train_val_anchors,
)
from patchtst.revin import build_revin_model_from_config

# RevIN-specific artifact paths (kept separate from the existing patchtst_v2)
PT_REVIN_MODEL_PATH = PT_MODELS_DIR / "patchtst_revin.pt"
PT_REVIN_METRICS_PATH = PT_MODELS_DIR / "metrics_revin.json"


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[int, int, int, torch.device]:
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
def evaluate(model, loader, device, progress_file=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, truths, r_idxs = [], [], []
    for batch in tqdm(loader, desc="val", leave=False, file=progress_file,
                      mininterval=2.0, ncols=100):
        x_daily, side, cal, y, r = batch
        x_daily = x_daily.to(device, non_blocking=True)
        side = side.to(device, non_blocking=True)
        cal = cal.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True).long()
        out = model(x_daily, side, cal, r).clamp(0.0, 5.0).cpu().numpy()
        preds.append(out)
        truths.append(y.numpy())
        r_idxs.append(r.cpu().numpy())
    return np.vstack(preds), np.vstack(truths), np.concatenate(r_idxs)


# ---------------------------------------------------------------------------
# Daily data load (with preprocessing pipeline reuse)
# ---------------------------------------------------------------------------

def _load_daily(path, is_train: bool, preproc_artifacts) -> pd.DataFrame:
    dtype = {"region_id": str, "date": str}
    for f in METEO_FEATURES:
        dtype[f] = np.float32
    if is_train:
        dtype["score"] = "Int8"

    chunks = []
    for chunk in pd.read_csv(path, dtype=dtype, chunksize=500_000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    if preproc_artifacts is not None:
        from preprocessing import apply_pipeline
        bounds, log_features, imputation_table, quantile_table = preproc_artifacts
        df = apply_pipeline(df, bounds, log_features, imputation_table, quantile_table)

    return df


def _subsample_majority(
    anchors: np.ndarray,
    targets_flat: np.ndarray,
    weights_flat: np.ndarray,
    keep_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if keep_frac >= 1.0:
        return anchors, targets_flat, weights_flat
    all_zero = (targets_flat == 0).all(axis=1)
    rng = np.random.default_rng(seed)
    zero_idx = np.where(all_zero)[0]
    nonzero_idx = np.where(~all_zero)[0]
    n_keep = int(round(len(zero_idx) * keep_frac))
    keep_zero = rng.choice(zero_idx, size=n_keep, replace=False)
    keep = np.sort(np.concatenate([keep_zero, nonzero_idx]))
    return anchors[keep], targets_flat[keep], weights_flat[keep]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    local_rank, rank, world_size, device = init_distributed()
    is_main = is_main_rank()
    distributed = world_size > 1

    def log(*args, **kwargs):
        if is_main:
            print(*args, **kwargs, flush=True)

    if is_main:
        _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        progress_log_path = PT_LOGS_DIR / f"progress_revin_{_ts}.log"
        progress_file = open(progress_log_path, "w")
        log(f"  [progress] tqdm bar -> {progress_log_path}")
    else:
        progress_file = open(os.devnull, "w")

    log(f"Device: {device}  rank={rank}/{world_size}  distributed={distributed}")
    log(f"[REVIN] training the RevIN-equipped variant (variant D: μ,σ as side input)")

    # -- Stage 0: preprocessing artifacts -------------------------------
    preproc_artifacts = None
    if USE_PREPROCESSING:
        from preprocessing import load_preprocessing_artifacts
        try:
            preproc_artifacts = load_preprocessing_artifacts()
            log("[Stage 0] Loaded preprocessing artifacts from LGBM track.")
        except FileNotFoundError:
            log("[Stage 0] No preprocessing artifacts found — running raw daily input.")

    # -- Stage 1: load daily train CSV (reuse cached pickle if present) ---
    t0 = time.time()
    if is_main:
        if PT_TRAIN_PKL.is_file():
            log(f"[Stage 1] reusing cached daily train -> {PT_TRAIN_PKL}")
            train_daily = pd.read_pickle(PT_TRAIN_PKL)
        else:
            log("[Stage 1] loading daily train.csv (rank 0)")
            train_daily = _load_daily(TRAIN_PATH, is_train=True,
                                      preproc_artifacts=preproc_artifacts)
            train_daily.to_pickle(PT_TRAIN_PKL)
        log(f"  daily train: {train_daily.shape}  ({time.time() - t0:.1f}s)")
    if distributed:
        dist.barrier()
    if not is_main:
        train_daily = pd.read_pickle(PT_TRAIN_PKL)
    log(f"  daily train: {train_daily.shape}")

    # -- Stage 2: per-region daily arrays + anchors ----------------------
    log("[Stage 2] building per-region daily arrays + anchor metadata")
    region_ids, daily_feats, daily_doys, weekly_scores = build_region_daily_arrays(
        train_daily, is_train=True,
    )
    del train_daily
    log(f"  regions={len(region_ids)}  "
        f"days/region={daily_feats[0].shape[0]}  "
        f"weeks/region={len(weekly_scores[0])}")
    if is_main:
        save_region_map(region_ids)

    region_sizes = [m.shape[0] for m in daily_feats]
    anchor_days = enumerate_anchor_days(region_sizes, daily_doys, is_train=True)
    score_lag_inputs = compute_score_lag_side_inputs_train(weekly_scores)
    targets = compute_anchor_targets(weekly_scores)

    # -- Stage 3: calendar-matched train/val split ----------------------
    log("[Stage 3] calendar-matched train/val split")
    test_woy_per_region = compute_test_anchor_woy_per_region(TEST_PATH)
    train_anchors, val_anchors = split_train_val_anchors(
        region_ids, daily_doys, anchor_days, test_woy_per_region,
        slack_weeks=CALENDAR_SLACK_WEEKS, last_year_only=CALENDAR_LAST_YEAR_ONLY,
    )
    log(f"  train anchors: {len(train_anchors):,}  "
        f"val anchors: {len(val_anchors):,}")

    def _is_valid(anchors):
        valid_mask = np.zeros(len(anchors), dtype=bool)
        for i, (r_idx, d_idx) in enumerate(anchors):
            w_idx = int(d_idx)   # Phase-13 anchors are ALREADY week indices (was //7 bug)
            t = targets[int(r_idx)][w_idx]
            valid_mask[i] = not np.isnan(t).any()
        return valid_mask

    train_mask = _is_valid(train_anchors)
    val_mask = _is_valid(val_anchors)
    train_anchors = train_anchors[train_mask]
    val_anchors = val_anchors[val_mask]
    log(f"  after NaN-target filter: train={len(train_anchors):,}  val={len(val_anchors):,}")

    train_targets_flat = np.stack([
        targets[int(r)][int(d)] for r, d in train_anchors   # week index (was //7 bug)
    ]).astype(np.float32)
    train_weights_flat = severity_sample_weights(train_targets_flat)
    log(f"  severity weights: mean={train_weights_flat.mean():.3f} "
        f"max={train_weights_flat.max():.1f}  pct>=2: "
        f"{(train_weights_flat>=2).mean()*100:.1f}%")

    train_anchors, train_targets_flat, train_weights_flat = _subsample_majority(
        train_anchors, train_targets_flat, train_weights_flat,
        keep_frac=MAJORITY_KEEP_FRAC, seed=SEED,
    )
    log(f"  after majority subsample (keep_frac={MAJORITY_KEEP_FRAC}): "
        f"train={len(train_anchors):,}")

    # -- Stage 4: channel norm — fit but DO NOT apply to the dataset ----
    # The RevIN model does its own per-instance normalization. We still fit
    # and save the global stats so the existing predict_pt.py path keeps
    # working for the non-RevIN variant.
    log("[Stage 4] fitting per-channel norm (saved but not applied; RevIN normalizes in-model)")
    train_last_day_per_region = [0] * len(region_ids)
    for r_idx, d_idx in train_anchors:
        if int(d_idx) > train_last_day_per_region[int(r_idx)]:
            train_last_day_per_region[int(r_idx)] = int(d_idx)
    mean, std = fit_channel_norm(daily_feats, train_last_day_per_region)
    if is_main:
        save_channel_norm(mean, std)
    log(f"  (unused for RevIN) mean[0]={mean[0]:.3f}  std[0]={std[0]:.3f}")

    # -- Datasets (apply_global_norm=False for RevIN) -------------------
    train_ds = DailyWindowDataset(
        daily_feats, daily_doys, score_lag_inputs, targets,
        train_anchors, mean, std,
        apply_global_norm=False,
    )
    val_ds = DailyWindowDataset(
        daily_feats, daily_doys, score_lag_inputs, targets,
        val_anchors, mean, std,
        apply_global_norm=False,
    )

    class WithIndex(_Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            return (*self.base[i], i)

    train_wrapped = WithIndex(train_ds)
    weights_t = torch.from_numpy(train_weights_flat)

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

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS_LOADER, pin_memory=True,
        persistent_workers=NUM_WORKERS_LOADER > 0,
    ) if is_main else None

    # -- Stage 5: RevIN model + optimizer -------------------------------
    base_model = build_revin_model_from_config(n_regions=len(region_ids)).to(device)
    n_params = sum(p.numel() for p in base_model.parameters())
    log(f"[Stage 5] RevINPatchTSTRegressor params={n_params/1e6:.2f}M  "
        f"(global batch={BATCH_SIZE * world_size})")
    model = DDP(base_model, device_ids=[local_rank]) if distributed else base_model

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    sched = build_lr_schedule(
        optim, steps_per_epoch * EPOCHS, steps_per_epoch * WARMUP_EPOCHS,
    )

    best_macro = float("inf")
    best_epoch = -1
    patience = 0
    history: list[dict] = []

    cluster_csv = MODELS_DIR / "region_clusters.csv"
    cluster_table = (
        pd.read_csv(cluster_csv, dtype={"region_id": str})
        if (USE_REGION_CLUSTER_FEATURES and cluster_csv.is_file()) else None
    )
    region_idx_to_cluster = None
    if cluster_table is not None:
        cluster_lookup = dict(zip(cluster_table["region_id"].astype(str),
                                  cluster_table["region_cluster_id"].astype(int)))
        region_idx_to_cluster = np.array(
            [cluster_lookup.get(rid, -1) for rid in region_ids], dtype=np.int16,
        )

    for epoch in range(EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{EPOCHS}",
                    disable=not is_main, file=progress_file,
                    mininterval=2.0, ncols=100)
        for batch in pbar:
            x_daily, side, cal, y, r, idx = batch
            x_daily = x_daily.to(device, non_blocking=True)
            side = side.to(device, non_blocking=True)
            cal = cal.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True).long()
            w = weights_t[idx].to(device, non_blocking=True)

            preds = model(x_daily, side, cal, r)
            per_sample_l1 = (preds - y).abs().mean(dim=1)
            loss = (per_sample_l1 * w).sum() / w.sum()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            sched.step()

            running_loss += loss.item() * x_daily.size(0)
            running_n += x_daily.size(0)
            if is_main and running_n % (BATCH_SIZE * 20) == 0:
                pbar.set_postfix(loss=running_loss / running_n,
                                 lr=optim.param_groups[0]["lr"])

        train_loss = running_loss / max(1, running_n)

        if is_main:
            preds_arr, truth_arr, r_idxs = evaluate(
                model.module if distributed else model, val_loader, device,
                progress_file=progress_file,
            )
            macro, per = macro_mae_per_horizon(preds_arr, truth_arr)
            log(f"  epoch {epoch+1}: train_loss={train_loss:.4f}  "
                f"val_macro_MAE={macro:.4f}  "
                f"per_h=[{', '.join(f'{p:.3f}' for p in per)}]")

            cluster_report = None
            if region_idx_to_cluster is not None:
                cluster_ids = region_idx_to_cluster[r_idxs]
                abs_err = np.abs(preds_arr - truth_arr).mean(axis=1)
                cluster_macro = {}
                for cid in sorted(np.unique(cluster_ids)):
                    mask = cluster_ids == cid
                    cluster_macro[int(cid)] = float(abs_err[mask].mean())
                log(f"  per-cluster val macro: " +
                    ", ".join(f"c{cid}={v:.3f}" for cid, v in cluster_macro.items()))
                cluster_report = cluster_macro

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_macro_mae": macro,
                "val_per_horizon": per,
                "cluster_macro": cluster_report,
            })

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
                }, PT_REVIN_MODEL_PATH)
                log(f"    new best -> saved to {PT_REVIN_MODEL_PATH}")
                stop_signal = 0
            else:
                patience += 1
                stop_signal = 1 if patience >= EARLY_STOP_PATIENCE else 0
        else:
            stop_signal = 0

        if distributed:
            stop_t = torch.tensor([stop_signal], device=device)
            dist.broadcast(stop_t, src=0)
            stop_signal = int(stop_t.item())
        if stop_signal:
            log(f"  early stopping at epoch {epoch+1}; best epoch={best_epoch}")
            break

    log(f"\nBest val macro MAE: {best_macro:.4f}  (epoch {best_epoch})")
    if is_main:
        with open(PT_REVIN_METRICS_PATH, "w") as f:
            json.dump({
                "best_val_macro_mae": best_macro,
                "best_epoch": best_epoch,
                "world_size": world_size,
                "history": history,
            }, f, indent=2)
        log(f"Wrote metrics -> {PT_REVIN_METRICS_PATH}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
