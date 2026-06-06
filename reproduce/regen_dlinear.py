"""Regenerate the two DLinear base legs by running the saved checkpoints.

Outputs (written to reproduce/out/):
  base_dlinear.csv          <- single base member (seed 42, kernel 25) == `submission_dlinear` leg (TDP blend)
  dlinear_ens_shared7.csv   <- unweighted mean of 7 shared-arch members == `submission_dlinear_ens_shared7` leg (ENS blend)

The 7 "shared" members (channel-independent=False) are 5 seeds + 2 kernel variants:
  base(seed42,k25), s11, s22, s33, s44  (kernel 25)   +   k13(seed42,k13), k101(seed42,k101)

We load the 1.1GB daily history ONCE and run all 7 checkpoints, instead of 7 separate
prediction processes. Predictions run on GPU (or CPU); because the trained models are GPU
artifacts with no determinism flags, the forward pass drifts ~1e-4 from the committed CSVs,
which moves the final blend by only ~1e-4 (corr 1.0000000). The models genuinely run each
time — nothing is copied from a cached CSV.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent          # /mnt/1stHDD/juiyun/DMFP
CODE = ROOT / "dlinear"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "patchtst"))

from config import USE_PREPROCESSING, SAMPLE_SUB_PATH                       # noqa: E402
from patchtst.config_pt import (                                           # noqa: E402
    HORIZONS, BATCH_SIZE, NUM_WORKERS_LOADER, PT_TRAIN_PKL,
    PT_NORM_PATH, PT_REGION_MAP_PATH, PT_MODELS_DIR,
)
from patchtst.dataset_pt import (                                          # noqa: E402
    build_region_daily_arrays, load_channel_norm, load_region_map,
    load_score_lag_side_inputs_test,
)
from patchtst.predict_pt import TestDailyDataset, _load_test_daily          # noqa: E402
from patchtst.dlinear import build_dlinear_model_from_config                # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
PRED_COLS = [f"pred_week{h}" for h in range(1, HORIZONS + 1)]

# (suffix, kernel) for each shared-architecture member. suffix "" = base (seed 42).
MEMBERS = [
    ("", 25), ("_s11", 25), ("_s22", 25), ("_s33", 25), ("_s44", 25),
    ("_k13", 13), ("_k101", 101),
]


def _member_paths(suffix: str):
    """Return (model, norm, regmap) paths, or None if this member was not trained
    (quick runs may train only a subset of the 7 members)."""
    model = PT_MODELS_DIR / f"patchtst_dlinear{suffix}.pt"
    norm = PT_NORM_PATH.with_name(PT_NORM_PATH.stem + suffix + PT_NORM_PATH.suffix)
    regmap = PT_REGION_MAP_PATH.with_name(PT_REGION_MAP_PATH.stem + suffix + PT_REGION_MAP_PATH.suffix)
    if all(p.is_file() for p in (model, norm, regmap)):
        return model, norm, regmap
    return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ---- shared inputs (built once) ----
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
    lag_per_region = [load_score_lag_side_inputs_test()[r] for r in region_ids]
    print(f"[shared] {len(region_ids)} regions, history rows/region={daily_feats[0].shape[0]}", flush=True)

    # ---- per-member prediction (skip members not trained in a quick run) ----
    member_preds = {}
    for suffix, kernel in MEMBERS:
        paths = _member_paths(suffix)
        if paths is None:
            print(f"[member {suffix or 'base':>5}] not trained -> skipped", flush=True)
            continue
        model_path, norm_path, regmap_path = paths
        os.environ["DLIN_KERNEL"] = str(kernel)
        os.environ["DLIN_INDIVIDUAL"] = "0"          # shared architecture
        mean, std = load_channel_norm(path=norm_path)
        region_map = load_region_map(path=regmap_path)
        region_emb_idx = np.asarray([region_map[r] for r in region_ids], dtype=np.int64)

        model = build_dlinear_model_from_config(n_regions=len(region_map)).to(device)
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"]); model.eval()

        ds = TestDailyDataset(daily_feats, daily_doys, lag_per_region, region_emb_idx, mean, std)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS_LOADER, pin_memory=True)
        preds = []
        with torch.no_grad():
            for x, side, cal, r in loader:
                out = model(x.to(device), side.to(device), cal.to(device),
                            r.to(device).long()).clamp(0, 5)
                preds.append(out.cpu().numpy())
        preds = np.vstack(preds)
        assert preds.shape == (len(region_ids), HORIZONS), preds.shape
        member_preds[suffix] = preds
        print(f"[member {suffix or 'base':>5}] kernel={kernel:3d}  "
              f"epoch={ckpt['epoch']}  val={ckpt['val_macro_mae']:.4f}  pred_mean={preds.mean():.3f}", flush=True)

    if "" not in member_preds:
        raise SystemExit("base DLinear member (suffix '') not found — train it first.")
    base = member_preds[""]                              # seed 42, kernel 25 -> dlinear leg (TDP)
    ens = np.mean(list(member_preds.values()), axis=0)   # mean of trained members -> ensemble leg (ENS)
    print(f"[ensemble] averaged {len(member_preds)} member(s): {sorted(member_preds)}", flush=True)

    sample = pd.read_csv(SAMPLE_SUB_PATH)
    order = sample[sample["region_id"].isin(set(region_ids))]["region_id"].tolist()
    idx = {r: i for i, r in enumerate(region_ids)}

    def write(arr, name):
        rows = [idx[r] for r in order]
        df = pd.DataFrame({"region_id": order})
        for h, c in enumerate(PRED_COLS):
            df[c] = arr[rows, h]
        path = OUT / name
        df.to_csv(path, index=False)
        print(f"Wrote {path}  shape={df.shape}", flush=True)

    write(base, "base_dlinear.csv")
    write(ens, "dlinear_ens_shared7.csv")


if __name__ == "__main__":
    main()
