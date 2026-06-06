"""Inference entry point for the PatchTST track — daily-resolution refactor.

Loads the trained model and runs one prediction per test region using:
  - the 91 days of test daily meteo data (the entire test window per region)
  - the saved score-lag side inputs from region_last_scores.csv (Phase 1b
    staleness-aware: test's score_lag1 = score at last training week, which
    is naturally 14 weeks before the first prediction target)
  - the saved channel normalization

Usage:
    python -m patchtst.predict_pt
    python -m patchtst.predict_pt --output /abs/path/submission.csv
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    USE_PREPROCESSING, TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH, METEO_FEATURES,
)
from patchtst.config_pt import (
    LOOKBACK_DAYS, HORIZONS, BATCH_SIZE, NUM_WORKERS_LOADER,
    PT_MODEL_PATH, PT_SUBMISSION_PATH, PT_LOGS_DIR, SCORE_LAG_OFFSETS,
    PT_TRAIN_PKL,
)
from patchtst.dataset_pt import (
    build_region_daily_arrays,
    load_channel_norm, load_region_map,
    load_score_lag_side_inputs_test,
    _MONTH_CUM,
)
from patchtst.model_pt import build_model_from_config


# ---------------------------------------------------------------------------
# Test-time dataset
# ---------------------------------------------------------------------------

class TestDailyDataset(Dataset):
    """One sample per region: the full 91-day test window + side inputs."""

    def __init__(
        self,
        daily_feats: list[np.ndarray],
        daily_doys: list[np.ndarray],
        score_lag_side_per_region: list[np.ndarray],
        region_emb_idx: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.daily_feats = daily_feats
        self.daily_doys = daily_doys
        self.score_lag = score_lag_side_per_region
        self.region_emb_idx = region_emb_idx
        self.mean = mean.reshape(1, -1)
        self.std = std.reshape(1, -1)

    def __len__(self):
        return len(self.daily_feats)

    def __getitem__(self, i: int):
        mat = self.daily_feats[i]
        # Always take the last LOOKBACK_DAYS rows (test has exactly 91 days)
        assert mat.shape[0] >= LOOKBACK_DAYS, (
            f"region has only {mat.shape[0]} days, need {LOOKBACK_DAYS}"
        )
        window = mat[-LOOKBACK_DAYS:]
        window = (window - self.mean) / self.std
        x_daily = window.T.astype(np.float32)

        side = self.score_lag[i]   # (N_SCORE_LAG + 2,)

        # Calendar features for the LAST test day (anchor)
        anchor_doy = int(self.daily_doys[i][-1])
        m = 1
        while m < 12 and _MONTH_CUM[m] < anchor_doy:
            m += 1
        cal = np.array([
            np.sin(2 * np.pi * m / 12),
            np.cos(2 * np.pi * m / 12),
            np.sin(2 * np.pi * anchor_doy / 365),
            np.cos(2 * np.pi * anchor_doy / 365),
        ], dtype=np.float32)

        return (
            torch.from_numpy(x_daily),
            torch.from_numpy(side),
            torch.from_numpy(cal),
            int(self.region_emb_idx[i]),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_test_daily(preproc_artifacts) -> pd.DataFrame:
    dtype = {"region_id": str, "date": str}
    for f in METEO_FEATURES:
        dtype[f] = np.float32
    chunks = []
    for chunk in pd.read_csv(TEST_PATH, dtype=dtype, chunksize=500_000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    if preproc_artifacts is not None:
        from preprocessing import apply_pipeline
        bounds, log_features, imputation_table, quantile_table = preproc_artifacts
        df = apply_pipeline(df, bounds, log_features, imputation_table, quantile_table)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_path: Path | str | None = None):
    if output_path is None:
        output_path = PT_SUBMISSION_PATH
    output_path = Path(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {output_path}")

    # Route tqdm to a separate progress file so the predict log stays clean.
    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_log_path = PT_LOGS_DIR / f"predict_progress_{_ts}.log"
    progress_file = open(progress_log_path, "w")
    print(f"[progress] tqdm bar -> {progress_log_path}")

    # -- Load preprocessing artifacts (mirror train) ---------------------
    preproc_artifacts = None
    if USE_PREPROCESSING:
        from preprocessing import load_preprocessing_artifacts
        try:
            preproc_artifacts = load_preprocessing_artifacts()
            print("[Stage 0] loaded preprocessing artifacts.")
        except FileNotFoundError:
            print("[Stage 0] no preprocessing artifacts — raw daily input.")

    # -- Load daily test --------------------------------------------------
    t0 = time.time()
    print("[Stage 1] loading daily test.csv")
    test_daily = _load_test_daily(preproc_artifacts)
    print(f"  daily test: {test_daily.shape}  ({time.time()-t0:.1f}s)")

    # Phase 13: PatchTST input is 260 weeks of per-region history. The test
    # window provides only 13 weeks; the prior 247 weeks come from each
    # region's TRAINING history. We load the cached train daily DataFrame
    # (PT_TRAIN_PKL — written by train_pt.py), aggregate to weekly per region,
    # and concatenate train_weekly + test_weekly per region. TestDailyDataset
    # then takes the LAST LOOKBACK_DAYS (=260) weeks ending at the test anchor.
    print(f"[Stage 1b] loading cached train daily from {PT_TRAIN_PKL}")
    train_daily = pd.read_pickle(PT_TRAIN_PKL)
    print(f"  daily train: {train_daily.shape}")

    # -- Per-region weekly arrays ----------------------------------------
    train_region_ids, train_weekly_feats, _, _ = build_region_daily_arrays(
        train_daily, is_train=False,
    )
    train_weekly_map = dict(zip(train_region_ids, train_weekly_feats))
    del train_daily

    test_region_ids, test_weekly_feats, test_weekly_doys, _ = build_region_daily_arrays(
        test_daily, is_train=False,
    )
    print(f"  test weekly rows/region={test_weekly_feats[0].shape[0]}  "
          f"train weekly rows/region={train_weekly_feats[0].shape[0]}")

    # Concatenate per region. Doys we keep as the test-window weekly doys
    # only (the anchor is the LAST test week, so only test_doys[-1] matters).
    region_ids: list[str] = []
    daily_feats: list[np.ndarray] = []
    daily_doys: list[np.ndarray] = []
    for rid, t_feats, t_doys in zip(test_region_ids, test_weekly_feats, test_weekly_doys):
        if rid not in train_weekly_map:
            print(f"  WARN: region {rid} not in training set; using test-only history")
            feats = t_feats
        else:
            feats = np.vstack([train_weekly_map[rid], t_feats])
        region_ids.append(rid)
        daily_feats.append(feats)
        daily_doys.append(t_doys)   # anchor week's doy is t_doys[-1]
    print(f"[Stage 2] regions={len(region_ids)}  "
          f"concat weekly rows/region={daily_feats[0].shape[0]} (lookback needs {LOOKBACK_DAYS})")

    # -- Load saved artifacts --------------------------------------------
    mean, std = load_channel_norm()
    region_map = load_region_map()
    region_emb_idx = np.asarray(
        [region_map[rid] for rid in region_ids], dtype=np.int64,
    )

    # Score-lag side inputs for test (from region_last_scores.csv saved by LGBM train.py)
    score_lag_table = load_score_lag_side_inputs_test()
    score_lag_per_region = []
    for rid in region_ids:
        if rid not in score_lag_table:
            raise RuntimeError(
                f"region_last_scores.csv missing region_id={rid}. "
                "Run LGBM train.py once to populate."
            )
        score_lag_per_region.append(score_lag_table[rid])

    # -- Model -----------------------------------------------------------
    model = build_model_from_config(n_regions=len(region_map)).to(device)
    ckpt = torch.load(PT_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Stage 3] loaded model from {PT_MODEL_PATH}  "
          f"(epoch {ckpt['epoch']}, val_macro_MAE={ckpt['val_macro_mae']:.4f})")

    # -- Inference -------------------------------------------------------
    test_ds = TestDailyDataset(
        daily_feats, daily_doys, score_lag_per_region,
        region_emb_idx, mean, std,
    )
    loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS_LOADER, pin_memory=True,
    )

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", file=progress_file,
                          mininterval=1.0, ncols=100):
            x_daily, side, cal, r = batch
            x_daily = x_daily.to(device, non_blocking=True)
            side = side.to(device, non_blocking=True)
            cal = cal.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True).long()
            out = model(x_daily, side, cal, r).clamp(0.0, 5.0).cpu().numpy()
            all_preds.append(out)
    preds = np.vstack(all_preds)
    assert preds.shape == (len(region_ids), HORIZONS), preds.shape

    sub = pd.DataFrame({"region_id": region_ids})
    for h in range(HORIZONS):
        sub[f"pred_week{h + 1}"] = preds[:, h]

    sample = pd.read_csv(SAMPLE_SUB_PATH)
    have = set(sub["region_id"])
    sample_subset = sample[sample["region_id"].isin(have)][["region_id"]]
    sub = sample_subset.merge(sub, on="region_id", how="left")
    assert not sub.isna().any().any(), "missing region predictions"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(f"Wrote {output_path}  shape={sub.shape}")
    print(sub.head().to_string(index=False))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchTST inference -> submission CSV.")
    p.add_argument(
        "-o", "--output", type=str, default=None,
        help=f"Output submission CSV path. Default: {PT_SUBMISSION_PATH}",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(output_path=args.output)
