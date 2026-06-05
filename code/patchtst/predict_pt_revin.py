"""Test-time inference for the RevIN-equipped PatchTST.

Mirrors predict_pt.py but loads the RevIN model and feeds RAW daily values
(no global norm) — the model's RevIN layer normalizes per instance.
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    USE_PREPROCESSING, TEST_PATH, SAMPLE_SUB_PATH, METEO_FEATURES,
)
from patchtst.config_pt import (
    LOOKBACK_DAYS, HORIZONS, BATCH_SIZE, NUM_WORKERS_LOADER,
    PT_LOGS_DIR, PT_TRAIN_PKL,
)
from patchtst.dataset_pt import (
    build_region_daily_arrays,
    load_region_map,
    load_score_lag_side_inputs_test,
    _MONTH_CUM,
)
from patchtst.revin import build_revin_model_from_config
from patchtst.train_pt_revin import PT_REVIN_MODEL_PATH

REVIN_SUBMISSION_PATH = Path(__file__).parent.parent.parent / "submission_patchtst_revin.csv"


class TestDailyDatasetRaw(Dataset):
    """One sample per region: the full 91-day test window, NO global norm
    (RevIN normalizes in-model)."""

    def __init__(
        self,
        daily_feats: list[np.ndarray],
        daily_doys: list[np.ndarray],
        score_lag_side_per_region: list[np.ndarray],
        region_emb_idx: np.ndarray,
    ):
        self.daily_feats = daily_feats
        self.daily_doys = daily_doys
        self.score_lag = score_lag_side_per_region
        self.region_emb_idx = region_emb_idx

    def __len__(self):
        return len(self.daily_feats)

    def __getitem__(self, i: int):
        mat = self.daily_feats[i]
        assert mat.shape[0] >= LOOKBACK_DAYS, (
            f"region has only {mat.shape[0]} days, need {LOOKBACK_DAYS}"
        )
        window = mat[-LOOKBACK_DAYS:]              # raw, no global norm
        x_daily = window.T.astype(np.float32)

        side = self.score_lag[i]

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


def main(output_path: Path | str | None = None):
    if output_path is None:
        output_path = REVIN_SUBMISSION_PATH
    output_path = Path(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {output_path}")

    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_log_path = PT_LOGS_DIR / f"predict_revin_progress_{_ts}.log"
    progress_file = open(progress_log_path, "w")
    print(f"[progress] tqdm bar -> {progress_log_path}")

    preproc_artifacts = None
    if USE_PREPROCESSING:
        from preprocessing import load_preprocessing_artifacts
        try:
            preproc_artifacts = load_preprocessing_artifacts()
            print("[Stage 0] loaded preprocessing artifacts.")
        except FileNotFoundError:
            print("[Stage 0] no preprocessing artifacts — raw daily input.")

    t0 = time.time()
    print("[Stage 1] loading daily test.csv")
    test_daily = _load_test_daily(preproc_artifacts)
    print(f"  daily test: {test_daily.shape}  ({time.time()-t0:.1f}s)")

    # Phase 13 / RevIN fix: the model needs 260 WEEKS of per-region history, but the
    # test window is only 13 weeks. Concatenate each region's TRAIN weekly history
    # (cached PT_TRAIN_PKL) + test weekly, then take the last LOOKBACK_DAYS weeks
    # (done in TestDailyDatasetRaw via mat[-LOOKBACK_DAYS:]). Mirrors predict_pt.py.
    print(f"[Stage 1b] loading cached train daily from {PT_TRAIN_PKL}")
    train_daily = pd.read_pickle(PT_TRAIN_PKL)
    train_region_ids, train_weekly_feats, _, _ = build_region_daily_arrays(
        train_daily, is_train=False,
    )
    train_weekly_map = dict(zip(train_region_ids, train_weekly_feats))
    del train_daily

    test_region_ids, test_weekly_feats, test_weekly_doys, _ = build_region_daily_arrays(
        test_daily, is_train=False,
    )
    region_ids, daily_feats, daily_doys = [], [], []
    for rid, t_feats, t_doys in zip(test_region_ids, test_weekly_feats, test_weekly_doys):
        feats = (np.vstack([train_weekly_map[rid], t_feats])
                 if rid in train_weekly_map else t_feats)
        region_ids.append(rid)
        daily_feats.append(feats)
        daily_doys.append(t_doys)   # anchor week's doy is t_doys[-1]
    print(f"[Stage 2] regions={len(region_ids)}  "
          f"concat weekly rows/region={daily_feats[0].shape[0]} (lookback needs {LOOKBACK_DAYS})")

    region_map = load_region_map()
    region_emb_idx = np.asarray(
        [region_map[rid] for rid in region_ids], dtype=np.int64,
    )

    score_lag_table = load_score_lag_side_inputs_test()
    score_lag_per_region = []
    for rid in region_ids:
        if rid not in score_lag_table:
            raise RuntimeError(
                f"region_last_scores.csv missing region_id={rid}. "
                "Run LGBM train.py once to populate."
            )
        score_lag_per_region.append(score_lag_table[rid])

    model = build_revin_model_from_config(n_regions=len(region_map)).to(device)
    ckpt = torch.load(PT_REVIN_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Stage 3] loaded model from {PT_REVIN_MODEL_PATH}  "
          f"(epoch {ckpt['epoch']}, val_macro_MAE={ckpt['val_macro_mae']:.4f})")

    test_ds = TestDailyDatasetRaw(
        daily_feats, daily_doys, score_lag_per_region, region_emb_idx,
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
    p = argparse.ArgumentParser(description="RevIN PatchTST inference -> submission CSV.")
    p.add_argument(
        "-o", "--output", type=str, default=None,
        help=f"Output submission CSV path. Default: {REVIN_SUBMISSION_PATH}",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(output_path=args.output)
