"""Build dlinear/patchtst/models_pt/daily_train.pkl from dataset/data/train.csv if missing.

This preprocessed daily history is the input the DLinear trainer reads. It is deterministic
data preparation (not a trained model), so it is rebuilt from the raw CSV rather than shipped
(it is ~1 GB). Re-run is a no-op if the pickle already exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dlinear"))
sys.path.insert(0, str(ROOT / "dlinear" / "patchtst"))

from config import USE_PREPROCESSING, TRAIN_PATH          # noqa: E402
from patchtst.config_pt import PT_TRAIN_PKL                # noqa: E402
from patchtst.train_pt import _load_daily                  # noqa: E402


def main():
    if PT_TRAIN_PKL.is_file():
        print(f"[build_daily] daily_train.pkl present -> {PT_TRAIN_PKL} (skip)")
        return
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(
            f"{TRAIN_PATH} not found. Put the competition train.csv at dataset/data/train.csv first."
        )
    preproc = None
    if USE_PREPROCESSING:
        from preprocessing import load_preprocessing_artifacts
        try:
            preproc = load_preprocessing_artifacts()
        except FileNotFoundError:
            pass  # preprocessing artifacts are optional; fall back to raw daily input
    print(f"[build_daily] building {PT_TRAIN_PKL} from {TRAIN_PATH} ...", flush=True)
    df = _load_daily(TRAIN_PATH, is_train=True, preproc_artifacts=preproc)
    PT_TRAIN_PKL.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(PT_TRAIN_PKL)
    print(f"[build_daily] wrote {PT_TRAIN_PKL}  shape={df.shape}", flush=True)


if __name__ == "__main__":
    main()
