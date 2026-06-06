"""Reproduce the two best Kaggle submissions BYTE-IDENTICALLY from the frozen base predictions.

Unlike the from-scratch pipeline in ../reproduce/ (which retrains the models and so differs by
a tiny amount), this script blends the *exact original model-output CSVs* and reproduces the
submitted files bit-for-bit.

    submission = SHRINK * (0.40 * D + 0.35 * teammate + 0.25 * phase1b)

  best_submissions/submission_blend_ENS_354025.csv  (Kaggle 0.7862)
      D = base_predictions/submission_dlinear_ens_shared7.csv   SHRINK = 0.9694903458056042
  best_submissions/submission_blend_TDP_354025.csv  (Kaggle 0.7864)
      D = base_predictions/submission_dlinear.csv               SHRINK = 0.975092816675869

The normalized member weights are exactly 0.40 (DLinear) / 0.35 (teammate) / 0.25 (phase1b);
SHRINK is the per-blend shrinkage applied to the summed prediction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE / "base_predictions"
PRED = [f"pred_week{h}" for h in range(1, 6)]

TEAMMATE = BASE / "submission_teammate_082.csv"
PHASE1B = BASE / "submission_phase1b_v2.csv"

BLENDS = {
    "submission_blend_ENS_354025.csv": dict(D=BASE / "submission_dlinear_ens_shared7.csv",
                                            shrink=0.9694903458056042),
    "submission_blend_TDP_354025.csv": dict(D=BASE / "submission_dlinear.csv",
                                            shrink=0.975092816675869),
}


def _load(path: Path, order: np.ndarray) -> np.ndarray:
    return pd.read_csv(path).set_index("region_id").loc[order].reset_index()[PRED].to_numpy(np.float64)


def main():
    # canonical row order = lexical sort of region_id (matches the original submissions)
    order = np.sort(pd.read_csv(TEAMMATE)["region_id"].to_numpy())
    t = _load(TEAMMATE, order)
    p = _load(PHASE1B, order)

    for name, cfg in BLENDS.items():
        d = _load(cfg["D"], order)
        blended = cfg["shrink"] * (0.40 * d + 0.35 * t + 0.25 * p)
        out = pd.DataFrame({"region_id": order})
        for i, c in enumerate(PRED):
            out[c] = blended[:, i]
        out.to_csv(HERE / name, index=False)
        print(f"Wrote {HERE / name}  shape={out.shape}")


if __name__ == "__main__":
    main()
