"""Blend the four base legs into the two best submissions.

Recipe (reverse-engineered to byte-exact on the original base CSVs):
    submission = SHRINK * (0.40 * D + 0.35 * teammate + 0.25 * phase1b_v2)
where the normalized member weights are exactly D=0.40, teammate=0.35, phase1b_v2=0.25
(the "354025" label = teammate/D/P = 35/40/25), and SHRINK is a per-blend constant.

  TDP_354025 : D = base_dlinear            SHRINK = 0.975092816675869
  ENS_354025 : D = dlinear_ens_shared7     SHRINK = 0.9694903458056041

Legs (all TRAINED FROM SCRATCH in the run, except the teammate's):
  D (dlinear / dlinear_ens) : trained by reproduce/train_dlinear.sh, predicted by regen_dlinear.py
  phase1b                   : trained + predicted by the lgbm/ LightGBM pipeline -> out/phase1b.csv
  teammate                  : FROZEN (frozen/submission_teammate_082.csv) -- the only component we do
                              not retrain; it is produced by the teammate's separate repo.

Because every retrained component differs from the original models (random init / stochastic
LightGBM / reduced iterations), the blend is NOT identical to the committed submissions; it is a
freshly-trained reproduction of the same recipe.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
FROZEN = HERE / "frozen"
PRED = [f"pred_week{h}" for h in range(1, 6)]

TEAMMATE = FROZEN / "submission_teammate_082.csv"   # frozen: the teammate's model (only exception)
PHASE1B = OUT / "phase1b.csv"                        # regenerated in-run by lgbm

BLENDS = {
    "submission_blend_TDP_354025.csv": dict(D=OUT / "base_dlinear.csv",        shrink=0.975092816675869),
    "submission_blend_ENS_354025.csv": dict(D=OUT / "dlinear_ens_shared7.csv", shrink=0.9694903458056041),
}


def _load(path: Path, order: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path)
    return df.set_index("region_id").loc[order].reset_index()[PRED].to_numpy(np.float64)


def main():
    # canonical region order = teammate file's order (any consistent order works; the
    # committed blends are written in lexical order, which we match for readability)
    order = np.sort(pd.read_csv(TEAMMATE)["region_id"].to_numpy())
    t = _load(TEAMMATE, order)
    p = _load(PHASE1B, order)

    for name, cfg in BLENDS.items():
        d = _load(cfg["D"], order)
        blended = cfg["shrink"] * (0.40 * d + 0.35 * t + 0.25 * p)
        out = pd.DataFrame({"region_id": order})
        for h, c in enumerate(PRED):
            out[c] = blended[:, h]
        dest = OUT / name
        out.to_csv(dest, index=False)

        msg = f"Wrote {dest}  shape={out.shape}  mean={blended.mean():.4f}"
        committed = ROOT / name
        if committed.is_file():
            ref = _load(committed, order)
            diff = np.abs(blended - ref)
            corr = np.corrcoef(blended.ravel(), ref.ravel())[0, 1]
            msg += (f"\n    vs committed {name}: maxabs={diff.max():.3e} "
                    f"mean={diff.mean():.3e} corr={corr:.7f}")
        print(msg, flush=True)


if __name__ == "__main__":
    main()
