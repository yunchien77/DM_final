# best_submissions/ — byte-identical reproduction of the two best submissions

| File | Kaggle public |
|------|---------------|
| `submission_blend_ENS_354025.csv` | **0.7862** |
| `submission_blend_TDP_354025.csv` | **0.7864** |

These are reproduced **bit-for-bit** from the frozen base model predictions:

```bash
python best_submissions/make_byte_identical.py
```

## Why a separate, frozen path

The two submissions are a shrunk weighted blend of three model outputs:

```
submission = SHRINK * ( 0.40 · D  +  0.35 · teammate  +  0.25 · phase1b )
```

| Blend | `D` | `SHRINK` |
|-------|-----|----------|
| ENS_354025 | `base_predictions/submission_dlinear_ens_shared7.csv` | `0.9694903458056042` |
| TDP_354025 | `base_predictions/submission_dlinear.csv` | `0.975092816675869` |

The blend itself is exact arithmetic, so given the **original model-output CSVs** it reproduces the
submitted files byte-for-byte. Those outputs are kept here as fixed files in `base_predictions/`
because the models that produced them (GPU-trained DLinear, multi-threaded LightGBM) are not
bit-reproducible when retrained — so this is the only way to regenerate the exact submitted bytes.

`base_predictions/`:
- `submission_dlinear.csv` — single DLinear (used by TDP)
- `submission_dlinear_ens_shared7.csv` — 7-member DLinear ensemble mean (used by ENS)
- `submission_teammate_082.csv` — teammate's LightGBM
- `submission_phase1b_v2.csv` — phase1b LightGBM (8000 iterations/horizon)

> For the full **train-from-scratch** pipeline (which retrains the models and lands at 0.7904), see
> [`../reproduce/`](../reproduce/). This folder is the byte-exact counterpart.
