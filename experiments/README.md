# experiments/ — approaches we tried

Reference code for the alternatives we explored during development. These informed the design of
the final pipeline (see `../report/`), but are **not** part of it.

> **Note:** this is a code-only snapshot for transparency. These scripts depend on the earlier
> project structure (not shipped here), so they are **reference, not runnable standalone**. The
> final, runnable pipeline is `dlinear/` + `lgbm/` + `reproduce/`.

## `neural_alternatives/` — sequence models we evaluated before choosing DLinear
PatchTST-family forecasters, each with a model definition + training + inference script:
- `revin.py` — reversible instance normalization variant
- `gru.py` — GRU recurrent forecaster
- `ssm.py` — state-space-model forecaster
- `autoformer.py` — Autoformer (decomposition + auto-correlation attention)

We settled on **DLinear** as the neural leg (simplest and most stable on this data); these are
the alternatives that motivated that choice.

## `ensemble_calibration/` — blending & calibration methods
The tooling behind the final blend recipe:
- `hillclimb.py` — greedy, positive-weight blend-weight search on per-cluster validation MAE
- `calibrate.py` — per-horizon isotonic-regression calibration
- `per_cluster_isotonic.py`, `per_region_shift.py` — cluster-/region-conditional post-processing we tested
- `make_submission.py`, `analyze.py` — blend assembly and distribution diagnostics

These experiments led to the final `0.40 / 0.35 / 0.25` weights and the per-blend shrinkage used in
`reproduce/blend.py`.

## `dmfp2_rewrite/` — a clean-slate re-architecture
An exploratory rewrite focused on distribution shift (train/test covariate shift): shift-aware
validation, anchor-based weekly construction, a proxy model, and physically-motivated drought
indices (`physical_models/linphys_model.py`, `physical_models/spei_model.py`).
