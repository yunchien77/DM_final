# Disaster / Drought Severity Forecasting — Group [FILL: group ID]

Forecast a per-region severity score (integer 0–5) for the next **5 weeks**
(`pred_week1`..`pred_week5`) for **2,248 regions** from daily meteorological history.

The final Kaggle submission is a weighted blend of three complementary models. **This repo trains
every component from scratch and blends them end-to-end** — only the teammate's component is reused
as a fixed file.

| Submission | Kaggle public (lower = better) |
|------------|-------------------------------|
| `submission_blend_ENS_354025.csv` (best) | **0.7862** |
| from-scratch reproduction (this code) | **0.7904** (corr 0.998) |

---

## 1. Method

```
submission = shrink × ( 0.40 · DLinear  +  0.35 · teammate_LGBM  +  0.25 · phase1b_LGBM )
```

| Component | What it is | Trained here? |
|-----------|------------|---------------|
| **DLinear** | Decomposition-linear neural forecaster (Zeng et al., 2023), severity-weighted L1 loss. `TDP` = single model; `ENS` = mean of seed/kernel members. | **Yes** (PyTorch, GPU) |
| **phase1b LightGBM** | Per-horizon gradient-boosted trees (L1) over 382 engineered tabular features. | **Yes** (LightGBM) |
| **teammate LightGBM** | Gap-stratified, isotonic-calibrated LightGBM (Kaggle ≈0.82), from a separate repo. | **No** — frozen file |
| `shrink` | Per-blend scalar: TDP `0.975092816675869`, ENS `0.9694903458056041`. | — |

The three components are only moderately correlated (Pearson 0.66–0.73), so blending reduces error;
the blend (0.7862) beats the strongest single component (teammate ≈0.82).

---

## 2. Setup

**Environment** (Python 3.10):
```bash
conda create -n DMFP python=3.10 -y
conda activate DMFP
pip install -r requirements.txt
# DLinear trains on GPU if available (CUDA torch); CPU also works (slower).
```

**Data** — the dataset is **not** included in the repo. Download the competition files and place
them here:
```
dataset/data/train.csv          dataset/data/test.csv          dataset/sample_submission.csv
```
On the first run the pipeline auto-builds the preprocessed daily history
(`dlinear/patchtst/models_pt/daily_train.pkl`, ~1 GB) from `train.csv`; all other model artifacts
are trained from scratch.

---

## 3. How to run (reproduce the submissions)

One command trains all components from scratch and writes the two submissions:

```bash
# QUICK — smoke-scale, just to see the whole pipeline run end-to-end (~30–40 min)
./reproduce.sh

# FULL — faithful run, LightGBM 8000 trees/horizon  (~6–8 h)  → reproduces 0.7862
FULL=1 ./reproduce.sh

# FULL but faster — LightGBM trimmed to 3000 trees  (~4.3 h on 16 cores)  → 0.7904 (corr 0.998)
FULL=1 LGBM_N_EST=3000 ./reproduce.sh
```

**Outputs** (in `reproduce/out/`):
```
submission_blend_ENS_354025.csv   <- submit this (best)
submission_blend_TDP_354025.csv
base_dlinear.csv  dlinear_ens_shared7.csv  phase1b.csv   (component predictions)
```

### Iteration count: 3000 vs 8000

The original `ENS_354025` (Kaggle **0.7862**) used **8000** LightGBM boosting iterations per horizon.
- `FULL=1 ./reproduce.sh` uses **8000** (default) → reproduces **0.7862**.
- `FULL=1 LGBM_N_EST=3000 ./reproduce.sh` trims to **3000** to keep run time reasonable → **0.7904**
  (within 0.004 of the original; correlation 0.998).
- DLinear members reproduce their original validation MAEs exactly, so the small gap is almost
  entirely the LightGBM iteration count.

### What the run does (the 4 stages of `reproduce/run.sh`)

| Stage | Command (under the hood) | Output |
|-------|--------------------------|--------|
| 1. Train DLinear | `reproduce/train_dlinear.sh` → `dlinear/patchtst/train_pt_dlinear.py` (quick: 3 members; full: 7) | checkpoints in `dlinear/patchtst/models_pt/` |
| 2. Predict DLinear | `reproduce/regen_dlinear.py` | `out/base_dlinear.csv`, `out/dlinear_ens_shared7.csv` |
| 3. Train + predict LightGBM | `lgbm/train.py` then `lgbm/predict.py` | `out/phase1b.csv` |
| 4. Blend | `reproduce/blend.py` | `out/submission_blend_{ENS,TDP}_354025.csv` |

### Tunable knobs (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `FULL` | `0` | `1` = full training (7 DLinear members, 100 epochs, 8000 LGBM trees) |
| `LGBM_N_EST` | quick 200 / full 8000 | LightGBM boosting iterations per horizon |
| `DLIN_EPOCHS` | quick 3 / full 100 | DLinear training epochs (early-stopping caps it) |
| `LGBM_THREADS` | 16 | CPU threads per LightGBM model |
| `N_WORKERS_CAP` | 16 | CPU processes for feature building |

Example — a 16-core, 3000-tree faithful run:
```bash
FULL=1 LGBM_N_EST=3000 LGBM_THREADS=16 N_WORKERS_CAP=16 ./reproduce.sh
```

---

## 4. Repository structure

```
reproduce.sh            one-command entry point  ->  reproduce/run.sh
requirements.txt        Python dependencies
dlinear/                DLinear component (PyTorch)
  patchtst/             model, dataset, training, config; models_pt/ (checkpoints + daily_train.pkl)
  models/               preprocessing + score-lag artifacts
lgbm/                   phase1b LightGBM component (trains 382 features + 5 per-horizon boosters)
reproduce/              orchestration
  run.sh                4-stage train+blend pipeline
  train_dlinear.sh      trains the DLinear members
  regen_dlinear.py      predicts DLinear members -> base_dlinear.csv, dlinear_ens_shared7.csv
  blend.py              the shrunk weighted blend
  frozen/               teammate's fixed prediction file (the only non-retrained leg)
  out/                  generated submissions
dataset/data/           train.csv, test.csv (+ sample_submission.csv)
report/                 LaTeX report (main.tex, references.bib)
archive/                old experiments — not needed to reproduce (kept for reference)
submission_blend_*.csv  the original best submissions (0.7862), preserved for reference
```

---

## 5. Notes

- A reproduction run trains fresh models (random init / stochastic boosting), so it re-runs the
  **recipe** faithfully (0.7862 → 0.7904) rather than producing a bit-identical file.
- The **teammate** component cannot be retrained here (its model lives in a separate repo); its
  prediction is included as `reproduce/frozen/submission_teammate_082.csv`.
- The DLinear members are 5 seeds {42,11,22,33,44} at kernel 25 plus 2 kernel variants {13,101} at
  seed 42, averaged equally (quick mode trains a subset).
