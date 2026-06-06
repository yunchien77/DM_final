# Disaster / Drought Severity Forecasting — Group 8

Forecast a per-region severity score (integer 0–5) for the next **5 weeks**
(`pred_week1`..`pred_week5`) for **2,248 regions** from daily meteorological history. The
submission is a weighted blend of three models; this repo trains them and produces the submission
end-to-end.

Best submission `submission_blend_ENS_354025.csv` → Kaggle public **0.7862**.

---

## How to run

### Step 1 — Environment

```bash
conda create -n DMFP python=3.10 -y
conda activate DMFP
pip install -r requirements.txt
```
DLinear trains on a CUDA GPU if one is available; CPU also works.

### Step 2 — Data

The dataset is not included. Download the competition files and put them here:
```
dataset/data/train.csv          dataset/data/test.csv          dataset/sample_submission.csv
```

### Step 3 — Reproduce the submission

A single command trains every component from scratch and writes the two submissions:

```bash
# Reproduce the best submission — LightGBM 8000 iterations/horizon  → Kaggle 0.7862
FULL=1 ./reproduce.sh

# Faster — LightGBM trimmed to 3000 iterations/horizon  → Kaggle 0.7904 (corr 0.998 with the 0.7862 submission)
FULL=1 LGBM_N_EST=3000 ./reproduce.sh

# Quick smoke run — reduced training, only to confirm the pipeline executes end-to-end
./reproduce.sh
```

### Step 4 — Output

The submissions are written to `reproduce/out/`:
```
submission_blend_ENS_354025.csv     <- submit this
submission_blend_TDP_354025.csv
```

---

### What each run does

`./reproduce.sh` calls `reproduce/run.sh`, which **trains every component from scratch** and then
blends:

| Stage | What it does | Output |
|-------|--------------|--------|
| 0. Prepare | builds the preprocessed daily history from `train.csv` (skipped if already built) | `dlinear/patchtst/models_pt/daily_train.pkl` |
| 1. Train DLinear | trains the DLinear members from random initialization (`FULL=1`: 7 members; quick: 3) | checkpoints in `dlinear/patchtst/models_pt/` |
| 2. Predict DLinear | runs the freshly-trained members | `out/base_dlinear.csv`, `out/dlinear_ens_shared7.csv` |
| 3. Train + predict LightGBM | builds 382 features and trains 5 per-horizon `phase1b` boosters from scratch, then predicts | `out/phase1b.csv` |
| 4. Blend | `shrink × (0.40·DLinear + 0.35·teammate + 0.25·phase1b)` | `out/submission_blend_{ENS,TDP}_354025.csv` |

The teammate component is read from `reproduce/frozen/submission_teammate_082.csv`.

### LightGBM iterations: 8000 vs 3000

The original `ENS_354025` (Kaggle **0.7862**) used **8000** LightGBM iterations per horizon — this is
the `FULL=1` default. Setting `LGBM_N_EST=3000` trims the iteration count and reaches **0.7904**
(within 0.004; correlation 0.998 vs the original). The DLinear members reproduce their original
validation MAEs, so the small gap is essentially the LightGBM iteration count.

### Tunable knobs (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `FULL` | `0` | `1` = full training (7 DLinear members, 100 epochs, 8000 LightGBM trees) |
| `LGBM_N_EST` | quick `200` / full `8000` | LightGBM iterations per horizon |
| `DLIN_EPOCHS` | quick `3` / full `100` | DLinear training epochs (early-stopping may cap it) |
| `LGBM_THREADS` | `16` | CPU threads per LightGBM model |
| `N_WORKERS_CAP` | `16` | CPU processes used for feature building |

Example — faithful run pinned to 16 cores:
```bash
FULL=1 LGBM_N_EST=8000 LGBM_THREADS=16 N_WORKERS_CAP=16 ./reproduce.sh
```

---

## Method (brief)

```
submission = shrink × ( 0.40 · DLinear  +  0.35 · teammate_LGBM  +  0.25 · phase1b_LGBM )
```
- **DLinear** — decomposition-linear neural forecaster (Zeng et al., 2023), severity-weighted L1
  loss; `TDP` = single model, `ENS` = mean of seed/kernel members.
- **phase1b LightGBM** — per-horizon gradient-boosted trees (L1) over 382 engineered features.
- **teammate LightGBM** — gap-stratified, isotonic-calibrated GBDT (`reproduce/frozen/...`).
- **shrink** — TDP `0.975092816675869`, ENS `0.9694903458056041`.

Full details, formulas, and experiments are in the submitted report

## Repository structure (brief)

```
dlinear/        DLinear component (training + inference)
lgbm/           phase1b LightGBM component (trains 382 features + 5 boosters)
reproduce/      orchestration: run.sh, train_dlinear.sh, regen_dlinear.py, blend.py, build_daily.py
experiments/    code-only snapshots of other approaches we tried
```
