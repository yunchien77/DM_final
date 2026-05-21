# Drought Severity Prediction

Predicts drought severity (score 0–5) for 1–5 weeks ahead, evaluated by MAE.

---

## Pipeline Overview

```
Raw daily meteorological data
    │
    ▼ [Stage 0] Preprocessing  (winsorization → imputation → log/sqrt → rank-norm)
    │
    ▼ [Stage 1] Climatology    (per-region long-term mean/std, monthly baselines, score stats)
    │
    ▼ [Stage 2] Proxy Ridge    (meteorological signals → drought proxy score)
    │
    ▼ [Stage 3] Feature matrix (parallel, 91-day rolling window, ~430 features)
    │
    ▼ [Stage 3.5] Adversarial Weights (reweight train to match test distribution)
    │
    ▼ [Stage 4] Model training
    │     4a: Phase 1 LightGBM → Feature Importance
    │     4b: Feature Pruning (drop bottom 25% by importance)
    │     4c-1: Phase 2 LightGBM (main model, Calendar-Matched Validation)
    │     4c-2: Gap-Stratified LightGBM (short/long gap strata)
    │     4d: XGBoost (GPU-accelerated, 20% blend weight)
    │     4e: Zero-Inflated two-stage model (59.6% zero optimization)
    │     4f: PatchTST (DataParallel across 6 GPUs, 364-day window)
    │
    ▼ [Stage 5] Isotonic Calibration + persistence
    │
    ▼ predict.py → submission.csv
```

---

## Key Design Decisions

### 1. Proxy Score (no score lag dependency)

The train/test gap averages ~163 weeks. Score autocorrelation `0.96^163 ≈ 0.001`, so score lag features are nearly useless for test predictions.

Instead, a **Proxy Score** is estimated from pure meteorological signals (VPD, precipitation deficit, dry-day ratio, temperature anomaly, etc.) via Ridge regression — ensuring identical feature spaces for train and test.

```
3 time scales (7d / 21d / 91d) × 7 physical signals = 21-D proxy features
Ridge → p7, p21, p91, p_main  (plus short–long differentials)
```

### 2. Per-horizon Models

One LightGBM model is trained per forecast week (fw=1..5):

- fw=1: uses the most recent 91 days, predicts 1 week ahead
- fw=5: uses 91 days excluding the last 28, predicts 5 weeks ahead (simulates data availability)

Each horizon has different optimal feature importances; independent models outperform a shared model with a `forecast_week` feature.

### 3. TemporalKFold

Walk-forward cross-validation with a purge gap to prevent leakage:

```
Fold 1: Train=[T1,T2,T3] → Val=[T4]  (purge_gap=13 steps)
Fold 2: Train=[T1..T4]   → Val=[T5]
Fold 3: Train=[T1..T5]   → Val=[T6]
Fold 4: Train=[T1..T6]   → Val=[T7]
```

Val is always temporally after Train. Purge gap = 13 steps (91d / 7d) eliminates sliding-window overlap.

### 4. Calendar-Matched Validation

Early stopping uses only validation samples whose calendar month matches the test period month. This makes the early-stopping signal representative of test conditions, reducing overfitting to out-of-season patterns.

### 5. Preprocessing Pipeline

| Step | Description |
|------|-------------|
| Per-region median imputation | Fill missing values using region-specific medians |
| Winsorization (p1/p99) | Cap extreme values for robustness |
| Log1p (prec) / sqrt (surf_pre) | Remove right skew |
| Rank normalization | Align train∪test pooled ECDF |

All artifacts are fit on train only and applied at inference time — no leakage.

### 6. Feature Engineering (~430 dimensions)

| Feature group | Description |
|---------------|-------------|
| Multi-scale rolling stats | 7/14/21/42/91d × mean/std/min/max + linear trend slope |
| Anomaly z-score | Current window vs. region long-term baseline |
| Monthly anomaly | Current 7d vs. same-month historical mean |
| Physical drought indices | VPD, dry-day ratio, precipitation deficit, heat degree-days, DTR, wind-dry index |
| Proxy score | p7/p21/p91/p_main + short–long differentials (3 variants) |
| Region score stats | mean/std/q25/q75/q90/nonzero rate, monthly means |
| Seasonal encoding | sin/cos(doy), sin/cos(month), quarter |
| Forecast week encoding | fw, sin/cos(fw) (constant per horizon in per-horizon mode) |

### 7. Sample Weighting

```
weight = 1 + 0.5·𝟙[y>0] + 3·𝟙[y≥3],  normalized to mean=1
```

Addresses class imbalance: score=0 accounts for ~62%, severe (≥3) for ~12%.

### 8. Gap-Stratified Models

Regions are split by train–test gap:
- **Short gap (<13 weeks):** ACF > 0.4, score lag features remain informative
- **Long gap (≥13 weeks):** ACF < 0.4, model relies primarily on meteorological features

Each stratum gets its own LightGBM ensemble; predictions are blended (soft or hard) based on the region's gap distance from the threshold.

### 9. Zero-Inflated Two-Stage Model

With ~59.6% zero scores, a two-stage approach is used:
- **Stage 1 (classifier):** P(score > 0) via LightGBM binary
- **Stage 2 (regressor):** E(score | score > 0) via LightGBM L1, trained on non-zero samples only

Final prediction = P(score > 0) × E(score | score > 0)

### 10. Ensemble Blending

```
final = lgbm_weight  × (gap_stratified_lgbm or main_lgbm)
      + xgb_weight   × xgboost          (if available)
      + zi_weight    × zero_inflated     (if available)
      + ptst_weight  × patchtst          (if available)
```

Default weights: XGB=0.20, ZI=0.15, PatchTST=0.20, LGBM=remainder (~0.45).

---

## Stage Caching

Each stage's output is persisted; already-existing artifacts are skipped automatically, allowing recovery from any stage after interruption.

| Stage | Output | Skip condition |
|-------|--------|----------------|
| 0 | `models/preprocessing.pkl` | File exists and not `--force` |
| 1 | `models/climatology.pkl` | File exists and not `--force` |
| 2 | `models/proxy_ridge.pkl` | File exists and not `--force` |
| 3 | `dataset/cache/train_features_<key8>.pkl` | SHA1 key matches (config + file mtime) |
| 4 | `models/per_horizon/lgbm_fw{1..5}.pkl` | Retrained every run |
| 5 | `models/calibrator.pkl` | Retrained every run |

**Cache key** = SHA1(feature-relevant config + train.csv size/mtime + test.csv size/mtime).  
Changing LGBM hyperparameters does not invalidate the cache; changing feature settings or data does.

---

## Directory Structure

```
drought/
├── dataset/
│   ├── data/
│   │   ├── train.csv              ← place here
│   │   ├── test.csv               ← place here
│   │   └── sample_submission.csv
│   └── cache/                     ← feature cache (auto-created)
├── code/
│   ├── config.py          Global settings (flags, hyperparameters, paths)
│   ├── logging_setup.py   Unified logging (console + file)
│   ├── cache.py           SHA1 feature cache
│   ├── preprocessing.py   Daily data preprocessing artifacts
│   ├── climatology.py     Per-region climatology baselines
│   ├── proxy.py           Proxy Score (meteorological → drought proxy)
│   ├── features.py        Feature engineering (91-day rolling window)
│   ├── temporal_cv.py     TemporalKFold implementation
│   ├── evaluation.py      Evaluation metrics
│   ├── data_pipeline.py   Parallel feature construction
│   ├── adversarial.py     Adversarial sample weighting
│   ├── model.py           LightGBM training / inference
│   ├── ensemble.py        XGBoost training / inference + blending
│   ├── gap_stratified.py  Gap-stratified LightGBM
│   ├── zero_inflated.py   Zero-inflated two-stage model
│   ├── patch_tst.py       PatchTST Transformer model
│   ├── train.py           Training pipeline (Stages 0–5)
│   └── predict.py         Inference pipeline
├── EDA/
│   ├── eda.py             Exploratory analysis
│   └── figures/           EDA figures (auto-created)
├── models/
│   └── per_horizon/       Per-horizon models (lgbm_fw1..5.pkl)
├── eval/                  Evaluation reports (eval_report.json)
├── logs/
├── submission.csv
└── README.md
```

---

## Installation

```bash
pip install lightgbm scikit-learn pandas numpy matplotlib scipy tqdm

# Optional: XGBoost ensemble
pip install xgboost>=2.0.0

# Optional: PatchTST deep learning
pip install torch>=2.0.0
```

---

## Usage

### Training

```bash
cd code

# Full training pipeline (Stages 0–5)
python train.py

# Resume from Stage 3 (feature rebuild)
python train.py --from-stage 3

# Retrain models only (cache already built)
python train.py --from-stage 4

# Force full rebuild (ignore all caches)
python train.py --force

# Skip PatchTST (when GPU is unavailable)
python train.py --skip-patchtst
```

### Inference

```bash
cd code

# Standard inference
python predict.py

# Custom output path
python predict.py --output /path/to/result.csv

# Force rebuild test features (ignore cache)
python predict.py --force-rebuild
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Primary competition metric (lower is better) |
| **RMSE** | More sensitive to large errors |
| **R²** | Explained variance (1=perfect, 0=mean prediction only) |
| **Bias** | Systematic over/under-prediction |
| **P95 abs error** | 95th percentile absolute error — tail performance |
| **MASE** | Relative to naïve baseline; <1 beats baseline |
| **MAE by score class** | Per-severity-level breakdown |
| **Per-region MAE** | Identifies hard-to-predict regions |

---

## Performance Settings

```python
# code/config.py

# Parallel feature engineering (default: half of CPU cores)
N_WORKERS = max(1, os.cpu_count() // 2)

# Reduce for faster debug runs
SAMPLE_FRAC        = 0.2
MAX_WIN_PER_REGION = 40

# GPU acceleration (LightGBM)
LGBM_DEVICE   = "gpu"
GPU_DEVICE_ID = 0

# Enable/disable model components
USE_XGBOOST       = True
USE_GAP_STRATIFIED = True
USE_ZERO_INFLATED  = True
USE_PATCHTST       = True
```
