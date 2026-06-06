"""PatchTST track configuration — daily-resolution refactor.

Phase 4 redesign that incorporates the Phase 1b insights:
  - Daily-resolution input matching the test window exactly (91 days × 9
    meteo channels). The prior weekly-stats version (LOOKBACK_WEEKS=52 of
    weekly aggregates) hit Kaggle 0.9769 because it inherited LGBM's
    aggregation losses and excluded score history entirely.
  - Score-lag side inputs at offsets [1, 2, 4, 8, 52] shifted by
    SCORE_LAG_TEST_GAP_SHIFT (=12 weeks) so the train-time
    (lag-source → first-target) distance equals test's 14 weeks.
  - Channel-mixing layer (post-encoder Linear over the channel dim)
    enables cross-feature interactions the original channel-independent
    PatchTST couldn't learn.
  - Calendar-matched validation per region (mirrors the LGBM track).
  - Severity sample weights re-enabled (ALPHA=1.0, BETA=2.0).

All paths land under code/patchtst/models_pt/ and code/patchtst/logs_pt/
to keep artifacts isolated from the LGBM track.
"""
import os
from pathlib import Path

PT_DIR = Path(__file__).parent
PT_MODELS_DIR = PT_DIR / "models_pt"
PT_LOGS_DIR = PT_DIR / "logs_pt"
PT_MODELS_DIR.mkdir(exist_ok=True)
PT_LOGS_DIR.mkdir(exist_ok=True)

PT_SUBMISSION_PATH = PT_DIR.parent.parent / "submission_patchtst.csv"

# --- Input shape ---------------------------------------------------------
# Phase 13: switch from 91-day daily to 260-week (5-year) WEEKLY aggregated
# input. Each timestep is the mean of 7 daily values, and the input sequence
# spans 5 years of per-region history (concatenating train weekly history
# with the test 13 weeks at inference). This gives PatchTST multi-year
# context the LGBM stack can't see, maximizing ensemble diversity.
#
# Variable names kept as `LOOKBACK_DAYS` / `PATCH_LEN_DAYS` /
# `PATCH_STRIDE_DAYS` to minimize code churn, but values are now WEEKS.
LOOKBACK_DAYS = 260      # WEEKS (5 years × 52)
# 9 weekly means + 3 surf_pre dynamics (drop/climb/std) + 2 prec extras
# (max, dry-day-count) + 1 tmp_max peak. The 6 extra channels preserve the
# within-week shock signal that pure weekly-mean aggregation erases — the
# Phase 8 diagnostic identified surf_pre_drop_3d_max (ρ=−0.61 with score)
# as the strongest single signal.
N_CHANNELS = 15
PATCH_LEN_DAYS = 4       # 4-week patches (~1 month each)
PATCH_STRIDE_DAYS = 4    # non-overlapping → 66 patches per channel

HORIZONS = 5

# --- Encoder -------------------------------------------------------------
D_MODEL = 96
N_HEADS = 4
N_LAYERS = 3
D_FF = 192
DROPOUT = 0.2

# Channel-mixing layer applied AFTER the channel-independent encoder.
# Without this the model can't learn interactions like tmp_max × humidity
# that LGBM picks up natively. Enabled by default in the refactor.
USE_CHANNEL_MIXING = True
CHANNEL_MIX_HIDDEN = 64  # bottleneck dim for the inter-channel Linear

# --- Side inputs --------------------------------------------------------
# Score-lag features (staleness-shifted) — Phase 8 retrospective showed the
# train↔test gap of ~163 weeks makes the lag autocorrelation 0.96^163 ≈ 0.001.
# The lag is informative on val (where we shift by only 14 weeks) but noise
# on test. Dropping it in LGBM aligned val and test better; PatchTST should
# get the same treatment. When False, dataset_pt returns zero side-input
# vectors so the architecture (and saved models) stay shape-compatible.
USE_SCORE_LAG_SIDE_INPUTS = False
SCORE_LAG_OFFSETS = [1, 2, 4, 8, 52]
SCORE_LAG_TEST_GAP_SHIFT = 12

# Calendar features of the anchor day fed into the head.
USE_CALENDAR_SIDE_INPUTS = True

REGION_EMB_DIM = 16

# --- Training ------------------------------------------------------------
BATCH_SIZE = 256
NUM_WORKERS_LOADER = 4
LR = 1e-3
WEIGHT_DECAY = 3e-5
EPOCHS = int(os.environ.get("DLIN_EPOCHS", "100"))   # set DLIN_EPOCHS=N for quick runs
WARMUP_EPOCHS = 2
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 10
SEED = 42

# Subsample majority-class anchors (all 5 future scores == 0) to this
# fraction of their original count. With ~60% zeros in train, keeping all
# at full weight floods the gradient with easy zero rows — we drop to 50%
# and rely on severity weights to lift the rare-event signal.
MAJORITY_KEEP_FRAC = 1.0

# Severity sample weighting on top of L1. weight = 1 + ALPHA*1[max_y>0] + BETA*1[max_y>=3]
# Matches the LGBM track. The previous run had both at 0 — that's one of
# the reasons the original PatchTST collapsed on extreme events.
SAMPLE_WEIGHT_ALPHA = 1.0
SAMPLE_WEIGHT_BETA = 2.0

# --- Validation ----------------------------------------------------------
# Calendar-matched per-region: hold out anchors whose woy is within
# CALENDAR_SLACK_WEEKS of the region's test_end_woy in the last training
# year. Mirrors the LGBM Phase 1 split so val MAE is directly comparable.
USE_CALENDAR_MATCHED_VAL = True
CALENDAR_SLACK_WEEKS = 6
CALENDAR_LAST_YEAR_ONLY = True

# --- Saved artifacts -----------------------------------------------------
PT_MODEL_PATH = PT_MODELS_DIR / "patchtst.pt"
PT_NORM_PATH = PT_MODELS_DIR / "channel_norm.npz"   # mean/std per daily channel
PT_REGION_MAP_PATH = PT_MODELS_DIR / "region_id_map.json"
PT_METRICS_PATH = PT_MODELS_DIR / "metrics.json"
PT_TRAIN_PKL = PT_MODELS_DIR / "daily_train.pkl"   # cached per-region daily arrays
