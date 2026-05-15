"""PatchTST track configuration.

Parallel to the L1 LightGBM pipeline. Reuses the daily->weekly aggregation
from data_pipeline.py but feeds the weekly stat matrix directly to a
channel-independent Transformer instead of flattening into lag features.

All paths land under code/patchtst/models_pt/ and code/patchtst/logs_pt/
to keep artifacts isolated from the LGBM track.
"""
from pathlib import Path

PT_DIR = Path(__file__).parent
PT_MODELS_DIR = PT_DIR / "models_pt"
PT_LOGS_DIR = PT_DIR / "logs_pt"
PT_MODELS_DIR.mkdir(exist_ok=True)
PT_LOGS_DIR.mkdir(exist_ok=True)

PT_SUBMISSION_PATH = PT_DIR.parent.parent / "submission_patchtst.csv"

# --- Input shape ---------------------------------------------------------
# Channels = 14 meteo features x 5 weekly stats = 70.
# We do NOT include the score history as a channel: at test time we have
# 13 unobserved score weeks at the tail, which would create train/test drift.
LOOKBACK_WEEKS = 52          # ~1 year of weekly stats
PATCH_LEN = 4                # 4 weeks per patch (~1 month)
PATCH_STRIDE = 2             # 50% overlap

HORIZONS = 5

# --- Encoder -------------------------------------------------------------
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 128
DROPOUT = 0.2

# Region embedding concatenated to the channel-pooled representation
# before the regression head. Adds per-region prior on top of meteo signal.
REGION_EMB_DIM = 16

# --- Training ------------------------------------------------------------
BATCH_SIZE = 256
NUM_WORKERS_LOADER = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
WARMUP_EPOCHS = 2
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 6
SEED = 42

# Subsample majority-class anchors (all 5 future scores == 0) to this fraction
# of their original count. Non-zero anchors are kept in full.
MAJORITY_KEEP_FRAC = 1.0

# Sample weighting on top of the L1 loss. weight = 1 + ALPHA*1[max_y>0] + BETA*1[max_y>=3]
# Mirrors the LGBM weighting scheme so subsampling and weighting compound.
SAMPLE_WEIGHT_ALPHA = 0.5    # lower than LGBM's 1.0 because subsampling already shifts the mix
SAMPLE_WEIGHT_BETA = 1.5

# --- Validation ----------------------------------------------------------
# Match the LGBM track: last 26 weeks per region are the validation holdout.
VALID_WEEKS = 26

# --- Saved artifacts -----------------------------------------------------
PT_MODEL_PATH = PT_MODELS_DIR / "patchtst.pt"
PT_NORM_PATH = PT_MODELS_DIR / "channel_norm.npz"   # mean/std per channel from train fold
PT_REGION_MAP_PATH = PT_MODELS_DIR / "region_id_map.json"
PT_METRICS_PATH = PT_MODELS_DIR / "metrics.json"
