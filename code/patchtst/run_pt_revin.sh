#!/usr/bin/env bash
# Train the RevIN-equipped PatchTST, dump val predictions, write the test
# submission. Mirrors run_pt.sh but targets the *_revin scripts so the existing
# patchtst_v2 checkpoint is left untouched.
#
# Pipeline (all three stages share one timestamp suffix on their log files):
#   1. torchrun --nproc_per_node=4 -m patchtst.train_pt_revin   (~2-4 h)
#         -> code/patchtst/models_pt/patchtst_revin.pt
#            code/patchtst/models_pt/metrics_revin.json
#   2. python -m patchtst.dump_val_preds_revin                  (~2 min, 1 GPU)
#         -> code/patchtst/models_pt/val_preds_patchtst_revin.npz
#   3. python -m patchtst.predict_pt_revin                      (~30 s, 1 GPU)
#         -> submission_patchtst_revin.csv  (at repo root)
#
# The script re-execs itself into a detached tmux session named "pt_revin" the
# first time it is invoked outside tmux, then attaches.
#
# Usage:
#   bash code/patchtst/run_pt_revin.sh                              # default submission name
#   bash code/patchtst/run_pt_revin.sh submission_revin_v1.csv      # custom name (CWD-relative)
#   bash code/patchtst/run_pt_revin.sh /abs/path/submission.csv     # absolute path honored verbatim
#
# tmux controls:
#   Detach:    Ctrl-b d
#   Reattach:  tmux attach -t pt_revin
#   Kill:      tmux kill-session -t pt_revin
set -euo pipefail

SCRIPT_PATH=$(realpath "$0")
SESSION=pt_revin

# First positional arg overrides PT_OUTPUT_PATH (env var is the fallback used
# when the outer invocation re-execs us inside tmux).
OUTPUT_PATH="${1:-${PT_OUTPUT_PATH:-}}"

# ---------------------------------------------------------------------------
# Outer invocation: re-exec into a detached tmux session, then attach.
# Inside-tmux work happens in the block below this conditional.
# ---------------------------------------------------------------------------
if [ -z "${TMUX:-}" ]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux session '$SESSION' already exists."
        echo "  attach with:  tmux attach -t $SESSION"
        echo "  or kill with: tmux kill-session -t $SESSION"
        exit 1
    fi
    # Forward OUTPUT_PATH to the inner shell via env var (tmux 2.6 has no
    # `new-session -e`, so prepend the assignment to the command).
    tmux new-session -d -s "$SESSION" \
        "PT_OUTPUT_PATH='$OUTPUT_PATH' bash '$SCRIPT_PATH'; echo; echo '=== finished. press any key to close ==='; read -n 1"
    exec tmux attach -t "$SESSION"
fi

# ---------------------------------------------------------------------------
# Inside tmux from here. Activate the conda env, set a shared timestamp,
# run the three pipeline stages back-to-back.
# ---------------------------------------------------------------------------

# Activate DMFP conda env (provides pytorch, catboost, lightgbm, etc.)
# shellcheck disable=SC1091
source /mnt/1stHDD/juiyun/miniforge3/etc/profile.d/conda.sh
conda activate DMFP

PY=/mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/python
TR=/mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/torchrun

# cd to code/ so the `python -m patchtst.<module>` imports resolve correctly
# and the relative log paths land under code/patchtst/logs_pt/.
cd "$(dirname "$SCRIPT_PATH")/.."

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p patchtst/logs_pt
TRAIN_LOG=patchtst/logs_pt/train_revin_${TS}.log
DUMP_LOG=patchtst/logs_pt/dump_val_preds_revin_${TS}.log
PRED_LOG=patchtst/logs_pt/predict_revin_${TS}.log

# PYTHONUNBUFFERED=1 forces line-buffered stdout so `tee` (and an external
# `tail -f`) sees output immediately rather than in 4 KB chunks. Without it,
# training would look frozen for minutes between epoch prints.
export PYTHONUNBUFFERED=1

# ---- Stage 1: training (4 GPUs, DDP via torchrun) ------------------------
# --standalone       : single-node rendezvous (no master_addr needed)
# --nproc_per_node=4 : 4 worker processes (1 per GPU on this machine)
# `; true` after the pipe keeps the script alive even if training crashes,
# so the DUMP/PREDICT markers still fire and the failure shows in the log.
echo "==> [1/3] training (4 GPUs via torchrun)  log=$TRAIN_LOG"
$TR --standalone --nproc_per_node=4 -m patchtst.train_pt_revin 2>&1 | tee "$TRAIN_LOG" || true
echo "=== TRAIN_DONE ==="

# ---- Stage 2: dump val predictions from the best checkpoint --------------
# Reads code/patchtst/models_pt/patchtst_revin.pt + cached daily_train.pkl
# and writes val_preds_patchtst_revin.npz in the same schema as the LGBM
# and CatBoost val_preds files so the ensemble can intersect on
# (region_id, week_idx).
echo "==> [2/3] dumping val_preds_patchtst_revin.npz  log=$DUMP_LOG"
$PY -m patchtst.dump_val_preds_revin 2>&1 | tee "$DUMP_LOG" || true
echo "=== DUMP_DONE ==="

# ---- Stage 3: test inference -> submission CSV ---------------------------
# Honors --output if a path was passed; otherwise writes to the default
# submission_patchtst_revin.csv at the repo root.
echo "==> [3/3] writing submission CSV  log=$PRED_LOG  output=${OUTPUT_PATH:-<default>}"
if [ -n "$OUTPUT_PATH" ]; then
    $PY -m patchtst.predict_pt_revin --output "$OUTPUT_PATH" 2>&1 | tee "$PRED_LOG" || true
else
    $PY -m patchtst.predict_pt_revin 2>&1 | tee "$PRED_LOG" || true
fi
echo "=== PREDICT_DONE ==="

# Summary line so when you attach later, the final scroll position tells you
# where to look. The outer `read -n 1` keeps the session alive until dismissed.
echo
echo "==> finished. artifacts:"
echo "    checkpoint: code/patchtst/models_pt/patchtst_revin.pt"
echo "    metrics:    code/patchtst/models_pt/metrics_revin.json"
echo "    val_preds:  code/patchtst/models_pt/val_preds_patchtst_revin.npz"
echo "    submission: ${OUTPUT_PATH:-$(pwd)/../submission_patchtst_revin.csv}"
