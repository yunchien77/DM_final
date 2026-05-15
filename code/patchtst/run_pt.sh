#!/usr/bin/env bash
# Train PatchTST on 4 GPUs via torchrun, then run single-GPU inference.
# Re-execs itself inside a detached tmux session named "patchtst" the first
# time it is invoked outside tmux, then attaches.
#
# Detach: Ctrl-b d.   Reattach: tmux attach -t patchtst.   Kill: tmux kill-session -t patchtst.
set -euo pipefail

SCRIPT_PATH=$(realpath "$0")
SESSION=patchtst

if [ -z "${TMUX:-}" ]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux session '$SESSION' already exists."
        echo "  attach with:  tmux attach -t $SESSION"
        echo "  or kill with: tmux kill-session -t $SESSION"
        exit 1
    fi
    tmux new-session -d -s "$SESSION" \
        "bash '$SCRIPT_PATH'; echo; echo '=== finished. press any key to close ==='; read -n 1"
    exec tmux attach -t "$SESSION"
fi

# ---- Inside tmux: actual work ------------------------------------------
PY=/mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/python
TR=/mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/torchrun

cd "$(dirname "$SCRIPT_PATH")/.."   # -> code/

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p patchtst/logs_pt
TRAIN_LOG=patchtst/logs_pt/train_${TS}.log
PRED_LOG=patchtst/logs_pt/predict_${TS}.log

echo "==> training (4 GPUs via torchrun)  log=$TRAIN_LOG"
$TR --standalone --nproc_per_node=4 -m patchtst.train_pt 2>&1 | tee "$TRAIN_LOG"

echo "==> predicting (single GPU)  log=$PRED_LOG"
$PY -m patchtst.predict_pt 2>&1 | tee "$PRED_LOG"

echo "==> done. submission_patchtst.csv at $(pwd)/../submission_patchtst.csv"
