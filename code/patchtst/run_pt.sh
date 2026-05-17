#!/usr/bin/env bash
# Train PatchTST on 4 GPUs via torchrun, then run single-GPU inference.
# Re-execs itself inside a detached tmux session named "patchtst" the first
# time it is invoked outside tmux, then attaches.
#
# Usage:
#   bash run_pt.sh                              # default: code/../submission_patchtst.csv
#   bash run_pt.sh submission_v2.csv            # custom name (CWD-relative)
#   bash run_pt.sh /abs/path/submission.csv     # absolute path honored verbatim
#
# Detach: Ctrl-b d.   Reattach: tmux attach -t patchtst.
# Kill:   tmux kill-session -t patchtst.
set -euo pipefail

SCRIPT_PATH=$(realpath "$0")
SESSION=patchtst
# First positional arg overrides PT_OUTPUT_PATH; env var is the fallback
# (set when the outer invocation re-execs us inside tmux).
OUTPUT_PATH="${1:-${PT_OUTPUT_PATH:-}}"

if [ -z "${TMUX:-}" ]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux session '$SESSION' already exists."
        echo "  attach with:  tmux attach -t $SESSION"
        echo "  or kill with: tmux kill-session -t $SESSION"
        exit 1
    fi
    # Forward OUTPUT_PATH to the inner shell via env var. tmux 2.6 does not
    # support `new-session -e`, so we prepend the assignment to the command.
    tmux new-session -d -s "$SESSION" \
        "PT_OUTPUT_PATH='$OUTPUT_PATH' bash '$SCRIPT_PATH'; echo; echo '=== finished. press any key to close ==='; read -n 1"
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

echo "==> predicting (single GPU)  log=$PRED_LOG  output=${OUTPUT_PATH:-<default>}"
if [ -n "$OUTPUT_PATH" ]; then
    $PY -m patchtst.predict_pt --output "$OUTPUT_PATH" 2>&1 | tee "$PRED_LOG"
else
    $PY -m patchtst.predict_pt 2>&1 | tee "$PRED_LOG"
fi

echo "==> done. submission at ${OUTPUT_PATH:-$(pwd)/../submission_patchtst.csv}"
