#!/usr/bin/env bash
# Train LGBM (5 horizons, L1) then run inference. Re-execs itself inside a
# detached tmux session named "lgbm" the first time it is invoked outside
# tmux, then attaches.
#
# Usage:
#   bash run.sh                              # default: ../submission.csv (per config.SUBMISSION_PATH)
#   bash run.sh submission_v2.csv            # custom name (CWD-relative)
#   bash run.sh /abs/path/submission.csv     # absolute path honored verbatim
#
# Detach: Ctrl-b d.   Reattach: tmux attach -t lgbm.
# Kill:   tmux kill-session -t lgbm.
set -euo pipefail

SCRIPT_PATH=$(realpath "$0")
SESSION=lgbm
# First positional arg overrides LGBM_OUTPUT_PATH; env var is the fallback
# (set when the outer invocation re-execs us inside tmux).
OUTPUT_PATH="${1:-${LGBM_OUTPUT_PATH:-}}"

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
        "LGBM_OUTPUT_PATH='$OUTPUT_PATH' bash '$SCRIPT_PATH'; echo; echo '=== finished. press any key to close ==='; read -n 1"
    exec tmux attach -t "$SESSION"
fi

# ---- Inside tmux: actual work ------------------------------------------
PY=/mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/python

cd "$(dirname "$SCRIPT_PATH")"   # -> code/

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
TRAIN_LOG=logs/train_${TS}.log
ANALYSIS_LOG=logs/analysis_${TS}.log
PRED_LOG=logs/predict_${TS}.log

echo "==> training (single process)  log=$TRAIN_LOG"
$PY -m train 2>&1 | tee "$TRAIN_LOG"

echo "==> analysis  log=$ANALYSIS_LOG"
$PY -m analysis 2>&1 | tee "$ANALYSIS_LOG"

echo "==> predicting  log=$PRED_LOG  output=${OUTPUT_PATH:-<default>}"
if [ -n "$OUTPUT_PATH" ]; then
    $PY -m predict --output "$OUTPUT_PATH" 2>&1 | tee "$PRED_LOG"
else
    $PY -m predict 2>&1 | tee "$PRED_LOG"
fi

echo "==> done. submission at ${OUTPUT_PATH:-$(pwd)/../submission.csv}"
echo "==> feature analysis at $(pwd)/diagnostics/feature_analysis.md"
