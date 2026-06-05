#!/usr/bin/env bash
# Usage: run_team.sh <work_dir> <output_submission> <tag>
set -o pipefail
PY=/mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/python
WORK="$1"; OUT="$2"; TAG="$3"
cd "$WORK/code" || exit 2
mkdir -p ../logs
echo "[$(date)] [$TAG] TRAIN start (cwd=$PWD)"
$PY -u train.py --from-stage 0 2>&1 | tee "../logs/${TAG}_train.log"
rc=${PIPESTATUS[0]}; if [ "$rc" -ne 0 ]; then echo "[$TAG] TRAIN FAILED rc=$rc"; exit "$rc"; fi
echo "[$(date)] [$TAG] PREDICT start"
$PY -u predict.py -o "$OUT" 2>&1 | tee "../logs/${TAG}_predict.log"
rc=${PIPESTATUS[0]}; if [ "$rc" -ne 0 ]; then echo "[$TAG] PREDICT FAILED rc=$rc"; exit "$rc"; fi
echo "[$(date)] [$TAG] DONE -> $OUT"
