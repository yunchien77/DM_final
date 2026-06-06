#!/usr/bin/env bash
# Train the DLinear members from scratch (weights initialised randomly, trained on the
# preprocessed daily history). Writes checkpoints to code/patchtst/models_pt/.
#
#   QUICK (default): 3 members, few epochs    -> fast, runnable demonstration
#   FULL=1         : all 7 members, full epochs -> the real ensemble
#
# Knobs:  FULL=1  DLIN_EPOCHS=<n>   bash reproduce/train_dlinear.sh
set -euo pipefail
PY=/mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/python
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO/dlinear"

FULL="${FULL:-0}"
if [ "$FULL" = "1" ]; then
    EPOCHS="${DLIN_EPOCHS:-100}"
    MEMBERS="base s11 s22 s33 s44 k13 k101"
else
    EPOCHS="${DLIN_EPOCHS:-3}"
    MEMBERS="base s11 k13"          # base + one seed + one kernel variant
fi
echo "[train_dlinear] FULL=$FULL  epochs=$EPOCHS  members: $MEMBERS"

for m in $MEMBERS; do
    echo "[train_dlinear] === member: $m ==="
    case "$m" in
        base) env DLIN_EPOCHS="$EPOCHS"                                  $PY -u -m patchtst.train_pt_dlinear ;;
        s11)  env DLIN_EPOCHS="$EPOCHS" DLIN_SEED=11                     $PY -u -m patchtst.train_pt_dlinear ;;
        s22)  env DLIN_EPOCHS="$EPOCHS" DLIN_SEED=22                     $PY -u -m patchtst.train_pt_dlinear ;;
        s33)  env DLIN_EPOCHS="$EPOCHS" DLIN_SEED=33                     $PY -u -m patchtst.train_pt_dlinear ;;
        s44)  env DLIN_EPOCHS="$EPOCHS" DLIN_SEED=44                     $PY -u -m patchtst.train_pt_dlinear ;;
        k13)  env DLIN_EPOCHS="$EPOCHS" DLIN_SUF=_k13  DLIN_KERNEL=13    $PY -u -m patchtst.train_pt_dlinear ;;
        k101) env DLIN_EPOCHS="$EPOCHS" DLIN_SUF=_k101 DLIN_KERNEL=101   $PY -u -m patchtst.train_pt_dlinear ;;
    esac
done
echo "[train_dlinear] done."
