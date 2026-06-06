#!/usr/bin/env bash
# End-to-end reproduction: TRAIN every component from scratch, then blend.
#
#   QUICK (default) : DLinear 3 members x few epochs + LightGBM ~200 trees/horizon  (~30-40 min)
#   FULL=1          : DLinear 7 members x full epochs + LightGBM 8000 trees/horizon (several hours)
#
# Only the teammate's component is not retrained (reproduce/frozen/submission_teammate_082.csv);
# it comes from her separate repo. Outputs land in reproduce/out/.
#
#   bash reproduce/run.sh            # quick
#   FULL=1 bash reproduce/run.sh     # full
set -euo pipefail
PY=/mnt/1stHDD/juiyun/miniforge3/envs/DMFP/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
mkdir -p "$HERE/out"

FULL="${FULL:-0}"
if [ "$FULL" = "1" ]; then LGBM_N_EST="${LGBM_N_EST:-8000}"; else LGBM_N_EST="${LGBM_N_EST:-200}"; fi
export LGBM_N_EST
echo "============================================================"
echo " Reproduce (train-from-scratch)  FULL=$FULL  LGBM_N_EST=$LGBM_N_EST"
echo "============================================================"

echo ""; echo "[0/4] Prepare dirs + build daily_train.pkl if missing ..."
mkdir -p "$REPO/lgbm/models" "$REPO/lgbm/diagnostics" "$REPO/dlinear/patchtst/models_pt"
$PY -u "$HERE/build_daily.py"

echo ""; echo "[1/4] Train DLinear members ..."
FULL="$FULL" bash "$HERE/train_dlinear.sh"

echo ""; echo "[2/4] Predict DLinear -> base_dlinear.csv + dlinear_ens_shared7.csv ..."
$PY -u "$HERE/regen_dlinear.py"

echo ""; echo "[3/4] Train + predict LightGBM (phase1b) -> out/phase1b.csv ..."
( cd "$REPO/lgbm" && $PY -u train.py )
( cd "$REPO/lgbm" && $PY -u predict.py -o "$HERE/out/phase1b.csv" )

echo ""; echo "[4/4] Blend -> two submissions ..."
$PY -u "$HERE/blend.py"

echo ""; echo "Done. Submissions in $HERE/out/"
