#!/bin/bash
set -e

cd "$(dirname "$0")/code"

echo "=========================================="
echo " Natural Disaster Severity Prediction"
echo "=========================================="

echo ""
echo "[TRAIN]"
python train.py

echo ""
echo "[PREDICT]"
python predict.py

echo ""
echo "Done. Submission saved to ../submission.csv"
