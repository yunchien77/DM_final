#!/usr/bin/env bash
# One-command reproduction entry point. Trains every component from scratch and blends.
#
#   ./reproduce.sh                      # QUICK  (~30-40 min): smoke-scale training
#   FULL=1 ./reproduce.sh               # FULL   (~4-5 h): faithful run, LightGBM 8000 trees
#   FULL=1 LGBM_N_EST=3000 ./reproduce.sh   # FULL but faster (~4.3 h): reproduces 0.7862 to within 0.004
#
# Outputs -> reproduce/out/submission_blend_{ENS,TDP}_354025.csv
exec bash "$(dirname "$0")/reproduce/run.sh" "$@"
