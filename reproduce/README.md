# reproduce/ — train-from-scratch pipeline

Trains every component from scratch and blends them into the two submissions. See the top-level
`README.md` for the full description; quick reference:

```bash
bash reproduce/run.sh          # quick mode (~30–40 min)
FULL=1 bash reproduce/run.sh   # full mode (several hours)
```

Steps (orchestrated by `run.sh`):

1. `train_dlinear.sh` — trains DLinear member(s) (`dlinear/patchtst/train_pt_dlinear.py`); writes
   checkpoints to `dlinear/patchtst/models_pt/`. Quick: 3 members × few epochs. Full: 7 members.
2. `regen_dlinear.py` — predicts the trained members → `out/base_dlinear.csv` (single, for TDP)
   and `out/dlinear_ens_shared7.csv` (member mean, for ENS).
3. `lgbm/train.py` + `predict.py` — trains the per-horizon phase1b LightGBM from scratch and
   writes `out/phase1b.csv`. Iterations via `LGBM_N_EST` (quick 200, full 8000).
4. `blend.py` — `shrink × (0.40·DLinear + 0.35·teammate + 0.25·phase1b)`. The teammate leg is the
   fixed file in `frozen/`; everything else is freshly trained.

Recipe (verified): weights normalise to exactly 0.40/0.35/0.25 (D/teammate/phase1b); shrink =
0.975092816675869 (TDP) and 0.9694903458056041 (ENS). A run is a fresh re-training of the recipe,
not a bit-identical copy of the original submissions.

Knobs: `FULL=1`, `DLIN_EPOCHS=<n>`, `LGBM_N_EST=<n>`.
