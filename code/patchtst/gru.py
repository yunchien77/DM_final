"""GRU regressor adapted to DMFP meteo -> 5-week drought score.

A recurrent inductive bias — different from the LGBM (trees), DLinear (linear), and PatchTST
(attention) tracks, so potentially DECORRELATED. RISK: it is nonlinear/deep, so like PatchTST
it may mis-extrapolate the +6C out-of-support shift (DLinear transferred BECAUSE it is linear).
Mitigations carried over from what made DLinear work: global-normed input (+6C -> offset),
representative random val (not the mild 0.58 slice), SMALL capacity (low overfit => better transfer).

Interface matches PatchTSTRegressor / DLinearRegressor so it reuses dataset_pt + the loop:
  forward(x_daily (B,M,L), x_side (B,S), x_cal (B,4), region_idx (B,)) -> (B,H)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GRURegressor(nn.Module):
    def __init__(self, n_channels: int, n_horizons: int, n_regions: int,
                 region_emb_dim: int, side_dim: int, calendar_dim: int,
                 hidden: int = 96, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=n_channels, hidden_size=hidden, num_layers=layers,
                          batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        nn.init.trunc_normal_(self.region_emb.weight, std=0.02)
        self.head = nn.Sequential(
            nn.Linear(hidden + side_dim + calendar_dim + region_emb_dim, 128),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_horizons),
        )

    def forward(self, x_daily, x_side, x_cal, region_idx):
        x = x_daily.transpose(1, 2)        # (B, M, L) -> (B, L, M): sequence of L weeks
        out, _ = self.gru(x)               # (B, L, hidden)
        last = out[:, -1, :]               # final-step hidden state (B, hidden)
        r = self.region_emb(region_idx)
        return self.head(torch.cat([last, x_side, x_cal, r], dim=-1))


def build_gru_model_from_config(n_regions: int) -> GRURegressor:
    import os
    from patchtst.config_pt import (
        N_CHANNELS, HORIZONS, REGION_EMB_DIM, SCORE_LAG_OFFSETS, DROPOUT,
    )
    hidden = int(os.environ.get("GRU_HIDDEN", 96))
    layers = int(os.environ.get("GRU_LAYERS", 2))
    side_dim = len(SCORE_LAG_OFFSETS) + 2
    return GRURegressor(
        n_channels=N_CHANNELS, n_horizons=HORIZONS, n_regions=n_regions,
        region_emb_dim=REGION_EMB_DIM, side_dim=side_dim, calendar_dim=4,
        hidden=hidden, layers=layers, dropout=DROPOUT,
    )
