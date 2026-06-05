"""Linear State-Space Model (diagonal LRU-style) for DMFP meteo -> 5-week drought score.

A LINEAR state recurrence h_t = a (.) h_{t-1} + B x_t with learnable per-dim decay a in (0,1).
The final state is the closed-form exponentially-weighted sum  h_L[i] = sum_t a_i^(L-1-t) (B x_t)[i],
computed as one einsum (no scan loop). The meteo->output path is FULLY LINEAR, so like DLinear it
EXTRAPOLATES the +6C shift (trees clip; GRU/Mamba mis-extrapolate). The ONLY nonlinearity is on the
auxiliary region/calendar SIDE head (sets the level, not the meteo extrapolation), mirroring DLinear.

Why it might be DECORRELATED from DLinear despite both being linear: DLinear uses a fixed full-window
trend/seasonal projection per channel; this SSM uses LEARNABLE MULTI-TIMESCALE exponential pooling +
cross-channel mixing (B). Different temporal inductive bias -> the shot at residual-corr < 0.55.

Interface matches DLinear/GRU so it reuses dataset_pt + the loop:
  forward(x_daily (B,M,L), x_side (B,S), x_cal (B,4), region_idx (B,)) -> (B,H)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearSSM(nn.Module):
    def __init__(self, n_channels: int, lookback: int, n_horizons: int, n_regions: int,
                 region_emb_dim: int, side_dim: int, calendar_dim: int,
                 state_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.L = lookback
        self.B = nn.Linear(n_channels, state_dim, bias=False)   # input projection (channel mixing)
        # learnable per-dim decay a = sigmoid(a_raw); init spans timescales a in ~[0.5, 0.98]
        a0 = torch.linspace(0.0, 4.0, state_dim)                # sigmoid(0)=0.5 ... sigmoid(4)=0.982
        self.a_raw = nn.Parameter(a0)
        self.C = nn.Linear(state_dim, n_horizons)               # linear readout of the final state
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        nn.init.trunc_normal_(self.region_emb.weight, std=0.02)
        self.side = nn.Sequential(
            nn.Linear(side_dim + calendar_dim + region_emb_dim, 64),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, n_horizons),
        )

    def forward(self, x_daily, x_side, x_cal, region_idx):
        x = x_daily.transpose(1, 2)            # (B, L, M)
        xb = self.B(x)                         # (B, L, d)
        a = torch.sigmoid(self.a_raw)          # (d,) in (0,1)
        exps = torch.arange(self.L - 1, -1, -1, device=x.device, dtype=x.dtype)  # (L,): L-1..0
        kernel = a.unsqueeze(0) ** exps.unsqueeze(1)            # (L, d): a_i^(L-1-t), recency-weighted
        h = torch.einsum("btd,td->bd", xb, kernel)             # (B, d) final state (linear in input)
        out = self.C(h)                                        # (B, H) linear readout
        r = self.region_emb(region_idx)
        return out + self.side(torch.cat([x_side, x_cal, r], dim=-1))


def build_ssm_model_from_config(n_regions: int) -> LinearSSM:
    import os
    from patchtst.config_pt import (
        LOOKBACK_DAYS, N_CHANNELS, HORIZONS, REGION_EMB_DIM, SCORE_LAG_OFFSETS, DROPOUT,
    )
    state_dim = int(os.environ.get("SSM_STATE", 64))
    side_dim = len(SCORE_LAG_OFFSETS) + 2
    return LinearSSM(
        n_channels=N_CHANNELS, lookback=LOOKBACK_DAYS, n_horizons=HORIZONS, n_regions=n_regions,
        region_emb_dim=REGION_EMB_DIM, side_dim=side_dim, calendar_dim=4,
        state_dim=state_dim, dropout=DROPOUT,
    )
