"""DLinear regressor (Zeng et al. 2023) adapted to DMFP meteo -> 5-week drought score.

Why DLinear for this problem: it is LINEAR, so it can EXTRAPOLATE the +6C out-of-support
shift (trees clip to the training boundary; PatchTST mis-extrapolated). The series is split
into a moving-average TREND + SEASONAL residual; each gets a shared per-channel Linear(L->H),
the channels are combined with learnable weights, and a small region/calendar/side term sets
the level. Tiny + simple => generalizes (the "fit less, transfer better" lesson), and its
linear-decomposition math is decorrelated from the LGBM / physics tracks.

Input/output interface matches PatchTSTRegressor so it reuses dataset_pt + the training loop:
  forward(x_daily (B,M,L), x_side (B,S), x_cal (B,4), region_idx (B,)) -> (B,H)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _MovingAvg(nn.Module):
    """Causal-ish moving average for series decomposition (DLinear), edge-padded."""
    def __init__(self, kernel: int):
        super().__init__()
        self.k = kernel
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, M, L)
        front = x[:, :, :1].repeat(1, 1, (self.k - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, self.k // 2)
        return self.avg(torch.cat([front, x, end], dim=-1))


class DLinearRegressor(nn.Module):
    def __init__(self, n_channels: int, lookback: int, n_horizons: int, n_regions: int,
                 region_emb_dim: int, side_dim: int, calendar_dim: int,
                 kernel: int = 25, dropout: float = 0.1, individual: bool = False):
        super().__init__()
        self.decomp = _MovingAvg(kernel)
        self.individual = individual
        if individual:
            # per-channel (M,H,L) weights — each channel its own L->H map (classic DLinear individual)
            self.wt = nn.Parameter(torch.randn(n_channels, n_horizons, lookback) * (1.0 / lookback))
            self.ws = nn.Parameter(torch.randn(n_channels, n_horizons, lookback) * (1.0 / lookback))
            self.bt = nn.Parameter(torch.zeros(n_channels, n_horizons))
        else:
            # shared (channel-independent) linear maps L -> H for trend + seasonal
            self.lin_trend = nn.Linear(lookback, n_horizons)
            self.lin_season = nn.Linear(lookback, n_horizons)
        # learnable per-channel combine weights (B,M,H) -> (B,H)
        self.chan_w = nn.Parameter(torch.ones(n_channels) / n_channels)
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        nn.init.trunc_normal_(self.region_emb.weight, std=0.02)
        # small additive level/season term from side inputs (region + calendar + score-lag)
        self.side = nn.Sequential(
            nn.Linear(side_dim + calendar_dim + region_emb_dim, 64),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, n_horizons),
        )

    def forward(self, x_daily, x_side, x_cal, region_idx):
        trend = self.decomp(x_daily)
        season = x_daily - trend
        if self.individual:
            t = torch.einsum("bml,mhl->bmh", trend, self.wt) + self.bt
            s = torch.einsum("bml,mhl->bmh", season, self.ws)
        else:
            t = self.lin_trend(trend)      # (B, M, H)
            s = self.lin_season(season)    # (B, M, H)
        y = (t + s) * self.chan_w.view(1, -1, 1)
        y = y.sum(dim=1)               # (B, H) — linear combine over channels
        r = self.region_emb(region_idx)
        adj = self.side(torch.cat([x_side, x_cal, r], dim=-1))
        return y + adj


def build_dlinear_model_from_config(n_regions: int) -> DLinearRegressor:
    import os
    from patchtst.config_pt import (
        LOOKBACK_DAYS, N_CHANNELS, HORIZONS, REGION_EMB_DIM, SCORE_LAG_OFFSETS, DROPOUT,
    )
    kernel = int(os.environ.get("DLIN_KERNEL", 25))            # decomposition window (variant knob)
    individual = os.environ.get("DLIN_INDIVIDUAL", "0") == "1"  # per-channel vs shared linear
    side_dim = len(SCORE_LAG_OFFSETS) + 2
    return DLinearRegressor(
        n_channels=N_CHANNELS, lookback=LOOKBACK_DAYS, n_horizons=HORIZONS,
        n_regions=n_regions, region_emb_dim=REGION_EMB_DIM,
        side_dim=side_dim, calendar_dim=4, kernel=kernel, dropout=DROPOUT, individual=individual,
    )
