"""Autoformer-encoder regressor (Wu et al. 2021) adapted to DMFP meteo -> 5-week drought score.

Autoformer = series decomposition (trend/seasonal) + Auto-Correlation (FFT period-based aggregation,
replacing self-attention). RISK: it is a DEEP NONLINEAR transformer (PatchTST family, which failed
0.95 / RevIN 1.23) and high-capacity (GRU-style overfit risk); its Auto-Correlation keys on PERIODICITY,
not the +6C LEVEL shift that is the real problem. Mitigations: (1) a LINEAR trend path (mean of the
moving-average trend, carrying the level) added to the head so the level can still EXTRAPOLATE like
DLinear even if the deep seasonal path can't; (2) modest capacity (d_model 64, 2 layers); (3) the
DLinear-proven setup (representative val, global-norm, dropout).

Encoder-only adaptation (we regress the SCORE from the meteo window, not autoregressive forecasting):
embed seasonal -> N Auto-Correlation encoder layers -> pool seasonal + pooled raw trend -> head -> (B,H).
Interface matches DLinear/SSM/GRU so it reuses dataset_pt + the loop.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from patchtst.dlinear import _MovingAvg


class SeriesDecomp(nn.Module):
    def __init__(self, kernel: int):
        super().__init__()
        self.ma = _MovingAvg(kernel)

    def forward(self, x):                 # x: (B, L, C)
        trend = self.ma(x.transpose(1, 2)).transpose(1, 2)
        return x - trend, trend           # seasonal, trend


class AutoCorrelation(nn.Module):
    """FFT-based auto-correlation + time-delay aggregation (Autoformer core)."""
    def __init__(self, factor: int = 1, dropout: float = 0.1):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def _time_delay_agg(self, values, corr):   # values, corr: (B, H, C, L)
        B, H, C, L = values.shape
        top_k = max(1, int(self.factor * math.log(L)))
        idx = torch.arange(L, device=values.device)
        weights, delay = torch.topk(corr, top_k, dim=-1)     # (B,H,C,top_k)
        w = torch.softmax(weights, dim=-1)
        vv = values.repeat(1, 1, 1, 2)                       # circular roll buffer
        out = torch.zeros_like(values)
        for i in range(top_k):
            d = idx.view(1, 1, 1, L) + delay[..., i:i + 1]   # (B,H,C,L) gather indices
            out = out + torch.gather(vv, -1, d) * w[..., i:i + 1]
        return out

    def forward(self, q, k, v):            # (B, L, H, E)
        B, L, H, E = q.shape
        qf = torch.fft.rfft(q.permute(0, 2, 3, 1).contiguous(), dim=-1)
        kf = torch.fft.rfft(k.permute(0, 2, 3, 1).contiguous(), dim=-1)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=L, dim=-1)          # (B,H,E,L)
        out = self._time_delay_agg(v.permute(0, 2, 3, 1).contiguous(), corr)
        return self.dropout(out.permute(0, 3, 1, 2))         # (B,L,H,E)


class AutoCorrLayer(nn.Module):
    def __init__(self, d_model, n_heads, factor, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.out = nn.Linear(d_model, d_model)
        self.inner = AutoCorrelation(factor, dropout)

    def forward(self, x):                  # (B, L, d)
        B, L, _ = x.shape; H = self.n_heads
        q = self.q(x).view(B, L, H, -1); k = self.k(x).view(B, L, H, -1); v = self.v(x).view(B, L, H, -1)
        o = self.inner(q, k, v).reshape(B, L, -1)
        return self.out(o)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, factor, kernel, dropout):
        super().__init__()
        self.attn = AutoCorrLayer(d_model, n_heads, factor, dropout)
        self.decomp1 = SeriesDecomp(kernel); self.decomp2 = SeriesDecomp(kernel)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1); self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(x))
        x, _ = self.decomp1(x)
        y = self.drop(F.gelu(self.conv1(x.transpose(1, 2))))
        y = self.drop(self.conv2(y)).transpose(1, 2)
        x, _ = self.decomp2(x + y)
        return x                            # seasonal


class AutoformerRegressor(nn.Module):
    def __init__(self, n_channels, lookback, n_horizons, n_regions, region_emb_dim, side_dim, calendar_dim,
                 d_model=64, n_heads=4, e_layers=2, d_ff=128, kernel=25, dropout=0.2, factor=1):
        super().__init__()
        self.init_decomp = SeriesDecomp(kernel)
        self.embed = nn.Linear(n_channels, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, factor, kernel, dropout)
                                     for _ in range(e_layers)])
        self.head_seasonal = nn.Linear(d_model, n_horizons)
        self.head_trend = nn.Linear(n_channels, n_horizons)   # LINEAR level path -> extrapolates
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        nn.init.trunc_normal_(self.region_emb.weight, std=0.02)
        self.side = nn.Sequential(
            nn.Linear(side_dim + calendar_dim + region_emb_dim, 64),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(64, n_horizons),
        )

    def forward(self, x_daily, x_side, x_cal, region_idx):
        x = x_daily.transpose(1, 2)                    # (B, L, M)
        seasonal_init, trend_init = self.init_decomp(x)
        h = self.embed(seasonal_init)                  # (B, L, d)
        for layer in self.layers:
            h = layer(h)
        out = self.head_seasonal(h.mean(dim=1)) + self.head_trend(trend_init.mean(dim=1))
        r = self.region_emb(region_idx)
        return out + self.side(torch.cat([x_side, x_cal, r], dim=-1))


def build_autoformer_model_from_config(n_regions: int) -> AutoformerRegressor:
    import os
    from patchtst.config_pt import (
        LOOKBACK_DAYS, N_CHANNELS, HORIZONS, REGION_EMB_DIM, SCORE_LAG_OFFSETS, DROPOUT,
    )
    side_dim = len(SCORE_LAG_OFFSETS) + 2
    return AutoformerRegressor(
        n_channels=N_CHANNELS, lookback=LOOKBACK_DAYS, n_horizons=HORIZONS, n_regions=n_regions,
        region_emb_dim=REGION_EMB_DIM, side_dim=side_dim, calendar_dim=4,
        d_model=int(os.environ.get("AUTO_DMODEL", 64)), n_heads=4,
        e_layers=int(os.environ.get("AUTO_LAYERS", 2)), d_ff=128, kernel=25, dropout=DROPOUT, factor=1,
    )
