"""Channel-independent PatchTST encoder + cross-channel mixing + side-input head.

Phase 4 refactor over the original PatchTST track:
  - Inputs are (B, M, L) batches of daily meteo: M=9 channels, L=91 days.
  - Each channel is patched and passed through a shared Transformer encoder
    (channel-independent weight sharing).
  - Per-channel patch representations are mean-pooled → (B, M, D_MODEL).
  - NEW: channel-mixing layer (Linear over M) lets the model learn cross-
    feature interactions (tmp_max × humidity, etc.) that the original CI
    PatchTST couldn't.
  - Channels pooled → (B, D_MODEL), concatenated with:
        score-lag side inputs (5 staleness-shifted lags + 2 extras = 7)
        calendar side inputs (sin/cos of month + doy = 4)
        region embedding (16-dim)
  - MLP head outputs 5 future-week scores.

Encoder block follows Zerveas 2021 (BatchNorm1d inside the Transformer).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _make_patches(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """(B, M, L) -> (B, M, N, P) by sliding `patch_len` window with `stride`.
    Trailing edge is padded `stride` times with the last value to ensure a
    well-formed tail patch (matches Nie et al. 2023)."""
    B, M, L = x.shape
    pad = x[:, :, -1:].expand(B, M, stride)
    x = torch.cat([x, pad], dim=-1)
    patches = x.unfold(dimension=-1, size=patch_len, step=stride)  # (B, M, N, P)
    return patches.contiguous()


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer block with BatchNorm1d (Zerveas 2021)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def _bn(self, x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
        B, N, D = x.shape
        return bn(x.reshape(B * N, D)).reshape(B, N, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self._bn(x + self.drop(attn_out), self.bn1)
        x = self._bn(x + self.drop(self.ff(x)), self.bn2)
        return x


class PatchTSTRegressor(nn.Module):
    """Channel-independent encoder + channel mixing + side-input head."""

    def __init__(
        self,
        n_channels: int,
        lookback: int,
        n_horizons: int,
        n_regions: int,
        patch_len: int,
        patch_stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        region_emb_dim: int,
        score_lag_dim: int,            # 7 (5 lags + weeks_since + max_prev8)
        calendar_dim: int,             # 4 (sin/cos month/doy)
        use_channel_mixing: bool,
        channel_mix_hidden: int,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.use_channel_mixing = use_channel_mixing

        # Patch count formula from the Nie et al. unfold pattern with one
        # trailing-pad step: N = (L - patch_len) // stride + 2
        n_patches = (lookback - patch_len) // patch_stride + 2
        self.n_patches = n_patches

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # NEW: channel-mixing layer. Operates on (B, M, D) over the M dim.
        # Implemented as a bottleneck MLP applied per-feature-dim across the
        # channel axis to keep parameter count modest.
        if use_channel_mixing:
            self.channel_mix = nn.Sequential(
                nn.Linear(n_channels, channel_mix_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(channel_mix_hidden, n_channels),
            )
        else:
            self.channel_mix = None

        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        nn.init.trunc_normal_(self.region_emb.weight, std=0.02)

        # Side input projections
        self.score_lag_dim = score_lag_dim
        self.calendar_dim = calendar_dim
        self.side_proj = nn.Sequential(
            nn.Linear(score_lag_dim + calendar_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        head_in = d_model + region_emb_dim + 64
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_horizons),
        )

    def forward(
        self,
        x_daily: torch.Tensor,        # (B, M, L)
        x_score_lag: torch.Tensor,    # (B, score_lag_dim)
        x_calendar: torch.Tensor,     # (B, calendar_dim)
        region_idx: torch.Tensor,     # (B,)
    ) -> torch.Tensor:
        B, M, L = x_daily.shape

        # Patch
        patches = _make_patches(x_daily, self.patch_len, self.patch_stride)  # (B, M, N, P)
        N = patches.shape[2]

        # Channel-independent encoding: fold M into batch
        z = patches.reshape(B * M, N, self.patch_len)
        z = self.patch_proj(z)         # (B*M, N, D)
        z = z + self.pos_emb           # broadcast over (B*M)
        for block in self.encoder:
            z = block(z)
        z = self.encoder_norm(z)

        # Pool over patches per channel
        z = z.mean(dim=1)              # (B*M, D)
        z = z.reshape(B, M, self.d_model)  # (B, M, D)

        # Channel mixing — operates over the M axis at each d position
        if self.channel_mix is not None:
            # (B, M, D) -> (B, D, M) -> Linear(M->H->M) -> (B, M, D)
            z_t = z.transpose(1, 2)               # (B, D, M)
            z_t = self.channel_mix(z_t)            # (B, D, M)
            z = z + z_t.transpose(1, 2)            # residual back to (B, M, D)

        # Aggregate across channels (mean pool)
        z = z.mean(dim=1)              # (B, D)

        # Side inputs
        s = torch.cat([x_score_lag, x_calendar], dim=-1)  # (B, score_lag + cal)
        s = self.side_proj(s)                              # (B, 64)

        r = self.region_emb(region_idx)                    # (B, E)
        h = torch.cat([z, r, s], dim=-1)
        return self.head(h)


def build_model_from_config(n_regions: int) -> PatchTSTRegressor:
    from patchtst.config_pt import (
        LOOKBACK_DAYS, N_CHANNELS, PATCH_LEN_DAYS, PATCH_STRIDE_DAYS, HORIZONS,
        D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, REGION_EMB_DIM,
        USE_CHANNEL_MIXING, CHANNEL_MIX_HIDDEN,
        SCORE_LAG_OFFSETS,
    )
    score_lag_dim = len(SCORE_LAG_OFFSETS) + 2   # +weeks_since, +max_prev8
    calendar_dim = 4                              # month_sin/cos + doy_sin/cos
    return PatchTSTRegressor(
        n_channels=N_CHANNELS,
        lookback=LOOKBACK_DAYS,
        n_horizons=HORIZONS,
        n_regions=n_regions,
        patch_len=PATCH_LEN_DAYS,
        patch_stride=PATCH_STRIDE_DAYS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        region_emb_dim=REGION_EMB_DIM,
        score_lag_dim=score_lag_dim,
        calendar_dim=calendar_dim,
        use_channel_mixing=USE_CHANNEL_MIXING,
        channel_mix_hidden=CHANNEL_MIX_HIDDEN,
    )
