"""RevIN-equipped PatchTST regressor for cross-year covariate shift.

Reversible Instance Normalization (Kim et al. ICLR 2022, arXiv:2105.06441)
adapted to DMFP's (meteo input → score output) setting:

  - Per-instance per-channel μ, σ are computed on the 91-day lookback window.
  - The daily input is normalized: x' = γ ⊙ (x − μ) / (σ + ε) + β, learnable
    affine γ, β per channel (initialized to 1, 0).
  - The score head receives the encoder output AS-IS — no reversal, because
    the score is a different modality from the input meteo. Classic RevIN's
    denormalize step doesn't apply.
  - Variant D extension: (μ, σ) themselves are concatenated to the head's
    side inputs (2 × N_CHANNELS scalars per instance). This lets the head
    keep the absolute-level signal (which LGBM relies on) while the encoder
    learns shape-only features (cross-year robust).

The base PatchTST encoder, channel-mixing, region embedding, and the score-
lag/calendar side inputs are reused unchanged from `model_pt.PatchTSTRegressor`.

Training script: `code/patchtst/train_pt_revin.py`. Artifacts land under
`code/patchtst/models_pt/patchtst_revin.{pt,...}` and don't collide with the
existing patchtst_v2 checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from patchtst.model_pt import _make_patches, TransformerEncoderBlock


class RevIN(nn.Module):
    """Per-instance per-channel normalization with learnable affine.

    Forward:  (B, M, L) -> normalized (B, M, L), plus (μ, σ) of shape (B, M)
              returned for downstream side-input concatenation.

    The model does NOT denormalize the head output (score is a different
    modality from input meteo).
    """

    def __init__(self, n_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_channels))
            self.beta = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, M, L)
        mu = x.mean(dim=-1)                          # (B, M)
        sigma = x.std(dim=-1, unbiased=False)        # (B, M)
        x_normed = (x - mu.unsqueeze(-1)) / (sigma.unsqueeze(-1) + self.eps)
        if self.affine:
            x_normed = x_normed * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
        return x_normed, mu, sigma


class RevINPatchTSTRegressor(nn.Module):
    """PatchTST + RevIN + (μ, σ) concatenated to side inputs (variant D).

    Mirrors `PatchTSTRegressor`'s architecture exactly except:
      1. RevIN runs first on x_daily (replaces dataset-side global norm).
      2. Per-instance (μ, σ) are concatenated to (score_lag, calendar) before
         the side projection, growing its input dim by 2 × n_channels.
    """

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
        score_lag_dim: int,
        calendar_dim: int,
        use_channel_mixing: bool,
        channel_mix_hidden: int,
        revin_eps: float = 1e-5,
        revin_affine: bool = True,
        use_instance_stats_side_input: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.use_channel_mixing = use_channel_mixing
        self.use_instance_stats_side_input = use_instance_stats_side_input

        # RevIN front-end
        self.revin = RevIN(n_channels, eps=revin_eps, affine=revin_affine)

        # Patch / encoder (identical to base PatchTSTRegressor)
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

        # Side projection: now also consumes (μ, σ) per channel
        extra_stats = 2 * n_channels if use_instance_stats_side_input else 0
        self.side_proj = nn.Sequential(
            nn.Linear(score_lag_dim + calendar_dim + extra_stats, 64),
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
        x_daily: torch.Tensor,
        x_score_lag: torch.Tensor,
        x_calendar: torch.Tensor,
        region_idx: torch.Tensor,
    ) -> torch.Tensor:
        B, M, L = x_daily.shape

        # RevIN front: per-instance per-channel normalize, return (μ, σ)
        x_normed, mu, sigma = self.revin(x_daily)

        # Patch + encode (identical pipeline to base model)
        patches = _make_patches(x_normed, self.patch_len, self.patch_stride)
        N = patches.shape[2]
        z = patches.reshape(B * M, N, self.patch_len)
        z = self.patch_proj(z)
        z = z + self.pos_emb
        for block in self.encoder:
            z = block(z)
        z = self.encoder_norm(z)

        z = z.mean(dim=1)                              # (B*M, D)
        z = z.reshape(B, M, self.d_model)              # (B, M, D)

        if self.channel_mix is not None:
            z_t = z.transpose(1, 2)
            z_t = self.channel_mix(z_t)
            z = z + z_t.transpose(1, 2)

        z = z.mean(dim=1)                              # (B, D)

        # Side inputs: append per-instance stats so the head sees absolute levels
        side_parts = [x_score_lag, x_calendar]
        if self.use_instance_stats_side_input:
            side_parts.extend([mu, sigma])
        s = torch.cat(side_parts, dim=-1)
        s = self.side_proj(s)

        r = self.region_emb(region_idx)
        h = torch.cat([z, r, s], dim=-1)
        return self.head(h)


def build_revin_model_from_config(n_regions: int) -> RevINPatchTSTRegressor:
    from patchtst.config_pt import (
        LOOKBACK_DAYS, N_CHANNELS, PATCH_LEN_DAYS, PATCH_STRIDE_DAYS, HORIZONS,
        D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, REGION_EMB_DIM,
        USE_CHANNEL_MIXING, CHANNEL_MIX_HIDDEN, SCORE_LAG_OFFSETS,
    )
    score_lag_dim = len(SCORE_LAG_OFFSETS) + 2
    calendar_dim = 4
    return RevINPatchTSTRegressor(
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
        revin_eps=1e-5,
        revin_affine=True,
        use_instance_stats_side_input=True,
    )
