"""Channel-independent PatchTST encoder with a cross-channel regression head.

Adaptation of Nie et al., ICLR 2023 ("A Time Series is Worth 64 Words") for
the DMFP setup:
  - Inputs are (B, M, L) batches: M meteo channels, L lookback weeks.
  - Each channel is patched and passed through a shared Transformer encoder
    (channel-independence, weight sharing across channels).
  - Per-channel representations are mean-pooled over patches, concatenated
    across channels, and fused with a learned region embedding through an
    MLP head that outputs 5 future scores.

The paper uses BatchNorm-1d inside the encoder block (citing Zerveas 2021);
we follow that choice.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_patches(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """(B, M, L) -> (B, M, N, P) by sliding a window of length `patch_len`.

    Following the paper, we pad the trailing edge with the last value so that
    the patch count is ceil((L - patch_len) / stride) + 1 + 1.
    """
    B, M, L = x.shape
    # Pad last value `stride` times to ensure the tail patch is well-formed.
    pad = x[:, :, -1:].expand(B, M, stride)
    x = torch.cat([x, pad], dim=-1)                                # (B, M, L + stride)
    patches = x.unfold(dimension=-1, size=patch_len, step=stride)  # (B, M, N, P)
    return patches.contiguous()


class TransformerEncoderBlock(nn.Module):
    """Vanilla pre-norm Transformer block but with BatchNorm1d (Zerveas 2021)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # BatchNorm over the feature dim — applied to (B*M*N, d_model) reshape.
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
        # x: (B, N, d) -> (B*N, d) -> BN -> back to (B, N, d)
        B, N, D = x.shape
        return bn(x.reshape(B * N, D)).reshape(B, N, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self._bn(x + self.drop(attn_out), self.bn1)
        x = self._bn(x + self.drop(self.ff(x)), self.bn2)
        return x


class PatchTSTRegressor(nn.Module):
    """Channel-independent encoder + cross-channel regression head."""

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
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.d_model = d_model

        # Inferred from the patching layout
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

        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        nn.init.trunc_normal_(self.region_emb.weight, std=0.02)

        head_in = n_channels * d_model + region_emb_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_horizons),
        )

    def forward(self, x: torch.Tensor, region_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (B, M, L) z-scored channel matrix.
            region_idx: (B,) long int region indices.

        Returns:
            (B, H) future-score predictions (raw, clip to [0, 5] outside).
        """
        B, M, L = x.shape
        patches = _make_patches(x, self.patch_len, self.patch_stride)  # (B, M, N, P)
        N = patches.shape[2]

        # Channel-independent forward: fold channels into the batch dim.
        z = patches.reshape(B * M, N, self.patch_len)
        z = self.patch_proj(z)                                          # (B*M, N, D)
        z = z + self.pos_emb                                            # broadcast over batch

        for block in self.encoder:
            z = block(z)
        z = self.encoder_norm(z)

        # Pool patches per channel
        z = z.mean(dim=1)                                               # (B*M, D)
        z = z.reshape(B, M * self.d_model)                              # (B, M*D)

        r = self.region_emb(region_idx)                                 # (B, E)
        h = torch.cat([z, r], dim=-1)
        return self.head(h)


def build_model_from_config(n_regions: int) -> PatchTSTRegressor:
    from patchtst.config_pt import (
        LOOKBACK_WEEKS, PATCH_LEN, PATCH_STRIDE, HORIZONS,
        D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, REGION_EMB_DIM,
    )
    from patchtst.dataset_pt import N_CHANNELS
    return PatchTSTRegressor(
        n_channels=N_CHANNELS,
        lookback=LOOKBACK_WEEKS,
        n_horizons=HORIZONS,
        n_regions=n_regions,
        patch_len=PATCH_LEN,
        patch_stride=PATCH_STRIDE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        region_emb_dim=REGION_EMB_DIM,
    )
