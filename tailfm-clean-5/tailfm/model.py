"""Velocity field v_theta(x_t, t) for windows x in R^{n x f}.

DiT-style: tokens are timesteps.  The input projection and the per-token MLPs mix
channels at every timestep, multi-head self-attention mixes time, and their
composition across depth represents the joint cross-channel/cross-time structure in
which co-crash behaviour lives.  Flow time t conditions every block through
adaLN-Zero, with gates initialised to zero so the model starts as the identity.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 100.0) -> torch.Tensor:
    """Sinusoidal Fourier features of t in [0, 1]; t shaped (B,)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * max_period
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (B, h, N, dh)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class DiTBlock(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = Attention(d, heads)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(),
                                 nn.Linear(mlp_ratio * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)   # adaLN-Zero
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), s1, sc1))
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s2, sc2))
        return x


class VelocityField(nn.Module):
    """pos_std sets the scale of the positional embedding relative to the projected
    input.  forward() computes in_proj(x) + pos, and with nn.Linear's default init the
    entries of in_proj(x) have sd ~0.75 in z-space at f=235; the usual std=0.02 makes
    position ~3% of the token signal, which LayerNorm then dilutes further.  Since the
    base is exchangeable in time and adaLN-Zero starts the network at the identity,
    the model has almost no gradient pressure to break that exchangeability, and at
    std=0.02 it does not: the generated windows are statistically indistinguishable
    from real windows with the time axis permuted.  std=0.1 reproduces the observed
    within-window squared-return ACF; larger values overshoot it and start copying
    individual training windows.
    """

    def __init__(self, f: int, n_max: int, d_model: int = 512,
                 depth: int = 4, heads: int = 4, mlp_ratio: int = 4,
                 pos_std: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(f, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_max, d_model))
        nn.init.trunc_normal_(self.pos, std=pos_std)
        self.t_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(),
                                   nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList(DiTBlock(d_model, heads, mlp_ratio)
                                    for _ in range(depth))
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.out_proj = nn.Linear(d_model, f)
        nn.init.zeros_(self.ada_out[1].weight); nn.init.zeros_(self.ada_out[1].bias)
        nn.init.zeros_(self.out_proj.weight);   nn.init.zeros_(self.out_proj.bias)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, n, f); t: (B,) or (B,1,1) in [0,1]. Returns (B, n, f)."""
        B, n, _ = x.shape
        c = self.t_mlp(timestep_embedding(t.reshape(B), self.d_model))
        h = self.in_proj(x) + self.pos[:, :n]
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.ada_out(c).chunk(2, dim=-1)
        return self.out_proj(modulate(self.norm_out(h), shift, scale))
