"""2D rotary position embedding for image tokens (axial: half the head dim rotates with the row,
half with the column). Applied after QK-norm. Tokens without a grid position (entities, registers,
the policy's step-token queries) are left unrotated (position 0 == identity).
"""
from __future__ import annotations

import torch


def rope2d_angles(grid_hw: tuple[int, int], head_dim: int, base: float = 100.0, device=None):
    """cos, sin of shape [H*W, head_dim // 2] for a row-major grid."""
    assert head_dim % 4 == 0, "2D RoPE needs head_dim divisible by 4"
    h, w = grid_hw
    q = head_dim // 4                                     # rotation pairs per axis
    inv = base ** (-torch.arange(q, device=device, dtype=torch.float32) / q)
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    ang = torch.cat([ys.reshape(-1, 1) * inv, xs.reshape(-1, 1) * inv], dim=-1)   # [HW, D/2]
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, token_dim: int, n: int):
    """Rotate the first ``n`` tokens along ``token_dim`` of x [..., D] (pairs = adjacent halves)."""
    xr = x.narrow(token_dim, 0, n)
    rest = x.narrow(token_dim, n, x.shape[token_dim] - n)
    shape = [1] * x.dim()
    shape[token_dim], shape[-1] = n, cos.shape[-1]
    c, s = cos.view(shape).to(x.dtype), sin.view(shape).to(x.dtype)
    a, b = xr[..., ::2], xr[..., 1::2]
    rot = torch.stack([a * c - b * s, a * s + b * c], dim=-1).flatten(-2)
    return torch.cat([rot, rest], dim=token_dim)
