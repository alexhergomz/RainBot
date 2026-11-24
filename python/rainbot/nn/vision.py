"""Vision: the rendered screen -> a sequence of embeddings.

Input is the latest rendered frame, downscaled by the mod: 160x96, RGB plus the movement-graph
distance channel. A pure CNN of depthwise GLU blocks turns it into a 10x6 feature map,
flattened into 60 tokens. There is no pooling to a single vector, so the policy can attend to
several places in the image at once (VLA-style).

| step    | resolution (WxH)  | channels    | operation                                                |
|---------|-------------------|-------------|----------------------------------------------------------|
| stem    | 160x96 -> 80x48   | 4->16->32   | pixel-unshuffle x2, 1x1 projection                       |
| stage 1 | 80x48             | 32          | 2 x DW-GLU inverted bottleneck (expand 4, DW 3x3, ExtraDW) |
| down    | 80x48 -> 40x24    | 128 -> 64   | pixel-unshuffle x2, 1x1                                  |
| stage 2 | 40x24             | 64          | 2 x DW-GLU inverted bottleneck (expand 4, DW 5x5)        |
| down    | 40x24 -> 20x12    | 256 -> 128  | pixel-unshuffle x2, 1x1                                  |
| stage 3 | 20x12             | 128         | 2 x DW-GLU inverted bottleneck (expand 4, DW 3x3)        |
| down    | 20x12 -> 10x6     | 512 -> 128  | pixel-unshuffle x2, 1x1                                  |
| fusion  | 60 + K (+4 reg)   | 128         | one GQA layer + GLU FFN, both residual; output all tokens |

The fusion layer is a bidirectional GQA (4 query heads, 1 KV head, head dim 32), QK
ScaleNorm, per-head sigmoid output gate, 4 register tokens that are discarded, and 2D RoPE on the
image tokens (rows and columns of the grid; entity and register tokens unrotated). Entity tokens
(the K nearest creatures, food and items, exact game state) join the image tokens there; drop them
with ``use_entities=False``.

The backbone reads this sequence through sliding-window cross-attention over the last few frames
(``nn/xattn.py``).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope, rope2d_angles
from .layers import GLU, ConvResidual, DWGLU2d, HeadScaleNorm, LayerScale, ScaleNorm, Unshuffle2d


class FusionAttention(nn.Module):
    """Bidirectional GQA, QK ScaleNorm, sigmoid output gate per head, no mask."""

    def __init__(self, d: int = 128, q_heads: int = 4, kv_heads: int = 1, head_dim: int = 32):
        super().__init__()
        self.Hq, self.Hkv, self.Dh = q_heads, kv_heads, head_dim
        self.q = nn.Linear(d, q_heads * head_dim, bias=False)
        self.kv = nn.Linear(d, 2 * kv_heads * head_dim, bias=False)
        self.q_norm = HeadScaleNorm(q_heads, head_dim)          # ScaleNorm default gain sqrt(Dh)
        self.k_norm = HeadScaleNorm(kv_heads, head_dim)
        self.gate = nn.Linear(d, q_heads, bias=True)
        self.o = nn.Linear(q_heads * head_dim, d, bias=False)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        """rope = (cos, sin, n_image): 2D RoPE on the first n_image tokens, after QK-norm."""
        B, N, _ = x.shape
        q = self.q_norm(self.q(x).view(B, N, self.Hq, self.Dh))
        k, v = self.kv(x).view(B, N, 2, self.Hkv, self.Dh).unbind(2)
        k = self.k_norm(k)
        if rope is not None:
            cos, sin, n_img = rope
            q, k = apply_rope(q, cos, sin, 1, n_img), apply_rope(k, cos, sin, 1, n_img)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = v.transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)    # [B, Hq, N, Dh], scale 1/sqrt(Dh)
        o = o * torch.sigmoid(self.gate(x)).transpose(1, 2).unsqueeze(-1)
        return self.o(o.transpose(1, 2).reshape(B, N, self.Hq * self.Dh))


class EntityEncoder(nn.Module):
    """GLU over [kind, type, AI behaviour, flags embeddings | rel pos, rel vel, scalars] -> d."""

    def __init__(self, n_types: int, n_behaviors: int, n_cont: int, d: int = 128, emb: int = 32):
        super().__init__()
        self.kind = nn.Embedding(3, emb)          # creature, food, item
        self.etype = nn.Embedding(n_types, emb)    # index 0 = absent
        self.behavior = nn.Embedding(n_behaviors, emb)
        self.glu = GLU(emb + n_cont, d, hidden=2 * d)

    def forward(self, kind, etype, behavior, cont):
        e = self.kind(kind) + self.etype(etype) + self.behavior(behavior)
        return self.glu(torch.cat([e, cont], dim=-1))


class Vision(nn.Module):
    def __init__(self, n_types: int = 256, n_behaviors: int = 128, n_entity_cont: int = 8,
                 d: int = 128, registers: int = 4, frame_hw: tuple[int, int] = (96, 160),
                 use_entities: bool = True, frame_c: int = 3):
        super().__init__()
        H, W = frame_hw
        assert H % 16 == 0 and W % 16 == 0, "four pixel-unshuffles need H, W divisible by 16"
        self.d, self.use_entities = d, use_entities
        self.stem = Unshuffle2d(frame_c, 32)
        self.stage1 = nn.Sequential(*[ConvResidual(32, DWGLU2d(32, 3, extra_dw=True)) for _ in range(2)])
        self.down1 = Unshuffle2d(32, 64)
        self.stage2 = nn.Sequential(*[ConvResidual(64, DWGLU2d(64, 5)) for _ in range(2)])
        self.down2 = Unshuffle2d(64, d)
        self.stage3 = nn.Sequential(*[ConvResidual(d, DWGLU2d(d, 3)) for _ in range(2)])
        self.down3 = Unshuffle2d(d, d)
        self.grid = (H // 16, W // 16)
        self.n_image = self.grid[0] * self.grid[1]
        # 2D RoPE (rows, columns of the 10x6 grid) replaces a learned position embedding.
        cos, sin = rope2d_angles(self.grid, 32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.entities = EntityEncoder(n_types, n_behaviors, n_entity_cont, d) if use_entities else None
        self.registers = nn.Parameter(torch.randn(1, registers, d))      # N(0,1), like nn.Embedding
        self.n_reg = registers
        self.attn_norm, self.attn, self.attn_scale = ScaleNorm(d), FusionAttention(d), LayerScale(d)
        self.ffn_norm, self.ffn, self.ffn_scale = ScaleNorm(d), GLU(d), LayerScale(d)

    def n_tokens(self, n_entities: int) -> int:
        return self.n_image + (n_entities if self.use_entities else 0)

    def forward(self, frame: torch.Tensor, kind=None, etype=None, behavior=None, ent_cont=None) -> torch.Tensor:
        """frame [B, C, H, W] uint8 or float in [0, 1] (RGB, then the distance channel); entity
        tensors [B, K(, ...)] -> [B, N, d].

        Pixels are centred to [-0.5, 0.5]. In [0, 1], black regions (most of a Rain World frame)
        are exact zero vectors after the bias-free stem, where ScaleNorm's gradient is largest. The
        large gradients compound across blocks and the depthwise convolutions spread them to lit
        neighbours, so gradient clipping would freeze the whole network."""
        x = frame.float() / 255.0 if frame.dtype == torch.uint8 else frame
        if os.environ.get("RAINBOT_FRAME_CENTER", "1") == "1":     # "0": uncentred [0, 1] input, for diagnosis
            x = x - 0.5
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        x = self.down1(self.stage1(self.stem(x)))
        x = self.down3(self.stage3(self.down2(self.stage2(x))))           # [B, d, 6, 10]
        tok = x.flatten(2).transpose(1, 2)                                # [B, 60, d], row-major
        parts = [tok]
        if self.use_entities and kind is not None:
            parts.append(self.entities(kind, etype, behavior, ent_cont))
        B = tok.shape[0]
        parts.append(self.registers.expand(B, -1, -1))
        tok = torch.cat(parts, dim=1)
        tok = tok + self.attn_scale(self.attn(self.attn_norm(tok), (self.rope_cos, self.rope_sin, self.n_image)))
        tok = tok + self.ffn_scale(self.ffn(self.ffn_norm(tok)))
        return tok[:, :tok.shape[1] - self.n_reg]                          # registers discarded
