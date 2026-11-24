"""Sliding-window cross-attention from the step tokens to recent frames' vision tokens.

The policy sees the screen VLA-style: each frame becomes a sequence of embeddings
(``nn/vision.py``), not one pooled vector. The backbone reads it through cross-attention instead of
folding image tokens into its own sequence. Folding would push every frame through the KDA
recurrences (which cannot be masked) and fill the SWA window with image tokens. Instead each step's
token attends to the vision tokens of the last ``window`` frames (a small sliding window for a bit
more capacity); KDA carries everything older, as it does for the self-attention layers.

Same conventions as every attention in the model: GQA, QK ScaleNorm (L2 + learned per-head
temperature), a sigmoid output gate per head, and 2D RoPE on the image-token keys (after QK-norm;
the step-token query sits at position 0). Keys get a learned frame-age embedding (0 = the current
frame) before QK-norm, so the model can tell the newest frame from older ones. The block wraps it pre-norm:
``x <- x + gamma * XAttn(ScaleNorm(x))``.

Exact, and without a [T x (T*N)] mask: the window is a set of ``window`` shifted views of one
key/value tensor (offset w = frame t - w for query t), each scored against the queries, then a
single softmax over all window x tokens. Episode resets and missing frames are masked per offset.

State across calls (rollout steps, PPO sequences): the raw keys/values of the previous
``window - 1`` frames, exactly like the SWA layer's window cache.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .kda import episode_ids
from .layers import HeadScaleNorm
from .rope import apply_rope, rope2d_angles


@dataclass
class XState:
    k: torch.Tensor        # [B, P, N, Hkv, Dh] raw keys (before age embedding and QK-norm)
    v: torch.Tensor        # [B, P, N, Hkv, Dh]
    valid: torch.Tensor    # [B, P] bool

    def detach(self) -> "XState":
        return XState(self.k.detach(), self.v.detach(), self.valid)

    def index(self, idx) -> "XState":
        return XState(self.k[idx], self.v[idx], self.valid[idx])


class XAttn(nn.Module):
    def __init__(self, d_model: int = 256, d_vis: int = 128, n_tokens: int = 74, q_heads: int = 8,
                 kv_heads: int = 2, head_dim: int = 32, window: int = 4, grid_hw: tuple | None = (6, 10)):
        super().__init__()
        self.Hq, self.Hkv, self.Dh, self.W, self.N = q_heads, kv_heads, head_dim, window, n_tokens
        self.q = nn.Linear(d_model, q_heads * head_dim, bias=False)
        self.kv = nn.Linear(d_vis, 2 * kv_heads * head_dim, bias=False)
        self.q_norm = HeadScaleNorm(q_heads, head_dim)          # ScaleNorm default gain sqrt(Dh)
        self.k_norm = HeadScaleNorm(kv_heads, head_dim)
        self.scale = head_dim ** -0.5                           # standard attention scale
        self.age = nn.Parameter(torch.randn(window, kv_heads, head_dim))   # N(0,1), like nn.Embedding
        # 2D RoPE on the image tokens' keys (the first grid_h*grid_w of each frame), after QK-norm.
        # The query is a step token with no place on the screen: position 0, i.e. unrotated, so the
        # keys carry their absolute screen position.
        self.n_img = 0
        if grid_hw is not None:
            cos, sin = rope2d_angles(tuple(grid_hw), head_dim)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            self.n_img = grid_hw[0] * grid_hw[1]
        self.gate = nn.Linear(d_model, q_heads, bias=True)
        self.o = nn.Linear(q_heads * head_dim, d_model, bias=False)

    def initial_state(self, batch: int, device=None) -> XState:
        P = self.W - 1
        z = torch.zeros(batch, P, self.N, self.Hkv, self.Dh, device=device)
        return XState(z, z.clone(), torch.zeros(batch, P, dtype=torch.bool, device=device))

    def forward(self, x: torch.Tensor, vis: torch.Tensor, frame_valid: torch.Tensor,
                state: XState | None, reset: torch.Tensor):
        """x [B, T, d] (already normed); vis [B, T, N, d_vis]; frame_valid, reset [B, T]."""
        B, T, _ = x.shape
        if state is None:
            state = self.initial_state(B, x.device)
        P, N, rep = self.W - 1, vis.shape[2], self.Hq // self.Hkv
        q = self.q_norm(self.q(x).view(B, T, self.Hq, self.Dh))
        k, v = self.kv(vis).view(B, T, N, 2, self.Hkv, self.Dh).unbind(3)
        k_all = torch.cat([state.k.to(k.dtype), k], 1)                   # [B, P+T, N, Hkv, Dh]
        v_all = torch.cat([state.v.to(v.dtype), v], 1)
        valid_all = torch.cat([state.valid, frame_valid.bool()], 1)        # [B, P+T]
        ep = episode_ids(reset)
        ep_all = torch.cat([ep.new_zeros(B, P), ep], 1)

        scores, values = [], []
        for w in range(self.W):
            sl = slice(P - w, P - w + T)                                   # frame t - w for query t
            kw = self.k_norm(k_all[:, sl] + self.age[w].to(k.dtype))
            if self.n_img:
                kw = apply_rope(kw, self.rope_cos, self.rope_sin, 2, self.n_img)
            kw = kw.repeat_interleave(rep, dim=3)                          # [B, T, N, Hq, Dh]
            s = torch.einsum("bthd,btnhd->bthn", q, kw) * self.scale
            ok = valid_all[:, sl] & (ep_all[:, sl] == ep)                  # [B, T]
            scores.append(s.masked_fill(~ok[:, :, None, None], float("-inf")))
            values.append(v_all[:, sl].repeat_interleave(rep, dim=3))
        S = torch.cat(scores, -1).float()                                  # [B, T, Hq, W*N]
        A = torch.softmax(S, -1).nan_to_num(0.0).to(q.dtype)               # no frame yet -> zero read
        V = torch.cat(values, 2)                                           # [B, T, W*N, Hq, Dh]
        o = torch.einsum("bthn,btnhd->bthd", A, V)
        o = o * torch.sigmoid(self.gate(x)).unsqueeze(-1)
        out = self.o(o.reshape(B, T, self.Hq * self.Dh))

        keep = (ep_all[:, -P:] == ep[:, -1:]) if P > 0 else valid_all[:, :0]
        new = XState(k_all[:, -P:], v_all[:, -P:], valid_all[:, -P:] & keep) if P > 0 else state
        return out, new
