"""Sliding-window GQA with attention sinks.

* 8 query heads, 2 KV heads, head dim 32; QK ScaleNorm (L2 + learned per-head temperature);
  a sigmoid output gate per head; no position encoding (the KDA layers carry position).
* Causal window W = 128 tokens: token t sees tokens t-127..t (flash-attn ``window_size=(127, 0)``).
* R = 4 learned sink key/value pairs per KV head that never slide out of the window.

Sinks. The guide computes windowed attention and sink attention separately and merges them with
log-sum-exp weights, because prepending sinks as tokens breaks flash-attn's sliding window. That
merge is exact, and so is the alternative used by the torch path here: put the sinks in the key
set as always-visible columns of an explicit mask. ``merge_sinks`` implements the LSE merge for the
flash-attn path, and ``tests/test_sequence_layers.py`` checks the two agree.

Backends. flash-attn (``flash_attn_func`` / ``flash_attn_with_kvcache``) when it is installed on a
CUDA device; otherwise PyTorch SDPA with an explicit boolean mask. SDPA's flash backend rejects
masks and falls back to slower kernels. At 256 tokens x (128 + 256 + 4) keys the masked math kernel
costs nothing measurable at this model size, and flash-attn does not build for RDNA2 cards.

Context across calls. The cache holds the last W tokens' keys and values in time order (a shifted
buffer rather than a ring: identical attention, simpler indexing, and W = 128 copies per step are
free). It is the "window state" PPO stores at the start of each training sequence.

Episode resets: a reset at token t hides every earlier key from t onward (sinks stay visible).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import HeadScaleNorm
from .kda import episode_ids

try:  # pragma: no cover - not available on this machine
    from flash_attn import flash_attn_func as _flash_attn_func
    HAVE_FLASH = True
except Exception:
    HAVE_FLASH = False


@dataclass
class SWAState:
    k: torch.Tensor        # [B, W, Hkv, Dh], already QK-normed
    v: torch.Tensor        # [B, W, Hkv, Dh]
    valid: torch.Tensor    # [B, W] bool

    def detach(self) -> "SWAState":
        return SWAState(self.k.detach(), self.v.detach(), self.valid)

    def index(self, idx) -> "SWAState":
        return SWAState(self.k[idx], self.v[idx], self.valid[idx])


def merge_sinks(o_w: torch.Tensor, lse_w: torch.Tensor, q: torch.Tensor,
                sink_k: torch.Tensor, sink_v: torch.Tensor) -> torch.Tensor:
    """Exact merge of windowed attention with sink attention.

    o_w [B, T, H, D], lse_w [B, H, T] (natural log), q [B, T, H, D] (already scaled so that logits
    are q.k), sink_k/sink_v [Hkv, R, D]. ``o = sigmoid(L_w - L_s) o_w + sigmoid(L_s - L_w) o_s``.
    """
    B, T, H, D = q.shape
    Hkv = sink_k.shape[0]
    sk = sink_k.repeat_interleave(H // Hkv, dim=0).float()           # [H, R, D]
    sv = sink_v.repeat_interleave(H // Hkv, dim=0).float()
    logits = torch.einsum("bthd,hrd->bhtr", q.float(), sk)
    lse_s = torch.logsumexp(logits, dim=-1)                           # [B, H, T]
    o_s = torch.einsum("bhtr,hrd->bthd", logits.softmax(-1), sv)
    w = torch.sigmoid(lse_w.float() - lse_s).transpose(1, 2).unsqueeze(-1)   # [B, T, H, 1]
    return (w * o_w.float() + (1 - w) * o_s).to(o_w.dtype)


class SWA(nn.Module):
    def __init__(self, d_model: int = 256, q_heads: int = 8, kv_heads: int = 2, head_dim: int = 32,
                 window: int = 128, sinks: int = 4):
        super().__init__()
        self.Hq, self.Hkv, self.Dh, self.W, self.R = q_heads, kv_heads, head_dim, window, sinks
        self.q_proj = nn.Linear(d_model, q_heads * head_dim, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * kv_heads * head_dim, bias=False)
        self.q_norm = HeadScaleNorm(q_heads, head_dim)          # ScaleNorm default gain sqrt(Dh)
        self.k_norm = HeadScaleNorm(kv_heads, head_dim)
        self.gate = nn.Linear(d_model, q_heads, bias=True)
        self.o_proj = nn.Linear(q_heads * head_dim, d_model, bias=False)
        self.sink_k = nn.Parameter(torch.randn(kv_heads, sinks, head_dim))   # N(0,1), like nn.Embedding
        self.sink_v = nn.Parameter(torch.randn(kv_heads, sinks, head_dim))

    def initial_state(self, batch: int, device=None, dtype=torch.float32) -> SWAState:
        return SWAState(torch.zeros(batch, self.W, self.Hkv, self.Dh, device=device, dtype=dtype),
                        torch.zeros(batch, self.W, self.Hkv, self.Dh, device=device, dtype=dtype),
                        torch.zeros(batch, self.W, dtype=torch.bool, device=device))

    def _sinks(self):
        return self.k_norm(self.sink_k.transpose(0, 1)).transpose(0, 1), self.sink_v

    def forward(self, x: torch.Tensor, state: SWAState | None = None,
                reset: torch.Tensor | None = None) -> tuple[torch.Tensor, SWAState]:
        B, T, _ = x.shape
        if state is None:
            state = self.initial_state(B, x.device, x.dtype)
        if reset is None:
            reset = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        reset = reset.bool()

        q = self.q_norm(self.q_proj(x).view(B, T, self.Hq, self.Dh))
        k, v = self.kv_proj(x).view(B, T, 2, self.Hkv, self.Dh).unbind(2)
        k = self.k_norm(k)

        if HAVE_FLASH and x.is_cuda and not bool(state.valid.any()) and not bool(reset[:, 1:].any()):
            o = self._flash(q, k, v)  # pragma: no cover
        else:
            o = self._masked(q, k, v, state, reset)

        o = o * torch.sigmoid(self.gate(x)).unsqueeze(-1)
        out = self.o_proj(o.reshape(B, T, self.Hq * self.Dh))
        return out, self._next_state(k, v, state, reset)

    # -- torch path ----------------------------------------------------------------------
    def _masked(self, q, k, v, state: SWAState, reset):
        B, T = q.shape[:2]
        W, R = self.W, self.R
        sk, sv = self._sinks()
        keys = torch.cat([sk.transpose(0, 1).unsqueeze(0).expand(B, R, self.Hkv, self.Dh).to(k.dtype),
                          state.k.to(k.dtype), k], dim=1)              # [B, R+W+T, Hkv, Dh]
        vals = torch.cat([sv.transpose(0, 1).unsqueeze(0).expand(B, R, self.Hkv, self.Dh).to(v.dtype),
                          state.v.to(v.dtype), v], dim=1)
        ep = episode_ids(reset)                                         # [B, T]
        t = torch.arange(T, device=q.device)
        # Prefix slot i holds time i - W.
        s_pre = torch.arange(W, device=q.device) - W
        pre_ok = (state.valid.unsqueeze(1)                              # [B, 1, W]
                  & (t.view(T, 1) - s_pre.view(1, W) <= W - 1).unsqueeze(0)
                  & (ep == 0).unsqueeze(-1))
        s = torch.arange(T, device=q.device)
        cur_ok = ((s.view(1, T) <= t.view(T, 1)) & (t.view(T, 1) - s.view(1, T) <= W - 1)).unsqueeze(0) \
            & (ep.unsqueeze(-1) == ep.unsqueeze(1))                    # [B, T, T]
        mask = torch.cat([torch.ones(B, T, R, dtype=torch.bool, device=q.device), pre_ok, cur_ok], dim=-1)
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), keys.transpose(1, 2), vals.transpose(1, 2),
            attn_mask=mask.unsqueeze(1), enable_gqa=True)   # default 1/sqrt(Dh)
        return o.transpose(1, 2)                                        # [B, T, Hq, Dh]

    def _flash(self, q, k, v):  # pragma: no cover - needs flash-attn on CUDA
        o, lse, _ = _flash_attn_func(q, k, v, causal=True,
                                     window_size=(self.W - 1, 0), return_attn_probs=True)
        sk, sv = self._sinks()
        return merge_sinks(o, lse, q * self.Dh ** -0.5, sk, sv)

    def _next_state(self, k, v, state: SWAState, reset) -> SWAState:
        B, T = k.shape[:2]
        ep = episode_ids(reset)
        allk = torch.cat([state.k.to(k.dtype), k], dim=1)[:, -self.W:]
        allv = torch.cat([state.v.to(v.dtype), v], dim=1)[:, -self.W:]
        ep_all = torch.cat([ep.new_zeros(B, self.W), ep], dim=1)[:, -self.W:]
        valid_all = torch.cat([state.valid, torch.ones(B, T, dtype=torch.bool, device=k.device)],
                              dim=1)[:, -self.W:]
        valid = valid_all & (ep_all == ep[:, -1:])
        return SWAState(allk, allv, valid)


def windowed_attention_with_lse(q, k, v, window: int):
    """Reference windowed causal attention returning (o [B,T,H,D], lse [B,H,T]); GQA by repeat.

    Used only by tests to check ``merge_sinks`` against the masked-sinks path.
    """
    B, T, H, D = q.shape
    rep = H // k.shape[2]
    kk = k.repeat_interleave(rep, dim=2).float()
    vv = v.repeat_interleave(rep, dim=2).float()
    logits = torch.einsum("bthd,bshd->bhts", q.float(), kk)
    t = torch.arange(T, device=q.device)
    ok = (t.view(1, T) <= t.view(T, 1)) & (t.view(T, 1) - t.view(1, T) <= window - 1)
    logits = logits.masked_fill(~ok, float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    o = torch.einsum("bhts,bshd->bthd", logits.softmax(-1), vv)
    return o, lse
