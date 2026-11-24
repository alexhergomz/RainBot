"""Kimi Delta Attention, following fla's ``KimiDeltaAttention`` layer.

The one change from the reference layer is the output norm: Kimi uses a gated RMSNorm, this layer
uses a per-head ScaleNorm (global rule 6.0) followed by the same sigmoid output gate from a
low-rank projection.

    q, k, v  = SiLU(shortconv_4(W x))                      causal depthwise conv, left padded
    q, k     = L2-normalised per head
    g        = -exp(A_log) * softplus(f_up(f_down(x)) + dt_bias)    log-decay, fp32, [H, d_k]
    beta     = sigmoid(W_beta x)                            one per head
    S_t      = (I - beta k k^T) Diag(exp g) S_{t-1} + beta k v^T
    o_t      = scale * S_t^T q_t
    out      = W_o[ HeadScaleNorm(o) * sigmoid(g_up(g_down(x))) ]

Kernels. The default torch path runs on any device in fp32: ``kda_chunked`` (chunkwise-parallel
WY/UT form) for sequences and ``kda_recurrence`` (the delta-rule recurrence, the same math as fla's
``naive_recurrent_kda``) for single-token rollout steps. ``tests/test_sequence_layers.py`` checks
them against each other and against fla's reference. With ``RAINBOT_KDA=fla`` on a CUDA device,
training uses fla's ``chunk_kda`` and rollout ``fused_recurrent_kda``, with the gate, beta-sigmoid
and QK L2-norm fused in-kernel, as fla's layer calls them.

Episode resets. ``reset[b, t]`` marks the first token of a new episode: the recurrent state and the
short-conv history are zeroed before that token. In the fla path this becomes variable-length
packing (``cu_seqlens``), with each segment's initial state either the carried state (segment begins
at t = 0 and is not a reset) or zeros.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import HeadScaleNorm

try:  # fla is optional: GPU-only Triton kernels.
    from fla.ops.kda import chunk_kda as _fla_chunk_kda
    from fla.ops.kda import fused_recurrent_kda as _fla_recurrent_kda
    HAVE_FLA = True
except Exception:  # pragma: no cover - depends on the machine
    HAVE_FLA = False

#: Kernel backend: "torch" (default) or "fla". fla's Triton kernels do not compile for RDNA2
#: (gfx1030, fla-core 0.6.0, triton-rocm 3.6.0), so the default is the torch chunked form. Set
#: RAINBOT_KDA=fla on hardware where fla works.
BACKEND = os.environ.get("RAINBOT_KDA", "torch")


@dataclass
class KDAState:
    S: torch.Tensor        # [B, H, d_k, d_v] fp32
    conv: torch.Tensor     # [B, conv_size - 1, C] last inputs to the short conv (q|k|v channels)

    def detach(self) -> "KDAState":
        return KDAState(self.S.detach(), self.conv.detach())

    def index(self, idx) -> "KDAState":
        return KDAState(self.S[idx], self.conv[idx])


def episode_ids(reset: torch.Tensor) -> torch.Tensor:
    """Per-token episode index within a call; the carried-in context counts as episode 0."""
    return torch.cumsum(reset.to(torch.int32), dim=1)


def short_conv(x: torch.Tensor, weight: torch.Tensor, cache: torch.Tensor,
               reset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise conv with history ``cache`` and episode resets, then SiLU.

    x [B, T, C]; weight [C, K] (last tap multiplies the current input); cache [B, K-1, C].
    An input contributes to output t only if it belongs to t's episode.
    """
    B, T, C = x.shape
    K = weight.shape[1]
    full = torch.cat([cache.to(x.dtype), x], dim=1)                      # [B, T+K-1, C]
    ep = episode_ids(reset)                                               # [B, T]
    ep_full = torch.cat([ep.new_zeros(B, K - 1), ep], dim=1)              # history = episode 0
    y = x.new_zeros(B, T, C)
    for j in range(K):                                                    # tap j looks back j steps
        src = full[:, K - 1 - j: K - 1 - j + T]
        same = (ep_full[:, K - 1 - j: K - 1 - j + T] == ep).unsqueeze(-1).to(x.dtype)
        y = y + src * same * weight[:, K - 1 - j].to(x.dtype)
    keep = (ep_full[:, -(K - 1):] == ep[:, -1:]).unsqueeze(-1).to(x.dtype)
    new_cache = full[:, -(K - 1):] * keep
    return F.silu(y), new_cache


def kda_recurrence(q, k, v, g, beta, S0, reset, scale):
    """Reference delta-rule recurrence, fp32. q,k [B,T,H,K] (already L2-normalised), v [B,T,H,V],
    g [B,T,H,K] log-decay, beta [B,T,H] in (0,1), S0 [B,H,K,V]. Returns o [B,T,H,V], S_T."""
    B, T = q.shape[:2]
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    S = S0.float()
    outs = []
    for t in range(T):
        S = S * (~reset[:, t]).float().view(B, 1, 1, 1)
        S = S * g[:, t].exp().unsqueeze(-1)
        kt, vt, bt = k[:, t], v[:, t], beta[:, t].unsqueeze(-1)
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, bt * (vt - pred))
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))
    return torch.stack(outs, dim=1), S


#: Log-decay written into the gate at a reset token: exp(-200) underflows to exactly 0 in fp32, so
#: the carried state is wiped, and |-200| keeps fp32 cumsum error near 1e-5 in log space.
RESET_LOG_DECAY = -200.0


def kda_chunked(q, k, v, g, beta, S0, reset, scale, chunk: int = 16):
    """Chunkwise-parallel KDA in torch: the WY/UT form of fla's ``naive_chunk_kda``, vectorised.

    Same inputs as ``kda_recurrence``. Resets are folded into the gate: a reset token's decay is
    ``exp(RESET_LOG_DECAY) = 0``, which zeroes the state it inherits (exactly the recurrence's
    reset) and keeps every chunk dense. Within a chunk of C tokens:

        L[c, i]  = beta_c sum_d k_cd k_id exp(G_cd - G_id)       (c > i)
        T        = (I + L)^-1 diag(beta)
        W = T (exp(G) * k),  U = T v
        per chunk: V' = U - W S;  o = (q exp(G)) S + Aqk V';  S <- exp(G_C) S + (exp(G_C - G) k)^T V'

    with G the within-chunk cumulative log-decay and Aqk[c, j] = sum_d q_cd k_jd exp(G_cd - G_jd)
    for j <= c.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    g = g.masked_fill(reset.view(B, T, 1, 1), RESET_LOG_DECAY)
    pad = (-T) % chunk
    if pad:
        z = lambda t: F.pad(t, [0, 0] * (t.dim() - 2) + [0, pad])
        q, k, v, g = z(q), z(k), z(v), z(g)
        beta = F.pad(beta, [0, 0, 0, pad])
    N, C = (T + pad) // chunk, chunk
    # [B, H, N, C, ...]
    r = lambda t: t.reshape(B, N, C, H, *t.shape[3:]).transpose(1, 3).transpose(2, 3)
    q, k, v, g = r(q * scale), r(k), r(v), r(g)
    beta = beta.reshape(B, N, C, H).permute(0, 3, 1, 2)                   # [B, H, N, C]
    G = g.cumsum(-2)                                                       # [B, H, N, C, K]

    diff = G.unsqueeze(-2) - G.unsqueeze(-3)                               # [.., c, i, K] = G_c - G_i
    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device), -1)
    incl = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device), 0)
    decay_l = torch.exp(diff.masked_fill(~lower.view(C, C, 1), float("-inf")))
    decay_i = torch.exp(diff.masked_fill(~incl.view(C, C, 1), float("-inf")))
    L = torch.einsum("bhncd,bhnid,bhncid->bhnci", k, k, decay_l) * beta.unsqueeze(-1)
    eye = torch.eye(C, device=q.device)
    Tm = torch.linalg.solve_triangular(eye + L, eye.expand_as(L), upper=False, unitriangular=True)
    Tm = Tm * beta.unsqueeze(-2)                                           # right-multiply diag(beta)
    W = Tm @ (G.exp() * k)
    U = Tm @ v
    Aqk = torch.einsum("bhncd,bhnid,bhncid->bhnci", q, k, decay_i)
    qg = q * G.exp()
    kd = torch.exp(G[..., -1:, :] - G) * k                                 # [B,H,N,C,K]
    gl = G[..., -1, :].exp()                                               # [B,H,N,K]

    S = S0.float()
    outs = []
    for n in range(N):
        vp = U[:, :, n] - W[:, :, n] @ S
        outs.append(qg[:, :, n] @ S + Aqk[:, :, n] @ vp)
        S = S * gl[:, :, n].unsqueeze(-1) + kd[:, :, n].transpose(-1, -2) @ vp
    o = torch.stack(outs, dim=2)                                           # [B,H,N,C,V]
    o = o.permute(0, 2, 3, 1, 4).reshape(B, N * C, H, V)[:, :T]
    return o, S


class KDA(nn.Module):
    def __init__(self, d_model: int = 256, heads: int = 4, head_dim: int = 64, conv_size: int = 4):
        super().__init__()
        self.d, self.H, self.Dk = d_model, heads, head_dim
        self.Dv = head_dim
        self.key_dim = heads * head_dim
        self.value_dim = heads * self.Dv
        self.conv_size = conv_size
        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        # Depthwise short conv weights for q|k|v, initialised like nn.Conv1d (fla's ShortConvolution).
        c = 2 * self.key_dim + self.value_dim
        self.conv_weight = nn.Parameter(torch.empty(c, conv_size))
        nn.init.kaiming_uniform_(self.conv_weight.view(c, 1, conv_size), a=math.sqrt(5))
        # Low-rank decay gate and output gate, rank = head_v_dim as in fla.
        self.f_down = nn.Linear(d_model, self.Dv, bias=False)
        self.f_up = nn.Linear(self.Dv, heads * head_dim, bias=False)
        self.b_proj = nn.Linear(d_model, heads, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(heads).uniform_(1, 16)))
        dt = torch.exp(torch.rand(heads * head_dim) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.g_down = nn.Linear(d_model, self.Dv, bias=False)
        self.g_up = nn.Linear(self.Dv, self.value_dim, bias=True)
        self.o_norm = HeadScaleNorm(heads, self.Dv)
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)
        self.scale = head_dim ** -0.5

    # -- state ---------------------------------------------------------------------------
    def initial_state(self, batch: int, device=None) -> KDAState:
        return KDAState(
            torch.zeros(batch, self.H, self.Dk, self.Dv, device=device),
            torch.zeros(batch, self.conv_size - 1, 2 * self.key_dim + self.value_dim, device=device),
        )

    def _use_fla(self, x: torch.Tensor) -> bool:
        if BACKEND == "torch":
            return False
        return HAVE_FLA and x.is_cuda

    # -- forward -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, state: KDAState | None = None,
                reset: torch.Tensor | None = None) -> tuple[torch.Tensor, KDAState]:
        # Always fp32: the delta-rule recurrence accumulates and inverts within chunks, which
        # overflows fp16, and autocast would re-cast its matmuls to fp16 even on fp32 inputs. KDA is
        # small, so the cost is negligible. The output returns in the caller's dtype.
        dt = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            o, st = self._forward(x.float(), state, reset)
        return o.to(dt) if dt in (torch.float16, torch.bfloat16) else o, st

    def _forward(self, x: torch.Tensor, state: KDAState | None = None,
                 reset: torch.Tensor | None = None) -> tuple[torch.Tensor, KDAState]:
        B, T, _ = x.shape
        if state is None:
            state = self.initial_state(B, x.device)
        if reset is None:
            reset = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        reset = reset.bool()

        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
        qkv, conv_cache = short_conv(qkv, self.conv_weight, state.conv, reset)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(B, T, self.H, self.Dk)
        k = k.view(B, T, self.H, self.Dk)
        v = v.view(B, T, self.H, self.Dv)
        f = self.f_up(self.f_down(x)).view(B, T, self.H, self.Dk)
        b = self.b_proj(x)

        if self._use_fla(x):
            o, S = self._fla(q, k, v, f, b, state.S, reset)
        else:
            g = -self.A_log.float().exp().view(1, 1, self.H, 1) * F.softplus(
                f.float() + self.dt_bias.float().view(self.H, self.Dk))
            fn = kda_recurrence if T == 1 else kda_chunked
            o, S = fn(F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1),
                      v, g, torch.sigmoid(b.float()), state.S, reset, self.scale)

        o = self.o_norm(o.to(x.dtype))
        gate = torch.sigmoid(self.g_up(self.g_down(x))).view(B, T, self.H, self.Dv)
        out = self.o_proj((o * gate).reshape(B, T, self.value_dim))
        return out, KDAState(S, conv_cache)

    def _fla(self, q, k, v, f, b, S0, reset):
        """fla kernels with variable-length packing at episode resets."""
        B, T, H, Kd = q.shape
        common = dict(A_log=self.A_log, dt_bias=self.dt_bias, scale=self.scale,
                      use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                      use_beta_sigmoid_in_kernel=True, output_final_state=True)
        if not bool(reset.any()):
            fn = _fla_recurrent_kda if (T == 1 and not torch.is_grad_enabled()) else _fla_chunk_kda
            o, S = fn(q, k, v, f, b, initial_state=S0.float(), **common)
            return o, S
        # Pack: one segment per (row, episode piece).
        starts = reset.clone()
        starts[:, 0] = True
        flat_starts = starts.reshape(-1).nonzero().squeeze(-1)
        cu = torch.cat([flat_starts, flat_starts.new_tensor([B * T])]).to(torch.int32)
        seg_row = flat_starts // T
        seg_carries = (flat_starts % T == 0) & ~reset.reshape(-1)[flat_starts]
        init = torch.zeros(len(flat_starts), H, Kd, self.Dv, device=q.device)
        init[seg_carries] = S0.float()[seg_row[seg_carries]]
        pk = lambda t: t.reshape(1, B * T, *t.shape[2:])
        o, S_seg = _fla_chunk_kda(pk(q), pk(k), pk(v), pk(f), pk(b), initial_state=init,
                                  cu_seqlens=cu.long(), **common)
        # Final state per row = final state of that row's last segment.
        last_seg = torch.zeros(B, dtype=torch.long, device=q.device)
        last_seg.scatter_reduce_(0, seg_row, torch.arange(len(flat_starts), device=q.device),
                                 reduce="amax", include_self=False)
        return o.reshape(B, T, H, self.Dv), S_seg[last_seg]
