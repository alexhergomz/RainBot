"""Building blocks shared by every module.

Global rules, applied everywhere including vision and the 1-D temporal convs:

* ScaleNorm only. ``g * x / sqrt(||x||^2 + d * eps)`` with one learned scalar ``g`` per norm,
  initialised to ``sqrt(d)``. For conv features the norm runs over channels at each position. No
  BatchNorm, LayerNorm, RMSNorm or GroupNorm anywhere in this package.
* Pre-norm residuals with LayerScale. ``x <- x + gamma * F(ScaleNorm(x))``, gamma a per-channel
  vector initialised to 0.1, so every block starts close to identity and needs no LR warmup.
* xATLU-gated GLUs. ``ATLU(z) = z * (arctan(z) + pi/2) / pi``; the expanded gate stretches the
  range from (0, 1) to (-alpha, 1 + alpha) with one trainable alpha per block, initialised to 0.
  The GLU is the second-order form ``W_out[(W_a x) * xATLU(W_b x)]`` with hidden width (8/3) d.
  ``order=1`` gives the paper's first-order xATGLU ``(W_a x) * g_alpha(W_b x)``, the documented
  fallback if the second-order form underperforms.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def glu_hidden(d: int) -> int:
    """(8/3) d, rounded up to a multiple of 8: parameter parity with a 4x MLP."""
    return int(math.ceil(8 * d / 3 / 8) * 8)


#: eps on the mean square, as nn.RMSNorm does by default (the dtype's machine epsilon). A 1e-6 floor
#: on the norm instead lets a zero vector (a black image patch) pass gradients at gain g / 1e-6 ~ 1e7
#: per norm, ~1e15 through the CNN. With eps on the mean square the gain at zero is at most
#: g / sqrt(d * eps) ~ 2.9e3, and the function is smooth.
NORM_EPS = torch.finfo(torch.float32).eps


def _rms_denominator(x: torch.Tensor, dim: int) -> torch.Tensor:
    """sqrt(||x||^2 + d * eps): the L2 norm with RMSNorm's mean-square epsilon."""
    xf = x.float()
    d = xf.shape[dim]
    return (xf.square().sum(dim=dim, keepdim=True) + d * NORM_EPS).sqrt()


class ScaleNorm(nn.Module):
    """``g * x / sqrt(||x||^2 + d * eps)`` over dimension ``dim``, one learned scalar ``g = sqrt(d)``.

    ``dim=-1`` for token features; ``dim=1`` for conv feature maps (normalise channels per position).
    """

    def __init__(self, d: int, dim: int = -1, eps: float = EPS):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.g = nn.Parameter(torch.tensor(math.sqrt(d)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() * (self.g / _rms_denominator(x, self.dim))).to(x.dtype)


class HeadScaleNorm(nn.Module):
    """ScaleNorm per attention head over the head dimension, with a learned per-head temperature.

    Used for QK-norm (L2 normalisation plus a learned per-head scale) and for the per-head output
    norm of the KDA layer. Input ``[..., H, D]``.
    """

    def __init__(self, heads: int, head_dim: int, init: float | None = None, eps: float = EPS):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.full((heads, 1), math.sqrt(head_dim) if init is None else init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() * (self.g / _rms_denominator(x, -1))).to(x.dtype)


class LayerScale(nn.Module):
    """Per-channel residual scale, initialised to 0.1 (shallow network)."""

    def __init__(self, d: int, init: float = 0.1, dim: int = -1):
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.full((d,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == -1:
            return x * self.gamma
        shape = [1] * x.dim()
        shape[self.dim] = -1
        return x * self.gamma.view(shape)


def xatlu_gate(z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """``g_alpha(z) = ((arctan z + pi/2) / pi) * (1 + 2 alpha) - alpha``."""
    return (torch.atan(z) + math.pi / 2) / math.pi * (1 + 2 * alpha) - alpha


class _XATLUFn(torch.autograd.Function):
    """``z * g_alpha(z)`` saving only ``z`` for backward.

    The eager composition keeps four full-size temporaries alive for autograd (atan, the gate, ...),
    which runs an 8 GB card out of memory on the vision stack. Analytic gradients:
        d/dz     = g(z) + z (1 + 2 alpha) / (pi (1 + z^2))
        d/dalpha = z (2 (arctan z + pi/2) / pi - 1)
    """

    @staticmethod
    def forward(ctx, z, alpha):
        ctx.save_for_backward(z, alpha)
        return z * xatlu_gate(z, alpha)

    @staticmethod
    def backward(ctx, go):
        z, alpha = ctx.saved_tensors
        up = torch.float64 if z.dtype == torch.float64 else torch.float32
        zf, af, gf = z.to(up), alpha.to(up), go.to(up)
        base = (torch.atan(zf) + math.pi / 2) / math.pi
        g = base * (1 + 2 * af) - af
        dz = g + zf * (1 + 2 * af) / (math.pi * (1 + zf * zf))
        gz = gf * dz
        da = (gf * zf * (2 * base - 1)).sum()
        return gz.to(z.dtype), da.to(alpha.dtype)


class _GatedXATLUFn(torch.autograd.Function):
    """``a * z * g_alpha(z)`` (the GLU product) saving only ``a`` and ``z``.

    Eager, autograd keeps ``a``, ``z`` and ``h = xATLU(z)`` (plus the product, saved by the next
    1x1 conv either way). Here ``h`` is recomputed instead: it is elementwise, so cheaper than
    storing a full-resolution tensor per vision block. Gradients are the eager ones exactly:
        d/da = h,   d/dz = a * h'(z),   d/dalpha = sum(a * z * (2 base - 1)).
    """

    @staticmethod
    def forward(ctx, a, z, alpha):
        ctx.save_for_backward(a, z, alpha)
        return a * (z * xatlu_gate(z, alpha))

    @staticmethod
    def backward(ctx, go):
        a, z, alpha = ctx.saved_tensors
        up = torch.float64 if z.dtype == torch.float64 else torch.float32
        zf, af, al, gf = z.to(up), a.to(up), alpha.to(up), go.to(up)
        base = (torch.atan(zf) + math.pi / 2) / math.pi
        g = base * (1 + 2 * al) - al
        h = zf * g
        dz = g + zf * (1 + 2 * al) / (math.pi * (1 + zf * zf))
        ga = gf * h
        gz = gf * af * dz
        dal = (gf * af * zf * (2 * base - 1)).sum()
        return ga.to(a.dtype), gz.to(z.dtype), dal.to(alpha.dtype)


def gated_xatlu(a: torch.Tensor, z: torch.Tensor, act: "XATLU") -> torch.Tensor:
    return _GatedXATLUFn.apply(a, z, act.alpha.to(z.dtype))


class XATLU(nn.Module):
    """``z * g_alpha(z)`` with one trainable alpha per block (init 0 = plain ATLU)."""

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(()))

    def gate(self, z: torch.Tensor) -> torch.Tensor:
        return xatlu_gate(z, self.alpha)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return _XATLUFn.apply(z, self.alpha.to(z.dtype))


class GLU(nn.Module):
    """``W_out[(W_a x) * xATLU(W_b x)]`` (order 2) or ``W_out[(W_a x) * g_alpha(W_b x)]`` (order 1).

    ``W_a`` and ``W_b`` are one fused matrix. ``d_out`` defaults to ``d_in``.
    """

    def __init__(self, d_in: int, d_out: int | None = None, hidden: int | None = None,
                 order: int = 2, bias: bool = False):
        super().__init__()
        d_out = d_in if d_out is None else d_out
        hidden = glu_hidden(d_in) if hidden is None else hidden
        self.hidden, self.order = hidden, order
        self.w_ab = nn.Linear(d_in, 2 * hidden, bias=bias)
        self.act = XATLU()
        self.w_out = nn.Linear(hidden, d_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w_ab(x).chunk(2, dim=-1)
        h = gated_xatlu(a, b, self.act) if self.order == 2 else a * self.act.gate(b)
        return self.w_out(h)


# ---------------------------------------------------------------------------------------------
# Depthwise-separable GLU inverted bottlenecks
# ---------------------------------------------------------------------------------------------


class DWGLU2d(nn.Module):
    """Inverted bottleneck, depthwise-separable GLU factorisation, 2-D.

    1. optional depthwise kxk before expansion (MobileNetV4 ExtraDW);
    2. 1x1 expansion to two branches a, b of E = expand * C channels;
    3. depthwise kxk on the gate branch b only (spatial context enters through the gate);
    4. a * xATLU(b);
    5. 1x1 projection back to C.

    The residual wrapper (ScaleNorm over channels, LayerScale) lives in ``ConvResidual``.
    """

    def __init__(self, c: int, k: int, expand: int = 4, extra_dw: bool = False):
        super().__init__()
        e = expand * c
        self.pre_dw = nn.Conv2d(c, c, k, padding=k // 2, groups=c, bias=False) if extra_dw else None
        self.expand = nn.Conv2d(c, 2 * e, 1, bias=False)
        self.gate_dw = nn.Conv2d(e, e, k, padding=k // 2, groups=e, bias=False)
        self.act = XATLU()
        self.project = nn.Conv2d(e, c, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_dw is not None:
            x = self.pre_dw(x)
        a, b = self.expand(x).chunk(2, dim=1)
        return self.project(gated_xatlu(a, self.gate_dw(b), self.act))


class DWGLU1d(nn.Module):
    """The same block over a 1-D (time) axis. Non-causal 'same' padding: it only mixes the
    8 ticks inside one token, so no causal padding is needed."""

    def __init__(self, c: int, k: int = 3, expand: int = 4):
        super().__init__()
        e = expand * c
        self.expand = nn.Conv1d(c, 2 * e, 1, bias=False)
        self.gate_dw = nn.Conv1d(e, e, k, padding=k // 2, groups=e, bias=False)
        self.act = XATLU()
        self.project = nn.Conv1d(e, c, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.expand(x).chunk(2, dim=1)
        return self.project(gated_xatlu(a, self.gate_dw(b), self.act))


class ConvResidual(nn.Module):
    """``x + gamma * block(ScaleNorm_channels(x))`` for ``[B, C, ...]`` feature maps."""

    def __init__(self, c: int, block: nn.Module):
        super().__init__()
        self.norm = ScaleNorm(c, dim=1)
        self.block = block
        self.scale = LayerScale(c, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale(self.block(self.norm(x)))


class Unshuffle2d(nn.Module):
    """Pixel-unshuffle x2 (lossless space-to-depth, no strided conv), then a 1x1 projection."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = nn.Conv2d(4 * c_in, c_out, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.pixel_unshuffle(x, 2))
