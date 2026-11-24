"""Plasticity diagnostics and ReDo resets ("Plasticity"; 9).

* **Dormant-unit ratio** (Sokar et al. 2023, ReDo): for every GLU hidden layer, a unit's score is
  its mean absolute activation over a batch divided by the layer's mean; a unit is dormant when its
  score is <= tau. Measured on the GLU's gated hidden ``a * xATLU(b)``, the input of ``w_out``.
* **ReDo reset** when the ratio exceeds ~10%: dormant units get fresh incoming weights (their rows of
  the fused ``w_ab``, both branches) and zero outgoing weights (their column of ``w_out``), so the
  network's function is unchanged at the moment of the reset; their optimiser state is cleared.
* **Stable rank** of the weight matrices (``||W||_F^2 / ||W||_2^2``) and the effective rank of the
  trunk features (Kumar et al. 2021: smallest k with the top-k singular values covering 99%).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..nn.layers import GLU

TAU = 0.025
RESET_ABOVE = 0.10


class DormantProbe:
    """Forward hooks recording mean |hidden| per GLU unit."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.acts: dict[str, torch.Tensor] = {}
        self.handles = []
        for name, m in model.named_modules():
            if isinstance(m, GLU):
                self.handles.append(m.w_out.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(mod, inp, out):
            h = inp[0].detach().float().abs()
            s = h.reshape(-1, h.shape[-1]).mean(0)
            prev = self.acts.get(name)
            self.acts[name] = s if prev is None else (prev + s) / 2
        return fn

    def close(self):
        for h in self.handles:
            h.remove()

    def dormant(self, tau: float = TAU) -> dict[str, torch.Tensor]:
        out = {}
        for name, s in self.acts.items():
            score = s / s.mean().clamp_min(1e-9)
            out[name] = score <= tau
        return out


def dormant_ratio(masks: dict[str, torch.Tensor]) -> float:
    n = sum(int(m.numel()) for m in masks.values())
    return sum(int(m.sum()) for m in masks.values()) / max(n, 1)


@torch.no_grad()
def redo_reset(model: nn.Module, masks: dict[str, torch.Tensor], optimizer=None) -> int:
    """Reinitialise dormant GLU units in place. Returns the number of units reset."""
    mods = dict(model.named_modules())
    total = 0
    for name, mask in masks.items():
        if not bool(mask.any()):
            continue
        glu: GLU = mods[name]
        idx = mask.nonzero().squeeze(-1).to(glu.w_ab.weight.device)
        H = glu.hidden
        fresh = torch.empty_like(glu.w_ab.weight)
        nn.init.kaiming_uniform_(fresh, a=math.sqrt(5))
        rows = torch.cat([idx, idx + H])
        glu.w_ab.weight[rows] = fresh[rows]
        if glu.w_ab.bias is not None:
            glu.w_ab.bias[rows] = 0
        glu.w_out.weight[:, idx] = 0
        if optimizer is not None:
            for p, sl in ((glu.w_ab.weight, (rows,)), (glu.w_out.weight, (slice(None), idx))):
                st = optimizer.state.get(p, {})
                for buf in st.values():
                    if torch.is_tensor(buf) and buf.shape == p.shape:
                        buf[sl] = 0
        total += int(idx.numel())
    return total


@torch.no_grad()
def weight_stable_ranks(model: nn.Module) -> dict[str, float]:
    out = {}
    for name, p in model.named_parameters():
        if p.ndim == 2 and min(p.shape) >= 8 and "embed" not in name:
            s = torch.linalg.matrix_norm(p.float(), ord=2)
            out[name] = float(p.float().square().sum() / s.square().clamp_min(1e-12))
    return out


@torch.no_grad()
def feature_effective_rank(h: torch.Tensor, delta: float = 0.01) -> int:
    """Smallest k such that the top-k singular values hold (1 - delta) of the total."""
    s = torch.linalg.svdvals(h.reshape(-1, h.shape[-1]).float())
    c = torch.cumsum(s, 0) / s.sum().clamp_min(1e-12)
    return int((c < 1 - delta).sum()) + 1
