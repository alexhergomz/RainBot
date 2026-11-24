"""MuSGD (Muon + SGD), partitioned by parameter type.

Two groups:

* muon: 2-D weight matrices (linear layers, 1x1 convs, dense convs). Update
  ``lr*muon_w * Muon + lr*sgd_w * SGD-Nesterov``, with decoupled weight decay.
* sgd: 1-D parameters (ScaleNorm g, LayerScale gamma, xATLU alpha, biases, KDA A_log and dt_bias),
  depthwise kernels (2-D and 1-D, including KDA's short convs), embeddings, learned tokens, actor
  logits and value output. SGD with Nesterov momentum 0.9.

Muon: Nesterov momentum 0.95 on the raw gradient, orthogonalised by 5 quintic Newton-Schulz steps
in bf16 (coefficients 3.4445, -4.7750, 2.0315), then scaled by Kimi's shape-based RMS-matching
factor ``0.2 * sqrt(max(m, n))`` so its per-element RMS matches an AdamW update. Same-shape matrices
are orthogonalised in one batched call.

The Muon/SGD blend follows Ultralytics' MuSGD (``ultralytics/optim/muon.py``, 8.4.x): matrices get
both an orthogonalised step weighted ``muon=0.5`` and a plain SGD-momentum step weighted
``sgd=0.5``, from separate momentum buffers. Two departures from that file, both from the design
guide: weight decay is decoupled (``p *= 1 - lr*wd``) and applied to matrices only, and the Muon
scale is Kimi's, not Ultralytics' ``sqrt(max(1, rows/cols))``.

Muon-family optimisers are well tested in supervised and language-model training, much less in on-policy RL.
``make_optimizer(kind="adamw")`` builds the AdamW control.
"""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7,
                   dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Approximate orthogonalisation of a batch of matrices [N, m, n]."""
    X = G.to(dtype)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    tall = G.size(-2) > G.size(-1)
    if tall:
        X = X.transpose(-2, -1)
    a, b, c = NS_COEFFS
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tall:
        X = X.transpose(-2, -1)
    return X.to(G.dtype)


def _is_depthwise(m: nn.Module) -> bool:
    return isinstance(m, (nn.Conv1d, nn.Conv2d)) and m.groups > 1 and m.groups == m.in_channels


#: Parameter names (suffix match) that are output layers or depthwise kernels despite being >= 2-D.
SGD_NAME_SUFFIXES = ("actor_logits.weight", "value.lin.weight", "conv_weight")


def partition(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    groups: dict[str, list] = {"muon": [], "sgd": []}
    seen = set()
    for mname, mod in model.named_modules():
        for pname, p in mod.named_parameters(recurse=False):
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            full = f"{mname}.{pname}" if mname else pname
            matrix = (isinstance(mod, nn.Linear) and pname == "weight") or \
                     (isinstance(mod, (nn.Conv1d, nn.Conv2d)) and pname == "weight" and not _is_depthwise(mod))
            if matrix and not full.endswith(SGD_NAME_SUFFIXES):
                groups["muon"].append((full, p))
            else:
                groups["sgd"].append((full, p))
    return groups


class MuSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 3e-4, momentum: float = 0.95, weight_decay: float = 0.01,
                 use_muon: bool = True, muon: float = 0.5, sgd: float = 0.5, ns_dtype=torch.bfloat16):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, use_muon=use_muon)
        super().__init__(params, defaults)
        self.muon_w, self.sgd_w, self.ns_dtype = muon, sgd, ns_dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in params:
                st = self.state[p]
                if not st:
                    st["sgd_buf"] = torch.zeros_like(p)
                    if group["use_muon"]:
                        st["muon_buf"] = torch.zeros_like(p)
            if wd:
                torch._foreach_mul_(params, 1 - lr * wd)                 # decoupled weight decay
            grads = [p.grad for p in params]
            sgd_lr = lr
            if group["use_muon"]:
                self._muon(params, grads, mom, lr * self.muon_w)
                sgd_lr = lr * self.sgd_w
            # SGD with Nesterov momentum (PyTorch convention: buf = m buf + g; step = g + m buf).
            bufs = [self.state[p]["sgd_buf"] for p in params]
            torch._foreach_mul_(bufs, mom)
            torch._foreach_add_(bufs, grads)
            upd = torch._foreach_add(grads, bufs, alpha=mom)
            torch._foreach_add_(params, upd, alpha=-sgd_lr)
        return loss

    def _muon(self, params, grads, mom, lr):
        bufs = [self.state[p]["muon_buf"] for p in params]
        torch._foreach_mul_(bufs, mom)
        torch._foreach_add_(bufs, grads, alpha=1 - mom)
        nest = torch._foreach_mul(bufs, mom)
        torch._foreach_add_(nest, grads, alpha=1 - mom)
        by_shape = defaultdict(list)
        for i, u in enumerate(nest):
            m2 = u.reshape(u.shape[0], -1)                                  # dense conv -> (out, in*k)
            by_shape[(tuple(m2.shape), m2.device)].append((i, m2))
        for (shape, _), items in by_shape.items():
            X = newton_schulz5(torch.stack([m for _, m in items]), dtype=self.ns_dtype)
            scale = 0.2 * max(shape) ** 0.5                                 # Kimi RMS matching
            for j, (i, _) in enumerate(items):
                params[i].add_(X[j].reshape(params[i].shape), alpha=-lr * scale)


def make_optimizer(model: nn.Module, kind: str = "musgd", lr: float = 3e-4, sgd_lr: float = 3e-3,
                   weight_decay: float = 0.01):
    """MuSGD partitioned by parameter type, or the AdamW control (decay on matrices only)."""
    groups = partition(model)
    muon = [p for _, p in groups["muon"]]
    other = [p for _, p in groups["sgd"]]
    if kind == "adamw":
        return torch.optim.AdamW([{"params": muon, "weight_decay": weight_decay},
                                  {"params": other, "weight_decay": 0.0}], lr=lr, eps=1e-5)
    if kind != "musgd":
        raise ValueError(kind)
    return MuSGD([{"params": muon, "lr": lr, "momentum": 0.95, "weight_decay": weight_decay,
                   "use_muon": True},
                  {"params": other, "lr": sgd_lr, "momentum": 0.9, "weight_decay": 0.0,
                   "use_muon": False}])
