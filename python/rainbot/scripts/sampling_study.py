"""Offline study of temporally correlated action sampling, to make the policy jitter less.

Works on the policy's own logits along a real rollout (<run>/logits_last.npz, saved by the trainer
every iteration).

Each decision is a categorical draw per action factor, a = argmax(logits / T + G). Independent
sampling draws fresh Gumbel noise G every decision. Correlated sampling keeps the noise's marginal
exactly Gumbel (so every decision is still drawn with the policy's own probability) but makes it
drift slowly: z_t = rho z_{t-1} + sqrt(1 - rho^2) e_t (Gaussian AR(1)), u = Phi(z), G = -log(-log u).
Consecutive decisions then agree unless the policy itself changes its mind.

Reported per scheme: how often the (x, y, jump) input changes between decisions, how often jump is
held, and the marginal check (mean log-prob of the chosen actions vs the policy's own expectation;
equal means each decision is still drawn from the policy). Humans: changes 0.04, jump held 0.05.

    python -m rainbot.scripts.sampling_study ../runs/grpo-bc6/logits_last.npz
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from scipy.special import ndtr

from ..nn.agent import split_logits


def factors(logits: np.ndarray, T: np.ndarray):
    """[n, 4, 12] logits, [n] temperatures -> list of 5 [n*4, k] log-prob arrays (decisions in order)."""
    temp = torch.as_tensor(T, dtype=torch.float32).view(-1, 1, 1)
    lg = torch.as_tensor(logits, dtype=torch.float32) / temp
    return [torch.log_softmax(p, -1).reshape(-1, p.shape[-1]).numpy() for p in split_logits(lg)]


def sample(logp: list, rho: float, rng: np.random.Generator) -> np.ndarray:
    n = logp[0].shape[0]
    out = np.zeros((n, len(logp)), np.int64)
    for f, lp in enumerate(logp):
        k = lp.shape[1]
        if rho <= 0.0:
            u = rng.random((n, k))
        else:
            z = np.zeros((n, k))
            z[0] = rng.standard_normal(k)
            e = rng.standard_normal((n, k)) * np.sqrt(1 - rho * rho)
            for t in range(1, n):
                z[t] = rho * z[t - 1] + e[t]
            u = ndtr(z)
        g = -np.log(-np.log(np.clip(u, 1e-12, 1 - 1e-12)))
        out[:, f] = np.argmax(lp + g, axis=1)
    return out


def stats(a: np.ndarray, logp: list) -> dict:
    ch = float(np.any(a[1:, :3] != a[:-1, :3], axis=1).mean())
    jump = float((a[:, 2] == 1).mean())
    chosen = np.mean([logp[f][np.arange(len(a)), a[:, f]].mean() for f in range(len(logp))])
    expect = np.mean([(np.exp(lp) * lp).sum(1).mean() for lp in logp])    # E[log p(a)] under the policy
    return {"changes": ch, "jump": jump, "logp_chosen": float(chosen), "logp_expected": float(expect)}


def main(path: str) -> None:
    d = np.load(path)
    L, Tm, ok = d["logits"].astype(np.float32), d["temp"], d["acted"]
    rng = np.random.default_rng(0)
    members = list(range(L.shape[0]))
    print(f"{path}: {int(ok.sum())} acted tokens, members {len(members)}")
    for rho in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98):
        rows = []
        for i in members:
            m = ok[i]
            if m.sum() < 20:
                continue
            lp = factors(L[i][m], Tm[i][m])
            rows.append(stats(sample(lp, rho, rng), lp))
        s = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        print(f"rho {rho:4.2f}: input changes {s['changes']:.3f}  jump held {s['jump']:.3f}  "
              f"mean log p(chosen) {s['logp_chosen']:.3f} vs policy's own {s['logp_expected']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
