"""Held-out jitter of a cloned policy.

The policy picks each token's decisions greedily, conditioned on its own earlier choices (closed
loop in actions, open loop in observations: the features are the human's), on the first 512 tokens
of 6 held-out segments. Reports how often its (x, y, jump) input changes against the humans', and
where in the 4-decision chunk the changes fall.

    python -m rainbot.scripts.bc_jitter ../runs/bc-kf/bc.pt [--eval]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .. import obs as O
from ..nn.agent import to_tensors
from ..train.bc import DemoSet, evaluate, human_sessions
from ..train.runner import load_agent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--eval", action="store_true", help="also the teacher-forced held-out evaluation")
    ap.add_argument("--segments", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=512)
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    vocab = O.Vocab()
    m = load_agent(a.ckpt, "cpu", vocab)
    ds = DemoSet(human_sessions(), vocab)
    if a.eval:
        print("teacher-forced:", evaluate(m, ds, "cpu"), flush=True)
    pos_p, pos_h = np.zeros(4, int), np.zeros(4, int)
    n = 0
    for f, lab, val in ds.val[:a.segments]:
        T = min(len(val), a.tokens)
        st = m.initial_state(1)
        acts = []
        for t in range(T):
            ft = to_tensors({k: v[t:t + 1][None] for k, v in f.items()}, "cpu")
            _, act, _, st, _ = m.act(ft, st, torch.tensor([[t == 0]]), greedy=True)
            acts.append(act[0, 0].numpy())
        p = np.concatenate(acts).reshape(-1, 5)
        y = lab[:T].reshape(-1, 5)
        cp = (p[1:, :3] != p[:-1, :3]).any(1)
        ch = (y[1:, :3] != y[:-1, :3]).any(1)
        idx = np.arange(1, len(p)) % 4
        for k in range(4):
            pos_p[k] += cp[idx == k].sum()
            pos_h[k] += ch[idx == k].sum()
        n += len(p) - 1
    print(f"{a.ckpt}: input-change rate (own choices) {pos_p.sum() / n:.4f}, humans {pos_h.sum() / n:.4f}")
    print(f"changes by position (0 = chunk boundary, 1-3 = inside): "
          f"policy {pos_p.tolist()}  humans {pos_h.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
