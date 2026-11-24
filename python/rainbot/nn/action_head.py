"""Autoregressive action head: a stack of causal convolutions over the decision timeline.

Drawing the 4 decisions of a chunk independently from one feature vector ties no decision to the
one before it, and the policy then changes input far more often than humans do (7.1% of decisions
against 0.7% on held-out human data, after cloning). Here every decision sees the decisions before
it, so imitation can learn the humans' persistence directly and RL can keep it, with no penalty
weight.

Each decision position p gets x_p = (its slot feature from the trunk) + (embedding of the action
taken at p-1; a 'none' index at an episode's first decision). A stack of residual causal conv
layers (kernel 2, dilations d_i: a receptive field of 1 + sum(d_i) decisions) mixes them; taps
never cross an episode start. The agent passes reset = True on every token when the head's
history is limited to its own chunk (``AgentConfig.action_chunk_only``). Training is
teacher-forced with the actions taken (one pass over all 4T positions); rollout samples the 4
decisions of a chunk one at a time.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..proto import ACTION_DIMS
from .layers import GLU, LayerScale, ScaleNorm


@dataclass
class ActState:
    cache: torch.Tensor     # [B, layers, D, d] last D normalised inputs of each layer (0 = none)
    prev: torch.Tensor      # [B, 5] long, the last decision taken (index n = none)

    def detach(self) -> "ActState":
        return ActState(self.cache.detach(), self.prev)

    def index(self, idx) -> "ActState":
        return ActState(self.cache[idx], self.prev[idx])


class CausalLayer(nn.Module):
    def __init__(self, d: int, dilation: int, glu_order: int):
        super().__init__()
        self.dilation = dilation
        self.norm = ScaleNorm(d)
        # Input [current | dilation back]. No bias, so a masked (zero) past contributes nothing.
        self.taps = nn.Linear(2 * d, d, bias=False)
        self.glu = GLU(d, order=glu_order)
        self.scale = LayerScale(d)

    def forward(self, x, ep, cache, cache_ep):
        """x [B, P, d], ep [B, P] episode ids; cache [B, D, d] with episode ids cache_ep [B, D]."""
        B, P, d = x.shape
        D = cache.shape[1]
        n = self.norm(x)
        full = torch.cat([cache.to(n.dtype), n], 1)                     # [B, D+P, d]
        ep_full = torch.cat([cache_ep, ep], 1)
        k = self.dilation
        past = full[:, D - k: D - k + P]
        same = (ep_full[:, D - k: D - k + P] == ep).unsqueeze(-1).to(n.dtype)
        y = self.taps(torch.cat([n, past * same], -1))
        x = x + self.scale(self.glu(y))
        keep = (ep_full[:, -D:] == ep[:, -1:]).unsqueeze(-1).to(n.dtype)
        return x, full[:, -D:] * keep


class ActionHead(nn.Module):
    def __init__(self, d: int = 64, dilations: tuple = (1, 2, 4, 8), glu_order: int = 2):
        super().__init__()
        self.d, self.D = d, max(dilations)
        self.prev_emb = nn.ModuleList([nn.Embedding(n + 1, d) for n in ACTION_DIMS])
        self.layers = nn.ModuleList([CausalLayer(d, k, glu_order) for k in dilations])

    def initial_state(self, batch: int, device=None) -> ActState:
        none = torch.tensor(ACTION_DIMS, device=device).expand(batch, -1).clone()
        return ActState(torch.zeros(batch, len(self.layers), self.D, self.d, device=device), none)

    def _embed(self, prev: torch.Tensor) -> torch.Tensor:
        return sum(e(prev[..., i]) for i, e in enumerate(self.prev_emb))

    def _run(self, x, ep, st: ActState):
        B = x.shape[0]
        cache_ep = ep.new_zeros(B, self.D)                               # carried history = episode 0
        caches = []
        for li, layer in enumerate(self.layers):
            x, c = layer(x, ep, st.cache[:, li], cache_ep)
            caches.append(c)
        return x, torch.stack(caches, 1)

    def forward(self, slots: torch.Tensor, act: torch.Tensor, st: ActState, reset: torch.Tensor):
        """Teacher-forced. slots [B, T, K, d], act [B, T, K, 5] (the decisions taken), reset [B, T]
        -> features [B, T, K, d], new state."""
        B, T, K, d = slots.shape
        ep = torch.cumsum(reset.to(torch.int64), 1).repeat_interleave(K, 1)      # [B, T*K]
        a = act.reshape(B, T * K, 5).long()
        prev = torch.cat([st.prev[:, None], a[:, :-1]], 1)
        ep_prev = torch.cat([ep.new_zeros(B, 1), ep[:, :-1]], 1)
        none = torch.tensor(ACTION_DIMS, device=a.device)
        prev = torch.where((ep_prev == ep).unsqueeze(-1), prev, none)
        x = slots.reshape(B, T * K, d) + self._embed(prev)
        x, cache = self._run(x, ep, st)
        return x.view(B, T, K, d), ActState(cache, a[:, -1])

    def step(self, slots: torch.Tensor, st: ActState, reset: torch.Tensor, choose):
        """Rollout, one token: slots [B, 1, K, d], reset [B, 1]; ``choose(feature [B, d]) ->
        (logits [B, 12], action [B, 5])`` picks each decision in turn. Returns logits [B, 1, K, 12],
        actions [B, 1, K, 5], new state."""
        B, _, K, d = slots.shape
        none = torch.tensor(ACTION_DIMS, device=slots.device)
        r = reset.reshape(B).bool()
        prev = torch.where(r.unsqueeze(-1), none, st.prev)
        cache = st.cache
        logits, acts = [], []
        for j in range(K):
            ep = (r & (j == 0)).long().view(B, 1)                         # a reset starts episode 1
            x = slots[:, 0, j:j + 1] + self._embed(prev).unsqueeze(1)
            x, cache = self._run(x, ep, ActState(cache, prev))
            lg, a = choose(x[:, 0])
            logits.append(lg)
            acts.append(a)
            prev = a.long()
        return torch.stack(logits, 1).unsqueeze(1), torch.stack(acts, 1).unsqueeze(1), ActState(cache, prev)
