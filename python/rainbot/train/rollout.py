"""Rollout storage, advantages and recurrent-state helpers, shared by the PPO and GRPO collectors."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .. import obs as O
from ..env.rain_env import Step
from ..nn.agent import split_logits
from .curriculum import EpisodeStats


def stomach(row) -> float:
    """Food in the stomach in pips, quarter pips included."""
    return float(row["food_in_stomach"]) + float(row["quarter_food"]) / 4.0


def death_cause(row) -> str:
    """Why the slugcat died, from the last row before the death (diagnostic only)."""
    if int(row["rain_hit"]):
        return "rain"            # the rain kills by flooding, so the lungs are empty too: check rain first
    if float(row["air_in_lungs"]) <= 0.05:
        return "drowned"
    if int(row["grabbed_by"]) or int(row["hit_taken"]):
        return "predator"
    return "other"


def has_event(step: Step, kind: str, reason: str | None = None) -> bool:
    return any(ev.get("type") == kind and (reason is None or ev.get("reason") == reason) for ev in step.events)


class Rollout:
    """[N, T] storage for one iteration: N games, T tokens."""

    def __init__(self, n: int, T: int):
        self.n, self.T = n, T
        self.feat: dict[str, np.ndarray] | None = None
        self.reset = np.zeros((n, T), bool)
        self.act = np.zeros((n, T, O.DECISIONS, 5), np.int64)
        self.logp = np.zeros((n, T), np.float32)
        self.temp = np.ones((n, T), np.float32)          # sampling temperature of the token's member
        self.logits = None                                # [n, T, 4, 12] float16, for the temperature fit
        self.value = np.zeros((n, T), np.float32)
        self.rew = np.zeros((n, T), np.float32)
        self.valid = np.ones((n, T), bool)
        self.acted = np.ones((n, T), bool)
        self.next_start = np.zeros((n, T), bool)
        self.deficits = np.zeros((n, T, 3), np.float32)
        self.stale = np.zeros((n, T), np.float32)      # ticks between the frame and the step's last row
        self.brightness = np.zeros((n, T), np.float32)  # mean frame pixel value: ~0 means no picture
        self.boot = np.zeros(n, np.float32)

    def put_feat(self, i: int, t: int, f: dict) -> None:
        if self.feat is None:
            self.feat = {k: np.zeros((self.n, self.T) + v.shape[1:], v.dtype) for k, v in f.items()}
        for k, v in f.items():
            self.feat[k][i, t] = v[0]

    def action_change_rate(self) -> float | None:
        """How often the (x, y, jump) input changes between consecutive decisions, mean over games."""
        rates = []
        for i in range(self.n):
            a = self.act[i][self.acted[i] & self.valid[i]].reshape(-1, 5)
            if len(a) > 1:
                rates.append(float(np.any(a[1:, :3] != a[:-1, :3], axis=1).mean()))
        return float(np.mean(rates)) if rates else None


@dataclass
class EnvTrack:
    """Per-game bookkeeping of the PPO collector."""

    step: Step | None = None
    stats: EpisodeStats = field(default_factory=EpisodeStats)
    stage: str = "L"
    last_food: float | None = 0.0
    last_pos: np.ndarray | None = None
    last_room: int = -1
    awaiting_reset: bool = True             # a reset is requested and not confirmed: steps are invalid
    reset_from_episode: int | None = None   # the mod's episode id when the reset was requested
    requested: bool = False
    rung: int = -1
    relocating: bool = False


@dataclass
class Member:
    """One GRPO group member: game ``i`` played it from pass step ``s`` for ``n`` steps."""

    i: int
    s: int = 0
    n: int | None = None
    refill: bool = False
    cut: bool = False                       # still alive when the pass budget ran out: left out
    stats: EpisodeStats | None = None
    rec: dict | None = None


def gae(ro: Rollout, gamma: float, lam: float):
    adv = np.zeros_like(ro.rew)
    last = np.zeros(ro.n, np.float32)
    for t in reversed(range(ro.T)):
        nonterm = 1.0 - ro.next_start[:, t].astype(np.float32)
        next_v = ro.boot if t == ro.T - 1 else ro.value[:, t + 1]
        delta = ro.rew[:, t] + gamma * next_v * nonterm - ro.value[:, t]
        last = delta + gamma * lam * nonterm * last
        last = np.where(ro.valid[:, t], last, 0.0)
        adv[:, t] = last
    return adv, adv + ro.value


def group_advantages(rew: np.ndarray, passes: list, gamma: float, adv: np.ndarray) -> float | None:
    """Group-normalise the return-to-go over every member of every pass that shared one seeded start.

    ``passes`` is a list of (offset, length, members). Members are aligned by their own episode step,
    so a refill that started later is compared with the others at the same step. Cut members are
    left out. Writes into ``adv``; returns the share of steps with no spread (no advantage), or None
    when no member finished.
    """
    L = min(p[1] for p in passes)
    returns, place = [], []
    for off, _, members in passes:
        for m in members:
            if m.cut:
                continue
            n = min(m.n, L - m.s)
            # float64: near-identical members differ by ~1e-9 in their shaped returns, rounding noise
            # that the 1e-6 epsilon would scale up. Steps after the member finished count as 0.
            r = np.zeros(L, np.float64)
            r[:n] = rew[m.i, off + m.s:off + m.s + n]
            G = np.zeros(L, np.float64)
            run = 0.0
            for k in reversed(range(L)):
                run = r[k] + gamma * run
                G[k] = run
            returns.append(G)
            place.append((m.i, off + m.s, n))
    if not returns:
        return None
    allG = np.stack(returns, 0)
    mu, sd = allG.mean(0), allG.std(0)
    tie = sd < 1e-6                                   # no spread: no advantage, not noise / eps
    for G, (i, t0, n) in zip(returns, place):
        adv[i, t0:t0 + n] = np.where(tie[:n], 0.0, (G[:n] - mu[:n]) / (sd[:n] + 1e-6)).astype(np.float32)
    return float(tie.mean())


# ---- per-member temperature (GRPO) -------------------------------------------------------------
# The behaviour clone is nearly deterministic, so group members at one temperature act alike and
# leave little to compare. The members of a group run the shared network at temperatures spread
# geometrically from 1 to tau_max, each trained on-policy at its own temperature. tau_max is fitted
# after every iteration so the hottest member samples at the entropy floor.

def member_tau(tau_max: float, k: int, members: int) -> float:
    return float(tau_max ** (k / max(members - 1, 1)))


def fit_tau_max(logits: np.ndarray, mask: np.ndarray, target: float) -> float | None:
    """The temperature at which the policy's mean entropy over ``logits[mask]`` reaches ``target``
    (1 if it is already above). None without data."""
    lg = torch.as_tensor(logits[mask], dtype=torch.float32)
    if lg.shape[0] == 0:
        return None
    if lg.shape[0] > 4096:
        lg = lg[torch.randperm(lg.shape[0])[:4096]]

    def entropy(t):
        e = 0.0
        for part in split_logits(lg / t):
            lp = F.log_softmax(part, dim=-1)
            e = e - (lp.exp() * lp).sum(-1).sum(-1)
        return float(e.mean())
    if entropy(1.0) >= target:
        return 1.0
    lo, hi = 1.0, 2.0
    while entropy(hi) < target and hi < 1e3:
        lo, hi = hi, hi * 2.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if entropy(mid) < target else (lo, mid)
    return hi


# ---- recurrent state ---------------------------------------------------------------------------
# Each block's state is a small dataclass of tensors with the batch on dim 0.

def clone_state(s):
    return type(s)(*[x.clone() for x in s.__dict__.values()])


def state_to(s, dev):
    return type(s)(*[x.to(dev) for x in s.__dict__.values()])


def stack_states(states: list):
    t = type(states[0])
    return t(*[torch.cat([getattr(s, n) for s in states], 0) for n in states[0].__dict__])


def scatter_state(dst, src, j: int, i: int) -> None:
    """Copy batch row ``j`` of ``src`` into row ``i`` of ``dst``."""
    for name, x in src.__dict__.items():
        getattr(dst, name)[i] = x[j]
