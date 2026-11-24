"""The food forecast's success probabilities, measured on the agent's own episodes.

A forecast that values a fruit as certain to be eaten once reached (only discounted for time) pays
99.5% of the fruit's value for walking up to it and 0.5% for the 40+ ticks of holding grab still
and quiet, and a fruit carried around uneaten keeps its value until the cycle ends. So the forecast
is the expected discounted value of the fruit under the agent's current behaviour, in three
measured factors, each with plenty of data even when bites are rare:

    held:      V_held[r]  = E[g^(ticks to the food increase) | holding, r eat ticks remaining]
    adjacent:  V_adj      = E[g^(ticks to the food increase) | not holding, food path <= 1 tile]
    far:       A[bin(d)]  = E[g^(ticks to food path <= 1 or holding) | not holding, path d]
    value of a fruit at path d = A[bin(d)] * V_adj

r = (min(eat_counter, 40) + 15 * (bites left - 1)) / eat_speed (Player.cs: the counter idles at 40,
a bite below 1, 15 per further bite; the curriculum's eat_speed divides it). 0 when nothing
follows within the cycle. Each factor is fitted under the order the game imposes (pava): the
counter only reaches 0 through every value in between and a fruit is only reached through every
nearer distance, so the value cannot fall as either gets closer.

The table is refitted between iterations and fixed within one, so the shaping telescopes
exactly inside every rollout (a potential may change between rollouts).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import reward as R

PATH_EDGES = [0, 2, 4, 8, 16, 32, 64, 128, 256, 10 ** 9]     # log2 bins; bin 0 = adjacent (0-1 tiles)
R_MAX = 40 + 15 * 8                                           # eat ticks remaining, upper end


@dataclass
class ForageTable:
    held: list = field(default_factory=lambda: [1.0] * (R_MAX + 1))
    adj: float = 1.0
    reach: list = field(default_factory=lambda: [1.0] * (len(PATH_EDGES) - 1))
    n_held: list = field(default_factory=list)
    n_adj: int = 0
    n_reach: list = field(default_factory=list)
    episodes: int = 0
    bites: int = 0

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> "ForageTable":
        return cls(**json.loads(Path(path).read_text()))


def pava_nonincreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic regression, non-increasing (pool adjacent violators)."""
    blocks: list[list[float]] = []
    for yi, wi in zip(y, w):
        blocks.append([float(yi), float(wi), 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    return np.concatenate([[m] * int(c) for m, _, c in blocks]) if blocks else np.zeros(0)


def _monotone_fill(mean: np.ndarray, n: np.ndarray, first: float) -> np.ndarray:
    """Isotonic fit over the bins with data; a bin without data takes the nearer fitted value
    (the order's upper bound; `first` if none is nearer)."""
    out = np.empty(len(mean))
    seen = n > 0
    if seen.any():
        out[seen] = pava_nonincreasing(mean[seen], n[seen])
    last = first
    for i in range(len(mean)):
        if seen[i]:
            last = out[i]
        else:
            out[i] = last
    return out


def eat_ticks_remaining(rows) -> np.ndarray:
    f = lambda k: np.asarray(rows[k], dtype=np.float64)
    k = np.maximum(f("eat_speed"), 1.0)
    r = (np.minimum(f("eat_counter"), R.EAT_IDLE) + R.EAT_NEXT_BITE * np.maximum(f("held_bites") - 1, 0)) / k
    return np.clip(np.ceil(r), 0, R_MAX).astype(int)


def path_bin(path) -> np.ndarray:
    return np.clip(np.searchsorted(PATH_EDGES, np.asarray(path), side="right") - 1, 0, len(PATH_EDGES) - 2)


EVENT, ENDED, CUT = 1, 2, 0


def _time_to(T: np.ndarray, hit: np.ndarray, strictly_after: bool, end_kind: int):
    """Per token: ticks to the first s (> t, or >= t) with hit[s] and kind EVENT, else ticks to
    the segment's end and `end_kind` (ENDED: the cycle ended first; CUT: censored)."""
    dur = np.empty(len(T)); kind = np.empty(len(T), int)
    nxt = -1
    for t in range(len(T) - 1, -1, -1):
        if not strictly_after and hit[t]:
            nxt = t
        if nxt >= 0:
            dur[t], kind[t] = T[nxt] - T[t], EVENT
        else:
            dur[t], kind[t] = T[-1] - T[t], end_kind
        if strictly_after and hit[t]:
            nxt = t
    return dur, kind


def discounted_incidence(dur: np.ndarray, kind: np.ndarray, g: float) -> float:
    """E[g^T_event] with the cycle's end as a competing event and cuts as censoring (the
    Aalen-Johansen estimator of the event's cumulative incidence, discounted). Mass beyond the
    last observed time, if any, is unknown and counted as no event (a lower bound)."""
    if len(dur) == 0:
        return float("nan")
    ts, inv = np.unique(dur, return_inverse=True)
    d_e = np.bincount(inv, weights=(kind == EVENT).astype(float), minlength=len(ts))
    d_c = np.bincount(inv, weights=(kind == ENDED).astype(float), minlength=len(ts))
    d_all = np.bincount(inv, minlength=len(ts)).astype(float)
    at_risk = d_all[::-1].cumsum()[::-1]
    S, V = 1.0, 0.0
    for t, e, c, n in zip(ts, d_e, d_c, at_risk):
        V += S * (e / n) * g ** t
        S *= 1.0 - (e + c) / n
    return float(V)


def _carry(m: np.ndarray, n: np.ndarray, prior_v, prior_n):
    """Bins without observations in this window keep the previous table's value, weighted by the
    observations it came from (the latest evidence for that state). Otherwise one iteration that
    never reaches a fruit resets every bin to 1.0, "certain to be eaten"."""
    m, n = m.copy(), n.copy()
    if prior_v is not None and prior_n is not None and len(prior_n) == len(n):
        pv, pn = np.asarray(prior_v, float), np.asarray(prior_n, int)
        use = (n == 0) & (pn > 0)
        m[use], n[use] = pv[use], pn[use]
    return m, n


def fit(episodes: list, gamma: float = R.GAMMA_TOKEN, censored: list | None = None,
        prior: "ForageTable | None" = None) -> ForageTable:
    """`episodes`: per-token row arrays (the game's state at each token, as the records store).
    `prior`: the previous table; its values fill the states this window did not visit.
    `censored[i]`: episode i was cut (a length cap or a replay's divergence), not ended by the
    cycle: in its last cycle a token with no event before the cut has an unknown outcome
    (censored), not "never". A cycle that ended (sleep, death) is a real end: nothing after it is
    the same fruit, so it is a competing event."""
    g = gamma ** (1.0 / R.TICKS_PER_TOKEN)
    nb = len(PATH_EDGES) - 1
    obs_held = [([], []) for _ in range(R_MAX + 1)]
    obs_adj: tuple = ([], [])
    obs_reach = [([], []) for _ in range(nb)]
    bites = 0
    for i_ep, rows in enumerate(episodes):
        if len(rows) < 2:
            continue
        cut_tail = bool(censored[i_ep]) if censored is not None else False
        ep = np.asarray(rows["episode"])
        cuts = np.r_[0, np.where(np.diff(ep) != 0)[0] + 1, len(rows)]
        for a, b in zip(cuts[:-1], cuts[1:]):           # one cycle at a time: nothing crosses a boundary
            seg = rows[a:b]
            if len(seg) < 2:
                continue
            end_kind = CUT if (cut_tail and b == len(rows)) else ENDED
            T = np.asarray(seg["tick"], np.float64)
            food = (np.asarray(seg["food_in_stomach"], np.float64)
                    + np.asarray(seg["quarter_food"], np.float64) / 4.0)
            bite = np.r_[False, np.diff(food) > 0]
            bites += int(bite.sum())
            held = (np.asarray(seg["held_points"]) > 0) & (np.asarray(seg["held_bites"]) > 0)
            path = np.asarray(seg["food_path"])
            adj = ~held & (path >= 0) & (path <= 1)
            db, kb = _time_to(T, bite, True, end_kind)
            dr, kr = _time_to(T, held | adj, False, end_kind)
            r = eat_ticks_remaining(seg)
            for t in np.where(held)[0]:
                obs_held[r[t]][0].append(db[t]); obs_held[r[t]][1].append(kb[t])
            obs_adj[0].extend(db[adj]); obs_adj[1].extend(kb[adj])
            far = np.where(~held & (path >= 0))[0]
            for t, pb in zip(far, path_bin(path[far])):
                obs_reach[pb][0].append(dr[t]); obs_reach[pb][1].append(kr[t])

    def est(o):
        return discounted_incidence(np.asarray(o[0], float), np.asarray(o[1], int), g)
    n_held = np.array([len(o[0]) for o in obs_held]); n_reach = np.array([len(o[0]) for o in obs_reach])
    m_held = np.array([est(o) if len(o[0]) else 0.0 for o in obs_held])
    m_reach = np.array([est(o) if len(o[0]) else 0.0 for o in obs_reach])
    if prior is not None:
        m_held, n_held = _carry(m_held, n_held, prior.held, prior.n_held)
        m_reach, n_reach = _carry(m_reach, n_reach, prior.reach, prior.n_reach)
    held_v = _monotone_fill(m_held, n_held, first=1.0)
    n_adj = len(obs_adj[0])
    # No cap by held_v[40]: that bin is a full fruit's eating time only at the real game's speed.
    # With the eat-speed ladder a full fruit is r = 70/K ticks, so bin 40 is rarely observed, can
    # fit to 0, and the cap would zero every reachable fruit.
    if n_adj:
        adj_v = est(obs_adj)
    elif prior is not None and prior.n_adj > 0:
        adj_v, n_adj = prior.adj, prior.n_adj
    else:
        adj_v = held_v[0]
    reach_v = _monotone_fill(m_reach, n_reach, first=1.0)
    reach_v[0] = 1.0                                     # adjacent is where "reach" ends
    reach_v = np.minimum.accumulate(reach_v)
    return ForageTable(held=held_v.tolist(), adj=float(adj_v), reach=reach_v.tolist(),
                       n_held=n_held.astype(int).tolist(), n_adj=int(n_adj),
                       n_reach=n_reach.astype(int).tolist(),
                       episodes=len(episodes), bites=bites)


def load_records(paths) -> list:
    out = []
    for p in paths:
        try:
            d = np.load(p, allow_pickle=False)
            if "rows" in d.files and d["rows"].dtype.names:
                out.append(d["rows"])
        except (OSError, ValueError):
            continue
    return out
