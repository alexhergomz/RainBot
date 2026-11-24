"""Offline study of candidate homeostatic values on recorded episodes; reward.py is not changed.

Candidates (second argument):
  sumsq        deficits measured down from each variable's maximum with min-max scaling (game limits),
               d_i = (max_i - v_i) / (max_i - min_i) in [0, 1] (food in pips, karma in levels, air),
               and V = -sum d_i^2.
  lehmer2      the same deficits, V = -sum d_i^2 / sum d_i (share-weighted, contraharmonic mean).
  extras       necessity deficits (food to tonight's requirement), V = -sum d_i^2, plus the pips above
               the requirement valued by tomorrow's expected foraging, paid at a sleep.
  extras_real  as extras, the pips above the requirement valued as tomorrow's realized start.
Reward r = V(after) - V(before) (across a cycle boundary the food deficit is held), shaping potential
Phi = V(forecast state) - V(state) with the forecast of reward.py.

Compared with reward.py ("old" in the output). Reported: single events, episode returns by outcome,
and whether the ranking of members inside a group changes (what GRPO learns from).

    python -m rainbot.scripts.reward_study ../runs/grpo-kf/episodes
"""
from __future__ import annotations

import collections
import glob
import re
import sys

import numpy as np

from .. import reward as R

G = R.GAMMA_TOKEN


def new_deficits(rows, forecast: bool = False) -> np.ndarray:
    f = lambda k: np.asarray(rows[k], dtype=np.float64)
    food = f("food_in_stomach") + f("quarter_food") / 4.0
    karma = f("karma")
    if forecast:
        pips, karma_up = R._forecast(rows, G)
        food, karma = food + pips, karma + karma_up
    max_food = np.maximum(f("max_food"), 1.0)
    d_food = (max_food - np.minimum(food, max_food)) / max_food          # min 0 pips
    has_karma = (f("karma_cap") >= 0) & (f("karma") >= 0)
    d_karma = np.where(has_karma, (R.KARMA_MAX - np.minimum(karma, R.KARMA_MAX)) / R.KARMA_MAX, 0.0)
    d_air = 1.0 - np.clip(f("air_in_lungs"), 0.0, 1.0)
    return np.stack([d_food, d_karma, d_air], axis=-1)


VALUE = "sumsq"


def value(d: np.ndarray) -> np.ndarray:
    """'lehmer2': share-weighted mean -sum d^2 / sum d (not monotone: shrinking a small deficit can
    lower it). 'sumsq': -sum d^2, marginal value 2 d_i: each need weighted by its own size."""
    if VALUE == "sumsq":
        return -(d ** 2).sum(-1)
    s = d.sum(-1)
    return np.where(s > 0, -(d ** 2).sum(-1) / np.where(s > 0, s, 1.0), 0.0)


def need_deficits(rows, forecast: bool = False) -> np.ndarray:
    """Necessities: food to tonight's requirement (scaled by it), karma, air."""
    f = lambda k: np.asarray(rows[k], dtype=np.float64)
    food, karma = f("food_in_stomach") + f("quarter_food") / 4.0, f("karma")
    if forecast:
        pips, karma_up = R._forecast(rows, G)
        food, karma = food + pips, karma + karma_up
    req = np.maximum(f("food_to_hibernate"), 1.0)
    d_food = (req - np.minimum(food, req)) / req
    has_karma = (f("karma_cap") >= 0) & (f("karma") >= 0)
    d_karma = np.where(has_karma, (R.KARMA_MAX - np.minimum(karma, R.KARMA_MAX)) / R.KARMA_MAX, 0.0)
    d_air = 1.0 - np.clip(f("air_in_lungs"), 0.0, 1.0)
    return np.stack([d_food, d_karma, d_air], axis=-1)


def extras_worth_realized(rows, food) -> np.ndarray:
    """No estimate: tomorrow starts with the extras in the stomach (the night spends the requirement),
    so its food deficit is (req - e) / req instead of 1; worth 1 - ((req - e) / req)^2."""
    f = lambda k: np.asarray(rows[k], dtype=np.float64)
    req = np.maximum(f("food_to_hibernate"), 1.0)
    e = np.clip(np.minimum(food, f("max_food")) - req, 0.0, None)
    return 1.0 - ((req - np.minimum(e, req)) / req) ** 2


def extras_worth(rows, food) -> np.ndarray:
    """What the pips above tonight's requirement are worth tomorrow: the change in tomorrow's expected
    value at its night, -E[((req - min(req, e + K)) / req)^2] with K ~ Poisson(human hungry eating
    rate x cycle length) pips foraged tomorrow, discounted over the cycle to tomorrow's night.
    'extras_real' uses the realized value instead (extras_worth_realized)."""
    if VALUE == "extras_real":
        return extras_worth_realized(rows, food)
    from math import exp, factorial
    f = lambda k: np.asarray(rows[k], dtype=np.float64)
    req = np.maximum(f("food_to_hibernate"), 1.0)
    cyc = np.where(f("cycle_length") > 0, f("cycle_length"), 9000.0)
    e = np.clip(np.minimum(food, f("max_food")) - req, 0.0, None)
    out = np.zeros(len(e))
    for i in range(len(e)):
        mu = R.HUNGRY_PIPS_PER_TICK * cyc[i]
        rq = int(round(req[i]))
        pk = np.array([exp(-mu) * mu ** k / factorial(k) for k in range(rq + 1)])
        pk[-1] = max(0.0, 1.0 - pk[:-1].sum())            # K >= req lumped
        def ev(x):
            got = np.minimum(rq, x + np.arange(rq + 1))
            return -float((pk * ((rq - got) / rq) ** 2).sum())
        out[i] = (ev(e[i]) - ev(0.0)) * G ** (cyc[i] / R.TICKS_PER_TOKEN)
    return out


def p_sleep_disc(rows) -> np.ndarray:
    """Discounted P(a sleep tonight), from the karma forecast run with a sleep worth +1 and a death 0
    (karma 0, cap 1: up = 1, down = 0, and a malnourished night = 1 as well)."""
    if "food_path" not in (rows.dtype.names or ()):
        return np.zeros(len(rows))
    r2 = rows.copy()
    r2["karma"] = 0
    r2["karma_cap"] = 1
    _, q = R._forecast(r2, G)
    return np.clip(np.asarray(q, np.float64), 0.0, 1.0)


def ext_value(rows, forecast: bool = False, slept: bool | None = None) -> np.ndarray:
    f = lambda k: np.asarray(rows[k], dtype=np.float64)
    food = f("food_in_stomach") + f("quarter_food") / 4.0
    if forecast:
        food = food + R._forecast(rows, G)[0]
    q = (p_sleep_disc(rows) if VALUE != "extras_real" else np.zeros(len(rows))) if slept is None \
        else np.full(len(rows), 1.0 if slept else 0.0)
    return -(need_deficits(rows, forecast) ** 2).sum(-1) + q * extras_worth(rows, food)


def new_reward(a, b, boundary: bool) -> float:
    if VALUE in ("extras", "extras_real"):
        # The night is booked where karma changes (with frozen endings, one token after the session
        # changes): food held there too, the extras realized on a sleep, lost on a death.
        if int(b["karma"][0]) != int(a["karma"][0]):
            boundary = True
        if boundary:
            slept = int(b["karma"][0]) > int(a["karma"][0])
            d0, d1 = need_deficits(a), need_deficits(b).copy()
            d1[..., 0] = d0[..., 0]
            v0 = ext_value(a)
            fa = float(a["food_in_stomach"][0]) + float(a["quarter_food"][0]) / 4.0
            v1 = -(d1 ** 2).sum(-1) + (extras_worth(a, np.array([fa])) if slept else 0.0)
            return float((v1 - v0)[0])
        return float((ext_value(b) - ext_value(a))[0])
    d0, d1 = new_deficits(a), new_deficits(b).copy()
    if boundary:
        d1[..., 0] = d0[..., 0]
    return float((value(d1) - value(d0))[0])


def new_phi(rows) -> np.ndarray:
    if VALUE in ("extras", "extras_real"):
        return ext_value(rows, forecast=True) - ext_value(rows)
    return value(new_deficits(rows, forecast=True)) - value(new_deficits(rows))


def old_reward(a, b, boundary: bool) -> float:
    return float((R.reward_across_cycle(a, b) if boundary else R.reward(a, b))[0])


def episode(rows, ended: str) -> dict:
    n = len(rows)
    out = {}
    for name, rew, phi in (("old", old_reward, lambda r: R.potential(r, G)), ("new", new_reward, new_phi)):
        base = np.zeros(n - 1)
        for k in range(n - 1):
            a, b = rows[k:k + 1], rows[k + 1:k + 2]
            base[k] = rew(a, b, int(a["session"][0]) != int(b["session"][0]))
        ph = np.asarray(phi(rows), np.float64)
        disc = G ** np.arange(n - 1)
        end_phi = ph[-1] if ended else 0.0           # a cut closes with Phi = 0
        out[name] = {"base": float((base * disc).sum()),
                     "shaped": float((base * disc).sum() + G ** (n - 1) * end_phi - ph[0]),
                     "events": base}
    return out


def classify(z) -> str:
    fd, ended = np.asarray(z["food"]), str(z["ended"])
    ins = np.asarray(z["rows"]["in_shelter"])
    if ended == "sleep":
        return "fed sleep" if fd.max() >= 4 else "malnourished sleep"
    if ended == "death":
        return "ate enough, died" if fd.max() >= 4 else "died hungry"
    if ended == "starve":
        return "starved"
    return "hid in shelter (cut)" if ins.mean() > 0.5 else "cut elsewhere"


def main(path: str, variant: str = "sumsq") -> None:
    global VALUE
    VALUE = variant
    print(f"### candidate: {variant}")
    files = sorted(glob.glob(f"{path}/*.npz"))
    by_class = collections.defaultdict(lambda: {"old": [], "new": []})
    groups = collections.defaultdict(list)
    eat_old, eat_new = collections.defaultdict(list), collections.defaultdict(list)
    night_old, night_new = collections.defaultdict(list), collections.defaultdict(list)
    for f in files:
        try:
            z = np.load(f)
            rows = z["rows"]
        except Exception:
            continue
        if len(rows) < 3:
            continue
        cls = classify(z)
        res = episode(rows, str(z["ended"]))
        for k in ("old", "new"):
            by_class[cls][k].append((res[k]["base"], res[k]["shaped"]))
        m = re.search(r"(it\d+_g\d+)_env\d(?:_m\d+)?_(\d{8}-\d{6})", f)
        if m:
            groups[(m.group(1), m.group(2))].append((res["old"]["shaped"], res["new"]["shaped"]))
        fd = np.asarray(rows["food_in_stomach"], np.float64) + np.asarray(rows["quarter_food"]) / 4.0
        ka = np.asarray(rows["karma"])
        for k in range(len(rows) - 1):
            if ka[k + 1] != ka[k]:
                kind = "karma +1 (a sleep)" if ka[k + 1] > ka[k] else "karma -1 (a death)"
                night_old[kind].append(res["old"]["events"][k])
                night_new[kind].append(res["new"]["events"][k])
            elif fd[k + 1] - fd[k] >= 0.99:
                lvl = int(round(fd[k]))
                eat_old[lvl].append(res["old"]["events"][k])
                eat_new[lvl].append(res["new"]["events"][k])
    print("== a pip eaten, by stomach before it (mean reward; n)")
    for lvl in sorted(eat_old):
        print(f"   {lvl} -> {lvl + 1} pips: old {np.mean(eat_old[lvl]):+.4f}  new {np.mean(eat_new[lvl]):+.4f}  "
              f"(n={len(eat_old[lvl])})")
    print("== a night booked (mean reward at the boundary; n)")
    for kind in sorted(night_old):
        print(f"   {kind:18s}: old {np.mean(night_old[kind]):+.4f}  new {np.mean(night_new[kind]):+.4f}  "
              f"(n={len(night_old[kind])})")
    print("== discounted returns by outcome: base (shaped)")
    for cls, v in sorted(by_class.items(), key=lambda kv: -np.mean([x[1] for x in kv[1]["new"]])):
        o, n_ = np.array(v["old"]), np.array(v["new"])
        print(f"   {cls:22s} n={len(o):3d}  old {o[:, 0].mean():+.3f} ({o[:, 1].mean():+.3f})   "
              f"new {n_[:, 0].mean():+.3f} ({n_[:, 1].mean():+.3f})")
    agree = tot = 0
    for v in groups.values():
        if len(v) < 2:
            continue
        a = np.array(v)
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                if abs(a[i, 0] - a[j, 0]) < 1e-6 and abs(a[i, 1] - a[j, 1]) < 1e-6:
                    continue
                tot += 1
                agree += (np.sign(a[i, 0] - a[j, 0]) == np.sign(a[i, 1] - a[j, 1]))
    print(f"== pairs of members in a group ranked the same way (shaped return): "
          f"{agree}/{tot} = {agree / max(tot, 1):.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../runs/grpo-kf/episodes",
         sys.argv[2] if len(sys.argv) > 2 else "sumsq")
