"""Check whether a recorded session is usable as training data.

The check is *action-label alignment*, not replay determinism. Rain World is RNG-heavy: a replay
diverges even when labels are perfect, so reproducing a trajectory proves nothing, and failing to
reproduce one proves nothing either.

What can go wrong, and would silently corrupt every action label, is an off-by-one: storing
a_{t-1} on row t. The mod records the action by two independent routes, and their agreement
proves the alignment. As a secondary, informational check, horizontal input should cause
horizontal acceleration with near-zero lag.

    python -m rainbot.scripts.check_session <path-to-.obs or dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from rainbot.data import session as S

MAX_LAG = 5


def xcorr_lag(cause: np.ndarray, effect: np.ndarray, max_lag: int = MAX_LAG):
    """Correlation of `cause` against `effect` shifted by each lag; returns (lags, corrs).

    Positive lag means the effect follows the cause, which is the physical expectation.
    """
    cause = cause - cause.mean()
    lags, corrs = [], []
    for lag in range(-max_lag, max_lag + 1):
        a = cause[max_lag:-max_lag] if max_lag else cause
        b = effect[max_lag + lag: len(effect) - max_lag + lag] if max_lag else effect
        n = min(len(a), len(b))
        a2, b2 = a[:n], b[:n] - b[:n].mean()
        denom = np.linalg.norm(a2) * np.linalg.norm(b2)
        lags.append(lag)
        corrs.append(0.0 if denom == 0 else float(a2 @ b2 / denom))
    return np.asarray(lags), np.asarray(corrs)


def check(path: Path) -> bool:
    sess = S.load(path)
    f = sess.schema.fields
    n = len(sess)
    print(f"\n=== {sess.path.name} ===")
    print(f"game {sess.schema.game_version} | {n} ticks | {sess.duration_s / 60:.1f} min wall "
          f"| {sess.schema.obs_bytes}B/tick | {len(f)} fields")
    if n < 200:
        print("FAIL: too short to say anything")
        return False

    obs = np.asarray(sess.obs)   # materialise; sessions are small enough
    ok = True

    # -- continuity --------------------------------------------------------------------
    d = np.diff(obs["tick"].astype(np.int64))
    gaps = int((d != 1).sum())
    print(f"tick continuity: {gaps} gaps" + ("" if not gaps else f"  (max jump {int(d.max())})"))
    if gaps > n * 0.001:
        print(f"FAIL: {gaps} tick gaps is too many; the recorder is dropping ticks")
        ok = False

    # -- alignment: exact, by cross-checking two independent captures -------------------
    # The mod records the action twice by different routes: `input_history` is read out of the
    # game's own 10-deep ring buffer *before* orig() runs, and `prev_action` is read from
    # input[0] *after*. They describe the same quantity one tick apart, so agreement proves the
    # alignment without any physics assumption.
    # Cross-correlating input against acceleration alone is too weak on human input: it is
    # heavily autocorrelated, the profile goes nearly flat and argmax lands on a neighbouring lag.
    hist = obs["input_history"].reshape(n, -1, 5).astype(np.int8)
    # v7 calls the label `action`; v6 recordings call it `prev_action`. In v7 a tick on which
    # checkInput did not run (in a pipe) has no ring-buffer entry, so only ticks where both
    # neighbours consumed input can be compared.
    label = "action" if "action" in obs.dtype.names else "prev_action"
    prev = obs[label].astype(np.int8)
    both = np.ones(n - 1, bool)
    if "consumed" in obs.dtype.names:
        c = obs["consumed"].astype(bool)
        both = c[1:] & c[:-1]
    agree = float((hist[1:, 0, :] == prev[:-1]).all(axis=1)[both].mean()) if both.any() else 0.0
    print(f"action labels: input_history[0] vs {label} (shift 1) agree on {agree * 100:.2f}%")
    if agree < 0.999:
        off = [lag for lag in (0, 2) if
               (hist[lag:, 0, :] == prev[:n - lag if lag else n]).all(axis=1).mean() > 0.999]
        print("FAIL: the two independent action captures disagree"
              + (f"; they align at shift {off[0]} instead, so the labels are off by "
                 f"{off[0] - 1} tick(s)" if off else ""))
        ok = False

    # Secondary, informational: input should visibly cause motion. Measured only while grounded
    # and not stunned: on a pole or mid-air, x does not map to horizontal acceleration.
    ax = np.diff(obs["vel0"][:, 0])
    x = prev[:-1, 0].astype(np.float64)
    m = (obs["standing"][:-1] > 0) & (obs["stun"][:-1] == 0)
    if m.sum() > 200:
        lags, corrs = xcorr_lag(x[m], ax[m])
        print(f"input->accel while grounded: peak lag {int(lags[np.argmax(corrs)]):+d}, "
              f"r={corrs.max():.3f} (informational; {m.sum()} ticks)")

    # -- coverage ----------------------------------------------------------------------
    alive = obs["alive"].astype(bool)
    asleep = obs["sleeping"].astype(bool)
    print(f"alive {alive.mean() * 100:.1f}% | sleeping {asleep.mean() * 100:.1f}% | "
          f"rooms visited {len(np.unique(obs['room']))}")

    present = (obs["cr_flags"] & 1).astype(bool)
    with_creatures = present.any(axis=1)
    print(f"creatures in view on {with_creatures.mean() * 100:.1f}% of ticks "
          f"(mean {present.sum(axis=1).mean():.2f} visible)")

    types = obs["cr_type"][present]
    names = sess.enum_names("creature_type")
    if len(types):
        uniq, cnt = np.unique(types, return_counts=True)
        top = sorted(zip(cnt, uniq), reverse=True)[:6]
        label = ", ".join(
            f"{names[u] if 0 <= u < len(names) else u}×{c}" for c, u in top)
        print(f"creature mix: {label}")

    # Only some AI classes declare a `behavior` field. Lizard/Scavenger/Vulture/Cicada/DropBug/
    # BigEel/JetFish do; Overseer, Snail and LanternMouse do not. So an unresolved slot is only
    # a problem when it belongs to a creature that should have one: report the breakdown by type.
    beh = obs["cr_behavior"][present]
    known = int((beh >= 0).sum())
    print(f"AI.behavior resolved on {100.0 * known / max(1, len(beh)):.1f}% of creature slots")
    if len(beh) and known < len(beh):
        miss = types[beh < 0]
        if len(miss):
            u, c = np.unique(miss, return_counts=True)
            lbl = ", ".join(f"{names[x] if 0 <= x < len(names) else x}×{k}"
                            for k, x in sorted(zip(c, u), reverse=True)[:5])
            print(f"  unresolved belong to: {lbl}  (expected for Overseer/Snail/LanternMouse)")

    # -- events ------------------------------------------------------------------------
    kinds = {}
    for e in sess.events:
        kinds[e.get("type", "?")] = kinds.get(e.get("type", "?"), 0) + 1
    print(f"events: {kinds if kinds else 'none'} | episodes: {len(sess.episodes())}")

    # A data set with no failures shows a model nothing of what dying looks like.
    if kinds.get("death", 0) == 0:
        print("NOTE: no deaths recorded. Fine for one session, but the corpus needs deaths and "
              "bad cycles or the world model never learns what dying looks like.")

    print("PASS" if ok else "FAIL")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help=".obs file, or a directory of them")
    args = ap.parse_args(argv)

    p = Path(args.path)
    paths = S.find_sessions(p) if p.is_dir() else [p]
    if not paths:
        print(f"no session-*.obs found under {p}", file=sys.stderr)
        return 1
    results = [check(q) for q in paths]
    print(f"\n{sum(results)}/{len(results)} sessions pass")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
