"""Verify the wire protocol against the running game (docs/PROTOCOL.md), by measurement.

    python -m rainbot.scripts.live_check [--instance 0] [--room SU_A22] [--skip-death]

Checks, each printed PASS/FAIL with the numbers behind it:

1. cadence:     every exchange chunk has 8 rows, ``tick`` consecutive, ``step_tick`` 0..7, no
                wall-clock gap > 1 s inside a chunk, also in pipes.
2. injection:   on ticks where checkInput ran, the consumed action equals the decision sent for
                that tick (the game legitimately rewrites a few: super-launch, roll).
3. reset:       a randomised reset lands: room, food, karma, timer as requested; episode +1.
4. pipe:        after walking around, rows with ``in_shortcut=1`` keep arriving, and the first row
                in the new room has ``room_changed=1``.
5. freeze:      the game survives a 45 s freeze with pings (PPO updates freeze it that long).
6. speed:       ``speed 4`` with ``render off`` raises ticks per second.
7. death:       ``kill``: session_end(death) precedes a final chunk that needs no answer, then
                session_start with session+1 and episode unchanged.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .. import obs as O
from ..env.rain_env import NEUTRAL, RainEnv

RESULTS: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name:10s} {detail}", flush=True)


class Driver:
    def __init__(self, env: RainEnv):
        self.env = env
        self.step = env.recv_step()
        self.chunks: list[np.ndarray] = []
        self.sent: list[np.ndarray] = []

    def advance(self, actions: np.ndarray | None = None):
        a = NEUTRAL if actions is None else actions.astype(np.int8)
        if self.step.needs_action:
            self.env.send_actions(a)
            self.sent.append(a)
        else:
            self.sent.append(None)
        self.step = self.env.recv_step()
        self.chunks.append(self.step.rows[:self.step.n_real].copy())
        return self.step

    def wait_for(self, pred, max_steps: int, actions_fn=None):
        for k in range(max_steps):
            s = self.advance(actions_fn(k) if actions_fn else None)
            if pred(s):
                return s
        return None


def walk(k: int) -> np.ndarray:
    """Wander: hold a direction for a while, jump now and then."""
    a = np.zeros((O.DECISIONS, 5), np.int8)
    a[:, 0] = 1 if (k // 40) % 2 == 0 else -1
    a[:, 2] = (k % 7) == 0
    return a


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--room", default="SU_A22")
    ap.add_argument("--skip-death", action="store_true")
    args = ap.parse_args(argv)

    env = RainEnv(instance=args.instance)
    sch = env.connect(retries=60)
    print(f"connected: protocol {sch.protocol}, game {sch.game_version}, row {sch.obs_bytes} B, "
          f"instance {sch.raw.get('instance')}", flush=True)
    env.command("autopilot on")
    d = Driver(env)
    world = env.world
    report("world", bool(world.get("rooms")), f"{len(world.get('rooms', []))} rooms, "
           f"{len(world.get('shelters', []))} shelters, region {world.get('region')}")

    # 1-2. cadence + injection, with varied actions.
    rng = np.random.default_rng(0)
    for k in range(120):
        a = rng.integers(-1, 2, size=(O.DECISIONS, 5)).astype(np.int8)
        a[:, 2:] = np.abs(a[:, 2:])
        d.advance(a)
    full = [(c, s) for c, s in zip(d.chunks[1:], d.sent[1:]) if len(c) == 8]
    sizes = [len(c) for c in d.chunks[1:]]
    tick_ok = all(np.all(np.diff(c["tick"].astype(np.int64)) == 1) for c, _ in full)
    st_ok = all(list(c["step_tick"]) == list(range(8)) for c, _ in full)
    gaps = max(float(np.diff(c["wall_ms"].astype(np.int64)).max()) for c, _ in full)
    report("cadence", tick_ok and st_ok and gaps < 1000,
           f"chunks {len(full)}/{len(sizes)} full; consecutive ticks {tick_ok}; step_tick 0..7 {st_ok}; "
           f"max intra-chunk wall gap {gaps:.0f} ms")
    # The decisions sent at an exchange drive the rows of the chunk that exchange returns.
    match, total = 0, 0
    pairs = list(zip(d.sent, d.chunks))
    for sent, c in pairs:
        if sent is None or len(c) != 8:
            continue
        for i, row in enumerate(c):
            if not row["consumed"]:
                continue
            total += 1
            match += bool(np.all(np.asarray(row["action"]) == sent[i // 2]))
    report("injection", total > 0 and match / total > 0.9,
           f"consumed action == sent decision on {match}/{total} ticks ({100 * match / max(total, 1):.1f}%)")

    # 3. reset.
    ep0 = int(d.step.rows[0]["episode"])
    env.request_reset(f"room={args.room} pos=random food=2 karma=1 timer_left=3000 creatures=clear "
                      f"fruit=3 seed=7")
    t0 = time.time()
    s = d.wait_for(lambda s: int(s.rows[0]["episode"]) != ep0 or
                   any(e.get("type") == "reset_failed" for e in s.events), 400)
    evs = [e for c in [s.events] if s for e in c]
    done = [e for e in evs if e.get("type") == "reset_done"]
    if s is None or not done:
        report("reset", False, f"no reset_done after 400 steps; events {evs[-5:]}")
    else:
        r = s.rows[0]
        names = world.get("rooms", [])
        room_name = names[int(r["room"]) - env.index.first_room] if names else "?"
        left = int(r["cycle_length"]) - int(r["cycle_timer"])
        ok = room_name == args.room and int(r["food_in_stomach"]) == 2 and int(r["karma"]) == 1 \
            and abs(left - 3000) < 400 and int(r["episode"]) == ep0 + 1
        report("reset", ok, f"room {room_name}, food {int(r['food_in_stomach'])}, karma {int(r['karma'])}, "
               f"timer_left {left}, episode {ep0}->{int(r['episode'])}, {time.time() - t0:.1f}s; "
               f"event {done[0]}")

    # 4. pipe transit: wander until the room changes.
    n0 = len(d.chunks)
    s = d.wait_for(lambda s: bool(np.any(s.rows[:s.n_real]["room_changed"])), 1500, walk)
    if s is None:
        report("pipe", False, "never changed room while wandering (1500 steps); check manually")
    else:
        rows = np.concatenate(d.chunks[n0:])
        in_pipe = int(rows["in_shortcut"].sum())
        changed = int(rows["room_changed"].sum())
        gap = float(np.diff(rows["wall_ms"].astype(np.int64)).max())
        report("pipe", in_pipe > 0 and changed >= 1,
               f"{in_pipe} rows in a shortcut, {changed} room_changed rows, max wall gap {gap:.0f} ms")

    # 5. freeze with pings.
    env.keepalive(True)
    time.sleep(45)
    env.keepalive(False)
    try:
        d.advance()
        d.advance()
        report("freeze", True, "45 s frozen with pings; link still up")
    except Exception as e:
        report("freeze", False, f"link lost after a 45 s freeze: {e}")
        return 1

    # 6. speed.
    def rate(n=60):
        t = time.time()
        k0 = int(d.step.rows[-1]["tick"])
        for _ in range(n):
            d.advance()
        return (int(d.step.rows[-1]["tick"]) - k0) / (time.time() - t)
    r1 = rate()
    env.command("speed 4")
    env.command("render off")
    d.advance()
    r4 = rate()
    env.command("render on")
    env.command("speed 1")
    report("speed", r4 > 1.8 * r1, f"{r1:.0f} ticks/s at speed 1, {r4:.0f} ticks/s at speed 4 + render off")

    # 7. death flow.
    if not args.skip_death:
        sess0, ep_now = int(d.step.rows[0]["session"]), int(d.step.rows[0]["episode"])
        env.command("kill")
        seen_end, final_ok = False, False
        for _ in range(3000):
            st = d.step
            if any(e.get("type") == "session_end" for e in st.events):
                seen_end = True
                final_ok = not st.needs_action
            if seen_end and any(e.get("type") == "session_start" for e in st.events):
                break
            d.advance()
        r = d.step.rows[0]
        ok = seen_end and final_ok and int(r["session"]) == sess0 + 1 and int(r["episode"]) == ep_now
        report("death", ok, f"session_end seen {seen_end}, final chunk unanswered {final_ok}, "
               f"session {sess0}->{int(r['session'])}, episode {ep_now}->{int(r['episode'])}")

    bad = [n for n, ok, _ in RESULTS if not ok]
    print("ALL PASS" if not bad else f"FAILED: {', '.join(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
