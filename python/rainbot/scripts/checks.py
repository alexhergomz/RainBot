"""Pre-training checks, sanity checks and probes.

    python -m rainbot.scripts.checks physics      [--instance 0] [--speeds 1,8]
    python -m rainbot.scripts.checks determinism  [--instance 0]
    python -m rainbot.scripts.checks throughput   [--instances 2] [--seconds 60]
    python -m rainbot.scripts.checks noop         [--tokens 3000]
    python -m rainbot.scripts.checks shuffle-rooms --ckpt runs/X/ckpt.pt
    python -m rainbot.scripts.checks memory-probe  --ckpt runs/X/ckpt.pt [--max-n 4]
    python -m rainbot.scripts.checks room-probe    --ckpt runs/X/ckpt.pt

Add ``--fake`` to run any of them against a FakeGame instead of Rain World (useful for testing
this file; the numbers only mean something against the real game).

Checks before training:
1. **Physics-preserving speed-up.** Replay one fixed input sequence from the same seeded reset at
   1x and at max speed; compare trajectories.
2. **Determinism under lockstep.** Same seed, same actions, same speed, twice; compare.
3. **Throughput.** Agent steps per second across instances, random actions.

Rain World is RNG-heavy and the mod seeds ``UnityEngine.Random`` on reset, but creature AI and
some systems keep state of their own, so exact equality is not expected everywhere. The checks
report *where* trajectories first diverge and by how much, which is what decides whether the
speed-up is safe: a divergence that begins with a creature is RNG; one that begins in the
slugcat's own body with nothing else around is physics.

Sanity checks:
* A NO-OP policy gets a negative return.
* Satiation (each extra pip pays less near the setpoint) is a property of the reward alone and is
  tested in ``tests/test_reward.py``.
* Shuffling room IDs at test time should hurt navigation; if it does not, the embeddings are not
  being used.

Probes:
* **Memory probe.** The agent passes a shelter, is moved N rooms away with the timer set low; the
  return rate as N grows says whether room embeddings + KDA memory are enough.
* **Room-embedding probe.** A linear probe from the backbone state to (room ID, previous room,
  distance to the nearest known shelter).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .. import obs as O
from ..env.rain_env import NEUTRAL, RainEnv
from ..train.curriculum import WorldGraph
from ..train.runner import PolicyRunner, Trace, load_agent

def _env(args, instance: int = 0, vocab=None) -> RainEnv:
    port = args.base_port + instance
    if args.fake:
        from ..env.fake_game import FakeGame
        FakeGame(port, seed=instance).start()
    e = RainEnv(instance=instance, port=port, vocab=vocab)
    e.connect(retries=30)
    e.command("autopilot on")
    return e


def _fixed_actions(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.zeros((n, O.DECISIONS, 5), np.int8)
    # Held directions with occasional jumps: momentum-heavy, like real play.
    d = rng.integers(-1, 2, size=(n // 4 + 1, 2)).repeat(4, axis=0)[:n]
    a[:, :, 0] = d[:, None, 0]
    a[:, :, 1] = d[:, None, 1]
    a[:, :, 2] = rng.random((n, O.DECISIONS)) < 0.15
    return a


_RUNNERS: dict[int, PolicyRunner] = {}


def _runner(env: RainEnv, **kw) -> PolicyRunner:
    """One runner per connection: it holds the pending step the mod is blocked on."""
    r = _RUNNERS.get(id(env))
    if r is None:
        r = _RUNNERS[id(env)] = PolicyRunner(env, **kw)
        r.first()
    else:
        for k, v in kw.items():
            setattr(r, k, v)
    return r


def _replay(env: RainEnv, spec: str, actions: np.ndarray, speed: float) -> np.ndarray:
    env.command(f"speed {speed}")
    r = _runner(env, action_fn=lambda t, s: actions[min(t, len(actions) - 1)])
    r.reset(spec)
    r.t = 0
    rows = []
    for t in range(len(actions)):
        rows.append(r.step.rows[:r.step.n_real].copy())
        r.advance(actions[t])
    return np.concatenate(rows)


def _compare(a: np.ndarray, b: np.ndarray, label: str) -> dict:
    n = min(len(a), len(b))
    d = np.linalg.norm(np.asarray(a["pos0"][:n], np.float64) - np.asarray(b["pos0"][:n], np.float64),
                       axis=-1)
    first = int(np.argmax(d > 1e-3)) if (d > 1e-3).any() else None
    out = {"check": label, "ticks": n, "max_pos_diff": float(d.max()), "first_divergence_tick": first}
    if first is not None:
        cr = np.asarray(a["cr_flags"][max(first - 40, 0):first + 1]) & 1
        out["creature_present_near_divergence"] = bool(cr.any())
        out["room_at_divergence"] = int(a["room"][first])
    return out


def cmd_physics(args):
    env = _env(args)
    acts = _fixed_actions(args.seed, args.tokens)
    spec = args.spec or f"room=shelter pos=den food=2 karma=1 creatures=clear peaceful=on seed={args.seed}"
    speeds = [float(s) for s in args.speeds.split(",")]
    runs = {sp: _replay(env, spec, acts, sp) for sp in speeds}
    base = runs[speeds[0]]
    return [_compare(base, runs[sp], f"speed {speeds[0]} vs {sp}") for sp in speeds[1:]]


def cmd_determinism(args):
    env = _env(args)
    acts = _fixed_actions(args.seed, args.tokens)
    spec = args.spec or f"room=shelter pos=den food=2 karma=1 creatures=keep seed={args.seed}"
    a = _replay(env, spec, acts, float(args.speeds.split(",")[0]))
    b = _replay(env, spec, acts, float(args.speeds.split(",")[0]))
    return [_compare(a, b, "same seed, same actions")]


def cmd_throughput(args):
    envs = [_env(args, i) for i in range(args.instances)]
    for e in envs:
        e.command(f"speed {args.speeds.split(',')[-1]}")
        e.command("render off")
    rng = np.random.default_rng(0)
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(len(envs))
    steps = list(pool.map(lambda e: e.recv_step(), envs))
    n, t0 = 0, time.time()
    while time.time() - t0 < args.seconds:
        def go(i):
            s = steps[i]
            if s.needs_action:
                a = rng.integers(-1, 2, size=(O.DECISIONS, 5)).astype(np.int8)
                a[:, 2:] = np.abs(a[:, 2:])
                envs[i].send_actions(a)
            return envs[i].recv_step()
        steps = list(pool.map(go, range(len(envs))))
        n += len(envs)
    dt = time.time() - t0
    return [{"check": "throughput", "instances": len(envs), "agent_steps_per_s": n / dt,
             "ticks_per_s": n * O.TICKS / dt, "sim_speed_x": n * O.TICKS / dt / 40.0}]


def cmd_noop(args):
    from .. import reward as R
    env = _env(args)
    r = _runner(env, action_fn=lambda t, s: NEUTRAL)
    r.reset(args.spec or "room=any pos=random food=0 creatures=keep")
    total, prev = 0.0, r.step.rows[r.step.n_real - 1]
    for _ in range(args.tokens):
        s = r.advance(NEUTRAL)
        last = s.rows[s.n_real - 1]
        total += float(R.reward(prev, last))
        prev = last
    return [{"check": "noop_return", "tokens": args.tokens, "return": total, "pass": total < 0}]


def _run_eval(env, agent, spec, tokens, shuffle=False):
    """Episodes from ``spec``; returns (hibernations, deaths, distinct rooms)."""
    if shuffle:
        tabs = env.vocab.tables.get("room", {})
        ids = list(tabs.values())
        perm = np.random.default_rng(0).permutation(ids)
        env.vocab.tables["room"] = dict(zip(tabs.keys(), perm.tolist()))
        env.index.set_world(env.index.first_room, env.world.get("rooms", []))
    r = PolicyRunner(env, agent)
    r.first()
    r.reset(spec)
    r.reset_memory()
    tr = r.run(tokens, reset_state=False)
    ev = [e for _, e in tr.events]
    sleeps = sum(e.get("reason") == "sleep" for e in ev if e.get("type") == "session_end")
    deaths = sum(e.get("reason") in ("death", "starve") for e in ev if e.get("type") == "session_end")
    rooms = len({int(x["room"]) for x in tr.rows})
    return sleeps, deaths, rooms


def cmd_shuffle_rooms(args):
    out = []
    for shuffle in (False, True):
        vocab = O.Vocab()
        agent = load_agent(args.ckpt, vocab=vocab)
        env = _env(args, vocab=vocab)
        tot = np.zeros(3)
        for k in range(args.episodes):
            tot += _run_eval(env, agent, f"room=any pos=random timer_left=6000 seed={k}", args.tokens, shuffle)
        env.close()
        out.append({"check": "shuffle_rooms" if shuffle else "baseline", "sleeps": tot[0], "deaths": tot[1],
                    "rooms": tot[2] / args.episodes})
    out.append({"check": "shuffle_hurts", "pass": out[1]["sleeps"] < out[0]["sleeps"]})
    return out


def cmd_memory_probe(args):
    vocab = O.Vocab()
    agent = load_agent(args.ckpt, vocab=vocab)
    env = _env(args, vocab=vocab)
    r = PolicyRunner(env, agent)
    r.first()
    g = WorldGraph(env.world)
    results = []
    for n in range(1, args.max_n + 1):
        ok = 0
        for k in range(args.episodes):
            shelter = sorted(g.shelters)[k % len(g.shelters)]
            # 1. The agent passes the shelter: start inside it and let it act for a while.
            r.reset(f"room={shelter} pos=den food=4 timer_left=20000 seed={k}")
            r.reset_memory()
            r.run(args.pass_tokens, reset_state=False)
            # 2. Moved N rooms away (memory kept), timer low.
            dist = _bfs(g, shelter)
            cands = sorted(rm for rm, d in dist.items() if d == n and rm not in g.shelters)
            if not cands:
                continue
            target = cands[k % len(cands)]
            r.reset(f"room={target} pos=random food=4 timer_left={args.timer_left} seed={k}")
            tr = r.run(args.timer_left // O.TICKS + 400, reset_state=False,
                       until=lambda s: any(e.get("type") == "session_end" for e in s.events))
            names = {i + env.index.first_room: nm for i, nm in enumerate(env.world.get("rooms", []))}
            reached = any(names.get(int(x["room"])) == shelter for x in tr.rows)
            slept = any(e.get("reason") == "sleep" for _, e in tr.events if e.get("type") == "session_end")
            ok += bool(reached and slept)
        results.append({"probe": "memory", "N": n, "return_rate": ok / max(args.episodes, 1)})
    return results


def _bfs(g: WorldGraph, src: str) -> dict:
    import collections
    dist, q = {src: 0}, collections.deque([src])
    while q:
        x = q.popleft()
        for y in g.adj.get(x, []):
            if y not in dist:
                dist[y] = dist[x] + 1
                q.append(y)
    return dist


def cmd_room_probe(args):
    """Ridge-regression linear probes from trunk features h to room id, previous room, shelter distance."""
    vocab = O.Vocab()
    agent = load_agent(args.ckpt, vocab=vocab)
    env = _env(args, vocab=vocab)
    r = PolicyRunner(env, agent)
    r.first()
    tr = Trace()
    for k in range(args.episodes):
        r.reset(f"room=any pos=random timer_left=8000 seed={k}")
        r.reset_memory()
        r.run(args.tokens, trace=tr, reset_state=False)
    H = np.stack(tr.h)
    rows = tr.array()
    room = np.asarray(rows["room"], np.int64)
    changes = np.nonzero(room[1:] != room[:-1])[0] + 1
    prev_room = room.copy()
    last_other = room[0]
    for t in range(len(room)):
        if t in set(changes):
            last_other = room[t - 1]
        prev_room[t] = last_other
    g = WorldGraph(env.world)
    names = env.world.get("rooms", [])
    dmap = g.dist_to_shelter()
    first = env.index.first_room
    dist = np.array([dmap.get(names[i - first], -1) if 0 <= i - first < len(names) else -1 for i in room],
                    np.float64)
    split = int(len(H) * 0.7)
    out = []
    for name, y, kind in (("room_id", room, "cls"), ("previous_room", prev_room, "cls"),
                          ("dist_to_shelter", dist, "reg")):
        out.append({"probe": name, **_linear_probe(H[:split], y[:split], H[split:], y[split:], kind)})
    return out


def _linear_probe(Xtr, ytr, Xte, yte, kind, lam=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr1 = np.c_[Xtr, np.ones(len(Xtr))]
    Xte1 = np.c_[Xte, np.ones(len(Xte))]
    A = Xtr1.T @ Xtr1 + lam * len(Xtr) * np.eye(Xtr1.shape[1])
    if kind == "reg":
        w = np.linalg.solve(A, Xtr1.T @ ytr)
        pred = Xte1 @ w
        ss = ((yte - yte.mean()) ** 2).sum() + 1e-9
        return {"r2": float(1 - ((yte - pred) ** 2).sum() / ss), "n_test": len(yte)}
    classes = np.unique(ytr)
    Y = (ytr[:, None] == classes[None]).astype(np.float64)
    W = np.linalg.solve(A, Xtr1.T @ Y)
    pred = classes[(Xte1 @ W).argmax(1)]
    chance = float(np.mean(yte == np.bincount(ytr).argmax())) if len(ytr) else 0.0
    return {"accuracy": float(np.mean(pred == yte)), "majority_baseline": chance, "n_test": len(yte),
            "classes": len(classes)}


def cmd_eat_rate(args):
    """Uniform random play from 'fruit in hand, eatCounter = N': how often does a pip arrive?

    Measures the eat-counter ladder: from a start counter the random policy clears often enough
    to learn from, up to the natural 40 it almost never clears.
    """
    env = _env(args)
    env.command("speed 16")
    env.command("render off")
    rng = np.random.default_rng(args.seed)

    def rand(t, s):
        a = rng.integers(-1, 2, size=(O.DECISIONS, 5)).astype(np.int8)
        a[:, 2:] = np.abs(a[:, 2:])
        return a
    agent = None
    if args.ckpt:
        vocab = env.vocab
        agent = load_agent(args.ckpt, device=args.device, vocab=vocab)
        env.index.from_schema(env.schema.enums)          # rebuild lookups with the loaded vocabulary
        if env.world:
            env.index.set_world(env.world.get("first_room_index", 0), env.world.get("rooms", []))
    r = _runner(env, action_fn=None if agent else rand)
    if agent is not None:
        r.agent, r.device, r.greedy = agent, args.device, args.greedy
    out = []
    for spec in args.counters.split(","):
        parts = spec.split(":")
        b, n = (parts[0], parts[1]) if len(parts) >= 2 else ("3", parts[0])
        speed = parts[2] if len(parts) >= 3 else "1"
        ate = 0
        for k in range(args.episodes):
            r.reset(f"room={args.room} pos=random food=0 hold=fruit eat_counter={n} bites={b} "
                    f"eat_speed={speed} creatures=clear timer_left=20000 seed={k}")
            f0 = int(r.step.rows[0]["food_in_stomach"]) * 4 + int(r.step.rows[0]["quarter_food"])
            if agent is not None:
                r.state = agent.initial_state(1, args.device)
            for _ in range(args.tokens):
                if agent is not None:
                    wire, _ = r.act(reset_state=False)
                    s = r.advance(wire)
                else:
                    s = r.advance(rand(0, None))
                last = s.rows[s.n_real - 1]
                f = int(last["food_in_stomach"]) * 4 + int(last["quarter_food"])
                if f > f0 or any(e.get("type") == "session_end" for e in s.events):
                    ate += f > f0
                    break
        out.append({"check": "eat_rate", "bites": int(b), "eat_counter": int(n), "eat_speed": int(speed),
                    "episodes": args.episodes, "ticks": args.tokens * O.TICKS, "ate": ate,
                    "rate": ate / args.episodes})
        print(json.dumps(out[-1]), flush=True)
    return []


COMMANDS = {"eat-rate": cmd_eat_rate, "physics": cmd_physics, "determinism": cmd_determinism,
            "throughput": cmd_throughput, "noop": cmd_noop, "shuffle-rooms": cmd_shuffle_rooms,
            "memory-probe": cmd_memory_probe, "room-probe": cmd_room_probe}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RainBot checks and probes")
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--base-port", type=int, default=54321)
    ap.add_argument("--instances", type=int, default=1)
    ap.add_argument("--speeds", default="1,8")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--spec", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--pass-tokens", type=int, default=150)
    ap.add_argument("--timer-left", type=int, default=2400)
    ap.add_argument("--counters", default="1:2,1:8,1:15,2:8,2:15,3:15,3:25,3:40",
                    help="bites:eat_counter[:eat_speed] for eat-rate")
    ap.add_argument("--room", default="SU_A22")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args(argv)
    for rec in COMMANDS[args.command](args):
        print(json.dumps(rec, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
