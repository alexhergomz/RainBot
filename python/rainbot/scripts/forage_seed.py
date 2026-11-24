"""Seed the food forecast's measured table (train/forage_fit.py) from older episode records.

Older records store only the actions, not the game's rows (eat counter, held item, food path), so
they are replayed open-loop on a running game (instance 0, speed 16) and the rows are kept up to the
replay's divergence from the recorded positions (the game's own nondeterminism: exact for ~100-1000
tokens). A replay cut at divergence is censored, not ended. The replayed rows
are kept in <run>/replay_rows/ so the seed can be refitted without the game.

    python -m rainbot.scripts.forage_seed --run ../runs/grpo-bc2 [--n 106]     # replay + fit
    python -m rainbot.scripts.forage_seed --run ../runs/grpo-bc2 --fit-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NEUTRAL = np.zeros((4, 5), np.int8)
DIVERGED_PX = 20.0          # one tile


def replay(records: list[Path], out_dir: Path) -> None:
    from ..env.rain_env import RainEnv
    out_dir.mkdir(parents=True, exist_ok=True)
    e = RainEnv(instance=0, timeout=300)
    e.connect(retries=60, wait_s=5)
    st = e.recv_step()
    e.command("speed 16")
    for p in records:
        o = out_dir / (p.stem + ".npz")
        if o.exists():
            continue
        r = np.load(p)
        spec, wire, acted, pos, room = str(r["spec"]), r["wire"], r["acted"], r["pos"], r["room"]
        for _ in range(6000):                  # settle: the mod refuses to reset a dead slugcat
            last = st.rows[st.n_real - 1]
            if int(last["alive"]) and not int(last["rain_hit"]) and st.needs_action:
                break
            if st.needs_action:
                e.send_actions(NEUTRAL)
            st = e.recv_step()
        e.request_reset(spec)
        ep0 = int(st.rows[0]["episode"])
        for _ in range(4000):
            if st.needs_action:
                e.send_actions(NEUTRAL)
            st = e.recv_step()
            if int(st.rows[0]["episode"]) != ep0:
                break
        if int(st.rows[0]["episode"]) == ep0:
            print("reset failed", p.name, flush=True)
            continue
        rows, dev = [], []
        for k in range(len(wire)):
            if st.needs_action:
                e.send_actions(wire[k] if acted[k] else NEUTRAL)
            st = e.recv_step()
            last = st.rows[st.n_real - 1]
            rows.append(last.copy())           # one row per token, as the trainer's records store
            d = float(np.linalg.norm(np.asarray(last["pos0"], np.float32) - pos[k]))
            dev.append(np.inf if int(last["room"]) != int(room[k]) else d)
            if dev[-1] > DIVERGED_PX:
                break
        dev = np.array(dev)
        keep = int(np.argmax(dev > DIVERGED_PX)) if (dev > DIVERGED_PX).any() else len(dev)
        np.savez_compressed(o, rows=np.stack(rows[:keep]) if keep else np.stack(rows[:1])[:0],
                            diverged=keep < len(wire), ended=str(r["ended"]) if "ended" in r.files else "")
        print(f"{p.stem}: {keep}/{len(wire)} tokens before divergence", flush=True)
    e.close()


def fit_seed(rows_dir: Path, out: Path) -> None:
    from ..train import forage_fit as FF
    eps, cens = [], []
    for f in sorted(rows_dir.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        if len(z["rows"]) < 2:
            continue
        eps.append(z["rows"])
        cens.append(bool(z["diverged"]) or str(z["ended"]) == "")
    tab = FF.fit(eps, censored=cens)
    tab.save(out)
    print(f"{len(eps)} replayed episodes, {tab.bites} bites: adj {tab.adj:.3f} held[40] {tab.held[40]:.3f} "
          f"held[0] {tab.held[0]:.3f} n_adj {tab.n_adj} n_held {sum(tab.n_held)} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=106, help="the most recent N records without rows")
    ap.add_argument("--fit-only", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    rows_dir = run / "replay_rows"
    if not a.fit_only:
        recs = []
        for p in sorted((run / "episodes").glob("it*.npz")):
            with np.load(p) as z:
                if "rows" not in z.files:
                    recs.append(p)
        replay(recs[-a.n:], rows_dir)
    fit_seed(rows_dir, run / "forage_seed.json")


if __name__ == "__main__":
    main()
