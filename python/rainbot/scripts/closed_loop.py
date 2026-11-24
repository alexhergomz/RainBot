"""Closed-loop comparison of cloned policies on the live game.

Each checkpoint plays the same recorded starts (episode records with their start tile, for example
rung-0 starts of a GRPO run) at temperature 1 (what the coldest GRPO member samples) until the night
ends or a token cap. Reported per checkpoint: (x, y, jump) input changes per decision, by position in
the 4-decision chunk (0 = chunk boundary), against the humans' 0.7%; fed sleeps; ends.

The held-out cloning evaluation (scripts/bc_jitter.py) is open-loop in observations: the trunk sees
the human's consumed actions, not the policy's, which inflates flips at chunk boundaries. Here the
trunk sees what the policy did.

    python -m rainbot.scripts.closed_loop --instance 0 \
        --starts '../runs/grpo-ar/episodes/it001*_g0_env0_*.npz' \
        ../runs/bc-ar/bc.pt ../runs/bc-chunk/bc.pt ../runs/bc-kf/bc.pt
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np

from .. import obs as O
from ..env.rain_env import RainEnv
from ..train.runner import PolicyRunner, load_agent
from .replay import start_spec


def spec_of(rec: str) -> str:
    return start_spec(np.load(rec))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--starts", required=True, help="glob of episode records whose starts to play")
    ap.add_argument("--n", type=int, default=8, help="starts to use")
    ap.add_argument("--cap", type=int, default=1200, help="token cap per episode")
    ap.add_argument("--instance", type=int, default=0)
    a = ap.parse_args(argv)
    recs = sorted(glob.glob(a.starts))
    specs = list(dict.fromkeys(spec_of(r) for r in recs))[:a.n]
    print(f"{len(specs)} starts", flush=True)
    for ck in a.ckpts:
        vocab = O.Vocab()
        agent = load_agent(ck, "cpu", vocab)
        e = RainEnv(instance=a.instance, timeout=600, vocab=vocab)      # a fresh link per checkpoint (its vocab)
        e.connect(retries=60, wait_s=5)
        runner = PolicyRunner(e, agent)
        runner.first()
        for c in ("render on", "autopilot on", "endings freeze", "speed 16"):
            e.command(c)
        pos = np.zeros(4, int)
        n_dec, fed, ends = 0, 0, {}
        for spec in specs:
            runner.reset(spec)
            runner.reset_memory()
            dec, end = [], "cut"
            for _ in range(a.cap):
                wire, _ = runner.act()
                dec.append(np.asarray(wire).reshape(4, 5))
                runner.advance(wire)
                ev = [x for x in runner.step.events if x.get("type") == "session_end"]
                if ev:
                    end = ev[-1].get("reason", "?")
                    last = runner.step.rows[runner.step.n_real - 1]
                    if end == "sleep" and float(last["food_in_stomach"]) >= float(last["food_to_hibernate"]):
                        fed += 1
                    break
            ends[end] = ends.get(end, 0) + 1
            d = np.concatenate(dec)
            ch = (d[1:, :3] != d[:-1, :3]).any(1)
            idx = np.arange(1, len(d)) % 4
            for k in range(4):
                pos[k] += ch[idx == k].sum()
            n_dec += len(d) - 1
            print(f"  {ck}: {spec.split()[0]} ended {end} after {len(dec)} tokens", flush=True)
        print(json.dumps({"ckpt": ck, "changes_per_decision": round(pos.sum() / max(n_dec, 1), 4),
                          "changes_by_position": pos.tolist(), "fed_sleeps": fed, "episodes": len(specs),
                          "ends": ends}), flush=True)
        e.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
