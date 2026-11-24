"""Native-resolution clips of the policy playing live.

Open-loop replays of recorded actions (scripts/replay.py) can diverge within a few tokens (the reset
and the game are not bit-identical run to run), and then the clip no longer shows what the agent
did. This instead loads a checkpoint and lets the policy play the recorded episode's start (its seeded
reset spec) closed-loop at 1x, sampling at its own temperature as in training, capturing every tick
with the mod's native frame ring, until the night ends (sleep, starve, death) or a token cap.

    python -m rainbot.scripts.demo --ckpt ../runs/grpo-bc6/ckpt.pt --instance 0 \
        --out ../runs/grpo-bc6/demos ../runs/grpo-bc6/episodes/X.npz ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import obs as O
from ..env.rain_env import RainEnv
from ..train.runner import PolicyRunner, load_agent
from .replay import PIECE_TOKENS, encode_previous, join_pieces, start_spec, wine_path


def demo(runner: PolicyRunner, e: RainEnv, rec: Path, out_dir: Path, cap: int) -> Path:
    r = np.load(rec, allow_pickle=False)
    spec = start_spec(r)
    work = out_dir / (rec.stem + "_parts")
    work.mkdir(parents=True, exist_ok=True)
    runner.reset(spec)
    runner.reset_memory()
    e.command("clip arm 400")
    pieces, ended, k = [], "", 0
    for k in range(cap):
        wire, _ = runner.act()
        runner.advance(wire)
        ends = [ev.get("reason", "") for ev in runner.step.events if ev.get("type") == "session_end"]
        last = (k + 1) % PIECE_TOKENS == 0
        if ends or last or k == cap - 1:
            raw = work / f"p{len(pieces):03d}.raw"
            e.command("clip dump " + wine_path(raw))
            encode_previous(pieces)
            pieces.append(raw)
            if ends:
                ended = ends[-1]
                break
            e.command("clip arm 400")
    # One more exchange so the last dump is processed.
    runner.advance(None)
    e.command("clip off")
    out = out_dir / f"{rec.stem}_live_{ended or 'cut'}_t{k + 1:03d}.mp4"
    join_pieces(pieces, work, out)
    print(f"{out}  ended '{ended}' after {k + 1} tokens", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="+", help="episode records (.npz) whose starts to play")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=1500, help="token cap per episode")
    ap.add_argument("--tau", type=float, default=1.0, help="sampling temperature (a GRPO member's T)")
    a = ap.parse_args(argv)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    vocab = O.Vocab()
    agent = load_agent(a.ckpt, "cpu", vocab)
    e = RainEnv(instance=a.instance, timeout=600, vocab=vocab)
    e.connect(retries=60, wait_s=5)
    runner = PolicyRunner(e, agent, tau=a.tau)
    runner.first()
    e.command("speed 1")
    e.command("render on")
    try:
        for p in a.episodes:
            demo(runner, e, Path(p), out_dir, a.cap)
    finally:
        e.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
