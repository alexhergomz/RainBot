"""Replay a training episode at normal speed and native resolution.

The trainer saves a record for every episode with an event (`runs/<run>/episodes/*.npz`): the
seeded reset spec and every action it sent. This plays it back open-loop on its own game instance
at 1x and captures every tick with the mod's native frame ring (`clip arm` / `clip dump`), in
pieces, into one MP4 at 40 fps (real time). The game is not fully deterministic (same seed and
actions match for ~400 ticks, then drift by fractions of a unit), so the replay compares the
slugcat's position with the recorded one every token and reports where it diverged, also in the
file name.

It renders only while the trainer is collecting (`runs/<run>/phase`): the learner and a rendering
game busy on the GPU together can crash the machine. While the trainer updates, the replay stops
sending actions, which freezes its game in lockstep.

    python -m rainbot.scripts.replay --run-dir ../runs/grpo-bc2 --latest 3 [--instance 4]
    python -m rainbot.scripts.replay --run-dir ../runs/grpo-bc2 runs/grpo-bc2/episodes/X.npz
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

from ..env.rain_env import RainEnv

NEUTRAL = np.zeros((4, 5), np.int8)
PIECE_TOKENS = 24            # 192 ticks per dump (~800 MB raw at 1366x768, encoded and deleted at once)
DIVERGE_PX = 20.0            # one tile


def wine_path(p: Path) -> str:
    return "Z:" + str(p.resolve()).replace("/", "\\")


def wait_collect(run_dir: Path | None, e: RainEnv) -> None:
    if run_dir is None:
        return
    ph = run_dir / "phase"
    while ph.exists() and ph.read_text().strip() == "update":
        e.keepalive(True)
        time.sleep(1.0)
    e.keepalive(False)


def start_spec(r) -> str:
    """The record's reset spec, with its recorded start tile when it has one: the reset's own tile
    choice can change between mod versions."""
    spec = str(r["spec"])
    if "start_tile" in r.files and int(r["start_tile"][0]) >= 0 and "pos=random" in spec:
        spec = spec.replace("pos=random", f"pos={int(r['start_tile'][0])},{int(r['start_tile'][1])}")
    return spec


def encode_piece(raw: Path, mp4: Path, fps: int = 40) -> int:
    for _ in range(600):                                      # the mod writes it on its next tick
        if raw.exists() and raw.stat().st_size >= 12:
            with open(raw, "rb") as fh:
                n, w, h = (int(x) for x in np.frombuffer(fh.read(12), np.int32))
            if raw.stat().st_size >= 12 + n * w * h * 3 + 4 * n:
                break
        time.sleep(0.1)
    else:
        raise RuntimeError(f"dump {raw} never completed")
    if n == 0:
        raw.unlink(missing_ok=True)
        return 0
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)]
    with open(raw, "rb") as fh:
        fh.seek(12)
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        left = n * w * h * 3
        while left > 0:
            chunk = fh.read(min(left, 1 << 24))
            if not chunk:
                break
            p.stdin.write(chunk)
            left -= len(chunk)
        p.stdin.close()
        p.wait(timeout=600)
    raw.unlink(missing_ok=True)
    return n


def encode_previous(pieces: list) -> None:
    """Encode the previous piece (the game has written it by this exchange) and drop its raw:
    ~600 MB per piece at 1366x768, and a whole-cycle episode has ~57 pieces, so keeping the raws
    until the end fills the disk."""
    if pieces and pieces[-1] is not None and pieces[-1].suffix == ".raw":
        mp4 = pieces[-1].with_suffix(".mp4")
        pieces[-1] = mp4 if encode_piece(pieces[-1], mp4) > 0 else None


def join_pieces(pieces: list, work: Path, out: Path) -> None:
    """Encode the pieces still raw, concatenate all into ``out`` and delete the parts."""
    mp4s = []
    for raw in pieces:
        if raw is None:
            continue
        if raw.suffix == ".mp4":
            mp4s.append(raw)
            continue
        mp4 = raw.with_suffix(".mp4")
        if encode_piece(raw, mp4) > 0:
            mp4s.append(mp4)
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{m.resolve()}'\n" for m in mp4s))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True)
    for m in mp4s:
        m.unlink(missing_ok=True)
    lst.unlink(missing_ok=True)
    work.rmdir()


def replay(e: RainEnv, rec_path: Path, out_dir: Path, run_dir: Path | None) -> Path:
    r = np.load(rec_path, allow_pickle=False)
    spec, wire, acted = start_spec(r), r["wire"], r["acted"]
    # Records of members cut by a pass budget (episodes/cut/) keep only what a replay needs.
    pos = r["pos"] if "pos" in r.files else None
    room = r["room"] if "room" in r.files else None
    events = [str(x) for x in r["events"]] if "events" in r.files else []
    work = out_dir / (rec_path.stem + "_parts")
    work.mkdir(parents=True, exist_ok=True)

    wait_collect(run_dir, e)
    st = e._last
    e.request_reset(spec)
    ep0 = int(st.rows[0]["episode"])
    for _ in range(4000):
        if st.needs_action:
            e.send_actions(NEUTRAL)
        st = e.recv_step()
        if int(st.rows[0]["episode"]) != ep0:
            break
    e.command("clip arm 400")
    pieces, diverged, max_dev = [], None, 0.0
    for k in range(len(wire)):
        wait_collect(run_dir, e)
        if st.needs_action:
            e.send_actions(wire[k] if acted[k] else NEUTRAL)
        st = e.recv_step()
        last = st.rows[st.n_real - 1]
        dev = float(np.linalg.norm(np.asarray(last["pos0"], np.float32) - pos[k])) if pos is not None else 0.0
        if room is not None and int(last["room"]) != int(room[k]):
            dev = float("inf")
        max_dev = max(max_dev, dev)
        if diverged is None and dev > DIVERGE_PX:
            diverged = k
        if (k + 1) % PIECE_TOKENS == 0 or k == len(wire) - 1:
            raw = work / f"p{len(pieces):03d}.raw"
            e.command("clip dump " + wine_path(raw))
            if k < len(wire) - 1:
                e.command("clip arm 400")
            encode_previous(pieces)
            pieces.append(raw)
    # One more exchange so the last dump is processed.
    if st.needs_action:
        e.send_actions(NEUTRAL)
    st = e.recv_step()
    e.command("clip off")
    e._last = st
    tag = "matched" if diverged is None else f"diverged_t{diverged:03d}"
    out = out_dir / f"{rec_path.stem}_{'_'.join(ev.split(':')[1] for ev in events[:2])}_{tag}.mp4"
    join_pieces(pieces, work, out)
    print(f"{out}  events {events}  max position error {max_dev:.1f} px"
          + ("" if diverged is None else f", diverged at token {diverged}"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="*", help="episode records (.npz); default: the latest in the run")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--latest", type=int, default=1, help="replay the N most recent records")
    ap.add_argument("--prefer", default="hibernate,shelter_fed,pip",
                    help="event kinds, most wanted first, for choosing among the latest records")
    ap.add_argument("--instance", type=int, default=4)
    ap.add_argument("--out", default="", help="output dir (default: <run>/replays)")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "replays"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = [Path(p) for p in args.episodes]
    if not recs:
        allr = sorted((run_dir / "episodes").glob("*.npz"), key=lambda p: p.stat().st_mtime)
        kinds = args.prefer.split(",")

        def rank(p):
            evs = [str(x).split(":")[1] for x in np.load(p)["events"]]
            best = min((i for i, kk in enumerate(kinds) if any(ev.startswith(kk) for ev in evs)),
                       default=len(kinds))
            return (best, -p.stat().st_mtime)
        recs = sorted(allr[-500:], key=rank)[:args.latest]
    e = RainEnv(instance=args.instance, timeout=600)
    e.connect(retries=60, wait_s=5)
    e._last = e.recv_step()
    e._t_recv_done = 0.0
    e.command("speed 1")
    try:
        for p in recs:
            replay(e, p, out_dir, run_dir)
    finally:
        e.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
