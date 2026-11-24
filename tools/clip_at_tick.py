"""Pull the video of a specific moment out of Steam's rolling recording buffer.

The watchdog reports *that* the agent wedged and what its state was; this turns that report
into something to look at. Given a tick, it finds the recording that contains it, decodes the
on-screen tick marker to locate the exact frame, and writes a clip or stills.

Reads the live buffer directly rather than the harvested archive, because the interesting
moment is usually the one that just happened: still being written, not yet harvested.

    python tools/clip_at_tick.py 27562                  # newest recording, +-6 s
    python tools/clip_at_tick.py 27562 --window 20 --stills 6
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harvest_recordings import decode_ticks, find_sessions, reconstruct   # noqa: E402
from rainbot.data import tickmarker as tm                                 # noqa: E402

SRC = Path.home() / ".steam/steam/userdata"
OUT = Path("/tmp/rainbot-clips")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tick", type=int)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--window", type=float, default=6.0, help="seconds either side")
    ap.add_argument("--stills", type=int, default=0, help="write N stills instead of a clip")
    ap.add_argument("--search", type=int, default=3, help="how many recent recordings to try")
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    sessions = sorted(find_sessions(a.src), key=lambda p: max(
        (q.stat().st_mtime for q in p.glob("*.m4s")), default=0), reverse=True)
    if not sessions:
        print("no recordings in the buffer")
        return 1

    tmp = a.out / "_full.mp4"
    for sess in sessions[:a.search]:
        print(f"{sess.name}: reconstructing")
        if not reconstruct(sess, tmp):
            continue
        info = tm.probe(tmp)
        ticks, geom = decode_ticks(tmp, info)
        if ticks is None or not len(ticks):
            print("  no tick marker decoded")
            continue
        have = ticks[ticks >= 0]
        if not len(have):
            continue
        print(f"  covers ticks {have.min()}-{have.max()} over {len(ticks)} frames")
        if not (have.min() <= a.tick <= have.max()):
            continue
        frame = int(np.argmin(np.abs(ticks - a.tick) + (ticks < 0) * 10 ** 9))
        t = frame / info["fps"]
        print(f"  tick {a.tick} -> frame {frame} at {t:.2f}s")
        start = max(0.0, t - a.window)
        if a.stills:
            step = max(1.0, (2 * a.window) / a.stills)
            paths = []
            for i in range(a.stills):
                dest = a.out / f"tick{a.tick}_{i}.png"
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                                "-ss", str(start + i * step), "-i", str(tmp),
                                "-frames:v", "1", str(dest)], check=False)
                paths.append(dest)
            print("\n".join(str(p) for p in paths if p.exists()))
        else:
            dest = a.out / f"tick{a.tick}.mp4"
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                            "-ss", str(start), "-i", str(tmp), "-t", str(2 * a.window),
                            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                            str(dest)], check=False)
            print(str(dest))
        return 0
    print(f"tick {a.tick} not found in the {a.search} most recent recordings")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
