"""Map every frame of a recording to the sim tick it depicts.

    python -m rainbot.scripts.align_video capture.mkv [--out capture.ticks.npz]

Reads the on-screen tick marker (see rainbot/data/tickmarker.py) frame by frame and writes a
frame -> tick table. That table is what makes the video joinable to a `.obs` session log: with
it, a frame and a state row can be matched exactly rather than correlated through wall-clock,
which at 60 fps against a 40 Hz sim would leave roughly a tick of slop.

Reports rather than hides the awkward cases: frames where the marker was occluded or the game
was not running decode as gaps, and ticks that go backwards mean the capture spans a restart.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from rainbot.data import tickmarker as tm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="default: <video>.ticks.npz")
    ap.add_argument("--calib-frames", type=int, default=120,
                    help="how many frames to try before giving up on locating the marker")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = ap.parse_args(argv)

    path = Path(args.video)
    info = tm.probe(path)
    print(f"{path.name}: {info['width']}x{info['height']} @ {info['fps']:.3f} fps"
          + (f", {info['frames']} frames" if info["frames"] else ""))

    # Solve for the geometry once, against a window of consecutive frames. Single-frame
    # calibration is not reliable: any white rectangle of the right proportions decodes cleanly
    # by coincidence. Requiring the candidate to count (advance, never go back, at the sim rate)
    # rules those out.
    # Try several points in the recording. A capture often starts before the game runs or on a
    # menu, where the tick is frozen, and a frozen tick looks like a static UI element to the
    # stability check.
    duration = info["frames"] / info["fps"] if info["frames"] else 0.0
    seeks = [0.0] + ([duration * f for f in (0.1, 0.3, 0.5, 0.75)] if duration > 20 else [5.0, 15.0])
    geom = None
    for seek in seeks:
        window = []
        for i, frame in tm.iter_frames(path, info["width"], info["height"], seek=seek):
            window.append(frame.copy())
            if len(window) >= args.calib_frames:
                break
        if not window:
            continue
        geom = tm.calibrate_stable(window, fps=info["fps"])
        if geom is not None:
            print(f"marker located over {len(window)} frames at t={seek:.0f}s: "
                  f"cell={geom[0]}px at ({geom[1]},{geom[2]})")
            break
    if geom is None:
        print(f"FAILED: no tick marker found in the first {args.calib_frames} frames.\n"
              "  Was [marker] enabled in rainbot.bridge.cfg when this was recorded, and is the\n"
              "  game window's top-left corner inside the capture region?", file=sys.stderr)
        return 1

    cell, x0, y0 = geom
    ticks = []
    for i, frame in tm.iter_frames(path, info["width"], info["height"]):
        d = tm.decode_frame(frame, cell, x0, y0)
        ticks.append(d.tick if d.ok else -1)
        if args.max_frames and i + 1 >= args.max_frames:
            break

    t = np.asarray(ticks, dtype=np.int64)
    good = t >= 0
    print(f"\ndecoded {good.sum()}/{len(t)} frames ({good.mean() * 100:.2f}%)")
    if good.sum() < 2:
        print("FAILED: nothing decoded", file=sys.stderr)
        return 1

    v = t[good]
    back = int((np.diff(v) < 0).sum())
    print(f"tick range {v.min()} .. {v.max()}  ({v.max() - v.min()} ticks "
          f"= {(v.max() - v.min()) / 40 / 60:.1f} min of sim)")
    if back:
        print(f"WARNING: tick decreases at {back} points: the capture likely spans a game "
              f"restart, so it must be split before joining to a session log")

    # A 60 fps capture of a 40 Hz sim should repeat each tick ~1.5 times on average.
    uniq = len(np.unique(v))
    print(f"{uniq} distinct ticks over {good.sum()} frames "
          f"({good.sum() / max(1, uniq):.2f} frames per tick; expect ~{info['fps'] / 40:.2f})")

    out = Path(args.out) if args.out else path.with_suffix(".ticks.npz")
    np.savez_compressed(out, frame_tick=t, fps=info["fps"], cell=cell, x0=x0, y0=y0)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
