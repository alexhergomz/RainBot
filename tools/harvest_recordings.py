"""Harvest Steam's rolling background recordings into an aligned, storage-bounded dataset.

Steam keeps a continuous buffer of recent gameplay as fragmented MP4 (an `init-stream0.m4s`
plus numbered chunks) under a per-session directory, and rotates it away. Run this periodically
and the footage becomes a permanent, growing video corpus at no cost to the play loop.

It:

* **Decodes the tick marker at native resolution, before any downscaling.** Alignment is then
  exact and stored as a small frame->tick table, so the archived video can be shrunk as far as
  storage demands without losing its correspondence to the state log.
* **Masks the marker out of the archive.** Once the mapping is extracted the strip is a small
  block of maximum-contrast pixels that a vision encoder could learn to read instead of the game.
* **Stays inside a storage budget**, pruning oldest-first: an unbounded archive grows by about
  1.2 GB per hour of play.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from rainbot.data import tickmarker as tm   # noqa: E402

APPID = "312520"
DEFAULT_SRC = Path.home() / ".steam/debian-installation/userdata"
DEFAULT_OUT = Path.home() / ".steam/steam/steamapps/common/Rain World/RainBotData/video"


def find_sessions(src: Path) -> list[Path]:
    out = []
    for base in src.glob("*/gamerecordings/video"):
        out.extend(sorted(base.glob(f"bg_{APPID}_*")))
    return out


def reconstruct(session: Path, dest: Path) -> bool:
    """Concatenate the fragmented stream into a playable file."""
    init = session / "init-stream0.m4s"
    chunks = sorted(session.glob("chunk-stream0-*.m4s"),
                    key=lambda p: int(re.search(r"(\d+)\.m4s$", p.name).group(1)))
    if not init.exists() or not chunks:
        return False
    with open(dest, "wb") as out:
        for part in [init] + chunks:
            out.write(part.read_bytes())
    return dest.stat().st_size > 0


def decode_ticks(path: Path, info: dict, calib_frames: int = 40):
    """(frame -> tick array, geometry) decoded at native resolution."""
    geom = None
    duration = info.get("duration") or (info["frames"] / info["fps"] if info["fps"] else 0)
    # Sample across the whole recording. A rolling buffer routinely opens on a menu or a
    # loading screen, where the tick is frozen and indistinguishable from a static overlay.
    seeks = ([duration * f for f in (0.3, 0.5, 0.7, 0.1, 0.9)] if duration > 60 else [0.0, 10.0])
    for seek in seeks:
        window = []
        for _, frame in tm.iter_frames(path, info["width"], info["height"], seek=seek):
            window.append(frame.copy())
            if len(window) >= calib_frames:
                break
        if window:
            geom = tm.calibrate_stable(window, fps=info["fps"])
            if geom is not None:
                break
    if geom is None:
        return None, None

    cell, x0, y0 = geom
    # Crop to the strip before piping: decoding whole 1080p frames only to read a 208x32 corner
    # moves ~60x more data than the job needs.
    w = tm.COLUMNS * cell + 4
    h = tm.ROWS * cell + 4
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
           "-vf", f"crop={w}:{h}:{max(0, x0 - 2)}:{max(0, y0 - 2)}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=n * 8)
    ticks = []
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            f = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            d = tm.decode_frame(f, cell, 2, 2)
            ticks.append(d.tick if d.ok else -1)
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()
    return np.asarray(ticks, dtype=np.int64), geom


def transcode(src: Path, dest: Path, geom, height: int, fps: int, crf: int) -> bool:
    cell, x0, y0 = geom if geom else (16, 0, 0)
    mw, mh = tm.COLUMNS * cell + 4, tm.ROWS * cell + 4
    # drawbox blanks the marker: with the mapping already extracted it is only a spurious
    # high-contrast feature for a vision model to latch onto.
    vf = (f"drawbox=x={max(0, x0 - 2)}:y={max(0, y0 - 2)}:w={mw}:h={mh}:color=black:t=fill,"
          f"scale=-2:{height},fps={fps}")
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                        "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", str(crf), str(dest)], capture_output=True)
    return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def prune(out: Path, budget_gb: float) -> None:
    vids = sorted(out.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in vids)
    limit = budget_gb * (1 << 30)
    while total > limit and len(vids) > 1:
        victim = vids.pop(0)
        total -= victim.stat().st_size
        for ext in (".mp4", ".ticks.npz"):
            victim.with_suffix(ext).unlink(missing_ok=True)
        print(f"  pruned {victim.name} (over {budget_gb:.0f} GB budget)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--height", type=int, default=360,
                    help="archive height; 160x128 is what the encoder wants")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=30)
    ap.add_argument("--budget-gb", type=float, default=12.0)
    ap.add_argument("--min-age", type=float, default=120.0,
                    help="seconds a session must be idle before harvesting, "
                         "so a live recording is not read mid-write")
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    manifest_path = a.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    sessions = find_sessions(a.src)
    print(f"{len(sessions)} Rain World recordings in the buffer")
    tmp = a.out / "_tmp.mp4"
    harvested = 0

    for sess in sessions:
        key = sess.name
        if key in manifest:
            continue
        age = time.time() - max(p.stat().st_mtime for p in sess.glob("*.m4s"))
        if age < a.min_age:
            print(f"  {key}: still being written, skipping")
            continue
        print(f"  {key}: reconstructing")
        if not reconstruct(sess, tmp):
            manifest[key] = {"status": "empty"}
            continue
        try:
            info = tm.probe(tmp)
            ticks, geom = decode_ticks(tmp, info)
            if ticks is None:
                print("    no tick marker found; archiving without alignment")
                ticks = np.full(0, -1, dtype=np.int64)
            else:
                good = int((ticks >= 0).sum())
                print(f"    decoded {good}/{len(ticks)} frames ({good / max(1, len(ticks)) * 100:.1f}%)")
            dest = a.out / f"{key}.mp4"
            if not transcode(tmp, dest, geom, a.height, a.fps, a.crf):
                print("    transcode failed")
                manifest[key] = {"status": "transcode_failed"}
                continue
            np.savez_compressed(a.out / f"{key}.ticks.npz", frame_tick=ticks,
                                fps=info["fps"], geom=np.asarray(geom if geom else (0, 0, 0)))
            mb = dest.stat().st_size / (1 << 20)
            manifest[key] = {"status": "ok", "frames": int(len(ticks)),
                             "aligned": int((ticks >= 0).sum()), "mb": round(mb, 1),
                             "seconds": round(info["frames"] / max(info["fps"], 1), 1)}
            print(f"    archived {mb:.0f} MB")
            harvested += 1
        finally:
            tmp.unlink(missing_ok=True)

    manifest_path.write_text(json.dumps(manifest, indent=1))
    prune(a.out, a.budget_gb)
    total = sum(p.stat().st_size for p in a.out.glob("*.mp4")) / (1 << 30)
    print(f"harvested {harvested} | archive now {total:.2f} GB / {a.budget_gb:.0f} GB budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
