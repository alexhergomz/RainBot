"""Decode the on-screen tick marker from recorded video frames.

The mod draws a binary strip in the top-left corner of the game window (see
`mod/RainBot/TickMarker.cs`): a 13x2 grid of cells, white = 1.

    cell 0   always white  } calibration
    cell 1   always black  }
    cells 2..25            24 bits of tick, least-significant first

A decoded frame gives the exact tick it shows, so video can be used as training data. OBS runs
at 60 fps and the game at 40 Hz, so frames do not map to ticks in a fixed way, and a `wall_ms`
correlation drifts over a long session.

Cells are sampled at their centres and thresholded midway between the two calibration cells. The
decode tolerates scaling, gamma and soft compressed edges. It fails if the strip is cropped or covered.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_BITS = 24
CALIB_CELLS = 2
TOTAL_CELLS = CALIB_CELLS + DATA_BITS
COLUMNS = 13
ROWS = 2

#: Minimum separation between the white and black calibration cells for a decode to be trusted.
#: Below this the strip is probably not in frame at all, and a decoded number would be noise.
MIN_CONTRAST = 40.0


#: Every sampled cell must sit within this fraction of the white/black gap from one of the two
#: levels. A misaligned or mis-scaled sampling grid straddles cell boundaries and produces
#: intermediate greys, so this is what separates a real lock from a coincidental one.
MAX_IMPURITY = 0.25


@dataclass
class Decoded:
    tick: int | None
    contrast: float
    #: Worst normalised distance of any cell from the nearer of the two levels. Low is good.
    impurity: float = 1.0

    @property
    def ok(self) -> bool:
        return self.tick is not None

    @property
    def score(self) -> float:
        """Quality of a candidate alignment; higher is better."""
        return 0.0 if not self.ok else self.contrast * (1.0 - self.impurity)


def decode_frame(frame: np.ndarray, cell_px: int, x0: int = 0, y0: int = 0) -> Decoded:
    """Decode one RGB frame (H, W, 3) uint8.

    `cell_px` is the marker cell size *in video pixels*, which equals the mod's `cell_px`
    config only when the game window was captured 1:1. Use `calibrate` to find it.
    """
    # Crop to the strip before greyscaling. Converting a whole 1080p frame costs ~38 ms and
    # would put a 9-hour 60 fps recording at ~20 hours of decoding; the strip is a few thousand
    # pixels, so cropping first makes it sub-millisecond.
    h, w = frame.shape[:2]
    ymax = y0 + ROWS * cell_px
    xmax = x0 + COLUMNS * cell_px
    if x0 < 0 or y0 < 0 or ymax > h or xmax > w:
        return Decoded(None, 0.0)
    grey = frame[y0:ymax, x0:xmax, :3].mean(axis=2)

    vals = np.empty(TOTAL_CELLS, dtype=np.float64)
    half = cell_px // 2
    for i in range(TOTAL_CELLS):
        col, row = i % COLUMNS, i // COLUMNS
        y = row * cell_px + half
        x = col * cell_px + half
        # Average a patch rather than a single pixel: compression noise on one pixel is much
        # larger than on a small mean, and the cell centre is the safest place to average.
        # Scale the patch with the cell so large cells get a correspondingly better estimate.
        r = max(1, cell_px // 6)
        y1, y2 = max(0, y - r), min(grey.shape[0], y + r + 1)
        x1, x2 = max(0, x - r), min(grey.shape[1], x + r + 1)
        vals[i] = grey[y1:y2, x1:x2].mean()

    white, black = vals[0], vals[1]
    contrast = float(white - black)
    if contrast < MIN_CONTRAST:
        return Decoded(None, contrast)

    # Reject alignments that merely happen to put a light pixel at cell 0 and a dark one at
    # cell 1. In a true lock every cell reads as almost exactly white or almost exactly black;
    # a wrong scale or offset samples across boundaries and lands in between.
    impurity = float(np.max(np.minimum(np.abs(vals - white), np.abs(vals - black))) / contrast)
    if impurity > MAX_IMPURITY:
        return Decoded(None, contrast, impurity)

    mid = (white + black) / 2.0
    bits = vals[CALIB_CELLS:] > mid
    tick = int(sum(int(b) << i for i, b in enumerate(bits)))
    return Decoded(tick, contrast, impurity)


def _white_run_candidates(frame: np.ndarray, max_candidates: int = 600):
    """Candidate (cell, x0, y0) triples, from maximal horizontal runs of white.

    Keys on white rather than on "extreme" pixels: Rain World's palette is dark enough that
    ordinary background reads as pure black, so a black-inclusive mask swallows the frame.
    Calibration cell 0 is white and cell 1 to its right is always black, so cell 0's horizontal
    run is exactly one cell wide.
    """
    grey = frame[..., :3].mean(axis=2)
    white = (grey >= 231)
    if not white.any():
        return []

    h, _ = white.shape
    seen, cands = set(), []
    for y in range(0, h, 2):            # the strip is ROWS*cell tall; every 2nd row cannot miss it
        row = white[y]
        if not row.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
        for st, en in zip(edges[0::2], edges[1::2]):
            cell = int(en - st)
            if not (4 <= cell <= 64):
                continue
            key = (int(st), cell)
            if key in seen:
                continue
            seen.add(key)
            cx = int(st) + cell // 2
            y0 = y
            while y0 > 0 and white[y0 - 1, cx]:
                y0 -= 1
            cands.append((cell, int(st), y0))
            if len(cands) >= max_candidates:
                return cands
    return cands


def calibrate(frame: np.ndarray, max_candidates: int = 600):
    """Locate the strip in a single frame. Returns (cell_px, x0, y0) or None.

    Single-frame calibration is inherently guessable-wrong: any white rectangle of the right
    proportions can decode cleanly by coincidence, and a UI element elsewhere on screen will
    happily produce a confident, meaningless number. Prefer `calibrate_stable`, which settles it
    by requiring the geometry to behave like a clock across consecutive frames.
    """
    best = None
    for cell, x0, y0 in _white_run_candidates(frame, max_candidates):
        for dc in (0, -1, 1):
            c = cell + dc
            if c < 4:
                continue
            for dx in (0, -1, 1):
                for dy in (0, -1, 1):
                    d = decode_frame(frame, c, x0 + dx, y0 + dy)
                    if d.ok and (best is None or d.score > best[0]):
                        best = (d.score, c, x0 + dx, y0 + dy)
    return None if best is None else (best[1], best[2], best[3])


def calibrate_stable(frames, sim_hz: float = 40.0, fps: float | None = None,
                     min_decoded: float = 0.9):
    """Find the geometry that behaves like a clock across `frames` (consecutive, in order).

    The marker is drawn at the game window's origin at a fixed cell size, so its geometry is fixed
    for a capture: solve for it once and lock it. The true strip counts: its ticks advance, never go
    back, and advance at the game's rate. A static UI element gives a constant and a misaligned grid
    gives noise, so neither passes.

    Returns (cell_px, x0, y0) or None.
    """
    frames = list(frames)
    if not frames:
        return None

    cands = set()
    for f in frames[: min(len(frames), 5)]:
        for cell, x0, y0 in _white_run_candidates(f):
            for dc in (0, -1, 1):
                for dx in (0, -1, 1):
                    for dy in (0, -1, 1):
                        if cell + dc >= 4 and x0 + dx >= 0 and y0 + dy >= 0:
                            cands.add((cell + dc, x0 + dx, y0 + dy))

    expected = (fps / sim_hz) if fps else None
    best = None
    for geom in cands:
        ticks = [decode_frame(f, *geom) for f in frames]
        got = [t.tick for t in ticks if t.ok]
        if len(got) < min_decoded * len(frames):
            continue
        arr = np.asarray(got, dtype=np.int64)
        d = np.diff(arr)
        if len(d) == 0 or (d < 0).any():        # a clock never runs backwards
            continue
        if arr[-1] == arr[0]:                   # constant: a static UI element, not a counter
            continue
        # Ticks must advance at roughly the sim rate relative to the frame rate.
        if expected is not None:
            rate = float(d.mean())
            if not (0.5 * (1.0 / expected) <= rate <= 2.0 * (1.0 / expected) or
                    0.5 <= rate / max(1e-9, 1.0 / expected) <= 2.0):
                continue
        score = len(got) + float((d > 0).mean())
        if best is None or score > best[0]:
            best = (score, geom)
    return None if best is None else best[1]


def iter_frames(path: str | Path, width: int, height: int, fps: float | None = None,
                seek: float = 0.0):
    """Stream raw RGB frames from a video via ffmpeg. Yields (index, array).

    `seek` skips ahead in seconds (placed before -i, so ffmpeg seeks rather than decodes and
    discards). Frame indices restart at 0 from the seek point.
    """
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error"]
    if seek:
        cmd += ["-ss", str(seek)]
    cmd += ["-i", str(path)]
    if fps:
        cmd += ["-vf", f"fps={fps}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n = width * height * 3
    # Swallow ffmpeg's stderr: callers routinely stop reading early (calibration only needs the
    # first few frames), which makes ffmpeg complain about a broken pipe. That is expected, and
    # letting it print buries the actual output.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=n * 4)
    try:
        i = 0
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                return
            yield i, np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def probe(path: str | Path) -> dict:
    """Video dimensions and frame rate, via ffprobe."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json", str(path)])
    doc = json.loads(out)
    s = doc["streams"][0]
    num, _, den = s["avg_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    frames = int(s.get("nb_frames", 0) or 0)
    # Fragmented MP4 (Steam's rolling recordings) reports nb_frames as N/A, so the frame count
    # comes from the container duration. A zero count would limit marker calibration to the first
    # seconds, which are still on the main menu.
    duration = float(doc.get("format", {}).get("duration") or 0.0)
    if not frames and duration and fps:
        frames = int(duration * fps)
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": fps, "frames": frames, "duration": duration or (frames / fps if fps else 0.0)}
