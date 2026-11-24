"""Short video clips of useful moments during training.

Built from the frames the agent already receives (one 160x96 frame per 8-tick step = 5 fps of game
time), so it costs no extra capture. When an event is marked (a pip eaten), the clip keeps the
``before`` steps that led up to it and records ``after`` more, then encodes an MP4 with ffmpeg,
upscaled 4x with nearest-neighbour so the pixels stay crisp. Bounded: at most ``per_iter`` clips per
iteration and ``max_mb`` on disk (oldest deleted first), because disk space is limited.
"""
from __future__ import annotations

import collections
import subprocess
from pathlib import Path

import numpy as np


class ClipRecorder:
    def __init__(self, out_dir: Path, n_envs: int, before: int = 48, after: int = 16,
                 per_iter: int = 2, max_mb: float = 200.0, scale: int = 4, fps: int = 5):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.buf = [collections.deque(maxlen=before + after) for _ in range(n_envs)]
        self.pending: list[list] = [[] for _ in range(n_envs)]       # [steps_left, label]
        self.after, self.per_iter, self.max_mb, self.scale, self.fps = after, per_iter, max_mb, scale, fps
        self.saved_this_iter = 0
        self.saved: list[Path] = []

    def new_iteration(self) -> None:
        self.saved_this_iter = 0

    def push(self, i: int, frame: np.ndarray) -> None:
        self.buf[i].append(frame[..., :3])          # RGB only (frames also carry the distance channel)
        for p in list(self.pending[i]):
            p[0] -= 1
            if p[0] <= 0:
                self.pending[i].remove(p)
                self._write(i, p[1])

    def mark(self, i: int, label: str, priority: bool = False) -> None:
        """Queue a clip. ``priority`` events (a stage's goal: entering a shelter fed, hibernating)
        get their own budget of ``per_iter`` more, so routine pips cannot crowd them out."""
        cap = self.per_iter * (2 if priority else 1)
        if self.saved_this_iter + sum(len(p) for p in self.pending) >= cap:
            return
        self.pending[i].append([self.after, label])

    def flush(self) -> None:
        """Write anything still pending (end of a segment), with whatever 'after' frames exist."""
        for i, ps in enumerate(self.pending):
            for p in list(ps):
                ps.remove(p)
                self._write(i, p[1])

    def _write(self, i: int, label: str) -> None:
        frames = list(self.buf[i])
        if len(frames) < 4 or self.saved_this_iter >= 2 * self.per_iter:
            return
        h, w = frames[0].shape[:2]
        path = self.out / f"{label}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-",
               "-vf", f"scale={w * self.scale}:{h * self.scale}:flags=neighbor",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)]
        try:
            subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames),
                           check=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"clip {label} not written: {e}")
            return
        self.saved_this_iter += 1
        self.saved.append(path)
        print(f"clip saved: {path}")
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        files = sorted(self.out.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        while files and total > self.max_mb * 1e6:
            f = files.pop(0)
            total -= f.stat().st_size
            f.unlink()

