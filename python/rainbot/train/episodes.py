"""Episode records: every GRPO member is saved to ``<run>/episodes/`` as a compressed npz.

A record holds what an exact replay needs (the seeded reset spec, every action sent, the start tile,
the code and mod versions) and what learning without a replay needs (per-token reward as the learner
saw it, deficits, positions, rooms, food, the game's rows, events, how it ended). Frames are not
stored: scripts/replay.py regenerates them. About 4 KB per episode.

Members cut by the pass budget go to ``episodes/cut/`` with the replay fields only.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .rollout import Member, Rollout

MOD_DLL = Path.home() / ".steam/steam/steamapps/common/Rain World/BepInEx/plugins/RainBot.dll"


def new_record() -> dict:
    return {"wire": [], "acted": [], "pos": [], "room": [], "food": [], "events": [], "rows": []}


def _versions() -> tuple[str, str]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    mod = hashlib.md5(MOD_DLL.read_bytes()).hexdigest() if MOD_DLL.exists() else ""
    return commit, mod


class EpisodeWriter:
    def __init__(self, run_dir: Path):
        self.dir = run_dir / "episodes"
        self._versions: tuple[str, str] | None = None
        self._disk_warned = False

    def save(self, members: list[Member], *, spec: str, stage: str, rung: int, iteration: int,
             tag_prefix: str, ro: Rollout, off: int, start_tile: dict) -> None:
        d = self.dir
        d.mkdir(exist_ok=True)
        if shutil.disk_usage(d).free < 1 << 30:
            if not self._disk_warned:
                print("!! less than 1 GB free: episode records are not being saved")
                self._disk_warned = True
            return
        if self._versions is None:
            self._versions = _versions()
        commit, mod = self._versions
        stamp = time.strftime("%Y%m%d-%H%M%S")
        played: dict[int, int] = {}
        for m in members:
            i, r, s0 = m.i, m.rec, m.s
            n = len(r["wire"])
            if n == 0:
                continue
            j = played[i] = played.get(i, -1) + 1        # members played on this game in this pass
            common = dict(spec=spec, stage=stage, rung=rung, iteration=iteration, instance=i,
                          code_commit=commit, mod_md5=mod, wire=np.stack(r["wire"]).astype(np.int8),
                          acted=np.array(r["acted"]), start_tile=np.array(start_tile.get(i, [-1, -1])))
            if m.cut:
                # Kept apart, so readers of episodes/*.npz skip them.
                (d / "cut").mkdir(exist_ok=True)
                np.savez_compressed(d / "cut" / f"{tag_prefix}_env{i}_m{j}_{stamp}.npz", cut=True, **common)
                continue
            slept = any("hibernate" in e for e in r["events"])
            np.savez_compressed(
                d / f"{tag_prefix}_env{i}{'' if j == 0 else f'_m{j}'}_{stamp}.npz",
                events=np.array(r["events"]), pos=np.stack(r["pos"]), room=np.array(r["room"]),
                food=np.array(r["food"]), reward=ro.rew[i, off + s0:off + s0 + n].copy(),
                deficits=ro.deficits[i, off + s0:off + s0 + n].copy(),
                rows=np.stack(r["rows"]) if r["rows"] else np.zeros(0),
                ended=np.array(r.get("ended", "hibernate" if slept else "")), **common)
