"""Observed player moves, for the mod's path lengths (PathInfo.cs).

The game's AI map has no connection for many moves a player makes (jumps up ledges, across gaps,
off poles): the Slugcat creature template only walks, climbs, reaches and drops. So on the AI map
alone some exits, fruit and shelters have no path at all. What the player did is evidence that a
move exists, and of how long it takes. This writes every recorded
trajectory, room by room, as runs of constant tile:

    speed <tiles per tick>
    <ROOM> t0,t1,x0,y0,x1,y1;t0,t1,x0,y0,x1,y1;...

(ticks the run started/ended, main and second body chunk tiles). The mod takes, in each room, the
standable tile of every run (its own StartTile rule, on the room's AI map) and adds an edge between
consecutive standable runs, length = ticks in between x the human travel speed, so the potential's
time conversion (length / speed) gives back the observed time. Sources: the recorded human sessions
(every tick) and the agent's episode records (one position per 8-tick token).

    python -m rainbot.scripts.observed_moves [--records ../runs/*/episodes] \
        [--out RainBotData/moves-SU.txt]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from .. import reward as R
from ..data import session as S
from ..train.bc import RB, human_sessions, load_world_names

TILE = 20.0


def _tiles(p):
    return np.floor(np.asarray(p, np.float64) / TILE).astype(int)


def _runs(tick, t0, t1, names):
    """Runs of constant (room, tiles) inside visits; yields (room name, [run, ...])."""
    key = np.concatenate([t0, t1], axis=1)
    i = 0
    n = len(tick)
    visits = []
    while i < n:
        j = i
        runs = []
        while j < n and names[j] == names[i] and (j == i or tick[j] - tick[j - 1] <= 8):
            k = j
            while (k + 1 < n and names[k + 1] == names[i] and tick[k + 1] - tick[k] <= 8
                   and (key[k + 1] == key[j]).all()):
                k += 1
            runs.append((int(tick[j]), int(tick[k]), *key[j].tolist()))
            j = k + 1
        if names[i] and len(runs) > 1:
            visits.append((names[i], runs))
        i = j
    return visits


def from_sessions(paths, world) -> list:
    first, rooms = world["first_room_index"], world["rooms"]
    out = []
    for p in paths:
        o = S.load(p).obs
        room = np.asarray(o["room"], np.int64) - first
        ok = ((room >= 0) & (room < len(rooms)) & (np.asarray(o["alive"]) > 0)
              & (np.asarray(o["grabbed_by"]) == 0))
        names = np.array([rooms[r] if g else "" for r, g in zip(room, ok)], dtype=object)
        out += _runs(np.asarray(o["tick"], np.int64), _tiles(o["pos0"]), _tiles(o["pos1"]), names)
    return out


def from_records(paths, world) -> list:
    first, rooms = world["first_room_index"], world["rooms"]
    out = []
    for p in paths:
        try:
            d = np.load(p, allow_pickle=False)
            pos, room = d["pos"], np.asarray(d["room"], np.int64) - first
        except (OSError, KeyError, ValueError):
            continue
        names = np.array([rooms[r] if 0 <= r < len(rooms) else "" for r in room], dtype=object)
        tick = np.arange(len(pos)) * R.TICKS_PER_TOKEN
        t = _tiles(pos)
        out += _runs(tick, t, t, names)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    runs = Path(__file__).resolve().parents[3] / "runs"
    ap.add_argument("--records", nargs="*", default=sorted(glob.glob(str(runs / "*" / "episodes"))))
    ap.add_argument("--region", default="SU")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    world = load_world_names(region=a.region)
    visits = from_sessions(human_sessions(), world)
    nh = len(visits)
    recs = [f for d in a.records for f in sorted(glob.glob(str(Path(d) / "*.npz")))]
    visits += from_records(recs, world)
    out = Path(a.out) if a.out else RB / f"moves-{a.region}.txt"
    with open(out, "w") as f:
        f.write(f"speed {R.SLUGCAT_TILES_PER_TICK}\n")
        for name, runs in visits:
            f.write(name + " " + ";".join(",".join(map(str, r)) for r in runs) + "\n")
    rooms = sorted({v[0] for v in visits})
    print(f"{nh} human visits, {len(visits) - nh} agent visits ({len(recs)} records), "
          f"{len(rooms)} rooms -> {out}")


if __name__ == "__main__":
    main()
