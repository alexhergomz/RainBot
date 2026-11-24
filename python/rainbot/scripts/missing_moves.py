"""Which player moves the game's AI map lacks, and the jump envelopes the mod adds for them.

Every airborne stretch in the human sessions, from the state the slugcat left (floor, pole, wall,
ledge, corridor, water) to the state it caught (same room, alive, no pipe), is checked against the
room's dumped AI map (RainBotData/rooms/<region>_<room>.json, the Slugcat template's allowed moves):
does the game have any route from the start tile to the end tile, and how long is it? A move is
"missing" when the map has no route, or one more than 3x longer than the straight line.
Writes RainBotData/jumps-SU.txt.

    python -m rainbot.scripts.missing_moves
"""
from __future__ import annotations

import base64
import collections
import glob
import heapq
import json
from pathlib import Path

import numpy as np

from ..data import session as Sx
from ..train.bc import RB, human_sessions, load_world_names

SUPPORT = {"Stand": "floor", "Crawl": "floor", "ClimbingOnBeam": "pole", "WallClimb": "wall",
           "CorridorClimb": "corridor", "Swimming": "water"}


def room_graph(region: str, name: str):
    p = RB / "rooms" / f"{region}_{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    e = np.frombuffer(base64.b64decode(d["moves_b64"]), np.uint8).reshape(-1, 5).astype(int)
    adj = collections.defaultdict(list)
    for ax, ay, bx, by, _ in e:
        adj[(ax, ay)].append(((bx, by), max(1, max(abs(bx - ax), abs(by - ay)))))
    return adj


def route(adj, a, b, limit=400):
    """Shortest AI-map route length from tile a to (around) tile b; None if none within `limit`."""
    goal = {(b[0] + dx, b[1] + dy) for dx in (-1, 0, 1) for dy in (-2, -1, 0, 1)}
    starts = [(a[0] + dx, a[1] + dy) for dx in (-1, 0, 1) for dy in (-2, -1, 0)]
    dist, pq = {}, []
    for s in starts:
        if s in adj:
            dist[s] = 0; pq.append((0, s))
    while pq:
        d, u = heapq.heappop(pq)
        if u in goal:
            return d
        if d > dist.get(u, 1e9) or d > limit:
            continue
        for v, w in adj[u]:
            if d + w < dist.get(v, 1e9):
                dist[v] = d + w; heapq.heappush(pq, (d + w, v))
    return None


def main():
    world = load_world_names()
    first, rooms = world["first_room_index"], world["rooms"]
    graphs = {}
    moves = []
    for p in human_sessions():
        s = Sx.load(p)
        o = s.obs
        bm, an = s.schema.enums["body_mode"], s.schema.enums["animation"]
        body, anim = np.asarray(o["body_mode"]), np.asarray(o["animation"])
        room, tick, alive = np.asarray(o["room"]), np.asarray(o["tick"]), np.asarray(o["alive"])
        pos = np.asarray(o["pos0"]) / 20.0
        def state(k):
            a = an[anim[k]] if 0 <= anim[k] < len(an) else ""
            if a == "LedgeGrab":
                return "ledge"
            b = bm[body[k]] if 0 <= body[k] < len(bm) else ""
            return SUPPORT.get(b)
        k, n = 0, len(o)
        last_sup = None
        while k < n:
            st = state(k)
            if st is not None:
                if last_sup is not None:
                    j, st0 = last_sup
                    air = k - j - 1
                    ok = (room[k] == room[j] and alive[j:k + 1].all() and (tick[k] - tick[j]) == (k - j)
                          and air >= 2 and np.abs(np.diff(pos[j:k + 1], axis=0)).max() < 3)
                    if ok:
                        moves.append((int(room[j]), st0, st, pos[j], pos[k], air))
                last_sup = (k, st)
            k += 1
    stats = collections.defaultdict(list)
    for r, s0, s1, a, b, air in moves:
        name = rooms[r - first] if 0 <= r - first < len(rooms) else None
        if name is None:
            continue
        if name not in graphs:
            graphs[name] = room_graph("SU", name)
        g = graphs[name]
        if g is None:
            continue
        ta, tb = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
        straight = max(abs(tb[0] - ta[0]), abs(tb[1] - ta[1]))
        if straight < 2:
            continue
        L = route(g, ta, tb)
        missing = L is None or L > 3 * straight + 3
        stats[(s0, s1)].append((missing, tb[0] - ta[0], tb[1] - ta[1], air, L))
    # The jump envelope per (from, to) kind, over every human jump of that kind (the game's map has
    # some, not all): horizontal reach and height gain at the 90th percentile, the fall any depth the
    # humans fell (10th percentile), time in the air at the median. The mod adds a link from every
    # tile of the first kind to every tile of the second within it, with a clear line between.
    env = {}
    for (s0, s1), v in stats.items():
        if len(v) < 5 or "water" in (s0, s1) or "corridor" in (s0, s1):
            continue
        dx = np.abs([x[1] for x in v])
        dy = np.array([x[2] for x in v])
        air = np.array([x[3] for x in v])
        to = "floor" if s1 == "ledge" else s1
        env[(s0, to)] = (int(np.ceil(np.percentile(dx, 90))), int(np.floor(np.percentile(dy, 10))),
                         int(np.ceil(np.percentile(dy, 90))), int(np.median(air)), len(v))
    # Wall to wall (climbing a channel by jumping between its walls): the human sessions have too few
    # examples, so the reach comes from the agent's own episodes (rows once per token: hops from
    # WallClimb across to another WallClimb with time in the air between).
    if ("wall", "wall") not in env:
        wall, dflt = None, None
        hops = []
        for f in glob.glob(str(Path(__file__).resolve().parents[3] / "runs" / "*" / "episodes" / "*.npz")):
            try:
                z = np.load(f)
                if "rows" not in z.files or len(z["rows"]) < 3:
                    continue
            except Exception:
                continue
            if wall is None:
                bmn = Sx.load(human_sessions()[0]).schema.enums["body_mode"]
                wall, dflt = bmn.index("WallClimb"), bmn.index("Default")
            r = z["rows"]
            b, ps, rm = np.asarray(r["body_mode"]), np.asarray(r["pos0"]) / 20.0, np.asarray(r["room"])
            k = 0
            while k < len(b) - 1:
                if b[k] == wall:
                    j = k + 1
                    while j < len(b) and j - k <= 4 and b[j] == dflt:
                        j += 1
                    if (j < len(b) and j > k + 1 and b[j] == wall and rm[j] == rm[k]
                            and abs(ps[j, 0] - ps[k, 0]) >= 1):
                        hops.append((ps[j, 0] - ps[k, 0], ps[j, 1] - ps[k, 1], (j - k) * 8))
                    k = j
                else:
                    k += 1
        if len(hops) >= 5:
            h = np.array(hops)
            env[("wall", "wall")] = (int(np.ceil(np.percentile(np.abs(h[:, 0]), 90))),
                                     int(np.floor(np.percentile(h[:, 1], 10))),
                                     int(np.ceil(np.percentile(h[:, 1], 90))), int(np.median(h[:, 2])), len(h))
    out = RB / "jumps-SU.txt"
    with open(out, "w") as f:
        f.write("# from to dx_max dy_min dy_max air_ticks n   (human jumps, rainbot.scripts.missing_moves)\n")
        for (a_, b_), (dxm, dyl, dyh, airm, n_) in sorted(env.items()):
            f.write(f"{a_} {b_} {dxm} {dyl} {dyh} {airm} {n_}\n")
    print("jump envelopes ->", out)
    for k_, v_ in sorted(env.items()):
        print("  ", k_, "dx<=%d dy %d..%d air %d ticks (n=%d)" % v_)
    print(f"{len(moves)} airborne moves in the human sessions; with a room map and >= 2 tiles: "
          f"{sum(len(v) for v in stats.values())}")
    print(f"{'from -> to':22s} {'n':>5s} {'missing':>8s}  "
          "missing moves: |dx| p50/p90/max, dy p10/p50/p90/max, air ticks p50/p90")
    for key, v in sorted(stats.items(), key=lambda kv: -len(kv[1])):
        m = [x for x in v if x[0]]
        line = f"{key[0]+' -> '+key[1]:22s} {len(v):5d} {len(m):8d}"
        if m:
            dx = np.abs([x[1] for x in m])
            dy = np.array([x[2] for x in m])
            air = np.array([x[3] for x in m])
            line += (f"  |dx| {np.percentile(dx,50):.0f}/{np.percentile(dx,90):.0f}/{dx.max()}"
                     f"  dy {np.percentile(dy,10):.0f}/{np.percentile(dy,50):.0f}/{np.percentile(dy,90):.0f}/{dy.max()}"
                     f"  air {np.percentile(air,50):.0f}/{np.percentile(air,90):.0f}")
        print(line)


if __name__ == "__main__":
    main()
