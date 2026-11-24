"""Reachability probe: warm up the mod's region survey, then reset into distinct start specs from
the episode records and read food_path / shelter_path on the first steady tokens.

    python -m rainbot.scripts.reach_probe [WARMUP_TOKENS] [ROOM ...]
"""
import glob
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))            # also runnable as a plain file
from rainbot.env.rain_env import RainEnv             # noqa: E402
from rainbot.train.bc import load_world_names        # noqa: E402

NEUTRAL = np.zeros((4, 5), np.int8)


def main():
    w = load_world_names()
    first, names = w["first_room_index"], w["rooms"]

    def room_name(r):
        return names[r - first] if 0 <= r - first < len(names) else str(r)

    specs, seen = [], set()
    for f in sorted(glob.glob(str(REPO / "runs" / "*" / "episodes" / "*.npz"))):
        d = np.load(f)
        s = str(d["spec"])
        room = [t for t in s.split() if t.startswith("room=")]
        key = room[0] if room else s
        if key not in seen:
            seen.add(key)
            specs.append(s)
    only = sys.argv[2:]
    if only:
        specs = [s for s in specs if any(("room=" + o + " ") in s for o in only)]
    print(len(specs), "distinct start rooms", flush=True)

    e = RainEnv(instance=0, timeout=300)
    e.connect(retries=80, wait_s=5)
    st = e.recv_step()
    e.command("speed 16")
    for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 1500):
        if st.needs_action:
            e.send_actions(NEUTRAL)
        st = e.recv_step()
    e.command("pathdebug")
    for spec in specs:
        for _ in range(6000):
            last = st.rows[st.n_real - 1]
            if int(last["alive"]) and not int(last["rain_hit"]) and st.needs_action:
                break
            if st.needs_action:
                e.send_actions(NEUTRAL)
            st = e.recv_step()
        e.request_reset(spec)
        ep0 = int(st.rows[0]["episode"])
        for _ in range(4000):
            if st.needs_action:
                e.send_actions(NEUTRAL)
            st = e.recv_step()
            if int(st.rows[0]["episode"]) != ep0:
                break
        vals = []
        for _ in range(6):
            if st.needs_action:
                e.send_actions(NEUTRAL)
            st = e.recv_step()
            last = st.rows[st.n_real - 1]
            vals.append((int(last["food_path"]), int(last["shelter_path"])))
        last = st.rows[st.n_real - 1]
        print(f"{room_name(int(last['room'])):>16} food {vals[-1][0]:>5} shelter {vals[-1][1]:>5}   [{spec}]",
              flush=True)
    e.command("pathdebug")
    for _ in range(3):
        if st.needs_action:
            e.send_actions(NEUTRAL)
        st = e.recv_step()
    e.close()


if __name__ == "__main__":
    main()
