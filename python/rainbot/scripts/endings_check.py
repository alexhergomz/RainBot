"""Live check of frozen endings (`endings freeze`, mod Lockstep.EndNow).

Plays a death (the rain arrives), a sleep (fed, still, in a shelter) and a starvation (no food in a
shelter at the rain). For each: the session_end reason, the booked karma on the next token, a frozen
world (position constant), and how long the next reset takes to land.

    python -m rainbot.scripts.endings_check          # instance 0 running
"""
import time

import numpy as np

from ..env.rain_env import RainEnv

N = np.zeros((4, 5), np.int8)
CASES = [
    ("death (rain)", "room=SU_A22 pos=random food=2 karma=3 timer_left=40 creatures=clear seed=5", "death", 2),
    ("sleep (fed, still)", "room=SU_S01 pos=21,16 food=4 karma=3 timer_left=9000 creatures=clear seed=5", "sleep", 4),
    ("starve (no food)", "room=SU_S01 pos=21,16 food=0 karma=3 timer_left=40 creatures=clear seed=5", "starve", 2),
]
NEXT = "room=SU_A22 pos=random food=2 karma=3 timer_left=9000 creatures=clear seed=7"


def main():
    e = RainEnv(instance=0, timeout=300)
    e.connect(retries=10, wait_s=2)
    st = e.recv_step()
    e.command("endings freeze")
    for name, spec, want, want_karma in CASES:
        e.request_reset(spec)
        ep0 = int(st.rows[0]["episode"])
        while int(st.rows[0]["episode"]) == ep0:
            e.send_actions(N); st = e.recv_step()
        reason, booked, frozen_pos, k = None, None, [], 0
        for k in range(3000):
            if k > 0:                                    # the reset's own chunk may already hold the ending
                e.send_actions(N); st = e.recv_step()
            ends = [ev for ev in st.events if ev.get("type") == "session_end"]
            if ends:
                reason = ends[-1].get("reason")
                if st.needs_action:
                    e.send_actions(N)
                st = e.recv_step()                       # the booked token
                booked = int(st.rows[-1]["karma"]), int(st.rows[-1]["alive"])
                for _ in range(5):
                    e.send_actions(N); st = e.recv_step()
                    frozen_pos.append(np.asarray(st.rows[-1]["pos0"], np.float32).copy())
                break
        moved = float(np.ptp(np.stack(frozen_pos), axis=0).max()) if frozen_pos else float("nan")
        t0 = time.time()
        e.request_reset(NEXT)
        ep0 = int(st.rows[0]["episode"])
        n = 0
        while int(st.rows[0]["episode"]) == ep0:
            e.send_actions(N); st = e.recv_step(); n += 1
        alive = int(st.rows[-1]["alive"])
        print(f"{name:>17}: ended '{reason}' after {k} tokens (want '{want}'), booked karma/alive {booked} "
              f"(want karma {want_karma}), frozen drift {moved:.2f}px, next reset in {time.time() - t0:.2f}s "
              f"/ {n} tokens, alive {alive}", flush=True)
    e.command("endings game")
    e.close()


if __name__ == "__main__":
    main()
