"""Live check of the mod's hot reload (mod/Loader).

Step, reload by command, step; disconnect, install a new build, reconnect, step. After each swap the
ticks inside a chunk must still be consecutive (a hook left behind by the old core would run the
lockstep twice per tick).

    python -m rainbot.scripts.reload_check          # instance 0 running
"""
import subprocess
import time
from pathlib import Path

import numpy as np

from ..env.rain_env import RainEnv

NEUTRAL = np.zeros((4, 5), np.int8)
REPO = Path(__file__).resolve().parents[3]


def steps(e, n, fresh=True):
    """`fresh`: just connected (the mod sends a chunk first); otherwise the game awaits actions."""
    if not fresh:
        e.send_actions(NEUTRAL)
    st = e.recv_step()
    ok = True
    for _ in range(n):
        e.send_actions(NEUTRAL)
        st = e.recv_step()
        t = np.asarray(st.rows["tick"][:st.n_real], np.int64)
        ok &= bool((np.diff(t) == 1).all())
    return st, ok


def main():
    e = RainEnv(instance=0, timeout=60)
    e.connect(retries=60, wait_s=5)
    st, ok = steps(e, 50)
    print("before:        consecutive ticks", ok, "tick", int(st.rows[0]["tick"]), flush=True)
    e.reload_mod()
    st, ok = steps(e, 50)
    print("after reload:  consecutive ticks", ok, "tick", int(st.rows[0]["tick"]), flush=True)
    e.close()
    subprocess.run(["touch", str(REPO / "mod/RainBot/Plugin.cs")], check=True)
    subprocess.run([str(REPO / "mod/build.sh")], check=True, capture_output=True)
    time.sleep(3)
    e.connect(retries=30, wait_s=1)
    st, ok = steps(e, 50)
    print("after rebuild: consecutive ticks", ok, "tick", int(st.rows[0]["tick"]), flush=True)
    e.command("pathdebug")
    st, _ = steps(e, 3, fresh=False)
    e.close()


if __name__ == "__main__":
    main()
