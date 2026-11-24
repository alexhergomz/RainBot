"""Watch the training games for crashes and stalls; print one line per new problem (for a Monitor).

    python3 -u tools/watch_games.py RUN_DIR N_INSTANCES

Checks every 30 s, so a game that crashes at launch is noticed at once:
  - a new BepInEx preloader crash log (preloader_*.log in the game folder),
  - fewer RainWorld.exe processes than instances while the supervisor runs,
  - a game's instance log silent for 5 minutes while the trainer collects,
  - the supervisor waiting for the games' bridges for more than 10 minutes.
"""
import glob
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(sys.argv[1])
N = int(sys.argv[2])
GAME = Path.home() / ".steam/steam/steamapps/common/Rain World"
DATA = GAME / "RainBotData"
SUP = RUN.parent / (RUN.name + ".supervisor.out")

seen_crash = set(glob.glob(str(GAME / "preloader_*.log")))
alerted = {}


def say(key: str, msg: str, every: float = 600.0) -> None:
    now = time.time()
    if now - alerted.get(key, 0.0) >= every:
        alerted[key] = now
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def supervisor_running() -> bool:
    return subprocess.run(["pgrep", "-f", "run_training.sh"], capture_output=True).returncode == 0


while True:
    for f in sorted(set(glob.glob(str(GAME / "preloader_*.log"))) - seen_crash):
        seen_crash.add(f)
        head = Path(f).read_text(errors="replace").splitlines()[:2]
        say("crash" + f, f"GAME CRASH at launch (BepInEx preloader): {Path(f).name}: {' '.join(head)[:200]}",
            0)
    if supervisor_running():
        n = int(subprocess.run(["pgrep", "-fc", r"^Z:.*RainWorld\.exe"], capture_output=True,
                               text=True).stdout or 0)
        if n < N:
            say("procs", f"only {n} of {N} games running")
        phase = (RUN / "phase").read_text().strip() if (RUN / "phase").exists() else ""
        if phase == "collect":
            for i in range(N):
                log = DATA / f"instance{i}.log"
                if log.exists() and time.time() - log.stat().st_mtime > 300:
                    idle_min = int((time.time() - log.stat().st_mtime) / 60)
                    say(f"stale{i}", f"game {i}: no log output for {idle_min} min while collecting")
        if SUP.exists():
            lines = SUP.read_text(errors="replace").splitlines()[-30:]
            launched = [k for k, l in enumerate(lines) if "launched game" in l]
            up = [k for k, l in enumerate(lines) if "all bridges up" in l]
            if launched and (not up or up[-1] < launched[-1]) and time.time() - SUP.stat().st_mtime > 600:
                say("bridges", "supervisor still waiting for the games' bridges after 10+ min")
    time.sleep(30)
