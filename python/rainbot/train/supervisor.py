"""Owns the game's lifecycle so an unattended run survives the night.

Rain World will crash, hang, or wedge on a screen the autopilot does not know about, and none
of that should end a training run. The supervisor launches the game, waits for the bridge,
runs the PPO trainer against it, and relaunches when things go wrong. The trainer resumes from
its run's checkpoint, so a crash costs at most the PPO iteration in flight.

Liveness is judged by *training progress* (metrics.jsonl grows once per PPO iteration), not by
the process existing. A Rain World that is running but stuck on a menu looks healthy to
`poll()` while producing nothing, and that is the failure an overnight run must detect.
"""
from __future__ import annotations

import argparse
import functools
import os
import signal
import subprocess
import time
from pathlib import Path

# Unbuffered: an overnight log that only appears when the process exits is useless.
print = functools.partial(__builtins__.print if not isinstance(__builtins__, dict)
                          else __builtins__["print"], flush=True)

REPO = Path(__file__).resolve().parents[3]
LAUNCH = REPO / "tools" / "launch.sh"


class GameProcess:
    """A launched Rain World, and the means to end it."""

    def __init__(self, launch_script: Path = LAUNCH, log: Path | None = None, args: list | None = None):
        self.launch_script = launch_script
        self.log = log
        self.args = list(args or [])
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        out = open(self.log, "ab") if self.log else subprocess.DEVNULL
        # New process group: Proton spawns a tree (proton -> wine -> RainWorld.exe) and killing
        # only the launcher leaves the game running and holding the port.
        self.proc = subprocess.Popen(
            ["/usr/bin/env", "bash", str(self.launch_script), *self.args],
            stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
        )
        print(f"[supervisor] launched game (pid {self.proc.pid})")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        # Proton's tree does not always die with the group; sweep the exe by name as a backstop.
        subprocess.run(["pkill", "-f", r"RainWorld\.exe"], capture_output=True)
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.proc = None
        print("[supervisor] game stopped")

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def _kill_stray_games(ports=(54321,)) -> None:
    """Kill any Rain World left over from an earlier run, and wait for the port to free.

    wineserver inherits the listening socket when the game dies, so a relaunch can find the
    port still bound by a dead game. Waiting for it to close avoids launching into a conflict
    that looks like a boot failure.
    """
    subprocess.run(["pkill", "-f", r"RainWorld\.exe"], capture_output=True)
    for _ in range(30):
        if not any(_port_bound("127.0.0.1", p) for p in ports):
            return
        time.sleep(1)


def _port_bound(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def port_open(host: str, port: int, timeout: float = 5.0) -> bool:
    """True only if a real mod answers the handshake.

    Not a bare connect(): when the game dies, wineserver inherits its listening socket and keeps
    accepting connections, so a plain TCP connect succeeds against a dead game and the collector
    would block forever. The handshake is the cheapest thing only a live mod can produce.
    """
    import socket
    from ..proto import recv_handshake
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            recv_handshake(s)
            return True
    except Exception:
        return False


def wait_for_bridge(host: str, port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open(host, port):
            return True
        time.sleep(2)
    return False


def run(args, trainer_args: list[str]) -> int:
    """Launch the game(s), run the PPO trainer against them, relaunch both when either fails.

    The trainer resumes from the run's own checkpoint, so a restart costs at most the iteration
    in flight. Liveness is judged by training progress (``metrics.jsonl`` grows once per PPO
    iteration), not by processes existing: a game wedged on a screen keeps its process and port
    alive while the trainer waits on it forever.
    """
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))

    n = args.instances
    script = LAUNCH if n == 1 else REPO / "tools" / "launch_instances.sh"
    game = GameProcess(launch_script=script, log=Path(args.game_log) if args.game_log else None,
                       args=[] if n == 1 else [str(n)])
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_dir / "metrics.jsonl"
    restarts = 0
    t0 = time.time()
    ports = [args.port + i for i in range(n)]

    while not stop["now"]:
        if args.attach and all(port_open(args.host, p) for p in ports):
            print("[supervisor] attaching to games that are already running")
        else:
            # Always stop whatever is running before launching: a game still booting answers no
            # handshake either, and launching a second copy leaves two contending for the port.
            game.stop()
            _kill_stray_games(ports)
            game.start()
            if not all(wait_for_bridge(args.host, p, args.boot_timeout) for p in ports):
                print(f"[supervisor] not every bridge came up within {args.boot_timeout}s; relaunching")
                restarts += 1
                continue
            print("[supervisor] all bridges up")

        cmd = [args.python, "-u", "-m", "rainbot.train.ppo", "--instances", str(n),
               "--base-port", str(args.port), "--run-dir", str(run_dir), "--seed", str(restarts)]
        if (run_dir / "ckpt.pt").exists():
            cmd += ["--resume", str(run_dir / "ckpt.pt")]
        cmd += trainer_args
        env = dict(os.environ, PYTHONPATH=str(REPO / "python"))
        print(f"[supervisor] starting trainer (restart #{restarts}): {' '.join(cmd[3:])}")
        trainer = subprocess.Popen(cmd, cwd=str(REPO / "python"), env=env)

        last_size = metrics.stat().st_size if metrics.exists() else 0
        last_change = time.time()
        while not stop["now"]:
            if trainer.poll() is not None:
                print(f"[supervisor] trainer exited ({trainer.returncode})")
                break
            if not args.attach and not game.alive:
                print("[supervisor] game process died")
                break
            size = metrics.stat().st_size if metrics.exists() else 0
            if size != last_size:
                last_size, last_change = size, time.time()
            elif time.time() - last_change > args.stall_timeout:
                print(f"[supervisor] no PPO iteration for {args.stall_timeout}s; treating as stalled")
                break
            time.sleep(10)

        trainer.terminate()
        try:
            trainer.wait(timeout=30)
        except subprocess.TimeoutExpired:
            trainer.kill()
        if stop["now"] or (trainer.returncode == 0 and trainer.poll() is not None):
            break
        if trainer.returncode == 2:
            print("[supervisor] trainer rejected its arguments (exit 2); not retrying")
            break
        restarts += 1
        args.attach = False
        print(f"[supervisor] cycling ({restarts} restarts, {(time.time()-t0)/3600:.1f}h elapsed)")

    game.stop()
    print(f"[supervisor] done: {restarts} restarts over {(time.time()-t0)/3600:.2f}h", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Own the game lifecycle around a PPO run. Unknown "
                                             "arguments are passed through to rainbot.train.ppo.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=54321, help="base port; instance i uses port+i")
    ap.add_argument("--instances", type=int, default=1)
    ap.add_argument("--run-dir", default=str(REPO / "runs" / time.strftime("%Y%m%d-%H%M%S")))
    ap.add_argument("--python", default=str(REPO / "python" / ".venv" / "bin" / "python"))
    ap.add_argument("--game-log", default="")
    # Proton boot plus the autopilot walking intro/menu/slugcat-select takes a couple of minutes cold.
    ap.add_argument("--boot-timeout", type=float, default=360.0)
    ap.add_argument("--stall-timeout", type=float, default=1200.0)
    ap.add_argument("--attach", action="store_true",
                    help="use games that are already running instead of launching them")
    args, rest = ap.parse_known_args(argv)
    rest = [x for x in rest if x != "--"]
    # Relative run dirs mean the repo root, not python/ (where the trainer runs).
    if not Path(args.run_dir).is_absolute():
        args.run_dir = str((Path.cwd() if Path.cwd() != REPO / "python" else REPO) / args.run_dir)
    return run(args, rest)


if __name__ == "__main__":
    raise SystemExit(main())
