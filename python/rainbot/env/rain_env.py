"""One game instance as an RL environment over the bridge protocol (docs/PROTOCOL.md).

``RainEnv`` owns one TCP connection. Its unit of time is the agent step: 8 ticks in, 4 decisions
out. The mod is blocked inside the game's update loop between ``recv`` and ``send``, so the
simulation cannot advance while Python thinks. A ``ping`` thread keeps the mod's receive timeout
from expiring while the learner holds the game frozen for an update.

Episodes: an episode starts at a ``reset`` (randomised initial state) and
continues through deaths and hibernations (the game's own respawn at the last shelter) until the
trainer ends it. Each cycle end arrives from the mod as a ``session_end`` event (reason ``death``
or ``sleep``). Karma changes show up in the rows of the next session, so the reward sees them
exactly where the game applies them.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .. import obs as O
from .. import proto
from ..proto import ProtocolError, Schema

DEFAULT_HOST = "127.0.0.1"
BASE_PORT = 54321
NEUTRAL = np.zeros((O.DECISIONS, 5), dtype=np.int8)


@dataclass
class Step:
    """What one agent step returns."""

    rows: np.ndarray              # [8] structured rows (padded by repeating the last real row)
    n_real: int                   # real rows in the chunk (< 8 when the session ended mid-chunk)
    events: list[dict] = field(default_factory=list)
    frame: np.ndarray | None = None  # [H, W, C] uint8, the latest rendered frame (row 0 = top)
    frame_tick: int = -1             # sim tick it was rendered after; -1 = none captured yet
    episode_start: bool = False   # first token of a new episode (reset the recurrent state)
    cycle_ended: int = 0          # number of session_end events since the previous step
    deaths: int = 0
    sleeps: int = 0
    #: Whether this step is an exchange the mod is blocked on. The final chunk of a session (it
    #: follows a ``session_end`` event, and may be partial) is not: the agent must not answer it,
    #: and its sampled action is recorded as not executed so the policy loss skips it.
    needs_action: bool = True
    wait_ms: float = 0.0


class RainEnv:
    def __init__(self, instance: int = 0, host: str = DEFAULT_HOST, port: int | None = None,
                 vocab: O.Vocab | None = None, timeout: float = 600.0, ping_s: float = 10.0):
        self.instance = instance
        self.host, self.port = host, (BASE_PORT + instance) if port is None else port
        self.timeout, self.ping_s = timeout, ping_s
        self.vocab = vocab if vocab is not None else O.Vocab()
        self.index = O.IndexMap(self.vocab)
        self.feat = O.Featurizer(self.index)
        self.sock: socket.socket | None = None
        self.schema: Schema | None = None
        self._send_lock = threading.Lock()
        self._ping_stop = threading.Event()
        self._ping_thread: threading.Thread | None = None
        self._t_recv_done: float | None = None
        self.think_ms: list[float] = []
        self._episode_id: int | None = None
        self.water_rooms: set[str] | None = None
        self.world: dict = {}
        self.events_log: list[dict] = []

    # ---- connection ---------------------------------------------------------------------
    def connect(self, retries: int = 1, wait_s: float = 2.0) -> Schema:
        last = None
        for _ in range(max(1, retries)):
            try:
                s = socket.create_connection((self.host, self.port), timeout=self.timeout)
                break
            except OSError as e:
                last = e
                time.sleep(wait_s)
        else:
            raise ConnectionError(f"instance {self.instance}: no game on port {self.port}: {last}")
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s
        self.schema = proto.recv_handshake(s)
        if self.schema.raw.get("ticks_per_step", O.TICKS) != O.TICKS or \
                self.schema.raw.get("decisions_per_step", O.DECISIONS) != O.DECISIONS:
            raise ProtocolError(f"mod chunking {self.schema.raw.get('ticks_per_step')}x"
                                f"{self.schema.raw.get('decisions_per_step')} != {O.TICKS}x{O.DECISIONS}")
        self.index.from_schema(self.schema.enums)
        return self.schema

    def reload_mod(self, retries: int = 30, wait_s: float = 1.0) -> Schema:
        """Swap in the newest mod build (mod/build.sh installs it) without relaunching the game:
        the loader unloads the running core, loads the new one and its config, and the link is
        re-established. Call between ``recv_step`` and ``send_actions``. Episode state is lost
        (the new core starts from free play); reset afterwards."""
        self.command("reload")
        self.close()
        time.sleep(wait_s)
        self._episode_id = None
        return self.connect(retries=retries, wait_s=wait_s)

    def close(self) -> None:
        self.keepalive(False)
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    # ---- framing ------------------------------------------------------------------------
    def _send(self, msg_type: int, payload: bytes) -> None:
        with self._send_lock:
            proto.send_frame(self.sock, msg_type, payload)

    def command(self, text: str) -> None:
        self._send(proto.MSG_CMD, text.encode("utf-8"))

    def keepalive(self, on: bool) -> None:
        """While on, a background thread pings every ``ping_s`` so a frozen game does not time out."""
        if on and self._ping_thread is None and self.sock is not None:
            self._ping_stop.clear()

            def run():
                while not self._ping_stop.wait(self.ping_s):
                    try:
                        self.command("ping")
                    except OSError:
                        return
            self._ping_thread = threading.Thread(target=run, daemon=True, name=f"ping{self.instance}")
            self._ping_thread.start()
        elif not on and self._ping_thread is not None:
            self._ping_stop.set()
            self._ping_thread.join(timeout=2)
            self._ping_thread = None

    def _on_event(self, ev: dict) -> None:
        self.events_log.append(ev)
        if len(self.events_log) > 10000:
            del self.events_log[:5000]
        self.index.on_event(ev)
        if ev.get("type") == "world":
            self.world = ev
        elif ev.get("type") == "water_rooms":
            self.water_rooms = set(ev.get("rooms") or [])     # from the mod's room survey
        elif ev.get("type") == "memreport":
            print(f"[mod{self.instance}] {ev.get('text')}", flush=True)

    def recv_step(self) -> Step:
        """Block for the next MSG_CHUNK; events that arrive first are collected into it."""
        assert self.sock is not None and self.schema is not None
        t0 = time.perf_counter()
        events: list[dict] = []
        while True:
            msg_type, payload = proto.recv_frame(self.sock)
            if msg_type == proto.MSG_EVENT:
                ev = _json(payload)
                self._on_event(ev)
                events.append(ev)
                continue
            if msg_type == proto.MSG_LOG:
                print(f"[mod{self.instance}] {payload.decode('utf-8', 'replace')}")
                continue
            if msg_type != proto.MSG_CHUNK:
                raise ProtocolError(f"unexpected message type {msg_type} (expected MSG_CHUNK)")
            n, rows, frame, frame_tick = proto.decode_chunk(payload, self.schema)
            if n == 0:
                continue
            break
        if n < O.TICKS:
            rows = np.concatenate([rows, np.repeat(rows[-1:], O.TICKS - n)])
        else:
            rows = rows.copy()
        t1 = time.perf_counter()
        ep = int(rows[0]["episode"]) if "episode" in rows.dtype.names else 0
        # `episode` is the mod's reset counter (it does not move on deaths or hibernation).
        start = self._episode_id is None or ep != self._episode_id
        self._episode_id = ep
        if frame is None:
            fw = self.schema.raw.get("frame") or {}
            h, w = fw.get("h", O.FRAME_H) or O.FRAME_H, fw.get("w", O.FRAME_W) or O.FRAME_W
            frame = np.zeros((h, w, O.FRAME_C), np.uint8)
        st = Step(rows=rows, n_real=n, events=events, episode_start=start, frame=frame.copy(),
                  frame_tick=int(frame_tick),
                  needs_action=not any(e.get("type") == "session_end" for e in events),
                  wait_ms=(t1 - t0) * 1000.0)
        for e in events:
            if e.get("type") == "session_end":
                st.cycle_ended += 1
                st.deaths += e.get("reason") == "death"
                st.sleeps += e.get("reason") == "sleep"
        self._t_recv_done = t1
        return st

    def send_actions(self, actions_wire: np.ndarray) -> None:
        """[4, 5] wire actions (x, y in {-1,0,1}; buttons {0,1})."""
        t = time.perf_counter()
        if self._t_recv_done is not None:
            # Think time per step, logged although the game is frozen meanwhile: a policy that
            # needs unbounded think time could not play in real time.
            self.think_ms.append((t - self._t_recv_done) * 1000.0)
            if len(self.think_ms) > 4096:
                del self.think_ms[:2048]
        self._send(proto.MSG_ACTION, proto.encode_actions(actions_wire))

    def request_reset(self, spec: str) -> None:
        """Queue a randomised reset (applied by the mod at the next tick boundary).

        Call between ``recv_step`` and ``send_actions``: the mod reads commands while blocked.
        """
        # `peaceful` is its own command in the protocol, not a reset key; the curriculum puts it
        # in the spec for convenience, so it is split out here and sent first.
        keep = []
        for tok in spec.split():
            if tok.startswith("peaceful="):
                self.command("peaceful " + tok.split("=", 1)[1])
            else:
                keep.append(tok)
        self.command("reset " + " ".join(keep) if keep else "reset")


def _json(payload: bytes) -> dict:
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return {"type": "raw", "raw": payload.decode("utf-8", "replace")}
