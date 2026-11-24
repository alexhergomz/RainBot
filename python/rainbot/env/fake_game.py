"""A small stand-in for Rain World that speaks the bridge protocol exactly.

It tests the whole Python side (env client, featurizer, reward, PPO, curriculum, probes) without
the game, without Proton, and without the GPU. It is not a physics model of Rain World. It is a
world with the same needs structure: food to find, a shelter to reach before the rain, water that
drains the lungs, a predator that kills, karma that rises on hibernation and falls on death. If the
trainer and the drive-reduction reward cannot learn to eat and go home here, they will not in the
real game.

World: rooms in a line, ``SU_F00 .. SU_F0{n-2}`` then a shelter ``SU_FS0``. Each room is 40x24
tiles (20 units per tile) with a floor at row 1, exits at both ends, fruit on the floor, a pool in
the middle of every odd room, and optionally one lizard. Movement: x action accelerates toward
4 u/tick, jump on the ground gives vy = 9, gravity 0.8.

Protocol behaviour mirrors the mod: exchange every 8 ticks (rows of the previous 8 post-tick
states out, 4 decisions in), commands received while blocked are applied after the exchange, a
session ending flushes a partial chunk, ``episode`` counts resets only and ``session`` counts game
sessions.
"""
from __future__ import annotations

import json
import socket
import struct
import threading

import numpy as np

from .. import obs as O
from .. import proto

ROOM_W, ROOM_H, TILE = 40, 24, 20.0
FLOOR = 1
ANIMS = ["None", "StandUp", "Run", "Jump", "Swim", "Dead"]
MODES = ["Default", "Stand", "Crawl", "Swimming", "Dead"]
CREATURES = ["Slugcat", "GreenLizard", "Fly"]
OBJECTS = ["Creature", "DangleFruit", "Rock", "Spear"]


class Layout:
    """Sequential field layout with explicit offsets, like the mod's ObsLayout."""

    def __init__(self):
        self.fields, self.size = [], 0

    def add(self, name, dt, n=1):
        self.fields.append({"name": name, "dtype": dt, "offset": self.size, **({"shape": [n]} if n > 1 else {})})
        self.size += np.dtype(dt).itemsize * n


def build_layout() -> Layout:
    L = Layout()
    for name, dt, n in [
        ("tick", "i4", 1), ("episode", "i4", 1), ("session", "i4", 1), ("step_tick", "u1", 1),
        ("clock", "i4", 1), ("wall_ms", "i8", 1), ("room", "i4", 1), ("room_changed", "u1", 1),
        ("camera", "i4", 1), ("in_shortcut", "u1", 1), ("consumed", "u1", 1), ("action", "i1", 5),
        ("alive", "u1", 1), ("sleeping", "u1", 1), ("standing", "u1", 1), ("exhausted", "u1", 1),
        ("grabbed_by", "u1", 1), ("can_jump", "u1", 1), ("stun", "u1", 1), ("roll_counter", "u1", 1),
        ("slide_counter", "u1", 1), ("animation", "i2", 1), ("animation_frame", "i2", 1),
        ("body_mode", "i2", 1), ("grasp0_type", "i2", 1), ("grasp1_type", "i2", 1),
        ("aerobic_level", "f4", 1), ("air_in_lungs", "f4", 1), ("pos0", "f4", 2), ("vel0", "f4", 2),
        ("pos1", "f4", 2), ("vel1", "f4", 2), ("cycle_timer", "i4", 1), ("cycle_length", "i4", 1),
        ("rain_hit", "u1", 1), ("in_shelter", "u1", 1), ("food_to_hibernate", "i2", 1),
        ("food_in_stomach", "i2", 1), ("quarter_food", "i2", 1), ("max_food", "i2", 1),
        ("karma", "i2", 1), ("karma_cap", "i2", 1), ("karma_protected", "u1", 1),
        ("input_history", "i1", 50),
        ("cr_type", "i2", 6), ("cr_ai_kind", "i2", 6), ("cr_behavior", "i2", 6), ("cr_flags", "u1", 6),
        ("cr_rel_pos", "f4", 12), ("cr_rel_vel", "f4", 12),
        ("food_type", "i2", 4), ("food_points", "i1", 4), ("food_flags", "u1", 4), ("food_rel_pos", "f4", 8),
        ("item_type", "i2", 4), ("item_flags", "u1", 4), ("item_rel_pos", "f4", 8),
        ("holding_edible", "u1", 1), ("eat_counter", "u1", 1), ("speed", "f4", 1),
        ("hit_dealt", "u1", 1), ("hit_taken", "u1", 1), ("hit_victim", "i2", 1),
        ("food_path", "i2", 1), ("food_path_points", "i1", 1), ("food_path_bites", "u1", 1),
        ("shelter_path", "i2", 1), ("held_points", "i1", 1), ("held_bites", "u1", 1), ("eat_speed", "u1", 1),
        ("malnourished", "u1", 1), ("food_hibernate_base", "i2", 1),
        ("sleep_path", "i2", 1), ("start_shelter", "u1", 1), ("ready_for_win", "u1", 1),
        ("still_ticks_left", "i2", 1), ("force_ticks_left", "i2", 1), ("sleep_speed", "u1", 1),
    ]:
        L.add(name, dt, n)
    return L


def schema_json(L: Layout) -> dict:
    return {
        "protocol": 10, "game_version": "fake", "obs_bytes": L.size, "ticks_per_step": O.TICKS,
        "ticks_per_decision": O.TICKS_PER_DECISION, "decisions_per_step": O.DECISIONS,
        "frame": {"w": O.FRAME_W, "h": O.FRAME_H, "c": 3, "header_bytes": 9},
        "action": {"dims": list(proto.ACTION_DIMS), "bytes": proto.ACTION_BYTES},
        "enums": {"animation": ANIMS, "body_mode": MODES, "creature_type": CREATURES,
                  "object_type": OBJECTS, "ai_kind": ["LizardAI"],
                  "ai_behavior": {"LizardAI": ["Idle", "Hunt"]}},
        "fields": L.fields,
    }


class FakeWorld:
    def __init__(self, n_rooms: int = 3, cycle_length: int = 1600, food_to_hibernate: int = 2,
                 max_food: int = 5, karma_cap: int = 4, lizard: bool = False, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.rooms = [f"SU_F{i:02d}" for i in range(n_rooms - 1)] + ["SU_FS0"]
        self.shelter = n_rooms - 1
        self.cycle_length, self.hib, self.max_food, self.cap = cycle_length, food_to_hibernate, max_food, karma_cap
        self.lizard_enabled = lizard
        self.karma = 1
        self.episode = 0
        self.session = 0
        self.tick = 0
        self._spawn_cycle(room=self.shelter, x=8 * TILE)
        self.food = 0

    # ---- state -------------------------------------------------------------------------
    def _spawn_cycle(self, room: int, x: float) -> None:
        self.room, self.x, self.y, self.vx, self.vy = room, x, (FLOOR + 1) * TILE, 0.0, 0.0
        self.timer, self.air, self.alive, self.dead_for, self.still = 0, 1.0, True, 0, 0
        self.fruit = {r: list(self.rng.uniform(4, ROOM_W - 4, size=2) * TILE)
                      for r in range(len(self.rooms)) if r != self.shelter}
        self.lizard = None
        if self.lizard_enabled:
            self.lizard = [0, 30 * TILE, (FLOOR + 1) * TILE]
        self.room_changed = True
        self.session += 1
        self.consumed_last = np.zeros(5, np.int8)
        self.history = np.zeros((10, 5), np.int8)

    def apply_reset(self, kv: dict) -> dict:
        if "seed" in kv:
            self.rng = np.random.default_rng(int(kv["seed"]))
        r = kv.get("room")
        if r == "shelter":
            room = self.shelter
        elif r in self.rooms:
            room = self.rooms.index(r)
        elif r == "any":
            room = int(self.rng.integers(len(self.rooms)))
        else:
            room = self.room
        self.session -= 1  # _spawn_cycle counts a session; a reset is not a new game session
        self._spawn_cycle(room, float(self.rng.uniform(3, ROOM_W - 3)) * TILE)
        self.session += 1
        if "food" in kv:
            self.food = int(kv["food"])
        if "karma" in kv:
            self.karma = min(int(kv["karma"]), self.cap)
        if "timer_left" in kv:
            self.timer = max(0, self.cycle_length - int(kv["timer_left"]))
        if kv.get("fruit"):
            self.fruit[room] = list(self.rng.uniform(3, ROOM_W - 3, size=int(kv["fruit"])) * TILE)
        if kv.get("creatures") == "clear":
            self.lizard = None
        if "GreenLizard" in kv.get("spawn", ""):
            self.lizard = [room, float(self.rng.uniform(3, ROOM_W - 3)) * TILE, (FLOOR + 1) * TILE]
        self.episode += 1
        return {"type": "reset_done", "t": self.tick, "episode": self.episode,
                "room": self.rooms[self.room], "food": self.food, "karma": self.karma, "timer": self.timer}

    def in_water(self) -> bool:
        return self.room % 2 == 1 and self.room != self.shelter and \
            17 * TILE < self.x < 23 * TILE and self.y < 4 * TILE

    # ---- one tick ----------------------------------------------------------------------
    def tick_once(self, a: np.ndarray) -> str | None:
        """Advance one tick with wire action a; returns 'death' / 'sleep' when the session ends."""
        self.tick += 1
        self.room_changed = False
        x, _, jmp, _, pckp = (int(v) for v in a)
        if not self.alive:
            self.dead_for += 1
            return "death" if self.dead_for >= 40 else None
        on_ground = self.y <= (FLOOR + 1) * TILE + 0.1
        self.vx += (4.0 * x - self.vx) * 0.3
        if jmp and on_ground and not self.history[0, 2]:
            self.vy = 9.0
        self.vy -= 0.8
        self.x += self.vx
        self.y = max((FLOOR + 1) * TILE, self.y + self.vy)
        if self.y <= (FLOOR + 1) * TILE:
            self.vy = 0.0
        # Room transitions at the edges.
        if self.x < 0:
            if self.room > 0:
                self.room -= 1
                self.x = (ROOM_W - 1) * TILE
                self.room_changed = True
            else:
                self.x, self.vx = 0.0, 0.0
        elif self.x > ROOM_W * TILE:
            if self.room < len(self.rooms) - 1:
                self.room += 1
                self.x = 1 * TILE
                self.room_changed = True
            else:
                self.x, self.vx = ROOM_W * TILE, 0.0
        # Eating.
        fr = self.fruit.get(self.room, [])
        if pckp:
            for i, fx in enumerate(fr):
                if abs(fx - self.x) < 30 and abs(self.y - (FLOOR + 1) * TILE) < 30:
                    fr.pop(i)
                    self.food = min(self.food + 1, self.max_food)
                    break
        # Breath.
        if self.in_water():
            self.air = max(0.0, self.air - 1.0 / 300)
            if self.air <= 0:
                self.alive = False
        else:
            self.air = min(1.0, self.air + 1.0 / 40)
        # Lizard.
        if self.lizard is not None and self.lizard[0] == self.room:
            d = self.x - self.lizard[1]
            if abs(d) < 12 * TILE:
                self.lizard[1] += np.sign(d) * 2.5
            if abs(d) < 15 and self.y < 3 * TILE:
                self.alive = False
        # Cycle.
        self.timer += 1
        if abs(self.vx) < 0.5 and on_ground:
            self.still += 1
        else:
            self.still = 0
        if self.room == self.shelter and self.food >= self.hib and self.still >= 40:
            return "sleep"
        if self.timer >= self.cycle_length + 40 and self.room != self.shelter:
            self.alive = False
        return None

    def end_session(self, reason: str) -> None:
        if reason == "sleep":
            self.karma = min(self.karma + 1, self.cap)
            self.food = max(self.food - self.hib, 0)
        else:
            self.karma = max(self.karma - 1, 0)
            self.food = 0
        self._spawn_cycle(self.shelter, 8 * TILE)

    # ---- rows --------------------------------------------------------------------------
    def row(self, dtype, consumed: np.ndarray) -> np.ndarray:
        r = np.zeros((), dtype=dtype)
        r["tick"], r["episode"], r["session"] = self.tick, self.episode, self.session
        r["step_tick"] = self.tick % 8
        r["room"], r["room_changed"] = self.room, self.room_changed
        r["consumed"], r["action"] = 1, consumed
        r["alive"] = self.alive
        on_ground = self.y <= (FLOOR + 1) * TILE + 0.1
        r["standing"], r["can_jump"] = on_ground, 5 if on_ground else 0
        r["animation"] = 5 if not self.alive else (4 if self.in_water() else (2 if abs(self.vx) > 0.5 else 1))
        r["body_mode"] = 4 if not self.alive else (3 if self.in_water() else 1)
        r["grasp0_type"] = r["grasp1_type"] = -1
        r["aerobic_level"], r["air_in_lungs"] = 0.0, self.air
        r["pos0"] = (self.x, self.y + 10)
        r["pos1"] = (self.x, self.y - 5)
        r["vel0"] = r["vel1"] = (self.vx, self.vy)
        r["cycle_timer"], r["cycle_length"] = self.timer, self.cycle_length
        r["rain_hit"] = self.timer >= self.cycle_length
        r["in_shelter"] = self.room == self.shelter
        r["food_to_hibernate"], r["food_in_stomach"], r["max_food"] = self.hib, self.food, self.max_food
        r["karma"], r["karma_cap"] = self.karma, self.cap
        r["food_path"], r["shelter_path"], r["eat_speed"] = -1, -1, 1   # the fake world has no paths
        r["food_hibernate_base"] = self.hib
        # Sleep fields, by this fake's own rule: fed, in the shelter, still for 40 ticks.
        sh = self.room == self.shelter
        r["sleep_path"] = 0 if sh else -1
        r["start_shelter"] = 0
        r["ready_for_win"] = sh and self.food >= self.hib
        r["still_ticks_left"] = max(0, 40 - self.still) if sh else -1
        r["force_ticks_left"] = -1
        r["sleep_speed"] = 1
        r["input_history"] = self.history.reshape(-1)
        r["cr_type"] = r["cr_ai_kind"] = r["cr_behavior"] = -1
        if self.lizard is not None and self.lizard[0] == self.room:
            r["cr_type"][0], r["cr_ai_kind"][0] = 1, 0
            r["cr_behavior"][0] = 1 if abs(self.x - self.lizard[1]) < 12 * TILE else 0
            r["cr_flags"][0] = 1
            r["cr_rel_pos"][0:2] = (self.lizard[1] - self.x, self.lizard[2] - self.y)
        fr = sorted(self.fruit.get(self.room, []), key=lambda fx: abs(fx - self.x))[:4]
        r["food_type"] = -1
        for i, fx in enumerate(fr):
            r["food_type"][i], r["food_points"][i], r["food_flags"][i] = 1, 1, 1
            r["food_rel_pos"][2 * i:2 * i + 2] = (fx - self.x, (FLOOR + 1) * TILE - self.y)
        r["item_type"] = -1
        r["speed"] = abs(self.vx)
        return r

    def frame(self) -> np.ndarray:
        """A crude rendering of the room around the slugcat: [H, W, 3] uint8, row 0 = top.

        One frame pixel per 4 world units, centred on the slugcat; floor grey, water blue, exits
        orange, fruit red, lizard green, slugcat white, off-room black.
        """
        H, W, px = O.FRAME_H, O.FRAME_W, 4.0
        img = np.zeros((H, W, 3), np.uint8)
        xs = self.x + (np.arange(W) - W / 2) * px
        ys = self.y + (H / 2 - np.arange(H)) * px                      # row 0 = top
        X, Y = np.meshgrid(xs, ys)
        inside = (X >= 0) & (X < ROOM_W * TILE) & (Y >= 0) & (Y < ROOM_H * TILE)
        img[inside] = (40, 40, 48)
        img[inside & (Y < (FLOOR + 1) * TILE)] = (120, 120, 120)
        if self.room % 2 == 1 and self.room != self.shelter:
            img[inside & (X > 17 * TILE) & (X < 23 * TILE) & (Y < 4 * TILE) & (Y >= (FLOOR + 1) * TILE)] = (40, 80, 200)
        img[inside & ((X < TILE) | (X > (ROOM_W - 1) * TILE)) & (Y < (FLOOR + 3) * TILE)] = (230, 140, 30)
        for fx in self.fruit.get(self.room, []):
            img[(np.abs(X - fx) < 8) & (np.abs(Y - (FLOOR + 1.5) * TILE) < 8)] = (220, 30, 30)
        if self.lizard is not None and self.lizard[0] == self.room:
            img[(np.abs(X - self.lizard[1]) < 20) & (np.abs(Y - self.lizard[2]) < 8)] = (40, 200, 60)
        img[(np.abs(X - self.x) < 6) & (np.abs(Y - (self.y + 6)) < 12)] = (250, 250, 250)
        return img


class FakeGame:
    """Serve one FakeWorld on a localhost port. Runs in a daemon thread."""

    def __init__(self, port: int, **world_kw):
        self.port = port
        self.layout = build_layout()
        self.schema = schema_json(self.layout)
        self.dtype = proto.parse_schema(json.dumps(self.schema).encode()).dtype
        self.world = FakeWorld(**world_kw)
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", port))
        self.srv.listen(1)
        self.thread = threading.Thread(target=self._serve, daemon=True, name=f"fake{port}")
        self.stop = threading.Event()
        self.ticks_simulated = 0

    def start(self) -> "FakeGame":
        self.thread.start()
        return self

    def _send(self, s, t, payload: bytes):
        s.sendall(struct.pack("<BI", t, len(payload)) + payload)

    def _event(self, s, ev: dict):
        self._send(s, proto.MSG_EVENT, json.dumps(ev).encode())

    def _serve(self):
        try:
            s, _ = self.srv.accept()
        except OSError:
            return
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        blob = json.dumps(self.schema).encode()
        s.sendall(proto.MAGIC + struct.pack("<I", len(blob)) + blob)
        w = self.world
        self._event(s, {"type": "world", "region": "SU", "first_room_index": 0, "rooms": w.rooms,
                        "shelters": [w.rooms[w.shelter]],
                        "connections": {n: [w.rooms[j] for j in (i - 1, i + 1) if 0 <= j < len(w.rooms)]
                                        for i, n in enumerate(w.rooms)}})
        actions = np.zeros((O.DECISIONS, 5), np.int8)
        buf: list[np.ndarray] = []
        try:
            while not self.stop.is_set():
                ended = None
                for k in range(O.TICKS):
                    a = actions[k // O.TICKS_PER_DECISION]
                    w.history = np.roll(w.history, 1, axis=0)
                    w.history[0] = a
                    ended = w.tick_once(a)
                    self.ticks_simulated += 1
                    buf.append(w.row(self.dtype, a))
                    if ended:
                        break
                if ended:
                    # Session over: event first, then the final flush, which is not an exchange.
                    self._event(s, {"type": "session_end", "t": w.tick, "reason": ended,
                                    "episode": w.episode})
                fr = w.frame()
                block = struct.pack("<iHHB", w.tick, O.FRAME_W, O.FRAME_H, 3) + fr.tobytes()
                self._send(s, proto.MSG_CHUNK, bytes([len(buf)]) + b"".join(r.tobytes() for r in buf) + block)
                buf = []
                if ended:
                    w.end_session(ended)
                    self._event(s, {"type": "session_start", "t": w.tick, "episode": w.episode})
                    actions = np.zeros((O.DECISIONS, 5), np.int8)
                    continue  # the next chunk starts the new session; no exchange for a partial one
                cmds = []
                while True:
                    t, payload = proto.recv_frame(s)
                    if t == proto.MSG_ACTION:
                        actions = np.frombuffer(payload, np.int8).reshape(O.DECISIONS, 5).copy()
                        break
                    if t == proto.MSG_CMD:
                        cmds.append(payload.decode())
                for c in cmds:
                    parts = c.split()
                    if parts and parts[0] == "reset":
                        kv = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
                        self._event(s, w.apply_reset(kv))
                        actions = np.zeros((O.DECISIONS, 5), np.int8)
                    elif parts and parts[0] == "kill":
                        w.alive = False
        except (OSError, proto.ProtocolError):
            pass
        finally:
            s.close()
            self.srv.close()

    def close(self):
        self.stop.set()
        try:
            self.srv.close()
        except OSError:
            pass
