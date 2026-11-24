"""Observation rows -> model inputs.

One agent step delivers 8 rows (one per tick, post-tick state). ``Featurizer.chunk`` turns a batch
of those into fixed-shape numpy arrays. The model owns the learned parts (embeddings, running
normalisation), so this file is pure, deterministic bookkeeping.

Names, not indices. Animation, body mode, creature type, object type, AI behaviour and room are
keyed by name through ``Vocab`` tables that persist in the checkpoint. ExtEnum indices are
registration order and shift with game version and mod set, room indices are region-local, and AI
kinds register in the order creatures are first met. An index-keyed embedding would mean different
things in different runs. The mod sends index->name tables at handshake and ``enum`` events for
tables that grow later. ``Vocab`` maps names to stable ids. Id 0 is always "absent/unknown".

The scales come from percentiles over 210k real ticks. The model applies running normalisation
 on top, so the scales only need to be in the right decade.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TICKS = 8
DECISIONS = 4
TICKS_PER_DECISION = 2
#: The rendered screen, downscaled by the mod: the agent's only visual input. One frame per step,
#: the latest rendered when the chunk was sent.
FRAME_H, FRAME_W = 96, 160
#: Frame channels: RGB plus the movement-graph distance field from the slugcat (protocol v11).
#: Frames with 3 channels (older mods, the fake game) get an empty distance channel.
FRAME_C = 4

POS_SCALE, VEL_SCALE, VEL_CLIP = 1000.0, 10.0, 4.0
ENT_POS_SCALE, ENT_POS_CLIP = 500.0, 6.0
N_ENT_CONT = 8
#: Entity tokens per frame: nearest creatures (6), food (4) and items (4), as the mod sends them.
N_ENTITIES = 6 + 4 + 4

#: Fixed vocabulary capacities. Embedding tables are this size; overflow maps to id 0 with a warning.
CAPACITY = {"animation": 64, "body_mode": 32, "object_type": 128, "creature_type": 128,
            "behavior": 128, "room": 512}


@dataclass
class Vocab:
    """name -> id tables, persisted with the model. Id 0 = absent / unknown."""

    tables: dict[str, dict[str, int]] = field(default_factory=dict)
    overflowed: set = field(default_factory=set)

    def id(self, table: str, name: str | None) -> int:
        if not name:
            return 0
        t = self.tables.setdefault(table, {})
        i = t.get(name)
        if i is None:
            if len(t) + 1 >= CAPACITY[table]:
                if (table, name) not in self.overflowed:
                    self.overflowed.add((table, name))
                    print(f"warning: vocab {table!r} full, {name!r} maps to unknown")
                return 0
            i = t[name] = len(t) + 1
        return i

    def state_dict(self) -> dict:
        return {k: dict(v) for k, v in self.tables.items()}

    def load_state_dict(self, d: dict) -> None:
        self.tables = {k: dict(v) for k, v in d.items()}


class IndexMap:
    """Per-connection ExtEnum index -> vocab id lookup arrays (index -1 -> 0)."""

    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.maps: dict[str, np.ndarray] = {}
        self.behaviors: dict[int, np.ndarray] = {}     # ai_kind index -> behavior idx -> vocab id
        self.ai_kind_names: dict[int, str] = {}
        self.rooms = np.zeros(1, dtype=np.int64)       # room index - first_room_index -> vocab id
        self.first_room = 0

    def set_table(self, key: str, table_name: str, names: list[str]) -> None:
        self.maps[key] = np.array([self.vocab.id(table_name, n) for n in names] + [0], dtype=np.int64)

    def from_schema(self, enums: dict) -> None:
        for key, tname in (("animation", "animation"), ("body_mode", "body_mode"),
                           ("creature_type", "creature_type"), ("object_type", "object_type")):
            self.set_table(key, tname, list(enums.get(key, [])))
        for i, name in enumerate(enums.get("ai_kind", [])):
            self.add_ai_kind(i, name, (enums.get("ai_behavior") or {}).get(name, []))

    def add_ai_kind(self, index: int, name: str, behaviors: list[str]) -> None:
        self.ai_kind_names[index] = name
        short = name.rsplit(".", 1)[-1]
        self.behaviors[index] = np.array(
            [self.vocab.id("behavior", f"{short}.{b}") for b in behaviors] + [0], dtype=np.int64)

    def set_world(self, first_room_index: int, rooms: list[str]) -> None:
        self.first_room = int(first_room_index)
        self.rooms = np.array([self.vocab.id("room", r) for r in rooms] + [0], dtype=np.int64)

    def on_event(self, ev: dict) -> None:
        t = ev.get("type")
        if t == "world":
            self.set_world(ev.get("first_room_index", 0), ev.get("rooms", []))
        elif t == "enum" and ev.get("table") == "ai_kind":
            self.add_ai_kind(int(ev["index"]), ev["name"], ev.get("behaviors", []))

    @staticmethod
    def _lookup(table: np.ndarray, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        ok = (idx >= 0) & (idx < len(table) - 1)
        return np.where(ok, table[np.clip(idx, 0, len(table) - 1)], 0)

    def lookup(self, key: str, idx) -> np.ndarray:
        return self._lookup(self.maps.get(key, np.zeros(1, np.int64)), idx)

    def room(self, idx) -> np.ndarray:
        return self._lookup(self.rooms, np.asarray(idx, dtype=np.int64) - self.first_room)

    def behavior(self, ai_kind, beh) -> np.ndarray:
        ai_kind = np.asarray(ai_kind, dtype=np.int64)
        out = np.zeros(ai_kind.shape, dtype=np.int64)
        for k in np.unique(ai_kind):
            if k < 0 or int(k) not in self.behaviors:
                continue
            m = ai_kind == k
            out[m] = self._lookup(self.behaviors[int(k)], np.asarray(beh)[m])
        return out


def _f(rows, name, width=None):
    a = np.asarray(rows[name], dtype=np.float32)
    return a if width is None else a.reshape(*a.shape[:2], width)


BODY_DIM = 2 + 2 + 2 + 2 + 2 + 17
NEEDS_DIM = 13


class Featurizer:
    """Rows [B, 8] (structured) -> dict of arrays, shapes documented per key."""

    def __init__(self, index: IndexMap):
        self.ix = index

    def chunk(self, rows: np.ndarray, frames: np.ndarray | None = None,
              frame_ticks: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """rows [B, 8]; frames [B, H, W, C] uint8 (the step's latest rendered frame)."""
        B, T = rows.shape
        assert T == TICKS, rows.shape
        out: dict[str, np.ndarray] = {}

        # ---- body, every tick ------------------------------------------------------------
        pos0, pos1 = _f(rows, "pos0", 2) / POS_SCALE, _f(rows, "pos1", 2) / POS_SCALE
        vel0 = np.clip(_f(rows, "vel0", 2) / VEL_SCALE, -VEL_CLIP, VEL_CLIP)
        vel1 = np.clip(_f(rows, "vel1", 2) / VEL_SCALE, -VEL_CLIP, VEL_CLIP)
        scal = np.stack([
            _f(rows, "alive"), _f(rows, "standing"), _f(rows, "exhausted"), _f(rows, "grabbed_by"),
            _f(rows, "can_jump") / 4.0, _f(rows, "stun") / 40.0, _f(rows, "roll_counter") / 20.0,
            _f(rows, "slide_counter") / 20.0, _f(rows, "aerobic_level"),
            _f(rows, "animation_frame") / 40.0, _f(rows, "in_shortcut"), _f(rows, "holding_edible"),
            np.minimum(_f(rows, "eat_counter"), 40.0) / 40.0, np.minimum(_f(rows, "speed"), 40.0) / 10.0,
            np.minimum(_f(rows, "hit_dealt"), 4.0), np.minimum(_f(rows, "hit_taken"), 4.0),
            _f(rows, "sleeping"),
        ], axis=-1)
        out["body"] = np.concatenate([pos0, pos1, (pos1 - pos0) * 50.0, vel0, vel1, scal],
                                     axis=-1).astype(np.float32)
        out["animation"] = self.ix.lookup("animation", rows["animation"])
        out["body_mode"] = self.ix.lookup("body_mode", rows["body_mode"])
        out["grasp"] = np.stack([self.ix.lookup("object_type", rows["grasp0_type"]),
                                 self.ix.lookup("object_type", rows["grasp1_type"])], axis=-1)

        # ---- needs and rain timer --------------------------------------------------------
        maxf = np.maximum(_f(rows, "max_food"), 1.0)
        food = _f(rows, "food_in_stomach") + _f(rows, "quarter_food") / 4.0
        hib = _f(rows, "food_to_hibernate")
        cap = np.maximum(_f(rows, "karma_cap"), 1.0)
        karma = np.maximum(_f(rows, "karma"), 0.0)
        L = np.maximum(_f(rows, "cycle_length"), 1.0)
        left = np.clip((L - _f(rows, "cycle_timer")) / L, -1.0, 1.0)
        out["needs"] = np.stack([
            food / maxf, np.maximum(hib - food, 0.0) / maxf, hib / maxf,
            karma / cap, np.maximum(cap - karma, 0) / cap, cap / 9.0, _f(rows, "karma_protected"),
            _f(rows, "air_in_lungs"), np.maximum(1.0 - _f(rows, "air_in_lungs"), 0.0),
            left, _f(rows, "rain_hit"), _f(rows, "in_shelter"), L / 40000.0,
        ], axis=-1).astype(np.float32)

        # ---- action consumed each tick, room ---------------------------------------------
        a = np.asarray(rows["action"], dtype=np.int64).reshape(B, T, 5).copy()
        a[..., :2] += 1
        a[..., :2] = np.clip(a[..., :2], 0, 2)
        a[..., 2:] = np.clip(a[..., 2:], 0, 1)
        out["action"] = a
        out["room"] = self.ix.room(rows["room"])
        out["room_changed"] = np.asarray(rows["room_changed"], dtype=np.bool_)

        # ---- vision: the frame, plus entity tokens at the step's last tick -----------------
        if frames is None:
            frames = np.zeros((B, FRAME_H, FRAME_W, FRAME_C), np.uint8)
        frames = np.asarray(frames, np.uint8)
        if frames.shape[-1] < FRAME_C:
            pad = np.zeros(frames.shape[:-1] + (FRAME_C - frames.shape[-1],), np.uint8)
            frames = np.concatenate([frames, pad], -1)
        out["frame"] = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))   # [B, C, H, W]
        ft = np.full(B, -1) if frame_ticks is None else np.asarray(frame_ticks)
        out["frame_valid"] = ft >= 0
        ents = self._entities(rows[:, -1:])
        out.update({k: v[:, 0] for k, v in ents.items()})                # [B, K(, 8)]
        return out

    def _entities(self, v: np.ndarray) -> dict[str, np.ndarray]:
        """Creatures, food and items as one entity set per row: [B, T, K]."""
        B, T = v.shape
        ct = self.ix.lookup("creature_type", v["cr_type"])                  # [B,T,Kc]
        cf = np.asarray(v["cr_flags"]).astype(np.int64)
        c_present = (cf & 1) > 0
        n_ct = CAPACITY["creature_type"]
        c_type = np.where(c_present, ct, 0)
        c_beh = np.where(c_present, self.ix.behavior(v["cr_ai_kind"], v["cr_behavior"]), 0)
        c_pos = _f(v, "cr_rel_pos").reshape(B, T, -1, 2)
        c_vel = _f(v, "cr_rel_vel").reshape(B, T, -1, 2)
        c_ext = np.stack([(cf >> 1) & 1, (cf >> 2) & 1, (cf >> 3) & 1], axis=-1).astype(np.float32)

        ff = np.asarray(v["food_flags"]).astype(np.int64)
        f_present = (ff & 1) > 0
        f_type = np.where(f_present, self.ix.lookup("object_type", v["food_type"]) + n_ct, 0)
        f_pos = _f(v, "food_rel_pos").reshape(B, T, -1, 2)
        f_ext = np.stack([np.asarray(v["food_points"], np.float32) / 4.0, ((ff >> 1) & 1).astype(np.float32),
                          np.zeros(ff.shape, np.float32)], axis=-1)

        itf = np.asarray(v["item_flags"]).astype(np.int64)
        i_present = (itf & 1) > 0
        i_type = np.where(i_present, self.ix.lookup("object_type", v["item_type"]) + n_ct, 0)
        i_pos = _f(v, "item_rel_pos").reshape(B, T, -1, 2)
        i_ext = np.stack([((itf >> 1) & 1).astype(np.float32), np.zeros(itf.shape, np.float32),
                          np.zeros(itf.shape, np.float32)], axis=-1)

        def cont(pos, vel, present, ext):
            m = present[..., None].astype(np.float32)
            p = np.clip(pos / ENT_POS_SCALE, -ENT_POS_CLIP, ENT_POS_CLIP) * m
            vv = np.clip(vel / VEL_SCALE, -VEL_CLIP, VEL_CLIP) * m
            return np.concatenate([p, vv, m, ext * m], axis=-1)

        kinds = [np.zeros_like(c_type), np.ones_like(f_type), np.full_like(i_type, 2)]
        return {
            "ent_kind": np.concatenate(kinds, axis=-1),
            "ent_type": np.concatenate([c_type, f_type, i_type], axis=-1),
            "ent_behavior": np.concatenate([c_beh, np.zeros_like(f_type), np.zeros_like(i_type)], axis=-1),
            "ent_cont": np.concatenate([cont(c_pos, c_vel, c_present, c_ext),
                                        cont(f_pos, np.zeros_like(f_pos), f_present, f_ext),
                                        cont(i_pos, np.zeros_like(i_pos), i_present, i_ext)],
                                       axis=-2).astype(np.float32),
        }


def n_entity_types() -> int:
    """Entity type ids: creature types first, then object types."""
    return CAPACITY["creature_type"] + CAPACITY["object_type"]
