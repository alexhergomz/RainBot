"""Reader for RainBot session logs written by the mod's log mode.

Layout of a `.obs` file:

    bytes6   magic "RBLOG\\x02"
    uint32   schema_len
    bytes    schema JSON
    ...      fixed-stride observation records, `obs_bytes` each

The stride is fixed, so the record array is memory-mapped and viewed as a structured dtype: a long
session needs no parsing and no copy.

The schema comes from the `.schema.json` sidecar when present, not from the copy in the header. Both
describe the same field layout, but the ExtEnum name tables grow as new creatures appear during a
session. The header holds the tables from the first tick; the sidecar is rewritten complete on close.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..proto import Schema, parse_schema

MAGIC = b"RBLOG\x02"


@dataclass
class Session:
    """One recorded play session."""

    path: Path
    schema: Schema
    obs: np.ndarray            # structured array, one row per tick
    events: list[dict]

    def __len__(self) -> int:
        return len(self.obs)

    @property
    def duration_s(self) -> float:
        """Wall-clock span, which is not len/40: the log spans menus and pauses too."""
        if "wall_ms" not in (self.schema.dtype.names or ()) or not len(self.obs):
            return len(self.obs) / 40.0
        return float(self.obs["wall_ms"][-1] - self.obs["wall_ms"][0]) / 1000.0

    def enum_names(self, key: str) -> list[str]:
        """index -> name table for an ExtEnum, e.g. 'creature_type', 'animation'."""
        return list(self.schema.raw.get("enums", {}).get(key, []))

    def behavior_names(self, ai_kind: int) -> list[str]:
        """Behavior table for one AI kind, indexed by the cr_behavior field."""
        kinds = self.schema.raw.get("enums", {}).get("ai_kind", [])
        if not 0 <= ai_kind < len(kinds):
            return []
        return list(self.schema.raw.get("enums", {}).get("ai_behavior", {}).get(kinds[ai_kind], []))

    def episodes(self) -> list[tuple[int, int]]:
        """Split into (start, end) row ranges on death events.

        Ranges are half-open and expressed as row indices, so `obs[a:b]` is one episode.
        """
        deaths = sorted(e["t"] for e in self.events if e.get("type") == "death")
        if "tick" not in (self.schema.dtype.names or ()):
            return [(0, len(self.obs))]
        ticks = self.obs["tick"]
        bounds, start = [], 0
        for d in deaths:
            end = int(np.searchsorted(ticks, d, side="right"))
            if end > start:
                bounds.append((start, end))
            start = end
        if start < len(self.obs):
            bounds.append((start, len(self.obs)))
        return bounds


def load(path: str | Path) -> Session:
    """Load a session by path to its `.obs` file (or its base name)."""
    p = Path(path)
    if p.suffix != ".obs":
        p = p.with_suffix(".obs")

    with open(p, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{p}: bad magic {magic!r}, expected {MAGIC!r}")
        (schema_len,) = struct.unpack("<I", f.read(4))
        header_schema = f.read(schema_len)
        data_start = f.tell()

    sidecar = p.with_suffix(".schema.json")
    schema = parse_schema(sidecar.read_bytes() if sidecar.exists() else header_schema)

    total = p.stat().st_size - data_start
    n, remainder = divmod(total, schema.obs_bytes)
    if remainder:
        # A session killed mid-write leaves a partial final record. Dropping it is correct;
        # silently reinterpreting the file would not be.
        print(f"warning: {p.name} ends with a {remainder}B partial record, dropping it")
    obs = np.memmap(p, dtype=schema.dtype, mode="r", offset=data_start, shape=(n,))

    events = []
    ev_path = p.with_suffix(".events.jsonl")
    if ev_path.exists():
        for line in ev_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return Session(path=p, schema=schema, obs=obs, events=events)


def find_sessions(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("session-*.obs"))
