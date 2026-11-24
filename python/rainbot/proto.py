"""RainBot bridge wire protocol (docs/PROTOCOL.md).

Transport is a localhost TCP stream, little-endian throughout. The game is frozen while Python
thinks, so a ~50 us loopback round trip against a 25 ms tick budget costs nothing. Shared memory
between Wine and Linux would add work for no gain.

The observation layout is self-describing. The mod owns the layout and sends it as JSON at
handshake time. Python builds a numpy structured dtype from that description, not from a
hand-mirrored struct, so the C# and Python layouts cannot drift apart without an error.

Framing, both directions:

    uint8  msg_type
    uint32 length
    bytes  payload[length]

Handshake, mod -> python, once on connect:

    bytes8 magic = b"RAINBOT\\x01"
    uint32 schema_len
    bytes  schema[schema_len]   # UTF-8 JSON, see SchemaError for the required shape
"""
from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

MAGIC = b"RAINBOT\x01"
PROTOCOL = 11

# mod -> python
MSG_OBS = 1
MSG_EVENT = 2
MSG_LOG = 3
#: ``uint8 n`` then ``n`` observation rows (one per tick, oldest first), then an optional frame.
MSG_CHUNK = 4
# python -> mod
MSG_ACTION = 1
MSG_CMD = 2

_HEADER = struct.Struct("<BI")

#: Factored action (x, y, jmp, thrw, pckp), not one 72-way softmax: the policy head has one
#: categorical per component.
ACTION_DIMS = (3, 3, 2, 2, 2)
ACTION_BYTES = 5

#: numpy dtype codes the mod may declare. Anything else is a protocol error, not a guess: a wrong
#: guess would corrupt data silently.
_ALLOWED_DTYPES = {"i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f4", "f8"}


class ProtocolError(RuntimeError):
    """Raised on malformed framing, bad magic, or a schema the client cannot honour."""


@dataclass(frozen=True)
class Schema:
    """Parsed handshake: what the mod says it will send, and how to read it."""

    protocol: int
    game_version: str
    obs_bytes: int
    dtype: np.dtype
    #: ExtEnum index -> name tables, pinned at capture time. ExtEnum integer values change with
    #: game version and mod set, so a dataset without its own tables would relabel itself. Most
    #: entries are ``list[str]`` indexed by the stored value. ``ai_behavior`` maps an AI class
    #: name to ``list[str]``, because each AI type declares its own Behavior enum.
    enums: dict[str, Any]
    raw: dict[str, Any]

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self.dtype.names or ())


def parse_schema(blob: bytes) -> Schema:
    """Turn the handshake JSON into a numpy structured dtype.

    Explicit per-field offsets and an explicit itemsize let the C# struct have any padding.
    numpy's default packing is never assumed to match it.
    """
    try:
        doc = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"handshake schema is not valid UTF-8 JSON: {e}") from e

    # The version check is permissive in both directions. Field names, dtypes and offsets all
    # arrive in the handshake, and the featurizer tolerates unknown and missing fields, so the
    # protocol number only guards the framing, which has not changed. A strict check would refuse
    # recorded sessions that parse correctly.
    got = doc.get("protocol")
    if got is None:
        raise ProtocolError("handshake carries no protocol number")
    if got > PROTOCOL:
        print(f"note: mod speaks protocol {got}, client built for {PROTOCOL}; "
              f"reading via the self-described schema")

    obs_bytes = int(doc["obs_bytes"])
    names, formats, offsets = [], [], []
    for f in doc["fields"]:
        name, dt, off = f["name"], f["dtype"], int(f["offset"])
        if dt not in _ALLOWED_DTYPES:
            raise ProtocolError(f"field {name!r} declares unsupported dtype {dt!r}")
        shape = tuple(int(s) for s in f.get("shape", ()))
        fmt = np.dtype(f"<{dt}")
        width = fmt.itemsize * int(np.prod(shape)) if shape else fmt.itemsize
        if off < 0 or off + width > obs_bytes:
            raise ProtocolError(
                f"field {name!r} at offset {off} (+{width}B) overruns obs_bytes={obs_bytes}"
            )
        names.append(name)
        formats.append((fmt, shape) if shape else fmt)
        offsets.append(off)

    if len(set(names)) != len(names):
        raise ProtocolError("handshake schema declares duplicate field names")

    dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": obs_bytes}
    )

    act = doc.get("action", {})
    if tuple(act.get("dims", ACTION_DIMS)) != ACTION_DIMS:
        raise ProtocolError(f"mod declares action dims {act.get('dims')!r}, expected {ACTION_DIMS}")

    return Schema(
        protocol=doc["protocol"],
        game_version=str(doc.get("game_version", "unknown")),
        obs_bytes=obs_bytes,
        dtype=dtype,
        enums=dict(doc.get("enums", {})),
        raw=doc,
    )


def encode_action(action) -> bytes:
    """(x, y, jmp, thrw, pckp) -> 5 signed bytes.

    x and y are in {-1, 0, 1} to match the game's InputPackage directly, so the C# side needs no
    encoding convention.
    """
    a = np.asarray(action, dtype=np.int8)
    if a.shape != (ACTION_BYTES,):
        raise ValueError(f"action must have shape ({ACTION_BYTES},), got {a.shape}")
    if not (-1 <= a[0] <= 1 and -1 <= a[1] <= 1):
        raise ValueError(f"x and y must be in {{-1,0,1}}, got {a[0]}, {a[1]}")
    if not np.all((a[2:] == 0) | (a[2:] == 1)):
        raise ValueError(f"button components must be 0 or 1, got {a[2:]}")
    return a.tobytes()


def encode_actions(actions) -> bytes:
    """[decisions, 5] actions -> wire bytes (one MSG_ACTION carries a whole chunk)."""
    a = np.asarray(actions, dtype=np.int8).reshape(-1, ACTION_BYTES)
    return b"".join(encode_action(row) for row in a)


def decode_chunk(payload: bytes, schema: "Schema"):
    """MSG_CHUNK payload -> (n real rows, rows, frame or None, frame_tick).

    The latest rendered frame may follow the rows: int32 frame_tick, uint16 w, uint16 h, uint8 c,
    then h*w*c bytes, row 0 = top.
    """
    n = payload[0]
    end = 1 + n * schema.obs_bytes
    if len(payload) < end:
        raise ProtocolError(f"chunk declares {n} rows of {schema.obs_bytes}B, carries {len(payload) - 1}B")
    rows = np.frombuffer(payload, dtype=schema.dtype, count=n, offset=1)
    frame, frame_tick = None, -1
    if len(payload) > end:
        frame_tick, w, h, c = struct.unpack_from("<iHHB", payload, end)
        body = memoryview(payload)[end + 9:]
        if len(body) != w * h * c:
            raise ProtocolError(f"frame {w}x{h}x{c} carries {len(body)}B")
        frame = np.frombuffer(body, dtype=np.uint8).reshape(h, w, c)
    return n, rows, frame, frame_tick


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes or raise. Short reads on a stream socket are normal."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ProtocolError(f"connection closed after {got}/{n} bytes")
        got += k
    return bytes(buf)


def recv_frame(sock: socket.socket) -> tuple[int, bytes]:
    msg_type, length = _HEADER.unpack(recv_exact(sock, _HEADER.size))
    return msg_type, recv_exact(sock, length)


def send_frame(sock: socket.socket, msg_type: int, payload: bytes) -> None:
    sock.sendall(_HEADER.pack(msg_type, len(payload)) + payload)


def recv_handshake(sock: socket.socket) -> Schema:
    magic = recv_exact(sock, len(MAGIC))
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic!r}, expected {MAGIC!r}")
    (schema_len,) = struct.unpack("<I", recv_exact(sock, 4))
    return parse_schema(recv_exact(sock, schema_len))
