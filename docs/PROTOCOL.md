# Bridge protocol (v11): chunked lockstep and screen frames

The contract between the mod (`mod/RainBot/`) and the Python agent (`python/rainbot/`). The design
(`docs/DESIGN.md`) fixes the timing: **one agent step = one
token = 8 game ticks = 4 decisions of 2 ticks each.** Everything here follows from that.

Transport: localhost TCP. Framing: `uint8 type, uint32 len, payload`. Handshake: magic
`RAINBOT\x01`, then a length-prefixed JSON schema. The observation **row** describes itself: Python
builds its numpy dtype from the schema's `fields`, never from a mirrored struct. The schema's
`fields` is authoritative.

## Version history

- **v6**: the base transport, framing, handshake, row layout and `autopilot` command.
- **v7**: chunked lockstep at simulation-tick granularity (this document's "Timing"). New row
  fields `tick`, `episode`, `session`, `step_tick`, `room_changed`, `in_shortcut`, `consumed`,
  `action`, `max_food`, `karma_protected`, `tiles`. Removed: `prev_action` (replaced by `action`),
  the 21x21 `u1` tile patch. `MSG_OBS` is unused from v7 on.
- **v8**: every `MSG_CHUNK` carries the latest rendered frame after its rows (see "Frames"). The
  schema has `"frame": {"w", "h", "c", "header_bytes"}` instead of `tile_w`/`tile_h`/`tile_bits`.
  The `tiles` row field is gone (the agent sees the screen instead). New commands `frame on|off` and `den ROOM`. New reset keys
  `fruit_near`, `hold`, `bites`, `eat_counter`.
- **v9**: sleep state by the game's rules: `sleep_path`, `start_shelter`, `ready_for_win`,
  `still_ticks_left`, `force_ticks_left`.
- **v10**: `sleep_speed` (row field and reset key).
- **v11**: a fourth frame channel, D: the movement-graph distance from the slugcat (see "Frames").

The v7 `tiles` field (`u2[24*32]`, a crop centred on the slugcat, row-major, row 0 = bottom) used
these bits: `solid` 1 = `TerrainType.Solid`; `floor` 2 = one-way platform (`TerrainType.Floor`);
`slope` 4; `pole_v` 8 / `pole_h` 16 = beams; `shortcut` 32 = a shortcut entrance
(`TerrainType.ShortcutEntrance`), not pipe bodies; `water` 64 = tile centre below the current
`FloatWaterLevel` of its column; `hazard` 128 = worm grass (the patch tiles plus the two tiles
above them, rescanned on room change and every 200 ticks); `self` 256 = the tiles of body chunks 0
and 1 (the vessel tile while in a pipe); `off_map` 512 = outside the room (a pipe in an unrealized
room is all `off_map`).

## Instances

`RAINBOT_INSTANCE` (environment variable, default `0`) selects the instance. The port is
`54321 + instance` (base port from config). Save data is isolated per instance (see
`mod/RainBot/Instance.cs`): 1.11.8 saves through Kittehface `UserData` under the name returned by
`Options.GetSaveFileName_SavOrExp()`, and that method is hooked so instance N > 0 uses its own file
`sav_rainbotN` in the shared prefix (instance 0 keeps the normal `sav`). Instances > 0 never write
the shared `options` file and record to `RainBotData/instance-N`. `tools/launch_instances.sh N`
launches N direct-Proton instances and seeds each `sav_rainbotN` from `sav` on first use. Proton
passes the variable through to the Windows process. The schema carries `"instance"`.

## Timing

The mod counts **simulation ticks**, not `Player.checkInput` calls (hook: `RainWorldGame.Update`).
A tick is an un-paused `RainWorldGame.Update` with a story session, `!GamePaused`, `processActive`,
and player 0 realized and either in a room or in a pipe. `checkInput` is not called while the
slugcat is inside a shortcut pipe, so an exchange per `checkInput` would span 54-80 frames on every
room change.

Per tick `t`:

1. If `t % 8 == 0` and an agent is connected: **exchange**. Send the 8 rows of the previous chunk
   (`MSG_CHUNK`), block until `MSG_ACTION` (4 decisions) arrives. Commands received while blocked
   are queued and applied right after the exchange, before the tick runs.
2. The injected input for this tick is decision `(t % 8) / 2` (each decision held for 2 ticks). It
   is injected in `RWInput.PlayerInput(0)`, so the game's own post-processing (superLaunchJump,
   Roll) still applies.
3. The tick runs (`orig`).
4. A row describing the **post-tick state** is appended to the chunk buffer, with `action` set to
   the `InputPackage` the sim consumed this tick (or the injected one if `checkInput` did not run,
   for example in a pipe; `consumed` says which).

So token `n` (rows of ticks `8n..8n+7`) is the result of the 4 decisions sent at the exchange
before tick `8n`. Python's policy reads token `n` and answers with decisions for ticks
`8(n+1)..8(n+1)+7`.

The first exchange after connecting, after a reset, or after a new game session starts happens
only once 8 rows exist; until then the slugcat gets neutral input. Rows never span a reset or a
session change: the buffer is cleared, and `episode` or `session` (below) changes.

**Exchanges and the final chunk.** Every `MSG_CHUNK` with `n = 8` sent during a session is an
exchange and is answered by exactly one `MSG_ACTION`. When the game process leaves
`RainWorldGame` (death screen, sleep screen, quit), the mod sends the `session_end` event
**first**, then flushes the buffered rows as a final `MSG_CHUNK` (`n` = 1..8, possibly a full 8)
and does **not** wait for an action. If the buffer is empty, no chunk is sent. Python identifies
the final chunk as "a `MSG_CHUNK` that follows a `session_end` event".

A `reset` is not lockstepped: from the command until `reset_done`/`reset_failed` the game runs
freely with neutral input (the slugcat travels through a pipe to the target room) and no chunks
are sent. Python sends `reset` as a command and still answers the chunk it is holding, if any.

**Frozen endings** (`endings freeze`, with an agent driving): a death, a starvation or a sleep
does not leave the game. The mod books the night by the game's own rules (karma, karma flower,
malnourishment), sends `session_end` with `"frozen":true`, the ending's rows as a final chunk,
then `session_start` with `"frozen":true`. The world then stays frozen: rows of the frozen state
go out in normal lockstep chunks until the next `reset`. With `endings game` (the default) the
game runs its own death, sleep and dream screens.

## Determinism

Drawing never consumes the simulation's random numbers: `RainWorldGame.GrafUpdate` saves and
restores `UnityEngine.Random.state`. `global_seed N` (-1 = off) re-seeds `UnityEngine.Random`
from (N, episode, session) at every new game session and every reset that carries no `seed=` of
its own. Goal: a run depends only on the seed and the actions, at any speed. This allows replays
instead of recordings, GRPO groups from identical states, and reproducible bugs. Check with
`scripts/checks.py determinism` at speed 16 and `physics 1,16`.

## Messages

mod -> python

| type | name | payload |
|---|---|---|
| 1 | `MSG_OBS` | unused since v7 |
| 2 | `MSG_EVENT` | UTF-8 JSON, one event |
| 3 | `MSG_LOG` | UTF-8 text |
| 4 | `MSG_CHUNK` | `uint8 n`, then `n * obs_bytes` bytes of rows, oldest first, then the frame block when frames are on |

python -> mod

| type | name | payload |
|---|---|---|
| 1 | `MSG_ACTION` | `decisions * 5` bytes: per decision `int8 x, int8 y, u8 jmp, u8 thrw, u8 pckp` |
| 2 | `MSG_CMD` | UTF-8 command line |

Python may send `MSG_CMD` at any time, including while the mod is blocked waiting for an action.
The mod reads commands inline in that wait and keeps waiting. It also polls the socket without
blocking once per tick for commands that arrive between exchanges (for example during a reset).
`ping` only keeps the receive timeout from expiring while the learner is busy (updates freeze the
game for tens of seconds).

## Frames

Every `MSG_CHUNK` (exchange and final) carries the latest rendered frame after its rows:

    int32 frame_tick, uint16 w, uint16 h, uint8 c (= 4), then h*w*4 bytes RGBD interleaved, row 0 = top

The header is 9 bytes (`header_bytes`). Captured at end of frame (`mod/RainBot/FrameCapture.cs`):
back buffer -> render texture -> repeated 2x bilinear halving -> exact size -> read back. Default
160x96 (`[vision] frame_w/h` in the config; 0 = no frames). `frame_tick` is the tick counter when
the frame was captured (-1 = none yet; the block is then black). The block is always sent in full
when frames are configured, so the chunk size is constant. Rendering must be on (`render off`
freezes the image). The tick marker is disabled while frames are on, so it is not drawn into what
the agent sees. Until v10 the block was `c = 3`, RGB only.

D (v11) is the slugcat's movement-graph distance to each pixel's tile (the game's moves, the
recorded moves and the human-jump links, with an arrival-time flood into the air), scaled per
room: 255 at the slugcat, 1 at the farthest reachable tile, 0 unreachable or another room.

## Schema

```json
{"protocol": 11, "game_version": "...", "obs_bytes": 437, "k_creatures": 6, "k_foods": 4, "k_items": 4,
 "eat_speed_limit": 3.6, "input_history": 10,
 "ticks_per_step": 8, "ticks_per_decision": 2, "decisions_per_step": 4,
 "frame": {"w": 160, "h": 96, "c": 4, "header_bytes": 9}, "instance": 0,
 "action": {"dims": [3,3,2,2,2], "bytes": 5},
 "creature_flags": {"present":1,"dead":2,"stunned":4,"grabbing_me":8},
 "enums": {...}, "fields": [...]}
```

`enums` has `animation`, `body_mode`, `creature_type` and `object_type`
(`AbstractPhysicalObject.AbstractObjectType` names, index order; `food_type`, `item_type`,
`grasp0_type`, `grasp1_type` index into it). `ai_kind` and `ai_behavior` only list AI classes met
before the handshake; later ones arrive as `enum` events (below).

## Row fields

| field | dtype | meaning |
|---|---|---|
| `tick` | i4 | simulation tick counter, monotonic for the life of the game process (never reset, so the video tick marker stays unique) |
| `episode` | i4 | increments **only** on a completed `reset`; episodes run through deaths and hibernation. Survives session changes |
| `session` | i4 | increments on every new `RainWorldGame` (respawn after death, new cycle after hibernation) |
| `step_tick` | u1 | row's position in its chunk, 0..7 (not `tick % 8`: chunks restart on reset/session/connect) |
| `room` | i4 | `abstractRoom.index` of the room the slugcat is in (abstract position while in a pipe) |
| `room_changed` | u1 | room differs from the previous row's |
| `in_shortcut` | u1 | slugcat is inside a pipe; `pos0`/`pos1` then both hold the vessel's tile centre and velocities are 0 |
| `consumed` | u1 | `checkInput` ran this tick (otherwise the action is what was injected) |
| `action` | i1[5] | input the sim consumed this tick |
| `max_food` | i2 | `slugcatStats.maxFood` (food meter size) |
| `karma_protected` | u1 | karma flower effect active (`reinforcedKarma`) |
| `food_flags`, `item_flags` | u1[4] | bit 1 = present; bit 2 = `AutomaticPickUp` (food) or a `Weapon` (item) |
| `food_path`, `shelter_path` | i2 | tiles along the Slugcat movement graph to the nearest edible / shelter (-1 = none or unknown); for the reward's shaping, not a policy input |
| `sleep_path` | i2 | in a shelter: tiles to the nearest sleep spot (0 on one); -1 outside a shelter |
| `still_ticks_left` | i2 | ticks still to stay without input for the door to close, once `ready_for_win` (-1 outside a shelter) |
| `force_ticks_left` | i2 | ticks still to hold down for a forced (malnourished) sleep (-1 when the food does not allow one) |
| `eat_speed`, `sleep_speed` | u1 | the reset's speed-up K (1 = the real game) |
| `paused` | n/a | not a field: paused frames are not ticks |

Full row (437 bytes), in order: `tick i4, clock i4, wall_ms i8, room i4, camera i4, episode i4,
session i4, step_tick u1, room_changed u1, in_shortcut u1, consumed u1, action i1[5], alive u1,
sleeping u1, standing u1, exhausted u1, grabbed_by u1, can_jump u1, stun u1, roll_counter u1,
slide_counter u1, animation i2, animation_frame i2, body_mode i2, grasp0_type i2, grasp1_type i2,
aerobic_level f4, air_in_lungs f4, pos0 f4[2], vel0 f4[2], pos1 f4[2], vel1 f4[2], cycle_timer i4,
cycle_length i4, rain_hit u1, in_shelter u1, food_to_hibernate i2, food_in_stomach i2,
quarter_food i2, karma i2, karma_cap i2, max_food i2, karma_protected u1, input_history i1[50],
cr_type i2[6], cr_ai_kind i2[6], cr_behavior i2[6], cr_flags u1[6], cr_rel_pos f4[12],
cr_rel_vel f4[12], food_type i2[4], food_points i1[4], food_flags u1[4], food_rel_pos f4[8],
item_type i2[4], item_flags u1[4], item_rel_pos f4[8], holding_edible u1, eat_counter u1, speed f4,
has_controller u1, has_ai u1, inject_fired u1, hit_dealt u1, hit_taken u1, hit_victim i2,
food_path i2, food_path_points i1, food_path_bites u1, shelter_path i2, held_points i1,
held_bites u1, eat_speed u1, malnourished u1, food_hibernate_base i2, sleep_path i2,
start_shelter u1, ready_for_win u1, still_ticks_left i2, force_ticks_left i2, sleep_speed u1`.
Field meanings not listed above are in `mod/RainBot/Observation.cs`.

## Events

```json
{"type":"room","t":..,"index":..,"name":"SU_A22","shelter":false}
{"type":"world","region":"SU","first_room_index":..,"rooms":["SU_A01", ...],
 "shelters":["SU_S01", ...], "gates":[...], "connections":{"SU_A01":["SU_A02", ...]}}
{"type":"session_start","t":..,"session":..,"episode":..}   // new RainWorldGame (cycle start/respawn)
{"type":"session_end","t":..,"reason":"death"|"starve"|"sleep"|"quit","session":..,"episode":..}
{"type":"enum","table":"ai_kind","index":3,"name":"LizardAI","behaviors":["Idle", ...]}
{"type":"death","t":..,"clock":..}
{"type":"karma","t":..,"from":..,"to":..}
{"type":"reset_done","t":..,"episode":..,"session":..,"room":"SU_A22","food":..,"karma":..,"timer":..,
 "spawned":[{"type":"GreenLizard","x":..,"y":..}, ...],"tile":[x,y]}
{"type":"reset_failed","reason":"..."}
{"type":"water_rooms","complete":true,"rooms":["SU_A22", ...]}
{"type":"clip_dumped","path":"...","frames":..}
{"type":"memreport","text":"..."}
```

`world` is sent on connect and whenever the region changes, so Python never needs the on-disk
dumps to know room names and connections. `rooms` is indexed by `room - first_room_index`, and
`connections` lists only connected (non -1) neighbours by name.

`session_end.reason`: `sleep` (`RainWorldGame.Win`), `death` (`GoToDeathScreen`,
`GoToRedsGameOver`), `starve` (`GoToStarveScreen`), else `death` if player 0 is dead, else `quit`.
A frozen ending sends `session_end` and `session_start` with `"frozen":true`.

`enum` is sent when the mod first meets an AI class after the handshake (the `ai_kind` table grows
lazily), always before the chunk whose rows first use that index. `behaviors` is that class's
Behavior ExtEnum names (empty if it has none).

`reset_done.tile` is the start tile the reset placed the slugcat on (`[-1,-1]` if none), so a
replay can ask for exactly it with `pos=x,y`.

`water_rooms` lists the rooms with standing water, learned by the mod's background room survey. It
is sent once per connection when the survey is complete, and on the `water_rooms` command.

## Commands

| command | effect |
|---|---|
| `ping` | no-op, keeps the link alive |
| `speed X` | sim speed multiplier (`framesPerSecond = 40 X`); with config `max_ticks_per_frame` > 3 the mod runs extra ticks per rendered frame beyond the game's own catch-up cap of 3 |
| `render on\|off` | skip `RainWorldGame.GrafUpdate` (sim unaffected) |
| `frame on\|off` | frame capture on or off (off keeps sending the last frame) |
| `seed N` | `UnityEngine.Random.InitState(N)` at the next tick boundary |
| `global_seed N` | run-level seed, -1 = off (see "Determinism") |
| `kill` | the slugcat dies |
| `reset k=v ...` | randomized initial state, below |
| `timer_left N` | set the running cycle's rain timer to N ticks left |
| `endings freeze\|game` | frozen endings (see "Timing") or the game's own flow |
| `peaceful on\|off` | while on, non-player creatures in the slugcat's room are removed, except ones a reset spawned |
| `autopilot on\|off` | clear death prompts and advance menu screens automatically |
| `den ROOM` | set the respawn shelter; with `kill`, this changes region (`reset` works within the loaded world) |
| `water_rooms` | send the `water_rooms` event now |
| `clip arm [N]` | record a native-resolution clip ring of N ticks (40..1200, default 400); while armed, one tick per rendered frame at real time |
| `clip off` | stop recording and free the ring |
| `clip dump PATH` | write the ring to PATH (`int32 n, int32 w, int32 h`, n frames of `w*h*3` RGB, row 0 = top, then n `int32` ticks) and send `clip_dumped` |
| `reload` | swap in the core build on disk now; the agent link closes and must reconnect |
| `memreport` | memory use by Unity asset type, as a `memreport` event |
| `pathdebug` | path diagnostics for the current room, as a `memreport` event |

`reset` keys, all optional:

| key | values | default |
|---|---|---|
| `room` | room name, or `shelter` (random shelter), or `any` (random non-gate room) | stay |
| `pos` | `den` (where the slugcat comes out of the entrance pipe it was routed through; in the current room: stay), `random` (random standable tile: AI-map `Floor`, not narrow, no water, air with headroom; unsafe tiles are redrawn), `x,y` tiles | `random` |
| `food` | pips in stomach (0..maxFood) | unchanged |
| `karma` | 0..karmaCap | unchanged |
| `malnourished` | 0, 1 | 0 |
| `timer_left` | ticks of cycle left before rain | unchanged |
| `creatures` | `keep`, `clear` | `keep` |
| `spawn` | `Type:n,Type:n` creature templates (template name or `CreatureTemplate.Type` value) placed at random standable tiles >= 10 tiles (Manhattan) from the slugcat when possible | none |
| `fruit` | n `DangleFruit` at random standable tiles in the room | 0 |
| `fruit_near` | n `DangleFruit` within 6 tiles (Manhattan) of the slugcat | 0 |
| `hold` | `fruit` (a `DangleFruit` in the free hand after settling), `none` | `none` |
| `bites` | 1..3, bites left on the held fruit | the game's |
| `eat_counter` | 1..40, the slugcat's starting `eatCounter` | the game's |
| `eat_speed` | 1..40: each tick the game counts down `eatCounter`, K-1 more are removed | 1 |
| `sleep_speed` | 1..40: each tick the game advances the shelter's still / hold-down counters, K-1 more are added | 1 |
| `climb` | `0`: the start must reach a shelter without a pole climb, after any wrong turn | climbing allowed |
| `shelter_within` | N >= 1: the start is at most N tiles of path from the pipe into a shelter (may be on a pole, holding it) | any |
| `items` | `spear:n,rock:n` | none |
| `seed` | int, seeds the placement RNG and `UnityEngine.Random` | random |

After placing the slugcat the reset settles it with neutral input for 2 x 40 ticks (placed,
settled, placed again on the same tile, settled again), zeroes the game's update-phase counters
(`updateShortCut`, `updateAbstractRoom`), then sets the rain timer and seeds `UnityEngine.Random`,
so the same seed gives the same start state whichever room the slugcat came from. Objects that
earlier resets spawned, and any `DangleFruit` without a level placement, are removed first,
wherever they are.

The reset is applied at a tick boundary: the mod clears the chunk buffer, increments `episode`,
emits `reset_done` (or `reset_failed`), and the next exchange carries 8 fresh rows.

Mechanics: moving to another room goes through the game's own
`ShortcutHandler.CreatureEnterFromAbstractRoom` via a random `Exit` node of the target room
(realize, wait until ready, pipe travel, camera move), so the slugcat drops what it holds and
anything holding it lets go. The reset fails (`reset_failed`) if the slugcat is dead, the room is
unknown or has no exit node, loading takes > 30 s or transit > 20 s, no start meets `climb=0` or
`shelter_within`, or any exception occurs; the game stays playable. `creatures=clear` removes the
room's creatures, including those in its dens. `seed` seeds both the placement RNG and
`UnityEngine.Random`.
