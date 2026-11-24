"""Curriculum, on story-mode Outskirts through the mod's ``reset`` command.

The stages run in story mode: the mod teleports into an
Outskirts room, clears or spawns creatures, places fruit and items, and sets food, karma and the rain
timer. Karma is real, so no karma emulation is needed, and stage F is the same code path with
nothing overridden.

Stages (advance when the metric holds for 3 windows in a row):
  L  the ladder (replaces stages A-C): whole cycles with food, eating, shelter and rain together,
     every skill harder at each rung (LADDER). Metric: fed hibernation rate > 80%.
  D  as L, predators added gradually. Metric: karma does not fall.
  E  spears and rocks present. Metric: karma does not fall with predators present.
  F  the full region, story mode, nothing overridden. Final stage.

The reward is always the full drive; the resets decide what a rung asks for. Every stage randomises
the initial state. About 20% of resets are drawn from earlier stages or earlier rungs.
The guide gives no thresholds; they are all listed in this module.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np

STAGES = ("L", "D", "E", "F")

#: The ladder: one stage instead of A-C, so every episode is the whole task (find food, eat it, reach
#: a shelter, sleep before the rain) and every skill gets harder at the same time. Separate stages
#: each drop skills an earlier one taught, and the agent forgets them. Every rung is a whole cycle.
#: Per rung:
#:   need     pips below the hibernation requirement at the start (None: anywhere from 1 to all of it)
#:   K        eat_speed and sleep_speed (20 -> 1, 1 = the real game). Holding still to eat and holding
#:            still (or down) for the shelter door are the same wall for a sampling policy, so both
#:            are walked back together.
#:   hold     a fruit already in hand (3 bites)
#:   near     extra fruit within 6 tiles of the start (more fruit is placed across the room)
#:   rooms    rooms from the nearest shelter (lo, hi). Only 4 rooms are one room from a shelter, and 3 of
#:            them need a climb or a ledge pull-up; 1-2 rooms gives 15, so a rung also has rooms
#:            without its new move.
#:   water    rooms with standing water allowed (the mod's survey reports them)
#:   climb    starts whose route to a shelter needs a pole climb allowed. Without it the mod's reset
#:            (`climb=0`) only places the slugcat where the game's AI map reaches a shelter pipe
#:            without a pole (SU_A37's pipe is in the ceiling, SU_B01's in a pit). A room with no
#:            such tile is remembered in climb_rooms and not drawn again for such rungs.
#: The rain timer is TIMER_CAP at every rung. A per-start timer (the time a human needs for the
#: start's own task) is ~180 ticks next to a shelter: no room for a single mistake, so most members
#: die in the rain and a group has nothing to compare. Real cycles are 20-30k ticks; difficulty
#: comes from the other dimensions.
LADDER = (
    dict(need=1, K=20, hold=True, near=2, rooms=(1, 1), water=False, climb=False),
    dict(need=1, K=10, hold=False, near=2, rooms=(1, 1), water=False, climb=False),
    # One new difficulty per rung (adding four at once stalls): more food at a slower eat, then
    # climbing, then water rooms, then a shelter 2 rooms away.
    dict(need=2, K=6, hold=False, near=2, rooms=(1, 1), water=False, climb=False),
    # Climbing as a reverse curriculum: the hard part is the last 8-10 tiles up to the shelter, so
    # these rungs start in the climbing rooms only (climb_only), first at most `within` tiles of path
    # from the shelter pipe, then farther, then anywhere.
    dict(need=2, K=6, hold=False, near=2, rooms=(1, 1), water=False, climb=True, climb_only=True, within=6),
    dict(need=2, K=6, hold=False, near=2, rooms=(1, 1), water=False, climb=True, climb_only=True, within=12),
    dict(need=2, K=6, hold=False, near=2, rooms=(1, 1), water=False, climb=True, climb_only=True),
    dict(need=2, K=6, hold=False, near=2, rooms=(1, 1), water=True, climb=True),
    dict(need=2, K=6, hold=False, near=2, rooms=(1, 2), water=True, climb=True),
    dict(need=2, K=4, hold=False, near=1, rooms=(1, 2), water=True, climb=True),
    dict(need=3, K=3, hold=False, near=1, rooms=(1, 2), water=True, climb=True),
    dict(need=3, K=2, hold=False, near=1, rooms=(1, 3), water=True, climb=True),
    dict(need=None, K=1, hold=False, near=0, rooms=(1, 3), water=True, climb=True),
)
#: The rain timer of every rung (ticks). The rollout keeps every token's frame, so the episode length
#: (timer + the rain's time to kill) must stay bounded.
TIMER_CAP = 9000
TICKS_PER_MIN = 40 * 60

#: Advancement uses the game's own requirements, not hand-picked numbers:
#:   L (every rung)  fed hibernation rate > 80%
#:   D, E            karma does not fall over the window: hibernations >= deaths
#:   F               final stage, nothing to pass
#: Metrics decide stage advancement only, never the reward or gradients.
HIBERNATION_RATE_C = 0.8
#: Longest measured time from the rain timer running out to the slugcat's death (SU_A53: 2184 ticks),
#: rounded up: whole-cycle episodes must contain the death, not cut it off.
RAIN_TO_DEATH_TICKS = 2400
#: Stage D: predators per reset, raised each time survival holds.
D_PREDATORS = ("GreenLizard:1", "GreenLizard:1,PinkLizard:1", "GreenLizard:2,PinkLizard:1")


@dataclass
class EpisodeStats:
    ticks: int = 0
    pips: float = 0.0
    sleeps: int = 0
    deaths: int = 0
    rooms: set = field(default_factory=set)
    idle_tokens: int = 0
    tokens: int = 0
    displacement: float = 0.0
    stage: str = "L"
    rung: int = -1           # ladder rung this episode was reset into (-1: none)
    fed_sleeps: int = 0      # sleeps with the hibernation requirement in the stomach (not malnourished)
    need: float = -1.0       # pips the episode's cycle started short of the requirement (-1: unknown)
    # Game requirements, read from the rows (Survivor defaults until a row is seen).
    food_to_hibernate: int = 4
    cycle_length: int = 0    # ticks; 0 = unknown
    drowned: int = 0
    death_causes: dict = field(default_factory=dict)   # rain / predator / drowned / other
    sheltered_before_rain: int = 0                     # cycles slept with rain not yet hit


class WorldGraph:
    """Room graph from the mod's ``world`` event: names, shelters, connections."""

    def __init__(self, world: dict):
        # Only the region's own rooms: the mod's world table also lists rooms of other regions
        # (OE_* rooms in the SU world) and gate rooms, and a reset cannot load those from here.
        reg = world.get("region")
        own = lambda r: (not reg or r.startswith(reg + "_")) and not r.startswith("GATE")
        self.rooms = [r for r in world.get("rooms", []) if own(r)]
        self.shelters = {r for r in world.get("shelters", []) if own(r)}
        self.adj = {k: list(v) for k, v in (world.get("connections") or {}).items()}

    def dist_to_shelter(self) -> dict[str, int]:
        dist = {s: 0 for s in self.shelters}
        q = collections.deque(self.shelters)
        while q:
            r = q.popleft()
            for n in self.adj.get(r, []):
                if n not in dist:
                    dist[n] = dist[r] + 1
                    q.append(n)
        return dist

    def rooms_within(self, lo: int, hi: int) -> list[str]:
        return [r for r, d in self.dist_to_shelter().items() if lo <= d <= hi]


class Curriculum:
    def __init__(self, stage: str = "L", seed: int = 0, windows_to_advance: int = 3,
                 replay_frac: float = 0.2, water_rooms: set | None = None):
        self.stage_idx = STAGES.index(stage)
        self.rng = np.random.default_rng(seed)
        self.windows_to_advance = windows_to_advance
        self.replay_frac = replay_frac
        self.passes = 0
        self.level = 0                 # ladder rung (L) or predator level (D)
        self.graph: WorldGraph | None = None
        self.water_rooms = water_rooms or set()
        self.climb_rooms: set = set()      # rooms with no climb-free start (the mod's reset said so)
        self.history: list[dict] = []

    last_rung: int = -1

    def l_rung(self, replay: bool = False) -> int:
        """The ladder rung for a new episode: the current one, or (replay_frac of the time, and
        whenever the ladder itself is replayed from a later stage) an earlier one."""
        if replay or self.stage != "L":
            return int(self.rng.integers(len(LADDER)))
        cur = min(self.level, len(LADDER) - 1)
        if cur > 0 and self.rng.random() < self.replay_frac:
            return int(self.rng.integers(cur))
        return cur

    @staticmethod
    def group_tokens(stage: str, spec: str, base: int) -> int:
        """GRPO episode length for a reset, in tokens.

        The episode is the whole rain cycle the reset sets (timer_left ticks / 8 per token) plus the
        rain's time to kill, rounded up to a multiple of ``base`` so the update can train it in
        base-token chunks. The slugcat has every chance to eat, travel, hibernate, or be caught by
        the rain. A spec with no timer (F) gets 9 chunks."""
        timer = None
        for tok in spec.split():
            if tok.startswith("timer_left="):
                timer = int(tok.split("=", 1)[1])
        # The rain kills 512-2184 ticks after the timer runs out, depending on the room. The episode
        # itself ends earlier, when the cycle ends.
        tokens = -(-(timer + RAIN_TO_DEATH_TICKS) // 8) if timer else 9 * base
        return -(-tokens // base) * base

    def episode_tokens(self, ep: EpisodeStats) -> int | None:
        return None

    @property
    def stage(self) -> str:
        return STAGES[self.stage_idx]

    def set_world(self, world: dict) -> None:
        if world.get("rooms"):
            self.graph = WorldGraph(world)

    # ---- reset specs ----------------------------------------------------------------------
    def sample_stage(self) -> str:
        if self.stage_idx > 0 and self.rng.random() < self.replay_frac:
            return STAGES[int(self.rng.integers(self.stage_idx))]
        return self.stage

    def reset_spec(self, stage: str | None = None, world: dict | None = None) -> tuple[str, str]:
        """(stage, reset command arguments) for a new episode.

        ``world`` is the requesting instance's own ``world`` event: instances can be in different
        regions, and a room name from another instance's world does not load.
        """
        st = stage or self.sample_stage()
        g = WorldGraph(world) if world and world.get("rooms") else self.graph
        r = self.rng
        seed = int(r.integers(1 << 30))
        rooms = [x for x in (g.rooms if g else []) if x not in (g.shelters if g else set())]
        pick = lambda xs: xs[int(r.integers(len(xs)))] if xs else "any"
        food = int(r.integers(0, 5))
        karma = int(r.integers(0, 5))
        if st == "L":
            rung = self.l_rung(replay=stage is None and st != self.stage)
            self.last_rung = rung
            lad = LADDER[rung]
            lo, hi = lad["rooms"]
            near = (g.rooms_within(lo, hi) if g else []) or rooms
            if not lad["water"]:
                near = [x for x in near if x not in self.water_rooms] or near
            if not lad["climb"]:
                near = [x for x in near if x not in self.climb_rooms] or near
            if lad.get("climb_only"):
                near = [x for x in near if x in self.climb_rooms] or near
            req = 4                                   # Survivor's requirement (food_to_hibernate)
            need = lad["need"] if lad["need"] is not None else int(r.integers(1, req + 1))
            # The stomach starts anywhere from the rung's hungriest case up to exactly fed: a fed
            # start asks whether extras are worth the risk. Extras are never given, only earned.
            # The pass is judged on the hungry starts only (evaluate).
            args = dict(room=pick(near), pos="random", food=req - need + int(r.integers(0, need + 1)),
                        karma=karma, timer_left=TIMER_CAP,
                        creatures="clear", peaceful="on", fruit=need + int(r.integers(0, 3)),
                        fruit_near=lad["near"], eat_speed=lad["K"], sleep_speed=lad["K"])
            if lad["hold"]:
                args.update(hold="fruit", bites=3, eat_counter=40)
            if not lad["climb"]:
                args["climb"] = 0
            if lad.get("within"):
                args["shelter_within"] = lad["within"]
        elif st == "D":
            near = g.rooms_within(1, 3) if g else []
            lvl = min(self.level, len(D_PREDATORS) - 1) if self.stage == "D" else len(D_PREDATORS) - 1
            args = dict(room=pick(near or rooms), pos="random", food=food, karma=karma,
                        timer_left=int(r.integers(3600, 9000)), creatures="clear", peaceful="off",
                        spawn=D_PREDATORS[lvl], fruit=int(r.integers(0, 3)))
        elif st == "E":
            args = dict(room=pick(rooms), pos="random", food=food, karma=karma,
                        timer_left=int(r.integers(3600, 9000)), creatures="keep", peaceful="off",
                        items=f"spear:{int(r.integers(1, 3))},rock:{int(r.integers(1, 4))}")
        else:  # F: the region as it is
            args = dict(room=pick(list(g.shelters)) if g and g.shelters else "shelter", pos="den",
                        creatures="keep", peaceful="off")
        args["seed"] = seed
        return st, " ".join(f"{k}={v}" for k, v in args.items())

    # ---- metrics and advancement ------------------------------------------------------------
    @staticmethod
    def metrics(eps: list[EpisodeStats], n_rooms_hint: int = 4) -> dict[str, float]:
        if not eps:
            return {}
        ticks = sum(e.ticks for e in eps) or 1
        cycles = sum(e.sleeps + e.deaths for e in eps)
        L = [e.cycle_length for e in eps if e.cycle_length > 0]
        cyc = float(np.median(L)) if L else 0.0
        hib = float(np.median([e.food_to_hibernate for e in eps]))
        pips_per_cycle = sum(e.pips for e in eps) / ticks * cyc if cyc else 0.0
        causes: dict = {}
        for e in eps:
            for k, v in e.death_causes.items():
                causes[k] = causes.get(k, 0) + v
        return {
            # game-grounded
            "pips_per_cycle": pips_per_cycle,
            "food_to_hibernate": hib,
            # Per cycle, not per tick: pips eaten / pips the cycle started short of, over the
            # window's cycles (each GRPO episode is one). A per-tick rate scaled to the cycle would
            # count the walk home and the wait at the door against the slugcat. >= 1: eats enough.
            "food_need_met": (sum(e.pips for e in eps if e.need >= 0)
                              / max(sum(e.need for e in eps if e.need >= 0), 1e-9)
                              if any(e.need > 0 for e in eps) else
                              (pips_per_cycle / hib if hib else 0.0)),
            "drowned": sum(e.drowned for e in eps),
            "deaths_by_cause": causes,
            "sheltered_before_rain": sum(e.sheltered_before_rain for e in eps),
            "karma_net": sum(e.sleeps - e.deaths for e in eps),
            # descriptive
            "episodes": len(eps),
            "pips_per_min": sum(e.pips for e in eps) / (ticks / TICKS_PER_MIN),
            "traversed_frac": float(np.mean([min(len(e.rooms) / n_rooms_hint, 1.0) for e in eps])),
            "idle_frac": sum(e.idle_tokens for e in eps) / max(sum(e.tokens for e in eps), 1),
            "hibernation_rate": sum(e.sleeps for e in eps) / max(cycles, 1),
            # Over episodes, not finished cycles: every GRPO episode is one cycle, and one cut at its
            # length without a sleep (for example fed, in the shelter, never still for the door) is
            # a cycle without a hibernation, not a cycle that does not count.
            "fed_hibernation_rate": sum(e.fed_sleeps for e in eps) / max(cycles, len(eps), 1),
            "survival_rate": sum(e.sleeps for e in eps) / max(cycles, 1),
            "cycles": cycles,
            "rooms_used": float(np.mean([len(e.rooms) for e in eps])),
            "displacement": float(np.mean([e.displacement for e in eps])),
            "tiles_per_token": sum(e.displacement for e in eps) / max(sum(e.tokens for e in eps), 1),
            "ate_frac": float(np.mean([e.pips > 0 for e in eps])),
        }

    def passes_threshold(self, m: dict) -> bool:
        """The stage's game requirement (see the table above HIBERNATION_RATE_C)."""
        if not m:
            return False
        st = self.stage
        if st == "L":
            # Fed sleeps only: a malnourished sleep is the shortcut of going home without eating. food_need_met is only logged: per cycle it is a stricter
            # copy of the fed-sleep rate.
            return m["cycles"] > 0 and m["fed_hibernation_rate"] > HIBERNATION_RATE_C
        if st in ("D", "E"):
            return m["cycles"] > 0 and m["karma_net"] >= 0
        return False

    def evaluate(self, finished: list[EpisodeStats], min_episodes: int = 4) -> dict:
        """Call once per evaluation window with the episodes of the *current* stage it finished."""
        eps = [e for e in finished if e.stage == self.stage]
        if self.stage == "L":
            # Judge the ladder only on episodes of the current rung that started hungry: a fed
            # start's fed sleep says nothing about foraging.
            eps = [e for e in eps if e.rung == min(self.level, len(LADDER) - 1)]
            eps = [e for e in eps if e.need != 0]
        m = self.metrics(eps)
        if len(eps) < min_episodes:
            return {"stage": self.stage, "level": self.level, "skipped": True, **m}
        ok = self.passes_threshold(m)
        self.passes = self.passes + 1 if ok else 0
        event = None
        if self.passes >= self.windows_to_advance:
            self.passes = 0
            if self.stage == "L" and self.level < len(LADDER) - 1:
                self.level += 1
                event = f"ladder -> rung {self.level} {LADDER[self.level]}"
            elif self.stage == "D" and self.level < len(D_PREDATORS) - 1:
                self.level += 1
                event = f"D predators -> {D_PREDATORS[self.level]}"
            elif self.stage_idx < len(STAGES) - 1:
                self.stage_idx += 1
                self.level = 0
                event = f"advance -> {self.stage}"
        rec = {"stage": self.stage, "level": self.level, "pass": ok, "passes": self.passes,
               "event": event, **m}
        self.history.append(rec)
        return rec

    def state_dict(self) -> dict:
        # The water rooms are saved because the mod's survey reports them only some time after each
        # game launch, and a resumed run must not pick its first starts without them. The start
        # generator is saved so that a restart does not replay the same starts from the run seed.
        return {"stage_idx": self.stage_idx, "passes": self.passes, "level": self.level,
                "water_rooms": sorted(self.water_rooms), "climb_rooms": sorted(self.climb_rooms),
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, d: dict) -> None:
        self.stage_idx, self.passes, self.level = d["stage_idx"], d["passes"], d["level"]
        if d.get("water_rooms"):
            self.water_rooms = set(d["water_rooms"])
        self.climb_rooms = set(d.get("climb_rooms") or [])
        if d.get("rng"):
            self.rng.bit_generator.state = d["rng"]
        else:                                   # older checkpoints: at least not the run seed's sequence
            self.rng = np.random.default_rng()
