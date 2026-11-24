"""Homeostatic reward: drive reduction over the needs the game tracks.

    r_t  = D(h_t) - D(h_{t+1}) (+ the extras, paid at a sleep)
    D(h) = sum_i d_i^2

Each necessity deficit d_i lies in [0, 1], measured over its own range:

| variable | game meaning                                      | deficit is the distance to          |
|----------|---------------------------------------------------|-------------------------------------|
| food     | pips in the stomach (quarter pips included)       | tonight's base requirement          |
| karma    | +1 per hibernation, -1 per death unless protected | the game's maximum (9)              |
| breath   | air left in the lungs                             | full lungs (1.0)                    |

With D a sum of squares, each need's marginal weight is its own size: the biggest deficit counts
most, every need still counts, and there is no weight to tune. Pips above the requirement (extras)
are not a need. They are paid once, when the game books a sleep, as tomorrow's smaller food
deficit (``extras_paid``).

Nothing else goes into the reward. Rain and predators need no terms: both kill, and death costs
karma. Idling loses because the food deficit never closes.

A step here is one token (8 ticks): ``h_t`` is read from the token's last row, as the mod exports
it. Nothing is inferred.
"""
from __future__ import annotations

import numpy as np

#: The game's maximum karma (index; 10 symbols). The karma setpoint is this, not the current cap.
#: With the cap as setpoint, meeting an echo (which raises the cap, SaveState.GhostEncounter) would
#: increase the deficit. With the maximum, the deficit only closes by raising the cap, as the
#: game's ending requires.
KARMA_MAX = 9.0


def deficits(rows, food_add=0.0, karma_add=0.0) -> np.ndarray:
    """Necessity deficits [..., 3] in [0, 1]: food, karma, air.

    Food is measured to tonight's base requirement (pips above it are extras, paid by
    ``extras_paid``), karma to the game's maximum, air to full lungs. Only food is clipped, at the
    requirement. ``food_add`` / ``karma_add`` add the forecast's pips and karma
    (``augmented_deficits``)."""
    f = lambda k: np.asarray(rows[k], dtype=np.float64)
    food = f("food_in_stomach") + f("quarter_food") / 4.0 + food_add
    req = np.maximum(hunger_setpoint(rows), 1.0)
    d_food = (req - np.minimum(food, req)) / req
    cap, karma = f("karma_cap"), f("karma")
    # Karma exists only in story sessions; outside one the mod reports -1.
    has_karma = (cap >= 0) & (karma >= 0)
    d_karma = np.where(has_karma, (KARMA_MAX - np.minimum(karma + karma_add, KARMA_MAX)) / KARMA_MAX, 0.0)
    d_air = 1.0 - np.minimum(f("air_in_lungs"), 1.0)
    return np.stack([d_food, d_karma, d_air], axis=-1)


def extras_paid(prev_rows, rows, slept) -> np.ndarray:
    """The extras, paid when the game books a sleep.

    Pips above tonight's requirement are tomorrow's starting food, so tomorrow's food deficit is
    (req - e) / req instead of 1: worth 1 - ((req - e) / req)^2 in the same squared units. The
    value is certain at the sleep, so it is paid then: nothing before it, nothing at a death.
    ``slept`` is the mod's own ending reason for this transition (session_end reason "sleep"). It
    is never inferred from karma, because at the karma cap a sleep adds no karma."""
    f1 = lambda k: np.asarray(rows[k], dtype=np.float64)
    slept = np.asarray(slept, bool)
    food = f1("food_in_stomach") + f1("quarter_food") / 4.0          # as booked (shelter food eaten)
    req = np.maximum(f1("food_to_hibernate"), 1.0)                   # tonight's (max food if malnourished)
    e = np.clip(np.minimum(food, f1("max_food")) - req, 0.0, None)
    return np.where(slept, 1.0 - ((req - np.minimum(e, req)) / req) ** 2, 0.0)


def hunger_setpoint(rows) -> np.ndarray:
    """The food setpoint: the species' normal hibernation threshold (food_hibernate_base).

    It is not the malnourished requirement (max food) the game sets after a malnourished sleep.
    Hunger is the distance to a normal night. The raised requirement changes tonight's outcome, so
    it lives in the karma forecast. This keeps sleeping fed > sleeping malnourished > dying without
    a longer horizon."""
    names = rows.dtype.names or ()
    key = "food_hibernate_base" if "food_hibernate_base" in names else "food_to_hibernate"
    return np.asarray(rows[key], dtype=np.float64)


def drive(d: np.ndarray) -> np.ndarray:
    """D = sum d_i^2: each need's marginal weight is its own size."""
    return np.sum(np.square(d), axis=-1)


def reward(prev_rows, rows, slept=False) -> np.ndarray:
    """r = D(h_prev) - D(h_now) (+ the extras when ``slept``: the mod booked a sleep here)."""
    return drive(deficits(prev_rows)) - drive(deficits(rows)) + extras_paid(prev_rows, rows, slept)


class DriveTracker:
    """Per-environment drive bookkeeping across tokens, including across session boundaries."""

    def __init__(self, n_envs: int):
        self.prev = np.full(n_envs, np.nan)
        self.prev_d = np.zeros((n_envs, 3))

    def step(self, i: int, last_row) -> tuple[float, np.ndarray]:
        d = deficits(last_row)
        D = float(drive(d))
        r = 0.0 if np.isnan(self.prev[i]) else float(self.prev[i] - D)
        self.prev[i] = D
        self.prev_d[i] = d
        return r, d

    def reset(self, i: int) -> None:
        """New episode: the first token has no predecessor to difference against."""
        self.prev[i] = np.nan


# ---- potential-based shaping (Ng, Harada & Russell 1999) -------------------------------------------
#
#   r'_t = r_t + gamma * Phi(s_{t+1}) - Phi(s_t),     Phi(s) = D(h(s)) - D(h~(s))
#
# h~ is h plus a forecast: what the nearest opportunity is worth if the slugcat goes straight to it,
# discounted by the return's own gamma over the ticks that takes. Phi is the drive reduction of that
# forecast, the value of the go-straight plan. Each potential forecasts a drive variable, in its own
# units, from quantities the game tracks. No free constants:
#
#   food    edible in hand: its points x gamma^(ticks to finish eating); otherwise the nearest
#           edible by path: points x gamma^(path / speed + ticks to eat it). Eating ticks follow the
#           game's state machine (Player.cs: eatCounter idles at 40, -1 per qualifying tick, a bite
#           below 1, then 15 per further bite), divided by the curriculum's eat_speed K.
#   karma   tonight's outcome under the game's rules: E = P(survive) * (+1 up to the cap) - (1 - P) * 1
#           (the death; nothing at karma 0 or under a karma flower), discounted to the rain. Survive =
#           in a shelter with >= 1 pip (malnourished sleep), or >= the requirement if malnourished
#           (else it starves). Outside, P = Poisson(human hungry eating rate x time left to forage
#           before leaving for the shelter) of collecting the pips still needed, 0 if no shelter is
#           reachable before the rain. Distance to the shelter enters through the foraging time.
#   breath  already continuous (air changes every tick): no forecast.
#
# Paths come from the mod (PathInfo.cs): tiles along the game's own slugcat movement graph and
# prebaked room-to-room lengths. Speed is SLUGCAT_TILES_PER_TICK, measured on human recordings.
# Rows without the path fields (older mods, the fake game) get no forecast.

#: Mean speed while a direction is held, human recordings (10 sessions, 47k ticks): mean 0.199,
#: median 0.186, p25 0.100, p75 0.266 tiles/tick. The mean, because time = length / speed.
SLUGCAT_TILES_PER_TICK = 0.199
#: The game's sleep rules (Player.Update): a sleep spot is more than 6 tiles from the shelter's
#: entrance; the door closes after 41 ticks without input (fed, readyForWin) or 261 ticks holding down
#: (forceSleepCounter > 260, a malnourished sleep).
SLEEP_SPOT_TILES, SLEEP_STILL_TICKS, FORCE_SLEEP_TICKS = 7, 41, 261
#: Pips per tick while hungry (outside shelters, alive), human recordings: 16 pips over 189,873 ticks
#: (0.20 per minute of game time). The rate of a Poisson process for the chance that the slugcat
#: collects the n pips it still needs before it must go to a shelter.
HUNGRY_PIPS_PER_TICK = 16.0 / 189873.0
EAT_IDLE = 40           # Player.cs: eatCounter's resting value
EAT_NEXT_BITE = 15      # Player.cs: eatCounter after each bite
TICKS_PER_TOKEN = 8
GAMMA_TOKEN = 0.999     # PPOConfig.gamma; the shaping must use the return's discount


#: Trip-scale travel speed (mean over 200-tick windows with a direction held), human recordings,
#: quantiles 5%..95% over 151 windows. P(arrive in time) = P(speed >= path / time left).
SPEED_QUANTILES = np.array([0.1096, 0.1206, 0.1301, 0.1448, 0.1588, 0.171, 0.1746, 0.181, 0.1874, 0.1944,
                            0.2007, 0.205, 0.2133, 0.2196, 0.2257, 0.2335, 0.2435, 0.248, 0.2705])
SPEED_PROBS = np.linspace(0.05, 0.95, 19)


def _p_speed_at_least(v: np.ndarray) -> np.ndarray:
    """P(trip speed >= v) from the measured quantiles, linear between them, 1 at speed 0 and 0 above
    1.1 x the fastest quantile."""
    xs = np.concatenate([[0.0], SPEED_QUANTILES, [SPEED_QUANTILES[-1] * 1.1]])
    cdf = np.concatenate([[0.0], SPEED_PROBS, [1.0]])
    v = np.asarray(v, dtype=np.float64)
    return 1.0 - np.interp(np.where(np.isfinite(v), v, 1e9), xs, cdf)


def _poisson_at_least(n: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """P(N >= n) for N ~ Poisson(mu), elementwise; n is a small non-negative integer array."""
    n = np.asarray(n, dtype=np.int64)
    mu = np.asarray(mu, dtype=np.float64)
    term = np.exp(-mu)
    below = np.zeros(np.broadcast(n, mu).shape)
    for i in range(int(n.max(initial=0))):
        below = below + np.where(i < n, term, 0.0)
        term = term * mu / (i + 1)
    return np.clip(1.0 - below, 0.0, 1.0)


# Measured food forecast (train/forage_fit.py): the trainer sets it between iterations, and it stays
# fixed within one. None = the time-discounted certainty forecast in _forecast (tests, the fake game,
# no data yet).
FORAGE_TABLE = None


def set_forage_table(table) -> None:
    global FORAGE_TABLE
    FORAGE_TABLE = table


def _food_forecast_measured(f, tab, gamma):
    from .train.forage_fit import R_MAX
    k = np.maximum(f("eat_speed"), 1.0)
    held_pts, held_bites = f("held_points"), f("held_bites")
    r = (np.minimum(f("eat_counter"), EAT_IDLE) + EAT_NEXT_BITE * np.maximum(held_bites - 1, 0)) / k
    r = np.clip(np.ceil(r), 0, R_MAX).astype(int)
    pips_held = np.where((held_pts > 0) & (held_bites > 0), held_pts * np.asarray(tab.held)[r], 0.0)
    # Food at a distance is discounted by human travel time along the path. What is reached is a
    # fruit within reach, worth what one is measured to be worth to this agent (picking it up and
    # eating it earn the rest of its value).
    g = gamma ** (1.0 / TICKS_PER_TOKEN)
    path, pts = f("food_path"), f("food_path_points")
    pips_far = np.where((path >= 0) & (pts > 0),
                        pts * tab.adj * g ** (np.maximum(path, 0.0) / SLUGCAT_TILES_PER_TICK), 0.0)
    return np.where(pips_held > 0, pips_held, pips_far)


def _forecast(rows, gamma: float):
    """(forecast pips, forecast karma levels), or zeros where the fields are absent."""
    names = rows.dtype.names or ()
    f = lambda k: np.asarray(rows[k], dtype=np.float64)
    shape = np.shape(rows)
    if "food_path" not in names:
        return np.zeros(shape), np.zeros(shape)
    g = gamma ** (1.0 / TICKS_PER_TOKEN)
    disc = lambda t: g ** np.clip(t, 0.0, 1e6)
    k = np.maximum(f("eat_speed"), 1.0)
    held_pts, held_bites = f("held_points"), f("held_bites")
    t_held = (np.minimum(f("eat_counter"), EAT_IDLE) + EAT_NEXT_BITE * np.maximum(held_bites - 1, 0)) / k
    pips_held = np.where((held_pts > 0) & (held_bites > 0), held_pts * g ** t_held, 0.0)
    path, pts, bites = f("food_path"), f("food_path_points"), f("food_path_bites")
    # Ticks to reach and eat the nearest edible.
    t_food = path / SLUGCAT_TILES_PER_TICK + (EAT_IDLE + EAT_NEXT_BITE * np.maximum(bites - 1, 0)) / k
    pips_far = np.where((path >= 0) & (pts > 0), pts * g ** np.maximum(t_food, 0.0), 0.0)
    pips = np.where(pips_held > 0, pips_held, pips_far)
    if FORAGE_TABLE is not None:
        pips = _food_forecast_measured(f, FORAGE_TABLE, gamma)
    karma, cap = f("karma"), f("karma_cap")
    food = f("food_in_stomach") + f("quarter_food") / 4.0
    req = f("food_to_hibernate")                       # tonight's requirement (max food if malnourished)
    maln = f("malnourished") > 0 if "malnourished" in names else np.zeros(shape, bool)
    spath = f("shelter_path")
    shelter = f("in_shelter") > 0
    has_timer = f("cycle_length") > 0
    t_left = np.where(has_timer, f("cycle_length") - f("cycle_timer"), 1e9)
    # Survival tonight under the game's rules (ShelterDoor): a malnourished slugcat must reach its
    # requirement or it starves; otherwise one pip is enough (a malnourished sleep, +1 karma like a
    # fed one; its cost, tomorrow's raised requirement, shows up in tomorrow's forecast).
    up = np.minimum(1.0, np.maximum(cap - karma, 0.0))
    protected = f("karma_protected") > 0 if "karma_protected" in names else np.zeros(shape, bool)
    down = np.where(protected | (karma <= 0), 0.0, 1.0)
    disc_rain = disc(t_left)
    max_food = f("max_food")
    cycle = np.where(has_timer, f("cycle_length"), 0.0)
    # A malnourished sleep: +1 tonight, but tomorrow's requirement is max food, and below it the
    # slugcat starves one cycle later, with P(collect max food in a cycle) to escape it.
    p_full_next = _poisson_at_least(np.ceil(max_food), HUNGRY_PIPS_PER_TICK * cycle)
    v_maln = up - disc(cycle) * (1.0 - p_full_next) * down
    # Pips still needed: for a fed sleep (the requirement) and to survive at all (one pip, or the
    # requirement when already malnourished: below it there is no malnourished sleep, only starving).
    need_fed = np.maximum(req - food, 0.0)
    need_any = np.where(maln, need_fed, np.maximum(1.0 - food, 0.0))
    # Outside (or leaving the shelter to forage): reach a shelter in time (P from the spread of
    # human trip speeds) having collected pips in the time left for foraging (Poisson, human hungry
    # eating rate). The night comes at the rain.
    t_shelter = np.where(shelter, 0.0, np.where(spath >= 0, spath / SLUGCAT_TILES_PER_TICK, np.inf))
    v_req = np.where(shelter, 0.0, np.where(spath >= 0, spath / np.maximum(t_left, 1.0), np.inf))
    p_reach = np.where(t_shelter < t_left, _p_speed_at_least(v_req), 0.0)
    mu = HUNGRY_PIPS_PER_TICK * np.maximum(t_left - np.where(np.isfinite(t_shelter), t_shelter, 0.0), 0.0)
    p_fed = _poisson_at_least(np.ceil(need_fed), mu)
    p_any = np.maximum(_poisson_at_least(np.ceil(need_any), mu), p_fed)
    night = p_fed * up + (p_any - p_fed) * v_maln - (1.0 - p_any) * down
    e_go = disc_rain * (p_reach * night - (1.0 - p_reach) * down)
    # Or go home now and sleep there. The game closes a shelter's door as soon as the slugcat stands
    # still on a sleep spot with its requirement eaten, or after 260 ticks holding down with at least
    # a pip (a malnourished sleep); it need not wait for the rain. Without this option a fed slugcat
    # far from home would be worth as much as one at the door, and nothing would pull it home.
    # Time: the path to the shelter and on to a sleep spot at the mean human speed, then the still or
    # hold-down ticks; the chance of making it before the rain from the human trip-speed spread.
    if not np.all(shelter):
        ks = np.maximum(f("sleep_speed"), 1.0) if "sleep_speed" in names else 1.0   # sleep-speed curriculum
        still_home, force_home = SLEEP_STILL_TICKS / ks, FORCE_SLEEP_TICKS / ks
        t_home = t_shelter + SLEEP_SPOT_TILES / SLUGCAT_TILES_PER_TICK
        tiles_home = np.where(spath >= 0, spath + SLEEP_SPOT_TILES, np.inf)

        def home(value, wait):
            # Reach a sleep spot in time (same human trip-speed spread as above); if not, the rain's death.
            p = _p_speed_at_least(tiles_home / np.maximum(t_left - wait, 1.0))
            return p * value * disc(t_home + wait) - (1.0 - p) * down * disc_rain
        # A fed night is also reachable by eating the nearest food first, when it covers the missing
        # pips (in hand: its eating ticks; else the walk to it and its eating). Without this, a
        # slugcat one pip short with a fruit in hand would see only the malnourished night and walk
        # home with the fruit uneaten.
        held_ok = (held_pts > 0) & (held_bites > 0)
        gain = np.where(held_ok, held_pts, np.where((path >= 0) & (pts > 0), pts, 0.0))
        t_gain = np.where(held_ok, t_held, np.maximum(t_food, 0.0))
        t_fed = np.where(need_fed <= 0, 0.0, np.where(gain >= need_fed, t_gain, np.inf))
        fed_ok = np.isfinite(t_fed)
        e_home_fed = np.where(fed_ok, home(up, still_home + np.where(fed_ok, t_fed, 0.0)), -np.inf)
        e_home_maln = np.where((need_any <= 0) & (need_fed > 0) & ~maln, home(v_maln, force_home), -np.inf)
        e_go = np.where(shelter, e_go, np.maximum(e_go, np.maximum(e_home_fed, e_home_maln)))
    # In a shelter it can also sleep with what it has, when the game closes the door
    # (Player.Update). Crediting the whole night on entry would make the policy park at the entrance.
    # The slugcat walks to a sleep spot (sleep_path tiles away), then either stays still for
    # still_ticks_left once ready_for_win (fed and out of the start shelter, or the rain has come),
    # or holds down for force_ticks_left (a malnourished sleep; fed if already malnourished and
    # full). Without ready_for_win the rain brings the night. The night's value follows the food held.
    by_food = np.where(need_fed <= 0, up, np.where(need_any <= 0, v_maln, -down))
    if "sleep_path" in names:
        sp, still, force = f("sleep_path"), f("still_ticks_left"), f("force_ticks_left")
        ready = f("ready_for_win") > 0
        t_spot = np.where(sp >= 0, sp / SLUGCAT_TILES_PER_TICK, np.inf)
        still_t = np.maximum(still, 0.0)
        e_still = np.where(ready & np.isfinite(t_spot), by_food * disc(t_spot + still_t), -np.inf)
        e_force = np.where((force >= 0) & np.isfinite(t_spot),
                           np.where(maln, up, v_maln) * disc(t_spot + force), -np.inf)
        e_rain = np.where(np.isfinite(t_spot), by_food * disc(np.maximum(t_left, t_spot) + still_t), -np.inf)
        e_now = np.maximum(np.maximum(e_still, e_force), e_rain)
    else:
        e_now = by_food                                # older rows: no sleep fields
    e = np.where(shelter, np.maximum(e_now, e_go), e_go)
    karma_change = np.where(has_timer | shelter, e, 0.0)
    return pips, karma_change


def augmented_deficits(rows, gamma: float = GAMMA_TOKEN) -> np.ndarray:
    pips, karma_up = _forecast(rows, gamma)
    return deficits(rows, food_add=pips, karma_add=karma_up)


def potential(rows, gamma: float = GAMMA_TOKEN) -> np.ndarray:
    return drive(deficits(rows)) - drive(augmented_deficits(rows, gamma))


def reward_across_cycle(prev_rows, rows, slept=False) -> np.ndarray:
    """The base reward across a cycle boundary (a sleep or a death), with the food deficit held at
    its pre-boundary value. The stomach change there is the game's bookkeeping (hibernation spends
    the requirement, a death rewinds to the saved stomach), not food the slugcat ate or lost.
    Scoring it would credit deaths with "refunded" pips. What the night changes is karma (and the
    malnourished state, which the forecast prices)."""
    d0, d1 = deficits(prev_rows), deficits(rows).copy()
    d1[..., 0] = d0[..., 0]
    return drive(d0) - drive(d1) + extras_paid(prev_rows, rows, slept)


def shaped_reward(prev_rows, rows, gamma: float, cycle_boundary: bool = False, slept: bool = False) -> np.ndarray:
    base = reward_across_cycle(prev_rows, rows, slept) if cycle_boundary else reward(prev_rows, rows, slept)
    return base + gamma * potential(rows, gamma) - potential(prev_rows, gamma)
