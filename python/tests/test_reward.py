"""Reward and shaping checks that need no game, plus curriculum specs and a GRPO
pass on fake games."""
import numpy as np

from rainbot import reward as R


def _row(food=0, quarter=0, hib=4, maxf=7, karma=2, cap=4, air=1.0):
    dt = np.dtype([("food_in_stomach", "i2"), ("quarter_food", "i2"), ("food_to_hibernate", "i2"),
                   ("max_food", "i2"), ("karma", "i2"), ("karma_cap", "i2"), ("air_in_lungs", "f4")])
    r = np.zeros((), dt)
    r["food_in_stomach"], r["quarter_food"], r["food_to_hibernate"], r["max_food"] = food, quarter, hib, maxf
    r["karma"], r["karma_cap"], r["air_in_lungs"] = karma, cap, air
    return r


def test_each_pip_pays_less_near_the_requirement_and_extras_nothing_before_the_night():
    pays = [float(R.reward(_row(food=f), _row(food=f + 1))) for f in range(0, 7)]
    below = pays[:4]
    assert all(p > 0 for p in below)
    assert all(a > b for a, b in zip(below, below[1:])), pays      # diminishing near the setpoint
    assert all(abs(p) < 1e-12 for p in pays[4:]), pays             # nothing above it


def test_deficits_are_necessities_on_their_own_range():
    d = R.deficits(_row(food=6, karma=9, cap=9, air=1.0))
    assert np.allclose(d, 0.0)                                      # fed (extras are not a need), full
    # karma setpoint = the game's maximum (9), not the current cap
    assert np.isclose(R.deficits(_row(food=6, karma=4, cap=4))[1], 5 / 9)
    d = R.deficits(_row(food=0, karma=0, air=0.0))
    assert np.allclose(d, [1.0, 1.0, 1.0])                         # each over its own range
    assert np.isclose(R.deficits(_row(food=3))[0], 1 / 4)


def test_extras_are_paid_at_a_sleep_only():
    # 6 pips (2 extras) booked at a sleep (karma 2 -> 3): tomorrow starts 2 pips in, worth 1 - (2/4)^2
    assert np.isclose(float(R.extras_paid(_row(food=6, karma=2), _row(food=6, karma=3), True)), 1 - (2 / 4) ** 2)
    assert float(R.extras_paid(_row(food=4, karma=2), _row(food=4, karma=3), True)) == 0.0     # fed, no extras
    assert float(R.extras_paid(_row(food=6, karma=2), _row(food=6, karma=1), False)) == 0.0    # a death
    assert float(R.extras_paid(_row(food=5, karma=2), _row(food=6, karma=2), False)) == 0.0    # eating, no night
    # at the karma cap a sleep adds no karma, and the extras are still paid
    assert np.isclose(float(R.extras_paid(_row(food=6, karma=4, cap=4), _row(food=6, karma=4, cap=4), True)), 0.75)
    # a sleep with extras beats one without, which beats a malnourished one
    fed = float(R.reward(_row(food=4, karma=2), _row(food=4, karma=3), slept=True))
    more = float(R.reward(_row(food=6, karma=2), _row(food=6, karma=3), slept=True))
    assert more > fed > 0


def test_death_costs_more_than_hibernation():
    before = _row(food=5, karma=2)
    slept = _row(food=1, karma=3)        # pips spent, +1 karma
    died = _row(food=0, karma=1)         # stomach emptied, -1 karma
    assert R.reward(before, slept) > R.reward(before, died)


def test_noop_drift_is_never_positive():
    # Standing still changes nothing in the needs, so it is worth exactly nothing; drowning costs.
    assert R.reward(_row(food=1), _row(food=1)) == 0
    assert R.reward(_row(air=1.0), _row(air=0.5)) < 0


def test_no_karma_outside_story_sessions():
    assert R.deficits(_row(karma=-1, cap=-1))[1] == 0


def test_curriculum_specs_only_use_protocol_reset_keys():
    """Every key the curriculum emits must be one the mod's Reset.Parse accepts (peaceful is a
    separate command, split out by RainEnv.request_reset)."""
    from rainbot.train.curriculum import STAGES, Curriculum
    allowed = {"room", "pos", "food", "karma", "timer_left", "creatures", "spawn", "fruit", "items", "seed",
               "fruit_near", "hold", "bites", "eat_counter", "eat_speed", "sleep_speed", "climb", "shelter_within"}
    c = Curriculum()
    c.set_world({"rooms": ["SU_A01", "SU_A02", "SU_S01"], "shelters": ["SU_S01"],
                 "connections": {"SU_A01": ["SU_A02"], "SU_A02": ["SU_A01", "SU_S01"], "SU_S01": ["SU_A02"]}})
    for st in STAGES:
        _, spec = c.reset_spec(st)
        keys = {t.split("=", 1)[0] for t in spec.split()} - {"peaceful"}
        assert keys <= allowed, (st, keys - allowed)


def _srow(food=0, hold=0, eat=40, shelter=0, karma=2, cap=4, air=1.0, path=-1, pts=1, bites=3,
          held_bites=3, spath=-1, timer=0, length=0, k=1, maln=0, hib=4, base=4):
    dt = np.dtype([("food_in_stomach", "i2"), ("quarter_food", "i2"), ("food_to_hibernate", "i2"),
                   ("max_food", "i2"), ("karma", "i2"), ("karma_cap", "i2"), ("air_in_lungs", "f4"),
                   ("holding_edible", "u1"), ("eat_counter", "u1"), ("in_shelter", "u1"),
                   ("food_path", "i2"), ("food_path_points", "i1"), ("food_path_bites", "u1"),
                   ("shelter_path", "i2"), ("held_points", "i1"), ("held_bites", "u1"), ("eat_speed", "u1"),
                   ("cycle_timer", "i4"), ("cycle_length", "i4"), ("malnourished", "u1"),
                   ("food_hibernate_base", "i2")])
    r = np.zeros((), dt)
    r["food_in_stomach"], r["food_to_hibernate"], r["max_food"] = food, hib, 7
    r["malnourished"], r["food_hibernate_base"] = maln, base
    r["karma"], r["karma_cap"], r["air_in_lungs"] = karma, cap, air
    r["holding_edible"], r["eat_counter"], r["in_shelter"] = hold, eat, shelter
    r["food_path"], r["food_path_points"], r["food_path_bites"] = path, pts, bites
    r["held_points"], r["held_bites"] = (1, held_bites) if hold else (0, 0)
    r["shelter_path"] = 0 if shelter else spath
    r["eat_speed"], r["cycle_timer"], r["cycle_length"] = k, timer, length
    return r


G = 0.999


def test_food_forecast_rises_as_the_path_shortens_and_the_bite_completes():
    phis = [float(R.potential(_srow(path=p))) for p in (200, 100, 20, 0)]
    assert all(a < b for a, b in zip(phis, phis[1:])), phis          # closer = more forecast
    chew = [float(R.potential(_srow(hold=1, eat=e, held_bites=b))) for b, e in ((3, 40), (3, 10), (1, 10), (1, 1))]
    assert all(a < b for a, b in zip(chew, chew[1:])), chew
    # picking up is instant in this model: a full fruit in hand = the same fruit at path 0
    assert np.isclose(R.potential(_srow(hold=1)), R.potential(_srow(path=0)))
    # completion dominates: the forecast is always below one real pip
    assert R.potential(_srow(hold=1, eat=1, held_bites=1)) < R.reward(_srow(), _srow(food=1))
    assert float(R.potential(_srow())) == 0.0                        # nothing in reach: no forecast


def test_food_forecast_is_nothing_when_fed():
    assert abs(float(R.potential(_srow(food=4, path=5)))) < 1e-12


def test_eat_speed_shortens_the_forecast_time():
    assert R.potential(_srow(hold=1, k=40)) > R.potential(_srow(hold=1, k=1))


def test_night_ordering_fed_beats_malnourished_beats_death():
    """Sleeping fed > sleeping malnourished > dying, in the episode's own return: the path, the
    cycle boundary (food held: no "refunded" pips) and the forecast kept on waking."""
    g = 0.999
    start = _srow(food=2, karma=3, cap=4, spath=40, timer=8000, length=9000)
    wake = dict(shelter=1, timer=0, length=9000)
    paths = {"fed": [_srow(food=4, karma=3, cap=4, spath=40, timer=8200, length=9000),
                     _srow(food=4, karma=3, cap=4, shelter=1, timer=8500, length=9000),
                     _srow(food=0, karma=4, cap=4, **wake)],
             "malnourished": [_srow(food=2, karma=3, cap=4, shelter=1, timer=8500, length=9000),
                              _srow(food=0, karma=4, cap=4, maln=1, hib=7, **wake)],
             "death": [_srow(food=2, karma=3, cap=4, spath=40, timer=9000, length=9000),
                       _srow(food=2, karma=2, cap=4, **wake)]}
    value = {}
    for k, rows in paths.items():
        seq = [start] + rows
        ret = 0.0
        for t, (a, b) in enumerate(zip(seq, seq[1:])):
            ret += g ** t * float(R.shaped_reward(a, b, g, cycle_boundary=(b is rows[-1])))
        value[k] = ret
    assert value["fed"] > value["malnourished"] > value["death"], value


def test_forecast_falls_with_distance_and_time_even_at_the_cap():
    near = float(R.potential(_srow(food=4, karma=4, cap=4, spath=100, timer=0, length=9000)))
    far = float(R.potential(_srow(food=4, karma=4, cap=4, spath=1500, timer=0, length=9000)))
    late = float(R.potential(_srow(food=4, karma=4, cap=4, spath=100, timer=8600, length=9000)))
    assert near > far and near > late                    # at the cap the death risk still bites
    hungry = float(R.potential(_srow(food=2, spath=100, timer=0, length=9000)))
    fed = float(R.potential(_srow(food=4, spath=100, timer=0, length=9000)))
    assert fed >= hungry


def test_shelter_rules():
    # a shelter with >= 1 pip: sleep now survives (+1); with 0 pips sleeping now starves, so the
    # forecast is the better of that and going out to forage
    assert R.potential(_srow(food=2, shelter=1, timer=0, length=9000)) > 0
    assert R.potential(_srow(food=2, shelter=1, timer=0, length=9000)) > \
        R.potential(_srow(food=0, shelter=1, timer=0, length=9000))
    # malnourished: below max food a shelter means starving
    assert R.potential(_srow(food=4, shelter=1, maln=1, hib=7, timer=8900, length=9000)) < 0


def test_rows_without_path_fields_get_no_forecast():
    assert abs(float(R.potential(_row(food=1)))) < 1e-12


def test_grpo_group_advantages_on_fake_games():
    """GRPO end to end on fake games: groups start identically, advantages are group-normalised."""
    from rainbot import obs as O
    from rainbot.env.fake_game import FakeGame
    from rainbot.env.rain_env import RainEnv
    from rainbot.train.ppo import PPOConfig, Trainer
    import tempfile
    port = 57300
    for i in range(3):
        FakeGame(port + i, seed=0).start()
    vocab = O.Vocab()
    cfg = PPOConfig(instances=3, base_port=port, algo="grpo", group_tokens=12, groups_per_update=1, stage="F",
                    epochs=1, minibatch_seqs=3, learner_device="cpu", actor_device="cpu",
                    run_dir=tempfile.mkdtemp())
    tr = Trainer(cfg, [RainEnv(instance=i, port=port + i, vocab=vocab) for i in range(3)])
    tr.start()
    ro, adv, seq_states = tr.collect_group()
    assert len(seq_states) == ro.T // cfg.group_tokens        # one stored state per chunk
    # Same seeded start: every member's first step is the same observation.
    first = ro.feat["body"][:, 0]
    assert np.allclose(first, first[0:1])
    Rp = cfg.group_repeats
    L = ro.T // Rp                                          # whole-cycle episodes: length from the rollout
    members = np.concatenate([adv[:, r * L:(r + 1) * L] for r in range(Rp)], 0)   # instances x repeats
    assert np.allclose(members.mean(0), 0, atol=1e-5)      # normalised across the whole group per step
    # The learner's teacher-forced log-probs equal the ones stored at rollout (autoregressive head:
    # each decision conditioned on the ones before it, state carried across segments).
    import torch
    from rainbot.nn.agent import log_prob_entropy, to_tensors
    from rainbot.train.rollout import stack_states
    L_ = cfg.group_tokens
    for k in range(ro.T // L_):
        sl = slice(k * L_, (k + 1) * L_)
        f = to_tensors({n: v[:, sl] for n, v in ro.feat.items()}, "cpu")
        init = [stack_states([seq_states[k][b].index(torch.tensor([i])) for i in range(3)])
                for b in range(len(seq_states[0]))]
        act = torch.as_tensor(ro.act[:, sl]).long()
        with torch.no_grad():
            lg, _, _, _ = tr.learner(f, init, torch.as_tensor(ro.reset[:, sl]), act=act)
        lp, _ = log_prob_entropy(lg / torch.as_tensor(ro.temp[:, sl]).view(3, L_, 1, 1), act)
        m = ro.valid[:, sl] & ro.acted[:, sl]
        assert np.allclose(lp.numpy()[m], ro.logp[:, sl][m], atol=1e-3), np.abs(lp.numpy()[m] - ro.logp[:, sl][m]).max()
    out = tr.update(ro, seq_states, adv, cfg.group_tokens)
    assert "pg" in out and np.isfinite(out["pg"])


def test_meeting_an_echo_is_not_punished():
    # An echo raises the cap and refills karma to it (SaveState.GhostEncounter): 4/4 -> 6/6.
    assert R.reward(_row(food=4, karma=4, cap=4), _row(food=4, karma=6, cap=6)) > 0


def test_shaping_telescopes_exactly_with_phi_zero_at_the_end():
    rng = np.random.default_rng(0)
    seq = [_srow(food=int(rng.integers(0, 5)), hold=int(rng.integers(0, 2)), eat=int(rng.integers(0, 41)),
                 shelter=int(rng.integers(0, 2)), karma=int(rng.integers(0, 5))) for _ in range(30)]
    g = 0.999
    shaped = [float(R.shaped_reward(a, b, g)) for a, b in zip(seq, seq[1:])]
    shaped[-1] -= g * float(R.potential(seq[-1]))           # the cut: Phi(terminal) = 0
    base = [float(R.reward(a, b)) for a, b in zip(seq, seq[1:])]
    ret = lambda rs: sum(g ** t * r for t, r in enumerate(rs))
    assert np.isclose(ret(shaped), ret(base) - float(R.potential(seq[0])))


def _with_fields(base, extra):
    """A copy of the structured row ``base`` with the fields ``extra`` [(name, dtype)] added as 0."""
    r = np.zeros((), np.dtype(base.dtype.descr + extra))
    for n in base.dtype.names:
        r[n] = base[n]
    return r


def _sleeprow(sleep_path, still=41, force=-1, ready=0, **kw):
    """_srow plus the sleep fields (the game's door-closing conditions)."""
    r = _with_fields(_srow(shelter=1, timer=2000, length=9000, **kw),
                     [("sleep_path", "i2"), ("start_shelter", "u1"), ("ready_for_win", "u1"),
                      ("still_ticks_left", "i2"), ("force_ticks_left", "i2")])
    r["sleep_path"], r["ready_for_win"], r["still_ticks_left"], r["force_ticks_left"] = sleep_path, ready, still, force
    return r


def test_malnourished_sleep_is_credited_as_the_game_closes_the_door():
    """The night comes as the game brings it: walk to a sleep spot, then hold down 260 ticks. A
    policy parked at the shelter entrance is not credited with the whole night."""
    kw = dict(food=2, karma=2, cap=4)
    at_entrance = float(R.potential(_sleeprow(sleep_path=8, force=261, **kw)))
    on_spot = float(R.potential(_sleeprow(sleep_path=0, force=261, **kw)))
    holding = [float(R.potential(_sleeprow(sleep_path=0, force=f, **kw))) for f in (200, 100, 0)]
    assert at_entrance < on_spot < holding[0] < holding[1] < holding[2], (at_entrance, on_spot, holding)


def test_ladder_specs_and_timers():
    """The ladder: every rung a whole cycle with food, eating and a shelter together; dry rooms first;
    the rain timer is the same cap at every rung."""
    from rainbot.train.curriculum import Curriculum, LADDER
    c = Curriculum(seed=1, replay_frac=0.0)
    c.set_world({"rooms": ["SU_A01", "SU_A02", "SU_S01", "SU_W01"], "shelters": ["SU_S01"],
                 "connections": {"SU_A01": ["SU_A02"], "SU_A02": ["SU_A01", "SU_S01"],
                                 "SU_S01": ["SU_A02", "SU_W01"], "SU_W01": ["SU_S01"]}})
    c.water_rooms = {"SU_W01"}
    for _ in range(20):
        _, spec = c.reset_spec("L")
        kv = dict(t.split("=", 1) for t in spec.split())
        assert kv["room"] == "SU_A02" and kv["hold"] == "fruit" and kv["eat_speed"] == "20" and kv["food"] in ("3", "4")
    c.level = len(LADDER) - 1
    assert all("hold=" not in c.reset_spec("L")[1] for _ in range(10))
    assert all("timer_left=9000" in c.reset_spec("L")[1] for _ in range(5))   # the cap at every rung


def test_fed_slugcat_outside_is_pulled_home():
    """A fed slugcat outside can go home and sleep at once (the game closes the door on a still,
    fed slugcat on a sleep spot): the forecast rises as the shelter gets closer, long before the rain."""
    far = float(R.potential(_srow(food=5, karma=2, cap=4, spath=300, timer=0, length=9000)))
    mid = float(R.potential(_srow(food=5, karma=2, cap=4, spath=100, timer=0, length=9000)))
    near = float(R.potential(_srow(food=5, karma=2, cap=4, spath=10, timer=0, length=9000)))
    assert far < mid < near, (far, mid, near)


def test_fruit_in_hand_covering_the_need_counts_as_a_fed_night():
    """One pip short with a fruit in hand, the fed night is in reach (eat it, then sleep): the karma
    forecast must see it, not only the malnourished night it would see without the fruit."""
    def karma(row):
        return float(R._forecast(row, G)[1])
    holding = _srow(food=3, hold=1, held_bites=3, eat=40, karma=2, cap=4, spath=100, timer=0, length=9000)
    empty = _srow(food=3, karma=2, cap=4, spath=100, timer=0, length=9000)
    fed = _srow(food=4, karma=2, cap=4, spath=100, timer=0, length=9000)
    assert karma(empty) < karma(holding) <= karma(fed), (karma(empty), karma(holding), karma(fed))


def test_ladder_rate_counts_cut_episodes_as_failures():
    from rainbot.train.curriculum import Curriculum, EpisodeStats
    eps = [EpisodeStats(stage="L", sleeps=1, fed_sleeps=1) for _ in range(3)] + \
          [EpisodeStats(stage="L") for _ in range(4)] + [EpisodeStats(stage="L", deaths=1)]
    assert abs(Curriculum.metrics(eps)["fed_hibernation_rate"] - 3 / 8) < 1e-9


def test_sleep_speed_shortens_the_wait_for_the_door():
    """The sleep-speed curriculum's K divides the still / hold-down ticks: going home is worth more."""
    def row(k):
        r = _with_fields(_srow(food=5, karma=2, cap=4, spath=40, timer=0, length=9000), [("sleep_speed", "u1")])
        r["sleep_speed"] = k
        return r
    assert float(R.potential(row(20))) > float(R.potential(row(1)))
