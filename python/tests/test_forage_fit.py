"""The measured food forecast (train/forage_fit.py)."""
import numpy as np

from rainbot import reward as R
from rainbot.train import forage_fit as FF


def test_discounted_incidence_no_censoring_is_the_plain_mean():
    g = 0.99
    dur = np.array([0, 5, 10, 20, 30.0]); kind = np.array([1, 1, 2, 1, 2])
    want = np.mean([g ** 0, g ** 5, 0, g ** 20, 0])
    assert abs(FF.discounted_incidence(dur, kind, g) - want) < 1e-12


def test_discounted_incidence_censoring_is_not_failure():
    # half the observations are cut before anything happened; the rest all succeed at t=10.
    # Leaving them out or counting them as 0 are both wrong; the estimator recovers 1 * g^10.
    g = 0.99
    dur = np.array([10, 10, 10, 10, 3, 3, 3, 3.0]); kind = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert abs(FF.discounted_incidence(dur, kind, g) - g ** 10) < 1e-12


def test_pava_is_monotone_and_keeps_the_weighted_mean():
    y = np.array([1.0, 0.6, 0.9, 0.2, 0.4]); w = np.array([1, 2, 1, 5, 1.0])
    f = FF.pava_nonincreasing(y, w)
    assert np.all(np.diff(f) <= 1e-12)
    assert abs((f * w).sum() - (y * w).sum()) < 1e-9


def test_measured_table_telescopes_and_rewards_the_press():
    tab = FF.ForageTable(held=list(np.linspace(1.0, 0.2, FF.R_MAX + 1)), adj=0.1,
                         reach=[1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01])
    dt = np.dtype([(k, "f8") for k in ("eat_speed", "held_points", "held_bites", "eat_counter", "food_path",
                                       "food_path_points", "food_path_bites", "karma", "karma_cap", "food_in_stomach", "quarter_food",
                                       "food_to_hibernate", "shelter_path", "in_shelter", "cycle_length",
                                       "cycle_timer", "max_food", "air_in_lungs", "food_hibernate_base")])
    rows = np.zeros(6, dt)
    rows["eat_speed"] = 1; rows["max_food"] = 7; rows["food_to_hibernate"] = 4; rows["food_hibernate_base"] = 4
    rows["air_in_lungs"] = 1; rows["karma"] = -1; rows["karma_cap"] = -1; rows["shelter_path"] = -1
    rows["food_path"] = [20, 1, -1, -1, -1, -1]; rows["food_path_points"] = 1
    rows["held_points"] = [0, 0, 1, 1, 1, 0]; rows["held_bites"] = [0, 0, 1, 1, 1, 0]
    rows["eat_counter"] = [40, 40, 40, 20, 1, 40]; rows["food_in_stomach"] = [1, 1, 1, 1, 1, 2]
    R.set_forage_table(tab)
    try:
        pips = R._forecast(rows, 0.999)[0]
        assert pips[3] - pips[2] > pips[1] - pips[0] > 0          # pressing earns more than approaching
        sh = sum(R.shaped_reward(rows[t], rows[t + 1], 1.0) for t in range(5))
        base = sum(R.reward(rows[t], rows[t + 1]) for t in range(5))
        assert abs(sh - (base + R.potential(rows[-1], 1.0) - R.potential(rows[0], 1.0))) < 1e-12
    finally:
        R.set_forage_table(None)


def test_unvisited_states_keep_the_previous_measurement():
    from rainbot.train import forage_fit as FF
    prior = FF.ForageTable(adj=0.2, n_adj=50, held=[0.5] * (FF.R_MAX + 1), n_held=[10] * (FF.R_MAX + 1),
                           reach=[1.0] + [0.3] * (len(FF.PATH_EDGES) - 2), n_reach=[0] + [20] * (len(FF.PATH_EDGES) - 2))
    tab = FF.fit([], prior=prior)                         # a window that never came near a fruit
    assert tab.adj == 0.2 and tab.held[40] == 0.5 and tab.reach[3] == 0.3
    assert FF.fit([]).adj == 1.0                          # never observed at all: the order's bound
