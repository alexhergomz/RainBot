import numpy as np

from rainbot.scripts.observed_moves import _runs


def test_runs_split_on_room_and_tick_gap():
    tick = np.array([0, 1, 2, 3, 4, 20, 21])
    t0 = np.array([[1, 1], [1, 1], [2, 1], [2, 1], [3, 1], [3, 1], [4, 1]])
    names = np.array(["A", "A", "A", "A", "B", "B", "B"], dtype=object)
    v = _runs(tick, t0, t0, names)
    # A: runs (0-1 at x=1), (2-3 at x=2); B before the tick gap is a single run (dropped), after it two
    assert v == [("A", [(0, 1, 1, 1, 1, 1), (2, 3, 2, 1, 2, 1)]),
                 ("B", [(20, 20, 3, 1, 3, 1), (21, 21, 4, 1, 4, 1)])]


def test_runs_skip_unnamed():
    tick = np.arange(4)
    t0 = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
    names = np.array(["", "", "C", "C"], dtype=object)
    assert _runs(tick, t0, t0, names) == [("C", [(2, 2, 3, 1, 3, 1), (3, 3, 4, 1, 4, 1)])]
