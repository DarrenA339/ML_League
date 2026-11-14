import math

from series import score_distribution, series_win_prob


def diff_for(p):
    return 400 * math.log10(p / (1 - p))


def test_matches_binomial_when_no_side_advantage():
    p = 0.6
    want = p**3 + 3 * p**3 * (1 - p) + 6 * p**3 * (1 - p) ** 2
    assert abs(series_win_prob(diff_for(p), 0.0, 5, True) - want) < 1e-12


def test_distribution_sums_to_one():
    for side in (True, False, None):
        for bo in (1, 3, 5, 7):
            dist = score_distribution(37.0, 25.0, bo, side)
            assert abs(sum(dist.values()) - 1) < 1e-12


def test_symmetric_when_game1_side_unknown():
    for d in (-120.0, 0.0, 55.0):
        assert abs(series_win_prob(d, 25.0, 5) + series_win_prob(-d, 25.0, 5) - 1) < 1e-12


def test_series_amplifies_game_edge():
    assert series_win_prob(diff_for(0.6), 0.0, 5, True) > series_win_prob(diff_for(0.6), 0.0, 3, True) > 0.6
