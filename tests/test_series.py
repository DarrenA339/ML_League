import math

import pytest

from elo import win_prob

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


def test_starting_from_a_score_matches_the_full_tree():
    # Total probability over game 1. After an A win B takes blue, after an A loss A does.
    d, s = 60.0, 25.0
    p1 = win_prob(d + s)
    full = score_distribution(d, s, 5, True)
    after_win = score_distribution(d, s, 5, False, start=(1, 0))
    after_loss = score_distribution(d, s, 5, True, start=(0, 1))
    for score in full:
        want = p1 * after_win.get(score, 0.0) + (1 - p1) * after_loss.get(score, 0.0)
        assert full[score] == pytest.approx(want, abs=1e-12)


def test_starting_score_edge_cases():
    d, s = 60.0, 25.0
    decider = score_distribution(d, s, 5, True, start=(2, 2))
    assert decider == pytest.approx({(3, 2): win_prob(d + s), (2, 3): 1 - win_prob(d + s)})
    assert score_distribution(d, s, 5, True, start=(3, 1)) == {(3, 1): 1.0}
    assert all(a >= 2 for a, _ in score_distribution(d, s, 5, None, start=(2, 0)))
    for start in ((1, 0), (0, 2), (2, 2), (1, 1)):
        for side in (True, False, None):
            assert sum(score_distribution(d, s, 5, side, start=start).values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        score_distribution(d, s, 5, True, start=(4, 0))
    with pytest.raises(ValueError):
        score_distribution(d, s, 3, True, start=(2, 2))


def test_p_next_only_replaces_the_next_game():
    d, s = 60.0, 25.0
    plain = score_distribution(d, s, 5, True, start=(1, 1))
    same = score_distribution(d, s, 5, True, start=(1, 1), p_next=win_prob(d + s))
    assert same == pytest.approx(plain)
    sure = score_distribution(d, s, 5, True, start=(1, 1), p_next=1.0)
    assert sure == pytest.approx(score_distribution(d, s, 5, False, start=(2, 1)))
    lost = series_win_prob(d, s, 5, True, start=(1, 1), p_next=0.0)
    assert lost == pytest.approx(series_win_prob(d, s, 5, True, start=(1, 2)))
    up = series_win_prob(d, s, 5, True, start=(1, 1), p_next=0.9)
    assert up > series_win_prob(d, s, 5, True, start=(1, 1)) > lost
