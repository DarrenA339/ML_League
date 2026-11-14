import numpy as np
import pandas as pd
import pytest

import v3
from data import add_end_times, homes_by_year, load_games_v3
from elo import Elo, EloParams, run_v3
from evalkit import paired_bootstrap, release_order, sigmoid
from make_fake_data import write_fake_data
from series import score_distribution


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    return load_games_v3(write_fake_data(tmp_path_factory.mktemp("fake"), seed=2))


def toy_games(rows):
    g = pd.DataFrame(rows, columns=["date", "league", "blue", "red", "blue_win", "gamelength"])
    g["date"] = pd.to_datetime(g["date"])
    g["year"] = g["date"].dt.year
    g["gameid"] = [f"g{i}" for i in range(len(g))]
    return g


def test_home_league_is_decided_per_year():
    g = toy_games([
        ("2022-02-01 10:00", "LCK", "Promoted", "Other", 1, 1800),   # stand-in for a lower league year
        ("2023-02-01 10:00", "LEC", "Promoted", "Fnatic", 1, 1800),
        ("2023-02-02 10:00", "LEC", "Promoted", "Fnatic", 0, 1800),
        ("2023-10-01 10:00", "WLDs", "Promoted", "Wildcard", 1, 1800),
        ("2024-10-01 10:00", "WLDs", "Promoted", "Wildcard", 1, 1800),
    ])
    homes = homes_by_year(g)
    assert homes[2022]["Promoted"] == "LCK"
    assert homes[2023]["Promoted"] == "LEC"
    assert homes[2024]["Promoted"] == "LEC"      # no domestic games, keeps last year's home
    assert homes[2023]["Wildcard"] == "Minor"    # never had a major home league


def test_changing_region_keeps_strength():
    m = Elo(EloParams(), {"A": "X", "B": "X", "C": "Y"})
    m.team.update({"A": 50.0, "B": -20.0, "C": 10.0})
    m.region.update({"X": 100.0, "Y": -40.0})
    before = {t: m.strength(t) for t in "ABC"}
    m.set_homes({"A": "Y", "B": "Z", "C": "Y"})   # A moves to a known region, B founds a new one
    for t in "ABC":
        assert m.strength(t) == pytest.approx(before[t])
    assert m.region["Z"] == pytest.approx(before["B"])
    assert m.team["B"] == pytest.approx(0.0)


def test_release_order_hides_games_in_progress():
    start, end = [0, 5, 20, 30], [10, 8, 25, 40]
    seen = {i: done for i, done in release_order(start, end)}
    assert seen[1] == []            # game 0 is still being played when game 1 starts
    assert sorted(seen[2]) == [0, 1]
    assert seen[3] == [2]


def test_end_time_is_capped_by_the_next_game_of_either_team():
    g = toy_games([
        ("2024-01-01 10:00", "LCK", "A", "B", 1, 7200),
        ("2024-01-01 11:00", "LCK", "B", "A", 0, 1800),
        ("2024-01-01 10:30", "LPL", "C", "D", 1, 1800),
    ]).sort_values("date").reset_index(drop=True)
    g = add_end_times(g)
    first = g[g["gameid"] == "g0"].iloc[0]
    assert first["end"] == pd.Timestamp("2024-01-01 11:00")
    assert (g["end"] > g["date"]).all()


def test_elo_v3_has_no_leakage(fake):
    games, _, homes = fake
    params = EloParams(k_team=32, k_region=24, side_adv=30.0)
    _, base, base_diff = run_v3(games, params, homes)
    for i in (50, len(games) // 2, len(games) - 2):
        changed = games.copy()
        changed.loc[i:, "blue_win"] = 1 - changed.loc[i:, "blue_win"]
        _, probs, diffs = run_v3(changed, params, homes)
        assert np.array_equal(probs[:i + 1], base[:i + 1])        # nothing at or before game i moved
        assert np.array_equal(diffs[:i + 1], base_diff[:i + 1])
        assert not np.array_equal(probs[i + 1:], base[i + 1:])    # and later games do react


def test_elo_v3_uses_game_one_for_game_two(fake):
    games, _, homes = fake
    params = EloParams(k_team=32, k_region=24, side_adv=30.0)
    _, base, _ = run_v3(games, params, homes)
    i = int(games.index[(games["game_in_series"] == 1) & (games["best_of"] == 5)][10])
    nxt = games.index[(games["series_id"] == games.loc[i, "series_id"]) & (games["game_in_series"] == 2)][0]
    changed = games.copy()
    changed.loc[i, "blue_win"] = 1 - changed.loc[i, "blue_win"]
    _, probs, _ = run_v3(changed, params, homes)
    assert probs[i] == base[i]
    assert probs[nxt] != base[nxt]


def test_elo_v3_ignores_an_overlapping_game():
    g = toy_games([
        ("2024-01-01 10:00", "LCK", "A", "B", 1, 1800),
        ("2024-01-02 10:00", "MSI", "A", "C", 1, 3600),
        ("2024-01-02 10:20", "MSI", "B", "D", 1, 1800),   # starts while A vs C is still going
        ("2024-01-03 10:00", "LPL", "C", "D", 1, 1800),
        ("2024-01-03 12:00", "LPL", "D", "C", 1, 1800),
    ])
    g = add_end_times(g)
    homes = {2024: {"A": "LCK", "B": "LCK", "C": "LPL", "D": "LPL"}}
    params = EloParams(k_team=32, k_region=24, side_adv=0.0)
    _, base, _ = run_v3(g, params, homes)
    flipped = g.copy()
    flipped.loc[1, "blue_win"] = 0
    _, probs, _ = run_v3(flipped, params, homes)
    assert probs[2] == base[2]      # B vs D cannot see the unfinished A vs C
    assert probs[3] != base[3]


def test_platt_roundtrip_matches_series_inputs():
    params, platt = EloParams(side_adv=40.0), {"a": -0.07, "b": 0.9}
    diff = 80.0
    want_blue = sigmoid(v3.calibrated_logit(diff, params, platt))
    want_red = 1 - sigmoid(v3.calibrated_logit(-diff, params, platt))
    d, s = v3.effective(diff, params, platt)
    assert score_distribution(d, s, 1, True)[(1, 0)] == pytest.approx(want_blue)
    assert score_distribution(d, s, 1, False)[(1, 0)] == pytest.approx(want_red)


def test_fit_platt_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1.2, 20000)
    y = (rng.random(20000) < sigmoid(0.1 + 0.7 * z)).astype(int)
    a, b = v3.fit_platt(z, y)
    assert a == pytest.approx(0.1, abs=0.05)
    assert b == pytest.approx(0.7, abs=0.05)


def test_paired_bootstrap_sees_a_real_improvement():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.2, 0.8, 4000)
    y = (rng.random(4000) < truth).astype(int)
    mean, lo, hi, _ = paired_bootstrap(truth, np.full(4000, 0.5), y, n_boot=500)
    assert hi < 0 and lo < mean < hi
    mean, lo, hi, _ = paired_bootstrap(truth, truth, y, n_boot=200)
    assert mean == 0 and lo == 0 and hi == 0
