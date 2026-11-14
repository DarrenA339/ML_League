"""Leakage and fallback tests for the draft features.

Three properties are checked for every feature, through the full pipeline
  truncation   cutting off all later games leaves a game's features unchanged
  own result   flipping the result of a game and of everything after it
               leaves that game's features unchanged
  series       game 2 of a series does change when game 1 changes
"""
import numpy as np
import pandas as pd
import pytest

from data import load_rows
from draft import (CHAMP_WINDOWS, History, build_dataset, build_picks, champ_table, feature_matrix,
                   fit_logistic, player_table, predict_logistic, raw_side_from_snapshot, replay)
from elo import EloParams
from make_fake_data import write_fake_data

ALL = ["elo_logit", "pc_wr", "pc_games", "ch_wr", "ch_pick", "ch_wr_x"]
USES_RESULTS = ["elo_logit", "pc_wr", "ch_wr", "ch_wr_x"]
PARAMS, PLATT = EloParams(k_team=32, k_region=24, side_adv=30.0), {"a": -0.05, "b": 0.95}
HP = dict(pc_shrink=5, window=60, champ_shrink=20)


def features(rows):
    data = build_dataset(rows, PARAMS, PLATT)
    X = feature_matrix(data["raw"], data["elo_logit"], names=ALL, **HP)
    return data["games"], pd.DataFrame(X, columns=ALL, index=data["games"]["gameid"])


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    return load_rows(write_fake_data(tmp_path_factory.mktemp("fake"), years=range(2021, 2024), seed=3))


@pytest.fixture(scope="module")
def base(rows):
    return features(rows)


def flip_results(rows, gameids):
    out = rows.copy()
    hit = out["gameid"].isin(set(gameids))
    out.loc[hit, "result"] = 1 - out.loc[hit, "result"]
    return out


@pytest.fixture(scope="module")
def cut_points(base):
    games, _ = base
    return [200, len(games) // 2, len(games) - 3]


@pytest.fixture(scope="module")
def truncated(rows, base, cut_points):
    games, _ = base
    return {i: features(rows[rows["gameid"].isin(set(games["gameid"][:i + 1]))])[1] for i in cut_points}


@pytest.fixture(scope="module")
def flipped(rows, base, cut_points):
    games, _ = base
    return {i: features(flip_results(rows, games["gameid"][i:]))[1] for i in cut_points}


@pytest.mark.parametrize("name", ALL)
def test_feature_ignores_all_later_games(name, base, truncated, cut_points):
    games, X = base
    for i in cut_points:
        ids = games["gameid"][:i + 1]
        assert np.array_equal(X.loc[ids, name].to_numpy(), truncated[i].loc[ids, name].to_numpy())


@pytest.mark.parametrize("name", ALL)
def test_feature_ignores_own_and_later_results(name, base, flipped, cut_points):
    games, X = base
    for i in cut_points:
        before, after = games["gameid"][:i + 1], games["gameid"][i + 1:]
        assert np.array_equal(X.loc[before, name].to_numpy(), flipped[i].loc[before, name].to_numpy())
        # ch_wr_x leaves out the two teams' own games, so the last few games of
        # the final series have nothing later to react to.
        if name in USES_RESULTS and not (name == "ch_wr_x" and i == cut_points[-1]):
            assert not np.array_equal(X.loc[after, name].to_numpy(), flipped[i].loc[after, name].to_numpy())


def series_pair(games):
    late = games[(games["year"] == games["year"].max()) & (games["best_of"] == 5)]
    g1 = late[late["game_in_series"] == 1].iloc[5]
    g2 = late[(late["series_id"] == g1["series_id"]) & (late["game_in_series"] == 2)].iloc[0]
    return g1["gameid"], g2["gameid"]


@pytest.mark.parametrize("name", USES_RESULTS)
def test_game_two_sees_the_result_of_game_one(name, rows, base):
    games, X = base
    g1, g2 = series_pair(games)
    _, changed = features(flip_results(rows, [g1]))
    assert changed.loc[g1, name] == X.loc[g1, name]
    if name == "ch_wr_x":
        assert changed.loc[g2, name] == X.loc[g2, name]   # by design it skips these two teams' games
    else:
        assert changed.loc[g2, name] != X.loc[g2, name]


def test_game_two_counts_the_picks_of_game_one(rows, base):
    games, X = base
    g1, g2 = series_pair(games)
    data = build_dataset(rows, PARAMS, PLATT)
    without = build_dataset(rows[rows["gameid"] != g1], PARAMS, PLATT)
    at = lambda d: d["raw"][(d["games"]["gameid"] == g2).to_numpy()][0]
    full, less = at(data), at(without)
    assert (full[:, :, 2] == less[:, :, 2] + 1).all()          # every player has one more game
    assert (full[:, :, 6] == less[:, :, 6] + 1).all()          # one more game in the 30 day window
    assert full[:, :, 4].sum() > less[:, :, 4].sum()           # champion pick counts moved


def test_first_game_ever_counts_nothing(rows):
    picks = build_picks(rows)
    raw, _ = replay(picks)
    assert not raw[0].any()


def toy_picks(spec):
    rows = []
    for i, (start, end, blue, red, win) in enumerate(spec):
        rows.append({
            "gameid": f"g{i}", "date": pd.Timestamp(start), "end": pd.Timestamp(end),
            "blue": blue, "red": red, "blue_win": win, "league": "LCK",
            "blue_players": [f"{blue}{k}" for k in range(5)], "blue_champs": list("abcde"),
            "red_players": [f"{red}{k}" for k in range(5)], "red_champs": list("vwxyz"),
        })
    return pd.DataFrame(rows)


def test_a_game_still_in_progress_is_invisible():
    picks = toy_picks([
        ("2024-01-01 10:00", "2024-01-01 11:00", "A", "B", 1),
        ("2024-01-01 10:30", "2024-01-01 11:10", "C", "D", 1),   # starts before A vs B ends
        ("2024-01-01 12:00", "2024-01-01 12:30", "C", "D", 1),
    ])
    raw, _ = replay(picks)
    assert not raw[1].any()                      # champion "a" has been picked but not finished
    assert raw[2, 0, 0, 4] == 2                  # by noon both earlier games count for champion "a"
    assert raw[2, 0, 0, 0] == 1 and raw[2, 0, 0, 1] == 1   # and C's own first game does too


def test_window_and_horizon_drop_old_games():
    picks = toy_picks([
        ("2022-01-01 10:00", "2022-01-01 10:30", "A", "B", 1),
        ("2024-01-01 10:00", "2024-01-01 10:30", "A", "B", 1),
        ("2024-02-15 10:00", "2024-02-15 10:30", "A", "B", 1),
    ])
    raw, _ = replay(picks)
    last = raw[2, 0, 0]
    assert last[0] == 1                           # the 2022 game is past the two year horizon
    w30, w60 = CHAMP_WINDOWS.index(30), CHAMP_WINDOWS.index(60)
    assert last[4 + 5 * w30] == 0 and last[4 + 5 * w60] == 1


def test_leave_out_columns_drop_only_the_two_teams_own_games():
    picks = toy_picks([
        ("2024-01-01 10:00", "2024-01-01 10:30", "A", "B", 1),
        ("2024-01-02 10:00", "2024-01-02 10:30", "C", "D", 0),
        ("2024-01-03 10:00", "2024-01-03 10:30", "A", "C", 1),
        ("2024-01-04 10:00", "2024-01-04 10:30", "A", "B", 1),
    ])
    raw, _ = replay(picks)
    w = 4 + 5 * CHAMP_WINDOWS.index(60)
    last = raw[3, 0, 0]                           # champion "a" before the second A vs B
    assert (last[w], last[w + 1]) == (3, 2)       # picked in all three earlier games, won two
    assert (last[w + 3], last[w + 4]) == (1, 0)   # only C vs D is left once A and B games go


def test_unseen_player_and_champion_fall_back_to_the_prior():
    hist = History()
    now = np.datetime64("2024-06-01")
    raw = hist.raw_side(["nobody"] * 5, ["Brand New Champ"] * 5, now, ("X", "Y"))
    pt, ct = player_table(raw, 5), champ_table(raw, 60, 20)
    assert (pt["overall_wr"] == 0.5).all() and (pt["shrunk_wr"] == 0.5).all()
    assert (ct["champ_wr"] == 0.5).all() and (ct["champ_pick_share"] == 0).all()
    X = feature_matrix(np.stack([raw, raw]), 0.3, names=ALL, **HP)
    assert X[0] == 0.3 and not X[1:].any()        # only Elo is left, so the model returns Elo


def test_shrunk_win_rate_moves_toward_the_player_average():
    raw = np.zeros((5, 4 + 5 * len(CHAMP_WINDOWS)))
    raw[0, :4] = [4, 4, 80, 40]                   # 4 and 0 on the champion, 50% overall
    assert player_table(raw, 4)["shrunk_wr"][0] == pytest.approx(0.75)
    assert player_table(raw, 36)["shrunk_wr"][0] == pytest.approx(0.55)


def test_snapshot_matches_live_history(rows):
    picks = build_picks(rows)
    _, hist = replay(picks)
    now = picks["end"].max().to_numpy() + np.timedelta64(1, "h")
    snap = hist.snapshot(now)
    for k in (-1, -40, -400):
        g = picks.iloc[k]
        teams = (g["blue"], g["red"])
        live = hist.raw_side(g["blue_players"], g["red_champs"], now, teams)
        saved = raw_side_from_snapshot(snap, g["blue_players"], g["red_champs"], teams)
        assert np.array_equal(live, saved)
        live = hist.raw_side(g["blue_players"], g["blue_champs"], now, teams)
        assert np.array_equal(live, raw_side_from_snapshot(snap, g["blue_players"], g["blue_champs"], teams))


def test_logistic_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 3)) * [1.0, 0.1, 5.0]
    y = (rng.random(4000) < 1 / (1 + np.exp(-(X[:, 0] + 4 * X[:, 1])))).astype(int)
    model = fit_logistic(X, y, ["a", "b", "c"])
    p = predict_logistic(model, X)
    assert p.shape == (4000,) and 0 < p.min() < p.max() < 1
    assert model["coef"][0] > 0.7 and abs(model["coef"][2]) < 0.15
    assert predict_logistic(model, X[0]).shape == (1,)
