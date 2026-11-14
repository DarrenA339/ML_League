"""The app must compute exactly what the backtest computed."""
import numpy as np
import pytest

import live
from build_artifacts import build_lookups
from data import load_rows
from draft import build_dataset, feature_matrix, fit_logistic, predict_logistic
from elo import EloParams
from make_fake_data import write_fake_data

PARAMS, PLATT = EloParams(k_team=32, k_region=24, side_adv=30.0), {"a": -0.05, "b": 0.95}
HP = dict(pc_shrink=5, window=60, champ_shrink=20)
NAMES = ["elo_logit", "ch_wr", "pc_wr", "pc_games"]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    rows = load_rows(write_fake_data(tmp_path_factory.mktemp("fake"), years=range(2021, 2024), seed=4))
    full = build_dataset(rows, PARAMS, PLATT)
    target = full["games"].iloc[-1]
    X = feature_matrix(full["raw"], full["elo_logit"], names=NAMES, **HP)
    model = fit_logistic(X[:-1], full["y"][:-1], NAMES)
    model["hyperparams"] = HP
    # Artifacts as the app would have them the moment before the last game.
    before = build_dataset(rows[rows["gameid"] != target["gameid"]], PARAMS, PLATT)
    lookups = build_lookups(before, PARAMS, PLATT, as_of=target["date"].to_numpy())
    lookups["elo"]["params"] = EloParams(**lookups["elo"]["params"])
    pick = full["picks"].set_index("gameid").loc[target["gameid"]]
    return full, X, model, lookups, target, pick


@pytest.mark.parametrize("a_is_blue", [True, False])
def test_app_matches_backtest_for_the_same_game(setup, a_is_blue):
    full, X, model, lookups, target, pick = setup
    want_blue = float(predict_logistic(model, X[-1])[0])
    blue = (target["blue"], pick["blue_names"], pick["blue_champs"])
    red = (target["red"], pick["red_names"], pick["red_champs"])
    a, b = (blue, red) if a_is_blue else (red, blue)
    got = live.draft_game(lookups, model, a[0], b[0], a_is_blue, a[1], a[2], b[1], b[2])
    assert got["blue_logit"] == pytest.approx(full["elo_logit"][-1])
    assert list(got["features"].values()) == pytest.approx(list(X[-1]))
    assert got["p"] == pytest.approx(want_blue if a_is_blue else 1 - want_blue)
    p_elo_blue = full["elo_p"][-1]
    assert got["p_elo"] == pytest.approx(p_elo_blue if a_is_blue else 1 - p_elo_blue)
    assert sum(s["points"] for s in got["steps"]) == pytest.approx(100 * (got["p"] - got["p_elo"]))


def test_swapping_teams_gives_the_complement(setup):
    _, _, model, lookups, target, pick = setup
    one = live.draft_game(lookups, model, target["blue"], target["red"], True,
                          pick["blue_names"], pick["blue_champs"], pick["red_names"], pick["red_champs"])
    two = live.draft_game(lookups, model, target["red"], target["blue"], False,
                          pick["red_names"], pick["red_champs"], pick["blue_names"], pick["blue_champs"])
    assert one["p"] + two["p"] == pytest.approx(1.0)
    assert one["p_elo"] + two["p_elo"] == pytest.approx(1.0)


def test_unknown_players_and_champions_use_the_prior(setup):
    _, _, model, lookups, target, _ = setup
    got = live.draft_game(lookups, model, target["blue"], target["red"], True,
                          ["Nobody"] * 5, ["New Champ"] * 5, ["Somebody"] * 5, ["Other Champ"] * 5)
    assert all(v == 0 for n, v in got["features"].items() if n != "elo_logit")
    assert all(not r["known"] and r["games"] == 0 and r["shrunk_wr"] == 0.5 and r["champ_wr"] == 0.5
               for r in got["tables"]["a"] + got["tables"]["b"])
    draft_steps = [s for s in got["steps"] if s["label"] != "Model recalibration of Elo"]
    assert all(s["points"] == pytest.approx(0.0) for s in draft_steps)


def test_no_draft_model_still_gives_elo_and_series(setup):
    _, _, _, lookups, target, _ = setup
    a, b = target["blue"], target["red"]
    assert live.draft_game(lookups, None, a, b, True, [], [], [], []) is None
    p_blue, _ = live.elo_game(lookups, a, b, True)
    p_red, _ = live.elo_game(lookups, a, b, False)
    assert p_blue > p_red                         # blue side helps
    p_series, dist = live.series_odds(lookups, a, b, 5, (1, 1), True)
    assert sum(dist.values()) == pytest.approx(1.0) and 0 < p_series < 1
    # A Bo1 from 0-0 is just the game.
    assert live.series_odds(lookups, a, b, 1, (0, 0), True)[0] == pytest.approx(p_blue)
    assert live.series_odds(lookups, a, b, 1, (0, 0), False)[0] == pytest.approx(p_red)
    # The draft number only replaces the next game.
    assert live.series_odds(lookups, a, b, 5, (2, 2), True, p_next=0.9)[0] == pytest.approx(0.9)


def test_player_names_resolve_to_the_right_team(setup):
    _, _, _, lookups, target, pick = setup
    name = pick["blue_names"][0]
    key, known = live.resolve_player(lookups, name.upper(), target["blue"])
    assert known and key == pick["blue_players"][0]
    key, known = live.resolve_player(lookups, "Nobody At All", target["blue"])
    assert not known and key == "name:Nobody At All"


def test_roster_is_the_latest_lineup(setup):
    full, _, _, lookups, target, _ = setup
    picks = full["picks"].iloc[:-1]
    last = picks[(picks["blue"] == target["blue"])].iloc[-1]
    assert set(lookups["rosters"][target["blue"]]) == set(last["blue_names"])
    assert len(lookups["champions"]) == len(set(lookups["champions"])) > 5
