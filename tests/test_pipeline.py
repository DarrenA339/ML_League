from backtest import log_loss
from data import load_games
from elo import EloParams, estimate_side_adv, run
from make_fake_data import TRUE_SIDE_ADV, write_fake_data


def test_recovers_known_structure(tmp_path):
    games, series, home = load_games(write_fake_data(tmp_path, seed=1))

    # every fake series is reconstructed with the right length
    assert set(series["best_of"]) == {3, 5}
    assert set(games.loc[games["league"] == "WLDs", "best_of"]) == {5}

    side = estimate_side_adv(games)
    assert abs(side - TRUE_SIDE_ADV) < 15

    model, probs, _ = run(games, EloParams(side_adv=side), home)
    assert model.region["LCK"] > model.region["LCS"]
    assert model.region["LPL"] > model.region["LCS"]

    mask = (games["intl_event"] & games["cross_region"] & (games["year"] >= 2023)).to_numpy()
    y = games["blue_win"].to_numpy()
    assert log_loss(probs[mask], y[mask]) < 0.67
