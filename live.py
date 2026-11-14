"""Fast single-game and series predictions from the saved artifacts.

The app only talks to this module. Nothing here reads the CSVs.
"""
import json
import os
import pickle

import numpy as np

import v3
from draft import (champ_table, feature_matrix, player_table, predict_logistic,
                   raw_side_from_snapshot, POSITIONS)
from elo import EloParams
from evalkit import sigmoid
from series import score_distribution

LOOKUPS = "lookups.pkl"
MODEL = "draft_model.json"


def load_artifacts(folder="models"):
    """Returns (lookups, draft model or None). The app still runs on Elo alone."""
    with open(os.path.join(folder, LOOKUPS), "rb") as f:
        lookups = pickle.load(f)
    lookups["elo"]["params"] = EloParams(**lookups["elo"]["params"])
    model = None
    path = os.path.join(folder, MODEL)
    if os.path.exists(path):
        with open(path) as f:
            model = json.load(f)
    return lookups, model


def strength(lookups, team):
    e = lookups["elo"]
    return e["team"].get(team, 0.0) + e["region"].get(e["home"].get(team), 0.0)


def elo_diff(lookups, a, b):
    return strength(lookups, a) - strength(lookups, b)


def elo_game(lookups, a, b, a_blue):
    """Calibrated Elo probability that A wins one game on the given side."""
    e = lookups["elo"]
    d = elo_diff(lookups, a, b)
    blue_logit = float(v3.calibrated_logit(d if a_blue else -d, e["params"], e["platt"]))
    p_blue = float(sigmoid(blue_logit))
    return (p_blue if a_blue else 1 - p_blue), blue_logit


def series_odds(lookups, a, b, best_of, score=(0, 0), a_blue=None, p_next=None):
    """(P(A wins the series), score distribution) from the current score.

    a_blue is A's side in the next game. p_next replaces A's win probability
    in that game only. Later games use Elo with the loser picking side.
    """
    e = lookups["elo"]
    d, s = v3.effective(elo_diff(lookups, a, b), e["params"], e["platt"])
    dist = score_distribution(d, s, best_of, a_blue, 400.0, None, tuple(score), p_next)
    need = best_of // 2 + 1
    return sum(p for (x, _), p in dist.items() if x == need), dist


def resolve_player(lookups, name, team=None):
    """Player key for a typed name. Prefers someone on this team, then whoever
    played most recently. An unknown name gets a fresh key, so the model falls
    back to its prior for that player."""
    name = (name or "").strip()
    found = lookups["players"].get(name.lower(), [])
    if not found:
        return f"name:{name}", False
    on_team = [p for p in found if p["team"] == team]
    best = max(on_team or found, key=lambda p: p["last"])
    return best["key"], True


def draft_game(lookups, model, a, b, a_blue, names_a, champs_a, names_b, champs_b):
    """Draft-adjusted probability that A wins this game, with the reasons.

    Returns None if there is no draft model.
    """
    if model is None:
        return None
    hp = model["hyperparams"]
    snap = lookups["snapshot"]
    teams = (a, b)
    keys, known, raw = {}, {}, {}
    for tag, team, names, champs in (("a", a, names_a, champs_a), ("b", b, names_b, champs_b)):
        resolved = [resolve_player(lookups, n, team) for n in names]
        keys[tag] = [k for k, _ in resolved]
        known[tag] = [ok for _, ok in resolved]
        raw[tag] = raw_side_from_snapshot(snap, keys[tag], champs, teams)

    p_elo, blue_logit = elo_game(lookups, a, b, a_blue)
    blue, red = ("a", "b") if a_blue else ("b", "a")
    X = feature_matrix(np.stack([raw[blue], raw[red]]), blue_logit,
                       hp["pc_shrink"], hp["window"], hp["champ_shrink"], model["features"])
    p_blue = float(predict_logistic(model, X)[0])
    z = (X - np.array(model["mean"])) / np.array(model["std"])
    sign = 1.0 if a_blue else -1.0
    # Contribution of each feature to A's logit, measured from an average game.
    parts = {n: sign * float(c * v) for n, c, v in zip(model["features"], model["coef"], z)}

    tables = {}
    for tag, names, champs in (("a", names_a, champs_a), ("b", names_b, champs_b)):
        pt = player_table(raw[tag], hp["pc_shrink"])
        ct = champ_table(raw[tag], hp["window"], hp["champ_shrink"])
        tables[tag] = [{
            "role": POSITIONS[i], "player": names[i], "champion": champs[i],
            "known": known[tag][i],
            "games": int(pt["games_on_champ"][i]), "wins": int(pt["wins_on_champ"][i]),
            "shrunk_wr": float(pt["shrunk_wr"][i]), "overall_wr": float(pt["overall_wr"][i]),
            "player_games": int(pt["player_games"][i]),
            "champ_picks": int(ct["champ_picks"][i]), "champ_wr": float(ct["champ_wr"][i]),
        } for i in range(5)]
    return {"p": p_blue if a_blue else 1 - p_blue, "p_elo": p_elo, "parts": parts, "tables": tables,
            "window": hp["window"]}
