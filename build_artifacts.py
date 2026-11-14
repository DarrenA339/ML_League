"""Rebuild everything the app loads, from the CSVs.

python3 build_artifacts.py data/*.csv

Writes models/lookups.pkl (Elo ratings, rosters, champion list and the player
and champion tables as of the last game in the data) and
models/draft_model.json. Run it again after downloading fresh CSVs.

The draft model keeps the features and settings chosen in
models/draft_choice.json and is refit on every year in the data, so it has
seen recent metas. Pass --fit-years 2022 2023 for the exact backtested fit.
Without models/draft_choice.json only the Elo tables are written and the app
runs on Elo alone.
"""
import argparse
import json
import os
import pickle
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import v3
from data import load_rows
from draft import POSITIONS, build_dataset, feature_matrix, fit_logistic


def build_lookups(data, params, platt, as_of=None):
    """Tables as of a moment. as_of defaults to just after the last game."""
    games, picks, hist, elo = data["games"], data["picks"], data["history"], data["elo_model"]
    if as_of is None:
        as_of = (picks["end"].max() + pd.Timedelta(seconds=1)).to_numpy()

    last_played = pd.concat([games.groupby("blue")["date"].max(), games.groupby("red")["date"].max()])
    last_played = last_played.groupby(level=0).max().sort_values(ascending=False)
    cutoff = games["date"].max() - pd.Timedelta(days=730)
    teams = [t for t, d in last_played.items() if d >= cutoff]

    rosters, players = {}, {}
    for tag in ("blue", "red"):
        for team, keys, names, when in zip(picks[tag], picks[f"{tag}_players"], picks[f"{tag}_names"], picks["date"]):
            for key, name in zip(keys, names):
                slot = players.setdefault(name.lower(), {})
                if key not in slot or when >= slot[key]["last"]:
                    slot[key] = {"key": key, "name": name, "team": team, "last": when}
            if team not in rosters or when >= rosters[team][0]:
                rosters[team] = (when, list(names))
    known = set(elo.team)
    return {
        "as_of": str(pd.Timestamp(as_of)),
        "last_game": str(picks["date"].max()),
        "elo": {"params": asdict(params), "platt": platt, "team": dict(elo.team),
                "region": dict(elo.region), "home": dict(elo.home)},
        "teams": teams,
        "rosters": {t: names for t, (_, names) in rosters.items() if t in known},
        "champions": sorted({c for col in ("blue_champs", "red_champs") for cs in picks[col] for c in cs}),
        "players": {n: list(v.values()) for n, v in players.items()},
        "positions": POSITIONS,
        "snapshot": hist.snapshot(as_of),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--params", default="params_v3.json")
    ap.add_argument("--choice", default="models/draft_choice.json")
    ap.add_argument("--out", default="models")
    ap.add_argument("--fit-years", type=int, nargs="+", help="default is every year in the data")
    args = ap.parse_args()

    params, platt = v3.load_params(args.params)
    data = build_dataset(load_rows(args.csvs), params, platt)
    os.makedirs(args.out, exist_ok=True)

    lookups = build_lookups(data, params, platt)
    with open(os.path.join(args.out, "lookups.pkl"), "wb") as f:
        pickle.dump(lookups, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {args.out}/lookups.pkl  as of {lookups['as_of']}  "
          f"{len(lookups['teams'])} teams, {len(lookups['champions'])} champions, "
          f"{len(lookups['snapshot']['player_champ'])} player and champion pairs")

    model_path = os.path.join(args.out, "draft_model.json")
    if not os.path.exists(args.choice):
        if os.path.exists(model_path):
            os.remove(model_path)
        print(f"{args.choice} not found, so no draft model was written. The app will run on Elo alone. "
              "Run draft_backtest.py first to get one.")
        return
    with open(args.choice) as f:
        choice = json.load(f)
    hp, names = choice["hyperparams"], choice["features"]
    games, y = data["games"], data["y"]
    years = args.fit_years or sorted(int(v) for v in games["year"].unique())
    mask = games["year"].isin(years).to_numpy()
    X = feature_matrix(data["raw"], data["elo_logit"], hp["pc_shrink"], hp["window"], hp["champ_shrink"], names)
    model = fit_logistic(X[mask], y[mask], names)
    model.update({
        "status": choice["status"], "name": choice["name"], "hyperparams": hp,
        "fit_years": years, "fit_games": int(mask.sum()),
        "select_log_loss": choice["select_log_loss"],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    with open(model_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"saved {model_path}  {choice['name']} ({choice['status']})  fit on {years}, {mask.sum()} games")
    for n, c, sd in zip(names, model["coef"], model["std"]):
        print(f"  {n:<10} coef {c:+.4f}   one sd = {sd:.4f}")


if __name__ == "__main__":
    main()
