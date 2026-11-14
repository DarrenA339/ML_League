"""Predict one series.

python predict.py data/*.csv --a "Gen.G" --b "Bilibili Gaming" --bo 5
"""
import argparse
import difflib
import json
import os

import pandas as pd

from data import load_games
from elo import EloParams, run
from series import score_distribution


def resolve(name, known):
    if name in known:
        return name
    close = difflib.get_close_matches(name, known, n=5, cutoff=0.4)
    raise SystemExit(f"No team named {name!r} in the data. Closest names {close}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--bo", type=int, default=5)
    ap.add_argument("--a-game1-side", choices=["blue", "red"],
                    help="leave out if game 1 sides are not announced yet")
    ap.add_argument("--before", help="only use games before this date, e.g. 2026-11-14")
    ap.add_argument("--params", default="params.json")
    args = ap.parse_args()

    games, _, home = load_games(args.csvs)
    if args.before:
        games = games[games["date"] < pd.Timestamp(args.before)].reset_index(drop=True)

    if os.path.exists(args.params):
        with open(args.params) as f:
            params = EloParams(**json.load(f))
    else:
        print(f"{args.params} not found, using default params. Run backtest.py first.\n")
        params = EloParams()

    model, _, _ = run(games, params, home)
    known = sorted(set(games["blue"]) | set(games["red"]))
    a, b = resolve(args.a, known), resolve(args.b, known)

    for t in (a, b):
        last = games.loc[(games["blue"] == t) | (games["red"] == t), "date"].max()
        print(f"{t:<20} region {home.get(t)}  last game in data {last:%Y-%m-%d}  "
              f"team {model.team[t]:+6.0f}  region offset {model.region_offset(t):+6.0f}  "
              f"total {model.strength(t):+6.0f}")

    d = model.diff(a, b)
    print(f"\nP({a} wins one game)  on blue {model.p_game(a, b, 'Blue'):.1%}  "
          f"on red {model.p_game(a, b, 'Red'):.1%}")

    a_blue = None if args.a_game1_side is None else args.a_game1_side == "blue"
    dist = score_distribution(d, params.side_adv, args.bo, a_blue, params.scale)
    need = args.bo // 2 + 1
    p_series = sum(p for (x, _), p in dist.items() if x == need)
    print(f"P({a} wins the Bo{args.bo})  {p_series:.1%}")
    for (x, y), p in sorted(dist.items(), key=lambda kv: -kv[1]):
        winner = a if x == need else b
        print(f"  {winner} {max(x, y)}-{min(x, y)}  {p:.1%}")

    print("\nregion offsets")
    for r, v in sorted(model.region.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {r:<8} {v:+6.0f}")

    latest = games["year"].max()
    active = set(games.loc[games["year"] == latest, "blue"]) | set(games.loc[games["year"] == latest, "red"])
    print(f"\ntop 10 teams active in {latest}")
    for t in sorted(active, key=model.strength, reverse=True)[:10]:
        print(f"  {t:<24} {model.strength(t):+6.0f}")


if __name__ == "__main__":
    main()
