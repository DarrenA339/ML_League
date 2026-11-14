"""Tune Elo on older international events and score it on newer ones.

python backtest.py data/*.csv --tune-years 2022 2023 2024 --test-years 2025 2026
"""
import argparse
import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from data import load_games
from elo import EloParams, estimate_side_adv, run
from series import series_win_prob

GRID = {
    "k_team": [12, 18, 24, 32, 40, 56],
    "k_region": [0, 4, 8, 12, 20, 32],
    "season_regress": [0.1, 0.25, 0.4, 0.6],
    "region_regress": [0.0, 0.15, 0.3, 0.5],
}


def log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def report(name, p, y):
    acc = np.mean((p > 0.5) == (y == 1))
    print(f"{name:<34} n={len(y):5d}  log loss {log_loss(p, y):.4f}  "
          f"brier {brier(p, y):.4f}  acc {acc:.3f}")


def calibration(p, y, bins=5):
    print("\ncalibration on test games (blue win prob)")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    for b in range(bins):
        m = idx == b
        if m.any():
            print(f"  {edges[b]:.1f} to {edges[b + 1]:.1f}  n={m.sum():4d}  "
                  f"predicted {p[m].mean():.3f}  observed {y[m].mean():.3f}")


def tune(games, home, mask, side_adv, rounds=2):
    """Coordinate descent over GRID, minimizing log loss on the masked games."""
    y = games["blue_win"].to_numpy()[mask]

    def score(params):
        _, probs, _ = run(games, params, home)
        return log_loss(probs[mask], y)

    params = EloParams(side_adv=side_adv)
    best = score(params)
    for _ in range(rounds):
        for name, values in GRID.items():
            for v in values:
                cand = replace(params, **{name: v})
                s = score(cand)
                if s < best - 1e-9:
                    best, params = s, cand
    return params


def series_eval(series, diffs, params, years):
    s = series[series["intl_event"] & series["cross_region"] & series["year"].isin(years)]
    if s.empty:
        return
    p = np.array([series_win_prob(diffs[r], params.side_adv, bo, True, params.scale)
                  for r, bo in zip(s["first_row"], s["best_of"])])
    y = s["a_won"].to_numpy()
    print()
    for bo in (3, 5):
        m = (s["best_of"] == bo).to_numpy()
        if m.any():
            report(f"test  intl Bo{bo} series", p[m], y[m])


def game_number_check(games, probs, mask):
    g = games[mask].assign(p=probs[mask])
    g = g[g["best_of"] == 5]
    if g.empty:
        return
    g = g.assign(fav_p=np.maximum(g["p"], 1 - g["p"]),
                 fav_won=((g["p"] >= 0.5) == (g["blue_win"] == 1)).astype(float))
    print("\nBo5 favorite by game number (a gap that grows late hints fearless draft matters)")
    for n, grp in g.groupby("game_in_series"):
        print(f"  game {n}  n={len(grp):4d}  predicted {grp['fav_p'].mean():.3f}  "
              f"observed {grp['fav_won'].mean():.3f}")


def list_leagues(games):
    t = games.groupby("league").agg(games=("gameid", "size"), cross_region=("cross_region", "mean"))
    t["flagged_international"] = games.groupby("league")["intl_event"].first()
    print(t.sort_values("cross_region", ascending=False).head(25).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--tune-years", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--test-years", type=int, nargs="+", default=[2025, 2026])
    ap.add_argument("--out", default="params.json")
    ap.add_argument("--list-leagues", action="store_true")
    args = ap.parse_args()

    games, series, home = load_games(args.csvs)
    print(f"{len(games)} games from {games['date'].min():%Y-%m-%d} to {games['date'].max():%Y-%m-%d}")
    if args.list_leagues:
        list_leagues(games)
        return

    target = games["intl_event"] & games["cross_region"]
    tune_mask = (target & games["year"].isin(args.tune_years)).to_numpy()
    test_mask = (target & games["year"].isin(args.test_years)).to_numpy()
    if not tune_mask.any() or not test_mask.any():
        raise SystemExit("No cross-region international games in a window. Run with "
                         "--list-leagues and check INTERNATIONAL_LEAGUES in config.py.")

    side_adv = estimate_side_adv(games[games["year"].isin(args.tune_years)])
    params = tune(games, home, tune_mask, side_adv)
    print("tuned params", asdict(params))

    _, probs, diffs = run(games, params, home)
    y = games["blue_win"].to_numpy()
    same_region = (~games["cross_region"] & games["year"].isin(args.test_years)).to_numpy()
    print("coin flip log loss is 0.6931\n")
    report("tune  cross-region intl games", probs[tune_mask], y[tune_mask])
    report("test  cross-region intl games", probs[test_mask], y[test_mask])
    report("test  same-region games", probs[same_region], y[same_region])
    calibration(probs[test_mask], y[test_mask])
    series_eval(series, diffs, params, args.test_years)
    game_number_check(games, probs, test_mask)

    with open(args.out, "w") as f:
        json.dump(asdict(params), f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
