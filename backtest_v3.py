"""Elo v3. Major leagues plus international events, per-year home regions,
tuned on every 2022 to 2023 game, k_region checked and Platt scaled on 2024.

python3 backtest_v3.py data/*.csv

This script never prints a 2025 or 2026 score. Those years are scored once by
final_test.py after every model choice is frozen.
"""
import argparse
import itertools
import json
from dataclasses import asdict, replace

import numpy as np

import v3
from config import FIT_YEARS, MAJOR_LEAGUES, SELECT_YEARS, V3_INTERNATIONAL, V3_LEAGUES
from data import load_games, load_games_v3, load_team_rows
from elo import EloParams, estimate_side_adv, run, run_v3
from evalkit import bootstrap_line, calibration_table, log_loss, metrics_line, sigmoid

GRID = {
    "k_team": [8, 12, 16, 20, 24, 28, 32, 40],
    "k_region": [0, 4, 8, 12, 16, 24, 32],
    "season_regress": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "region_regress": [0.0, 0.1, 0.2, 0.3, 0.5],
}
# Natural limits. A grid is never widened past these.
BOUNDS = {"k_team": (1, 200), "k_region": (0, 400), "season_regress": (0.0, 1.0), "region_regress": (0.0, 1.0)}


def grid_search(games, homes, side_adv, score_mask, grid, cache):
    y = games["blue_win"].to_numpy()[score_mask]
    best, best_params = None, None
    names = list(grid)
    for combo in itertools.product(*(grid[n] for n in names)):
        if combo not in cache:
            params = EloParams(side_adv=side_adv, **dict(zip(names, combo)))
            _, probs, _ = run_v3(games, params, homes)
            cache[combo] = probs
        s = log_loss(cache[combo][score_mask], y)
        if best is None or s < best - 1e-12:
            best, best_params = s, dict(zip(names, combo))
    return best_params, best


def widen(grid, best):
    """Extend any grid whose best value sits on an edge that is not a natural limit."""
    changed = False
    for name, values in grid.items():
        lo, hi = BOUNDS[name]
        step_hi = values[-1] - values[-2]
        step_lo = values[1] - values[0]
        if best[name] == values[-1] and values[-1] < hi:
            grid[name] = values + [round(min(hi, values[-1] + step_hi * (i + 1)), 4) for i in range(2)]
            grid[name] = sorted(set(grid[name]))
            changed = True
        elif best[name] == values[0] and values[0] > lo:
            grid[name] = sorted(set([round(max(lo, values[0] - step_lo * (i + 1)), 4) for i in range(2)] + values))
            changed = True
    return changed


def tune(games, homes, side_adv, score_mask, out):
    grid = {k: list(v) for k, v in GRID.items()}
    cache = {}
    for round_no in range(1, 8):
        best, score = grid_search(games, homes, side_adv, score_mask, grid, cache)
        out(f"  round {round_no}  best {best}  log loss {score:.5f}  ({len(cache)} runs so far)")
        if not widen(grid, best):
            break
        out(f"  best value on a grid edge, widened to {grid}")
    edges = [n for n, v in grid.items() if best[n] in (v[0], v[-1])]
    for n in edges:
        out(f"  note  {n} = {best[n]} is at its natural limit {BOUNDS[n]}")
    return EloParams(side_adv=side_adv, **best), grid, cache


def v1_probs_for(paths, gameids):
    """The committed v1 Elo (params.json, all leagues, one home per team) on the same games."""
    games, _, home = load_games(paths)
    with open("params.json") as f:
        params = EloParams(**json.load(f))
    _, probs, _ = run(games, params, home)
    lookup = dict(zip(games["gameid"], probs))
    return np.array([lookup[g] for g in gameids])


def section(out, title, models, y, mask, bins):
    out(f"\n{title}")
    for name, p in models.items():
        out(metrics_line(name, p[mask], y[mask]))
    out("  calibration (bins of predicted blue win probability)")
    out(calibration_table({n: p[mask] for n, p in models.items()}, y[mask], bins=bins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", default="params_v3.json")
    ap.add_argument("--report", default="results/v3_backtest.txt")
    args = ap.parse_args()

    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    team_rows = load_team_rows(args.csvs)
    games, series, homes = load_games_v3(team_rows=team_rows)
    y = games["blue_win"].to_numpy()
    year = games["year"]
    out("Elo v3 backtest")
    out(f"{len(games)} games in the v3 set from {games['date'].min():%Y-%m-%d} to {games['date'].max():%Y-%m-%d}")
    out("fit on 2022 to 2023, calibrated and compared on 2024. 2025 and 2026 are not scored here.")

    out("\nleague codes found in the data (games by calendar year)")
    all_codes = set(team_rows["league"].unique())
    missing = sorted(V3_LEAGUES - all_codes)
    out(f"  codes in config.py missing from the data  {missing if missing else 'none'}")
    table = games.groupby(["league", "year"]).size().unstack(fill_value=0)
    table["kind"] = ["international" if c in V3_INTERNATIONAL else "major" for c in table.index]
    out("  " + table.to_string().replace("\n", "\n  "))

    fit_mask = year.isin(FIT_YEARS).to_numpy()
    sel_mask = year.isin(SELECT_YEARS).to_numpy()
    fit_games = games[fit_mask].reset_index(drop=True)
    side_adv = estimate_side_adv(fit_games)
    out(f"\nblue side advantage from 2022 to 2023 game 1s  {side_adv:.1f} Elo")

    out(f"\ngrid search, full grid, log loss on all {fit_mask.sum()} games of 2022 to 2023")
    out(f"  starting grid {GRID}")
    params, grid, cache = tune(fit_games, homes, side_adv, np.ones(len(fit_games), dtype=bool), out)
    out(f"  final grid {grid}")
    out(f"  tuned params {asdict(params)}")

    # Diagnostic only. Ratings start cold in January 2022, which could favour a large k.
    warm = (fit_games["year"] == 2023).to_numpy()
    names = list(grid)
    warm_best = min(cache, key=lambda c: log_loss(cache[c][warm], fit_games["blue_win"].to_numpy()[warm]))
    out(f"  check  best params if only 2023 is scored (2022 as warm up)  {dict(zip(names, warm_best))}")

    # k_region is the one setting the fit years cannot pin down. Every region
    # offset starts at zero in January 2022, so 2022 to 2023 rewards a very
    # large k_region just to get the offsets off the ground. 2024 is the first
    # year that starts warm, so it gets to veto that value under the usual rule.
    xr = games["cross_region"].to_numpy()
    out(f"\nk_region check on 2024 (other settings fixed at the tuned values)")
    out(f"  the tuned k_region {params.k_region} was fit while region offsets were still climbing from zero")
    by_kr = {}
    for kr in grid["k_region"]:
        _, pr, _ = run_v3(games, replace(params, k_region=kr), homes)
        by_kr[kr] = pr
        tag = "  tuned on 2022 to 2023" if kr == params.k_region else ""
        out(f"  k_region {kr:3d}  2024 cross-region games n={int((sel_mask & xr).sum())} "
            f"log loss {log_loss(pr[sel_mask & xr], y[sel_mask & xr]):.4f}  "
            f"2024 all games log loss {log_loss(pr[sel_mask], y[sel_mask]):.4f}{tag}")
    best_kr = min(by_kr, key=lambda kr: log_loss(by_kr[kr][sel_mask], y[sel_mask]))
    tuned_kr = params.k_region
    if best_kr != tuned_kr:
        line = bootstrap_line(f"k_region {best_kr} vs tuned {tuned_kr}, 2024 all games",
                              by_kr[best_kr][sel_mask], by_kr[tuned_kr][sel_mask], y[sel_mask])
        out(line)
        out(bootstrap_line(f"k_region {best_kr} vs tuned {tuned_kr}, 2024 cross-region",
                           by_kr[best_kr][sel_mask & xr], by_kr[tuned_kr][sel_mask & xr], y[sel_mask & xr]))
        if "better than noise" in line:
            params = replace(params, k_region=best_kr)
            out(f"  decision  k_region {best_kr} replaces {tuned_kr}. It wins on 2024 by more than noise.")
            out("  caveat    it was picked as the best of the grid on 2024, so the interval above is a little optimistic.")
        else:
            out(f"  decision  keep the tuned k_region {tuned_kr}. The 2024 gain is within noise.")
    else:
        out("  decision  2024 agrees with the tuned value.")
    out(f"  final params {asdict(params)}")

    model, raw, diffs = run_v3(games, params, homes)
    z = v3.raw_logit(diffs, params)
    a, b = v3.fit_platt(z[sel_mask], y[sel_mask])
    platt = {"a": a, "b": b}
    cal = sigmoid(a + b * z)
    out(f"\nPlatt scaling fit on all {sel_mask.sum()} games of 2024  p = sigmoid({a:+.4f} + {b:.4f} * elo_logit)")
    out("  a slope under 1 means raw Elo was overconfident")

    v1 = v1_probs_for(args.csvs, games["gameid"])
    models = {"Elo v1 (baseline)": v1, "Elo v3 raw": raw, "Elo v3 + Platt": cal}
    intl = (games["intl_event"] & games["cross_region"]).to_numpy()
    out("\nnotes on reading the tables")
    out("  Elo v1 was tuned on 2022 to 2024 international games, so it has already seen 2024.")
    out("  Elo v3 + Platt is in sample on 2024 for its two Platt numbers and for k_region.")
    out("  all major-league games means the whole v3 set, domestic and international.")

    for label, m in (("2022 to 2023 (fit years)", fit_mask), ("2024 (selection year)", sel_mask)):
        section(out, f"{label}, cross-region international games", models, y, m & intl, bins=5)
        section(out, f"{label}, all major-league games", models, y, m, bins=10)

    out("\npaired bootstrap over games on 2024, log loss difference (negative favours the first model)")
    for name, m in (("cross-region international", sel_mask & intl), ("all major-league games", sel_mask)):
        out(bootstrap_line(f"v3 raw vs v1, {name}", raw[m], v1[m], y[m]))
        out(bootstrap_line(f"v3 + Platt vs v1, {name}", cal[m], v1[m], y[m]))
        out(bootstrap_line(f"v3 + Platt vs v3 raw, {name}", cal[m], raw[m], y[m]))

    out(f"\nregion offsets after the last game in the data ({games['date'].max():%Y-%m-%d})")
    latest = int(year.max())
    active = set(games.loc[year == latest, "blue"]) | set(games.loc[year == latest, "red"])
    counts = {}
    for t in active:
        counts[model.home.get(t)] = counts.get(model.home.get(t), 0) + 1
    for r, v in sorted(model.region.items(), key=lambda kv: -kv[1]):
        out(f"  {r:<8} {v:+7.0f}   teams active in {latest}  {counts.get(r, 0)}")
    out(f"\ntop 10 teams active in {latest} (team rating plus region offset)")
    for t in sorted(active, key=model.strength, reverse=True)[:10]:
        out(f"  {t:<26} {model.home.get(t):<6} team {model.team[t]:+6.0f}  region {model.region_offset(t):+6.0f}  "
            f"total {model.strength(t):+6.0f}")

    v3.save_params(args.out, params, platt, {
        "fit_years": FIT_YEARS, "platt_fit_years": SELECT_YEARS,
        "leagues": sorted(V3_LEAGUES), "grid": grid,
        "k_region_tuned_on_fit_years": tuned_kr, "k_region_chosen_on": "2024" if params.k_region != tuned_kr else "2022 to 2023",
    })
    out(f"\nsaved {args.out}")
    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {args.report}")


if __name__ == "__main__":
    main()
