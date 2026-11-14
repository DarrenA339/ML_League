"""Write fake Oracle's Elixir CSVs with known team and region strengths.

Lets you test the pipeline end to end before downloading real data, and the
tests use it to check the model recovers strengths it should.

python make_fake_data.py fake_data
python backtest.py fake_data/*.csv --tune-years 2021 2022 2023 --test-years 2024 2025
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from elo import win_prob

TRUE_REGION = {"LCK": 100.0, "LPL": 80.0, "LEC": 0.0, "LCS": -60.0}
TRUE_SIDE_ADV = 25.0
POSITIONS = ["top", "jng", "mid", "bot", "sup"]


class FakeLeague:
    def __init__(self, seed=0, teams_per_region=8):
        self.rng = np.random.default_rng(seed)
        self.offset = dict(TRUE_REGION)
        self.skill, self.league = {}, {}
        for r in TRUE_REGION:
            for i in range(teams_per_region):
                name = f"{r} Squad {i}"
                self.skill[name] = self.rng.normal(0, 60)
                self.league[name] = r
        self.rows, self.gid, self.clock = [], 0, None

    def strength(self, t):
        return self.skill[t] + self.offset[self.league[t]]

    def play_game(self, blue, red, league, game_no):
        p = win_prob(self.strength(blue) - self.strength(red) + TRUE_SIDE_ADV)
        blue_win = int(self.rng.random() < p)
        self.gid += 1
        date = self.clock.strftime("%Y-%m-%d %H:%M:%S")
        for side, team, res in (("Blue", blue, blue_win), ("Red", red, 1 - blue_win)):
            base = dict(gameid=f"SYN-{self.gid}", datacompleteness="complete", league=league,
                        year=self.clock.year, date=date, game=game_no, patch=14.1, side=side,
                        teamname=team, result=res, gamelength=1800, golddiffat15=0)
            for pos in POSITIONS:
                self.rows.append({**base, "position": pos, "playername": f"{team} {pos}"})
            self.rows.append({**base, "position": "team", "playername": None})
        self.clock += pd.Timedelta(minutes=45)
        return blue_win

    def play_series(self, a, b, league, best_of):
        need, wins = best_of // 2 + 1, {a: 0, b: 0}
        blue, red = (a, b) if self.rng.random() < 0.5 else (b, a)
        game_no = 1
        while max(wins.values()) < need:
            blue_won = self.play_game(blue, red, league, game_no)
            winner, loser = (blue, red) if blue_won else (red, blue)
            wins[winner] += 1
            blue, red = loser, winner  # loser takes blue
            game_no += 1
        self.clock += pd.Timedelta(hours=8)
        return a if wins[a] == need else b

    def event(self, league, teams, best_of):
        for i, a in enumerate(teams):
            for b in teams[i + 1:]:
                if self.league[a] != self.league[b]:
                    self.play_series(a, b, league, best_of)

    def season(self, year):
        for t in self.skill:
            self.skill[t] = 0.7 * self.skill[t] + self.rng.normal(0, 40)
        for r in self.offset:
            self.offset[r] += self.rng.normal(0, 10)

        self.clock = pd.Timestamp(f"{year}-01-15 08:00")
        standings = {}
        for r in TRUE_REGION:
            teams = [t for t in self.skill if self.league[t] == r]
            wins = dict.fromkeys(teams, 0)
            for _ in range(2):
                for i, a in enumerate(teams):
                    for b in teams[i + 1:]:
                        wins[self.play_series(a, b, r, 1)] += 1
            standings[r] = sorted(teams, key=lambda t: -wins[t])

        self.clock = pd.Timestamp(f"{year}-05-01 08:00")
        self.event("MSI", [t for r in TRUE_REGION for t in standings[r][:2]], 3)
        self.clock = pd.Timestamp(f"{year}-10-10 08:00")
        self.event("WLDs", [t for r in TRUE_REGION for t in standings[r][:3]], 5)


def write_fake_data(out_dir, years=range(2021, 2026), seed=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim, paths = FakeLeague(seed), []
    for year in years:
        sim.rows = []
        sim.season(year)
        path = out_dir / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
        pd.DataFrame(sim.rows).to_csv(path, index=False)
        paths.append(str(path))
    return paths


if __name__ == "__main__":
    for p in write_fake_data(sys.argv[1] if len(sys.argv) > 1 else "fake_data"):
        print(p)
