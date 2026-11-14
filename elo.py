"""Team Elo plus a per-region offset that only moves on cross-region games.

A team's strength is team[t] + region[home(t)]. In a same-region game the
region terms cancel, so domestic play only sorts teams within a region. MSI,
EWC and Worlds are the only games that move the LCK vs LPL gap.
"""
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EloParams:
    k_team: float = 24.0
    k_region: float = 8.0
    side_adv: float = 20.0         # Elo points for being on blue side
    season_regress: float = 0.25   # pull teams toward their region average each new year
    region_regress: float = 0.15   # pull region offsets toward 0 each new year
    scale: float = 400.0


def win_prob(diff, scale=400.0):
    return 1.0 / (1.0 + 10.0 ** (-diff / scale))


def estimate_side_adv(games, scale=400.0):
    # Only game 1s. Later in a series the previous loser picks blue, so weaker
    # teams sit on blue more often and raw blue win rate understates the edge.
    if "game_in_series" in games:
        games = games[games["game_in_series"] == 1]
    p = games["blue_win"].mean()
    return scale * math.log10(p / (1 - p))


class Elo:
    def __init__(self, params, home):
        self.p = params
        self.home = home
        self.team = defaultdict(float)
        self.region = defaultdict(float)
        self.year = None
        self.active = set()

    def region_offset(self, team):
        r = self.home.get(team)
        return self.region.get(r, 0.0) if r is not None else 0.0

    def strength(self, team):
        return self.team.get(team, 0.0) + self.region_offset(team)

    def diff(self, a, b):
        """A's neutral-side strength minus B's."""
        return self.strength(a) - self.strength(b)

    def p_game(self, a, b, a_side=None):
        d = self.diff(a, b)
        if a_side == "Blue":
            d += self.p.side_adv
        elif a_side == "Red":
            d -= self.p.side_adv
        return win_prob(d, self.p.scale)

    def start_year(self, year):
        if self.year is not None and year != self.year:
            self._new_season()
        self.year = year

    def _new_season(self):
        # Move each region's average active-team rating into the region offset
        # so "team" and "region" stay cleanly separated, then regress both.
        by_region = defaultdict(list)
        for t in self.active:
            by_region[self.home.get(t)].append(t)
        for region, teams in by_region.items():
            if region is None:
                continue
            mean = sum(self.team[t] for t in teams) / len(teams)
            self.region[region] += mean
            for t in self.team:
                if self.home.get(t) == region:
                    self.team[t] -= mean
        for t in self.team:
            self.team[t] *= 1 - self.p.season_regress
        for r in self.region:
            self.region[r] *= 1 - self.p.region_regress
        self.active = set()

    def set_homes(self, new_home):
        """Switch to a new team to region map without changing anyone's strength.

        A team that changes region keeps team plus region constant. A region
        seen for the first time starts at the average strength of the known
        teams moving into it, so its newcomers start at that level too.
        """
        movers = [t for t in self.team if new_home.get(t) != self.home.get(t)]
        strength = {t: self.strength(t) for t in movers}
        fresh = defaultdict(list)
        for t in movers:
            r = new_home.get(t)
            if r is not None and r not in self.region:
                fresh[r].append(strength[t])
        for r, vals in fresh.items():
            self.region[r] = sum(vals) / len(vals)
        self.home = new_home
        for t in movers:
            self.team[t] = strength[t] - self.region_offset(t)

    def update(self, blue, red, blue_win, p=None):
        if p is None:
            p = self.p_game(blue, red, "Blue")
        delta = blue_win - p
        self.team[blue] += self.p.k_team * delta
        self.team[red] -= self.p.k_team * delta
        rb, rr = self.home.get(blue), self.home.get(red)
        if rb is not None and rr is not None and rb != rr:
            self.region[rb] += self.p.k_region * delta
            self.region[rr] -= self.p.k_region * delta
        self.active.update((blue, red))
        return p


def run(games, params, home):
    """Replay every game in date order.

    Returns the fitted model, each game's pre-game blue win probability, and
    each game's pre-game neutral diff (blue minus red). Every prediction only
    uses earlier games, so the probabilities are out of sample by construction.
    """
    model = Elo(params, home)
    n = len(games)
    probs, diffs = np.empty(n), np.empty(n)
    rows = zip(games["year"], games["blue"], games["red"], games["blue_win"])
    for i, (year, blue, red, win) in enumerate(rows):
        model.start_year(year)
        diffs[i] = model.diff(blue, red)
        probs[i] = model.update(blue, red, win)
    return model, probs, diffs


def run_v3(games, params, homes):
    """Replay with per-year home regions and strict finish times.

    homes is {year: {team: region}}. A game's result only reaches the ratings
    once its end time has passed, so two overlapping games never see each
    other. Needs the end column from data.add_end_times.
    """
    from evalkit import release_order

    model = Elo(params, {})
    n = len(games)
    probs, diffs = np.empty(n), np.empty(n)
    years = games["year"].to_numpy()
    blue, red = games["blue"].to_numpy(), games["red"].to_numpy()
    win = games["blue_win"].to_numpy()
    start, end = games["date"].to_numpy(), games["end"].to_numpy()
    applied = np.zeros(n, dtype=bool)
    for i, finished in release_order(start, end):
        for j in finished:
            model.update(blue[j], red[j], win[j], probs[j])
            applied[j] = True
        if years[i] != model.year:
            model.start_year(years[i])
            model.set_homes(homes[int(years[i])])
        diffs[i] = model.diff(blue[i], red[i])
        probs[i] = model.p_game(blue[i], red[i], "Blue")
    for j in sorted(np.flatnonzero(~applied), key=lambda j: end[j]):
        model.update(blue[j], red[j], win[j], probs[j])
    return model, probs, diffs
