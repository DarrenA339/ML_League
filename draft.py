"""Draft features and the single-game draft model.

Every feature for a game is built from games that had already finished when
it started. History comes from every league in the data (a rookie's academy
games count), but the model is only fit and scored on the v3 game set.

Features, all blue minus red, averaged over the five players of a team
  elo_logit   calibrated, side-adjusted Elo v3 logit
  pc_wr       player's win rate on this champion, shrunk toward the player's
              own overall win rate, minus that overall win rate
  pc_games    log(1 + player's games on this champion)
  ch_wr       champion's recent pro win rate shrunk toward 50%, minus 50%
  ch_pick     share of recent pro games the champion was picked in
  ch_wr_x     like ch_wr but leaving out every game either of these two teams
              played, so it cannot stand in for the teams' own recent form
"""
import bisect
from collections import defaultdict

import numpy as np
import pandas as pd

from data import add_end_times, build_games
from evalkit import release_order, sigmoid

POSITIONS = ["top", "jng", "mid", "bot", "sup"]
PLAYER_HORIZON_DAYS = 730          # player history looks back two years
CHAMP_WINDOWS = (30, 60, 90)       # candidate recent windows for champion form
PLAYER_PRIOR_GAMES = 20            # player overall win rate is shrunk toward 50% with this weight
DAY = np.timedelta64(1, "D")

FEATURES = ["elo_logit", "pc_wr", "pc_games", "ch_wr", "ch_pick"]
PLAYER_FEATURES = ["pc_wr", "pc_games"]
CHAMP_FEATURES = ["ch_wr", "ch_pick"]

# layout of the raw count array, last axis
N_PC, W_PC, N_P, W_P = 0, 1, 2, 3
PER_WINDOW = 5                            # then per window: picks, wins, games in window,
RAW_WIDTH = 4 + PER_WINDOW * len(CHAMP_WINDOWS)   # picks and wins without these two teams' games


def pair_key(a, b):
    return (a, b) if a <= b else (b, a)


def player_key(playerid, playername):
    if isinstance(playerid, str) and playerid:
        return playerid
    return f"name:{playername}"


def build_picks(rows):
    """One row per game from every league, with both lineups in role order.

    Columns gameid, date, end, blue, red, blue_win, league and the lists
    blue_players, blue_champs, red_players, red_champs (plus *_names).
    Games without exactly one player per role on each side are dropped.
    """
    games = build_games(rows[rows["position"] == "team"], aliases={})
    games = add_end_times(games)
    p = rows[rows["position"].isin(POSITIONS)].dropna(subset=["champion", "playername"])
    p = p.drop_duplicates(["gameid", "side", "position"]).copy()
    p["key"] = [player_key(i, n) for i, n in zip(p["playerid"], p["playername"])]
    p["position"] = pd.Categorical(p["position"], POSITIONS, ordered=True)
    p = p.sort_values(["gameid", "side", "position"])
    grouped = p.groupby(["gameid", "side"], observed=True).agg(
        players=("key", list), names=("playername", list), champs=("champion", list))
    grouped = grouped[grouped["players"].str.len() == 5]
    out = games[["gameid", "date", "end", "blue", "red", "blue_win", "league"]].copy()
    for side, tag in (("Blue", "blue"), ("Red", "red")):
        part = grouped.xs(side, level="side").reindex(out["gameid"])
        out[f"{tag}_players"] = part["players"].to_numpy()
        out[f"{tag}_names"] = part["names"].to_numpy()
        out[f"{tag}_champs"] = part["champs"].to_numpy()
    out = out[out["blue_players"].notna() & out["red_players"].notna()]
    return out.sort_values(["date", "gameid"]).reset_index(drop=True)


class _Tally:
    """Finish times and running wins for one key, appended in time order."""
    __slots__ = ("ends", "cum")

    def __init__(self):
        self.ends, self.cum = [], [0]

    def add(self, end, win):
        self.ends.append(end)
        self.cum.append(self.cum[-1] + win)

    def since(self, cutoff):
        i = bisect.bisect_left(self.ends, cutoff)
        return len(self.ends) - i, self.cum[-1] - self.cum[i]


class History:
    """Everything known from finished games. Only add() changes it."""

    def __init__(self):
        self.player = defaultdict(_Tally)
        self.player_champ = defaultdict(_Tally)
        self.champ = defaultdict(_Tally)
        self.champ_team = defaultdict(_Tally)   # (champion, a team in that game)
        self.champ_pair = defaultdict(_Tally)   # (champion, both teams in that game)
        self.games = _Tally()

    def add(self, end, players, champs, win, team=None, opponent=None):
        for p, c in zip(players, champs):
            self.player[p].add(end, win)
            self.player_champ[(p, c)].add(end, win)
            self.champ[c].add(end, win)
            if team is not None:
                self.champ_team[(c, team)].add(end, win)
                self.champ_team[(c, opponent)].add(end, win)
                self.champ_pair[(c, pair_key(team, opponent))].add(end, win)

    def add_game(self, end):
        self.games.add(end, 0)

    def _since(self, table, key, cutoff):
        return table[key].since(cutoff) if key in table else (0, 0)

    def raw_side(self, players, champs, now, teams=None):
        """Raw counts for one team at time now. Shape (5, RAW_WIDTH).

        teams is the two teams in the game being predicted. Their own games
        are what the last two columns of each window leave out.
        """
        out = np.zeros((5, RAW_WIDTH))
        horizon = now - PLAYER_HORIZON_DAYS * DAY
        for i, (p, c) in enumerate(zip(players, champs)):
            out[i, N_PC], out[i, W_PC] = self._since(self.player_champ, (p, c), horizon)
            out[i, N_P], out[i, W_P] = self._since(self.player, p, horizon)
            for w, days in enumerate(CHAMP_WINDOWS):
                cutoff = now - days * DAY
                b = 4 + PER_WINDOW * w
                n, wins = self._since(self.champ, c, cutoff)
                out[i, b], out[i, b + 1] = n, wins
                out[i, b + 2] = self.games.since(cutoff)[0]
                if teams is not None:
                    for t in teams:
                        tn, tw = self._since(self.champ_team, (c, t), cutoff)
                        n, wins = n - tn, wins - tw
                    pn, pw = self._since(self.champ_pair, (c, pair_key(*teams)), cutoff)
                    n, wins = n + pn, wins + pw
                out[i, b + 3], out[i, b + 4] = n, wins
        return out

    def snapshot(self, now):
        """Plain dict tables as of now, for the app. Same numbers raw_side gives."""
        horizon = now - PLAYER_HORIZON_DAYS * DAY
        keep = lambda table, cutoff: {k: v for k, v in ((k, t.since(cutoff)) for k, t in table.items()) if v[0]}
        return {
            "as_of": str(pd.Timestamp(now)),
            "player": keep(self.player, horizon),
            "player_champ": keep(self.player_champ, horizon),
            "champ": {days: keep(self.champ, now - days * DAY) for days in CHAMP_WINDOWS},
            "champ_team": {days: keep(self.champ_team, now - days * DAY) for days in CHAMP_WINDOWS},
            "champ_pair": {days: keep(self.champ_pair, now - days * DAY) for days in CHAMP_WINDOWS},
            "games": {days: self.games.since(now - days * DAY)[0] for days in CHAMP_WINDOWS},
        }


def raw_side_from_snapshot(snap, players, champs, teams=None):
    """raw_side for the app, read from the saved tables."""
    out = np.zeros((5, RAW_WIDTH))
    for i, (p, c) in enumerate(zip(players, champs)):
        out[i, N_PC], out[i, W_PC] = snap["player_champ"].get((p, c), (0, 0))
        out[i, N_P], out[i, W_P] = snap["player"].get(p, (0, 0))
        for w, days in enumerate(CHAMP_WINDOWS):
            b = 4 + PER_WINDOW * w
            n, wins = snap["champ"][days].get(c, (0, 0))
            out[i, b], out[i, b + 1] = n, wins
            out[i, b + 2] = snap["games"][days]
            if teams is not None:
                for t in teams:
                    tn, tw = snap["champ_team"][days].get((c, t), (0, 0))
                    n, wins = n - tn, wins - tw
                pn, pw = snap["champ_pair"][days].get((c, pair_key(*teams)), (0, 0))
                n, wins = n + pn, wins + pw
            out[i, b + 3], out[i, b + 4] = n, wins
    return out


def replay(picks, want=None):
    """Walk every game in start order and collect raw counts.

    Returns (raw, history). raw has shape (games, 2, 5, RAW_WIDTH) with side 0
    blue and side 1 red, and is all zero for games not in want. A game enters
    the history only once its end time has passed.
    """
    n = len(picks)
    start, end = picks["date"].to_numpy(), picks["end"].to_numpy()
    cols = [picks[c].to_numpy() for c in ("blue_players", "blue_champs", "red_players", "red_champs")]
    blue, red = picks["blue"].to_numpy(), picks["red"].to_numpy()
    win = picks["blue_win"].to_numpy()
    wanted = np.ones(n, dtype=bool) if want is None else np.asarray(want)
    raw = np.zeros((n, 2, 5, RAW_WIDTH))
    hist = History()
    applied = np.zeros(n, dtype=bool)

    def apply(j):
        hist.add(end[j], cols[0][j], cols[1][j], int(win[j]), blue[j], red[j])
        hist.add(end[j], cols[2][j], cols[3][j], 1 - int(win[j]), red[j], blue[j])
        hist.add_game(end[j])
        applied[j] = True

    for i, finished in release_order(start, end):
        for j in finished:
            apply(j)
        if wanted[i]:
            teams = (blue[i], red[i])
            raw[i, 0] = hist.raw_side(cols[0][i], cols[1][i], start[i], teams)
            raw[i, 1] = hist.raw_side(cols[2][i], cols[3][i], start[i], teams)
    for j in sorted(np.flatnonzero(~applied), key=lambda j: end[j]):
        apply(j)
    return raw, hist


def player_table(raw_side, pc_shrink):
    """Per player numbers behind the features, for one team. Shape (5,) each."""
    n_pc, w_pc, n_p, w_p = (raw_side[..., i] for i in (N_PC, W_PC, N_P, W_P))
    overall = (w_p + 0.5 * PLAYER_PRIOR_GAMES) / (n_p + PLAYER_PRIOR_GAMES)
    shrunk = (w_pc + pc_shrink * overall) / (n_pc + pc_shrink)
    return {"games_on_champ": n_pc, "wins_on_champ": w_pc, "player_games": n_p,
            "overall_wr": overall, "shrunk_wr": shrunk}


def champ_table(raw_side, window, champ_shrink):
    b = 4 + PER_WINDOW * CHAMP_WINDOWS.index(window)
    n_c, w_c, total, n_x, w_x = (raw_side[..., b + i] for i in range(PER_WINDOW))
    return {"champ_picks": n_c, "champ_wr": (w_c + 0.5 * champ_shrink) / (n_c + champ_shrink),
            "champ_pick_share": n_c / np.maximum(total, 1),
            "champ_picks_x": n_x, "champ_wr_x": (w_x + 0.5 * champ_shrink) / (n_x + champ_shrink)}


def side_features(raw_side, pc_shrink, window, champ_shrink):
    """Team level values for one side. Works on (5, W) or (games, 5, W)."""
    pt = player_table(raw_side, pc_shrink)
    ct = champ_table(raw_side, window, champ_shrink)
    return {
        "pc_wr": (pt["shrunk_wr"] - pt["overall_wr"]).mean(axis=-1),
        "pc_games": np.log1p(pt["games_on_champ"]).mean(axis=-1),
        "ch_wr": (ct["champ_wr"] - 0.5).mean(axis=-1),
        "ch_pick": ct["champ_pick_share"].mean(axis=-1),
        "ch_wr_x": (ct["champ_wr_x"] - 0.5).mean(axis=-1),
    }


def feature_matrix(raw, elo_logit, pc_shrink, window, champ_shrink, names=FEATURES):
    """Blue minus red features. raw is (games, 2, 5, W) or (2, 5, W)."""
    blue = side_features(raw[..., 0, :, :], pc_shrink, window, champ_shrink)
    red = side_features(raw[..., 1, :, :], pc_shrink, window, champ_shrink)
    cols = {k: blue[k] - red[k] for k in blue}
    cols["elo_logit"] = np.asarray(elo_logit, dtype=float)
    return np.stack([np.asarray(cols[n], dtype=float) for n in names], axis=-1)


def fit_logistic(X, y, names):
    """Logistic regression on standardized features, returned as a plain dict."""
    from sklearn.linear_model import LogisticRegression
    mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit((X - mean) / std, np.asarray(y).astype(int))
    return {"features": list(names), "mean": mean.tolist(), "std": std.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


def predict_logistic(model, X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    z = (X - np.array(model["mean"])) / np.array(model["std"])
    return sigmoid(z @ np.array(model["coef"]) + model["intercept"])


def build_dataset(rows, params, platt):
    """Everything the draft backtest, the final test and the artifact builder share.

    Runs Elo v3 over the v3 game set and the draft replay over every league,
    then lines the two up game by game.
    """
    import v3
    from data import load_games_v3
    from elo import run_v3

    games, series, homes = load_games_v3(team_rows=rows[rows["position"] == "team"])
    elo_model, _, diffs = run_v3(games, params, homes)
    games["elo_logit"] = v3.calibrated_logit(diffs, params, platt)
    games["elo_diff"] = diffs

    picks = build_picks(rows)
    in_set = picks["gameid"].isin(set(games["gameid"])).to_numpy()
    raw_all, hist = replay(picks, want=in_set)
    pos = pd.Series(np.arange(len(picks)), index=picks["gameid"])
    have = games["gameid"].isin(pos.index).to_numpy()
    dropped = int((~have).sum())
    games = games[have].reset_index(drop=True)
    idx = pos.loc[games["gameid"]].to_numpy()
    return {
        "games": games, "y": games["blue_win"].to_numpy(), "raw": raw_all[idx],
        "elo_logit": games["elo_logit"].to_numpy(), "elo_p": sigmoid(games["elo_logit"].to_numpy()),
        "picks": picks, "history": hist, "elo_model": elo_model, "homes": homes,
        "series": series, "dropped": dropped,
    }
