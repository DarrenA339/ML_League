"""Turn Oracle's Elixir CSVs into one row per game and one row per series."""
import hashlib
import os

import pandas as pd

from config import (INTERNATIONAL_LEAGUES, LEAGUE_ALIASES, MAJOR_LEAGUES, MINOR_REGION,
                    V3_INTERNATIONAL, V3_LEAGUES, V3_REGION_ALIASES)

# Elo only needs results, so LPL rows marked datacompleteness=partial are fine here.
NEEDED = {"gameid", "league", "date", "game", "side", "position", "teamname", "result"}
OPTIONAL = {"gamelength"}
PLAYER_COLS = NEEDED | OPTIONAL | {"playername", "playerid", "champion"}


def load_rows(paths):
    """Every row (team and player) with the columns any model here needs."""
    paths = sorted(paths)
    cache = _cache_path(paths)
    if cache and os.path.exists(cache):
        return pd.read_pickle(cache)
    frames = []
    for path in paths:
        df = pd.read_csv(path, usecols=lambda c: c in PLAYER_COLS, low_memory=False)
        missing = NEEDED - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        frames.append(df)
    rows = pd.concat(frames, ignore_index=True)
    if cache:
        rows.to_pickle(cache)
    return rows


def _cache_path(paths):
    """Slim copy of the CSVs next to them, keyed on name, size and modified
    time, so a fresh download is always re-read. Only used inside data/."""
    folder = os.path.dirname(os.path.abspath(paths[0]))
    if os.path.basename(folder) != "data":
        return None
    key = "|".join(f"{os.path.basename(p)}:{os.path.getsize(p)}:{os.path.getmtime(p)}" for p in paths)
    key += "|" + ",".join(sorted(PLAYER_COLS))
    return os.path.join(folder, f".rows_{hashlib.md5(key.encode()).hexdigest()[:12]}.pkl")


def load_team_rows(paths):
    rows = load_rows(paths)
    return rows[rows["position"] == "team"].reset_index(drop=True)


def build_games(team_rows, aliases=LEAGUE_ALIASES):
    df = team_rows.dropna(subset=["gameid", "teamname", "side", "result", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["game"] = pd.to_numeric(df["game"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["gameid", "side"])

    blue = df[df["side"] == "Blue"].set_index("gameid")
    red = df[df["side"] == "Red"].set_index("gameid")
    keep = ["league", "date", "game", "teamname", "result"]
    if "gamelength" in df.columns:
        keep.append("gamelength")
    games = blue[keep].join(red[["teamname"]], rsuffix="_red", how="inner")
    games = games.rename(columns={"teamname": "blue", "teamname_red": "red", "result": "blue_win"})
    games = games[games["blue"] != games["red"]].reset_index()

    games["league"] = games["league"].replace(aliases)
    games["year"] = games["date"].dt.year
    games["blue_win"] = games["blue_win"].astype(int)
    return games.sort_values(["date", "gameid"]).reset_index(drop=True)


def home_leagues(games):
    """Each team's region is the domestic league it played the most games in."""
    long = pd.concat([
        games[["blue", "league"]].rename(columns={"blue": "team"}),
        games[["red", "league"]].rename(columns={"red": "team"}),
    ])
    domestic = long[~long["league"].isin(INTERNATIONAL_LEAGUES)]
    return domestic.groupby("team")["league"].agg(lambda s: s.value_counts().index[0]).to_dict()


def tag_regions(games, home):
    games = games.copy()
    games["blue_region"] = games["blue"].map(home)
    games["red_region"] = games["red"].map(home)
    games["cross_region"] = (games["blue_region"].notna() & games["red_region"].notna()
                             & (games["blue_region"] != games["red_region"]))
    games["intl_event"] = games["league"].isin(INTERNATIONAL_LEAGUES)
    return games


def tag_series(games, max_gap_hours=12):
    """Group consecutive games between the same two teams into series."""
    g = games.copy()
    g["pair"] = [" vs ".join(sorted(p)) for p in zip(g["blue"], g["red"])]
    g = g.sort_values(["pair", "date"])
    by_pair = g.groupby("pair", sort=False)
    first = by_pair.cumcount() == 0
    gap = by_pair["date"].diff() > pd.Timedelta(hours=max_gap_hours)
    restart = by_pair["game"].diff() < 0
    g["series_id"] = (first | gap | restart).cumsum()
    g = g.sort_values(["date", "gameid"]).reset_index(drop=True)
    g["game_in_series"] = g.groupby("series_id").cumcount() + 1
    return g


def series_table(games):
    """One row per completed Bo3 or Bo5. Team A is whoever was on blue in game 1."""
    g = games.reset_index().rename(columns={"index": "row"})
    first = g.groupby("series_id").first()
    g["a"] = g["series_id"].map(first["blue"])
    g["a_won_game"] = ((g["blue"] == g["a"]) == (g["blue_win"] == 1)).astype(int)
    agg = g.groupby("series_id").agg(a_wins=("a_won_game", "sum"), n=("a_won_game", "size"))
    s = first[["row", "blue", "red", "league", "year", "intl_event", "cross_region"]].join(agg)
    s = s.rename(columns={"row": "first_row", "blue": "a", "red": "b"})
    s["b_wins"] = s["n"] - s["a_wins"]
    need = s[["a_wins", "b_wins"]].max(axis=1)
    s = s[need.isin([2, 3]) & (s["a_wins"] != s["b_wins"])].copy()
    s["best_of"] = 2 * s[["a_wins", "b_wins"]].max(axis=1) - 1
    s["a_won"] = (s["a_wins"] > s["b_wins"]).astype(int)
    return s.reset_index()


def load_games(paths):
    games = build_games(load_team_rows(paths))
    home = home_leagues(games)
    games = tag_series(tag_regions(games, home))
    series = series_table(games)
    games["best_of"] = games["series_id"].map(series.set_index("series_id")["best_of"]).fillna(1).astype(int)
    return games, series, home


# ---- v3 ----

def homes_by_year(games):
    """{year: {team: region}} with the home league decided separately each year.

    A team's home in a year is the major league it played the most games in
    that year. A team with no major league games that year keeps last year's
    home, and a team that has never had one is a MINOR_REGION wildcard. This
    uses the schedule only, never results.
    """
    long = pd.concat([
        games[["year", "blue", "league"]].rename(columns={"blue": "team"}),
        games[["year", "red", "league"]].rename(columns={"red": "team"}),
    ])
    domestic = long[long["league"].isin(MAJOR_LEAGUES)]
    counts = domestic.groupby(["year", "team", "league"]).size().reset_index(name="n")
    counts = counts.sort_values(["year", "team", "n", "league"], ascending=[True, True, False, True])
    top = counts.drop_duplicates(["year", "team"])
    out, current = {}, {}
    for year in sorted(games["year"].unique()):
        this = top[top["year"] == year]
        current = dict(current)
        for team, league in zip(this["team"], this["league"]):
            current[team] = V3_REGION_ALIASES.get(league, league)
        played = set(long.loc[long["year"] == year, "team"])
        for team in played:
            current.setdefault(team, MINOR_REGION)
        out[int(year)] = current
    return out


def add_end_times(games):
    """end = when the result is known. Start plus game length, capped at the
    next start of either team so a series always sees its own earlier games."""
    g = games
    length = pd.to_timedelta(pd.to_numeric(g.get("gamelength"), errors="coerce"), unit="s") \
        if "gamelength" in g.columns else pd.Series(pd.NaT, index=g.index)
    end = g["date"] + length.fillna(pd.Timedelta(minutes=30))
    long = pd.concat([
        pd.DataFrame({"row": g.index, "team": g["blue"], "date": g["date"]}),
        pd.DataFrame({"row": g.index, "team": g["red"], "date": g["date"]}),
    ]).sort_values(["team", "date", "row"])
    long["next"] = long.groupby("team")["date"].shift(-1)
    nxt = long.groupby("row")["next"].min().reindex(g.index)
    g = g.copy()
    g["end"] = end.where(nxt.isna() | (end <= nxt), nxt)
    return g


def load_games_v3(paths=None, team_rows=None):
    """Major leagues plus international events only, with per-year home regions."""
    if team_rows is None:
        team_rows = load_team_rows(paths)
    games = build_games(team_rows, aliases={})
    games = games[games["league"].isin(V3_LEAGUES)].reset_index(drop=True)
    homes = homes_by_year(games)
    home_of = lambda col: [homes[y].get(t) for y, t in zip(games["year"], games[col])]
    games["blue_region"], games["red_region"] = home_of("blue"), home_of("red")
    games["cross_region"] = games["blue_region"] != games["red_region"]
    games["intl_event"] = games["league"].isin(V3_INTERNATIONAL)
    games = add_end_times(tag_series(games))
    series = series_table(games)
    games["best_of"] = games["series_id"].map(series.set_index("series_id")["best_of"]).fillna(1).astype(int)
    return games, series, homes
