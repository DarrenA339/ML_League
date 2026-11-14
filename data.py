"""Turn Oracle's Elixir CSVs into one row per game and one row per series."""
import pandas as pd

from config import INTERNATIONAL_LEAGUES, LEAGUE_ALIASES

# Elo only needs results, so LPL rows marked datacompleteness=partial are fine here.
NEEDED = {"gameid", "league", "date", "game", "side", "position", "teamname", "result"}


def load_team_rows(paths):
    frames = []
    for path in paths:
        df = pd.read_csv(path, usecols=lambda c: c in NEEDED, low_memory=False)
        missing = NEEDED - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        frames.append(df[df["position"] == "team"])
    return pd.concat(frames, ignore_index=True)


def build_games(team_rows):
    df = team_rows.dropna(subset=["gameid", "teamname", "side", "result", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["game"] = pd.to_numeric(df["game"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["gameid", "side"])

    blue = df[df["side"] == "Blue"].set_index("gameid")
    red = df[df["side"] == "Red"].set_index("gameid")
    games = blue[["league", "date", "game", "teamname", "result"]].join(
        red[["teamname"]], rsuffix="_red", how="inner")
    games = games.rename(columns={"teamname": "blue", "teamname_red": "red", "result": "blue_win"})
    games = games[games["blue"] != games["red"]].reset_index()

    games["league"] = games["league"].replace(LEAGUE_ALIASES)
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
