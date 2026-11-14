"""Live series and draft predictions.

python3 -m streamlit run app.py

Loads the artifacts in models/ (python3 build_artifacts.py data/*.csv makes
them). Runs on Elo alone when there is no draft model.
"""
import csv
import os
from datetime import datetime

import pandas as pd
import streamlit as st

import live

MODELS_DIR = os.environ.get("WORLDS_ELO_MODELS", "models")
LOG_PATH = os.environ.get("WORLDS_ELO_LOG", os.path.join("predictions", "log.csv"))
ROLES = ["Top", "Jungle", "Mid", "Bot", "Support"]
FEARLESS = ["Off", "Own team", "Both teams"]
LOG_FIELDS = ["time", "team_a", "team_b", "best_of", "score_a", "score_b", "game_number", "team_a_side",
              "fearless", "team_a_players", "team_a_picks", "team_b_players", "team_b_picks",
              "elo_p_game", "draft_p_game", "elo_p_series", "draft_p_series", "draft_model", "data_through"]

st.set_page_config(page_title="Worlds Elo", layout="wide")


@st.cache_resource(show_spinner=False)
def load(folder, stamp):
    return live.load_artifacts(folder)


def artifact_stamp(folder):
    files = [os.path.join(folder, f) for f in (live.LOOKUPS, live.MODEL)]
    return tuple(os.path.getmtime(f) if os.path.exists(f) else 0 for f in files)


def pct(p):
    return f"{100 * p:.1f}%"


def points(delta):
    return f"{100 * delta:+.1f} points"


def score_table(a, b, columns):
    """columns is {heading: distribution}. One row per final score, A wins first."""
    scores = sorted({s for dist in columns.values() for s in dist}, key=lambda s: (s[0] < s[1], -s[0] + s[1]))
    rows = []
    for x, y in scores:
        winner = a if x > y else b
        row = {"Final score": f"{winner} {max(x, y)} to {min(x, y)}"}
        for heading, dist in columns.items():
            row[heading] = 100 * dist.get((x, y), 0.0)
        rows.append(row)
    config = {h: st.column_config.ProgressColumn(h, format="%.1f%%", min_value=0, max_value=100) for h in columns}
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config=config)


def player_frame(rows, window):
    return pd.DataFrame([{
        "Role": ROLES[i],
        "Player": r["player"] + ("" if r["known"] else " (new)"),
        "Champion": r["champion"],
        "Games on champion": r["games"],
        "Record": f"{r['wins']} W {r['games'] - r['wins']} L",
        "Shrunk win rate": 100 * r["shrunk_wr"],
        "Player overall": 100 * r["overall_wr"],
        f"Champion picks, {window} days": r["champ_picks"],
        f"Champion win rate, {window} days": 100 * r["champ_wr"],
    } for i, r in enumerate(rows)])


def append_log(row):
    folder = os.path.dirname(LOG_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


# ---------- artifacts ----------

if not os.path.exists(os.path.join(MODELS_DIR, live.LOOKUPS)):
    st.title("Worlds Elo")
    st.error("No model files found. Build them first, then reload this page.")
    st.code("python3 build_artifacts.py data/*.csv")
    st.stop()

lookups, model = load(MODELS_DIR, artifact_stamp(MODELS_DIR))
teams = sorted(lookups["teams"], key=str.lower)
champions = lookups["champions"]
data_through = lookups["last_game"][:10]


def default_index(name, fallback):
    return teams.index(name) if name in teams else fallback


# ---------- match setup ----------

st.title("Worlds Elo")
st.caption(f"Ratings and player history use games through {data_through}. "
           "Rebuild the model files after downloading newer data.")

with st.sidebar:
    st.header("Match")
    team_a = st.selectbox("Team A", teams, index=default_index("Gen.G", 0))
    team_b = st.selectbox("Team B", teams, index=default_index("Bilibili Gaming", min(1, len(teams) - 1)))
    best_of = st.radio("Best of", [1, 3, 5], index=2, horizontal=True)
    need = best_of // 2 + 1
    st.subheader("Series score so far")
    left, right = st.columns(2)
    score_a = left.number_input(f"{team_a} wins", min_value=0, max_value=need - 1, value=0, step=1,
                                key=f"score_a_{best_of}")
    score_b = right.number_input(f"{team_b} wins", min_value=0, max_value=need - 1, value=0, step=1,
                                 key=f"score_b_{best_of}")
    side = st.radio(f"{team_a} side this game", ["Blue", "Red"], horizontal=True)
    st.header("Fearless draft")
    fearless = st.radio("Hide champions used earlier in the series", FEARLESS, horizontal=True,
                        help="Own team hides only what that team already played. "
                             "Both teams hides every champion either team already played.")

if team_a == team_b:
    st.warning("Pick two different teams.")
    st.stop()

a_blue = side == "Blue"
score = (int(score_a), int(score_b))
game_number = sum(score) + 1
series_key = f"{team_a}|{team_b}|{best_of}"
used_keys = {"a": f"used_a_{series_key}", "b": f"used_b_{series_key}"}

if fearless != "Off":
    with st.sidebar:
        st.multiselect(f"{team_a} earlier picks", champions, key=used_keys["a"],
                       help="Logging a prediction adds that game's picks here.")
        st.multiselect(f"{team_b} earlier picks", champions, key=used_keys["b"])
used = {tag: list(st.session_state.get(key, [])) for tag, key in used_keys.items()}


def hidden_for(tag):
    if fearless == "Own team":
        return set(used[tag])
    if fearless == "Both teams":
        return set(used["a"]) | set(used["b"])
    return set()


# ---------- before the draft ----------

p_elo, _ = live.elo_game(lookups, team_a, team_b, a_blue)
p_series_elo, dist_elo = live.series_odds(lookups, team_a, team_b, best_of, score, a_blue)

st.subheader(f"Game {game_number} of a best of {best_of}, {team_a} on {side.lower()} side")
m1, m2, m3 = st.columns(3)
m1.metric(f"{team_a} wins this game, Elo", pct(p_elo))
m2.metric(f"{team_a} wins the series, Elo", pct(p_series_elo))
m3.metric(f"{team_b} wins the series, Elo", pct(1 - p_series_elo))
strength_a, strength_b = live.strength(lookups, team_a), live.strength(lookups, team_b)
st.caption(f"Elo v3 rating {team_a} {strength_a:+.0f}, {team_b} {strength_b:+.0f}. "
           "Later games assume the loser of each game takes blue side.")

# ---------- draft input ----------

st.subheader("Draft")
if model is None:
    st.info("No draft model file found, so this page runs on Elo alone. "
            "You can still enter picks and log them.")
elif model.get("status") != "adopted":
    st.warning("The draft number is experimental. It did not beat Elo by more than noise in the 2024 "
               "backtest, so treat Elo as the number of record.")

names, picks = {}, {}
columns = st.columns(2)
for col, tag, team in ((columns[0], "a", team_a), (columns[1], "b", team_b)):
    with col:
        team_side = side if tag == "a" else ("Red" if a_blue else "Blue")
        st.markdown(f"**{team}** ({team_side.lower()} side)")
        roster = lookups["rosters"].get(team, [""] * 5)
        options = [c for c in champions if c not in hidden_for(tag)]
        names[tag], picks[tag] = [], []
        for i, role in enumerate(ROLES):
            c1, c2 = st.columns([1, 1])
            names[tag].append(c1.text_input(f"{role} player", value=roster[i], key=f"player_{tag}_{team}_{i}"))
            champ_key = f"champ_{tag}_{series_key}_{game_number}_{i}"
            if st.session_state.get(champ_key) not in options:
                st.session_state.pop(champ_key, None)
            picks[tag].append(c2.selectbox(f"{role} champion", options, index=None, key=champ_key,
                                           placeholder="Type to search"))

chosen = [c for c in picks["a"] + picks["b"] if c]
complete = len(chosen) == 10
duplicates = sorted({c for c in chosen if chosen.count(c) > 1})
if duplicates:
    st.warning("Picked more than once in this game, " + ", ".join(duplicates) + ".")

# ---------- after the draft ----------

result, p_series_draft = None, None
if complete and model is not None:
    result = live.draft_game(lookups, model, team_a, team_b, a_blue,
                             names["a"], picks["a"], names["b"], picks["b"])
    p_series_draft, dist_draft = live.series_odds(lookups, team_a, team_b, best_of, score, a_blue,
                                                  p_next=result["p"])
    label = "draft adjusted" if model.get("status") == "adopted" else "experimental draft number"
    st.subheader("With this draft")
    d1, d2, d3 = st.columns(3)
    d1.metric(f"{team_a} wins this game, {label}", pct(result["p"]),
              delta=points(result["p"] - result["p_elo"]) + " vs Elo")
    d2.metric(f"{team_a} wins the series", pct(p_series_draft),
              delta=points(p_series_draft - p_series_elo) + " vs Elo")
    d3.metric(f"{team_b} wins the series", pct(1 - p_series_draft),
              delta=points(p_series_elo - p_series_draft) + " vs Elo")
    st.caption("The series number uses the draft probability for this game and Elo for every game after it.")

    st.markdown(f"**Where the change comes from, in points of {team_a} win probability**")
    why = pd.DataFrame([{"Piece": s["label"], "Points": s["points"]} for s in result["steps"]]
                       + [{"Piece": "Total change vs Elo", "Points": 100 * (result["p"] - result["p_elo"])}])
    st.dataframe(why, hide_index=True, width="content",
                 column_config={"Points": st.column_config.NumberColumn(format="%+.2f")})
    st.caption("Recalibration is the model's own small adjustment to Elo and has nothing to do with the picks. "
               "Draft effects in the backtest are small, a few points at most.")

    window = result["window"]
    number_config = {
        "Shrunk win rate": st.column_config.NumberColumn(format="%.1f%%"),
        "Player overall": st.column_config.NumberColumn(format="%.1f%%"),
        f"Champion win rate, {window} days": st.column_config.NumberColumn(format="%.1f%%"),
    }
    for tag, team in (("a", team_a), ("b", team_b)):
        st.markdown(f"**{team}**")
        st.dataframe(player_frame(result["tables"][tag], window), hide_index=True, width="stretch",
                     column_config=number_config)
    st.caption("Games on champion and player overall cover the last two years in every league in the data. "
               "Shrunk win rate pulls the record on the champion toward the player's overall win rate. "
               f"Champion numbers cover all pro games in the last {window} days and are pulled toward 50%. "
               "A player marked new has no history, so the model uses its prior for them.")
elif not complete:
    st.info("Pick all ten champions to see the draft number.")

st.subheader("Final score odds from the current score")
if result is not None:
    score_table(team_a, team_b, {"Elo": dist_elo, "With this draft": dist_draft})
else:
    score_table(team_a, team_b, {"Elo": dist_elo})

# ---------- log ----------


def log_prediction():
    append_log({
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "team_a": team_a, "team_b": team_b, "best_of": best_of,
        "score_a": score[0], "score_b": score[1], "game_number": game_number,
        "team_a_side": side, "fearless": fearless,
        "team_a_players": "|".join(names["a"]), "team_a_picks": "|".join(c or "" for c in picks["a"]),
        "team_b_players": "|".join(names["b"]), "team_b_picks": "|".join(c or "" for c in picks["b"]),
        "elo_p_game": round(p_elo, 4),
        "draft_p_game": "" if result is None else round(result["p"], 4),
        "elo_p_series": round(p_series_elo, 4),
        "draft_p_series": "" if p_series_draft is None else round(p_series_draft, 4),
        "draft_model": "" if model is None else f"{model.get('name', '')} ({model.get('status', '')})",
        "data_through": data_through,
    })
    # Remember this game's picks for fearless in the next game.
    for tag in ("a", "b"):
        seen = list(st.session_state.get(used_keys[tag], []))
        st.session_state[used_keys[tag]] = seen + [c for c in picks[tag] if c and c not in seen]
    st.session_state["logged"] = f"Logged game {game_number} to {LOG_PATH}"


st.button("Log prediction", type="primary", on_click=log_prediction)
if st.session_state.get("logged"):
    st.success(st.session_state.pop("logged"))
st.caption("Team A win probabilities are saved with the picks. Logging before the draft is finished "
           "saves the Elo numbers and leaves the draft number empty.")
