# worlds-elo

Cross-region Elo for pro League of Legends, exact best-of-N series odds, a small draft model on top, and a local app for using all three during a live series. Built for the Worlds 2026 final and reusable for any pro series.

## Setup

```
python3 -m pip install -r requirements.txt
```

Download the yearly match data CSVs from Oracle's Elixir (https://oracleselixir.com/tools/downloads) into `data/`. Get 2022 through 2026, and download 2026 again right before predicting so the latest games are in it.

## Run the app

```
python3 build_artifacts.py data/*.csv
python3 -m streamlit run app.py
```

The first command takes about 20 seconds and rewrites `models/lookups.pkl` and `models/draft_model.json`. Run it again whenever you download newer CSVs. The second opens the app in your browser.

In the app you pick the two teams, the series length, the current score and which side team A is on. It shows the Elo odds for the game and the series straight away. Rosters are prefilled from each team's most recent game and can be edited. Once all ten champions are picked it shows the draft-adjusted odds for this game, where the change comes from, a per-player table, and the series odds using the draft number for this game and Elo for the rest. The fearless setting hides champions used earlier in the series, and Log prediction appends a row to `predictions/log.csv` and remembers that game's picks for fearless.

If `models/draft_model.json` is missing the app still runs on Elo alone.

## Results

Splits are fixed. Fit on 2022 to 2023, choose on 2024, test on 2025 to 2026. Log loss, lower is better. A coin flip scores 0.6931.

| Games | Elo v1 | Elo v3 | Elo v3 plus draft |
| --- | --- | --- | --- |
| 2024, all major-league games (2855) | 0.6212 | 0.6039 | 0.5977 |
| 2024, cross-region international (166) | 0.6444 | 0.6355 | 0.6259 |
| 2025 to 2026, all major-league games (6731) | 0.6384 | 0.6273 | 0.6225 |
| 2025 to 2026, cross-region international (391) | 0.6201 | 0.6066 | 0.6073 |

On 2025 to 2026 a paired bootstrap over games gives these 95% intervals for the log loss difference.

| Comparison | All major-league games | Cross-region international |
| --- | --- | --- |
| Elo v3 against Elo v1 | -0.0156 to -0.0067 | -0.0388 to +0.0097 |
| Draft against Elo v3 | -0.0067 to -0.0028 | -0.0081 to +0.0094 |

So Elo v3 and the draft model both beat what came before them on the full set of games by more than noise. On cross-region international games alone, which is what a Worlds final is, the samples are too small to separate any of the three, and the draft model is level with Elo v3 there.

How clean is the test? 2025 to 2026 had already been viewed twice before v3 existed, once for v1 and once for v2, which overfit and was rejected. Nothing in v3 or the draft model was tuned on those years and they were scored once at the very end, but knowing how v1 failed there shaped the v3 design. Treat the 2025 to 2026 numbers as nearly clean, not pristine.

Full reports with accuracy, Brier score and calibration tables are in `results/`.

- `results/v3_backtest.txt` is the Elo v3 tuning and the 2024 comparison
- `results/draft_backtest.txt` is the draft model selection on 2024
- `results/final_test.txt` is the single 2025 to 2026 run
- `results/backtest.txt` and `results/backtest_v2.txt` are the v1 and v2 records

## How it works

### Elo v3

Each team has a rating and each region has an offset, and a team's strength is the sum. Domestic games only move team ratings. Cross-region games move both, which is the only way the model learns how the LCK compares to the LPL. What changed from v1

- Only major leagues and international events are used (the list is in `config.py`). In v1 the minor leagues formed islands that never played a major region, so their offsets meant nothing and the LFL came out level with the LPL.
- A team's home region is decided per year, so a team promoted into the LEC is an LEC team from that year on and is not counted as an ERL team playing cross-region games.
- Tuning scores every 2022 to 2023 game, domestic and international, with a full grid over all four settings.
- `k_region` is the one setting 2024 was allowed to overrule. Every region offset starts at zero in 2022, so the fit years reward a very large value just to get the offsets moving. That value was badly overconfident on 2024, and a smaller one beat it by more than noise.
- Platt scaling fit on 2024 goes on top. With the smaller `k_region` it changes little.
- A game's result only reaches the ratings once the game has ended, so overlapping games never see each other.

EWC is not in the league list. To count those games, add the code to `V3_INTERNATIONAL` in `config.py` and rerun the three backtest commands below.

### Draft model

A logistic regression on four standardized numbers, each one blue side minus red side.

- The calibrated, side-adjusted Elo v3 logit
- Champion recent win rate. Each pick's win rate over all pro games in the last 60 days, shrunk toward 50%, averaged over the five picks
- Player win rate on champion. Each player's record on the champion over the last two years, shrunk toward that player's overall win rate, minus that overall win rate
- Player games on champion, on a log scale

Champion pick share was tried and dropped because it added nothing on 2024. Gradient boosting was tried on the same inputs and was worse than logistic regression. Leaving both teams' own games out of the champion win rate changed nothing, so that signal is about the meta and not a second helping of team form.

Player and champion history comes from every league in the data, so a rookie's academy games count. Unseen players and champions fall back to the prior. The draft moves a game by a few points of win probability at most. Most of the prediction is still Elo.

The model in `models/draft_model.json` keeps the features and settings chosen on 2024 and is refit on every year in the data so it has seen recent metas. `python3 build_artifacts.py data/*.csv --fit-years 2022 2023` gives the exact fit that was backtested.

### Series odds

Series odds are computed exactly by enumerating every path through the series, with the loser of each game taking blue. `series.score_distribution` can start from any score and can take a different probability for the next game, which is how the app mixes the draft number with Elo.

### Leakage

Every feature for a game uses only games that had finished before it started. Within a series, game 2 sees game 1 and never itself. `tests/test_draft.py` and `tests/test_v3.py` check this for every feature by cutting off later games, by flipping later results, and by flipping game 1 and watching game 2 move. `tests/test_live.py` checks that the app computes exactly what the backtest computed for the same game.

## Rebuild everything from scratch

```
python3 backtest_v3.py data/*.csv
python3 draft_backtest.py data/*.csv
python3 build_artifacts.py data/*.csv
```

The first writes `params_v3.json`, the second writes `models/draft_choice.json`, the third writes the files the app loads. `final_test.py` scores 2025 to 2026 and refuses to run again once `results/final_test.txt` exists, so a second look has to be deliberate.

The v1 scripts still work and still reproduce the first prediction.

```
python3 predict.py data/*.csv --a "Gen.G" --b "Bilibili Gaming" --bo 5
```

No real data handy? `python3 make_fake_data.py fake_data` writes CSVs in the same format with known team and region strengths.

## Known limitations

- Team name is the identity, so a rebrand starts from scratch
- Elo has no roster awareness beyond the yearly regression
- There are only a couple hundred cross-region games per year, so the region gap is the noisiest part of any LCK against LPL prediction
- The draft model ignores bans, pick order, matchups and synergies
- PCS, VCS and LJL have been second-tier leagues under the LCP since 2025 but still count as major leagues here
- Point estimates only, no uncertainty bands

## Tests

```
python3 -m pytest tests
```
