# worlds-elo

Cross-region Elo for pro League of Legends plus exact best-of-N series odds. Built for the Worlds 2026 final and reusable for any pro series.

## Setup

```
pip install -r requirements.txt
```

Download the yearly match data CSVs from Oracle's Elixir (https://oracleselixir.com/tools/downloads) into `data/`. Get 2022 through 2026, and re-download 2026 right before predicting so the latest games are in it.

## Run

```
python backtest.py data/*.csv --list-leagues
python backtest.py data/*.csv --tune-years 2022 2023 2024 --test-years 2025 2026
python predict.py data/*.csv --a "Gen.G" --b "Bilibili Gaming" --bo 5
```

The first command checks that the international league codes in `config.py` match the data. The second tunes on older events, scores on newer ones and writes `params.json`. The third uses those params. Add `--a-game1-side blue` once sides are announced, and `--before 2026-11-14` if you rerun after the final so it can't see the result.

No real data handy? `python make_fake_data.py fake_data` writes CSVs in the same format with known team and region strengths.

## How it works

Each team has a rating and each region has an offset, and a team's strength is the sum. Domestic games only move team ratings because the offsets cancel. Cross-region games at MSI, EWC, First Stand and Worlds move both, which is the only way the model learns how the LCK compares to the LPL. Each new year, team ratings regress toward their region average and offsets regress toward zero.

Blue side advantage is estimated from game 1s only. Later in a series the previous loser picks blue, so weaker teams sit on blue more often and raw blue win rate understates the edge.

Series odds are computed exactly by enumerating every path through the series, with the loser of each game taking blue. `score_distribution` takes an `adjust` hook for per-game effects such as fearless draft.

Every prediction uses only earlier games, so backtest numbers are out of sample. Hyperparameters are tuned on the tune years only.

## Known limitations

- Team name is the identity, so a rebrand starts from scratch
- No roster awareness beyond the yearly regression
- Only a couple hundred cross-region games per year, so the region gap is the noisiest part of any LCK vs LPL prediction
- Point estimates only, no uncertainty bands yet

## Tests

```
python -m pytest tests
```
