"""Settings you are most likely to need to edit."""

# Oracle's Elixir league codes for cross-region events.
# Run `python backtest.py data/*.csv --list-leagues` and fix this if a code differs.
INTERNATIONAL_LEAGUES = {"WLDs", "MSI", "EWC", "FST"}

# Renamed or merged regions. Teams in the left league count as the right region.
# Judgment call (VCS and LJL teams only joined the LCP in 2025), so adjust if
# you care about those regions. It does not affect LCK vs LPL.
LEAGUE_ALIASES = {
    "LTA N": "LCS",
    "LTA S": "CBLOL",
    "PCS": "LCP",
    "LJL": "LCP",
    "VCS": "LCP",
}
