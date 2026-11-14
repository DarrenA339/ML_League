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


# ---- v3 ----
# Elo v3 only uses leagues whose teams actually meet at international events,
# so every region offset is tied to the others by real cross-region games.
MAJOR_LEAGUES = {"LCK", "LPL", "LEC", "LCS", "LTA N", "LTA S", "CBLOL", "LCP", "PCS", "VCS", "LJL"}

# LTA is the 2025 cross-conference event between LTA N and LTA S teams.
# EWC is left out on purpose. Add it here if you want those games counted.
V3_INTERNATIONAL = {"WLDs", "MSI", "FST", "LTA"}

V3_LEAGUES = MAJOR_LEAGUES | V3_INTERNATIONAL

# Region a home league counts as. PCS, VCS and LJL stay separate regions here
# because before 2025 they never played each other domestically.
V3_REGION_ALIASES = {"LTA N": "LCS", "LTA S": "CBLOL"}

# Teams at an international event with no major home league in the data
# (LLA, TCL, LCO and LCL wildcards) share one region so its offset can learn
# how weak they are.
MINOR_REGION = "Minor"

FIT_YEARS = [2022, 2023]
SELECT_YEARS = [2024]
TEST_YEARS = [2025, 2026]


def enable_ewc():
    """Sensitivity switch. Counts EWC as an international event for this run
    only. The frozen v3 results were produced without it."""
    V3_INTERNATIONAL.add("EWC")
    V3_LEAGUES.add("EWC")
