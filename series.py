"""Exact best-of-N series probabilities from a game-level Elo diff.

Exact enumeration instead of Monte Carlo. A Bo5 has at most 2^5 paths, so
there is no reason to sample.
"""
from collections import defaultdict

from elo import win_prob


def score_distribution(diff, side_adv, best_of=5, a_blue_game1=None, scale=400.0, adjust=None):
    """Probability of each final score (a_wins, b_wins).

    diff is A's neutral rating minus B's. The loser of each game picks side and
    takes blue whenever blue is the stronger side. a_blue_game1=None averages
    over both game 1 sides. adjust(d, game_number) returns a modified diff for
    that game, which is the hook for things like fearless draft wearing down
    the favorite's champion pool in later games.
    """
    if a_blue_game1 is None:
        x = score_distribution(diff, side_adv, best_of, True, scale, adjust)
        y = score_distribution(diff, side_adv, best_of, False, scale, adjust)
        return {k: (x.get(k, 0.0) + y.get(k, 0.0)) / 2 for k in set(x) | set(y)}

    need = best_of // 2 + 1
    loser_takes_blue = side_adv >= 0
    out = defaultdict(float)

    def step(a_wins, b_wins, a_blue, prob):
        if a_wins == need or b_wins == need:
            out[(a_wins, b_wins)] += prob
            return
        d = diff + (side_adv if a_blue else -side_adv)
        if adjust is not None:
            d = adjust(d, a_wins + b_wins + 1)
        p = win_prob(d, scale)
        step(a_wins + 1, b_wins, not loser_takes_blue, prob * p)  # B lost, B picks side
        step(a_wins, b_wins + 1, loser_takes_blue, prob * (1 - p))  # A lost, A picks side

    step(0, 0, a_blue_game1, 1.0)
    return dict(out)


def series_win_prob(diff, side_adv, best_of=5, a_blue_game1=None, scale=400.0, adjust=None):
    need = best_of // 2 + 1
    dist = score_distribution(diff, side_adv, best_of, a_blue_game1, scale, adjust)
    return sum(p for (a, _), p in dist.items() if a == need)
