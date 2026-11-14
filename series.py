"""Exact best-of-N series probabilities from a game-level Elo diff.

Exact enumeration instead of Monte Carlo. A Bo5 has at most 2^5 paths, so
there is no reason to sample.
"""
from collections import defaultdict

from elo import win_prob


def score_distribution(diff, side_adv, best_of=5, a_blue_game1=None, scale=400.0, adjust=None,
                       start=(0, 0), p_next=None):
    """Probability of each final score (a_wins, b_wins).

    diff is A's neutral rating minus B's. The loser of each game picks side and
    takes blue whenever blue is the stronger side. a_blue_game1=None averages
    over both game 1 sides. adjust(d, game_number) returns a modified diff for
    that game, which is the hook for things like fearless draft wearing down
    the favorite's champion pool in later games.

    start is the current series score. When it is not (0, 0), a_blue_game1 is
    A's side in the next game to be played. p_next, if given, replaces A's win
    probability in that next game only (a draft-adjusted number, say) and every
    game after it goes back to diff and side_adv.
    """
    need = best_of // 2 + 1
    a0, b0 = start
    if min(a0, b0) < 0 or max(a0, b0) > need or (a0 == need and b0 == need):
        raise ValueError(f"score {a0}-{b0} is not possible in a best of {best_of}")
    if p_next is not None and not 0.0 <= p_next <= 1.0:
        raise ValueError("p_next must be a probability")
    if a_blue_game1 is None:
        x = score_distribution(diff, side_adv, best_of, True, scale, adjust, start, p_next)
        y = score_distribution(diff, side_adv, best_of, False, scale, adjust, start, p_next)
        return {k: (x.get(k, 0.0) + y.get(k, 0.0)) / 2 for k in set(x) | set(y)}

    loser_takes_blue = side_adv >= 0
    out = defaultdict(float)

    def step(a_wins, b_wins, a_blue, prob, override=None):
        if a_wins == need or b_wins == need:
            out[(a_wins, b_wins)] += prob
            return
        if override is not None:
            p = override
        else:
            d = diff + (side_adv if a_blue else -side_adv)
            if adjust is not None:
                d = adjust(d, a_wins + b_wins + 1)
            p = win_prob(d, scale)
        if p > 0:
            step(a_wins + 1, b_wins, not loser_takes_blue, prob * p)  # B lost, B picks side
        if p < 1:
            step(a_wins, b_wins + 1, loser_takes_blue, prob * (1 - p))  # A lost, A picks side

    step(a0, b0, a_blue_game1, 1.0, p_next)
    return dict(out)


def series_win_prob(diff, side_adv, best_of=5, a_blue_game1=None, scale=400.0, adjust=None,
                    start=(0, 0), p_next=None):
    need = best_of // 2 + 1
    dist = score_distribution(diff, side_adv, best_of, a_blue_game1, scale, adjust, start, p_next)
    return sum(p for (a, _), p in dist.items() if a == need)
