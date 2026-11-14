"""Shared scoring helpers so every model is reported the same way."""
import heapq

import numpy as np

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def per_game_log_loss(p, y):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_loss(p, y):
    return float(np.mean(per_game_log_loss(p, y)))


def brier(p, y):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def accuracy(p, y):
    return float(np.mean((np.asarray(p) > 0.5) == (np.asarray(y) == 1)))


def metrics_line(name, p, y):
    if len(y) == 0:
        return f"  {name:<28} n=    0"
    return (f"  {name:<28} n={len(y):5d}  log loss {log_loss(p, y):.4f}  "
            f"brier {brier(p, y):.4f}  acc {accuracy(p, y):.3f}")


def calibration_table(models, y, bins=10):
    """models is {name: probs}. One block per model, binned by predicted blue win prob."""
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    lines = []
    for name, p in models.items():
        p = np.asarray(p, dtype=float)
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        lines.append(f"  {name}")
        for b in range(bins):
            m = idx == b
            if m.any():
                lines.append(f"    {edges[b]:.1f} to {edges[b + 1]:.1f}  n={m.sum():5d}  "
                             f"predicted {p[m].mean():.3f}  observed {y[m].mean():.3f}")
    return "\n".join(lines)


def paired_bootstrap(p_new, p_base, y, n_boot=5000, seed=0):
    """Log loss of new minus base on the same games, resampling games with replacement.

    Returns (mean difference, 2.5th percentile, 97.5th percentile, share of
    resamples where new is better). Negative means new is better.
    """
    d = per_game_log_loss(p_new, y) - per_game_log_loss(p_base, y)
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 500):
        size = min(500, n_boot - start)
        idx = rng.integers(0, n, size=(size, n))
        means[start:start + size] = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float(np.mean(means < 0))


def bootstrap_line(name, p_new, p_base, y, **kw):
    mean, lo, hi, share = paired_bootstrap(p_new, p_base, y, **kw)
    verdict = "better than noise" if hi < 0 else ("worse than noise" if lo > 0 else "within noise")
    return (f"  {name:<44} n={len(y):5d}  log loss diff {mean:+.4f}  "
            f"95% interval {lo:+.4f} to {hi:+.4f}  better in {share:.0%} of resamples  {verdict}")


def release_order(start, end):
    """For games sorted by start time, yield (i, finished) where finished lists
    the games whose end time is at or before game i's start and that have not
    been released yet. Anything still in progress when game i starts stays hidden.
    """
    pending = []
    for i in range(len(start)):
        done = []
        while pending and pending[0][0] <= start[i]:
            done.append(heapq.heappop(pending)[1])
        yield i, done
        heapq.heappush(pending, (end[i], i))
