"""Elo v3 helpers shared by the backtests, the artifact builder and the app."""
import json
import math
from dataclasses import asdict

import numpy as np

from elo import EloParams

LN10_OVER_400 = math.log(10) / 400.0


def raw_logit(diff, params):
    """Blue win logit from the neutral diff (blue minus red), before calibration."""
    return (np.asarray(diff, dtype=float) + params.side_adv) * math.log(10) / params.scale


def fit_platt(z, y):
    """Fit p = sigmoid(a + b z) by maximum likelihood. Returns (a, b)."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(np.asarray(z, dtype=float).reshape(-1, 1), np.asarray(y).astype(int))
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def calibrated_logit(diff, params, platt):
    """Blue win logit after Platt scaling."""
    return platt["a"] + platt["b"] * raw_logit(diff, params)


def effective(diff, params, platt):
    """(diff, side_adv) on the plain 400 scale that reproduce the calibrated
    probabilities, so series.py can be used unchanged."""
    k = math.log(10) / params.scale
    return (platt["b"] * diff * k / LN10_OVER_400,
            (platt["b"] * params.side_adv * k + platt["a"]) / LN10_OVER_400)


def save_params(path, params, platt, extra=None):
    out = {"elo": asdict(params), "platt": platt}
    out.update(extra or {})
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def load_params(path="params_v3.json"):
    with open(path) as f:
        raw = json.load(f)
    return EloParams(**raw["elo"]), raw["platt"]
