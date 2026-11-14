"""Choose the draft model on 2024 and compare it to Elo v3.

python3 draft_backtest.py data/*.csv

Fit on 2022 to 2023, choose on 2024. This script never prints a 2025 or 2026
score. Run final_test.py once afterwards for those.
"""
import argparse
import json
import os

import numpy as np

import v3
from config import FIT_YEARS, SELECT_YEARS
from data import load_rows
from draft import (CHAMP_WINDOWS, PLAYER_FEATURES, build_dataset,
                   feature_matrix, fit_logistic, predict_logistic)
from evalkit import bootstrap_line, calibration_table, log_loss, metrics_line, paired_bootstrap

PC_SHRINK = [1, 2, 5, 10, 20, 40, 80]
CHAMP_SHRINK = [10, 20, 40, 80, 160, 320, 640]
DEFAULT = {"pc_shrink": 10, "window": 60, "champ_shrink": 20}


def fit_predict(data, names, hp, fit_mask):
    X = feature_matrix(data["raw"], data["elo_logit"], hp["pc_shrink"], hp["window"], hp["champ_shrink"], names)
    model = fit_logistic(X[fit_mask], data["y"][fit_mask], names)
    return model, predict_logistic(model, X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--params", default="params_v3.json")
    ap.add_argument("--report", default="results/draft_backtest.txt")
    ap.add_argument("--choice", default="models/draft_choice.json")
    args = ap.parse_args()
    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    params, platt = v3.load_params(args.params)
    data = build_dataset(load_rows(args.csvs), params, platt)
    games, y, elo = data["games"], data["y"], data["elo_p"]
    fit = games["year"].isin(FIT_YEARS).to_numpy()
    sel = games["year"].isin(SELECT_YEARS).to_numpy()
    intl = (games["intl_event"] & games["cross_region"]).to_numpy()

    out("Draft model backtest")
    out(f"fit on 2022 to 2023 ({fit.sum()} games), chosen on 2024 ({sel.sum()} games). "
        "2025 and 2026 are not scored here.")
    out(f"games dropped for an incomplete lineup  {data['dropped']}")
    out("baseline is Elo v3 + Platt on exactly the same games. Its Platt numbers were fit on 2024.")
    out("history for player and champion features comes from every league in the data.")

    def score(p):
        return log_loss(p[sel], y[sel])

    out("\nstep 1  logistic regression on the Elo logit alone (should match the baseline)")
    m_elo, p_lr_elo = fit_predict(data, ["elo_logit"], DEFAULT, fit)
    out(f"  Elo v3 + Platt   2024 log loss {score(elo):.4f}")
    out(f"  LR on Elo logit  2024 log loss {score(p_lr_elo):.4f}")

    out("\nstep 2  player features, shrinkage strength chosen on 2024")
    out("  shrunk win rate = (wins on champ + s * player overall win rate) / (games on champ + s)")
    hp = dict(DEFAULT)
    res = {}
    for s in PC_SHRINK:
        _, p = fit_predict(data, ["elo_logit"] + PLAYER_FEATURES, {**hp, "pc_shrink": s}, fit)
        res[s] = score(p)
        out(f"  s = {s:3d}  2024 log loss {res[s]:.5f}")
    hp["pc_shrink"] = min(res, key=res.get)
    spread = max(res.values()) - min(res.values())
    out(f"  chosen s = {hp['pc_shrink']}. The whole curve spans {spread:.5f}, so the choice barely matters.")

    out("\nstep 3  champion win rate, window and shrinkage chosen on 2024")
    out("  shrunk win rate = (wins in window + m * 0.5) / (picks in window + m)")
    res = {}
    for w in CHAMP_WINDOWS:
        row = []
        for m in CHAMP_SHRINK:
            _, p = fit_predict(data, ["elo_logit", "ch_wr"], {**hp, "window": w, "champ_shrink": m}, fit)
            res[(w, m)] = score(p)
            row.append(f"m={m} {res[(w, m)]:.4f}")
        out(f"  window {w:2d} days  " + "  ".join(row))
    hp["window"], hp["champ_shrink"] = min(res, key=res.get)
    out(f"  chosen window {hp['window']} days, m = {hp['champ_shrink']}")

    out("\n  check  is champion win rate just the two teams' own recent form in disguise?")
    _, p_in = fit_predict(data, ["elo_logit", "ch_wr"], hp, fit)
    _, p_x = fit_predict(data, ["elo_logit", "ch_wr_x"], hp, fit)
    out(f"  all games in the window                    2024 log loss {score(p_in):.4f}")
    out(f"  leaving out every game of these two teams  2024 log loss {score(p_x):.4f}")
    out("  " + ("No. The gain survives without their own games, so it is a meta signal."
                if score(p_x) - score(p_in) < 0.0005 else
                "Partly. Some of the gain comes from the teams' own recent results."))

    out("\nstep 4  forward selection on 2024")
    out("  start from Elo, add the block that helps most, keep it only if the interval excludes zero")
    blocks = {"champion win rate": ["ch_wr"], "player on champion": list(PLAYER_FEATURES),
              "champion pick share": ["ch_pick"]}
    current_names, current_p, current_label = ["elo_logit"], elo, "Elo v3 + Platt"
    current_model, steps = None, []
    remaining = dict(blocks)
    while remaining:
        trial = {}
        for label, cols in remaining.items():
            trial[label] = fit_predict(data, current_names + cols, hp, fit)
        best = min(trial, key=lambda k: score(trial[k][1]))
        for label, (_, p) in trial.items():
            out(bootstrap_line(f"+ {label} vs {current_label}", p[sel], current_p[sel], y[sel]))
        mean, lo, hi, _ = paired_bootstrap(trial[best][1][sel], current_p[sel], y[sel])
        if hi >= 0:
            out(f"  stop. Nothing left beats {current_label} by more than noise.")
            break
        current_names = current_names + remaining.pop(best)
        current_model, current_p = trial[best]
        current_label = "Elo + " + " + ".join(steps + [best])
        steps.append(best)
        out(f"  keep {best}")

    adopted = bool(steps)
    if adopted:
        status, shown, names = "adopted", current_label, current_names
        model_shown, p_shown = current_model, current_p
    else:
        status, shown, names = "experimental", "Elo + champion win rate + player on champion", \
            ["elo_logit", "ch_wr"] + PLAYER_FEATURES
        model_shown, p_shown = fit_predict(data, names, hp, fit)

    all_models = {"Elo v3 + Platt (baseline)": elo, shown: p_shown}
    for label, mask in (("2022 to 2023 (fit years, in sample for the draft model)", fit),
                        ("2024 cross-region international games", sel & intl),
                        ("2024 all major-league games", sel)):
        out(f"\n{label}")
        for name, p in all_models.items():
            out(metrics_line(name, p[mask], y[mask]))

    out("\npaired bootstrap over games on 2024, log loss difference (negative favours the draft model)")
    out(bootstrap_line("draft vs Elo, all major-league", p_shown[sel], elo[sel], y[sel]))
    out(bootstrap_line("draft vs Elo, cross-region international", p_shown[sel & intl], elo[sel & intl], y[sel & intl]))

    out("\nstep 5  gradient boosting check on the same features")
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = feature_matrix(data["raw"], data["elo_logit"], hp["pc_shrink"], hp["window"], hp["champ_shrink"], names)
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)
    gb.fit(X[fit], y[fit])
    p_gb = gb.predict_proba(X)[:, 1]
    out(metrics_line("gradient boosting, 2024 all", p_gb[sel], y[sel]))
    gb_line = bootstrap_line("gradient boosting vs logistic regression", p_gb[sel], p_shown[sel], y[sel])
    out(gb_line)
    out("  " + ("It wins by more than noise. Worth a closer look, not adopted automatically."
                if "better than noise" in gb_line else
                "Not kept. It does not beat logistic regression on 2024 by more than noise."))

    out("\ndecision")
    mean, lo, hi, _ = paired_bootstrap(p_shown[sel], elo[sel], y[sel])
    if adopted:
        out(f"  {shown} beats Elo v3 on 2024 by more than noise and is adopted.")
        out("  caveat  s, the window and m were also picked on 2024 and three blocks were tried,")
        out("          so the 2024 gain is somewhat optimistic. final_test.py gives the clean number.")
        out("  caveat  the gain is small next to the gap between teams. The draft moves a game by a")
        out("          few points of win probability, not tens.")
    else:
        out("  No draft model beats Elo v3 on 2024 by more than noise.")
        out(f"  Elo v3 stays the model of record. The app shows {shown} as an experimental number.")
    out(f"  shown in the app  {shown}")
    out(f"  2024 log loss {score(p_shown):.4f} vs Elo {score(elo):.4f}  diff {mean:+.4f} ({lo:+.4f} to {hi:+.4f})")
    fitted, preds, candidates = {shown: model_shown}, {shown: p_shown}, {shown: names}

    out(f"\ncoefficients of {shown} on standardized features (fit on 2022 to 2023)")
    m = fitted[shown]
    for n, c, sd in zip(m["features"], m["coef"], m["std"]):
        out(f"  {n:<10} coef {c:+.4f}   one sd = {sd:.4f}")
    out(f"  intercept  {m['intercept']:+.4f}")

    out("\ncalibration on 2024, all major-league games")
    out(calibration_table({"Elo v3 + Platt (baseline)": elo[sel], shown: preds[shown][sel]}, y[sel], bins=10))
    out("\ncalibration on 2024, cross-region international games")
    out(calibration_table({"Elo v3 + Platt (baseline)": elo[sel & intl], shown: preds[shown][sel & intl]},
                          y[sel & intl], bins=5))

    os.makedirs(os.path.dirname(args.choice), exist_ok=True)
    with open(args.choice, "w") as f:
        json.dump({
            "status": status, "name": shown, "features": candidates[shown], "hyperparams": hp,
            "fit_years": FIT_YEARS, "select_years": SELECT_YEARS,
            "select_log_loss": {"elo": score(elo), "draft": score(preds[shown]),
                                "diff": mean, "lo": lo, "hi": hi, "n": int(sel.sum())},
        }, f, indent=2)
    out(f"\nsaved {args.choice}")
    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {args.report}")


if __name__ == "__main__":
    main()
