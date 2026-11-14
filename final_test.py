"""Score the frozen models on 2025 to 2026. Run this once.

python3 final_test.py data/*.csv

Nothing here is tuned. Elo v3 comes from params_v3.json, the draft model is
the one recorded in models/draft_choice.json, refit on 2022 to 2023 exactly as
it was when it was chosen. The script refuses to overwrite its own report so
a second look has to be deliberate (--force).
"""
import argparse
import json
import os
import sys

import v3
from backtest_v3 import v1_probs_for
from config import FIT_YEARS, TEST_YEARS
from data import load_rows
from draft import build_dataset, feature_matrix, fit_logistic, predict_logistic
from evalkit import bootstrap_line, calibration_table, metrics_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--report", default="results/final_test.txt")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.exists(args.report) and not args.force:
        sys.exit(f"{args.report} already exists. The test years have been scored. "
                 "Read that file, or pass --force if you really mean to look again.")
    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    params, platt = v3.load_params()
    with open("models/draft_choice.json") as f:
        choice = json.load(f)
    hp = choice["hyperparams"]
    data = build_dataset(load_rows(args.csvs), params, platt)
    games, y, elo = data["games"], data["y"], data["elo_p"]
    fit = games["year"].isin(FIT_YEARS).to_numpy()
    test = games["year"].isin(TEST_YEARS).to_numpy()
    intl = (games["intl_event"] & games["cross_region"]).to_numpy()

    X = feature_matrix(data["raw"], data["elo_logit"], hp["pc_shrink"], hp["window"], hp["champ_shrink"],
                       choice["features"])
    draft = predict_logistic(fit_logistic(X[fit], y[fit], choice["features"]), X)
    v1 = v1_probs_for(args.csvs, games["gameid"])
    draft_name = f"Draft ({choice['status']})"
    models = {"Elo v1 (baseline)": v1, "Elo v3 + Platt": elo, draft_name: draft}

    out("Final test on 2025 to 2026")
    out("Every setting was frozen before this ran. Elo v3 was fit on 2022 to 2023 with k_region and")
    out("Platt scaling from 2024. The draft model was fit on 2022 to 2023 and chosen on 2024.")
    out(f"draft model  {choice['name']}  {hp}")
    out("")
    out("How clean is this? 2025 to 2026 had already been looked at twice (v1 and the rejected v2)")
    out("before v3 was designed. v3 was never tuned on these years, but knowing that v1 was")
    out("overconfident there shaped the design, so treat this as nearly clean, not pristine.")
    out("Elo v1 itself was scored on these games before, with the same params.json.")

    for label, mask in (("cross-region international games", test & intl), ("all major-league games", test)):
        out(f"\n2025 to 2026, {label}")
        for name, p in models.items():
            out(metrics_line(name, p[mask], y[mask]))
        for yr in TEST_YEARS:
            m = mask & (games["year"] == yr).to_numpy()
            out(f"  {yr} only")
            for name, p in models.items():
                out("  " + metrics_line(name, p[m], y[m]))
        out("  paired bootstrap over games, log loss difference (negative favours the first model)")
        out(bootstrap_line("Elo v3 + Platt vs Elo v1", elo[mask], v1[mask], y[mask]))
        out(bootstrap_line("Draft vs Elo v3 + Platt", draft[mask], elo[mask], y[mask]))
        out(bootstrap_line("Draft vs Elo v1", draft[mask], v1[mask], y[mask]))
        out("  calibration (bins of predicted blue win probability)")
        out(calibration_table({n: p[mask] for n, p in models.items()}, y[mask],
                              bins=5 if "international" in label else 10))

    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nsaved {args.report}")


if __name__ == "__main__":
    main()
