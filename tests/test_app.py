"""Drive the Streamlit app headlessly against small fake artifacts."""
import csv
import json
import pickle
from pathlib import Path

import pytest

from build_artifacts import build_lookups
from data import load_rows
from draft import build_dataset, feature_matrix, fit_logistic
from elo import EloParams
from make_fake_data import write_fake_data

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = Path(__file__).resolve().parents[1] / "app.py"
PARAMS, PLATT = EloParams(k_team=32, k_region=24, side_adv=30.0), {"a": -0.05, "b": 0.95}
HP = dict(pc_shrink=5, window=60, champ_shrink=20)
NAMES = ["elo_logit", "ch_wr", "pc_wr", "pc_games"]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    rows = load_rows(write_fake_data(tmp_path_factory.mktemp("fake"), years=range(2021, 2024), seed=5))
    data = build_dataset(rows, PARAMS, PLATT)
    folder = tmp_path_factory.mktemp("models")
    with open(folder / "lookups.pkl", "wb") as f:
        pickle.dump(build_lookups(data, PARAMS, PLATT), f)
    X = feature_matrix(data["raw"], data["elo_logit"], names=NAMES, **HP)
    model = fit_logistic(X, data["y"], NAMES)
    model.update({"status": "adopted", "name": "test model", "hyperparams": HP})
    return folder, model


def start(monkeypatch, folder, log, model=None):
    path = folder / "draft_model.json"
    if model is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(json.dumps(model))
    monkeypatch.setenv("WORLDS_ELO_MODELS", str(folder))
    monkeypatch.setenv("WORLDS_ELO_LOG", str(log))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def by_label(widgets, text):
    return [w for w in widgets if text in w.label]


def pick_all(at):
    for box in by_label(at.selectbox, "champion"):
        role = box.label.split()[0]
        prefix = {"Jungle": "Jng"}.get(role, role if role != "Support" else "Sup")
        box.set_value(next(o for o in box.options if o.startswith(prefix)))
    return at.run()


def test_runs_before_the_draft(monkeypatch, artifacts, tmp_path):
    folder, model = artifacts
    at = start(monkeypatch, folder, tmp_path / "log.csv", model)
    labels = [m.label for m in at.metric]
    assert any("wins this game, Elo" in l for l in labels)
    assert sum("wins the series, Elo" in l for l in labels) == 2
    assert len(by_label(at.selectbox, "champion")) == 10
    players = by_label(at.text_input, "player")
    assert len(players) == 10 and all(p.value for p in players)      # rosters are prefilled
    assert any("Pick all ten champions" in i.value for i in at.info)


def test_full_draft_then_log(monkeypatch, artifacts, tmp_path):
    folder, model = artifacts
    log = tmp_path / "predictions" / "log.csv"
    at = pick_all(start(monkeypatch, folder, log, model))
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert any("draft adjusted" in l for l in labels)
    at.button[0].click().run()
    assert not at.exception, at.exception
    rows = list(csv.DictReader(open(log)))
    assert len(rows) == 1
    row = rows[0]
    assert row["game_number"] == "1" and row["team_a_side"] == "Blue"
    assert len(row["team_a_picks"].split("|")) == 5 and all(row["team_a_picks"].split("|"))
    assert 0 < float(row["elo_p_game"]) < 1 and 0 < float(row["draft_p_game"]) < 1
    at.button[0].click().run()
    assert len(list(csv.DictReader(open(log)))) == 2                # appends, never overwrites


def test_fearless_hides_earlier_picks(monkeypatch, artifacts, tmp_path):
    folder, model = artifacts
    at = pick_all(start(monkeypatch, folder, tmp_path / "log.csv", model))
    a_picks = [b.value for b in by_label(at.selectbox, "champion")[:5]]
    b_picks = [b.value for b in by_label(at.selectbox, "champion")[5:]]
    at.button[0].click().run()
    by_label(at.radio, "Hide champions")[0].set_value("Own team")
    by_label(at.number_input, "wins")[0].set_value(1)
    at = at.run()
    assert not at.exception, at.exception
    boxes = by_label(at.selectbox, "champion")
    assert all(b.value is None for b in boxes)                       # new game, empty draft
    assert not set(a_picks) & set(boxes[0].options)                  # A cannot see its own earlier picks
    only_b = set(b_picks) - set(a_picks)
    assert only_b <= set(boxes[0].options)                           # but still sees B's
    by_label(at.radio, "Hide champions")[0].set_value("Both teams")
    at = at.run()
    boxes = by_label(at.selectbox, "champion")
    assert not (set(a_picks) | set(b_picks)) & set(boxes[0].options)
    assert not (set(a_picks) | set(b_picks)) & set(boxes[9].options)


def test_runs_on_elo_alone_without_a_draft_model(monkeypatch, artifacts, tmp_path):
    folder, _ = artifacts
    log = tmp_path / "log.csv"
    at = pick_all(start(monkeypatch, folder, log, model=None))
    assert not at.exception, at.exception
    assert any("Elo alone" in i.value for i in at.info)
    assert not any("draft adjusted" in m.label for m in at.metric)
    at.button[0].click().run()
    row = list(csv.DictReader(open(log)))[0]
    assert row["draft_p_game"] == "" and 0 < float(row["elo_p_game"]) < 1


def test_ui_copy_has_no_colons_or_em_dashes(monkeypatch, artifacts, tmp_path):
    folder, model = artifacts
    at = pick_all(start(monkeypatch, folder, tmp_path / "log.csv", model))
    text = []
    for group in (at.title, at.header, at.subheader, at.caption, at.markdown, at.info, at.warning):
        text += [w.value for w in group]
    for group in (at.metric, at.selectbox, at.text_input, at.radio, at.number_input, at.button):
        text += [w.label for w in group]
    assert len(text) > 30
    for t in text:
        assert ":" not in t and "—" not in t and "–" not in t, t
