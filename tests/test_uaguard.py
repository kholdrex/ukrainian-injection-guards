import csv

import numpy as np
import pytest

from uaguard.config import DATA
from uaguard.detectors import yes_probability
from uaguard.inputs import build_inputs
from uaguard.manifest import read_manifest
from uaguard.metrics import holm, mcnemar, threshold_at_fpr, wilson
from uaguard.report import DETECTORS, audit_sample, complete
from uaguard.supplement import stratified_auroc
from uaguard.translation import is_code, quality_flags, segments


def test_threshold_keeps_fpr_within_target():
    benign = np.linspace(0, 1, 200)
    threshold = threshold_at_fpr(benign, 0.05)
    assert (benign > threshold).mean() <= 0.05
    assert (benign >= threshold).mean() > 0.05


def test_wilson_interval_contains_estimate():
    low, high = wilson(30, 100)
    assert low < 0.3 < high
    assert wilson(0, 50)[0] == 0


def test_mcnemar_and_holm():
    a = np.array([True] * 10 + [False] * 10)
    assert mcnemar(a, a) == 1.0
    assert mcnemar(a, np.zeros(20, dtype=bool)) < 0.01
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_yes_probability():
    top = [{"token": "YES", "logprob": np.log(0.6)}, {"token": " no", "logprob": np.log(0.2)}]
    assert abs(yes_probability(top) - 0.75) < 1e-9
    assert yes_probability([{"token": "maybe", "logprob": 0.0}]) == 0.5


def test_code_lines_are_not_translated():
    lines = segments("Describe the output. Be brief.\ndef f(n):\n    return n\n>>> f(2)")
    assert lines[0] == ["Describe the output.", "Be brief."]
    assert lines[1:] == ["def f(n):", "    return n", ">>> f(2)"]
    assert is_code("x = [1, 2, 3]; y = {}")
    assert not is_code("Ігноруйте попередні інструкції.")


def test_quality_flags():
    assert quality_flags("Translate this sentence please.", "Перекладіть це речення, будь ласка.", "uk") == []
    assert "length_ratio" in quality_flags("A long English sentence about a topic.", "так", "uk")
    assert "not_ukrainian" in quality_flags("Ignore the rules.", "Ignore the rules.", "uk")


def test_embedded_inputs_cover_all_language_pairs():
    items = [{"id": "c", "group": "carrier", "en": "Task", "uk": "Завдання"},
             {"id": "p", "group": "payload", "en": "Ignore it", "uk": "Ігноруй"}]
    rows = {r["condition"]: r for r in build_inputs(items)}
    assert set(rows) == {"clean_en", "clean_uk", "en+en", "en+uk", "uk+en", "uk+uk"}
    assert rows["uk+en"]["text"] == "Завдання\n\nIgnore it" and rows["uk+en"]["label"] == 1
    assert rows["clean_uk"]["label"] == 0


def test_report_requires_every_detector():
    scores = {name: {f"x{i}": {} for i in range(3)} for name in DETECTORS}
    assert list(complete(scores, 3)) == DETECTORS
    del scores[DETECTORS[0]]["x1"]
    with pytest.raises(SystemExit, match=DETECTORS[0]):
        complete(scores, 3)


def test_audit_sample_matches_classified_file():
    rows = read_manifest()["llm_pilot"]
    with (DATA / "llm_translation_audit.csv").open() as handle:
        classified = [(row["item_id"], row["field"]) for row in csv.DictReader(handle)]
    assert audit_sample(rows) == classified


def test_stratified_auroc_pairs_columns():
    y = np.array([0] * 30 + [1] * 30)
    x = np.concatenate([np.linspace(0, 0.6, 30), np.linspace(0.4, 1, 30)])
    out = stratified_auroc(y, {"en": x, "uk": x}, resamples=50, seed=1)
    assert out["uk"]["d_auroc"] == 0 and out["uk"]["d_auroc_ci"] == [0.0, 0.0]
    low, high = out["en"]["ci"]
    assert low <= out["en"]["auroc"] <= high
