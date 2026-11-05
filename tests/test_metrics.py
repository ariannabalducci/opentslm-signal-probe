import math

import pytest

from src.metrics import accuracy, extract_label, macro_f1, summarise
from src.perturb import BY_NAME


@pytest.mark.parametrize("text, expected", [
    ("The signal shows strong periodic peaks ... Answer: running", "running"),
    ("... Answer: walking_up.", "walking_up"),
    ("... Answer: Walking up", "walking_up"),
    ("... Answer: walking-down<|eot_id|>", "walking_down"),
    ("Answer: sitting. Answer: lying", "lying"),  # last answer wins, as in OpenTSLM
    ("... Answer: **standing**\nExtra text", "standing"),
    ("No final answer here", None),
    ("... Answer: jogging", None),
    ("... Answer:", None),
])
def test_extract_label(text, expected):
    assert extract_label(text) == expected


def test_accuracy_counts_none_as_wrong():
    assert accuracy(["a", None, "c"], ["a", "b", "c"]) == pytest.approx(2 / 3)


def test_macro_f1_none_adds_only_false_negatives():
    refs = ["running", "running", "lying", "lying"]
    preds = ["running", None, "lying", "running"]
    # running: tp=1 fp=1 fn=1 -> 0.5 ; lying: tp=1 fp=0 fn=1 -> 2/3
    assert macro_f1(preds, refs) == pytest.approx((0.5 + 2 / 3) / 2)


def test_macro_f1_perfect():
    refs = ["biking", "sitting", "walking"]
    assert macro_f1(refs, refs) == 1.0


def test_summarise_reports_who_wins_a_conflict():
    conditions = [BY_NAME["original"], BY_NAME["stats_from_other"]]
    records = [
        {"condition": "original", "label": "running", "donor_label": "lying", "prediction": "running"},
        {"condition": "original", "label": "lying", "donor_label": "running", "prediction": None},
        # Statistics come from the donor; the model follows them in 2 of 3 cases.
        {"condition": "stats_from_other", "label": "running", "donor_label": "lying", "prediction": "lying"},
        {"condition": "stats_from_other", "label": "lying", "donor_label": "running", "prediction": "running"},
        {"condition": "stats_from_other", "label": "sitting", "donor_label": "biking", "prediction": "sitting"},
    ]
    rows = {r["condition"]: r for r in summarise(records, conditions)}
    assert rows["original"]["accuracy"] == 0.5
    assert rows["original"]["unparsed"] == 0.5
    assert rows["original"]["follows_signal"] is None
    swap = rows["stats_from_other"]
    assert swap["follows_text"] == pytest.approx(2 / 3)
    assert swap["follows_signal"] == pytest.approx(1 / 3)
    assert swap["accuracy"] == swap["follows_signal"]  # the series is the sample's own


def test_summarise_handles_missing_condition():
    rows = summarise([], [BY_NAME["noise_signal"]])
    assert rows[0]["n"] == 0
    assert math.isnan(rows[0]["accuracy"])
