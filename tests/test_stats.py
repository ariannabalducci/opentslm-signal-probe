import json
import math

import numpy as np
import pytest

from src.stats import binom_two_sided, chance_follows, mcnemar_exact, wilson


def test_wilson_matches_known_interval():
    lo, hi = wilson(252, 400)  # 0.630
    assert lo == pytest.approx(0.58166, abs=1e-5)
    assert hi == pytest.approx(0.67587, abs=1e-5)


def test_wilson_handles_extremes():
    lo, hi = wilson(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0 < hi < 0.1
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_binomial_two_sided():
    assert binom_two_sided(5, 10) == pytest.approx(1.0)
    assert binom_two_sided(0, 10) == pytest.approx(2 / 1024)
    assert binom_two_sided(10, 10) == binom_two_sided(0, 10)
    assert binom_two_sided(155, 216) < 1e-9


def test_mcnemar_counts_only_discordant_pairs():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    only_a, only_b, p = mcnemar_exact(a, b)
    assert (only_a, only_b) == (2, 1)
    assert p == pytest.approx(binom_two_sided(2, 3))


def test_chance_follows_expectation_is_exact():
    preds = ["a"] * 30 + ["b"] * 10
    refs = ["a"] * 20 + ["b"] * 20
    expected, p = chance_follows(preds, refs, n_perm=200)
    assert expected == pytest.approx((30 * 20 + 10 * 20) / 40)
    assert 0 < p <= 1


def test_chance_follows_flags_perfect_agreement():
    labels = list("abcdefgh") * 10
    _, p = chance_follows(labels, labels, n_perm=500)
    assert p < 0.01


def test_torch_statistics_match_opentslm_formula(recording):
    torch = pytest.importorskip("torch")
    from src.data import channel_stats, stats_text

    series = torch.tensor(recording.tolist(), dtype=torch.float32)
    means = series.mean(dim=1, keepdim=True)
    stds = torch.clamp(series.std(dim=1, keepdim=True), min=1e-6)
    expected = [f"mean {m:.4f} and std {s:.4f}:" for m, s in
                zip(means.squeeze().tolist(), stds.squeeze().tolist())]
    got = [t.split("it has ")[1] for t in stats_text(*channel_stats(recording))]
    assert got == expected


def test_resume_drops_truncated_line(tmp_path):
    pytest.importorskip("torch")
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_probe", Path(__file__).resolve().parents[1] / "scripts" / "run_probe.py")
    run_probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_probe)

    path = tmp_path / "pred.jsonl"
    good = json.dumps({"condition": "original", "index": 3})
    path.write_text(good + "\n" + '{"condition": "orig', encoding="utf-8")
    assert run_probe.already_done(path) == {("original", 3)}
    assert path.read_text(encoding="utf-8") == good + "\n"


def test_stats_module_has_no_scipy_dependency():
    import src.stats as stats
    assert "scipy" not in open(stats.__file__, encoding="utf-8").read().split('"""', 2)[2]
    assert np.isfinite(wilson(1, 2)[0])
