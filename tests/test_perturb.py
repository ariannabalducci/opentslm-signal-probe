import numpy as np
import pytest

from src.data import channel_stats, normalise, stats_text, bare_text
from src.perturb import BY_NAME, CONDITIONS, build_inputs, condition_rng


def run(name, signal, donor, seed=0):
    return build_inputs(signal, donor, BY_NAME[name], condition_rng(0, 0, seed))


def test_original_reproduces_opentslm_input(recording):
    series, texts = run("original", recording, None)
    np.testing.assert_allclose(series, normalise(recording))
    assert texts == stats_text(*channel_stats(recording))


def test_shuffle_keeps_values_and_stated_statistics(recording):
    series, texts = run("shuffled_time", recording, None)
    original = normalise(recording)
    np.testing.assert_allclose(np.sort(series, axis=1), np.sort(original, axis=1), rtol=1e-6)
    assert not np.allclose(series, original)
    # Mean and std are permutation-invariant, so the text stays truthful.
    assert texts == stats_text(*channel_stats(recording))


def test_shuffle_uses_one_permutation_for_all_axes(recording):
    # A strictly increasing x-axis makes the permutation recoverable from the output.
    ramped = recording.copy()
    ramped[0] = np.arange(128, dtype=np.float32)
    series, _ = run("shuffled_time", ramped, None)
    original = normalise(ramped)
    perm = np.searchsorted(original[0], series[0])
    np.testing.assert_allclose(series[1], original[1][perm])
    np.testing.assert_allclose(series[2], original[2][perm])


def test_noise_replaces_series_and_keeps_text(recording):
    series, texts = run("noise_signal", recording, None)
    assert series.shape == recording.shape
    assert abs(series.mean()) < 0.2
    assert not np.allclose(series, normalise(recording))
    assert texts == stats_text(*channel_stats(recording))


def test_signal_swap_takes_series_from_donor_only(recording, other_recording):
    series, texts = run("signal_from_other", recording, other_recording)
    np.testing.assert_allclose(series, normalise(other_recording))
    assert texts == stats_text(*channel_stats(recording))


def test_stats_swap_takes_text_from_donor_only(recording, other_recording):
    series, texts = run("stats_from_other", recording, other_recording)
    np.testing.assert_allclose(series, normalise(recording))
    assert texts == stats_text(*channel_stats(other_recording))


def test_no_stats_conditions_use_bare_text(recording):
    assert run("no_stats", recording, None)[1] == bare_text()
    assert run("noise_no_stats", recording, None)[1] == bare_text()


@pytest.mark.parametrize("name", ["signal_from_other", "stats_from_other"])
def test_swap_without_donor_fails_loudly(recording, name):
    with pytest.raises(ValueError, match="donor"):
        run(name, recording, None)


def test_randomness_is_reproducible_per_condition_and_sample(recording):
    a = build_inputs(recording, None, BY_NAME["noise_signal"], condition_rng(2, 17, 42))[0]
    b = build_inputs(recording, None, BY_NAME["noise_signal"], condition_rng(2, 17, 42))[0]
    c = build_inputs(recording, None, BY_NAME["noise_signal"], condition_rng(2, 18, 42))[0]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_condition_names_are_unique():
    names = [c.name for c in CONDITIONS]
    assert len(names) == len(set(names))
