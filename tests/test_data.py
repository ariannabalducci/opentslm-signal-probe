import numpy as np

from src.config import TIME_SERIES_LABELS
from src.data import bare_text, channel_stats, normalise, prompt_features, stats_text


def test_stats_match_unbiased_torch_convention(recording):
    means, stds = channel_stats(recording)
    np.testing.assert_allclose(means, recording.mean(axis=1), rtol=1e-6)
    # torch.Tensor.std defaults to the unbiased estimator; OpenTSLM relies on that default.
    np.testing.assert_allclose(stds, recording.std(axis=1, ddof=1), rtol=1e-5)


def test_std_is_clamped_like_opentslm():
    flat = np.ones((3, 128), dtype=np.float32)
    _, stds = channel_stats(flat)
    assert np.all(stds == np.float32(1e-6))


def test_normalised_series_have_zero_mean_unit_std(recording):
    series = normalise(recording)
    np.testing.assert_allclose(series.mean(axis=1), 0, atol=1e-5)
    np.testing.assert_allclose(series.std(axis=1, ddof=1), 1, atol=1e-4)


def test_stats_text_is_opentslm_format(recording):
    texts = stats_text(*channel_stats(recording))
    means, stds = channel_stats(recording)
    assert texts[0] == f"{TIME_SERIES_LABELS[0]}, it has mean {means[0]:.4f} and std {stds[0]:.4f}:"
    assert len(texts) == 3


def test_bare_text_drops_only_the_statistics():
    assert bare_text() == [f"{label}:" for label in TIME_SERIES_LABELS]


def test_prompt_features_are_the_six_stated_numbers(recording, other_recording):
    stack = np.stack([recording, other_recording])
    features = prompt_features(stack)
    assert features.shape == (2, 6)
    means, stds = channel_stats(recording)
    np.testing.assert_allclose(features[0], np.round(np.concatenate([means, stds]), 4))
