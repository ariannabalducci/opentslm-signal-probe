"""Load HAR-CoT and rebuild the two channels OpenTSLM reads.

For each axis OpenTSLM computes the mean and the (unbiased) standard deviation
of the raw signal, feeds the z-normalised series to its encoder, and writes the
two statistics into the text that precedes the series:

    "The following is the accelerometer data on the x-axis, it has mean
     -0.1234 and std 0.5678:"

Normalising removes level and scale from the series, so those two numbers
reach the model only through the text. That is the asymmetry the probe tests.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AXES, MIN_STD, SPLIT_FILES, TIME_SERIES_LABELS


def load_split(data_dir, split):
    """Return (signals, labels) for one split.

    signals: float32 array (N, 3, T), axes in x, y, z order.
    labels:  str array (N,).
    """
    path = Path(data_dir) / SPLIT_FILES[split]
    df = pd.read_csv(path, usecols=[*AXES, "label"])
    # The released CSVs repeat their header row in places; OpenTSLM drops these too.
    df = df[df["x_axis"] != "x_axis"].reset_index(drop=True)
    # The series are stored as Python-list literals, which are valid JSON and
    # parse far faster with json than with ast.literal_eval.
    signals = np.stack(
        [np.array([json.loads(s) for s in df[axis]], dtype=np.float32) for axis in AXES],
        axis=1,
    )
    return signals, df["label"].to_numpy(dtype=str)


def channel_stats(signal):
    """Per-axis mean and std, computed as OpenTSLM does (float32, unbiased, clamped).

    Works on a single sample (3, T) or a stack (N, 3, T); reduces the last axis.
    """
    signal = np.asarray(signal, dtype=np.float32)
    means = signal.mean(axis=-1)
    stds = np.maximum(signal.std(axis=-1, ddof=1), np.float32(MIN_STD))
    return means, stds


def normalise(signal):
    """Z-normalise each axis with its own statistics, as OpenTSLM's dataset does."""
    means, stds = channel_stats(signal)
    return (np.asarray(signal, dtype=np.float32) - means[..., None]) / stds[..., None]


def stats_text(means, stds):
    """The three sentences OpenTSLM puts before the three series."""
    return [
        f"{label}, it has mean {m:.4f} and std {s:.4f}:"
        for label, m, s in zip(TIME_SERIES_LABELS, means, stds)
    ]


def bare_text():
    """The same sentences with the statistics removed.

    The model never saw this format in training, so a drop in accuracy under it
    mixes two effects: losing the statistics and reading unfamiliar text. The
    swap conditions in `perturb` avoid that confound and are the main evidence.
    """
    return [f"{label}:" for label in TIME_SERIES_LABELS]


def prompt_features(signals):
    """The six numbers each prompt states, at the precision the model reads them.

    Returns an (N, 6) array: mean x, y, z, then std x, y, z, rounded to 4 decimals.
    """
    means, stds = channel_stats(signals)
    return np.round(np.concatenate([means, stds], axis=-1), 4)
