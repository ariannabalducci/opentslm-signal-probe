"""The probe's conditions.

A condition chooses, independently, where the series comes from and where the
stated statistics come from. Keeping the two factors separate is what makes the
swap conditions readable: when the series belongs to one recording and the
statistics to another recording of a different class, the prediction shows
which of the two the model follows.
"""
from dataclasses import dataclass

import numpy as np

from .data import bare_text, channel_stats, normalise, stats_text

SIGNAL_SOURCES = ("own", "shuffled", "noise", "donor")
TEXT_SOURCES = ("own", "donor", "none")


@dataclass(frozen=True)
class Condition:
    name: str
    signal: str  # one of SIGNAL_SOURCES
    text: str  # one of TEXT_SOURCES
    question: str


CONDITIONS = (
    Condition("original", "own", "own",
              "Reference: the input the model was trained on"),
    Condition("shuffled_time", "shuffled", "own",
              "Does the model use the temporal order of the series?"),
    Condition("noise_signal", "noise", "own",
              "How far do the stated statistics go on their own?"),
    Condition("signal_from_other", "donor", "own",
              "Conflict: series from a recording of another class"),
    Condition("stats_from_other", "own", "donor",
              "Conflict: statistics from a recording of another class"),
    Condition("no_stats", "own", "none",
              "How far does the series go on its own? (unseen text format)"),
    Condition("noise_no_stats", "noise", "none",
              "Floor: neither channel carries information"),
)
BY_NAME = {c.name: c for c in CONDITIONS}


def condition_rng(condition_index, sample_index, seed):
    """Independent, reproducible randomness per (condition, sample).

    Seeding per pair rather than per run means a resumed run draws exactly the
    same noise and permutations as an uninterrupted one.
    """
    return np.random.default_rng([seed, condition_index, sample_index])


def build_inputs(signal, donor, condition, rng):
    """Inputs for one sample under one condition.

    signal, donor: raw (3, T) recordings; donor may be None when unused.
    Returns (series, texts): a (3, T) float32 array of normalised series and
    the three sentences that precede them.
    """
    if condition.signal == "own":
        series = normalise(signal)
    elif condition.signal == "shuffled":
        # One permutation for all three axes: every time step keeps its 3-D
        # acceleration vector and only the order is lost. Mean and std are
        # permutation-invariant, so the stated statistics stay truthful.
        series = normalise(signal)[:, rng.permutation(signal.shape[-1])]
    elif condition.signal == "noise":
        series = rng.standard_normal(np.shape(signal)).astype(np.float32)
    elif condition.signal == "donor":
        series = normalise(_require(donor, condition))
    else:
        raise ValueError(f"Unknown signal source {condition.signal!r}")

    if condition.text == "own":
        texts = stats_text(*channel_stats(signal))
    elif condition.text == "donor":
        texts = stats_text(*channel_stats(_require(donor, condition)))
    elif condition.text == "none":
        texts = bare_text()
    else:
        raise ValueError(f"Unknown text source {condition.text!r}")

    return series, texts


def _require(donor, condition):
    if donor is None:
        raise ValueError(f"Condition {condition.name!r} needs a donor recording")
    return donor
