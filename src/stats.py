"""Small exact tests for the results section, with no SciPy dependency.

* `wilson`: 95% Wilson interval for a proportion.
* `binom_two_sided`: exact two-sided binomial test against p = 0.5, used on the
  swap answers that follow exactly one of the two channels.
* `mcnemar_exact`: exact McNemar test for two classifiers scored on the same
  samples (the binomial test on the discordant pairs).
* `chance_follows`: how often an answer would match a reference label by chance,
  keeping the model's own label distribution (the expectation over shuffling
  the predictions across samples), with a one-sided permutation p-value.
"""
import math
import random
from collections import Counter

import numpy as np


def wilson(k, n, z=1.959963984540054):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def binom_two_sided(k, n):
    """Exact two-sided binomial test of k out of n against p = 0.5."""
    if n == 0:
        return 1.0
    tail = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(tail + 1)) / 2 ** n
    return min(1.0, 2 * p)


def mcnemar_exact(correct_a, correct_b):
    """Exact McNemar test on paired correctness vectors.

    Returns (only_a, only_b, p): samples only A gets right, only B gets right,
    and the two-sided p-value on those discordant pairs.
    """
    correct_a, correct_b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    return only_a, only_b, binom_two_sided(only_a, only_a + only_b)


def chance_follows(predictions, references, n_perm=10000, seed=0):
    """Chance level for `predictions` matching `references`, and a one-sided p.

    The expected count under shuffling is exact: sum over labels of
    count_pred(label) * count_ref(label) / n. The p-value is the share of
    permutations that match at least as often as observed.
    """
    n = len(references)
    pred_counts, ref_counts = Counter(predictions), Counter(references)
    expected = sum(pred_counts[label] * ref_counts[label] for label in ref_counts) / n
    observed = sum(p == r for p, r in zip(predictions, references))
    # Python's `random` rather than NumPy's Generator: its stream for a given seed
    # is stable across Python and NumPy versions, so CI can check the committed
    # p-values exactly.
    rng = random.Random(seed)
    preds = [str(p) for p in predictions]
    refs = [str(r) for r in references]
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(preds)
        hits += sum(p == r for p, r in zip(preds, refs)) >= observed
    return expected, (hits + 1) / (n_perm + 1)
