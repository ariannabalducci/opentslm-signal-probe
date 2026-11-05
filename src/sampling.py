"""Which test samples the probe runs on, and which recording each one borrows from.

Generating a chain-of-thought rationale costs seconds per sample, so the probe
runs on a class-balanced subset of the test split rather than all 8222 samples.
Balancing also keeps macro-F1 from being dominated by `sitting`, which is 28 %
of the test split.
"""
import numpy as np

from .config import LABELS


def stratified_subset(labels, per_class, seed):
    """Indices of up to `per_class` samples per label, sorted, drawn without replacement."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    picked = []
    for label in LABELS:
        pool = np.flatnonzero(labels == label)
        if len(pool) == 0:
            continue
        picked.append(rng.choice(pool, size=min(per_class, len(pool)), replace=False))
    return np.sort(np.concatenate(picked))


def assign_donors(labels, subset, seed):
    """For each index in `subset`, a donor index from `subset` with a different label.

    Donors come from the subset itself, so every recording the swap conditions
    use is also evaluated unperturbed.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    subset = np.asarray(subset)
    subset_labels = labels[subset]
    donors = np.empty_like(subset)
    for i, label in enumerate(subset_labels):
        candidates = subset[subset_labels != label]
        if len(candidates) == 0:
            raise ValueError("Swap conditions need at least two classes in the subset")
        donors[i] = rng.choice(candidates)
    return donors
