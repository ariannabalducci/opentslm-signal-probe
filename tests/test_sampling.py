import numpy as np

from src.config import LABELS
from src.sampling import assign_donors, stratified_subset


def labels_fixture():
    rng = np.random.default_rng(0)
    counts = [473, 444, 1057, 2342, 1264, 1508, 542, 592]  # HAR-CoT test split
    labels = np.repeat(np.array(LABELS), counts)
    rng.shuffle(labels)
    return labels


def test_subset_is_balanced_and_reproducible():
    labels = labels_fixture()
    a = stratified_subset(labels, 50, seed=42)
    b = stratified_subset(labels, 50, seed=42)
    np.testing.assert_array_equal(a, b)
    values, counts = np.unique(labels[a], return_counts=True)
    assert list(values) == sorted(LABELS)
    assert set(counts) == {50}
    assert len(np.unique(a)) == len(a)


def test_small_classes_are_taken_whole():
    labels = np.array(["biking"] * 3 + ["lying"] * 10)
    subset = stratified_subset(labels, 5, seed=0)
    assert (labels[subset] == "biking").sum() == 3
    assert (labels[subset] == "lying").sum() == 5


def test_donors_come_from_subset_with_another_label():
    labels = labels_fixture()
    subset = stratified_subset(labels, 20, seed=42)
    donors = assign_donors(labels, subset, seed=43)
    assert donors.shape == subset.shape
    assert set(donors.tolist()) <= set(subset.tolist())
    assert np.all(labels[donors] != labels[subset])
