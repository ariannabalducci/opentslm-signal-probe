import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def recording():
    """A raw (3, 128) recording with distinct level and scale per axis."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, 128, dtype=np.float32)
    return np.stack([
        0.2 + 0.5 * np.sin(3 * t),
        -1.0 + 0.1 * rng.standard_normal(128),
        9.8 + 2.0 * np.cos(t),
    ]).astype(np.float32)


@pytest.fixture
def other_recording():
    rng = np.random.default_rng(1)
    return (rng.standard_normal((3, 128)) * [[0.3], [1.5], [0.7]] + [[1.0], [0.0], [-9.0]]).astype(np.float32)
