"""
Signal probe for OpenTSLM on HAR-CoT.

OpenTSLM reads each accelerometer axis through two channels: a z-normalised
series fed to its time-series encoder, and a sentence stating the raw mean and
standard deviation of that axis. This package rebuilds both channels exactly as
OpenTSLM does, then perturbs one at a time, so the model's predictions show
which channel actually drives its answer.

The modules here need only NumPy and pandas; the GPU part lives in
`scripts/run_probe.py`.
"""

__all__ = ["config", "data", "perturb", "sampling", "metrics"]
__version__ = "0.1.0"
