# Does OpenTSLM read the signal? A probe on HAR-CoT

[![CI](https://github.com/ariannabalducci/opentslm-signal-probe/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ariannabalducci/opentslm-signal-probe/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[OpenTSLM](https://github.com/StanfordBDHG/OpenTSLM) feeds time series to a
language model as a native modality. On human activity recognition (HAR-CoT) it
reads each accelerometer axis through **two channels**:

1. the **z-normalised series**, through its time-series encoder;
2. a **sentence** that states the raw mean and standard deviation of that axis:
   *"The following is the accelerometer data on the x-axis, it has mean -0.1234 and std 0.5678:"*.

Normalising removes level and scale from the series, so those numbers reach the
model only through the text. This repository asks which channel the model's
answer actually depends on.

## Why the question is worth asking

**The six numbers in the prompt already solve most of the task.** A gradient-boosting
classifier that sees only the per-axis mean and std — no series, no temporal
information — gets:

| Classifier on the 6 prompt numbers | Test accuracy | Test macro-F1 | Balanced subset accuracy | Balanced subset macro-F1 |
|---|---|---|---|---|
| majority class | 0.285 | 0.055 | 0.125 | 0.028 |
| logistic regression | 0.533 | 0.330 | 0.352 | 0.272 |
| gradient boosting | **0.689** | **0.606** | 0.598 | 0.586 |

Trained on the 68 542 training samples, tested on all 8 222 test samples and on the
400-sample balanced subset the probe uses (`python scripts/stats_baseline.py`, about
20 s on a laptop). For scale, the OpenTSLM paper reports an F1 of about 65 on HAR.

This does not show that OpenTSLM uses the shortcut, only that the shortcut exists
and is strong. Two further facts make it worth testing:

* Gravity sets the mean of each axis, and the std tracks movement intensity: these
  separate lying from standing and running from sitting without any temporal pattern.
* The training rationales were written by a teacher model that was **told the correct
  label** (the released `prompt` column ends with *"You MUST end your response with
  'Answer: sitting'"*). A model trained on such rationales learns to produce a
  convincing story for a label, not necessarily to derive it from the signal.

## The probe

Every condition changes one channel and keeps the other, on the same 400 test
samples (50 per class, fixed seed). Swapped inputs always come from a recording
of a **different class**.

| Condition | Series | Statistics in the text | Question |
|---|---|---|---|
| `original` | own | own | Reference: the input the model was trained on |
| `shuffled_time` | own, time steps shuffled | own | Does the model use temporal order? |
| `noise_signal` | Gaussian noise | own | How far do the statistics go alone? |
| `signal_from_other` | other recording | own | Conflict: which channel wins? |
| `stats_from_other` | own | other recording | Conflict: which channel wins? |
| `no_stats` | own | removed | How far does the series go alone? |
| `noise_no_stats` | Gaussian noise | removed | Floor |

The two **swap conditions** are the main evidence: the text format stays exactly
the one the model was trained on, and the answer shows whether it follows the
class of the series or the class of the statistics. `no_stats` is reported but
weighed less, because the model never saw sentences without numbers in training.

Details that keep the comparison fair:

* The unperturbed input is rebuilt exactly as OpenTSLM builds it (unbiased std,
  the same clamp, the same 4-decimal formatting), and `check_against_opentslm.py`
  verifies it against OpenTSLM's own dataset class before any run.
* Shuffling uses one permutation for all three axes, so each time step keeps its
  3-D acceleration vector and the stated statistics remain true.
* Generation is greedy. OpenTSLM's batched `generate` right-pads prompts of
  different length, which is unsafe for a decoder-only model; the probe embeds
  each sample separately and left-pads the batch itself, and checks that batched
  and single-sample generation agree before trusting it.
* Answers are parsed like OpenTSLM's evaluation does; an unparseable answer counts
  as wrong and never as a prediction of some class.

## Results

*To be filled in after the GPU run* (`notebooks/run_on_colab.ipynb`, model
`OpenTSLM/llama-3.2-1b-har-sp`). The table produced by `scripts/analyze.py` reports,
per condition, accuracy, macro-F1, the share of unparseable answers and, for the
swap conditions, how often the answer follows the series and how often the statistics.

## Reproduce

Local, CPU only (data, baseline, tests):

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/stats_baseline.py
pytest
```

The model runs need a GPU: open `notebooks/run_on_colab.ipynb` in Colab. The base
LLMs are gated on Hugging Face, so accept the Llama 3.2 (or Gemma 3) licence and
use a read token. Runs are resumable and write raw rationales, so every number in
the results table can be recomputed with `scripts/analyze.py` from the committed
predictions.

## Layout

```
src/        rebuild OpenTSLM's inputs, conditions, sampling, metrics (NumPy only)
scripts/    download_data, stats_baseline, check_against_opentslm, run_probe, analyze
notebooks/  Colab runner
tests/      unit tests for every condition and metric
results/    baseline scores and model predictions
```

## Credits

OpenTSLM and the HAR-CoT data are by Langer et al., *OpenTSLM: Time-Series Language
Models for Reasoning over Multivariate Medical Text- and Time-Series Data*
([arXiv:2510.02410](https://arxiv.org/abs/2510.02410)), released under the MIT licence.
This probe is independent work and is not affiliated with the OpenTSLM authors.
