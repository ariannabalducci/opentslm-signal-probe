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
| logistic regression | 0.533 | 0.331 | 0.355 | 0.276 |
| gradient boosting | **0.689** | **0.606** | 0.598 | 0.586 |

Trained on the 68 542 training samples, tested on all 8 222 test samples and on the
400-sample balanced subset the probe uses (`python scripts/stats_baseline.py`, about
20 s on a laptop). The statistics are computed exactly as OpenTSLM computes them
(torch, float32). The numbers need scikit-learn 1.7.2, pinned in `requirements.txt`:
scikit-learn 1.9.1 fits a different gradient-boosting model (0.580 on the subset), and
the versions used are recorded in `results/stats_baseline.json`. For scale, the
OpenTSLM paper reports an F1 of about 65 on HAR-CoT.

This does not show that OpenTSLM uses the shortcut, only that the shortcut exists
and is strong. Two further facts make it worth testing:

* Gravity sets the mean of each axis, and the std tracks movement intensity: these
  separate lying from standing and running from sitting without any temporal pattern.
* The training rationales were written by a teacher model that saw a plot of the
  window, a choice between two activities, and **the correct label**: the released
  `prompt` column ends with *You MUST end your response with "Answer: sitting":*
  (with the true class). A model trained on such rationales learns to produce a
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
  the same clamp, the same 4-decimal formatting, the same torch float32
  reductions), and `check_against_opentslm.py` verifies it against OpenTSLM's own
  dataset class before any run: on all 8 222 test samples, 0 text mismatches and a
  maximum series difference of 0 (`results/input_check.json`, against OpenTSLM at
  commit 2968f4b).
* Shuffling uses one permutation for all three axes, so each time step keeps its
  3-D acceleration vector and the stated statistics remain true.
* Generation is greedy and runs one sample at a time through OpenTSLM's own
  `generate`, the path its evaluation uses. A left-padded batched path exists, but
  on `llama-3.2-1b-har-sp` the `--check-batching` comparison failed in the Colab run
  (6 of 8 labels identical to single-sample generation; console output, not saved),
  so all reported results use `--batch-size 1` and the script refuses larger batches
  unless `--trust-batching` is passed.
* Answers are parsed with a slightly more lenient version of OpenTSLM's parser
  (first line after the last `Answer:`, spaces and hyphens mapped to `_`); on all
  2 800 outputs it gives the same labels as OpenTSLM's own `extract_answer`
  (`evaluation/opentslm/parse_predictions.py`, after lower-casing). An
  unparseable answer counts as wrong and never as a prediction of some class.

## Results

Model `OpenTSLM/llama-3.2-1b-har-sp`, 400 balanced test samples per condition,
greedy decoding one sample at a time (`notebooks/run_on_colab.ipynb`, T4 GPU).
Every rationale ended in a parseable label. All numbers below come from
`python scripts/analyze.py results/predictions_llama-3.2-1b-har-sp.jsonl`, which
writes them to `results/predictions_llama-3.2-1b-har-sp.analysis.json`; CI recomputes
them and fails if they change. The run's sample and donor indices are in the
`.manifest.json` next to the predictions.

| Condition | Series | Statistics | Accuracy (95% Wilson CI) | Macro-F1 | Follows series | Follows statistics |
|---|---|---|---|---|---|---|
| `original` | own | own | **0.630** (0.582–0.676) | 0.611 | - | - |
| `shuffled_time` | shuffled | own | 0.380 (0.334–0.428) | 0.315 | - | - |
| `noise_signal` | noise | own | 0.362 (0.317–0.411) | 0.298 | - | - |
| `signal_from_other` | other class | own | 0.388 (0.341–0.436) | 0.357 | 61 / 400 | **155 / 400** |
| `stats_from_other` | own | other class | 0.175 (0.141–0.215) | 0.159 | 70 / 400 | **152 / 400** |
| `no_stats` | own | removed | 0.275 (0.234–0.321) | 0.191 | - | - |
| `noise_no_stats` | noise | removed | 0.125 (0.096–0.161) | 0.028 | - | - |

Chance accuracy is 0.125. With neither channel informative the model answers
`biking` for all 400 samples. In the follows columns chance is about 50 / 400 for
either column: the expected count if the model's answers were shuffled across
samples, keeping its own label distribution.

**When the two channels disagree, the answer follows the statistics.** In both swap
conditions the model names the class of the statistics 2.2 to 2.5 times as often as
the class of the series (155 vs 61 and 152 vs 70; two-sided binomial test on the
answers that follow exactly one of the two, p = 1.3·10⁻¹⁰ and 3.8·10⁻⁸). Following
the statistics is about three times chance in both conditions (permutation p < 0.001).
Following the series is above chance in `stats_from_other` (70 vs 50, p = 0.001)
but not clearly in `signal_from_other` (61 vs 50, p = 0.06). The direction holds
whichever channel is swapped, so it is not an artefact of one perturbation.

**The model does read the series, but only part of the task depends on it.**
Shuffling the time steps while keeping the true statistics drops accuracy from 0.63
to 0.38, about the level of replacing the series with noise (0.36): the series
contributes, and its contribution depends on the order of the time steps (or the
encoder treats a shuffled series much like noise).

Pooling both swap conditions and grouping the answers by the class the statistics
come from (series wins vs statistics wins):

| Class of the statistics | Pooled | `original`: answers of this class (of 400) |
|---|---|---|
| running | 7 vs 73 | 55 |
| sitting | 10 vs 74 | 98 |
| standing | 10 vs 53 | 47 |
| walking | 12 vs 45 | 94 |
| biking | 12 vs 29 | 50 |
| walking_down | 27 vs 14 | 19 |
| walking_up | 31 vs 9 | 20 |
| lying | 22 vs 10 | 17 |

The statistics win clearly for `running`, `sitting`, `standing` and `walking`, the
classes that gravity (mean) and movement intensity (std) already separate, and less
sharply for `biking`. The series wins for `walking_up`, `walking_down` and `lying` (for `walking_down` only
when pooled: 8 vs 11 in `signal_from_other`, 19 vs 3 in `stats_from_other`). Those
are also the three labels the model rarely gives even on unperturbed input (17–20
of 400 answers each, against 50 true samples), so "following" their statistics is
capped by the model's own label prior, not only by what it reads in the series.

**Against the shortcut, the gap is small and its significance depends on the
baseline.** On the same 400 samples OpenTSLM reaches 0.630 and the gradient-boosting
classifier that sees only the six numbers in the prompt reaches 0.598. They give the
same label on 75% of the samples; in a paired comparison 41 samples are right only
for the model and 28 only for the classifier (exact McNemar p = 0.15). With
scikit-learn 1.9.1 the classifier reaches 0.580 instead and the same test gives
p = 0.03, so "barely ahead" is the safer reading than "no better". The model also
uses the statistics only coarsely: given the true statistics and a noise series it
reaches 0.36, while the classifier gets 0.60 from the same six numbers.

Limits of what this shows:

* One model, one dataset. The Gemma checkpoint and the Flamingo variant were not run.
* `no_stats` (0.275) mixes losing the statistics with reading a sentence format the
  model never saw, so it is weaker evidence than the swap conditions.
* About 45% of answers under a swap follow neither class; the probe measures which
  channel wins when one does, not why the others go elsewhere.
* The finding is about this trained checkpoint and its prompt format, not about the
  architecture: a model trained without the statistics in the text, or with them
  randomised, would answer a different question.

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
use a read token. The full run (2 800 rationales) takes about 4.5 hours on a T4.
Runs are resumable and write raw rationales, so every number in
the results section can be recomputed with `scripts/analyze.py` from the committed
predictions. Each run's manifest records the probe and OpenTSLM commits, library
versions and GPU (the committed manifest predates this and has only the sample
indices). `check_against_opentslm.py --out` saves the input check, as in
`results/input_check.json`.

## Layout

```
src/        rebuild OpenTSLM's inputs, conditions, sampling, metrics (NumPy; torch when installed)
scripts/    download_data, stats_baseline, check_against_opentslm, run_probe, analyze
notebooks/  Colab runner
tests/      unit tests for every condition and metric
results/    baseline scores, input check, model predictions, manifest and analysis
```

## Credits

OpenTSLM and the HAR-CoT data are by Langer et al., *OpenTSLM: Time-Series Language
Models for Reasoning over Multivariate Medical Text- and Time-Series Data*
([arXiv:2510.02410](https://arxiv.org/abs/2510.02410)). The OpenTSLM code is released
under the MIT licence; HAR-CoT is downloaded from the authors' release, not
redistributed here.
This probe is independent work and is not affiliated with the OpenTSLM authors.
