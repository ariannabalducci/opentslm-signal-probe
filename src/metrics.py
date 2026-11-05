"""Parse the model's answers and score each condition.

Unparseable answers count as wrong, never as a prediction of some class: they
add a false negative to the true class and no false positive anywhere. This is
the same convention as OpenTSLM's own HAR evaluation with `allowed_labels`.
"""
import re
from collections import defaultdict

from .config import LABELS

ANSWER_MARKER = "Answer:"
_SPECIAL_TOKENS = re.compile(r"<\|.*?\|>")


def extract_label(text):
    """The label after the last 'Answer:' in a generated rationale, or None.

    Follows OpenTSLM's parser (take what follows the last 'Answer:', drop
    special tokens), then also accepts 'walking up' and 'walking-up' for
    'walking_up'. Anything that is still not one of the eight labels is None.
    """
    if ANSWER_MARKER not in text:
        return None
    answer = _SPECIAL_TOKENS.sub("", text.rsplit(ANSWER_MARKER, 1)[-1])
    lines = answer.strip().splitlines()
    if not lines:
        return None
    answer = lines[0].strip().lower().strip(".,!?;:\"'* ")
    answer = re.sub(r"[\s-]+", "_", answer)
    return answer if answer in LABELS else None


def accuracy(predictions, references):
    if not references:
        return float("nan")
    return sum(p == r for p, r in zip(predictions, references)) / len(references)


def macro_f1(predictions, references):
    """Macro-F1 over the eight labels; None predictions only add false negatives."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for p, r in zip(predictions, references):
        if p == r:
            tp[r] += 1
            continue
        fn[r] += 1
        if p in LABELS:
            fp[p] += 1
    scores = []
    for label in LABELS:
        if tp[label] + fp[label] + fn[label] == 0:
            continue  # label absent from both references and predictions
        denom = 2 * tp[label] + fp[label] + fn[label]
        scores.append(2 * tp[label] / denom)
    return sum(scores) / len(scores) if scores else float("nan")


def summarise(records, conditions):
    """One row per condition from the probe's JSONL records.

    Each record has: condition, label, donor_label, prediction.
    For a swap condition, `follows_signal` and `follows_text` give the share of
    answers that match the class of the series and of the stated statistics.
    """
    by_condition = defaultdict(list)
    for record in records:
        by_condition[record["condition"]].append(record)

    rows = []
    for condition in conditions:
        recs = by_condition.get(condition.name, [])
        preds = [r["prediction"] for r in recs]
        own = [r["label"] for r in recs]
        row = {
            "condition": condition.name,
            "series": condition.signal,
            "statistics": condition.text,
            "n": len(recs),
            "accuracy": accuracy(preds, own),
            "macro_f1": macro_f1(preds, own),
            "unparsed": sum(p is None for p in preds) / len(preds) if preds else float("nan"),
            "follows_signal": None,
            "follows_text": None,
        }
        if "donor" in (condition.signal, condition.text):
            donor = [r["donor_label"] for r in recs]
            signal_ref = donor if condition.signal == "donor" else own
            text_ref = donor if condition.text == "donor" else own
            row["follows_signal"] = accuracy(preds, signal_ref)
            row["follows_text"] = accuracy(preds, text_ref)
        rows.append(row)
    return rows
