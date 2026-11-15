"""Turn the probe's JSONL predictions into every number the README reports.

    python scripts/analyze.py results/predictions_llama-3.2-1b-har-sp.jsonl

Writes, next to the input:
  *.summary.csv    one row per condition (accuracy, macro-F1, follows shares)
  *.analysis.json  intervals, swap counts and tests, chance levels, per-class
                   pattern, and the paired comparison with the stats-only baseline
and prints the Markdown tables used in the README.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABELS  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.perturb import BY_NAME, CONDITIONS  # noqa: E402
from src.stats import binom_two_sided, chance_follows, mcnemar_exact, wilson  # noqa: E402

SWAPS = ("signal_from_other", "stats_from_other")


def fmt(value):
    return "-" if value is None else f"{value:.3f}"


def refs(records, condition):
    """(series class, statistics class) for each record of a condition."""
    c = BY_NAME[condition]
    series = [r["donor_label"] if c.signal == "donor" else r["label"] for r in records]
    stats = [r["donor_label"] if c.text == "donor" else r["label"] for r in records]
    return series, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--baseline", type=Path, default=ROOT / "results" / "stats_baseline.json",
                        help="stats_baseline.py output with per-sample subset predictions")
    args = parser.parse_args()

    with args.predictions.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    by = defaultdict(list)
    for r in records:
        by[r["condition"]].append(r)
    for recs in by.values():
        recs.sort(key=lambda r: r["index"])

    # --- summary.csv (unchanged format) ---------------------------------------
    rows = summarise(records, CONDITIONS)
    out = args.predictions.with_suffix(".summary.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    analysis = {"source": args.predictions.name, "chance_accuracy": 1 / len(LABELS), "conditions": {}}

    print("| Condition | Series | Statistics | n | Accuracy (95% Wilson CI) | Macro-F1 | Unparsed |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        recs = by.get(r["condition"], [])
        k = sum(x["prediction"] == x["label"] for x in recs)
        lo, hi = wilson(k, len(recs))
        analysis["conditions"][r["condition"]] = {
            "n": r["n"], "correct": k, "accuracy": r["accuracy"], "wilson_95": [lo, hi],
            "macro_f1": r["macro_f1"], "unparsed": r["unparsed"],
            "prediction_counts": dict(sorted(Counter(str(x["prediction"]) for x in recs).items())),
        }
        print(f"| `{r['condition']}` | {r['series']} | {r['statistics']} | {r['n']} "
              f"| {r['accuracy']:.3f} ({lo:.3f}–{hi:.3f}) | {fmt(r['macro_f1'])} | {fmt(r['unparsed'])} |")

    # --- swap conditions -------------------------------------------------------
    print("\n| Swap condition | Follows series | Follows statistics | Neither | Chance per column | Binomial p (series vs statistics) |")
    print("|---|---|---|---|---|---|")
    pooled = defaultdict(lambda: [0, 0])
    analysis["swaps"] = {}
    for name in SWAPS:
        recs = by.get(name, [])
        if not recs:
            continue
        preds = [r["prediction"] for r in recs]
        series, stats = refs(recs, name)
        s = sum(p == x for p, x in zip(preds, series))
        t = sum(p == x for p, x in zip(preds, stats))
        exp_s, p_s = chance_follows(preds, series)
        exp_t, p_t = chance_follows(preds, stats)
        p = binom_two_sided(t, s + t)
        per_class = defaultdict(lambda: [0, 0])
        for pred, se, st in zip(preds, series, stats):
            per_class[st][0] += pred == se
            per_class[st][1] += pred == st
            pooled[st][0] += pred == se
            pooled[st][1] += pred == st
        analysis["swaps"][name] = {
            "n": len(recs), "follows_series": s, "follows_statistics": t, "neither": len(recs) - s - t,
            "binomial_p_two_sided": p,
            "chance_follows_series": exp_s, "perm_p_follows_series_above_chance": p_s,
            "chance_follows_statistics": exp_t, "perm_p_follows_statistics_above_chance": p_t,
            "by_statistics_class": {k: {"series": v[0], "statistics": v[1]} for k, v in sorted(per_class.items())},
        }
        n = len(recs)
        print(f"| `{name}` | {s} ({s / n:.3f}; p={p_s:.3g}) | {t} ({t / n:.3f}; p={p_t:.3g}) "
              f"| {n - s - t} ({(n - s - t) / n:.3f}) | {exp_s:.1f} / {exp_t:.1f} | {p:.2g} |")
    print("p in the follows columns: one-sided permutation test against chance (10 000 shuffles).")

    print("\n| Class of the statistics | " + " | ".join(f"`{n}` series vs statistics" for n in SWAPS)
          + " | Pooled | `original`: times the model answers this class |")
    print("|---|" + "---|" * (len(SWAPS) + 2))
    original_counts = Counter(r["prediction"] for r in by.get("original", []))
    for label in LABELS:
        cells = []
        for name in SWAPS:
            c = analysis["swaps"].get(name, {}).get("by_statistics_class", {}).get(label)
            cells.append(f"{c['series']} vs {c['statistics']}" if c else "-")
        ps = pooled[label]
        print(f"| {label} | " + " | ".join(cells) + f" | {ps[0]} vs {ps[1]} | {original_counts[label]} |")
    analysis["pooled_by_statistics_class"] = {k: {"series": v[0], "statistics": v[1]} for k, v in sorted(pooled.items())}

    # --- paired comparison with the stats-only baseline ------------------------
    if args.baseline.exists() and by.get("original"):
        base = json.loads(args.baseline.read_text(encoding="utf-8"))
        subset_preds = base.get("subset_predictions", {})
        recs = by["original"]
        analysis["vs_baseline"] = {"library_versions": base.get("library_versions")}
        print("\n| Model on `original` vs stats-only classifier (same samples) | Model | Classifier "
              "| Only model right | Only classifier right | Same label | McNemar exact p |")
        print("|---|---|---|---|---|---|---|")
        for clf, preds in subset_preds.items():
            if clf == "majority class":
                continue
            clf_pred = [preds[str(r["index"])] for r in recs]
            model_ok = [r["prediction"] == r["label"] for r in recs]
            clf_ok = [p == r["label"] for p, r in zip(clf_pred, recs)]
            a, b, p = mcnemar_exact(model_ok, clf_ok)
            same = sum(p_ == r["prediction"] for p_, r in zip(clf_pred, recs)) / len(recs)
            analysis["vs_baseline"][clf] = {
                "model_accuracy": sum(model_ok) / len(recs), "classifier_accuracy": sum(clf_ok) / len(recs),
                "only_model_right": a, "only_classifier_right": b, "same_label_share": same, "mcnemar_p": p,
            }
            print(f"| {clf} | {sum(model_ok) / len(recs):.3f} | {sum(clf_ok) / len(recs):.3f} "
                  f"| {a} | {b} | {same:.3f} | {p:.2g} |")

    path = args.predictions.with_suffix(".analysis.json")
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(analysis, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"\nWritten to {out.name} and {path.name}")


if __name__ == "__main__":
    main()
