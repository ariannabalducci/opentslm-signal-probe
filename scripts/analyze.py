"""Turn the probe's JSONL predictions into the results table.

    python scripts/analyze.py results/predictions_llama-3.2-1b-har-sp.jsonl

Writes a CSV next to the input and prints a Markdown table for the README.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import summarise  # noqa: E402
from src.perturb import CONDITIONS  # noqa: E402


def fmt(value):
    return "-" if value is None else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()

    with args.predictions.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    rows = summarise(records, CONDITIONS)

    out = args.predictions.with_suffix(".summary.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("| Condition | Series | Statistics | n | Accuracy | Macro-F1 | Unparsed | Follows series | Follows statistics |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| `{r['condition']}` | {r['series']} | {r['statistics']} | {r['n']} "
              f"| {fmt(r['accuracy'])} | {fmt(r['macro_f1'])} | {fmt(r['unparsed'])} "
              f"| {fmt(r['follows_signal'])} | {fmt(r['follows_text'])} |")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
