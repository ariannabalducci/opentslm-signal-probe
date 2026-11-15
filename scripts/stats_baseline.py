"""How much of HAR-CoT can be solved from the six numbers the prompt states?

Fits small classifiers on the per-axis mean and std that OpenTSLM writes into
its prompt (rounded to 4 decimals, as the model reads them) and nothing else:
no series, no temporal information. If they score close to OpenTSLM, the text
alone could carry most of the task, and the probe's swap conditions become the
way to tell whether the model actually relies on it.

Runs on a laptop in about 20 s; needs NumPy, pandas and scikit-learn (pinned to
1.7.2 in requirements.txt: newer releases change the gradient-boosting fit, e.g.
1.9.1 gives 0.580 instead of 0.598 on the balanced subset). The library versions
and whether the statistics were computed with torch are written to the output.

    python scripts/download_data.py
    python scripts/stats_baseline.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABELS, SEED  # noqa: E402
from src.data import load_split, prompt_features  # noqa: E402
from src.sampling import stratified_subset  # noqa: E402


def library_versions():
    import pandas
    import sklearn
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pandas.__version__, "scikit-learn": sklearn.__version__}


def stats_backend():
    try:
        import torch
    except ImportError:
        return "numpy"
    return f"torch {torch.__version__}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=ROOT / "data" / "raw", type=Path)
    parser.add_argument("--per-class", default=50, type=int,
                        help="size of the balanced subset the probe uses (default: 50)")
    parser.add_argument("--out", default=ROOT / "results" / "stats_baseline.json", type=Path)
    args = parser.parse_args()

    print("Loading splits...")
    train_x, train_y = load_split(args.data_dir, "train")
    test_x, test_y = load_split(args.data_dir, "test")
    f_train, f_test = prompt_features(train_x), prompt_features(test_x)
    subset = stratified_subset(test_y, args.per_class, SEED)

    models = {
        "majority class": DummyClassifier(strategy="most_frequent"),
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "gradient boosting": HistGradientBoostingClassifier(random_state=SEED),
    }

    results = {
        "features": "per-axis mean and std stated in the prompt (6 numbers)",
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "n_balanced_subset": int(len(subset)),
        "library_versions": library_versions(),
        "statistics_backend": stats_backend(),
        "models": {},
        "subset_predictions": {},
    }
    for name, model in models.items():
        model.fit(f_train, train_y)
        pred = model.predict(f_test)
        results["models"][name] = {
            "test_accuracy": accuracy_score(test_y, pred),
            "test_macro_f1": f1_score(test_y, pred, labels=list(LABELS), average="macro"),
            "subset_accuracy": accuracy_score(test_y[subset], pred[subset]),
            "subset_macro_f1": f1_score(test_y[subset], pred[subset], labels=list(LABELS), average="macro"),
            "per_class_recall_subset": {
                label: float(np.mean(pred[subset][test_y[subset] == label] == label)) for label in LABELS
            },
        }
        # Per-sample predictions on the probe's subset, for the paired test in analyze.py.
        results["subset_predictions"][name] = {str(int(i)): str(pred[i]) for i in subset}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(f"\nTrained on {len(train_y)} samples; six prompt numbers per sample.\n")
    print("| Classifier | Test accuracy | Test macro-F1 | Balanced subset accuracy | Balanced subset macro-F1 |")
    print("|---|---|---|---|---|")
    for name, r in results["models"].items():
        print(f"| {name} | {r['test_accuracy']:.3f} | {r['test_macro_f1']:.3f} "
              f"| {r['subset_accuracy']:.3f} | {r['subset_macro_f1']:.3f} |")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
