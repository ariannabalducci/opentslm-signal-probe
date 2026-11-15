"""Check that the probe's unperturbed input is exactly what OpenTSLM builds.

Loads the test split through OpenTSLM's own HARCoTQADataset and compares, for
the first N samples, the three sentences and the three normalised series with
the probe's `original` condition. Run once on Colab before the probe: every
other condition is a controlled change of this input, so it has to match.

    python scripts/check_against_opentslm.py --data-dir ../OpenTSLM/src/data/har_cot
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_split  # noqa: E402
from src.perturb import BY_NAME, build_inputs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=ROOT / "data" / "raw", type=Path)
    parser.add_argument("-n", default=500, type=int)
    parser.add_argument("--out", default=None, type=Path, help="also write the result as JSON")
    args = parser.parse_args()

    from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset

    reference = HARCoTQADataset("test", EOS_TOKEN="")
    signals, labels = load_split(args.data_dir, "test")
    assert len(reference) == len(labels), (len(reference), len(labels))

    text_mismatch, series_err = 0, 0.0
    for i in range(min(args.n, len(labels))):
        ref = reference[i]
        assert ref["label"] == labels[i], f"sample {i}: label order differs"
        series, texts = build_inputs(signals[i], None, BY_NAME["original"], None)
        if list(ref["time_series_text"]) != texts:
            text_mismatch += 1
            if text_mismatch <= 3:
                print(f"sample {i}:\n  opentslm: {ref['time_series_text']}\n  probe:    {texts}")
        series_err = max(series_err, float(np.abs(np.asarray(ref["time_series"]) - series).max()))

    checked = min(args.n, len(labels))
    print(f"Checked {checked} samples: {text_mismatch} text mismatches, "
          f"max series difference {series_err:.2e}")
    if args.out:
        import opentslm
        import torch
        record = {
            "checked_samples": checked, "text_mismatches": text_mismatch,
            "max_series_difference": series_err,
            "probe_commit": _git_head(ROOT),
            "opentslm_commit": _git_head(Path(opentslm.__file__).resolve().parent),
            "torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
        }
        with args.out.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(record, f, indent=1)
            f.write("\n")
        print(f"Written to {args.out}")
    sys.exit(0 if text_mismatch == 0 and series_err < 1e-4 else 1)


def _git_head(path):
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=path, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
