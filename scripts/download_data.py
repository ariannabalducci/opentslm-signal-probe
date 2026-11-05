"""Download the released HAR-CoT splits into data/raw/.

Same archive OpenTSLM's loader fetches (about 140 MB zipped, 580 MB of CSV).

    python scripts/download_data.py
"""
import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import HAR_COT_URL, SPLIT_FILES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=ROOT / "data" / "raw", type=Path)
    args = parser.parse_args()

    expected = [args.data_dir / name for name in SPLIT_FILES.values()]
    if all(p.exists() for p in expected):
        print(f"HAR-CoT already present in {args.data_dir}")
        return

    args.data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "har_cot.zip"
        print(f"Downloading {HAR_COT_URL} ...")
        urllib.request.urlretrieve(HAR_COT_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            # Extract only the three expected CSVs, by name, never arbitrary paths.
            for name in SPLIT_FILES.values():
                (args.data_dir / name).write_bytes(zf.read(name))

    missing = [p.name for p in expected if not p.exists()]
    if missing:
        sys.exit(f"Missing after extraction: {missing}")
    print(f"HAR-CoT ready in {args.data_dir}")


if __name__ == "__main__":
    main()
