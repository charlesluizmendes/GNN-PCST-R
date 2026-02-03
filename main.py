import argparse
import sys
from pathlib import Path

from src import auto_clean, build_snapshots, generate_instances, populate_labels
from test import validate_dataset

def _run(mod, argv):
    prev = sys.argv
    sys.argv = argv
    try:
        mod.main()
    finally:
        sys.argv = prev

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="input")
    ap.add_argument("--output_dir", default="output")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    input_dir = str((root / args.input_dir).resolve())
    out_base = (root / args.output_dir).resolve()

    snapshots_dir = str((out_base / "snapshots").resolve())
    instances_dir = str((out_base / "instances").resolve())
    labels_dir = str((out_base / "labels").resolve())
    validate_dir = str((out_base / "validate").resolve())

    _run(
        auto_clean,
        [
            "auto_clean.py",
            "--input_dir",
            str(out_base),
        ],
    )

    _run(
        build_snapshots,
        [
            "build_snapshots.py",
            "--input_dir",
            input_dir,
            "--output_dir",
            snapshots_dir,
        ],
    )

    _run(
        generate_instances,
        [
            "generate_instances.py",
            "--input_dir",
            snapshots_dir,
            "--output_dir",
            instances_dir,
        ],
    )

    _run(
        populate_labels,
        [
            "populate_labels.py",
            "--input_dir",
            snapshots_dir,
            instances_dir,
            "--output_dir",
            labels_dir,
        ],
    )

    _run(
        validate_dataset,
        [
            "validate_dataset.py",
            "--input_dir",
            str(out_base),
            "--output_dir",
            validate_dir,
        ],
    )

if __name__ == "__main__":
    main()