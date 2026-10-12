"""Validate the package and launch the canonical fixed-parameter DECAT runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def verify_package(package_root: Path, config: dict) -> None:
    """Check required files and the fixed dataset/split contract without importing ML libraries."""
    checks = [
        ("data", package_root / config["data"]["path"], config["data"]["sha256"]),
        ("split", package_root / config["split"]["path"], config["split"]["sha256"]),
    ]
    sources = config["runtime"]["source_files"]
    if not sources:
        raise ValueError("The configuration contains no source-integrity records.")
    checks.extend(
        (f"source {relative}", package_root / relative, expected)
        for relative, expected in sources.items()
    )
    for label, path, expected in checks:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
            )
    split = json.loads(checks[1][1].read_text(encoding="utf-8-sig"))
    if split["source_sha256"] != config["data"]["sha256"]:
        raise ValueError("Split source SHA-256 does not match the dataset.")
    members = [
        list(map(int, split[key])) for key in ("train_idx", "val_idx", "test_idx")
    ]
    expected_sizes = [
        config["split"][key] for key in ("train_n", "validation_n", "test_n")
    ]
    if [len(part) for part in members] != expected_sizes:
        raise ValueError("Split sizes do not match the configuration.")
    flattened = [index for part in members for index in part]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(
        range(config["data"]["row_count"])
    ):
        raise ValueError(
            "Split members must be disjoint and cover every dataset row once."
        )
    print("Package integrity OK.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate files and split membership without training.",
    )
    parser.add_argument(
        "--output-root", type=Path, help="Directory for local training outputs."
    )
    args = parser.parse_args()
    package_root = Path(__file__).resolve().parent
    config_path = package_root / "configs" / "decat.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    verify_package(package_root, config)
    if args.verify_only:
        return

    for key, value in config["environment"].items():
        os.environ.setdefault(key, str(value))
    os.environ.setdefault(
        "DECAT_DATA_PATH_OVERRIDE", str(package_root / config["data"]["path"])
    )
    os.environ.setdefault(
        "DECAT_SPLIT_PATH_OVERRIDE", str(package_root / config["split"]["path"])
    )
    os.environ.setdefault("DECAT_PARAMS_JSON", json.dumps(config["params"]))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if args.output_root:
        os.environ["DECAT_OUTPUT_ROOT_OVERRIDE"] = str(args.output_root.resolve())
    else:
        os.environ.setdefault(
            "DECAT_OUTPUT_ROOT_OVERRIDE", str(package_root / "outputs" / "training")
        )

    sys.path.insert(0, str(package_root / "src"))
    sys.path.insert(0, str(package_root))
    os.chdir(package_root)
    from decat.training.protocol import main as run_training

    run_training()


if __name__ == "__main__":
    main()
