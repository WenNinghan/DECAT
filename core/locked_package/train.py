"""Validate the original workbook and launch DECAT with supplied molecular inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def verify_package(package_root: Path, config: dict) -> None:
    """Check original data and source integrity without importing ML libraries."""
    checks = [
        ("data", package_root / config["data"]["path"], config["data"]["sha256"]),
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
    print("Package integrity OK.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate the package; also prepare and validate --inputs when supplied.",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        help="UTF-8 CSV with NO., SMILES, category27_label and optional CAS; required for training.",
    )
    parser.add_argument(
        "--output-root", type=Path, help="Directory for local training outputs."
    )
    args = parser.parse_args()
    package_root = Path(__file__).resolve().parent
    config_path = package_root / "configs" / "decat.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    verify_package(package_root, config)
    for key, value in config["environment"].items():
        os.environ.setdefault(key, str(value))
    if not args.inputs:
        if args.verify_only:
            return
        parser.error("Training requires --inputs with structures and category labels.")
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else package_root / "outputs" / "training"
    )
    sys.path.insert(0, str(package_root / "src"))
    from decat.data.inputs import prepare_training_inputs

    data_path, split_path = prepare_training_inputs(
        (package_root / config["data"]["path"]).resolve(),
        args.inputs.resolve(),
        output_root,
        search_seeds=int(config["split"]["search_seeds"]),
    )
    print(f"Training inputs and split prepared in {data_path.parent}", flush=True)
    if args.verify_only:
        return
    os.environ["DECAT_DATA_PATH_OVERRIDE"] = str(data_path)
    os.environ["DECAT_SPLIT_PATH_OVERRIDE"] = str(split_path)
    os.environ.setdefault("DECAT_PARAMS_JSON", json.dumps(config["params"]))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ["DECAT_OUTPUT_ROOT_OVERRIDE"] = str(output_root)

    sys.path.insert(0, str(package_root))
    os.chdir(package_root)
    from decat.training.protocol import main as run_training

    run_training()


if __name__ == "__main__":
    main()
