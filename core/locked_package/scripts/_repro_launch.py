"""Launch locked DECAT reproduction from the package root.

Patches the hardcoded PROJECT path inside run_decat_v21_blind_validation_fixed.py
so the package is self-contained.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    runner = package_root / "run_decat_v21_blind_validation_fixed.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)

    # Ensure package model code is importable.
    src = package_root / "src"
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(package_root))

    # Force data/split/output if not already set by the PowerShell wrapper.
    os.environ.setdefault(
        "DECAT_DATA_PATH_OVERRIDE",
        str(package_root / "data" / "反应logk指纹数据_25类_含环境无机物_Other_1626条.csv"),
    )
    os.environ.setdefault(
        "DECAT_SPLIT_PATH_OVERRIDE",
        str(package_root / "data" / "固定划分_1626_同分子异pH非均分_75-12.5-12.5.json"),
    )
    os.environ.setdefault("DECAT_CATEGORY_A_FAMILY_OVERRIDE", "other")
    os.environ.setdefault("DECAT_STRICT_SEED", "242")
    os.environ.setdefault("DECAT_MODEL_SEED", "242")
    os.environ.setdefault("DECAT_PARAMETER_PROFILE", "25class_stack")
    os.environ.setdefault("DECAT_FINAL_COMPONENT", "nn_plus_res_rf")
    os.environ.setdefault("DECAT_UNMASK_TEST", "1")
    os.environ.setdefault("DECAT_TEST_GUIDED", "0")
    os.environ.setdefault("DECAT_CONDITION_INTERPOLATION", "1")
    os.environ.setdefault("DECAT_SAVE_ARTIFACTS", "1")
    os.environ.setdefault("DECAT_MAX_EPOCHS", "120")
    os.environ.setdefault("DECAT_EARLY_STOP", "36")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    # Monkeypatch PROJECT before the runner module body uses it:
    # load source, replace PROJECT assignment, exec.
    source = runner.read_text(encoding="utf-8")
    # Rewrite absolute PROJECT line to package root.
    rewritten = []
    for line in source.splitlines(keepends=True):
        if line.startswith("PROJECT = Path("):
            rewritten.append(f"PROJECT = Path(r'''{package_root}''')\n")
        else:
            rewritten.append(line)
    code = compile("".join(rewritten), str(runner), "exec")
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(runner),
        "__package__": None,
    }
    # Provide package-local src/decat to imports that use the package src layout
    os.chdir(package_root)
    exec(code, globals_dict)


if __name__ == "__main__":
    main()
