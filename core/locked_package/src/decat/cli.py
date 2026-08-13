# -*- coding: utf-8 -*-
"""Command line entrypoint for the self-contained DECAT V9 package."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "LOCKED_SEED242_INTERNAL_HOLDOUT.json"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "输出" / "transformer_v9_transformer_centered"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    return dict(json.loads(path.read_text(encoding="utf-8-sig")) or {})


def _project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def _resolve_cfg_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    out["data_csv_path"] = str(_project_path(out.get("data_csv_path", "data/反应logk指纹数据_25类.csv")))
    out["fixed_split_json"] = str(_project_path(out.get("fixed_split_json", "data/固定划分_1637继承1696成员归属.json")))
    out["output_root"] = str(_project_path(out.get("output_root", "输出/transformer_v9_transformer_centered")))
    return out


def _write_runtime_config(cfg: dict[str, Any], base_config_path: Path) -> Path:
    runtime_dir = PROJECT_DIR / "输出" / "_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    payload = _resolve_cfg_paths(cfg)
    env = dict(payload.get("env", {}) or {})
    env["TRANSFORMER_V7_DATA_CSV"] = payload["data_csv_path"]
    env["TRANSFORMER_V7_FIXED_SPLIT_JSON"] = payload["fixed_split_json"]
    env["TRANSFORMER_V6_FIXED_SPLIT_JSON"] = payload["fixed_split_json"]
    payload["env"] = env
    payload["script"] = "src/decat/transformer_v9_transformer_centered.py"
    payload["run_script"] = "python -m decat"
    payload["config_source"] = str(base_config_path)
    runtime_config = runtime_dir / "decat_v9_runtime_config.json"
    runtime_config.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return runtime_config


def _apply_runtime_env(cfg: dict[str, Any], runtime_config: Path, *, output_root: Path, max_epochs: int | None, early_stop: int | None, device: str | None, amp: bool | None) -> None:
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["DECAT_PROJECT_DIR"] = str(PROJECT_DIR)
    os.environ["DECAT_OUTPUT_ROOT"] = str(output_root)
    os.environ["TRANSFORMER_V9_OUTPUT_ROOT"] = str(output_root)
    os.environ["TRANSFORMER_V7_FIXED_PARAMS_JSON"] = str(runtime_config)
    os.environ["TRANSFORMER_V7_MODE"] = "fixed_json"
    os.environ["TRANSFORMER_DETERMINISTIC"] = "1"
    os.environ.setdefault("TRANSFORMER_V3_ENABLE_FUSION", "1")
    os.environ.setdefault("TRANSFORMER_V3_SKIP_ARTIFACTS", "0")
    os.environ.setdefault("TRANSFORMER_V3_DEFER_BEST_EXPORT", "0")
    os.environ.setdefault("TRANSFORMER_CMA_ENABLE", "0")
    os.environ.setdefault("TRANSFORMER_EXPORT_UNIQUE_ATOM_GROUPS", "0")

    if device:
        os.environ["TRANSFORMER_DEVICE"] = device
    if amp is not None:
        os.environ["TRANSFORMER_AMP"] = "1" if amp else "0"
    if max_epochs is not None:
        os.environ["TRANSFORMER_V7_FIXED_MAX_EPOCHS"] = str(int(max_epochs))
    if early_stop is not None:
        os.environ["TRANSFORMER_V7_FIXED_EARLY_STOP"] = str(int(early_stop))

    for key, value in dict(cfg.get("env", {}) or {}).items():
        os.environ.setdefault(str(key), str(value))


def _check(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg_paths(_load_json(args.config))
    checks = [
        ("project_dir", PROJECT_DIR, PROJECT_DIR.is_dir()),
        ("config", args.config, args.config.is_file()),
        ("data_csv_path", Path(cfg["data_csv_path"]), Path(cfg["data_csv_path"]).is_file()),
        ("fixed_split_json", Path(cfg["fixed_split_json"]), Path(cfg["fixed_split_json"]).is_file()),
        ("core_module", PACKAGE_DIR / "transformer_v9_transformer_centered.py", (PACKAGE_DIR / "transformer_v9_transformer_centered.py").is_file()),
    ]
    ok = True
    for name, path, passed in checks:
        print(f"[DECAT check] {name:<18} {'OK' if passed else 'MISSING'} {path}")
        ok = ok and bool(passed)
    try:
        importlib.import_module("decat.transformer_v9_transformer_centered")
        print("[DECAT check] import_core        OK")
    except Exception as exc:
        print(f"[DECAT check] import_core        FAILED {exc!r}")
        ok = False
    return 0 if ok else 1


def _run_fixed(args: argparse.Namespace) -> int:
    cfg_raw = _load_json(args.config)
    cfg = _resolve_cfg_paths(cfg_raw)
    output_root = Path(args.output_root or cfg.get("output_root") or DEFAULT_OUTPUT_ROOT).resolve()
    runtime_config = _write_runtime_config(cfg, args.config.resolve())
    _apply_runtime_env(
        cfg,
        runtime_config,
        output_root=output_root,
        max_epochs=args.max_epochs,
        early_stop=args.early_stop,
        device=args.device,
        amp=args.amp,
    )
    if str(PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR))
    from . import transformer_v9_transformer_centered as v9

    print("[DECAT] project:", PROJECT_DIR)
    print("[DECAT] config:", args.config.resolve())
    print("[DECAT] runtime config:", runtime_config)
    print("[DECAT] output root:", output_root)
    print("[DECAT] data:", cfg["data_csv_path"])
    print("[DECAT] split:", cfg["fixed_split_json"])
    return int(v9.main() or 0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DECAT V9 self-contained workflow.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="DECAT locked V9 config.")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run the locked fixed-parameter V9 workflow.")
    run.add_argument("--output-root", type=Path, default=None, help="Output root, defaults to config output_root.")
    run.add_argument("--max-epochs", type=int, default=None, help="Override fixed max epochs.")
    run.add_argument("--early-stop", type=int, default=None, help="Override fixed early-stop patience.")
    run.add_argument("--device", type=str, default=None, help="Override device: cuda or cpu.")
    run.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable AMP.")

    smoke = sub.add_parser("smoke", help="One-epoch CPU smoke test.")
    smoke.add_argument("--output-root", type=Path, default=None, help="Output root.")

    sub.add_parser("check", help="Check package paths and imports.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "run"
    if args.command == "check":
        return _check(args)
    if args.command == "smoke":
        args.max_epochs = 1
        args.early_stop = 1
        args.device = "cpu"
        args.amp = False
        return _run_fixed(args)
    return _run_fixed(args)


if __name__ == "__main__":
    raise SystemExit(main())
