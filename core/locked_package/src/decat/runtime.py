"""Run state, reproducibility controls and path handling."""

from __future__ import annotations
import os
import random
import warnings
from datetime import datetime
from typing import Optional, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from rdkit import RDLogger
import decat.config as _config

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
SCRIPT_DIR = os.path.dirname(__file__)


def _resolve_default_data_csv_path() -> str:
    override = str(
        os.environ.get(
            "DECAT_RUNTIME_DATA_CSV",
            os.environ.get(
                "DECAT_RUNTIME_DATA_CSV_FALLBACK",
                os.environ.get("DECAT_RUNTIME_DATA_CSV_FALLBACK_2", ""),
            ),
        )
    ).strip()
    if override and os.path.isfile(override):
        return override
    candidates = [
        os.path.join(
            os.path.dirname(os.path.dirname(SCRIPT_DIR)),
            "data",
            "decat_features_1626.csv",
        )
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[-1]


def _resolve_output_root() -> str:
    """Resolve the base output directory for the current project."""
    override = str(
        os.environ.get(
            "DECAT_RUNTIME_OUTPUT_ROOT", os.environ.get("DECAT_OUTPUT_ROOT", "")
        )
    ).strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(
        os.path.dirname(os.path.dirname(SCRIPT_DIR)), "outputs", "training"
    )


def _get_data_csv_path() -> str:
    path = _resolve_default_data_csv_path()
    return path if path else context.data_csv_path


DATASET_CACHE_MAX_ENTRIES = 12


def get_run_output_dir(prefix: Optional[str] = None) -> str:
    base = context.run_output_dir if context.run_output_dir else context.out_dir
    os.makedirs(base, exist_ok=True)
    if prefix:
        path = os.path.join(base, str(prefix))
        os.makedirs(path, exist_ok=True)
        return path
    return base


def _set_global_random_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _config._env_bool("DECAT_RUNTIME_DETERMINISTIC", False):
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _init_run_output_dir() -> str:
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(context.out_dir, run_name)
    if os.path.exists(run_dir):
        suffix = 1
        while True:
            candidate = os.path.join(context.out_dir, f"{run_name}_{suffix}")
            if not os.path.exists(candidate):
                run_dir = candidate
                break
            suffix += 1
    os.makedirs(run_dir, exist_ok=True)
    context.run_output_dir = run_dir
    return str(context.run_output_dir)


def _to_prefix_model_name(model_mode: str) -> str:
    m = str(model_mode).strip().lower()
    if m == "attn":
        return "Attn"
    if m == "mlp":
        return "MLP"
    return "Dual"


def _get_objective_target() -> str:
    t = (
        str(
            _config._env_value(
                "DECAT_RUNTIME_OBJECTIVE_TARGET",
                "DECAT_RUNTIME_OBJECTIVE_TARGET_FALLBACK",
                _config._env_value(
                    "DECAT_RUNTIME_OBJECTIVE_TARGET_FALLBACK_2",
                    "DECAT_RUNTIME_OBJECTIVE_TARGET_FALLBACK_3",
                    context.objective_target,
                ),
            )
        )
        .strip()
        .lower()
    )
    if t not in {"val", "test"}:
        t = str(context.objective_target)
    return t


def _reset_best_result_state(objective_target: str = "val") -> None:
    context.best_result.clear()
    context.best_result.update(
        {
            "iter": 0,
            "objective_target": str(objective_target),
            "objective_r2": float("-inf"),
            "val_r2": float("-inf"),
            "val_rmse": float("nan"),
            "test_r2": float("nan"),
            "test_rmse": float("nan"),
            "train_r2": float("nan"),
            "train_rmse": float("nan"),
            "model_mode": "",
            "fusion_mode": "",
            "params": {},
        }
    )


from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class TrainingHooks:
    """Explicit protocol dependencies; None selects the standard implementation."""

    build_split: Callable | None = None
    subset_factory: Callable | None = None
    metrics: Callable | None = None
    fusion: Callable | None = None
    model_factory: Callable | None = None
    fixed_params_provider: Callable | None = None


@dataclass
class RunContext:
    """State for one training run, shared without copied module globals."""

    data_csv_path: str = ""
    out_dir: str = ""
    best_result: dict = field(default_factory=dict)
    iteration: int = 0
    split_cache: dict = field(default_factory=dict)
    fp_rank_cache: dict = field(default_factory=dict)
    dataset_base_cache: dict = field(default_factory=dict)
    bit_substructure_cache: dict = field(default_factory=dict)
    objective_target: str = "val"
    run_output_dir: str | None = None
    last_trained_model: Any = None
    hooks: TrainingHooks = field(default_factory=TrainingHooks)


context = RunContext()


def resolve_hook(name: str, default):
    """Return the explicit training-protocol override, or the normal implementation."""
    configured = getattr(context.hooks, name)
    return default if configured is None else configured


context.data_csv_path = _resolve_default_data_csv_path()
context.out_dir = _resolve_output_root()
context.best_result: Dict[str, object] = {
    "iter": 0,
    "objective_target": "val",
    "objective_r2": float("-inf"),
    "val_r2": float("-inf"),
    "val_rmse": float("nan"),
    "test_r2": float("nan"),
    "test_rmse": float("nan"),
    "train_r2": float("nan"),
    "train_rmse": float("nan"),
    "model_mode": "",
    "fusion_mode": "",
    "params": {},
}
context.iteration = 0
context.split_cache: Dict[str, np.ndarray] = {}
context.fp_rank_cache: Dict[Tuple[int, str], np.ndarray] = {}
context.dataset_base_cache: Dict[Tuple[str, int, int, int, int], Dict[str, object]] = {}
context.bit_substructure_cache: Dict[
    Tuple[str, int, int, Tuple[int, ...], Tuple[int, ...]], Dict[int, Dict[str, object]]
] = {}
context.objective_target = "val"
context.run_output_dir: Optional[str] = None
context.last_trained_model: Optional[nn.Module] = None
os.makedirs(context.out_dir, exist_ok=True)
