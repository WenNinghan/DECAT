"""Canonical fixed-parameter training and evaluation workflow."""

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple
from skopt.utils import use_named_args
import decat.config as _config
import decat.plotting.predictions as _plotting_predictions
import decat.runtime as _runtime
import decat.training.artifacts as _training_artifacts
import decat.training.steps as _steps


@use_named_args(_config.space)
def objective(
    fp_bits: int,
    topk_features: int,
    d_model: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    max_fp_tokens: int,
    n_layers: int,
    n_heads: int,
    model_mode: str,
    fp_select_method: str,
    enable_chem_attn_bias: bool,
    norm_first: bool,
    topk_ckpt_ensemble: int,
    use_ckpt_ensemble: bool,
    lambda_inv: float,
    lambda_dro: float,
    lambda_physics: float,
    dro_tau: float,
    generalization_penalty: float,
) -> float:
    parameters = _steps.TrialParameters(
        fp_bits=fp_bits,
        topk_features=topk_features,
        d_model=d_model,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        max_fp_tokens=max_fp_tokens,
        n_layers=n_layers,
        n_heads=n_heads,
        model_mode=model_mode,
        fp_select_method=fp_select_method,
        enable_chem_attn_bias=enable_chem_attn_bias,
        norm_first=norm_first,
        topk_ckpt_ensemble=topk_ckpt_ensemble,
        use_ckpt_ensemble=use_ckpt_ensemble,
        lambda_inv=lambda_inv,
        lambda_dro=lambda_dro,
        lambda_physics=lambda_physics,
        dro_tau=dro_tau,
        generalization_penalty=generalization_penalty,
    )
    prepared = _steps.prepare_trial(parameters, _runtime.context)
    if prepared is None:
        return 1000000.0
    checkpoints = _steps.fit_checkpoints(prepared)
    evaluated = _steps.evaluate_fusion(prepared, checkpoints)
    return _steps.record_trial(prepared, checkpoints, evaluated)


def _load_fixed_params_payload() -> Tuple[Dict[str, object], str, list]:
    path = str(
        _config._env_value(
            "DECAT_RUNTIME_FIXED_PARAMS_JSON",
            "DECAT_RUNTIME_FIXED_PARAMS_JSON_FALLBACK",
            _config._env_value("DECAT_RUNTIME_FIXED_PARAMS_JSON_FALLBACK_2", None, ""),
        )
    ).strip()
    if not path:
        path = _training_artifacts._find_latest_best_params_json() or ""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            "Cannot find fixed-params json. Set DECAT_RUNTIME_FIXED_PARAMS_JSON to an existing file."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = payload.get("params", payload)
    vec = _config._params_to_bo_vector(params)
    if vec is None:
        raise ValueError(f"Failed to convert params json to BO vector: {path}")
    return (payload, path, list(vec))


def _resolve_fixed_split_path_from_payload(
    payload: Optional[Dict[str, object]] = None
) -> str:
    payload = payload or {}
    explicit = str(
        _config._env_value(
            "DECAT_RUNTIME_FIXED_SPLIT_JSON",
            "DECAT_RUNTIME_FIXED_SPLIT_JSON_FALLBACK",
            _config._env_value("DECAT_RUNTIME_FIXED_SPLIT_JSON_FALLBACK_2", None, ""),
        )
    ).strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    payload_split = str(payload.get("fixed_split_json", "")).strip()
    if payload_split and os.path.isfile(payload_split):
        return payload_split
    candidates = [
        os.path.join(
            os.path.dirname(os.path.dirname(_runtime.SCRIPT_DIR)), "data", "split.json"
        ),
        os.path.join(
            os.path.dirname(os.path.dirname(_runtime.SCRIPT_DIR)), "data", "split.json"
        ),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return payload_split or explicit or candidates[0]


def run_formal_dual_innov_best() -> Optional[str]:
    """Run the optional dual-expert preset through the shared training pipeline."""
    fixed_split_path = os.path.join(
        os.path.dirname(os.path.dirname(_runtime.SCRIPT_DIR)), "data", "split.json"
    )
    env_updates = {
        "DECAT_RUNTIME_MODE": "formal_dual_innov_best",
        "DECAT_RUNTIME_OBJECTIVE_TARGET": "val",
        "DECAT_RUNTIME_SKIP_ARTIFACTS": "0",
        "DECAT_RUNTIME_DEFER_BEST_EXPORT": "0",
        "DECAT_RUNTIME_USE_WARM_START": "0",
        "DECAT_RUNTIME_PROGRESS_LOG": "0",
        "DECAT_RUNTIME_DETERMINISTIC": "1",
        "DECAT_RUNTIME_SEED": str(
            _config._env_value(
                "DECAT_RUNTIME_FORMAL_SEED", "DECAT_RUNTIME_FORMAL_SEED_FALLBACK", "42"
            )
        ),
        "DECAT_RUNTIME_MAX_EPOCHS": str(
            _config._env_value(
                "DECAT_RUNTIME_FORMAL_MAX_EPOCHS",
                "DECAT_RUNTIME_FORMAL_MAX_EPOCHS_FALLBACK",
                "36",
            )
        ),
        "DECAT_RUNTIME_EARLY_STOP": str(
            _config._env_value(
                "DECAT_RUNTIME_FORMAL_EARLY_STOP",
                "DECAT_RUNTIME_FORMAL_EARLY_STOP_FALLBACK",
                "8",
            )
        ),
        "DECAT_RUNTIME_MIN_DELTA": str(
            _config._env_value(
                "DECAT_RUNTIME_FORMAL_MIN_DELTA",
                "DECAT_RUNTIME_FORMAL_MIN_DELTA_FALLBACK",
                "1e-4",
            )
        ),
    }
    if os.path.isfile(fixed_split_path):
        env_updates["DECAT_RUNTIME_FIXED_SPLIT_JSON"] = fixed_split_path
        env_updates["DECAT_RUNTIME_FIXED_SPLIT_JSON_FALLBACK"] = fixed_split_path
    backup = _config._temporary_env_set(env_updates)
    try:
        _runtime.context.iteration = 0
        _runtime.context.split_cache = {}
        _runtime.context.fp_rank_cache = {}
        _runtime.context.run_output_dir = None
        _runtime._reset_best_result_state(objective_target="val")
        _plotting_predictions._apply_plot_defaults_if_needed()
        run_output_dir = _runtime._init_run_output_dir()
        print("=" * 108)
        print("DECAT formal dual-innovation run start")
        print("Output dir:", run_output_dir)
        print("Formal vector:", _config.FORMAL_DUAL_INNOV_BEST_VECTOR)
        print("=" * 108)
        result = objective(list(_config.FORMAL_DUAL_INNOV_BEST_VECTOR))
        if result is not None:
            print(f"[DECAT] Formal run objective (val R2) = {-float(result):.6f}")
        save_path = _training_artifacts._save_best_result()
        if save_path:
            print("[DECAT] Saved best params:", save_path)
        return run_output_dir
    finally:
        _config._temporary_env_restore(backup)


def run_fixed_params_once() -> Optional[str]:
    """Run one fixed-parameter JSON configuration without Bayesian optimization."""
    payload, config_path, vec = _runtime.resolve_hook(
        "fixed_params_provider", _load_fixed_params_payload
    )()
    fixed_split_path = _resolve_fixed_split_path_from_payload(payload)
    objective_target = (
        str(
            _config._env_value(
                "DECAT_RUNTIME_FIXED_OBJECTIVE_TARGET",
                "DECAT_RUNTIME_FIXED_OBJECTIVE_TARGET_FALLBACK",
                "test",
            )
        )
        .strip()
        .lower()
        or "test"
    )
    env_updates = {
        key: value
        for key, value in _config._extract_payload_env(payload).items()
        if os.environ.get(key) is None
    }
    env_updates.update(
        {
            "DECAT_RUNTIME_MODE": "fixed_json",
            "DECAT_RUNTIME_OBJECTIVE_TARGET": objective_target,
            "DECAT_RUNTIME_SKIP_ARTIFACTS": str(
                _config._env_value(
                    "DECAT_RUNTIME_FIXED_SKIP_ARTIFACTS",
                    "DECAT_RUNTIME_FIXED_SKIP_ARTIFACTS_FALLBACK",
                    "0",
                )
            ),
            "DECAT_RUNTIME_DEFER_BEST_EXPORT": "0",
            "DECAT_RUNTIME_USE_WARM_START": "0",
            "DECAT_RUNTIME_PROGRESS_LOG": "0",
            "DECAT_RUNTIME_ENABLE_FUSION": str(
                _config._env_value(
                    "DECAT_RUNTIME_FIXED_ENABLE_FUSION",
                    "DECAT_RUNTIME_FIXED_ENABLE_FUSION_FALLBACK",
                    "1",
                )
            ),
            "DECAT_RUNTIME_DETERMINISTIC": "1",
            "DECAT_RUNTIME_SEED": str(
                _config._env_value(
                    "DECAT_RUNTIME_FIXED_SEED",
                    "DECAT_RUNTIME_FIXED_SEED_FALLBACK",
                    "42",
                )
            ),
            "DECAT_RUNTIME_MAX_EPOCHS": str(
                _config._env_value(
                    "DECAT_RUNTIME_FIXED_MAX_EPOCHS",
                    "DECAT_RUNTIME_FIXED_MAX_EPOCHS_FALLBACK",
                    "36",
                )
            ),
            "DECAT_RUNTIME_EARLY_STOP": str(
                _config._env_value(
                    "DECAT_RUNTIME_FIXED_EARLY_STOP",
                    "DECAT_RUNTIME_FIXED_EARLY_STOP_FALLBACK",
                    "9",
                )
            ),
            "DECAT_RUNTIME_MIN_DELTA": str(
                _config._env_value(
                    "DECAT_RUNTIME_FIXED_MIN_DELTA",
                    "DECAT_RUNTIME_FIXED_MIN_DELTA_FALLBACK",
                    "1e-4",
                )
            ),
            "DECAT_RUNTIME_CKPT_ENSEMBLE_WEIGHTED": str(
                _config._env_value(
                    "DECAT_RUNTIME_FIXED_WEIGHTED_CKPT",
                    "DECAT_RUNTIME_FIXED_WEIGHTED_CKPT_FALLBACK",
                    "1",
                )
            ),
            "DECAT_RUNTIME_CKPT_ENSEMBLE_TEMP": str(
                _config._env_value(
                    "DECAT_RUNTIME_FIXED_CKPT_TEMP",
                    "DECAT_RUNTIME_FIXED_CKPT_TEMP_FALLBACK",
                    "0.005",
                )
            ),
        }
    )
    if os.path.isfile(fixed_split_path):
        env_updates["DECAT_RUNTIME_FIXED_SPLIT_JSON"] = fixed_split_path
        env_updates["DECAT_RUNTIME_FIXED_SPLIT_JSON_FALLBACK"] = fixed_split_path
    backup = _config._temporary_env_set(env_updates)
    try:
        _runtime.context.iteration = 0
        _runtime.context.split_cache = {}
        _runtime.context.fp_rank_cache = {}
        _runtime.context.run_output_dir = None
        _runtime._reset_best_result_state(objective_target=objective_target)
        _plotting_predictions._apply_plot_defaults_if_needed()
        run_output_dir = _runtime._init_run_output_dir()
        print("=" * 108)
        print("DECAT fixed-params run start")
        print("Output dir:", run_output_dir)
        print("Config json:", config_path)
        print("Fixed vector:", vec)
        print("=" * 108)
        result = objective(list(vec))
        if result is not None:
            print(
                f"[DECAT] Fixed-params run objective ({objective_target} R2) = {-float(result):.6f}"
            )
        save_path = _training_artifacts._save_best_result()
        if save_path:
            print("[DECAT] Saved best params:", save_path)
        summary_path = os.path.join(run_output_dir, "fixed_params_run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "config_source": config_path,
                    "fixed_split_json": (
                        fixed_split_path if os.path.isfile(fixed_split_path) else None
                    ),
                    "vector": list(vec),
                    "json_payload": payload,
                    "best_result": dict(_runtime.context.best_result),
                    "env": {k: os.environ.get(k) for k in env_updates.keys()},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("[DECAT] Saved fixed-params summary:", summary_path)
        return run_output_dir
    finally:
        _config._temporary_env_restore(backup)


def main() -> None:
    import decat.experiments.ablation as _experiments_ablation
    import decat.experiments.retrain as _experiments_retrain
    import decat.experiments.search as _experiments_search

    mode = (
        str(
            _config._env_value(
                "DECAT_RUNTIME_MODE",
                "DECAT_RUNTIME_MODE_FALLBACK",
                _config._env_value("DECAT_RUNTIME_MODE_FALLBACK_2", None, "bo"),
            )
        )
        .strip()
        .lower()
    )
    _config._setdefault_env(
        "DECAT_RUNTIME_OBJECTIVE_TARGET",
        os.environ.get("DECAT_RUNTIME_OBJECTIVE_TARGET_FALLBACK", "val"),
    )
    run_ablation_only = bool(
        _config._env_bool_pair(
            "DECAT_RUNTIME_RUN_ABLATION",
            "DECAT_RUNTIME_RUN_ABLATION_FALLBACK",
            _config._env_bool_pair(
                "DECAT_RUNTIME_RUN_ABLATION_FALLBACK_2", None, False
            ),
        )
        or mode in {"ablation", "abl", "stat"}
    )
    run_ablation_after_bo = _config._env_bool_pair(
        "DECAT_RUNTIME_RUN_ABLATION_AFTER_BO",
        "DECAT_RUNTIME_RUN_ABLATION_AFTER_BO_FALLBACK",
        _config._env_bool_pair(
            "DECAT_RUNTIME_RUN_ABLATION_AFTER_BO_FALLBACK_2", None, False
        ),
    )
    run_final_retrain = _config._env_bool_pair(
        "DECAT_RUNTIME_RUN_FINAL_RETRAIN",
        "DECAT_RUNTIME_RUN_FINAL_RETRAIN_FALLBACK",
        False,
    ) or mode in {"retrain", "final_retrain"}
    if run_ablation_only:
        _experiments_ablation.run_ablation_significance()
        return
    if mode in {"formal", "formal_dual_innov", "formal_dual_innov_best"}:
        run_formal_dual_innov_best()
        return
    if mode in {"fixed", "fixed_json", "fixed_params", "single"}:
        run_fixed_params_once()
        return
    if mode in {"retrain", "final_retrain"}:
        _experiments_retrain.run_final_retrain_on_trainval()
        return
    _experiments_search.run_bayesian_optimization()
    if run_final_retrain:
        _experiments_retrain.run_final_retrain_on_trainval()
    if run_ablation_after_bo:
        _experiments_ablation.run_ablation_significance()
