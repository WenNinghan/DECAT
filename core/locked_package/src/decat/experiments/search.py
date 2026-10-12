"""Optional hyperparameter search using the shared training workflow."""

from __future__ import annotations
import glob
import json
import os
from skopt import gp_minimize
import decat.config as _config
import decat.evaluation as _evaluation
import decat.plotting.predictions as _plotting_predictions
import decat.runtime as _runtime
import decat.training.artifacts as _training_artifacts
import decat.training.workflow as _training_workflow


def _collect_bo_warm_start_points() -> list:
    if not _config._env_bool_pair("DECAT_RUNTIME_USE_WARM_START", None, True):
        return []
    points = []
    default_point = [
        3147,
        914,
        212,
        32,
        0.003,
        0.0015,
        0.1,
        115,
        2,
        4,
        "dual",
        "rf",
        True,
        False,
        3,
        True,
        0.18,
        0.28,
        0.08,
        0.05,
        0.03,
    ]
    points.append(default_point)
    warm_topn = int(
        _config._env_value(
            "DECAT_RUNTIME_WARM_START_TOPN",
            "DECAT_RUNTIME_WARM_START_TOPN_FALLBACK",
            _config._env_value("DECAT_RUNTIME_WARM_START_TOPN_FALLBACK_2", None, "5"),
        )
    )
    warm_topn = max(0, min(12, warm_topn))
    if warm_topn > 0:
        patterns = [
            os.path.join(_runtime.context.out_dir, "run_*", "best_params.json"),
            os.path.join(_runtime.context.out_dir, "best_params_*.json"),
            os.path.join(_runtime.context.out_dir, "best_params_*.json"),
            os.path.join(_runtime.context.out_dir, "best_params_*.json"),
            os.path.join(_runtime.context.out_dir, "best_params_*.json"),
            os.path.join(_runtime.context.out_dir, "best_params_*.json"),
            os.path.join(_runtime.context.out_dir, "best_params_*.json"),
        ]
        warm_files = []
        for pattern in patterns:
            warm_files.extend(glob.glob(pattern))
        warm_files = [f for f in warm_files if os.path.isfile(f)]
        warm_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        for warm_file in warm_files[:warm_topn]:
            try:
                with open(warm_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                params = payload.get("params", payload)
                vec = _config._params_to_bo_vector(params)
                if vec is not None:
                    points.append(vec)
            except Exception:
                pass
    warm_json = str(
        _config._env_value("DECAT_RUNTIME_WARM_START_JSON", None, "")
    ).strip()
    if not warm_json and _config._env_bool_pair(
        "DECAT_RUNTIME_USE_LATEST_BEST_AS_WARM_START", None, True
    ):
        try:
            warm_json = _training_artifacts._find_latest_best_params_json() or ""
        except Exception:
            warm_json = ""
    if warm_json and os.path.isfile(warm_json):
        try:
            with open(warm_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            params = payload.get("params", payload)
            vec = _config._params_to_bo_vector(params)
            if vec is not None:
                points.append(vec)
        except Exception:
            pass
    manual = str(
        _config._env_value("DECAT_RUNTIME_WARM_START_VECTOR", None, "")
    ).strip()
    if manual:
        try:
            toks = [t.strip() for t in manual.split(",")]
            if len(toks) in {21, 22}:
                if len(toks) == 22:
                    toks = toks[:14] + toks[15:]
                vec = [
                    int(float(toks[0])),
                    int(float(toks[1])),
                    int(float(toks[2])),
                    int(float(toks[3])),
                    float(toks[4]),
                    float(toks[5]),
                    float(toks[6]),
                    int(float(toks[7])),
                    int(float(toks[8])),
                    int(float(toks[9])),
                    _config._normalize_model_mode_value(toks[10]),
                    (
                        str(toks[11]).strip().lower()
                        if str(toks[11]).strip().lower() in {"rf", "f_regression"}
                        else "rf"
                    ),
                    str(toks[12]).strip().lower() in {"1", "true", "yes", "y", "on"},
                    str(toks[13]).strip().lower() in {"1", "true", "yes", "y", "on"},
                    int(float(toks[14])),
                    str(toks[15]).strip().lower() in {"1", "true", "yes", "y", "on"},
                    float(toks[16]),
                    float(toks[17]),
                    float(toks[18]),
                    float(toks[19]),
                    float(toks[20]),
                ]
                points.append(vec)
        except Exception:
            pass

    def _point_in_space(point: list) -> bool:
        if len(point) != len(_config.space):
            return False
        for value, dim in zip(point, _config.space):
            if hasattr(dim, "categories"):
                if value not in set(dim.categories):
                    return False
                continue
            low, high = dim.bounds
            try:
                v = float(value)
            except Exception:
                return False
            if v < float(low) or v > float(high):
                return False
        return True

    uniq = []
    seen = set()
    for p in points:
        if not _point_in_space(list(p)):
            continue
        key = tuple(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(list(p))
    return uniq


def run_bayesian_optimization() -> None:
    _runtime.context.iteration = 0
    _runtime.context.split_cache = {}
    _runtime.context.fp_rank_cache = {}
    _runtime.context.run_output_dir = None
    _runtime._reset_best_result_state(_runtime._get_objective_target())
    _plotting_predictions._apply_plot_defaults_if_needed()
    run_output_dir = _runtime._init_run_output_dir()
    n_calls = int(
        _config._env_value(
            "DECAT_RUNTIME_N_CALLS",
            "DECAT_RUNTIME_N_CALLS_FALLBACK",
            _config._env_value(
                "DECAT_RUNTIME_N_CALLS_FALLBACK_2",
                "DECAT_RUNTIME_N_CALLS_FALLBACK_3",
                "100",
            ),
        )
    )
    n_initial_points = int(
        _config._env_value(
            "DECAT_RUNTIME_N_INITIAL_POINTS",
            "DECAT_RUNTIME_N_INITIAL_POINTS_FALLBACK",
            _config._env_value(
                "DECAT_RUNTIME_N_INITIAL_POINTS_FALLBACK_2",
                "DECAT_RUNTIME_N_INITIAL_POINTS_FALLBACK_3",
                "12",
            ),
        )
    )
    bo_seed = int(
        _config._env_value(
            "DECAT_RUNTIME_BO_SEED",
            "DECAT_RUNTIME_BO_SEED_FALLBACK",
            _config._env_value(
                "DECAT_RUNTIME_BO_SEED_FALLBACK_2",
                "DECAT_RUNTIME_BO_SEED_FALLBACK_3",
                "42",
            ),
        )
    )
    objective_target = _runtime._get_objective_target()
    print("=" * 108)
    print("DECAT (Causal-Invariant) Bayesian optimization start")
    print("Data:", _runtime._get_data_csv_path())
    print("Output dir:", run_output_dir)
    print(
        "n_calls:", n_calls, "| n_initial_points:", n_initial_points, "| seed:", bo_seed
    )
    print("Device hint:", os.environ.get("DECAT_RUNTIME_DEVICE", "auto"))
    print(f"Objective target: maximize {objective_target} R2")
    print("=" * 108)
    print(
        " iter | R2_train | RMSE_train | R2_val | RMSE_val | R2_eval | RMSE_eval | model | fp_bits | topk | tok | dm | h | ly | bs | dr | ep |   lr   |   wd"
    )
    warm_points = _collect_bo_warm_start_points()
    if warm_points:
        max_x0 = max(0, int(n_calls) - 1)
        if max_x0 <= 0:
            warm_points = []
        elif len(warm_points) > max_x0:
            warm_points = warm_points[:max_x0]
        n_initial_points = max(
            0, min(int(n_initial_points), int(n_calls) - len(warm_points))
        )
        if warm_points:
            print(
                f"[DECAT] BO warm-start points: {len(warm_points)} | adjusted n_initial_points: {n_initial_points}"
            )
    gp_kwargs = dict(
        func=_training_workflow.objective,
        dimensions=_config.space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=bo_seed,
        verbose=False,
    )
    if warm_points:
        gp_kwargs["x0"] = warm_points
    result = gp_minimize(**gp_kwargs)
    best_obj = -float(result.fun)
    print("=" * 108)
    print("Optimization finished")
    print(f"Best objective ({objective_target}) R2 = {best_obj:.4f}")
    defer_best_export = _config._env_bool_pair(
        "DECAT_RUNTIME_DEFER_BEST_EXPORT", None, True
    )
    skip_artifacts = _config._env_bool_pair(
        "DECAT_RUNTIME_SKIP_ARTIFACTS", "DECAT_RUNTIME_SKIP_ARTIFACTS_FALLBACK", False
    )
    if defer_best_export and (not skip_artifacts):
        print("=" * 108)
        print("Finalize best artifacts after BO with candidate/seed sweep...")
        finalize_seed_text = str(
            _config._env_value(
                "DECAT_RUNTIME_FINAL_SEEDS",
                "DECAT_RUNTIME_FINAL_SEEDS_FALLBACK",
                _config._env_value(
                    "DECAT_RUNTIME_FINAL_SEEDS_FALLBACK_2",
                    "DECAT_RUNTIME_FINALIZE_SEED",
                    str(bo_seed),
                ),
            )
        ).strip()
        finalize_seeds = _config._parse_seeds(finalize_seed_text)
        candidate_vectors = _config._dedupe_candidate_vectors(
            [list(result.x)] + list(warm_points or [])
        )
        env_updates = {
            "DECAT_RUNTIME_DEFER_BEST_EXPORT": "0",
            "DECAT_RUNTIME_SKIP_ARTIFACTS": "0",
            "DECAT_RUNTIME_DETERMINISTIC": "1",
        }
        backup = _config._temporary_env_set(env_updates)
        try:
            _runtime.context.iteration = 0
            _runtime.context.split_cache = {}
            _runtime.context.fp_rank_cache = {}
            _runtime._reset_best_result_state(objective_target=objective_target)
            total_runs = int(len(candidate_vectors) * len(finalize_seeds))
            run_i = 0
            for vec in candidate_vectors:
                for seed in finalize_seeds:
                    run_i += 1
                    os.environ["DECAT_RUNTIME_SEED"] = str(seed)
                    print(
                        f"[DECAT] Final sweep {run_i}/{total_runs} | seed={seed} | vec={vec}"
                    )
                    _ = _training_workflow.objective(vec)
        finally:
            _config._temporary_env_restore(backup)
        print("=" * 108)
        print("Final export finished")
        print(
            f"Final exported test R2 = {_evaluation._safe_float(_runtime.context.best_result.get('test_r2')):.4f}"
        )
        print(
            f"Final exported test RMSE = {_evaluation._safe_float(_runtime.context.best_result.get('test_rmse')):.4f}"
        )
    else:
        save_path = _training_artifacts._save_best_result()
        if save_path:
            print("Saved best params:", save_path)
