"""Optional controlled ablations and paired comparisons."""

from __future__ import annotations
import csv
import json
import os
from datetime import datetime
from typing import Dict, Tuple
import numpy as np
import decat.config as _config
import decat.evaluation as _evaluation
import decat.plotting.predictions as _plotting_predictions
import decat.runtime as _runtime
import decat.training.artifacts as _training_artifacts
import decat.training.workflow as _training_workflow


def _load_ablation_base_params() -> Tuple[Dict[str, object], str]:
    path = str(
        _config._env_value("DECAT_RUNTIME_ABLATION_PARAMS_JSON", None, "")
    ).strip()
    if not path:
        path = _training_artifacts._find_latest_best_params_json() or ""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            "Cannot find best-params json. Set DECAT_RUNTIME_ABLATION_PARAMS_JSON to an existing file."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = payload.get("params", payload)
    out = {
        "fp_bits": int(params.get("fp_bits", 4096)),
        "topk_features": int(params.get("topk_features", params.get("k", 320))),
        "d_model": int(params.get("d_model", 256)),
        "batch_size": int(params.get("batch_size", 32)),
        "lr": float(params.get("lr", params.get("learning_rate", 0.0001))),
        "weight_decay": float(params.get("weight_decay", 0.01)),
        "dropout": float(params.get("dropout", 0.2)),
        "max_fp_tokens": int(params.get("max_fp_tokens", 96)),
        "n_layers": int(params.get("n_layers", 1)),
        "n_heads": int(params.get("n_heads", 4)),
        "model_mode": _config._normalize_model_mode(
            params.get("model_mode", payload.get("model_mode", "dual"))
        ),
        "fp_select_method": str(
            params.get("fp_select", params.get("fp_select_method", "rf"))
        ),
        "enable_chem_attn_bias": bool(params.get("enable_chem_attn_bias", True)),
        "norm_first": bool(params.get("norm_first", False)),
        "topk_checkpoint_ensemble": int(params.get("topk_checkpoint_ensemble", 3)),
        "use_checkpoint_ensemble": bool(params.get("use_checkpoint_ensemble", True)),
        "lambda_inv": float(params.get("lambda_inv", 0.18)),
        "lambda_dro": float(params.get("lambda_dro", 0.28)),
        "lambda_physics": float(params.get("lambda_physics", 0.08)),
        "dro_tau": float(params.get("dro_tau", 0.05)),
        "generalization_penalty": float(params.get("generalization_penalty", 0.03)),
    }
    return (out, path)


def run_ablation_significance() -> None:
    _runtime.context.iteration = 0
    _runtime.context.split_cache = {}
    _runtime.context.fp_rank_cache = {}
    if _runtime.context.run_output_dir is None:
        _plotting_predictions._apply_plot_defaults_if_needed()
        _runtime._init_run_output_dir()
    params, params_path = _load_ablation_base_params()
    objective_target = _runtime._get_objective_target()
    seed_text = str(
        _config._env_value("DECAT_RUNTIME_ABLATION_SEEDS", None, "11,19,23,29,37")
    )
    seeds = _config._parse_seeds(seed_text)
    base_mode = _config._normalize_model_mode(params.get("model_mode", "dual"))
    print("=" * 108)
    print("DECAT ablation + significance start")
    print("Base params:", params_path)
    print("Objective target:", objective_target)
    print("Seeds:", seeds)
    print("=" * 108)
    base_env = {
        "DECAT_RUNTIME_ENABLE_DUAL": "1",
        "DECAT_RUNTIME_GATE_EXTRA_FEATURES": "1",
        "DECAT_RUNTIME_ENABLE_CORRECTION_HEAD": "1",
        "DECAT_RUNTIME_ENABLE_FUSION": "1",
        "DECAT_RUNTIME_ENABLE_CHEM_ATTN_BIAS": "1",
        "DECAT_RUNTIME_SKIP_ARTIFACTS": "1",
        "DECAT_RUNTIME_DETERMINISTIC": "1",
        "DECAT_RUNTIME_OBJECTIVE_TARGET_FALLBACK_4": "val",
    }
    variants = [
        {"name": "baseline", "mode": base_mode, "env": {}},
        {
            "name": "no_adaptive_fusion",
            "mode": base_mode,
            "env": {"DECAT_RUNTIME_ENABLE_FUSION": "0"},
        },
        {
            "name": "no_physicochemical_prior",
            "mode": base_mode,
            "env": {},
            "params": {"lambda_physics": 0.0},
        },
    ]
    if base_mode in {"dual", "attn"}:
        variants.append(
            {
                "name": "no_chem_attn_bias",
                "mode": base_mode,
                "env": {"DECAT_RUNTIME_ENABLE_CHEM_ATTN_BIAS": "0"},
            }
        )
    if base_mode == "dual":
        variants.extend(
            [
                {
                    "name": "no_dual_expert",
                    "mode": "dual",
                    "env": {"DECAT_RUNTIME_ENABLE_DUAL": "0"},
                },
                {
                    "name": "no_gate_extra_features",
                    "mode": "dual",
                    "env": {"DECAT_RUNTIME_GATE_EXTRA_FEATURES": "0"},
                },
                {
                    "name": "no_correction_head",
                    "mode": "dual",
                    "env": {"DECAT_RUNTIME_ENABLE_CORRECTION_HEAD": "0"},
                },
            ]
        )
    trial_rows = []
    for variant in variants:
        name = str(variant["name"])
        mode = _config._normalize_model_mode(variant.get("mode", base_mode))
        for seed in seeds:
            env_updates = dict(base_env)
            env_updates.update(
                {k: str(v) for k, v in dict(variant.get("env", {})).items()}
            )
            env_updates["DECAT_RUNTIME_SEED"] = str(seed)
            backup = _config._temporary_env_set(env_updates)
            try:
                _runtime.context.iteration = 0
                _runtime._reset_best_result_state(objective_target=objective_target)
                variant_params = dict(params)
                variant_params.update(dict(variant.get("params", {})))
                x = _config._params_to_vector(variant_params, model_mode_override=mode)
                _ = _training_workflow.objective(x)
                trial_rows.append(
                    {
                        "variant": name,
                        "seed": int(seed),
                        "model_mode": mode,
                        "objective_target": objective_target,
                        "objective_r2": _evaluation._safe_float(
                            _runtime.context.best_result.get("objective_r2")
                        ),
                        "val_r2": _evaluation._safe_float(
                            _runtime.context.best_result.get("val_r2")
                        ),
                        "val_rmse": _evaluation._safe_float(
                            _runtime.context.best_result.get("val_rmse")
                        ),
                        "test_r2": _evaluation._safe_float(
                            _runtime.context.best_result.get("test_r2")
                        ),
                        "test_rmse": _evaluation._safe_float(
                            _runtime.context.best_result.get("test_rmse")
                        ),
                    }
                )
                print(
                    f"[ablation] {name:<24s} seed={seed:<4d} val_r2={_evaluation._safe_float(_runtime.context.best_result.get('val_r2')):7.4f} test_r2={_evaluation._safe_float(_runtime.context.best_result.get('test_r2')):7.4f} test_rmse={_evaluation._safe_float(_runtime.context.best_result.get('test_rmse')):8.4f}"
                )
            finally:
                _config._temporary_env_restore(backup)
    if not trial_rows:
        print("[ablation] no trial rows generated.")
        return
    names = sorted({str(r["variant"]) for r in trial_rows})
    summary_rows = []
    baseline = [r for r in trial_rows if str(r["variant"]) == "baseline"]
    base_by_seed = {int(r["seed"]): r for r in baseline}
    for name in names:
        rows = [r for r in trial_rows if str(r["variant"]) == name]
        val_r2_arr = np.asarray([float(r["val_r2"]) for r in rows], dtype=float)
        test_r2_arr = np.asarray([float(r["test_r2"]) for r in rows], dtype=float)
        test_rmse_arr = np.asarray([float(r["test_rmse"]) for r in rows], dtype=float)
        row_out = {
            "variant": name,
            "n_runs": int(len(rows)),
            "val_r2_mean": float(np.nanmean(val_r2_arr)),
            "val_r2_std": float(np.nanstd(val_r2_arr)),
            "test_r2_mean": float(np.nanmean(test_r2_arr)),
            "test_r2_std": float(np.nanstd(test_r2_arr)),
            "test_rmse_mean": float(np.nanmean(test_rmse_arr)),
            "test_rmse_std": float(np.nanstd(test_rmse_arr)),
            "delta_test_r2_vs_baseline": float("nan"),
            "delta_test_rmse_vs_baseline": float("nan"),
            "p_ttest_test_r2_vs_baseline": float("nan"),
            "p_wilcoxon_test_r2_vs_baseline": float("nan"),
            "p_ttest_test_rmse_vs_baseline": float("nan"),
            "p_wilcoxon_test_rmse_vs_baseline": float("nan"),
        }
        if name != "baseline":
            common = [r for r in rows if int(r["seed"]) in base_by_seed]
            if common:
                base_test_r2 = np.asarray(
                    [float(base_by_seed[int(r["seed"])]["test_r2"]) for r in common],
                    dtype=float,
                )
                base_test_rmse = np.asarray(
                    [float(base_by_seed[int(r["seed"])]["test_rmse"]) for r in common],
                    dtype=float,
                )
                alt_test_r2 = np.asarray(
                    [float(r["test_r2"]) for r in common], dtype=float
                )
                alt_test_rmse = np.asarray(
                    [float(r["test_rmse"]) for r in common], dtype=float
                )
                row_out["delta_test_r2_vs_baseline"] = float(
                    np.nanmean(alt_test_r2 - base_test_r2)
                )
                row_out["delta_test_rmse_vs_baseline"] = float(
                    np.nanmean(alt_test_rmse - base_test_rmse)
                )
                r2_stats = _evaluation._paired_significance(base_test_r2, alt_test_r2)
                rmse_stats = _evaluation._paired_significance(
                    base_test_rmse, alt_test_rmse
                )
                row_out["p_ttest_test_r2_vs_baseline"] = float(r2_stats["p_ttest"])
                row_out["p_wilcoxon_test_r2_vs_baseline"] = float(
                    r2_stats["p_wilcoxon"]
                )
                row_out["p_ttest_test_rmse_vs_baseline"] = float(rmse_stats["p_ttest"])
                row_out["p_wilcoxon_test_rmse_vs_baseline"] = float(
                    rmse_stats["p_wilcoxon"]
                )
        summary_rows.append(row_out)
    out_dir = (
        _runtime.context.run_output_dir
        if _runtime.context.run_output_dir
        else _runtime.context.out_dir
    )
    os.makedirs(out_dir, exist_ok=True)
    trial_csv = os.path.join(out_dir, "ablation_trials_decat.csv")
    summary_csv = os.path.join(out_dir, "ablation_summary_decat.csv")
    summary_json = os.path.join(out_dir, "ablation_summary_decat.json")
    with open(trial_csv, "w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "variant",
            "seed",
            "model_mode",
            "objective_target",
            "objective_r2",
            "val_r2",
            "val_rmse",
            "test_r2",
            "test_rmse",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in trial_rows:
            writer.writerow(r)
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "variant",
            "n_runs",
            "val_r2_mean",
            "val_r2_std",
            "test_r2_mean",
            "test_r2_std",
            "test_rmse_mean",
            "test_rmse_std",
            "delta_test_r2_vs_baseline",
            "delta_test_rmse_vs_baseline",
            "p_ttest_test_r2_vs_baseline",
            "p_wilcoxon_test_r2_vs_baseline",
            "p_ttest_test_rmse_vs_baseline",
            "p_wilcoxon_test_rmse_vs_baseline",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "objective_target": objective_target,
                "params_path": params_path,
                "seeds": seeds,
                "base_params": params,
                "trials": trial_rows,
                "summary": summary_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("=" * 108)
    print("Ablation summary:")
    for r in summary_rows:
        print(
            f"{r['variant']:<24s} test_r2={r['test_r2_mean']:.4f}±{r['test_r2_std']:.4f} test_rmse={r['test_rmse_mean']:.4f}±{r['test_rmse_std']:.4f}"
        )
    print("Saved:", trial_csv)
    print("Saved:", summary_csv)
    print("Saved:", summary_json)
