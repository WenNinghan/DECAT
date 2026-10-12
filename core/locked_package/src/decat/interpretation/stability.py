"""Cross-seed and cross-fold attribution stability."""

from __future__ import annotations
import copy
import os
from collections import Counter
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import decat.config as _config
import decat.data.dataset as _data_dataset
import decat.data.splitting as _data_splitting
import decat.interpretation.attribution as _interpretation_attribution
import decat.interpretation.motifs as _interpretation_motifs
import decat.models.decat as _models_decat
import decat.runtime as _runtime
import decat.training.engine as _training_engine
import decat.training.losses as _training_losses
import decat.training.sampling as _training_sampling


def _train_stability_model_for_fold(
    *,
    dataset: _data_dataset.FingerprintReactionDataset,
    train_idx: np.ndarray,
    hold_idx: np.ndarray,
    fp_bits: int,
    topk_features: int,
    model_mode: str,
    cfg_base: _config.FingerprintConfig,
    fp_select_method: str,
    seed: int,
) -> Tuple[Optional[nn.Module], Optional[DataLoader]]:
    train_idx = np.asarray(train_idx, dtype=int).reshape(-1)
    hold_idx = np.asarray(hold_idx, dtype=int).reshape(-1)
    if train_idx.size < 16 or hold_idx.size < 8:
        return (None, None)
    _runtime._set_global_random_seed(int(seed))
    dataset.fit_scalers(train_idx)
    train_subset, hold_subset, fp_dim_for_model = (
        _data_splitting._make_selected_subsets_for_indices(
            dataset=dataset,
            train_idx=train_idx,
            eval_idx=hold_idx,
            fp_bits=int(fp_bits),
            topk_features=int(topk_features),
            fp_select_method=str(fp_select_method),
        )
    )
    cfg = copy.deepcopy(cfg_base)
    cfg.max_epochs = int(
        _config._env_value(
            "DECAT_RUNTIME_CMA_STABILITY_MAX_EPOCHS", None, min(int(cfg.max_epochs), 80)
        )
    )
    cfg.early_stopping_patience = int(
        _config._env_value(
            "DECAT_RUNTIME_CMA_STABILITY_EARLY_STOP",
            None,
            min(int(cfg.early_stopping_patience), 20),
        )
    )
    cfg.batch_size = int(
        _config._env_value(
            "DECAT_RUNTIME_CMA_STABILITY_BATCH_SIZE", None, cfg.batch_size
        )
    )
    cfg.min_delta = float(
        _config._env_value("DECAT_RUNTIME_CMA_STABILITY_MIN_DELTA", None, cfg.min_delta)
    )
    if model_mode in {"dual", "attn"}:
        cfg.max_fp_tokens = min(int(cfg.max_fp_tokens), int(fp_dim_for_model))
    cfg.base_numeric_dim = int(getattr(dataset, "base_num_dim", dataset.num_dim))
    model_fold = _models_decat._build_model_for_mode(
        model_mode, fp_dim_for_model, dataset.num_dim, cfg
    )
    device_env = str(os.environ.get("DECAT_RUNTIME_DEVICE", "")).strip()
    trainer = _training_engine.FingerprintTrainer(
        model_fold, cfg, device=device_env or None
    )
    use_cuda = str(trainer.device).lower().startswith("cuda")
    train_sampler, _ = _training_sampling._build_category_balanced_sampler(
        dataset, train_subset, train_idx
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=use_cuda,
    )
    hold_loader = DataLoader(
        hold_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )
    physics_meta_np = _training_losses._build_category_physics_meta(dataset, train_idx)
    physics_prior_t = torch.as_tensor(
        physics_meta_np["sign"], dtype=torch.float32, device=trainer.device
    )
    physics_meta_t = _training_losses._physics_meta_to_tensors(
        physics_meta_np, trainer.device
    )
    best_state = None
    best_monitor_r2 = float("-inf")
    best_monitor_rmse = float("inf")
    patience = 0
    for _epoch in range(1, int(cfg.max_epochs) + 1):
        _ = _training_engine._train_epoch_causal_invariant(
            trainer=trainer,
            dataloader=train_loader,
            physics_sign_prior=physics_prior_t,
            physics_meta=physics_meta_t,
            epoch=int(_epoch),
            max_epochs=int(cfg.max_epochs),
        )
        _, val_r2_epoch, val_rmse_epoch = trainer.evaluate(hold_loader)
        if cfg.scheduler_type == "plateau" and trainer.scheduler is not None:
            trainer.scheduler.step(
                val_rmse_epoch if np.isfinite(val_rmse_epoch) else 1.0
            )
        improved = False
        if np.isfinite(val_r2_epoch):
            improved = val_r2_epoch > best_monitor_r2 + float(cfg.min_delta)
        elif np.isfinite(val_rmse_epoch):
            improved = val_rmse_epoch < best_monitor_rmse - float(cfg.min_delta)
        if improved:
            best_monitor_r2 = float(val_r2_epoch)
            best_monitor_rmse = float(val_rmse_epoch)
            best_state = copy.deepcopy(trainer.model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= int(cfg.early_stopping_patience):
            break
    if best_state is not None:
        trainer.model.load_state_dict(best_state)
    return (trainer.model, hold_loader)


def _compute_cross_seed_fold_stability(
    *,
    model: nn.Module,
    dataset: _data_dataset.FingerprintReactionDataset,
    train_indices: np.ndarray,
    candidate_topk: int,
    final_topk: int,
    min_support: int,
    batch_size: int,
    ig_steps: int,
    fp_bits: int,
    topk_features: int,
    fp_select_method: str,
) -> Dict[str, float]:
    enabled = _config._env_bool("DECAT_RUNTIME_CMA_STABILITY_TRUE", True)
    if not enabled:
        return {}
    seed_text = str(
        _config._env_value("DECAT_RUNTIME_CMA_STABILITY_SEEDS", None, "11,19,23,29,37")
    ).strip()
    seeds = _config._parse_seeds(seed_text)
    fold_count = int(
        max(2, int(_config._env_value("DECAT_RUNTIME_CMA_STABILITY_FOLDS", None, 5)))
    )
    model_mode = _models_decat._infer_model_mode_from_instance(model)
    cfg_base = _models_decat._build_cma_cfg_from_model(model)
    train_indices = np.asarray(train_indices, dtype=int).reshape(-1)
    train_indices = train_indices[(train_indices >= 0) & (train_indices < len(dataset))]
    if train_indices.size < max(24, fold_count * 4):
        return {}
    hit_counter = Counter()
    run_counter = Counter()
    total_runs = 0
    fp_bits = int(
        fp_bits
        if fp_bits > 0
        else getattr(dataset, "max_fp_bits", dataset.fingerprint_dim)
    )
    topk_features = int(max(1, topk_features))
    fp_select_method = str(fp_select_method).strip().lower() or "rf"
    print(
        f"[DECAT] Starting Stability Analysis: {len(seeds)} seeds x {fold_count} folds..."
    )
    for seed in seeds:
        kf = KFold(n_splits=int(fold_count), shuffle=True, random_state=int(seed))
        for fold_id, (sub_train_pos, sub_hold_pos) in enumerate(
            kf.split(train_indices)
        ):
            sub_train_idx = np.asarray(train_indices[sub_train_pos], dtype=int)
            sub_hold_idx = np.asarray(train_indices[sub_hold_pos], dtype=int)
            try:
                model_fold, hold_loader = _train_stability_model_for_fold(
                    dataset=dataset,
                    train_idx=sub_train_idx,
                    hold_idx=sub_hold_idx,
                    fp_bits=fp_bits,
                    topk_features=topk_features,
                    model_mode=model_mode,
                    cfg_base=cfg_base,
                    fp_select_method=fp_select_method,
                    seed=int(seed * 100 + fold_id),
                )
            except Exception as e:
                print(
                    f"[DECAT] Warning: Stability fold failed (seed={seed}, fold={fold_id}): {e}"
                )
                continue
            if model_fold is None or hold_loader is None:
                continue
            try:
                explain = _interpretation_attribution._collect_explain_arrays(
                    hold_loader, max_samples=None
                )
                fp_np = np.asarray(
                    explain.get("fingerprint", np.zeros((0, 0), dtype=np.float32)),
                    dtype=np.float32,
                )
                num_np = np.asarray(
                    explain.get("numeric", np.zeros((0, 0), dtype=np.float32)),
                    dtype=np.float32,
                )
                feature_labels = list(explain.get("feature_labels", []) or [])
                if (
                    fp_np.ndim != 2
                    or fp_np.shape[0] == 0
                    or fp_np.shape[1] == 0
                    or (not feature_labels)
                ):
                    continue
                local_rows, _, _, local_ranked = (
                    _interpretation_motifs._compute_consensus_rows_core(
                        model=model_fold,
                        fingerprint_np=fp_np,
                        numeric_np=num_np,
                        feature_labels=feature_labels,
                        candidate_topk=int(candidate_topk),
                        final_topk=int(final_topk),
                        min_support=int(min_support),
                        batch_size=int(batch_size),
                        ig_steps=int(ig_steps),
                        stability_mode="none",
                        bootstrap_rounds=0,
                        bootstrap_frac=0.75,
                        stability_override=None,
                        weight_attn=0.2,
                        weight_ig=0.2,
                        weight_delta=0.3,
                        weight_stab=0.0,
                        weight_support=0.1,
                    )
                )
                if not local_rows:
                    continue
                total_runs += 1
                top_names = [str(x) for x in local_ranked[: int(final_topk)]]
                for name in top_names:
                    hit_counter[name] += 1
                for row in local_rows:
                    run_counter[str(row.get("bit_name", ""))] += 1
            except Exception:
                continue
    stability_map = {}
    if total_runs <= 0:
        return stability_map
    for name, hits in hit_counter.items():
        stability_map[str(name)] = float(hits / max(total_runs, 1))
    return stability_map


def export_consensus_motif_artifacts(
    model: nn.Module,
    dataset: _data_dataset.FingerprintReactionDataset,
    explain_loader: DataLoader,
    train_indices: np.ndarray,
    prefix: str,
) -> Dict[str, object]:
    import decat.plotting.structures as _plotting_structures

    out_dir = _runtime.get_run_output_dir(prefix)
    if explain_loader is None:
        return {}
    max_samples = int(
        max(64, int(_config._env_value("DECAT_RUNTIME_CMA_MAX_SAMPLES", None, 512)))
    )
    ig_steps = int(
        max(4, int(_config._env_value("DECAT_RUNTIME_CMA_IG_STEPS", None, 8)))
    )
    bootstrap_rounds = int(
        max(8, int(_config._env_value("DECAT_RUNTIME_CMA_BOOTSTRAP_ROUNDS", None, 32)))
    )
    bootstrap_frac = float(
        _config._env_value("DECAT_RUNTIME_CMA_BOOTSTRAP_FRAC", None, 0.75)
    )
    candidate_topk = int(
        max(12, int(_config._env_value("DECAT_RUNTIME_CMA_CANDIDATE_TOPK", None, 24)))
    )
    final_topk = int(
        max(5, int(_config._env_value("DECAT_RUNTIME_CMA_TOPK", None, 10)))
    )
    min_support = int(
        max(2, int(_config._env_value("DECAT_RUNTIME_CMA_MIN_SUPPORT", None, 5)))
    )
    batch_size = int(_config._env_value("DECAT_RUNTIME_CMA_BATCH_SIZE", None, 32))
    weight_attn = float(_config._env_value("DECAT_RUNTIME_CMA_W_ATTENTION", None, 0.2))
    weight_ig = float(_config._env_value("DECAT_RUNTIME_CMA_W_IG", None, 0.2))
    weight_delta = float(_config._env_value("DECAT_RUNTIME_CMA_W_DELTA", None, 0.3))
    weight_stab = float(_config._env_value("DECAT_RUNTIME_CMA_W_STABILITY", None, 0.2))
    weight_support = float(_config._env_value("DECAT_RUNTIME_CMA_W_SUPPORT", None, 0.1))
    explain = _interpretation_attribution._collect_explain_arrays(
        explain_loader, max_samples=max_samples
    )
    fp_np = np.asarray(
        explain.get("fingerprint", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32
    )
    num_np = np.asarray(
        explain.get("numeric", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32
    )
    feature_labels = list(explain.get("feature_labels", []) or [])
    if (
        fp_np.ndim != 2
        or fp_np.shape[0] == 0
        or fp_np.shape[1] == 0
        or (not feature_labels)
    ):
        return {}
    rows, _, _, _ = _interpretation_motifs._compute_consensus_rows_core(
        model=model,
        fingerprint_np=fp_np,
        numeric_np=num_np,
        feature_labels=feature_labels,
        candidate_topk=int(candidate_topk),
        final_topk=int(final_topk),
        min_support=int(min_support),
        batch_size=int(batch_size),
        ig_steps=int(ig_steps),
        stability_mode="bootstrap",
        bootstrap_rounds=int(bootstrap_rounds),
        bootstrap_frac=float(bootstrap_frac),
        stability_override=None,
        weight_attn=float(weight_attn),
        weight_ig=float(weight_ig),
        weight_delta=float(weight_delta),
        weight_stab=float(weight_stab),
        weight_support=float(weight_support),
    )
    if not rows:
        return {}
    topk_features = int(fp_np.shape[1])
    fp_bits = int(getattr(dataset, "max_fp_bits", dataset.fingerprint_dim))
    fp_select_method = (
        str(
            _config._env_value(
                "DECAT_RUNTIME_FP_SELECT", "DECAT_RUNTIME_FP_SELECT_FALLBACK", "rf"
            )
        )
        .strip()
        .lower()
    )
    true_stability_map = _compute_cross_seed_fold_stability(
        model=model,
        dataset=dataset,
        train_indices=np.asarray(train_indices, dtype=int),
        candidate_topk=int(candidate_topk),
        final_topk=int(final_topk),
        min_support=int(min_support),
        batch_size=int(batch_size),
        ig_steps=int(ig_steps),
        fp_bits=int(fp_bits),
        topk_features=int(topk_features),
        fp_select_method=str(fp_select_method),
    )
    if true_stability_map:
        for row in rows:
            row["stability_score"] = float(
                true_stability_map.get(str(row.get("bit_name", "")), 0.0)
            )
            row["stability_source"] = "seed_fold"
        attn_cons = _interpretation_attribution._minmax_scale_array(
            np.asarray(
                [float(row.get("attention_score", 0.0)) for row in rows],
                dtype=np.float32,
            )
        )
        ig_cons = _interpretation_attribution._minmax_scale_array(
            np.asarray(
                [float(row.get("ig_score", 0.0)) for row in rows], dtype=np.float32
            )
        )
        delta_cons = _interpretation_attribution._minmax_scale_array(
            np.asarray(
                [float(row.get("perturb_score", 0.0)) for row in rows], dtype=np.float32
            )
        )
        stab_cons = _interpretation_attribution._minmax_scale_array(
            np.asarray(
                [float(row.get("stability_score", 0.0)) for row in rows],
                dtype=np.float32,
            )
        )
        supp_cons = _interpretation_attribution._minmax_scale_array(
            np.asarray(
                [float(row.get("support_frac", 0.0)) for row in rows], dtype=np.float32
            )
        )
        for pos, row in enumerate(rows):
            row["consensus_score"] = float(
                weight_attn * float(attn_cons[pos])
                + weight_ig * float(ig_cons[pos])
                + weight_delta * float(delta_cons[pos])
                + weight_stab * float(stab_cons[pos])
                + weight_support * float(supp_cons[pos])
            )
        rows.sort(
            key=lambda row: (
                float(row.get("consensus_score", float("-inf"))),
                float(row.get("perturb_score", 0.0)),
                float(row.get("attention_score", 0.0)),
                float(row.get("ig_score", 0.0)),
                float(row.get("stability_score", 0.0)),
                float(row.get("support_count", 0.0)),
            ),
            reverse=True,
        )
    else:
        for row in rows:
            row["stability_source"] = "bootstrap"
    dataset.fit_scalers(np.asarray(train_indices, dtype=int))
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "consensus_motif_scores.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    metric_map = {str(row["bit_name"]): dict(row) for row in rows}
    ranked_bit_names = [str(row["bit_name"]) for row in rows]
    sub_info = _interpretation_motifs.export_attention_substructure_artifacts(
        dataset=dataset,
        train_indices=train_indices,
        ranked_bit_names=ranked_bit_names,
        prefix=prefix,
        artifact_tag="consensus",
        extra_bit_metrics=metric_map,
    )
    selected_rows = list((sub_info or {}).get("selected_rows", []) or [])
    bit_rows = list((sub_info or {}).get("bit_rows", []) or [])
    functional_rows = list((sub_info or {}).get("functional_rows", []) or [])
    if functional_rows:
        functional_rows = _interpretation_motifs._supplement_functional_group_rows(
            functional_rows,
            output_dir=out_dir,
            top_n=int(
                _config._env_value("DECAT_RUNTIME_TOP_FUNCTIONAL_GROUPS", None, 10)
            ),
        )
        pd.DataFrame(functional_rows).to_csv(
            os.path.join(out_dir, "consensus_motif_family_scores.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        _plotting_structures._plot_top_functional_groups(
            functional_rows,
            out_dir,
            artifact_tag="consensus",
            top_n=int(
                _config._env_value("DECAT_RUNTIME_TOP_FUNCTIONAL_GROUPS", None, 10)
            ),
        )
        _plotting_structures._plot_consensus_top_motifs(
            functional_rows,
            out_dir,
            top_n=int(
                _config._env_value("DECAT_RUNTIME_TOP_FUNCTIONAL_GROUPS", None, 10)
            ),
        )
        _plotting_structures._plot_top_functional_group_structures(
            output_dir=out_dir,
            top_n=int(
                _config._env_value("DECAT_RUNTIME_TOP_FUNCTIONAL_GROUPS", None, 10)
            ),
        )
    elif selected_rows:
        family_counter = Counter()
        family_support = Counter()
        for row in selected_rows:
            fg_name = str(row.get("functional_group", "")).strip() or "Unassigned"
            family_counter[fg_name] += float(row.get("consensus_score", 0.0))
            family_support[fg_name] += int(row.get("support_count", 0))
        family_rows = [
            {
                "functional_group": str(name),
                "consensus_score_sum": float(score),
                "support_count_sum": int(family_support.get(name, 0)),
            }
            for name, score in family_counter.most_common()
        ]
        pd.DataFrame(family_rows).to_csv(
            os.path.join(out_dir, "consensus_motif_family_scores.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        _plotting_structures._plot_consensus_top_motifs(
            selected_rows, out_dir, top_n=final_topk
        )
    else:
        _plotting_structures._plot_consensus_top_motifs(rows, out_dir, top_n=final_topk)
    bit_to_family = {}
    source_rows = bit_rows if bit_rows else selected_rows
    for row in source_rows:
        bit_name = str(row.get("bit_name", "")).strip()
        family = str(row.get("functional_group", "")).strip()
        if bit_name and family:
            bit_to_family[bit_name] = family
    return {
        "ranked_bit_names": ranked_bit_names,
        "selected_rows": selected_rows,
        "functional_rows": functional_rows,
        "bit_to_family": bit_to_family,
    }
