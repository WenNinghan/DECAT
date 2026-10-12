"""Explicit stages for preparing, fitting, evaluating and recording one trial."""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Subset

import decat.config as _config
import decat.data.dataset as _data_dataset
import decat.data.features as _data_features
import decat.data.splitting as _data_splitting
import decat.evaluation as _evaluation
import decat.models.attention as _models_attention
import decat.models.decat as _models_decat
import decat.models.experts as _models_experts
import decat.plotting.attention as _plotting_attention
import decat.runtime as _runtime
import decat.training.artifacts as _training_artifacts
import decat.training.engine as _training_engine
import decat.training.ensemble as _training_ensemble
import decat.training.losses as _training_losses
import decat.training.sampling as _training_sampling


@dataclass
class TrialParameters:
    """One objective's model, optimizer and regularization choices."""

    fp_bits: int
    topk_features: int
    d_model: int
    batch_size: int
    lr: float
    weight_decay: float
    dropout: float
    max_fp_tokens: int
    n_layers: int
    n_heads: int
    model_mode: str
    fp_select_method: str
    enable_chem_attn_bias: bool
    norm_first: bool
    topk_ckpt_ensemble: int
    use_ckpt_ensemble: bool
    lambda_inv: float
    lambda_dro: float
    lambda_physics: float
    dro_tau: float
    generalization_penalty: float


@dataclass
class TrialLoaders:
    train: DataLoader
    validation: DataLoader
    train_evaluation: DataLoader
    test: DataLoader


@dataclass
class PreparedTrial:
    """Prepared data and training objects, with the owning run context."""

    parameters: TrialParameters
    context: _runtime.RunContext
    iter_id: int
    objective_target: str
    progress_log: bool
    cfg: _config.FingerprintConfig
    dataset: _data_dataset.FingerprintReactionDataset
    train_idx: Any
    val_idx: Any
    test_idx: np.ndarray
    top_idx: np.ndarray | None
    fp_dim_for_model: int
    trainer: _training_engine.FingerprintTrainer
    loaders: TrialLoaders
    train_sampler: Any
    sampler_info: dict
    physics_prior_t: torch.Tensor
    physics_meta_t: dict


@dataclass
class FittedCheckpoints:
    """Validation-selected states and epoch diagnostics in selection order."""

    best_state: dict | None
    best_epoch: int
    epoch_history: list
    top_states: list
    top_state_records: list


@dataclass
class SplitEvaluation:
    truth: np.ndarray
    prediction: np.ndarray
    indices: np.ndarray
    r2: float
    rmse: float


@dataclass
class TrialEvaluation:
    """Checkpoint/fusion predictions and metadata needed by reporting."""

    train: SplitEvaluation
    validation: SplitEvaluation
    test: SplitEvaluation
    model_label: str
    fusion_note: str
    fusion_result: dict | None
    use_fusion: bool
    use_weighted_ckpt_ensemble: bool
    ckpt_ensemble_temp: float
    component_names_plot: list
    component_test_preds_plot: list
    global_blend_test_pred_plot: Any


def _resolve_hook(context: _runtime.RunContext, name: str, default):
    """Resolve a protocol hook at its original call site from this run."""
    configured = getattr(context.hooks, name)
    return default if configured is None else configured


def prepare_trial(
    parameters: TrialParameters, context: _runtime.RunContext
) -> PreparedTrial | None:
    """Set seeds, resolve protocol data/model hooks and construct loaders."""
    context.iteration += 1
    iter_id = context.iteration
    objective_target = _runtime._get_objective_target()
    progress_log = _config._env_bool_pair("DECAT_RUNTIME_PROGRESS_LOG", None, False)
    if progress_log:
        print(
            f"[DECAT] Iter {iter_id} start | mode={parameters.model_mode} | fp_bits={int(parameters.fp_bits)} | topk={int(parameters.topk_features)}",
            flush=True,
        )
    seed_env = os.environ.get("DECAT_RUNTIME_SEED")
    if seed_env is not None and str(seed_env).strip() != "":
        try:
            seed_val = int(str(seed_env).strip())
            random.seed(seed_val)
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_val)
            if _config._env_bool("DECAT_RUNTIME_DETERMINISTIC", False):
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                except Exception:
                    pass
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
        except Exception:
            pass
    cfg = _config.FingerprintConfig()
    cfg.learning_rate = float(parameters.lr)
    cfg.weight_decay = float(parameters.weight_decay)
    cfg.d_model = int(parameters.d_model)
    cfg.dropout = float(parameters.dropout)
    cfg.batch_size = int(parameters.batch_size)
    cfg.max_fp_tokens = int(parameters.max_fp_tokens)
    cfg.n_layers = int(parameters.n_layers)
    cfg.n_heads = int(parameters.n_heads)
    cfg.norm_first = bool(parameters.norm_first)
    cfg.max_epochs = int(
        _config._env_value(
            "DECAT_RUNTIME_MAX_EPOCHS", "DECAT_RUNTIME_MAX_EPOCHS_FALLBACK", "120"
        )
    )
    cfg.early_stopping_patience = int(
        _config._env_value(
            "DECAT_RUNTIME_EARLY_STOP", "DECAT_RUNTIME_EARLY_STOP_FALLBACK", "36"
        )
    )
    cfg.min_delta = float(
        _config._env_value(
            "DECAT_RUNTIME_MIN_DELTA", "DECAT_RUNTIME_MIN_DELTA_FALLBACK", "1e-4"
        )
    )
    parameters.topk_ckpt_ensemble = max(1, min(5, parameters.topk_ckpt_ensemble))
    parameters.use_ckpt_ensemble = bool(parameters.use_ckpt_ensemble)
    parameters.fp_select_method = str(parameters.fp_select_method).strip().lower()
    if parameters.fp_select_method not in {"rf", "f_regression"}:
        parameters.fp_select_method = "rf"
    os.environ["DECAT_RUNTIME_FP_SELECT"] = str(parameters.fp_select_method)
    os.environ["DECAT_RUNTIME_ENABLE_CHEM_ATTN_BIAS"] = (
        "1" if bool(parameters.enable_chem_attn_bias) else "0"
    )
    os.environ["DECAT_RUNTIME_USE_CKPT_ENSEMBLE"] = (
        "1" if bool(parameters.use_ckpt_ensemble) else "0"
    )
    os.environ["DECAT_RUNTIME_TOPK_CHECKPOINT_ENSEMBLE"] = str(
        int(parameters.topk_ckpt_ensemble)
    )
    os.environ["DECAT_RUNTIME_LAMBDA_INV"] = str(float(parameters.lambda_inv))
    os.environ["DECAT_RUNTIME_LAMBDA_DRO"] = str(float(parameters.lambda_dro))
    os.environ["DECAT_RUNTIME_LAMBDA_PHYSICS"] = str(float(parameters.lambda_physics))
    os.environ["DECAT_RUNTIME_DRO_TAU"] = str(float(parameters.dro_tau))
    os.environ["DECAT_RUNTIME_GENERALIZATION_PENALTY"] = str(
        float(parameters.generalization_penalty)
    )
    if cfg.d_model % cfg.n_heads != 0:
        cfg.d_model = max(cfg.n_heads, cfg.d_model // cfg.n_heads * cfg.n_heads)
    dataset = _data_dataset.FingerprintReactionDataset(
        _runtime._get_data_csv_path(),
        max_fp_bits=int(parameters.fp_bits),
        fingerprint_scale=cfg.fingerprint_scale,
    )
    cfg.base_numeric_dim = int(getattr(dataset, "base_num_dim", dataset.num_dim))
    if len(dataset) < 80:
        print(f"{iter_id:4d} | dataset too small ({len(dataset)})")
        return None
    split = _resolve_hook(context, "build_split", _data_splitting._build_split_indices)(
        dataset
    )
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = np.asarray(split["test_idx"], dtype=int)
    dataset.fit_scalers(train_idx)
    use_topk = _config._env_bool_pair(
        "DECAT_RUNTIME_USE_TOPK", "DECAT_RUNTIME_USE_TOPK_FALLBACK", True
    )
    if use_topk:
        rank = _data_features._get_fp_ranking(
            dataset, train_idx, int(parameters.fp_bits), parameters.fp_select_method
        )
        k = int(min(max(1, parameters.topk_features), dataset.fingerprint_dim))
        top_idx = rank[:k]
        train_subset = _resolve_hook(
            context, "subset_factory", _data_dataset.SelectedFeatureSubset
        )(dataset, train_idx, top_idx)
        val_subset = _resolve_hook(
            context, "subset_factory", _data_dataset.SelectedFeatureSubset
        )(dataset, val_idx, top_idx)
        test_subset = _resolve_hook(
            context, "subset_factory", _data_dataset.SelectedFeatureSubset
        )(dataset, test_idx, top_idx)
        fp_dim_for_model = int(k)
    else:
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        test_subset = Subset(dataset, test_idx)
        fp_dim_for_model = int(dataset.fingerprint_dim)
        top_idx = None
    if parameters.model_mode in {"dual", "attn"}:
        cfg.max_fp_tokens = min(int(cfg.max_fp_tokens), int(fp_dim_for_model))
    if parameters.model_mode == "mlp":
        model = _models_attention.FingerprintTransformer(
            fp_dim_for_model, dataset.num_dim, cfg
        )
    elif parameters.model_mode == "attn":
        model = _models_experts._build_attention_expert(
            fp_dim_for_model, dataset.num_dim, cfg
        )
    else:
        model = _resolve_hook(
            context, "model_factory", _models_decat.DualExpertRegressor
        )(fp_dim_for_model, dataset.num_dim, cfg)
    device_env = str(os.environ.get("DECAT_RUNTIME_DEVICE", "")).strip()
    trainer = _training_engine.FingerprintTrainer(model, cfg, device=device_env or None)
    use_cuda = str(trainer.device).lower().startswith("cuda")
    train_sampler, sampler_info = _training_sampling._build_category_balanced_sampler(
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
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )
    train_eval_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_subset,
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
    return PreparedTrial(
        parameters=parameters,
        context=context,
        iter_id=iter_id,
        objective_target=objective_target,
        progress_log=progress_log,
        cfg=cfg,
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        top_idx=top_idx,
        fp_dim_for_model=fp_dim_for_model,
        trainer=trainer,
        loaders=TrialLoaders(train_loader, val_loader, train_eval_loader, test_loader),
        train_sampler=train_sampler,
        sampler_info=sampler_info,
        physics_prior_t=physics_prior_t,
        physics_meta_t=physics_meta_t,
    )


def fit_checkpoints(prepared: PreparedTrial) -> FittedCheckpoints:
    """Train epochs, retain improving validation checkpoints and restore the best."""
    if prepared.progress_log:
        print(
            f"[DECAT] Iter {prepared.iter_id} training (max_epochs={prepared.cfg.max_epochs}) ...",
            flush=True,
        )
    best_state = None
    best_epoch = 0
    best_monitor_r2 = float("-inf")
    best_monitor_rmse = float("inf")
    patience = 0
    epoch_history = []
    top_states = []
    top_state_records = []
    for epoch in range(1, prepared.cfg.max_epochs + 1):
        train_stats = _training_engine._train_epoch_causal_invariant(
            trainer=prepared.trainer,
            dataloader=prepared.loaders.train,
            physics_sign_prior=prepared.physics_prior_t,
            physics_meta=prepared.physics_meta_t,
            epoch=int(epoch),
            max_epochs=int(prepared.cfg.max_epochs),
        )
        snapshot_state = None
        _, val_r2_epoch, val_rmse_epoch = prepared.trainer.evaluate(
            prepared.loaders.validation
        )
        monitor_r2 = val_r2_epoch
        monitor_rmse = val_rmse_epoch
        if (
            prepared.cfg.scheduler_type == "plateau"
            and prepared.trainer.scheduler is not None
        ):
            prepared.trainer.scheduler.step(
                monitor_rmse if np.isfinite(monitor_rmse) else 1.0
            )
        improved = False
        if np.isfinite(monitor_r2):
            improved = monitor_r2 > best_monitor_r2 + prepared.cfg.min_delta
        elif np.isfinite(monitor_rmse):
            improved = monitor_rmse < best_monitor_rmse - prepared.cfg.min_delta
        epoch_history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_stats.get("loss", float("nan"))),
                "train_pred_loss": float(train_stats.get("pred_loss", float("nan"))),
                "train_inv_loss": float(train_stats.get("inv_loss", float("nan"))),
                "train_dro_loss": float(train_stats.get("dro_loss", float("nan"))),
                "train_phys_loss": float(train_stats.get("phys_loss", float("nan"))),
                "val_r2": float(val_r2_epoch),
                "val_rmse": float(val_rmse_epoch),
                "eval_state": "raw",
            }
        )
        if improved:
            snapshot_state = copy.deepcopy(prepared.trainer.model.state_dict())
        if improved and snapshot_state is not None:
            best_monitor_r2 = float(monitor_r2)
            best_monitor_rmse = float(monitor_rmse)
            best_state = snapshot_state
            best_epoch = int(epoch)
            top_states.append(
                (float(monitor_r2), int(epoch), copy.deepcopy(best_state))
            )
            top_states.sort(key=lambda x: float(x[0]), reverse=True)
            top_states = top_states[: prepared.parameters.topk_ckpt_ensemble]
            top_state_records = [
                {"epoch": int(ep), "val_r2": float(r2)} for r2, ep, _ in top_states
            ]
            checkpoint_export = os.environ.get(
                "DECAT_EXPORT_TOP_CHECKPOINTS", ""
            ).strip()
            if checkpoint_export:
                os.makedirs(checkpoint_export, exist_ok=True)
                scores = [float(r2) for r2, _, _ in top_states]
                if _config._env_bool_pair(
                    "DECAT_RUNTIME_CKPT_ENSEMBLE_WEIGHTED", None, True
                ):
                    weights = _evaluation._softmax_weights(
                        scores,
                        temperature=float(
                            _config._env_value(
                                "DECAT_RUNTIME_CKPT_ENSEMBLE_TEMP", None, "0.02"
                            )
                        ),
                    )
                else:
                    weights = np.ones(len(scores), dtype=float) / float(len(scores))
                manifest = {
                    "states": [],
                    "weights": [float(value) for value in weights],
                }
                for rank, (score, saved_epoch, state) in enumerate(top_states, start=1):
                    filename = f"top{rank}_epoch{int(saved_epoch)}.pth"
                    torch.save(
                        {
                            "state_dict": state,
                            "epoch": int(saved_epoch),
                            "val_r2": float(score),
                        },
                        os.path.join(checkpoint_export, filename),
                    )
                    manifest["states"].append(
                        {
                            "file": filename,
                            "epoch": int(saved_epoch),
                            "val_r2": float(score),
                        }
                    )
                with open(
                    os.path.join(checkpoint_export, "top_checkpoint_ensemble.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(manifest, handle, ensure_ascii=False, indent=2)
            patience = 0
        else:
            patience += 1
        if patience >= prepared.cfg.early_stopping_patience:
            break
    if prepared.progress_log:
        print(
            f"[DECAT] Iter {prepared.iter_id} training done | best_epoch={best_epoch}",
            flush=True,
        )
    if best_state is not None:
        prepared.trainer.model.load_state_dict(best_state)
    prepared.context.last_trained_model = prepared.trainer.model
    return FittedCheckpoints(
        best_state=best_state,
        best_epoch=best_epoch,
        epoch_history=epoch_history,
        top_states=top_states,
        top_state_records=top_state_records,
    )


def evaluate_fusion(
    prepared: PreparedTrial, checkpoints: FittedCheckpoints
) -> TrialEvaluation:
    """Predict train/validation/test in order, then apply validation-selected fusion."""
    use_weighted_ckpt_ensemble = _config._env_bool_pair(
        "DECAT_RUNTIME_CKPT_ENSEMBLE_WEIGHTED", None, True
    )
    ckpt_ensemble_temp = float(
        _config._env_value("DECAT_RUNTIME_CKPT_ENSEMBLE_TEMP", None, "0.02")
    )
    if prepared.parameters.use_ckpt_ensemble and len(checkpoints.top_states) > 1:
        state_pack = list(
            checkpoints.top_states[: prepared.parameters.topk_ckpt_ensemble]
        )
        pred_train_list = []
        pred_val_list = []
        pred_test_list = []
        state_scores = []
        y_train = pred_train = idx_train = None
        y_val = pred_val = idx_val = None
        y_test = pred_test = idx_test = None
        for r2_ckpt, _, st in state_pack:
            prepared.trainer.model.load_state_dict(st)
            (
                y_train_i,
                pred_train_i,
                idx_train_i,
                y_val_i,
                pred_val_i,
                idx_val_i,
                y_test_i,
                pred_test_i,
                idx_test_i,
            ) = _evaluation._predict_split_triplet(
                prepared.trainer.model,
                prepared.trainer.device,
                prepared.loaders.train_evaluation,
                prepared.loaders.validation,
                prepared.loaders.test,
            )
            if y_train is None:
                y_train, idx_train = (y_train_i, idx_train_i)
                y_val, idx_val = (y_val_i, idx_val_i)
                y_test, idx_test = (y_test_i, idx_test_i)
            pred_train_list.append(np.asarray(pred_train_i, dtype=np.float32))
            pred_val_list.append(np.asarray(pred_val_i, dtype=np.float32))
            pred_test_list.append(np.asarray(pred_test_i, dtype=np.float32))
            state_scores.append(float(r2_ckpt))
        if use_weighted_ckpt_ensemble:
            w_ckpt = _evaluation._softmax_weights(
                state_scores, temperature=ckpt_ensemble_temp
            )
        else:
            w_ckpt = np.ones(len(pred_train_list), dtype=float) / float(
                len(pred_train_list)
            )
        w_col = w_ckpt.reshape(-1, 1)
        pred_train = np.sum(np.vstack(pred_train_list) * w_col, axis=0).astype(
            np.float32
        )
        pred_val = np.sum(np.vstack(pred_val_list) * w_col, axis=0).astype(np.float32)
        pred_test = np.sum(np.vstack(pred_test_list) * w_col, axis=0).astype(np.float32)
        if checkpoints.best_state is not None:
            prepared.trainer.model.load_state_dict(checkpoints.best_state)
    else:
        (
            y_train,
            pred_train,
            idx_train,
            y_val,
            pred_val,
            idx_val,
            y_test,
            pred_test,
            idx_test,
        ) = _evaluation._predict_split_triplet(
            prepared.trainer.model,
            prepared.trainer.device,
            prepared.loaders.train_evaluation,
            prepared.loaders.validation,
            prepared.loaders.test,
        )
    train_r2, train_rmse = _resolve_hook(
        prepared.context, "metrics", _evaluation._metrics
    )(y_train, pred_train)
    val_r2, val_rmse = _resolve_hook(prepared.context, "metrics", _evaluation._metrics)(
        y_val, pred_val
    )
    test_r2, test_rmse = _resolve_hook(
        prepared.context, "metrics", _evaluation._metrics
    )(y_test, pred_test)
    model_label = str(prepared.parameters.model_mode)
    fusion_note = "none"
    fusion_result = None
    use_fusion = _config._env_bool_pair(
        "DECAT_RUNTIME_ENABLE_FUSION", "DECAT_RUNTIME_ENABLE_FUSION_FALLBACK", True
    )
    if use_fusion and y_train.size > 20 and (y_val.size > 10):
        if prepared.progress_log:
            print(f"[DECAT] Iter {prepared.iter_id} adaptive fusion ...", flush=True)
        fp_scaled = prepared.dataset.fingerprint_scaled
        if prepared.top_idx is not None:
            fp_scaled = fp_scaled[:, prepared.top_idx]
        num_all = np.concatenate(
            [prepared.dataset.ph_scaled, prepared.dataset.category], axis=1
        ).astype(np.float32)
        x_all = np.concatenate([fp_scaled.astype(np.float32), num_all], axis=1).astype(
            np.float32
        )
        if idx_train.size > 0 and np.all(idx_train >= 0):
            x_train = x_all[idx_train]
        else:
            x_train = x_all[prepared.train_idx]
        if idx_val.size > 0 and np.all(idx_val >= 0):
            x_val = x_all[idx_val]
        else:
            x_val = x_all[prepared.val_idx]
        if idx_test.size > 0 and np.all(idx_test >= 0):
            x_test = x_all[idx_test]
        else:
            x_test = x_all[prepared.test_idx]
        fusion_result = _resolve_hook(
            prepared.context, "fusion", _training_ensemble._adaptive_fusion
        )(
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            pred_train_nn=pred_train,
            pred_val_nn=pred_val,
            pred_test_nn=pred_test,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            random_state=42 + prepared.iter_id,
            optimize_target="val",
            groups_train=np.asarray(
                [
                    Chem.MolToSmiles(
                        Chem.MolFromSmiles(str(prepared.dataset.smiles[index])),
                        canonical=True,
                        isomericSmiles=True,
                    )
                    for index in (
                        idx_train
                        if idx_train.size > 0 and np.all(idx_train >= 0)
                        else prepared.train_idx
                    )
                ],
                dtype=object,
            ),
        )
        if fusion_result is not None:
            base_target_r2 = test_r2 if prepared.objective_target == "test" else val_r2
            fused_target_r2 = float(
                fusion_result.get("r2_test", float("nan"))
                if prepared.objective_target == "test"
                else fusion_result.get("r2_val", float("nan"))
            )
            if np.isfinite(fused_target_r2) and (
                not np.isfinite(base_target_r2)
                or fused_target_r2 > base_target_r2 + 0.0001
            ):
                pred_train = np.asarray(fusion_result["train_pred"], dtype=np.float32)
                pred_test = np.asarray(fusion_result["test_pred"], dtype=np.float32)
                train_r2, train_rmse = _resolve_hook(
                    prepared.context, "metrics", _evaluation._metrics
                )(y_train, pred_train)
                val_r2 = float(fusion_result["r2_val"])
                val_rmse = float(fusion_result["rmse_val"])
                test_r2, test_rmse = _resolve_hook(
                    prepared.context, "metrics", _evaluation._metrics
                )(y_test, pred_test)
                model_label = f"{prepared.parameters.model_mode}+fusion"
                if "weights" in fusion_result:
                    fusion_note = f"{fusion_result['mode']}:{fusion_result['weights']}"
                else:
                    fusion_note = str(fusion_result.get("mode", "fusion"))
    component_names_plot = []
    component_test_preds_plot = []
    global_blend_test_pred_plot = None
    if isinstance(fusion_result, dict):
        component_names_plot = list(fusion_result.get("component_names", []) or [])
        component_test_preds_plot = list(
            fusion_result.get("component_test_preds", []) or []
        )
        global_blend_test_pred_plot = fusion_result.get("global_blend_test_pred", None)
    return TrialEvaluation(
        train=SplitEvaluation(y_train, pred_train, idx_train, train_r2, train_rmse),
        validation=SplitEvaluation(y_val, pred_val, idx_val, val_r2, val_rmse),
        test=SplitEvaluation(y_test, pred_test, idx_test, test_r2, test_rmse),
        model_label=model_label,
        fusion_note=fusion_note,
        fusion_result=fusion_result,
        use_fusion=use_fusion,
        use_weighted_ckpt_ensemble=use_weighted_ckpt_ensemble,
        ckpt_ensemble_temp=ckpt_ensemble_temp,
        component_names_plot=component_names_plot,
        component_test_preds_plot=component_test_preds_plot,
        global_blend_test_pred_plot=global_blend_test_pred_plot,
    )


def record_trial(
    prepared: PreparedTrial, checkpoints: FittedCheckpoints, evaluated: TrialEvaluation
) -> float:
    """Update the run objective and perform the requested best-result exports."""
    line = f"{prepared.iter_id:4d} | {evaluated.train.r2:8.4f} | {evaluated.train.rmse:10.4f} | {evaluated.validation.r2:7.4f} | {evaluated.validation.rmse:9.4f} | {evaluated.test.r2:7.4f} | {evaluated.test.rmse:9.4f} | {evaluated.model_label:<12s} | fp={int(prepared.parameters.fp_bits):<4d} | k={int(prepared.fp_dim_for_model):<4d} | tok={int(prepared.cfg.max_fp_tokens):<3d} | dm={int(prepared.cfg.d_model):<4d} | h={int(prepared.cfg.n_heads):<2d} | ly={int(prepared.cfg.n_layers):<2d} | bs={int(prepared.cfg.batch_size):<3d} | dr={prepared.cfg.dropout:4.2f} | ep={checkpoints.best_epoch:<3d} | lr={prepared.cfg.learning_rate:.5f} | wd={prepared.cfg.weight_decay:.2e}"
    current_target_r2 = (
        evaluated.test.r2
        if prepared.objective_target == "test"
        else evaluated.validation.r2
    )
    objective_score = float(current_target_r2)
    if (
        prepared.objective_target == "val"
        and np.isfinite(evaluated.train.r2)
        and np.isfinite(evaluated.validation.r2)
        and (prepared.parameters.generalization_penalty > 0.0)
    ):
        objective_score -= float(prepared.parameters.generalization_penalty) * max(
            0.0, float(evaluated.train.r2 - evaluated.validation.r2)
        )
    is_best = np.isfinite(objective_score) and objective_score > float(
        prepared.context.best_result["objective_r2"]
    )
    if is_best:
        prepared.context.best_result["iter"] = int(prepared.iter_id)
        prepared.context.best_result["objective_target"] = str(
            prepared.objective_target
        )
        prepared.context.best_result["objective_r2"] = float(objective_score)
        prepared.context.best_result["val_r2"] = float(evaluated.validation.r2)
        prepared.context.best_result["val_rmse"] = float(evaluated.validation.rmse)
        prepared.context.best_result["test_r2"] = float(evaluated.test.r2)
        prepared.context.best_result["test_rmse"] = float(evaluated.test.rmse)
        prepared.context.best_result["train_r2"] = float(evaluated.train.r2)
        prepared.context.best_result["train_rmse"] = float(evaluated.train.rmse)
        prepared.context.best_result["model_mode"] = str(evaluated.model_label)
        prepared.context.best_result["fusion_mode"] = str(evaluated.fusion_note)
        prepared.context.best_result["run_output_dir"] = prepared.context.run_output_dir
        prepared.context.best_result["params"] = {
            "fp_bits": int(prepared.parameters.fp_bits),
            "topk_features": int(prepared.fp_dim_for_model),
            "d_model": int(prepared.cfg.d_model),
            "batch_size": int(prepared.cfg.batch_size),
            "lr": float(prepared.cfg.learning_rate),
            "weight_decay": float(prepared.cfg.weight_decay),
            "dropout": float(prepared.cfg.dropout),
            "max_fp_tokens": int(prepared.cfg.max_fp_tokens),
            "n_layers": int(prepared.cfg.n_layers),
            "n_heads": int(prepared.cfg.n_heads),
            "model_mode": str(prepared.parameters.model_mode),
            "best_epoch": int(checkpoints.best_epoch),
            "fp_select": str(
                _config._env_value(
                    "DECAT_RUNTIME_FP_SELECT", "DECAT_RUNTIME_FP_SELECT_FALLBACK", "rf"
                )
            ),
            "enable_fusion": bool(evaluated.use_fusion),
            "balanced_sampler": bool(prepared.train_sampler is not None),
            "enable_chem_attn_bias": bool(
                _config._env_bool_pair(
                    "DECAT_RUNTIME_ENABLE_CHEM_ATTN_BIAS",
                    "DECAT_RUNTIME_ENABLE_CHEM_ATTN_BIAS_FALLBACK",
                    True,
                )
            ),
            "norm_first": bool(prepared.cfg.norm_first),
            "topk_checkpoint_ensemble": int(prepared.parameters.topk_ckpt_ensemble),
            "use_checkpoint_ensemble": bool(prepared.parameters.use_ckpt_ensemble),
            "ckpt_ensemble_weighted": bool(evaluated.use_weighted_ckpt_ensemble),
            "ckpt_ensemble_temp": float(evaluated.ckpt_ensemble_temp),
            "causal_invariant": True,
            "lambda_inv": float(
                _config._env_value(
                    "DECAT_RUNTIME_LAMBDA_INV",
                    "DECAT_RUNTIME_LAMBDA_INV_FALLBACK",
                    "0.18",
                )
            ),
            "lambda_dro": float(
                _config._env_value(
                    "DECAT_RUNTIME_LAMBDA_DRO",
                    "DECAT_RUNTIME_LAMBDA_DRO_FALLBACK",
                    "0.28",
                )
            ),
            "lambda_physics": float(
                _config._env_value(
                    "DECAT_RUNTIME_LAMBDA_PHYSICS",
                    "DECAT_RUNTIME_LAMBDA_PHYSICS_FALLBACK",
                    "0.08",
                )
            ),
            "dro_tau": float(
                _config._env_value(
                    "DECAT_RUNTIME_DRO_TAU", "DECAT_RUNTIME_DRO_TAU_FALLBACK", "0.05"
                )
            ),
            "generalization_penalty": float(prepared.parameters.generalization_penalty),
        }
        print("\x1b[92m" + line + "\x1b[0m")
        defer_best_export = _config._env_bool_pair(
            "DECAT_RUNTIME_DEFER_BEST_EXPORT", None, True
        )
        skip_artifacts = _config._env_bool_pair(
            "DECAT_RUNTIME_SKIP_ARTIFACTS",
            "DECAT_RUNTIME_SKIP_ARTIFACTS_FALLBACK",
            False,
        )
        if not defer_best_export and (not skip_artifacts):
            save_path = _training_artifacts._save_best_result()
            if save_path:
                print("[DECAT] Saved best params:", save_path)
                try:
                    model_save_path = os.path.join(
                        (
                            prepared.context.run_output_dir
                            if prepared.context.run_output_dir
                            else prepared.context.out_dir
                        ),
                        "decat_best.pth",
                    )
                    torch.save(prepared.trainer.model.state_dict(), model_save_path)
                    print(f"[DECAT] Saved best model checkpoint: {model_save_path}")
                except Exception as e:
                    print(f"[DECAT] Failed to save model checkpoint: {e}")
            out_dir = _training_artifacts._export_best_plots(
                model=prepared.trainer.model,
                dataset=prepared.dataset,
                explain_loader=prepared.loaders.train_evaluation,
                test_loader=prepared.loaders.test,
                test_subset_indices=np.asarray(prepared.test_idx, dtype=int),
                y_train_true=np.asarray(evaluated.train.truth, dtype=np.float32),
                y_train_pred=np.asarray(evaluated.train.prediction, dtype=np.float32),
                idx_train_pred=np.asarray(evaluated.train.indices, dtype=int),
                y_test_true=np.asarray(evaluated.test.truth, dtype=np.float32),
                y_test_pred=np.asarray(evaluated.test.prediction, dtype=np.float32),
                idx_test_pred=np.asarray(evaluated.test.indices, dtype=int),
                component_names=evaluated.component_names_plot,
                component_test_preds=evaluated.component_test_preds_plot,
                global_blend_test_pred=evaluated.global_blend_test_pred_plot,
                iter_id=int(prepared.iter_id),
                model_mode=str(prepared.parameters.model_mode),
                fp_bits=int(prepared.parameters.fp_bits),
                fp_dim_for_model=int(prepared.fp_dim_for_model),
            )
            if out_dir:
                print("[DECAT] Saved best plots:", out_dir)
                _plotting_attention.export_training_diagnostics(
                    output_dir=out_dir,
                    model=prepared.trainer.model,
                    test_loader=prepared.loaders.test,
                    epoch_history=checkpoints.epoch_history,
                    top_state_records=checkpoints.top_state_records,
                    sampler_info=prepared.sampler_info,
                    fusion_result=evaluated.fusion_result,
                )
    else:
        print(line)
    if not np.isfinite(objective_score):
        return 1000000.0
    return -float(objective_score)
