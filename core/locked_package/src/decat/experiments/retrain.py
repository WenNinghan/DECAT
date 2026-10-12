"""Optional final fitting on combined training and validation records."""

from __future__ import annotations
import json
import os
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from torch.utils.data import DataLoader
import decat.config as _config
import decat.data.dataset as _data_dataset
import decat.data.splitting as _data_splitting
import decat.models.attention as _models_attention
import decat.models.decat as _models_decat
import decat.models.experts as _models_experts
import decat.runtime as _runtime
import decat.training.engine as _training_engine
import decat.training.losses as _training_losses


def run_final_retrain_on_trainval() -> None:
    """Retrain best model on train+val, evaluate on test only (Plan B)."""
    if not _runtime.context.best_result.get("params"):
        print("[DECAT] No best params found, skip final retrain")
        return
    enable_final_retrain = _config._env_bool_pair(
        "DECAT_RUNTIME_ENABLE_FINAL_RETRAIN",
        "DECAT_RUNTIME_ENABLE_FINAL_RETRAIN_FALLBACK",
        _config._env_bool_pair(
            "DECAT_RUNTIME_ENABLE_FINAL_RETRAIN_FALLBACK_2", None, True
        ),
    )
    if not enable_final_retrain:
        print("[DECAT] Final retrain disabled by env var")
        return
    print("=" * 108)
    print("[DECAT] Final Retrain: train on train+val, evaluate on test only")
    print("=" * 108)
    best_params = dict(_runtime.context.best_result["params"])
    best_epoch = int(best_params.get("best_epoch", 50))
    epoch_multiplier = float(
        _config._env_value("DECAT_RUNTIME_FINAL_EPOCH_MULTIPLIER", None, "1.15")
    )
    final_epochs = max(10, int(best_epoch * epoch_multiplier))
    print(f"Best epoch from BO: {best_epoch}")
    print(f"Final training epochs: {final_epochs} (multiplier={epoch_multiplier})")
    print(f"Best params: {best_params}")
    fp_bits = int(best_params.get("fp_bits", 2048))
    topk_features = int(best_params.get("topk_features", 256))
    d_model = int(best_params.get("d_model", 256))
    batch_size = int(best_params.get("batch_size", 32))
    lr = float(best_params.get("lr", 0.0001))
    weight_decay = float(best_params.get("weight_decay", 0.01))
    dropout = float(best_params.get("dropout", 0.1))
    max_fp_tokens = int(best_params.get("max_fp_tokens", 64))
    n_layers = int(best_params.get("n_layers", 2))
    n_heads = int(best_params.get("n_heads", 4))
    model_mode = str(best_params.get("model_mode", "attn"))
    norm_first = bool(best_params.get("norm_first", False))
    dataset = _data_dataset.FingerprintReactionDataset(
        _runtime._get_data_csv_path(), max_fp_bits=fp_bits, fingerprint_scale=False
    )
    split = _runtime.resolve_hook("build_split", _data_splitting._build_split_indices)(
        dataset
    )
    trainval_idx = np.asarray(split["train_val_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)
    dataset.fit_scalers(trainval_idx)
    print(f"Train+Val size: {len(trainval_idx)}, Test size: {len(test_idx)}")
    fp_select_method = str(
        _config._env_value(
            "DECAT_RUNTIME_FP_SELECT", "DECAT_RUNTIME_FP_SELECT_FALLBACK", "rf"
        )
    )
    if topk_features < dataset.fingerprint_dim:
        cache_key = (int(fp_bits), str(fp_select_method))
        if cache_key in _runtime.context.fp_rank_cache:
            ranked_indices = _runtime.context.fp_rank_cache[cache_key]
        else:
            X_train = dataset.fingerprint[trainval_idx]
            y_train = dataset.logk_raw[trainval_idx].reshape(-1)
            if fp_select_method == "rf":
                seed = int(
                    _config._env_value(
                        "DECAT_RUNTIME_FINAL_SEED",
                        "DECAT_RUNTIME_FIXED_SEED_FALLBACK",
                        _config._env_value("DECAT_RUNTIME_SEED", None, "42"),
                    )
                )
                selector = RandomForestRegressor(
                    n_estimators=50, max_depth=8, random_state=seed, n_jobs=-1
                )
                selector.fit(X_train, y_train)
                importances = selector.feature_importances_
            else:
                f_scores, _ = f_regression(X_train, y_train)
                importances = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
            ranked_indices = np.argsort(importances)[::-1]
            _runtime.context.fp_rank_cache[cache_key] = ranked_indices
        selected_indices = ranked_indices[:topk_features]
    else:
        selected_indices = np.arange(dataset.fingerprint_dim, dtype=int)
    fp_dim_final = int(len(selected_indices))
    trainval_ds = _runtime.resolve_hook(
        "subset_factory", _data_dataset.SelectedFeatureSubset
    )(dataset, trainval_idx, selected_indices)
    test_ds = _runtime.resolve_hook(
        "subset_factory", _data_dataset.SelectedFeatureSubset
    )(dataset, test_idx, selected_indices)
    trainval_loader = DataLoader(
        trainval_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    cfg = _config.FingerprintConfig()
    cfg.d_model = d_model
    cfg.batch_size = batch_size
    cfg.learning_rate = lr
    cfg.weight_decay = weight_decay
    cfg.dropout = dropout
    cfg.max_fp_tokens = max_fp_tokens
    cfg.n_layers = n_layers
    cfg.n_heads = n_heads
    cfg.norm_first = bool(norm_first)
    cfg.max_epochs = final_epochs
    cfg.early_stopping_patience = final_epochs + 1
    cfg.scheduler_type = "none"
    cfg.base_numeric_dim = int(getattr(dataset, "base_num_dim", dataset.num_dim))
    if model_mode == "attn":
        model = _models_experts._build_attention_expert(
            fp_dim_final, dataset.num_dim, cfg
        )
    elif model_mode == "dual":
        model = _runtime.resolve_hook(
            "model_factory", _models_decat.DualExpertRegressor
        )(fp_dim_final, dataset.num_dim, cfg)
    else:
        model = _models_attention.FingerprintTransformer(
            fp_dim_final, dataset.num_dim, cfg
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = _training_engine.FingerprintTrainer(model, cfg, device=device)
    physics_meta_np = _training_losses._build_category_physics_meta(
        dataset, trainval_idx
    )
    physics_prior_t = torch.as_tensor(
        physics_meta_np["sign"], dtype=torch.float32, device=trainer.device
    )
    physics_meta_t = _training_losses._physics_meta_to_tensors(
        physics_meta_np, trainer.device
    )
    print(f"Training for {final_epochs} epochs on train+val...")
    for epoch in range(1, final_epochs + 1):
        train_stats = _training_engine._train_epoch_causal_invariant(
            trainer=trainer,
            dataloader=trainval_loader,
            physics_sign_prior=physics_prior_t,
            physics_meta=physics_meta_t,
            epoch=int(epoch),
            max_epochs=int(final_epochs),
        )
        if epoch % 10 == 0 or epoch == final_epochs:
            print(
                f"Epoch {epoch}/{final_epochs} | loss={train_stats.get('loss', 0):.4f}"
            )
    _, test_r2, test_rmse = trainer.evaluate(test_loader)
    trainer.model.eval()
    pred_test_list = []
    y_test_list = []
    with torch.no_grad():
        for batch in test_loader:
            fp = batch["fingerprint"].to(trainer.device)
            num = batch["numeric"].to(trainer.device)
            y_raw = batch["logk_raw"].numpy()
            pred_scaled = trainer.model(fp, num).detach().cpu().numpy().reshape(-1, 1)
            pred_raw = dataset.logk_scaler.inverse_transform(pred_scaled).reshape(-1)
            pred_test_list.append(pred_raw)
            y_test_list.append(y_raw.reshape(-1))
    y_test = np.concatenate(y_test_list)
    pred_test = np.concatenate(pred_test_list)
    print("=" * 108)
    print(f"[DECAT] Final Retrain Results:")
    print(f"Test R² = {test_r2:.4f}")
    print(f"Test RMSE = {test_rmse:.4f}")
    print(
        f"Previous BO best test R² = {_runtime.context.best_result.get('test_r2', float('nan')):.4f}"
    )
    print(
        f"Improvement = {test_r2 - _runtime.context.best_result.get('test_r2', 0):.4f}"
    )
    print("=" * 108)
    _runtime.context.best_result["final_retrain_test_r2"] = float(test_r2)
    _runtime.context.best_result["final_retrain_test_rmse"] = float(test_rmse)
    _runtime.context.best_result["final_retrain_epochs"] = int(final_epochs)
    if _runtime.context.run_output_dir:
        final_result_path = os.path.join(
            _runtime.context.run_output_dir, "final_retrain_result.json"
        )
        with open(final_result_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "test_r2": float(test_r2),
                    "test_rmse": float(test_rmse),
                    "epochs": int(final_epochs),
                    "trainval_size": int(len(trainval_idx)),
                    "test_size": int(len(test_idx)),
                    "params": best_params,
                },
                f,
                indent=2,
            )
        print(f"Saved final retrain result: {final_result_path}")
