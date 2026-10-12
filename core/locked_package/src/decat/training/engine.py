"""Epoch execution and optimizer mechanics."""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset
import decat.config as _config
import decat.data.dataset as _data_dataset
import decat.runtime as _runtime
import decat.training.losses as _training_losses


class FingerprintTrainer:
    """Own a model, optimizer and its batch prediction/training utilities."""

    def __init__(
        self,
        model: nn.Module,
        config: _config.FingerprintConfig,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        if str(self.config.scheduler_type).lower() == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-06
            )
        else:
            self.scheduler = None
        self.use_amp = bool(str(self.device).lower().startswith("cuda"))
        self.amp_dtype = torch.float16
        self.scaler = (
            torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None
        )

    def _get_base_dataset(self, dataloader: DataLoader):
        ds = dataloader.dataset
        if isinstance(
            ds,
            _runtime.resolve_hook(
                "subset_factory", _data_dataset.SelectedFeatureSubset
            ),
        ):
            return ds.base
        if isinstance(ds, Subset) and hasattr(ds, "dataset"):
            return ds.dataset
        return ds

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        preds_all = []
        y_all = []
        base_ds = self._get_base_dataset(dataloader)
        with torch.no_grad():
            for batch in dataloader:
                fingerprint = batch["fingerprint"].to(self.device)
                numeric = batch["numeric"].to(self.device)
                y = batch["logk"].to(self.device)
                y_raw = batch["logk_raw"].numpy()
                preds_scaled = self.model(fingerprint, numeric)
                loss = self.criterion(preds_scaled, y)
                bs = int(fingerprint.size(0))
                total_loss += float(loss.item()) * bs
                n_samples += bs
                preds_scaled_np = preds_scaled.detach().cpu().numpy().reshape(-1, 1)
                preds_raw = base_ds.logk_scaler.inverse_transform(
                    preds_scaled_np
                ).reshape(-1)
                preds_all.append(preds_raw)
                y_all.append(y_raw.reshape(-1))
        avg_loss = total_loss / max(1, n_samples)
        if preds_all:
            y_true = np.concatenate(y_all)
            y_pred = np.concatenate(preds_all)
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            if y_true.size >= 2:
                r2 = float(r2_score(y_true, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            else:
                r2 = float("nan")
                rmse = float("nan")
        else:
            r2 = float("nan")
            rmse = float("nan")
        return (float(avg_loss), float(r2), float(rmse))


def _train_epoch_causal_invariant(
    trainer: FingerprintTrainer,
    dataloader: DataLoader,
    physics_sign_prior: Optional[torch.Tensor] = None,
    physics_meta: Optional[Dict[str, torch.Tensor]] = None,
    epoch: int = 1,
    max_epochs: int = 1,
) -> Dict[str, float]:
    trainer.model.train()
    device = str(trainer.device)
    non_blocking = device.lower().startswith("cuda")
    use_amp = bool(getattr(trainer, "use_amp", False) and non_blocking)
    amp_dtype = getattr(trainer, "amp_dtype", torch.float16)
    scaler = getattr(trainer, "scaler", None)
    lam_inv = float(
        _config._env_value(
            "DECAT_RUNTIME_LAMBDA_INV", "DECAT_RUNTIME_LAMBDA_INV_FALLBACK", "0.18"
        )
    )
    lam_dro = float(
        _config._env_value(
            "DECAT_RUNTIME_LAMBDA_DRO", "DECAT_RUNTIME_LAMBDA_DRO_FALLBACK", "0.28"
        )
    )
    dro_tau = float(
        _config._env_value(
            "DECAT_RUNTIME_DRO_TAU", "DECAT_RUNTIME_DRO_TAU_FALLBACK", "0.05"
        )
    )
    lam_phys = float(
        _config._env_value(
            "DECAT_RUNTIME_LAMBDA_PHYSICS",
            "DECAT_RUNTIME_LAMBDA_PHYSICS_FALLBACK",
            "0.08",
        )
    )
    physics_delta = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_DELTA",
            "DECAT_RUNTIME_PHYSICS_DELTA_FALLBACK",
            "0.04",
        )
    )
    phys_smooth_w = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_SMOOTH_W",
            "DECAT_RUNTIME_PHYSICS_SMOOTH_W_FALLBACK",
            "0.45",
        )
    )
    phys_mono_w = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_MONO_W",
            "DECAT_RUNTIME_PHYSICS_MONO_W_FALLBACK",
            "0.55",
        )
    )
    physics_mode = (
        str(
            _config._env_value(
                "DECAT_RUNTIME_PHYSICS_MODE",
                "DECAT_RUNTIME_PHYSICS_MODE_FALLBACK",
                "finite_diff",
            )
        )
        .strip()
        .lower()
    )
    physics_conf_min = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_CONF_MIN",
            "DECAT_RUNTIME_PHYSICS_CONF_MIN_FALLBACK",
            "0.25",
        )
    )
    physics_colloc_w = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_COLLOC_W",
            "DECAT_RUNTIME_PHYSICS_COLLOC_W_FALLBACK",
            "0.35",
        )
    )
    physics_use_colloc = _config._env_bool_pair(
        "DECAT_RUNTIME_PHYSICS_ENABLE_COLLOCATION",
        "DECAT_RUNTIME_PHYSICS_ENABLE_COLLOCATION_FALLBACK",
        False,
    )
    physics_fd_conf_gated = _config._env_bool_pair(
        "DECAT_RUNTIME_PHYSICS_FINITEDIFF_CONF_GATED",
        "DECAT_RUNTIME_PHYSICS_FINITEDIFF_CONF_GATED_FALLBACK",
        False,
    )
    physics_fd_range_gated = _config._env_bool_pair(
        "DECAT_RUNTIME_PHYSICS_FINITEDIFF_RANGE_GATED",
        "DECAT_RUNTIME_PHYSICS_FINITEDIFF_RANGE_GATED_FALLBACK",
        False,
    )
    physics_fd_smooth_floor = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_FINITEDIFF_SMOOTH_FLOOR",
            "DECAT_RUNTIME_PHYSICS_FINITEDIFF_SMOOTH_FLOOR_FALLBACK",
            "0.15",
        )
    )
    physics_slope_target_w = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_SLOPE_TARGET_W",
            "DECAT_RUNTIME_PHYSICS_SLOPE_TARGET_W_FALLBACK",
            "0.70",
        )
    )
    physics_slope_clip = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_SLOPE_CLIP",
            "DECAT_RUNTIME_PHYSICS_SLOPE_CLIP_FALLBACK",
            "2.50",
        )
    )
    physics_pinn_curv_w = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_CURVATURE_W",
            "DECAT_RUNTIME_PHYSICS_CURVATURE_W_FALLBACK",
            "0.0",
        )
    )
    physics_pinn_curv_conf_min = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_CURVATURE_CONF_MIN",
            "DECAT_RUNTIME_PHYSICS_CURVATURE_CONF_MIN_FALLBACK",
            "0.18",
        )
    )
    physics_safe_target = (
        str(_config._env_value("DECAT_RUNTIME_PHYSICS_SAFE_TARGET", None, "attn"))
        .strip()
        .lower()
    )
    if physics_safe_target not in {"final", "attn"}:
        physics_safe_target = "attn"
    physics_warmup_frac = float(
        _config._env_value("DECAT_RUNTIME_PHYSICS_WARMUP_FRAC", None, "0.30")
    )
    physics_warmup_frac = float(max(0.0, min(0.95, physics_warmup_frac)))
    physics_start_frac = float(
        _config._env_value("DECAT_RUNTIME_PHYSICS_START_FRAC", None, "0.0")
    )
    physics_start_frac = float(max(0.0, min(0.95, physics_start_frac)))
    physics_max_gate = float(
        _config._env_value("DECAT_RUNTIME_PHYSICS_MAX_GATE", None, "1.0")
    )
    physics_max_gate = float(max(0.0, min(1.0, physics_max_gate)))
    physics_conf_power = float(
        _config._env_value("DECAT_RUNTIME_PHYSICS_CONF_POWER", None, "1.0")
    )
    physics_conf_power = float(max(0.25, min(4.0, physics_conf_power)))
    start_epoch = float(max_epochs) * physics_start_frac
    warmup_epochs = max(1.0, float(max_epochs) * max(1e-06, physics_warmup_frac))
    physics_epoch_gate = (
        float(min(1.0, max(0.0, (float(epoch) - start_epoch) / warmup_epochs)))
        * physics_max_gate
    )
    use_physics = lam_phys > 1e-12 and physics_epoch_gate > 1e-12
    use_invariant = lam_inv > 1e-12
    use_dro = lam_dro > 1e-12
    physics_meta = dict(physics_meta or {})
    if (
        not physics_meta
        and physics_sign_prior is not None
        and (physics_sign_prior.numel() > 0)
    ):
        n_cat = int(physics_sign_prior.numel())
        physics_meta = {
            "sign": physics_sign_prior.to(torch.float32),
            "confidence": torch.ones(
                (n_cat,), dtype=torch.float32, device=physics_sign_prior.device
            ),
            "target_slope": torch.zeros(
                (n_cat,), dtype=torch.float32, device=physics_sign_prior.device
            ),
            "ph_lo": torch.full(
                (n_cat,), -1.0, dtype=torch.float32, device=physics_sign_prior.device
            ),
            "ph_hi": torch.full(
                (n_cat,), 1.0, dtype=torch.float32, device=physics_sign_prior.device
            ),
            "valid": (torch.abs(physics_sign_prior.to(torch.float32)) > 0.5).to(
                torch.float32
            ),
        }
    total = {"loss": 0.0, "pred": 0.0, "inv": 0.0, "dro": 0.0, "phys": 0.0, "n": 0.0}
    for batch in dataloader:
        fingerprint = batch["fingerprint"].to(device, non_blocking=non_blocking)
        numeric = batch["numeric"].to(device, non_blocking=non_blocking)
        y = batch["logk"].to(device, non_blocking=non_blocking)
        category = batch["category"].to(device, non_blocking=non_blocking)
        base_idx = batch.get("base_idx")
        if base_idx is not None:
            base_idx = base_idx.to(device, non_blocking=non_blocking).long()
        trainer.optimizer.zero_grad(set_to_none=True)
        if use_amp:
            from torch.cuda.amp import autocast

            with autocast(enabled=True, dtype=amp_dtype):
                loss, pred_loss, inv_loss, dro_loss, phys_loss = (
                    _training_losses.compute_batch_losses(
                        base_idx=base_idx,
                        category=category,
                        dro_tau=dro_tau,
                        fingerprint=fingerprint,
                        lam_dro=lam_dro,
                        lam_inv=lam_inv,
                        lam_phys=lam_phys,
                        numeric=numeric,
                        phys_mono_w=phys_mono_w,
                        phys_smooth_w=phys_smooth_w,
                        physics_colloc_w=physics_colloc_w,
                        physics_conf_min=physics_conf_min,
                        physics_conf_power=physics_conf_power,
                        physics_delta=physics_delta,
                        physics_epoch_gate=physics_epoch_gate,
                        physics_fd_conf_gated=physics_fd_conf_gated,
                        physics_fd_range_gated=physics_fd_range_gated,
                        physics_fd_smooth_floor=physics_fd_smooth_floor,
                        physics_meta=physics_meta,
                        physics_mode=physics_mode,
                        physics_pinn_curv_conf_min=physics_pinn_curv_conf_min,
                        physics_pinn_curv_w=physics_pinn_curv_w,
                        physics_safe_target=physics_safe_target,
                        physics_sign_prior=physics_sign_prior,
                        physics_slope_clip=physics_slope_clip,
                        physics_slope_target_w=physics_slope_target_w,
                        physics_use_colloc=physics_use_colloc,
                        trainer=trainer,
                        use_amp=use_amp,
                        use_dro=use_dro,
                        use_invariant=use_invariant,
                        use_physics=use_physics,
                        y=y,
                    )
                )
            if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                scaler.scale(loss).backward()
                scaler.unscale_(trainer.optimizer)
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=5.0)
                scaler.step(trainer.optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=5.0)
                trainer.optimizer.step()
        else:
            loss, pred_loss, inv_loss, dro_loss, phys_loss = (
                _training_losses.compute_batch_losses(
                    base_idx=base_idx,
                    category=category,
                    dro_tau=dro_tau,
                    fingerprint=fingerprint,
                    lam_dro=lam_dro,
                    lam_inv=lam_inv,
                    lam_phys=lam_phys,
                    numeric=numeric,
                    phys_mono_w=phys_mono_w,
                    phys_smooth_w=phys_smooth_w,
                    physics_colloc_w=physics_colloc_w,
                    physics_conf_min=physics_conf_min,
                    physics_conf_power=physics_conf_power,
                    physics_delta=physics_delta,
                    physics_epoch_gate=physics_epoch_gate,
                    physics_fd_conf_gated=physics_fd_conf_gated,
                    physics_fd_range_gated=physics_fd_range_gated,
                    physics_fd_smooth_floor=physics_fd_smooth_floor,
                    physics_meta=physics_meta,
                    physics_mode=physics_mode,
                    physics_pinn_curv_conf_min=physics_pinn_curv_conf_min,
                    physics_pinn_curv_w=physics_pinn_curv_w,
                    physics_safe_target=physics_safe_target,
                    physics_sign_prior=physics_sign_prior,
                    physics_slope_clip=physics_slope_clip,
                    physics_slope_target_w=physics_slope_target_w,
                    physics_use_colloc=physics_use_colloc,
                    trainer=trainer,
                    use_amp=use_amp,
                    use_dro=use_dro,
                    use_invariant=use_invariant,
                    use_physics=use_physics,
                    y=y,
                )
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=5.0)
            trainer.optimizer.step()
        bs = float(fingerprint.size(0))
        total["loss"] += float(loss.detach().cpu().item()) * bs
        total["pred"] += float(pred_loss.detach().cpu().item()) * bs
        total["inv"] += float(inv_loss.detach().cpu().item()) * bs
        total["dro"] += float(dro_loss.detach().cpu().item()) * bs
        total["phys"] += float(phys_loss.detach().cpu().item()) * bs
        total["n"] += bs
    n = max(1.0, float(total["n"]))
    return {
        "loss": total["loss"] / n,
        "pred_loss": total["pred"] / n,
        "inv_loss": total["inv"] / n,
        "dro_loss": total["dro"] / n,
        "phys_loss": total["phys"] / n,
    }
