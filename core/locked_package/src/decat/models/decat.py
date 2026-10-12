"""DECAT model composition shared by training and inference."""

from __future__ import annotations
import copy
from typing import Dict
import torch
import torch.nn as nn
import decat.config as _config
import decat.models.attention as _models_attention
import decat.models.experts as _models_experts
import decat.runtime as _runtime


def _infer_model_mode_from_instance(model: nn.Module) -> str:
    import decat.interpretation.attribution as _interpretation_attribution

    if isinstance(model, _runtime.resolve_hook("model_factory", DualExpertRegressor)):
        return "dual"
    if _interpretation_attribution._resolve_attention_model(model) is not None:
        return "attn"
    return "mlp"


def _build_model_for_mode(
    model_mode: str,
    fingerprint_dim: int,
    numeric_dim: int,
    cfg: _config.FingerprintConfig,
) -> nn.Module:
    mode = str(model_mode).strip().lower()
    if mode == "mlp":
        return _models_attention.FingerprintTransformer(
            fingerprint_dim, numeric_dim, cfg
        )
    if mode == "attn":
        return _models_experts._build_attention_expert(
            fingerprint_dim, numeric_dim, cfg
        )
    return _models_attention.FingerprintTransformer(fingerprint_dim, numeric_dim, cfg)


def _build_cma_cfg_from_model(model: nn.Module) -> _config.FingerprintConfig:
    base_cfg = getattr(model, "config", None)
    cfg = _config.FingerprintConfig()
    if base_cfg is not None:
        for name in (
            "d_model",
            "dropout",
            "activation",
            "n_heads",
            "n_layers",
            "max_fp_tokens",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "max_epochs",
            "early_stopping_patience",
            "save_interval",
            "min_delta",
            "scheduler_type",
            "fingerprint_scale",
            "norm_first",
            "attn_pooling",
            "fp_bit_dropout",
            "base_numeric_dim",
            "chemistry_aware_attention",
            "n_categories",
            "pooling",
        ):
            if hasattr(base_cfg, name):
                setattr(cfg, name, copy.deepcopy(getattr(base_cfg, name)))
    return cfg


class DualExpertRegressor(nn.Module):
    """Combine chemistry-aware attention with a gated residual expert."""

    def __init__(
        self, fingerprint_dim: int, numeric_dim: int, config: _config.FingerprintConfig
    ):
        super().__init__()
        self.config = config
        self.input_numeric_dim = int(numeric_dim)
        self.ph_basis_degree = int(
            max(
                1,
                min(
                    3,
                    int(_config._env_value("DECAT_RUNTIME_PH_BASIS_DEGREE", None, "1")),
                ),
            )
        )
        requested_category_embed_dim = int(
            _config._env_value("DECAT_RUNTIME_CATEGORY_EMBED_DIM", None, "0")
        )
        category_dim = max(0, self.input_numeric_dim - 1)
        self.category_embed_dim = (
            min(category_dim, requested_category_embed_dim)
            if requested_category_embed_dim > 0 and category_dim > 0
            else category_dim
        )
        self.internal_numeric_dim = self.ph_basis_degree + self.category_embed_dim
        self.category_dropout = nn.Dropout(
            float(
                max(
                    0.0,
                    min(
                        0.5,
                        float(
                            _config._env_value(
                                "DECAT_RUNTIME_CATEGORY_DROPOUT", None, "0.0"
                            )
                        ),
                    ),
                )
            )
        )
        if self.category_embed_dim < category_dim:
            self.category_encoder = nn.Sequential(
                nn.Linear(category_dim, self.category_embed_dim, bias=False), nn.GELU()
            )
        else:
            self.category_encoder = None
        self.enable_dual = _config._env_bool_pair(
            "DECAT_RUNTIME_ENABLE_DUAL", "DECAT_RUNTIME_ENABLE_DUAL_FALLBACK", True
        )
        self.use_gate_extra_features = _config._env_bool_pair(
            "DECAT_RUNTIME_GATE_EXTRA_FEATURES",
            "DECAT_RUNTIME_GATE_EXTRA_FEATURES_FALLBACK",
            True,
        )
        self.enable_correction_head = _config._env_bool_pair(
            "DECAT_RUNTIME_ENABLE_CORRECTION_HEAD",
            "DECAT_RUNTIME_ENABLE_CORRECTION_HEAD_FALLBACK",
            True,
        )
        self.attention_min_gate = float(
            _config._env_value(
                "DECAT_RUNTIME_ATTENTION_MIN_GATE",
                "DECAT_RUNTIME_ATTENTION_MIN_GATE_FALLBACK",
                "0.0",
            )
        )
        self.attention_gate_logit_bias = float(
            _config._env_value(
                "DECAT_RUNTIME_ATTENTION_GATE_LOGIT_BIAS",
                "DECAT_RUNTIME_ATTENTION_GATE_LOGIT_BIAS_FALLBACK",
                "0.0",
            )
        )
        self.use_residual_blend = _config._env_bool_pair(
            "DECAT_RUNTIME_USE_RESIDUAL_BLEND", None, True
        )
        self.residual_max = float(
            _config._env_value("DECAT_RUNTIME_RESIDUAL_MAX", None, "0.85")
        )
        self.residual_min = float(
            _config._env_value("DECAT_RUNTIME_RESIDUAL_MIN", None, "0.0")
        )
        self.attention_ph_only = _config._env_bool_pair(
            "DECAT_RUNTIME_ATTN_PH_ONLY", None, False
        )
        self.attn_expert = _models_experts._build_attention_expert(
            fingerprint_dim=fingerprint_dim,
            numeric_dim=(
                self.ph_basis_degree
                if self.attention_ph_only
                else self.internal_numeric_dim
            ),
            config=config,
        )
        self.mlp_expert = (
            _models_attention.FingerprintTransformer(
                fingerprint_dim=fingerprint_dim,
                numeric_dim=self.internal_numeric_dim,
                config=config,
            )
            if self.enable_dual
            else None
        )
        hidden = max(32, int(config.d_model // 2))
        gate_in_dim = self.internal_numeric_dim + (
            3 if self.use_gate_extra_features else 0
        )
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(hidden, 2),
        )
        self.corr_head = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(hidden, 1),
        )

    def _encode_numeric(self, numeric: torch.Tensor) -> torch.Tensor:
        ph = numeric[:, :1].to(torch.float32)
        ph_basis = torch.cat(
            [ph.pow(power) for power in range(1, self.ph_basis_degree + 1)], dim=1
        )
        category = self.category_dropout(numeric[:, 1:].to(torch.float32))
        if self.category_encoder is None:
            return torch.cat([ph_basis, category], dim=1)
        return torch.cat([ph_basis, self.category_encoder(category)], dim=1)

    def forward_components(
        self, fingerprint: torch.Tensor, numeric: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return expert components for residual-safe auxiliary losses."""
        if numeric.dim() == 1:
            numeric = numeric.unsqueeze(-1)
        numeric = self._encode_numeric(numeric)
        attention_numeric = (
            numeric[:, : self.ph_basis_degree] if self.attention_ph_only else numeric
        )
        pred_attn = self.attn_expert(fingerprint, attention_numeric)
        fp_density = (
            (torch.abs(fingerprint) > 1e-06).to(torch.float32).mean(dim=1, keepdim=True)
        )
        fp_energy = torch.abs(fingerprint).to(torch.float32).mean(dim=1, keepdim=True)
        if not self.enable_dual or self.mlp_expert is None:
            return {
                "final": pred_attn,
                "attn": pred_attn,
                "mlp": pred_attn,
                "residual_gate": torch.zeros_like(pred_attn),
                "correction": torch.zeros_like(pred_attn),
            }
        pred_mlp = self.mlp_expert(fingerprint, numeric)
        pred_gap = torch.abs(pred_attn - pred_mlp).unsqueeze(-1)
        if self.use_gate_extra_features:
            gate_input = torch.cat(
                [numeric.to(torch.float32), fp_density, fp_energy, pred_gap], dim=1
            )
        else:
            gate_input = numeric.to(torch.float32)
        logits = self.gate_net(gate_input)
        if abs(float(self.attention_gate_logit_bias)) > 1e-12:
            logits = logits.clone()
            logits[:, 0] = logits[:, 0] + float(self.attention_gate_logit_bias)
        weights = torch.softmax(logits, dim=1)
        min_attn = float(max(0.0, min(0.45, self.attention_min_gate)))
        if min_attn > 1e-12:
            weights = weights.clone()
            weights[:, 0] = min_attn + (1.0 - min_attn) * weights[:, 0]
            weights[:, 1] = (1.0 - min_attn) * weights[:, 1]
        if self.use_residual_blend:
            residual_gate = weights[:, 1].clamp(
                float(max(0.0, min(0.95, self.residual_min))),
                float(max(0.0, min(0.98, self.residual_max))),
            )
            blended = pred_attn + residual_gate * (pred_mlp - pred_attn)
        else:
            blended = weights[:, 0] * pred_attn + weights[:, 1] * pred_mlp
        if not self.enable_correction_head:
            return {
                "final": blended,
                "attn": pred_attn,
                "mlp": pred_mlp,
                "residual_gate": (
                    residual_gate if self.use_residual_blend else weights[:, 1]
                ),
                "correction": torch.zeros_like(blended),
            }
        corr_in = torch.cat(
            [pred_attn.unsqueeze(-1), pred_mlp.unsqueeze(-1), fp_density, fp_energy],
            dim=1,
        )
        correction = torch.tanh(self.corr_head(corr_in)).squeeze(-1) * 0.35
        final = blended + correction
        return {
            "final": final,
            "attn": pred_attn,
            "mlp": pred_mlp,
            "residual_gate": (
                residual_gate if self.use_residual_blend else weights[:, 1]
            ),
            "correction": correction,
        }

    def forward(self, fingerprint: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        return self.forward_components(fingerprint, numeric)["final"]


class ContextualDualExpertRegressor(DualExpertRegressor):
    """DECAT predictor with the contextual correction used by the released checkpoints."""

    def __init__(self, fingerprint_dim: int, numeric_dim: int, model_config):
        super().__init__(fingerprint_dim, numeric_dim, model_config)
        hidden = max(16, int(model_config.d_model // 4))
        self.contextual_residual = nn.Sequential(
            nn.Linear(numeric_dim + 1, hidden),
            nn.GELU(),
            nn.Dropout(float(model_config.dropout)),
            nn.Linear(hidden, 1),
        )

    def forward_components(self, fingerprint: torch.Tensor, numeric: torch.Tensor):
        components = super().forward_components(fingerprint, numeric)
        context = torch.cat(
            [numeric.to(torch.float32), components["residual_gate"].unsqueeze(-1)],
            dim=1,
        )
        adjustment = torch.tanh(self.contextual_residual(context)).squeeze(-1) * 0.15
        components["correction"] = components["correction"] + adjustment
        components["final"] = components["final"] + adjustment
        return components
