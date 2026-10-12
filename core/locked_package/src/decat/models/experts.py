"""Attention and gated residual expert components."""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import decat.config as _config
import decat.data.features as _data_features
import decat.models.attention as _models_attention


class GatedFusion(nn.Module):
    """Learn a gate for combining fingerprint and numeric representations."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.linear_mol = nn.Linear(d_model, d_model)
        self.linear_num = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mol_feat: torch.Tensor, num_feat: torch.Tensor) -> torch.Tensor:
        h_mol = self.linear_mol(mol_feat)
        h_num = self.linear_num(num_feat)
        z = self.gate(torch.cat([mol_feat, num_feat], dim=1))
        fused = z * h_mol + (1.0 - z) * h_num
        return self.norm(self.dropout(fused))


class ChemBiasAttentionRegressor(nn.Module):
    """Attention expert with chemistry-aware feature and condition processing."""

    def __init__(
        self,
        fingerprint_dim: int,
        numeric_dim: int,
        config: Optional[_config.FingerprintConfig] = None,
    ):
        super().__init__()
        self.config = config or _config.FingerprintConfig
        self.fingerprint_dim = int(fingerprint_dim)
        self.numeric_dim = int(numeric_dim)
        self.d_model = int(self.config.d_model)
        self.max_fp_tokens = int(
            min(getattr(self.config, "max_fp_tokens", 128), self.fingerprint_dim)
        )
        self.uses_sparse_tokens = self.max_fp_tokens < self.fingerprint_dim
        self.attn_pooling = (
            str(getattr(self.config, "attn_pooling", "attn")).strip().lower() or "attn"
        )
        self.fp_bit_dropout = float(getattr(self.config, "fp_bit_dropout", 0.0) or 0.0)
        self.n_heads = int(getattr(self.config, "n_heads", 4))
        self.num_input_dim = max(1, self.numeric_dim)
        self.enable_chem_bias = _config._env_bool_pair(
            "DECAT_RUNTIME_ENABLE_CHEM_ATTN_BIAS",
            "DECAT_RUNTIME_ENABLE_CHEM_ATTN_BIAS_FALLBACK",
            True,
        )
        self.chem_bias_alpha = float(
            _config._env_value(
                "DECAT_RUNTIME_CHEM_ATTN_ALPHA",
                "DECAT_RUNTIME_CHEM_ATTN_ALPHA_FALLBACK",
                "0.30",
            )
        )
        self.chem_bias_learnable_gate = _config._env_bool_pair(
            "DECAT_RUNTIME_CHEM_BIAS_LEARNABLE_GATE", None, False
        )
        self.chem_bias_gate_init = float(
            _config._env_value("DECAT_RUNTIME_CHEM_BIAS_GATE_INIT", None, "0.08")
        )
        self.chem_bias_gate_max = float(
            _config._env_value(
                "DECAT_RUNTIME_CHEM_BIAS_GATE_MAX",
                None,
                str(max(1e-06, self.chem_bias_alpha)),
            )
        )
        self.enable_learned_bit_bias = _config._env_bool_pair(
            "DECAT_RUNTIME_ENABLE_LEARNED_BIT_ATTN_BIAS",
            "DECAT_RUNTIME_ENABLE_LEARNED_BIT_ATTN_BIAS_FALLBACK",
            True,
        )
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.fp_index_embed = nn.Embedding(self.fingerprint_dim, self.d_model)
        self.chem_bit_bias = nn.Parameter(
            torch.zeros(self.fingerprint_dim, dtype=torch.float32)
        )
        self.fp_val_embed = nn.Linear(1, self.d_model)
        self.fp_token_norm = nn.LayerNorm(self.d_model)
        self.fp_token_dropout = nn.Dropout(float(self.config.dropout))
        self.num_proj = nn.Sequential(
            nn.Linear(self.num_input_dim, self.d_model),
            nn.GELU() if self.config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )
        self.fusion = GatedFusion(self.d_model, dropout=self.config.dropout)
        self.encoder_layers = nn.ModuleList(
            [
                _models_attention.ChemistryAwareTransformerEncoderLayer(
                    self.d_model,
                    self.n_heads,
                    dropout=float(self.config.dropout),
                    norm_first=bool(getattr(self.config, "norm_first", False)),
                )
                for _ in range(int(getattr(self.config, "n_layers", 2)))
            ]
        )
        self.pool_norm = nn.LayerNorm(self.d_model)
        self.pool_attn = nn.Linear(self.d_model, 1)
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU() if self.config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.d_model // 2, 1),
        )
        self.numeric_bias_proj = nn.Linear(max(1, self.numeric_dim), 1)
        self.attn_cache = {"self": []}
        self.capture_attn = False
        self.last_attn_bias = None
        self.last_key_padding_mask = None
        self.last_token_indices = None
        self.last_token_active = None
        self.last_chem_bias_scale = None
        if self.chem_bias_learnable_gate:
            gate_max = float(max(1e-06, self.chem_bias_gate_max))
            gate_init = float(
                max(1e-06, min(gate_max - 1e-06, self.chem_bias_gate_init))
            )
            init_prob = float(max(1e-06, min(1.0 - 1e-06, gate_init / gate_max)))
            init_logit = np.log(init_prob / (1.0 - init_prob))
            self.chem_bias_gate_logit = nn.Parameter(
                torch.tensor(float(init_logit), dtype=torch.float32)
            )
        else:
            self.register_parameter("chem_bias_gate_logit", None)
        if self.enable_learned_bit_bias:
            nn.init.normal_(self.chem_bit_bias, mean=0.0, std=0.015)

    def _chem_bias_scale(self, device: torch.device) -> torch.Tensor:
        if self.chem_bias_learnable_gate and self.chem_bias_gate_logit is not None:
            gate_max = torch.as_tensor(
                float(max(1e-06, self.chem_bias_gate_max)),
                dtype=torch.float32,
                device=device,
            )
            scale = (
                torch.sigmoid(self.chem_bias_gate_logit.to(torch.float32)) * gate_max
            )
        else:
            scale = torch.as_tensor(
                float(self.chem_bias_alpha), dtype=torch.float32, device=device
            )
        self.last_chem_bias_scale = float(scale.detach().cpu().item())
        return scale

    def _build_tokens(
        self, fingerprint: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.uses_sparse_tokens:
            active_full = torch.abs(fingerprint) > 1e-06
            order = torch.argsort(~active_full, dim=1, stable=True)
            idx = order[:, : self.max_fp_tokens]
            vals = fingerprint.gather(1, idx)
            active_bits = torch.abs(vals) > 1e-06
            fp_bits_tokens = self.fp_index_embed(idx) + self.fp_val_embed(
                vals.unsqueeze(-1)
            )
            fp_bits_tokens = fp_bits_tokens * active_bits.unsqueeze(-1).to(
                fp_bits_tokens.dtype
            )
            cls = self.cls_token.expand(vals.size(0), -1, -1)
            fp_tokens = torch.cat([cls, fp_bits_tokens], dim=1)
            key_padding_mask = torch.cat(
                [
                    torch.zeros(
                        (vals.size(0), 1), dtype=torch.bool, device=fingerprint.device
                    ),
                    ~active_bits,
                ],
                dim=1,
            )
            token_values = vals
        else:
            vals = fingerprint
            active_bits = torch.abs(vals) > 1e-06
            idx = (
                torch.arange(self.fingerprint_dim, device=fingerprint.device)
                .unsqueeze(0)
                .expand(vals.size(0), -1)
            )
            fp_bits_tokens = self.fp_index_embed(idx) + self.fp_val_embed(
                vals.unsqueeze(-1)
            )
            fp_bits_tokens = fp_bits_tokens * active_bits.unsqueeze(-1).to(
                fp_bits_tokens.dtype
            )
            cls = self.cls_token.expand(vals.size(0), -1, -1)
            fp_tokens = torch.cat([cls, fp_bits_tokens], dim=1)
            key_padding_mask = torch.cat(
                [
                    torch.zeros(
                        (vals.size(0), 1), dtype=torch.bool, device=fingerprint.device
                    ),
                    ~active_bits,
                ],
                dim=1,
            )
            token_values = vals
        all_pad = key_padding_mask.all(dim=1)
        if all_pad.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad, 0] = False
        self.last_token_indices = idx.detach()
        self.last_token_active = active_bits.detach()
        return (fp_tokens, key_padding_mask, token_values)

    def _build_attention_bias(
        self,
        token_values: torch.Tensor,
        token_indices: torch.Tensor,
        numeric: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.enable_chem_bias:
            return None
        bit_strength = torch.abs(token_values).to(torch.float32).clamp(0.0, 1.0)
        if self.enable_learned_bit_bias and token_indices is not None:
            bit_prior = torch.tanh(
                self.chem_bit_bias[token_indices.long()].to(torch.float32)
            )
            bit_strength = (bit_strength * (1.0 + 0.75 * bit_prior)).clamp(0.0, 2.0)
        bsz, n_tokens = bit_strength.shape
        length = n_tokens + 1
        pair = bit_strength.unsqueeze(2) * bit_strength.unsqueeze(1)
        bias = torch.zeros(
            (bsz, length, length), dtype=torch.float32, device=token_values.device
        )
        bias[:, 1:, 1:] = pair
        bias[:, 0, 1:] = bit_strength
        bias[:, 1:, 0] = bit_strength
        numeric_full = _data_features._match_feature_dim(
            numeric.to(torch.float32), max(1, self.numeric_dim)
        )
        num_gate = torch.tanh(self.numeric_bias_proj(numeric_full)).view(-1, 1, 1)
        bias = bias * (1.0 + 0.25 * num_gate)
        bias = torch.clamp(bias * self._chem_bias_scale(token_values.device), -2.0, 2.0)
        valid = (~key_padding_mask).to(torch.float32)
        bias = bias * valid.unsqueeze(1) * valid.unsqueeze(2)
        row_mean = bias.mean(dim=-1, keepdim=True)
        return bias - row_mean

    def forward(self, fingerprint: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        if numeric.dim() == 1:
            numeric = numeric.unsqueeze(-1)
        if self.training and self.fp_bit_dropout > 0.0:
            p = float(max(0.0, min(1.0, self.fp_bit_dropout)))
            if p > 0.0:
                active = torch.abs(fingerprint) > 1e-06
                drop = (torch.rand_like(fingerprint) < p) & active
                if drop.any():
                    fingerprint = fingerprint.masked_fill(drop, 0.0)
        fp_tokens, key_padding_mask, token_values = self._build_tokens(fingerprint)
        fp_tokens = self.fp_token_dropout(self.fp_token_norm(fp_tokens))
        attn_bias = self._build_attention_bias(
            token_values, self.last_token_indices, numeric, key_padding_mask
        )
        self.last_key_padding_mask = key_padding_mask.detach()
        self.last_attn_bias = attn_bias.detach() if attn_bias is not None else None
        self.attn_cache["self"] = []
        for layer in self.encoder_layers:
            fp_tokens = layer(
                fp_tokens, key_padding_mask=key_padding_mask, attn_mask=attn_bias
            )
            if self.capture_attn and layer.last_attn is not None:
                self.attn_cache["self"].append(layer.last_attn.detach())
        if self.attn_pooling in {"cls", "class"}:
            mol_feat = fp_tokens[:, 0, :]
        elif self.attn_pooling in {"mean", "avg", "average"}:
            keep = (~key_padding_mask).to(fp_tokens.dtype)
            keep_bits = keep[:, 1:]
            denom = keep_bits.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_bits = (fp_tokens[:, 1:, :] * keep_bits.unsqueeze(-1)).sum(
                dim=1
            ) / denom
            has_bits = (keep_bits.sum(dim=1) > 0).unsqueeze(-1)
            mol_feat = torch.where(has_bits, mean_bits, fp_tokens[:, 0, :])
        else:
            scores = self.pool_attn(fp_tokens).squeeze(-1).float()
            scores = scores.masked_fill(key_padding_mask, -1000000000.0)
            w = torch.softmax(scores, dim=1).to(fp_tokens.dtype).unsqueeze(-1)
            mol_feat = (fp_tokens * w).sum(dim=1)
        mol_feat = self.pool_norm(mol_feat)
        num_feat = self.num_proj(
            _data_features._match_feature_dim(numeric, self.num_input_dim)
        )
        fused = self.fusion(mol_feat, num_feat)
        return self.regressor(fused).squeeze(-1)


def _build_attention_expert(
    fingerprint_dim: int, numeric_dim: int, config: _config.FingerprintConfig
) -> nn.Module:
    use_bias = _config._env_bool_pair(
        "DECAT_RUNTIME_ENABLE_CHEM_ATTN_BIAS",
        "DECAT_RUNTIME_ENABLE_CHEM_ATTN_BIAS_FALLBACK",
        True,
    )
    if use_bias:
        return ChemBiasAttentionRegressor(
            fingerprint_dim=fingerprint_dim, numeric_dim=numeric_dim, config=config
        )
    return _models_attention.AttentionFingerprintTransformer(
        fingerprint_dim=fingerprint_dim, numeric_dim=numeric_dim, config=config
    )
