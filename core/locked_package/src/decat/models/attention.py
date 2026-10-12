"""Transformer encoders and chemistry-aware attention layers."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import decat.config as _config


class FingerprintTransformer(nn.Module):
    """Encode selected fingerprint tokens and numeric reaction conditions."""

    def __init__(
        self,
        fingerprint_dim: int,
        numeric_dim: int,
        config: Optional[_config.FingerprintConfig] = None,
    ):
        import decat.models.experts as _models_experts

        super().__init__()
        self.config = config or _config.FingerprintConfig
        self.fingerprint_dim = int(fingerprint_dim)
        self.numeric_dim = int(numeric_dim)
        self.d_model = int(self.config.d_model)
        self.num_input_dim = max(1, self.numeric_dim)
        act = nn.GELU() if self.config.activation == "gelu" else nn.ReLU()
        self.fp_proj = nn.Sequential(
            nn.Linear(self.fingerprint_dim, self.d_model),
            act,
            nn.Dropout(self.config.dropout),
        )
        self.num_proj = nn.Sequential(
            nn.Linear(self.num_input_dim, self.d_model),
            act,
            nn.Dropout(self.config.dropout),
        )
        self.fusion = _models_experts.GatedFusion(
            self.d_model, dropout=float(self.config.dropout)
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model, max(8, self.d_model // 2)),
            nn.GELU() if self.config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(max(8, self.d_model // 2), 1),
        )

    def forward(self, fingerprint: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        import decat.data.features as _data_features

        if numeric.dim() == 1:
            numeric = numeric.unsqueeze(-1)
        fp_feat = self.fp_proj(fingerprint)
        num_feat = self.num_proj(
            _data_features._match_feature_dim(numeric, self.num_input_dim)
        )
        fused = self.fusion(fp_feat, num_feat)
        return self.regressor(fused).squeeze(-1)


class _TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.last_attn = None

    def forward(self, x: torch.Tensor):
        attn_out, attn_w = self.self_attn(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        self.last_attn = attn_w
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.linear2(self.dropout(self.act(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        return x


class AttentionFingerprintTransformer(nn.Module):
    """Attention-pooling fingerprint encoder with inspectable token weights."""

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
        self.num_token_dim = max(1, self.numeric_dim)
        self.max_fp_tokens = int(min(self.config.max_fp_tokens, self.fingerprint_dim))
        self.uses_sparse_tokens = self.max_fp_tokens < self.fingerprint_dim
        self.fp_embed = nn.Linear(1, self.d_model)
        self.fp_index_embed = nn.Embedding(self.fingerprint_dim, self.d_model)
        self.fp_pos = nn.Parameter(torch.zeros(1, self.max_fp_tokens, self.d_model))
        self.num_embed = nn.Linear(1, self.d_model)
        self.encoder_layers = nn.ModuleList(
            [
                _TransformerEncoderLayer(
                    self.d_model,
                    int(self.config.n_heads),
                    dropout=float(self.config.dropout),
                )
                for _ in range(int(self.config.n_layers))
            ]
        )
        self.cross_attn = nn.MultiheadAttention(
            self.d_model,
            int(self.config.n_heads),
            dropout=float(self.config.dropout),
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.d_model)
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model, max(8, self.d_model // 2)),
            nn.GELU() if self.config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(max(8, self.d_model // 2), 1),
        )
        self.attn_cache = {"self": [], "cross": None}
        self.capture_attn = False
        self.last_token_indices = None

    def forward(self, fingerprint: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        import decat.data.features as _data_features

        if numeric.dim() == 1:
            numeric = numeric.unsqueeze(-1)
        numeric = _data_features._match_feature_dim(numeric, self.num_token_dim)
        if self.uses_sparse_tokens:
            vals, idx = torch.topk(fingerprint, k=self.max_fp_tokens, dim=1)
            fp_tokens = (
                self.fp_index_embed(idx)
                + self.fp_embed(vals.unsqueeze(-1))
                + self.fp_pos
            )
            self.last_token_indices = idx.detach()
        else:
            fp_tokens = (
                self.fp_embed(fingerprint.unsqueeze(-1))
                + self.fp_pos[:, : self.fingerprint_dim, :]
            )
            dense_idx = (
                torch.arange(self.fingerprint_dim, device=fingerprint.device)
                .unsqueeze(0)
                .expand(fingerprint.size(0), -1)
            )
            self.last_token_indices = dense_idx.detach()
        num_tokens = self.num_embed(numeric.unsqueeze(-1))
        self.attn_cache["self"] = []
        for layer in self.encoder_layers:
            fp_tokens = layer(fp_tokens)
            if self.capture_attn and layer.last_attn is not None:
                self.attn_cache["self"].append(layer.last_attn.detach())
        cross_out, cross_w = self.cross_attn(
            fp_tokens,
            num_tokens,
            num_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        if self.capture_attn:
            self.attn_cache["cross"] = cross_w.detach()
        fp_tokens = self.cross_norm(fp_tokens + cross_out)
        pooled = fp_tokens.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)


class ChemistryAwareTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with an additive chemistry attention bias."""

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.1, norm_first: bool = False
    ):
        super().__init__()
        self.n_heads = int(n_heads)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.last_attn = None
        self.norm_first = bool(norm_first)

    def _expand_attn_mask(
        self, attn_mask: Optional[torch.Tensor], batch_size: int
    ) -> Optional[torch.Tensor]:
        if attn_mask is None:
            return None
        if attn_mask.dim() == 2:
            return attn_mask
        if attn_mask.dim() != 3:
            return None
        if attn_mask.size(0) == 1 and batch_size > 1:
            attn_mask = attn_mask.expand(batch_size, -1, -1)
        if attn_mask.size(0) == batch_size:
            return attn_mask.repeat_interleave(self.n_heads, dim=0)
        return None

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        attn_mask_expanded = self._expand_attn_mask(attn_mask, x.size(0))
        if self.norm_first:
            x_norm = self.norm1(x)
            attn_out, attn_w = self.self_attn(
                x_norm,
                x_norm,
                x_norm,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask_expanded,
                need_weights=True,
                average_attn_weights=False,
            )
            self.last_attn = attn_w
            x = x + self.dropout(attn_out)
            x_norm = self.norm2(x)
            ff = self.linear2(self.dropout(self.act(self.linear1(x_norm))))
            x = x + self.dropout(ff)
            return x
        attn_out, attn_w = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask_expanded,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attn = attn_w
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.linear2(self.dropout(self.act(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        return x


class ChemistryAwareAttentionTransformer(AttentionFingerprintTransformer):
    """Fingerprint attention model with optional category and pH conditioning."""

    def __init__(
        self,
        *args,
        chemistry_aware_attention: bool = True,
        n_categories: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.chemistry_aware_attention = chemistry_aware_attention
        self.n_categories = n_categories
        encoder_layer = ChemistryAwareTransformerEncoderLayer(
            self.d_model, self.n_heads, self.dropout_rate, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        if self.chemistry_aware_attention and n_categories > 1:
            self.chem_bias_param = nn.Parameter(torch.zeros(n_categories, n_categories))
            nn.init.xavier_uniform_(self.chem_bias_param)
        else:
            self.chem_bias_param = None
        self.expert_a = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, 1),
        )
        self.expert_b = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model // 2, 1),
        )
        self.uncertainty_gate = nn.Linear(self.d_model, 2)

    def forward(
        self,
        fingerprint: torch.Tensor,
        numeric: torch.Tensor,
        category: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = fingerprint.shape
        device = fingerprint.device
        fp_idx = fingerprint[:, :, 0].long()
        fp_val = fingerprint[:, :, 1].float().unsqueeze(-1)
        if self.training and self.fp_bit_dropout > 0:
            mask = torch.rand_like(fp_val) > self.fp_bit_dropout
            fp_val = fp_val * mask
        padding_mask = fp_idx < 0
        clean_idx = fp_idx.clone()
        clean_idx[padding_mask] = 0
        x_fp = self.idx_emb(clean_idx) + self.val_emb(fp_val)
        x_num = self.num_emb(numeric).unsqueeze(1)
        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x_num, x_fp], dim=1)
            full_padding_mask = torch.cat(
                [torch.zeros(B, 2, dtype=torch.bool, device=device), padding_mask],
                dim=1,
            )
        else:
            x = torch.cat([x_num, x_fp], dim=1)
            full_padding_mask = torch.cat(
                [torch.zeros(B, 1, dtype=torch.bool, device=device), padding_mask],
                dim=1,
            )
        if self.chem_bias_param is not None and category is not None:
            for layer in self.encoder.layers:
                if hasattr(layer.self_attn, "chemistry_aware_bias"):
                    layer.self_attn.chemistry_aware_bias = self.chem_bias_param
        x = self.encoder(x, src_key_padding_mask=full_padding_mask)
        if self.pooling == "attn":
            q = self.pool_query.expand(B, -1, -1)
            pool_out, _ = self.pool_attn(q, x, x, key_padding_mask=full_padding_mask)
            feature = pool_out.squeeze(1)
        elif self.pooling == "cls":
            feature = x[:, 0, :]
        else:
            mask = (~full_padding_mask).float().unsqueeze(-1)
            sum_x = torch.sum(x * mask, dim=1)
            len_x = torch.sum(mask, dim=1).clamp(min=1e-09)
            feature = sum_x / len_x
        out_a = self.expert_a(feature)
        out_b = self.expert_b(feature)
        gate_logits = self.uncertainty_gate(feature)
        weights = F.softmax(gate_logits, dim=1)
        final_pred = weights[:, 0:1] * out_a + weights[:, 1:2] * out_b
        return final_pred
