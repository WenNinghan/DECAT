"""Attention, integrated-gradient and perturbation attribution."""

from __future__ import annotations
from collections import Counter
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import decat.data.dataset as _data_dataset
import decat.models.attention as _models_attention
import decat.models.experts as _models_experts


def _resolve_attention_model(model: nn.Module) -> Optional[nn.Module]:
    if isinstance(
        model,
        (
            _models_attention.AttentionFingerprintTransformer,
            _models_attention.ChemistryAwareAttentionTransformer,
            _models_experts.ChemBiasAttentionRegressor,
        ),
    ):
        return model
    attn_expert = getattr(model, "attn_expert", None)
    if isinstance(
        attn_expert,
        (
            _models_attention.AttentionFingerprintTransformer,
            _models_attention.ChemistryAwareAttentionTransformer,
            _models_experts.ChemBiasAttentionRegressor,
        ),
    ):
        return attn_expert
    return None


def _capture_attention_once(model: nn.Module, test_loader: DataLoader):
    attn_model = _resolve_attention_model(model)
    if attn_model is None:
        return (None, None)
    device = next(attn_model.parameters()).device
    first_batch = None
    try:
        was_training = bool(attn_model.training)
        attn_model.capture_attn = True
        attn_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                fp = batch["fingerprint"].to(device)
                numeric = batch["numeric"].to(device)
                _ = attn_model(fp, numeric)
                first_batch = batch
                break
        if was_training:
            attn_model.train()
    finally:
        attn_model.capture_attn = False
    return (attn_model, first_batch)


def _token_labels_from_cache(
    attn_model: nn.Module, token_count_no_cls: int, fp_labels: Optional[list]
) -> list:
    labels = [f"token_{i}" for i in range(token_count_no_cls)]
    idx_tensor = getattr(attn_model, "last_token_indices", None)
    if idx_tensor is None:
        if fp_labels is not None and len(fp_labels) >= token_count_no_cls:
            return list(fp_labels[:token_count_no_cls])
        return labels
    try:
        idx_np = idx_tensor.detach().cpu().numpy()
        if idx_np.ndim != 2 or idx_np.shape[1] != token_count_no_cls:
            return labels
        out = []
        for j in range(idx_np.shape[1]):
            col = idx_np[:, j].astype(int)
            if col.size == 0:
                out.append(labels[j])
                continue
            uniq, cnt = np.unique(col, return_counts=True)
            bit_id = int(uniq[int(np.argmax(cnt))])
            if fp_labels is not None and 0 <= bit_id < len(fp_labels):
                out.append(str(fp_labels[bit_id]))
            else:
                out.append(f"fp_{bit_id}")
        return out
    except Exception:
        return labels


def _collect_category_bit_attention_stats(
    model: nn.Module, test_loader: DataLoader
) -> Optional[Dict[str, object]]:
    attn_model = _resolve_attention_model(model)
    if attn_model is None or not hasattr(attn_model, "attn_cache"):
        return None
    base_ds = _data_dataset._unwrap_dataset(test_loader.dataset)
    cat_cols = list(getattr(base_ds, "category_cols", []) or [])
    fp_labels = _data_dataset._get_fp_feature_labels(test_loader.dataset)
    if fp_labels is None:
        return None
    global_sum: Dict[str, float] = {}
    global_cnt: Dict[str, int] = {}
    cat_sum: Dict[str, Dict[str, float]] = {}
    cat_cnt: Dict[str, Dict[str, int]] = {}
    cat_n: Dict[str, int] = {}
    was_training = bool(attn_model.training)
    device = next(attn_model.parameters()).device
    try:
        attn_model.capture_attn = True
        attn_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                fp = batch["fingerprint"].to(device)
                numeric = batch["numeric"].to(device)
                category = batch["category"].detach().cpu().numpy()
                _ = attn_model(fp, numeric)
                self_list = attn_model.attn_cache.get("self", [])
                if not self_list:
                    continue
                last_attn = self_list[-1].detach().cpu()
                if last_attn.ndim != 4:
                    continue
                last_attn = last_attn.mean(dim=1)
                if last_attn.shape[-1] <= 1:
                    continue
                cls_scores = last_attn[:, 0, 1:].numpy()
                token_idx = getattr(attn_model, "last_token_indices", None)
                if token_idx is None:
                    continue
                token_idx = token_idx.detach().cpu().numpy()
                if token_idx.ndim != 2 or token_idx.shape != cls_scores.shape:
                    continue
                cat_ids = (
                    np.argmax(category, axis=1).astype(int)
                    if category.ndim == 2 and category.shape[1] > 0
                    else np.zeros(cls_scores.shape[0], dtype=int)
                )
                for row in range(cls_scores.shape[0]):
                    c_int = int(cat_ids[row]) if row < cat_ids.size else 0
                    c_label = (
                        str(cat_cols[c_int]).replace("cat_", "")
                        if 0 <= c_int < len(cat_cols)
                        else str(c_int)
                    )
                    cat_n[c_label] = int(cat_n.get(c_label, 0) + 1)
                    cat_sum.setdefault(c_label, {})
                    cat_cnt.setdefault(c_label, {})
                    for bit_id, score in zip(
                        token_idx[row].tolist(), cls_scores[row].tolist()
                    ):
                        bit_id = int(bit_id)
                        bit_name = (
                            str(fp_labels[bit_id])
                            if 0 <= bit_id < len(fp_labels)
                            else f"fp_{bit_id}"
                        )
                        val = float(score)
                        global_sum[bit_name] = float(
                            global_sum.get(bit_name, 0.0) + val
                        )
                        global_cnt[bit_name] = int(global_cnt.get(bit_name, 0) + 1)
                        cat_sum[c_label][bit_name] = float(
                            cat_sum[c_label].get(bit_name, 0.0) + val
                        )
                        cat_cnt[c_label][bit_name] = int(
                            cat_cnt[c_label].get(bit_name, 0) + 1
                        )
    finally:
        attn_model.capture_attn = False
        if was_training:
            attn_model.train()
    if not global_sum or not cat_n:
        return None
    return {
        "global_sum": global_sum,
        "global_cnt": global_cnt,
        "cat_sum": cat_sum,
        "cat_cnt": cat_cnt,
        "cat_n": cat_n,
        "fp_labels": fp_labels,
        "cat_cols": cat_cols,
    }


def _minmax_scale_array(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    mask = np.isfinite(x)
    if not mask.any():
        return np.zeros_like(x, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    vals = x[mask]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if abs(vmax - vmin) < 1e-12:
        out[mask] = 1.0 if vmax > 0 else 0.0
        return out
    out[mask] = (vals - vmin) / (vmax - vmin)
    return out


def _row_normalize_nonnegative(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    denom = arr.sum(axis=1, keepdims=True)
    denom = np.where(denom > 1e-12, denom, 1.0)
    return arr / denom


def _collect_explain_arrays(
    explain_loader: DataLoader, max_samples: Optional[int] = None
) -> Dict[str, object]:
    feature_labels = _data_dataset._get_fp_feature_labels(explain_loader.dataset) or []
    fp_blocks = []
    num_blocks = []
    idx_blocks = []
    total = 0
    fallback_idx = 0
    cap = None if max_samples is None else int(max(1, max_samples))
    for batch in explain_loader:
        fp_np = (
            batch["fingerprint"].detach().cpu().numpy().astype(np.float32, copy=False)
        )
        num_np = batch["numeric"].detach().cpu().numpy().astype(np.float32, copy=False)
        base_idx = batch.get("base_idx")
        if isinstance(base_idx, torch.Tensor):
            idx_np = base_idx.detach().cpu().numpy().astype(int, copy=False).reshape(-1)
        elif base_idx is None:
            idx_np = np.arange(fallback_idx, fallback_idx + fp_np.shape[0], dtype=int)
        else:
            idx_np = np.asarray(base_idx, dtype=int).reshape(-1)
        fallback_idx += int(fp_np.shape[0])
        if cap is not None and total + fp_np.shape[0] > cap:
            keep = max(0, cap - total)
            if keep <= 0:
                break
            fp_np = fp_np[:keep]
            num_np = num_np[:keep]
            idx_np = idx_np[:keep]
        if fp_np.size == 0:
            continue
        fp_blocks.append(fp_np)
        num_blocks.append(num_np)
        idx_blocks.append(idx_np)
        total += int(fp_np.shape[0])
        if cap is not None and total >= cap:
            break
    if not fp_blocks:
        return {
            "fingerprint": np.zeros((0, len(feature_labels)), dtype=np.float32),
            "numeric": np.zeros((0, 0), dtype=np.float32),
            "sample_idx": np.asarray([], dtype=int),
            "feature_labels": feature_labels,
        }
    fingerprint = np.concatenate(fp_blocks, axis=0).astype(np.float32, copy=False)
    numeric = np.concatenate(num_blocks, axis=0).astype(np.float32, copy=False)
    sample_idx = np.concatenate(idx_blocks, axis=0).astype(int, copy=False)
    if not feature_labels:
        feature_labels = [f"fp_{i}" for i in range(int(fingerprint.shape[1]))]
    return {
        "fingerprint": fingerprint,
        "numeric": numeric,
        "sample_idx": sample_idx,
        "feature_labels": list(feature_labels),
    }


def _collect_attention_sample_matrix_from_arrays(
    model: nn.Module,
    fingerprint_np: np.ndarray,
    numeric_np: np.ndarray,
    feature_labels: list,
    batch_size: int = 64,
) -> np.ndarray:
    attn_model = _resolve_attention_model(model)
    n_samples = int(np.asarray(fingerprint_np).shape[0])
    n_features = len(feature_labels)
    out = np.zeros((n_samples, n_features), dtype=np.float32)
    if attn_model is None or n_samples == 0 or n_features == 0:
        return out
    device = next(attn_model.parameters()).device
    was_training = bool(attn_model.training)
    attn_model.eval()
    attn_model.capture_attn = True
    try:
        for start in range(0, n_samples, int(max(1, batch_size))):
            end = min(n_samples, start + int(max(1, batch_size)))
            fp_t = torch.from_numpy(
                np.asarray(fingerprint_np[start:end], dtype=np.float32)
            ).to(device)
            num_t = torch.from_numpy(
                np.asarray(numeric_np[start:end], dtype=np.float32)
            ).to(device)
            with torch.no_grad():
                _ = attn_model(fp_t, num_t)
            self_list = attn_model.attn_cache.get("self", [])
            idx_tensor = getattr(attn_model, "last_token_indices", None)
            if not self_list or idx_tensor is None:
                continue
            attn = self_list[-1].detach().mean(dim=1)
            if (
                isinstance(
                    attn_model, _models_attention.ChemistryAwareAttentionTransformer
                )
                and attn.size(-1) > 1
            ):
                token_scores = attn[:, 0, 1:]
            else:
                token_scores = attn.mean(dim=1)
            idx_np = idx_tensor.detach().cpu().numpy()
            active_t = getattr(attn_model, "last_token_active", None)
            active_np = (
                active_t.detach().cpu().numpy().astype(bool)
                if isinstance(active_t, torch.Tensor)
                else None
            )
            score_np = token_scores.detach().cpu().numpy()
            for row in range(idx_np.shape[0]):
                bit_pos = np.asarray(idx_np[row], dtype=int).reshape(-1)
                vals = np.asarray(score_np[row], dtype=np.float32).reshape(-1)
                size = min(bit_pos.size, vals.size)
                if size <= 0:
                    continue
                bit_pos = bit_pos[:size]
                vals = np.clip(vals[:size], 0.0, None)
                if active_np is not None and row < active_np.shape[0]:
                    keep_mask = np.asarray(active_np[row], dtype=bool).reshape(-1)[
                        :size
                    ]
                else:
                    keep_mask = np.ones((size,), dtype=bool)
                if not keep_mask.any():
                    continue
                vals_keep = vals[keep_mask]
                if float(vals_keep.sum()) > 1e-12:
                    vals_keep = vals_keep / float(vals_keep.sum())
                bit_keep = bit_pos[keep_mask]
                for local_idx, val in zip(bit_keep.tolist(), vals_keep.tolist()):
                    if 0 <= int(local_idx) < n_features:
                        out[start + row, int(local_idx)] += float(val)
    finally:
        attn_model.capture_attn = False
        if was_training:
            attn_model.train()
    return out


def _integrated_gradients_fingerprint(
    model: nn.Module, fingerprint: torch.Tensor, numeric: torch.Tensor, steps: int = 8
) -> torch.Tensor:
    steps = int(max(4, steps))
    baseline = torch.zeros_like(fingerprint)
    total_grad = torch.zeros_like(fingerprint)
    alphas = torch.linspace(
        0.0, 1.0, steps + 1, device=fingerprint.device, dtype=fingerprint.dtype
    )[1:]
    model.zero_grad(set_to_none=True)
    for alpha in alphas:
        fp_interp = (baseline + alpha * (fingerprint - baseline)).detach()
        fp_interp.requires_grad_(True)
        pred = model(fp_interp, numeric)
        grad = torch.autograd.grad(
            pred.sum(),
            fp_interp,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        total_grad = total_grad + grad.detach()
        model.zero_grad(set_to_none=True)
    return (fingerprint - baseline) * (total_grad / float(steps))


def _collect_ig_sample_matrix_from_arrays(
    model: nn.Module,
    fingerprint_np: np.ndarray,
    numeric_np: np.ndarray,
    batch_size: int = 32,
    steps: int = 8,
) -> np.ndarray:
    n_samples = int(np.asarray(fingerprint_np).shape[0])
    n_features = int(np.asarray(fingerprint_np).shape[1]) if n_samples > 0 else 0
    out = np.zeros((n_samples, n_features), dtype=np.float32)
    if n_samples == 0 or n_features == 0:
        return out
    device = next(model.parameters()).device
    was_training = bool(model.training)
    model.eval()
    try:
        for start in range(0, n_samples, int(max(1, batch_size))):
            end = min(n_samples, start + int(max(1, batch_size)))
            fp_t = torch.from_numpy(
                np.asarray(fingerprint_np[start:end], dtype=np.float32)
            ).to(device)
            num_t = torch.from_numpy(
                np.asarray(numeric_np[start:end], dtype=np.float32)
            ).to(device)
            ig_t = _integrated_gradients_fingerprint(model, fp_t, num_t, steps=steps)
            out[start:end] = (
                torch.abs(ig_t).detach().cpu().numpy().astype(np.float32, copy=False)
            )
    finally:
        if was_training:
            model.train()
    return out


def _compute_mask_delta_scores_from_arrays(
    model: nn.Module,
    fingerprint_np: np.ndarray,
    numeric_np: np.ndarray,
    candidate_local_idx: list,
    batch_size: int = 64,
) -> Dict[int, Dict[str, float]]:
    out = {
        int(i): {"delta_sum": 0.0, "delta_count": 0, "delta_max": 0.0}
        for i in candidate_local_idx
    }
    n_samples = int(np.asarray(fingerprint_np).shape[0])
    if n_samples == 0 or not candidate_local_idx:
        return out
    device = next(model.parameters()).device
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, n_samples, int(max(1, batch_size))):
                end = min(n_samples, start + int(max(1, batch_size)))
                fp_t = torch.from_numpy(
                    np.asarray(fingerprint_np[start:end], dtype=np.float32)
                ).to(device)
                num_t = torch.from_numpy(
                    np.asarray(numeric_np[start:end], dtype=np.float32)
                ).to(device)
                pred_ref = model(fp_t, num_t).detach()
                for local_idx in candidate_local_idx:
                    local_idx = int(local_idx)
                    if local_idx < 0 or local_idx >= fp_t.size(1):
                        continue
                    active_mask = torch.abs(fp_t[:, local_idx]) > 1e-06
                    if not bool(active_mask.any()):
                        continue
                    fp_masked = fp_t.clone()
                    fp_masked[active_mask, local_idx] = 0.0
                    pred_masked = model(fp_masked, num_t).detach()
                    delta = torch.abs(pred_masked - pred_ref)
                    delta_active = delta[active_mask]
                    out[local_idx]["delta_sum"] += float(delta_active.sum().item())
                    out[local_idx]["delta_count"] += int(delta_active.numel())
                    out[local_idx]["delta_max"] = max(
                        out[local_idx]["delta_max"], float(delta_active.max().item())
                    )
    finally:
        if was_training:
            model.train()
    return out


def _compute_bootstrap_stability_scores(
    sample_score_matrix: np.ndarray,
    candidate_local_idx: list,
    rounds: int = 32,
    frac: float = 0.75,
    topk: int = 10,
    seed: int = 42,
) -> Dict[int, float]:
    candidate_local_idx = [int(i) for i in candidate_local_idx]
    out = {int(i): 0.0 for i in candidate_local_idx}
    mat = np.asarray(sample_score_matrix, dtype=np.float32)
    if (
        mat.ndim != 2
        or mat.shape[0] == 0
        or mat.shape[1] == 0
        or (not candidate_local_idx)
    ):
        return out
    n_samples = int(mat.shape[0])
    boot_rounds = int(max(1, rounds))
    boot_topk = int(max(1, min(int(topk), len(candidate_local_idx))))
    boot_n = int(max(8, min(n_samples, round(float(frac) * n_samples))))
    rng = np.random.default_rng(int(seed))
    hit_count = Counter()
    for _ in range(boot_rounds):
        if boot_n >= n_samples:
            chosen = np.arange(n_samples, dtype=int)
        else:
            chosen = np.asarray(
                rng.choice(n_samples, size=boot_n, replace=False), dtype=int
            )
        mean_scores = np.asarray(mat[chosen].mean(axis=0), dtype=np.float32)
        candidate_scores = np.asarray(
            [
                mean_scores[i] if 0 <= i < mean_scores.size else -np.inf
                for i in candidate_local_idx
            ],
            dtype=np.float32,
        )
        order = np.argsort(candidate_scores)[::-1][:boot_topk]
        for pos in order.tolist():
            hit_count[int(candidate_local_idx[int(pos)])] += 1
    for local_idx in candidate_local_idx:
        out[int(local_idx)] = float(
            hit_count.get(int(local_idx), 0) / max(boot_rounds, 1)
        )
    return out


def _resolve_chem_attention_model(
    model: nn.Module,
) -> Optional[_models_attention.ChemistryAwareAttentionTransformer]:
    if isinstance(model, _models_attention.ChemistryAwareAttentionTransformer):
        return model
    attn_expert = getattr(model, "attn_expert", None)
    if isinstance(attn_expert, _models_attention.ChemistryAwareAttentionTransformer):
        return attn_expert
    return None
