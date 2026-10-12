"""Numerical evaluation, prediction arrays and statistical helpers."""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
import decat.data.dataset as _data_dataset

try:
    from scipy.stats import gaussian_kde, pearsonr, ttest_rel, wilcoxon
except Exception:
    gaussian_kde = None
    pearsonr = None
    ttest_rel = None
    wilcoxon = None


def _safe_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 2:
        return (float("nan"), float("nan"))
    return (
        float(r2_score(y_true, y_pred)),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def _predict_raw(
    model: nn.Module, dataloader: DataLoader, device: str, return_indices: bool = False
) -> Tuple[np.ndarray, ...]:
    model.eval()
    base_ds = _data_dataset._get_base_dataset(dataloader.dataset)
    y_true_list = []
    y_pred_list = []
    idx_list = []
    non_blocking = str(device).lower().startswith("cuda")
    with torch.no_grad():
        for batch in dataloader:
            fp = batch["fingerprint"].to(device, non_blocking=non_blocking)
            numeric = batch["numeric"].to(device, non_blocking=non_blocking)
            y_raw = batch["logk_raw"].detach().cpu().numpy().reshape(-1)
            preds_scaled = model(fp, numeric)
            preds_scaled_np = (
                preds_scaled.detach().to(torch.float32).cpu().numpy().reshape(-1, 1)
            )
            preds_raw = base_ds.logk_scaler.inverse_transform(preds_scaled_np).reshape(
                -1
            )
            base_idx = batch.get("base_idx")
            if isinstance(base_idx, torch.Tensor):
                idx_np = base_idx.detach().cpu().numpy().reshape(-1)
            elif base_idx is None:
                idx_np = np.full((y_raw.shape[0],), -1, dtype=int)
            else:
                idx_np = np.asarray(base_idx).reshape(-1)
            y_true_list.append(y_raw)
            y_pred_list.append(preds_raw)
            idx_list.append(idx_np.astype(int, copy=False))
    if not y_true_list:
        empty = np.asarray([], dtype=np.float32)
        if return_indices:
            return (empty, empty, np.asarray([], dtype=int))
        return (empty, empty)
    y_true = np.concatenate(y_true_list).astype(np.float32)
    y_pred = np.concatenate(y_pred_list).astype(np.float32)
    idx = np.concatenate(idx_list).astype(int, copy=False)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    idx = idx[mask]
    if idx.size > 0 and np.all(idx >= 0):
        order = np.argsort(idx, kind="stable")
        y_true = y_true[order]
        y_pred = y_pred[order]
        idx = idx[order]
    if return_indices:
        return (y_true, y_pred, idx)
    return (y_true, y_pred)


def _predict_split_triplet(
    model: nn.Module,
    device: str,
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    y_train, pred_train, idx_train = _predict_raw(
        model, train_eval_loader, device, return_indices=True
    )
    y_val, pred_val, idx_val = _predict_raw(
        model, val_loader, device, return_indices=True
    )
    y_test, pred_test, idx_test = _predict_raw(
        model, test_loader, device, return_indices=True
    )
    return (
        y_train,
        pred_train,
        idx_train,
        y_val,
        pred_val,
        idx_val,
        y_test,
        pred_test,
        idx_test,
    )


def _softmax_weights(scores: list, temperature: float = 0.02) -> np.ndarray:
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        return np.ones(arr.shape[0], dtype=float) / float(arr.shape[0])
    t = float(max(0.0001, temperature))
    z = (arr - float(np.max(arr))) / t
    z = np.clip(z, -60.0, 60.0)
    w = np.exp(z)
    s = float(np.sum(w))
    if s <= 0.0 or not np.isfinite(s):
        return np.ones(arr.shape[0], dtype=float) / float(arr.shape[0])
    return w / s


def _paired_significance(
    base_vals: np.ndarray, alt_vals: np.ndarray
) -> Dict[str, float]:
    out = {"p_ttest": float("nan"), "p_wilcoxon": float("nan")}
    if base_vals.size < 2 or alt_vals.size < 2:
        return out
    mask = np.isfinite(base_vals) & np.isfinite(alt_vals)
    b = base_vals[mask]
    a = alt_vals[mask]
    if b.size < 2:
        return out
    if ttest_rel is not None:
        try:
            _, p = ttest_rel(b, a, nan_policy="omit")
            out["p_ttest"] = float(p)
        except Exception:
            pass
    if wilcoxon is not None:
        try:
            _, p = wilcoxon(b, a)
            out["p_wilcoxon"] = float(p)
        except Exception:
            pass
    return out
