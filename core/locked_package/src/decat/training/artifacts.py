"""Local checkpoint and optional analysis artifact export."""

from __future__ import annotations
import csv
import glob
import json
import os
from datetime import datetime
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import decat.chemistry as _chemistry
import decat.config as _config
import decat.data.dataset as _data_dataset
import decat.evaluation as _evaluation
import decat.interpretation.motifs as _interpretation_motifs
import decat.interpretation.stability as _interpretation_stability
import decat.plotting.attention as _plotting_attention
import decat.plotting.predictions as _plotting_predictions
import decat.runtime as _runtime


def _indices_to_category_labels(
    dataset: _data_dataset.FingerprintReactionDataset, indices: np.ndarray
) -> np.ndarray:
    """Map base indices to category labels aligned with prediction arrays."""
    idx = np.asarray(indices, dtype=int).reshape(-1)
    labels = np.asarray(["unknown"] * idx.shape[0], dtype=object)
    if idx.size == 0:
        return labels
    try:
        cat_cols = list(getattr(dataset, "category_cols", []))
        cat_mat = np.asarray(dataset.category, dtype=np.float32)
        if (
            cat_mat.ndim == 2
            and cat_mat.shape[1] > 0
            and (len(cat_cols) == cat_mat.shape[1])
        ):
            valid = (idx >= 0) & (idx < cat_mat.shape[0])
            if np.any(valid):
                cat_idx = np.argmax(cat_mat[idx[valid]], axis=1).astype(int)
                labels_valid = [
                    (
                        _chemistry._pretty_category_label(cat_cols[i])
                        if 0 <= i < len(cat_cols)
                        else "unknown"
                    )
                    for i in cat_idx
                ]
                labels[valid] = np.asarray(labels_valid, dtype=object)
    except Exception:
        pass
    return labels


def _reorder_by_subset_indices(
    values: np.ndarray, value_indices: np.ndarray, subset_indices: np.ndarray
) -> np.ndarray:
    """Reorder value array to match Subset/DataLoader sample order."""
    arr = np.asarray(values).reshape(-1)
    idx_val = np.asarray(value_indices, dtype=int).reshape(-1)
    idx_ref = np.asarray(subset_indices, dtype=int).reshape(-1)
    if arr.size == 0 or idx_val.size == 0 or idx_ref.size == 0:
        return arr
    try:
        pos = {int(v): i for i, v in enumerate(idx_val.tolist())}
        order = [pos[int(i)] for i in idx_ref.tolist() if int(i) in pos]
        if len(order) == idx_ref.size:
            return arr[np.asarray(order, dtype=int)]
    except Exception:
        pass
    return arr


def _save_fusion_component_artifacts(
    output_dir: str,
    y_true_test: np.ndarray,
    y_pred_fused: np.ndarray,
    component_names: Optional[list],
    component_test_preds: Optional[list],
    global_blend_test_pred: Optional[np.ndarray] = None,
) -> None:
    """Save component-level fusion comparison tables/plots for the test split."""
    if not output_dir:
        return
    y_true = np.asarray(y_true_test, dtype=np.float32).reshape(-1)
    y_fused = np.asarray(y_pred_fused, dtype=np.float32).reshape(-1)
    if y_true.size == 0 or y_fused.size == 0:
        return
    names = list(component_names or [])
    preds = list(component_test_preds or [])
    rows = []
    for name, pred in zip(names, preds):
        pred_arr = np.asarray(pred, dtype=np.float32).reshape(-1)
        n = min(y_true.size, pred_arr.size)
        if n < 2:
            continue
        r2, rmse = _runtime.resolve_hook("metrics", _evaluation._metrics)(
            y_true[:n], pred_arr[:n]
        )
        rows.append(
            {"model": str(name), "r2_test": float(r2), "rmse_test": float(rmse)}
        )
    if global_blend_test_pred is not None:
        gb = np.asarray(global_blend_test_pred, dtype=np.float32).reshape(-1)
        n = min(y_true.size, gb.size)
        if n >= 2:
            r2, rmse = _runtime.resolve_hook("metrics", _evaluation._metrics)(
                y_true[:n], gb[:n]
            )
            rows.append(
                {
                    "model": "global_blend",
                    "r2_test": float(r2),
                    "rmse_test": float(rmse),
                }
            )
    n = min(y_true.size, y_fused.size)
    if n >= 2:
        r2, rmse = _runtime.resolve_hook("metrics", _evaluation._metrics)(
            y_true[:n], y_fused[:n]
        )
        rows.append(
            {"model": "final_fusion", "r2_test": float(r2), "rmse_test": float(rmse)}
        )
    if not rows:
        return
    try:
        csv_path = os.path.join(output_dir, "fusion_component_metrics.csv")
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "r2_test", "rmse_test"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    except Exception:
        pass
    labels = [str(r["model"]) for r in rows]
    r2_vals = [float(r["r2_test"]) for r in rows]
    rmse_vals = [float(r["rmse_test"]) for r in rows]
    x = np.arange(len(labels))
    try:
        fig, axes = plt.subplots(1, 2, figsize=(max(8, len(labels) * 1.2), 4.2))
        ax1, ax2 = axes
        ax1.bar(x, r2_vals, color="#4C78A8")
        ax1.set_title("Test R2 by component")
        ax1.set_xticks(x, labels, rotation=25, ha="right")
        ax1.grid(axis="y", alpha=0.25)
        ax2.bar(x, rmse_vals, color="#F58518")
        ax2.set_title("Test RMSE by component")
        ax2.set_xticks(x, labels, rotation=25, ha="right")
        ax2.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        _plotting_predictions._savefig_with_pdf(
            fig,
            os.path.join(output_dir, "fusion_component_metrics.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    except Exception:
        pass


def _export_best_plots(
    *,
    model: nn.Module,
    dataset: _data_dataset.FingerprintReactionDataset,
    explain_loader: Optional[DataLoader],
    test_loader: DataLoader,
    test_subset_indices: np.ndarray,
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    idx_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    idx_test_pred: np.ndarray,
    component_names: Optional[list],
    component_test_preds: Optional[list],
    global_blend_test_pred: Optional[np.ndarray],
    iter_id: int,
    model_mode: str,
    fp_bits: int,
    fp_dim_for_model: int,
) -> Optional[str]:
    """Generate plots that are consistent with the actually selected (possibly fused) predictions."""
    model_name = _runtime._to_prefix_model_name(model_mode)
    prefix_name = (
        f"iter{int(iter_id)}_fp{int(fp_bits)}_k{int(fp_dim_for_model)}_{model_name}"
    )
    y_train_true = np.asarray(y_train_true, dtype=np.float32).reshape(-1)
    y_train_pred = np.asarray(y_train_pred, dtype=np.float32).reshape(-1)
    y_test_true = np.asarray(y_test_true, dtype=np.float32).reshape(-1)
    y_test_pred = np.asarray(y_test_pred, dtype=np.float32).reshape(-1)
    idx_train_pred = np.asarray(idx_train_pred, dtype=int).reshape(-1)
    idx_test_pred = np.asarray(idx_test_pred, dtype=int).reshape(-1)
    test_subset_indices = np.asarray(test_subset_indices, dtype=int).reshape(-1)
    try:
        if y_test_true.size > 0 and y_test_pred.size > 0:
            _plotting_predictions.plot_pred_vs_true_and_residuals(
                y_test_true, y_test_pred, prefix_name
            )
        if (
            y_train_true.size > 0
            and y_train_pred.size > 0
            and (y_test_true.size > 0)
            and (y_test_pred.size > 0)
        ):
            _plotting_predictions.plot_train_test_vs_true_with_band(
                y_train_true=y_train_true,
                y_train_pred=y_train_pred,
                y_test_true=y_test_true,
                y_test_pred=y_test_pred,
                prefix=prefix_name,
            )
            _plotting_predictions.plot_train_density_test_overlay(
                y_train_true=y_train_true,
                y_train_pred=y_train_pred,
                y_test_true=y_test_true,
                y_test_pred=y_test_pred,
                prefix=prefix_name,
            )
            if _config._env_bool("DECAT_RUNTIME_PLOT_CATEGORY_METRICS", True):
                y_train_cat = _indices_to_category_labels(dataset, idx_train_pred)
                y_test_cat = _indices_to_category_labels(dataset, idx_test_pred)
                if (
                    y_train_cat.size == y_train_true.size
                    and y_test_cat.size == y_test_true.size
                ):
                    _plotting_predictions.plot_category_metrics_paper(
                        np.concatenate([y_train_true, y_test_true]),
                        np.concatenate([y_train_pred, y_test_pred]),
                        np.concatenate([y_train_cat, y_test_cat]),
                        prefix_name,
                    )
        y_test_pred_loader = _reorder_by_subset_indices(
            values=y_test_pred,
            value_indices=idx_test_pred,
            subset_indices=test_subset_indices,
        )
        if _config._env_bool("DECAT_RUNTIME_PLOT_FEATURE_CORR_NETWORK20", True):
            _plotting_attention.plot_feature_corr_network_top20(
                test_loader, y_test_pred_loader, prefix_name
            )
        try:
            top_idx, feat_names, bit_map, rank_idx = (
                _plotting_attention.plot_attention_cls_and_category(
                    model, test_loader, prefix_name
                )
            )
            ranked_bit_names = []
            heatmap_focus_bits = []
            family_focus_map = {}
            if (
                isinstance(feat_names, list)
                and isinstance(rank_idx, np.ndarray)
                and (rank_idx.size > 0)
            ):
                ranked_bit_names = [
                    str(feat_names[int(i)])
                    for i in rank_idx.tolist()
                    if 0 <= int(i) < len(feat_names)
                ]
                heatmap_focus_bits = list(ranked_bit_names)
            _interpretation_motifs.export_attention_substructure_artifacts(
                dataset,
                idx_train_pred,
                ranked_bit_names,
                prefix_name,
                artifact_tag="attn",
            )
            if (
                _config._env_bool("DECAT_RUNTIME_CMA_ENABLE", True)
                and explain_loader is not None
            ):
                consensus_info = (
                    _interpretation_stability.export_consensus_motif_artifacts(
                        model=model,
                        dataset=dataset,
                        explain_loader=explain_loader,
                        train_indices=idx_train_pred,
                        prefix=prefix_name,
                    )
                )
                if isinstance(consensus_info, dict):
                    consensus_ranked_bit_names = list(
                        consensus_info.get("ranked_bit_names", []) or []
                    )
                    family_focus_map = dict(
                        consensus_info.get("bit_to_family", {}) or {}
                    )
                    if consensus_ranked_bit_names:
                        heatmap_focus_bits = list(consensus_ranked_bit_names)
            if _config._env_bool("DECAT_RUNTIME_PLOT_ATTN_CLS_BY_CATEGORY", True):
                _plotting_attention.plot_attn_cls_by_category(
                    model, test_loader, prefix_name, focus_bits=heatmap_focus_bits
                )
                if family_focus_map:
                    _plotting_attention.plot_attn_family_by_category(
                        model, test_loader, prefix_name, bit_to_family=family_focus_map
                    )
        except Exception:
            pass
        if _config._env_bool("DECAT_RUNTIME_KEEP_CORE_IMAGES_ONLY", True):
            _plotting_predictions.cleanup_attention_output_images(
                _runtime.get_run_output_dir(prefix_name)
            )
        out_dir = _runtime.get_run_output_dir(prefix_name)
        _save_fusion_component_artifacts(
            output_dir=out_dir,
            y_true_test=y_test_true,
            y_pred_fused=y_test_pred,
            component_names=component_names,
            component_test_preds=component_test_preds,
            global_blend_test_pred=global_blend_test_pred,
        )
        return out_dir
    except Exception as e:
        print(f"[DECAT] WARN: failed to generate best plots for {prefix_name}: {e}")
        return None


def _save_best_result() -> Optional[str]:
    if not np.isfinite(
        _evaluation._safe_float(
            _runtime.context.best_result.get("objective_r2", float("nan"))
        )
    ):
        return None
    if _runtime.context.run_output_dir:
        path = os.path.join(_runtime.context.run_output_dir, "best_params.json")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_runtime.context.out_dir, f"best_params_{stamp}.json")
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_csv_path": str(_runtime.context.data_csv_path),
        "run_output_dir": (
            str(_runtime.context.run_output_dir)
            if _runtime.context.run_output_dir
            else None
        ),
        "iter": int(_runtime.context.best_result.get("iter", 0)),
        "objective_target": str(
            _runtime.context.best_result.get("objective_target", "val")
        ),
        "objective_r2": _evaluation._safe_float(
            _runtime.context.best_result.get("objective_r2")
        ),
        "train_r2": _evaluation._safe_float(
            _runtime.context.best_result.get("train_r2")
        ),
        "train_rmse": _evaluation._safe_float(
            _runtime.context.best_result.get("train_rmse")
        ),
        "val_r2": _evaluation._safe_float(_runtime.context.best_result.get("val_r2")),
        "val_rmse": _evaluation._safe_float(
            _runtime.context.best_result.get("val_rmse")
        ),
        "test_r2": _evaluation._safe_float(_runtime.context.best_result.get("test_r2")),
        "test_rmse": _evaluation._safe_float(
            _runtime.context.best_result.get("test_rmse")
        ),
        "model_mode": str(_runtime.context.best_result.get("model_mode", "")),
        "fusion_mode": str(_runtime.context.best_result.get("fusion_mode", "")),
        "params": _runtime.context.best_result.get("params", {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _find_latest_best_params_json() -> Optional[str]:
    patterns = [
        os.path.join(_runtime.context.out_dir, "run_*", "best_params.json"),
        os.path.join(_runtime.context.out_dir, "best_params_*.json"),
        os.path.join(_runtime.context.out_dir, "best_params_*.json"),
        os.path.join(_runtime.context.out_dir, "best_params_*.json"),
        os.path.join(_runtime.context.out_dir, "best_params_*.json"),
        os.path.join(_runtime.context.out_dir, "best_params_*.json"),
        os.path.join(_runtime.context.out_dir, "best_params_*.json"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]
