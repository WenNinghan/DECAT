"""Attention, correlation and training diagnostic plots."""

from __future__ import annotations
import os
from collections import Counter
from typing import Dict, Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import decat.config as _config
import decat.data.dataset as _data_dataset
import decat.evaluation as _evaluation
import decat.interpretation.attribution as _interpretation_attribution
import decat.models.attention as _models_attention
import decat.plotting.predictions as _plotting_predictions
import decat.runtime as _runtime


def plot_feature_corr_network_top20(
    test_loader: DataLoader, y_pred_loader: np.ndarray, prefix: str
) -> None:
    out_dir = _runtime.get_run_output_dir(prefix)
    y_pred = np.asarray(y_pred_loader, dtype=np.float32).reshape(-1)
    if y_pred.size < 10:
        return
    fp_blocks = []
    ph_blocks = []
    cat_blocks = []
    for batch in test_loader:
        fp_blocks.append(batch["fingerprint"].detach().cpu().numpy().astype(np.float32))
        ph_blocks.append(
            batch["pH"].detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
        )
        cat_blocks.append(batch["category"].detach().cpu().numpy().astype(np.float32))
    if not fp_blocks:
        return
    x_fp = np.vstack(fp_blocks)
    x_ph = (
        np.vstack(ph_blocks)
        if ph_blocks
        else np.zeros((x_fp.shape[0], 0), dtype=np.float32)
    )
    x_cat = (
        np.vstack(cat_blocks)
        if cat_blocks
        else np.zeros((x_fp.shape[0], 0), dtype=np.float32)
    )
    n = min(x_fp.shape[0], y_pred.size)
    if n < 10:
        return
    x_fp = x_fp[:n]
    x_ph = x_ph[:n]
    x_cat = x_cat[:n]
    y_pred = y_pred[:n]
    fp_labels = _data_dataset._get_fp_feature_labels(test_loader.dataset)
    if fp_labels is None or len(fp_labels) < x_fp.shape[1]:
        fp_labels = [f"fp_{i}" for i in range(x_fp.shape[1])]
    feat_blocks = []
    feat_names = []
    if x_ph.size > 0:
        feat_blocks.append(x_ph[:, :1])
        feat_names.append("pH")
    if x_cat.size > 0:
        feat_blocks.append(np.argmax(x_cat, axis=1).astype(np.float32).reshape(-1, 1))
        feat_names.append("category")
    feat_blocks.append(x_fp)
    feat_names.extend([str(x) for x in fp_labels[: x_fp.shape[1]]])
    x_all = np.concatenate(feat_blocks, axis=1).astype(np.float32)
    corr_with_pred = np.zeros(x_all.shape[1], dtype=np.float32)
    pvals_with_pred = np.ones(x_all.shape[1], dtype=np.float32)
    for j in range(x_all.shape[1]):
        col = x_all[:, j]
        if np.std(col) <= 1e-09:
            continue
        try:
            if _evaluation.pearsonr is not None:
                r, p = _evaluation.pearsonr(col, y_pred)
            else:
                r = float(np.corrcoef(col, y_pred)[0, 1])
                p = 1.0
        except Exception:
            r, p = (0.0, 1.0)
        corr_with_pred[j] = 0.0 if not np.isfinite(r) else float(r)
        pvals_with_pred[j] = 1.0 if not np.isfinite(p) else float(p)
    selected_idx = []
    for name in ("pH", "category"):
        if name in feat_names:
            selected_idx.append(int(feat_names.index(name)))
    for idx in np.argsort(np.abs(corr_with_pred))[::-1].tolist():
        idx = int(idx)
        if idx in selected_idx:
            continue
        selected_idx.append(idx)
        if len(selected_idx) >= 20:
            break
    selected_idx = selected_idx[:20]
    if len(selected_idx) < 3:
        return
    top_names = [str(feat_names[i]) for i in selected_idx]
    x_top = x_all[:, selected_idx]
    node_corr = corr_with_pred[selected_idx]
    node_pvals = pvals_with_pred[selected_idx]
    vmax = float(max(0.2, np.nanmax(np.abs(node_corr))))
    node_sizes = 300 + 2200 * np.clip(np.abs(node_corr), 0, 1)
    node_edge_colors = ["#222222" if float(p) < 0.05 else "#9a9a9a" for p in node_pvals]
    node_edge_widths = [2.2 if float(p) < 0.05 else 1.0 for p in node_pvals]
    cmap_nodes = mpl.colors.LinearSegmentedColormap.from_list(
        "target_corr_bright", ["#3B82F6", "#FFFFFF", "#EF4444"], N=256
    )
    norm_nodes = mpl.colors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap_edges = mpl.colors.LinearSegmentedColormap.from_list(
        "feat_corr_bright", ["#EC4899", "#FFFFFF", "#22C55E"], N=256
    )
    norm_edges = mpl.colors.TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
    n_nodes = len(top_names)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 1.0
    coords = {i: (radius * np.cos(a), radius * np.sin(a)) for i, a in enumerate(angles)}
    fig, ax = plt.subplots(figsize=(12.8, 12.8))
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    lim = radius + 0.32
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    edge_candidates = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            xi = x_top[:, i]
            xj = x_top[:, j]
            if np.std(xi) <= 1e-09 or np.std(xj) <= 1e-09:
                continue
            try:
                if _evaluation.pearsonr is not None:
                    r, p = _evaluation.pearsonr(xi, xj)
                else:
                    r = float(np.corrcoef(xi, xj)[0, 1])
                    p = 1.0
            except Exception:
                continue
            if np.isfinite(r):
                edge_candidates.append(
                    (i, j, float(r), 1.0 if not np.isfinite(p) else float(p))
                )
    edge_candidates.sort(key=lambda item: abs(item[2]), reverse=True)
    for i, j, r, p in edge_candidates[: min(40, len(edge_candidates))]:
        if abs(r) < 0.2 and p >= 0.05:
            continue
        xi, yi = coords[i]
        xj, yj = coords[j]
        width = 2.6 + 7.2 * min(abs(r), 1.0)
        alpha = 0.15 + 0.6 * min(abs(r), 1.0)
        color = cmap_edges(norm_edges(r))
        linestyle = "--" if p >= 0.05 else "-"
        line_alpha = alpha * (0.35 if p >= 0.05 else 1.0)
        ax.plot(
            [xi, xj],
            [yi, yj],
            color=color,
            linewidth=width,
            alpha=line_alpha,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=1,
        )
    xs = [coords[i][0] for i in range(n_nodes)]
    ys = [coords[i][1] for i in range(n_nodes)]
    ax.scatter(
        xs,
        ys,
        c=node_corr,
        cmap=cmap_nodes,
        norm=norm_nodes,
        s=node_sizes,
        edgecolors=node_edge_colors,
        linewidths=node_edge_widths,
        zorder=3,
    )
    import matplotlib.patheffects as pe

    label_fontsize = 9.2 if n_nodes <= 20 else 8.2
    for i, name in enumerate(top_names):
        x, y = coords[i]
        angle = float(angles[i])
        dx = float(np.cos(angle))
        dy = float(np.sin(angle))
        node_r_pts = float(np.sqrt(max(float(node_sizes[i]), 1.0) / np.pi))
        offset_pts = node_r_pts + 6.0
        ha = "center" if abs(dx) < 0.2 else "left" if dx > 0 else "right"
        va = "center" if abs(dy) < 0.2 else "bottom" if dy > 0 else "top"
        ax.annotate(
            str(name),
            xy=(x, y),
            xytext=(offset_pts * dx, offset_pts * dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=label_fontsize,
            fontfamily="Arial",
            fontweight="bold",
            color="#111111",
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
            zorder=4,
            annotation_clip=False,
        )
    handles = [
        Line2D(
            [0], [0], color=cmap_edges(norm_edges(0.35)), lw=3.6, label="Positive corr"
        ),
        Line2D(
            [0], [0], color=cmap_edges(norm_edges(-0.35)), lw=3.6, label="Negative corr"
        ),
        Line2D(
            [0], [0], color="gray", lw=2.0, linestyle="--", label="Non-significant edge"
        ),
        plt.scatter(
            [],
            [],
            s=120,
            facecolors="none",
            edgecolors="k",
            linewidths=2.2,
            label="Node significant",
        ),
        plt.scatter(
            [],
            [],
            s=120,
            facecolors="none",
            edgecolors="#777777",
            linewidths=1.0,
            label="Node non-significant",
        ),
    ]
    legend_top = 0.18
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_top),
        frameon=False,
        fontsize=8,
        ncol=3,
        handlelength=2.2,
        handletextpad=0.55,
        columnspacing=1.8,
        labelspacing=0.7,
    )
    fig.subplots_adjust(bottom=legend_top, right=0.8, top=0.93, left=0.06)
    ax_pos = ax.get_position()
    cbar_width = 0.024
    cbar_x = ax_pos.x1 + 0.02
    sm_edges = plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm_edges)
    sm_edges.set_array([])
    cax_edges = fig.add_axes(
        [cbar_x, ax_pos.y0 + 0.6 * ax_pos.height, cbar_width, 0.32 * ax_pos.height]
    )
    cbar_edges = fig.colorbar(sm_edges, cax=cax_edges)
    cbar_edges.ax.set_title("Feature-Feature Corr", fontsize=8, pad=8)
    cbar_edges.ax.tick_params(labelsize=7, pad=2)
    sm_nodes = plt.cm.ScalarMappable(cmap=cmap_nodes, norm=norm_nodes)
    sm_nodes.set_array([])
    cax_nodes = fig.add_axes(
        [cbar_x, ax_pos.y0 + 0.22 * ax_pos.height, cbar_width, 0.32 * ax_pos.height]
    )
    cbar_nodes = fig.colorbar(sm_nodes, cax=cax_nodes)
    cbar_nodes.ax.set_title("Feature-Target Corr", fontsize=8, pad=8)
    cbar_nodes.ax.tick_params(labelsize=7, pad=2)
    fig.suptitle("feature_corr_network_top20", fontsize=12, y=0.97)
    _plotting_predictions._savefig_with_pdf(
        fig,
        os.path.join(out_dir, "feature_corr_network_top20.png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    if _runtime.context.run_output_dir and prefix:
        _plotting_predictions._savefig_with_pdf(
            fig,
            os.path.join(
                _runtime.context.run_output_dir,
                f"{prefix}_feature_corr_network_top20.png",
            ),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)


def plot_attn_cls_by_category(
    model: nn.Module,
    test_loader: DataLoader,
    prefix: str,
    focus_bits: Optional[list] = None,
) -> None:
    out_dir = _runtime.get_run_output_dir(prefix)
    stats = _interpretation_attribution._collect_category_bit_attention_stats(
        model, test_loader
    )
    if not isinstance(stats, dict):
        return
    global_sum = dict(stats.get("global_sum", {}))
    global_cnt = dict(stats.get("global_cnt", {}))
    cat_sum = dict(stats.get("cat_sum", {}))
    cat_cnt = dict(stats.get("cat_cnt", {}))
    cat_n = dict(stats.get("cat_n", {}))
    heatmap_topn = int(
        max(5, int(_config._env_value("DECAT_RUNTIME_ATTN_CAT_HEATMAP_TOPN", None, 20)))
    )
    min_cat_samples = int(
        max(
            1,
            int(
                _config._env_value(
                    "DECAT_RUNTIME_ATTN_CAT_HEATMAP_MIN_CATEGORY", None, 8
                )
            ),
        )
    )
    min_cell_support = int(
        max(
            1,
            int(
                _config._env_value(
                    "DECAT_RUNTIME_ATTN_CAT_HEATMAP_MIN_SUPPORT", None, 3
                )
            ),
        )
    )
    row_norm_mode = (
        str(_config._env_value("DECAT_RUNTIME_ATTN_CAT_HEATMAP_ROW_NORM", None, "max"))
        .strip()
        .lower()
    )
    min_cross_categories = int(
        max(
            1,
            int(_config._env_value("DECAT_RUNTIME_ATTN_CAT_HEATMAP_MIN_CATS", None, 3)),
        )
    )
    if row_norm_mode not in {"max", "sum"}:
        row_norm_mode = "max"
    if focus_bits:
        top_bits = []
        seen = set()
        for name in [str(x) for x in focus_bits]:
            if name in seen:
                continue
            seen.add(name)
            top_bits.append(name)
            if len(top_bits) >= heatmap_topn:
                break
    else:
        top_bits = sorted(
            global_sum.keys(),
            key=lambda name: global_sum[name] / max(global_cnt.get(name, 1), 1),
            reverse=True,
        )[:heatmap_topn]
    original_top_bits = list(top_bits)
    global_ranked_bits = sorted(
        global_sum.keys(),
        key=lambda name: global_sum[name] / max(global_cnt.get(name, 1), 1),
        reverse=True,
    )
    supported_top_bits = []
    for bit_name in top_bits:
        support_cats = 0
        for cat_name in cat_n.keys():
            if int(cat_cnt.get(cat_name, {}).get(bit_name, 0)) >= min_cell_support:
                support_cats += 1
        if support_cats >= min_cross_categories:
            supported_top_bits.append(bit_name)
    top_bits = supported_top_bits[:heatmap_topn]
    if len(top_bits) < heatmap_topn:
        seen_bits = set(top_bits)
        for bit_name in list(original_top_bits) + list(global_ranked_bits):
            if bit_name in seen_bits:
                continue
            top_bits.append(bit_name)
            seen_bits.add(bit_name)
            if len(top_bits) >= heatmap_topn:
                break
    cat_order = sorted(
        [name for name in cat_n.keys() if int(cat_n.get(name, 0)) >= min_cat_samples],
        key=lambda name: (-int(cat_n[name]), str(name)),
    )
    if not top_bits or not cat_order:
        return
    mat = np.zeros((len(cat_order), len(top_bits)), dtype=np.float32)
    support_mat = np.zeros((len(cat_order), len(top_bits)), dtype=np.float32)
    for r, cat_name in enumerate(cat_order):
        for c, bit_name in enumerate(top_bits):
            s = float(cat_sum.get(cat_name, {}).get(bit_name, 0.0))
            n = int(cat_cnt.get(cat_name, {}).get(bit_name, 0))
            mat[r, c] = float(s / max(n, 1)) if n > 0 else 0.0
            support_mat[r, c] = float(n)
    if row_norm_mode == "sum":
        row_denom = np.sum(mat, axis=1, keepdims=True)
    else:
        row_denom = np.max(mat, axis=1, keepdims=True)
    row_denom = np.where(row_denom > 1e-12, row_denom, 1.0)
    mat_norm = mat / row_denom
    mask = support_mat < float(min_cell_support)
    mat_plot = mat_norm.copy()
    mat_plot[mask] = np.nan
    fig_w = max(8.4, min(13.5, 1.8 + 0.58 * len(top_bits)))
    fig_h = max(6.8, min(12.8, 1.8 + 0.72 * len(cat_order)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = mpl.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="#D1D5DB")
    im = ax.imshow(mat_plot, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized avg CLS attention", fontsize=11)
    ax.set_xticks(np.arange(len(top_bits)))
    ax.set_xticklabels(top_bits, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(cat_order)))
    ax.set_yticklabels(
        [f"{cat_name}\n(n={int(cat_n[cat_name])})" for cat_name in cat_order],
        fontsize=10,
    )
    ax.set_xlabel("Fingerprint bit", fontsize=12)
    ax.set_ylabel("Category", fontsize=12)
    title_topn = int(max(len(top_bits), min(heatmap_topn, len(original_top_bits))))
    ax.set_title(
        f"{prefix} - CLS Attention by Category (Consensus-focused Top-{title_topn})",
        fontsize=16,
        pad=14,
    )
    fig.tight_layout()
    _plotting_predictions._savefig_with_pdf(
        fig,
        os.path.join(out_dir, "attn_cls_by_category.png"),
        dpi=320,
        bbox_inches="tight",
    )
    if _runtime.context.run_output_dir and prefix:
        _plotting_predictions._savefig_with_pdf(
            fig,
            os.path.join(
                _runtime.context.run_output_dir, f"{prefix}_attn_cls_by_category.png"
            ),
            dpi=320,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_attn_family_by_category(
    model: nn.Module,
    test_loader: DataLoader,
    prefix: str,
    bit_to_family: Dict[str, str],
) -> None:
    out_dir = _runtime.get_run_output_dir(prefix)
    stats = _interpretation_attribution._collect_category_bit_attention_stats(
        model, test_loader
    )
    if (
        not isinstance(stats, dict)
        or not isinstance(bit_to_family, dict)
        or (not bit_to_family)
    ):
        return
    cat_sum = dict(stats.get("cat_sum", {}))
    cat_cnt = dict(stats.get("cat_cnt", {}))
    cat_n = dict(stats.get("cat_n", {}))
    min_cat_samples = int(
        max(
            1,
            int(
                _config._env_value(
                    "DECAT_RUNTIME_ATTN_CAT_HEATMAP_MIN_CATEGORY", None, 8
                )
            ),
        )
    )
    min_cell_support = int(
        max(
            1,
            int(
                _config._env_value(
                    "DECAT_RUNTIME_ATTN_CAT_HEATMAP_MIN_SUPPORT", None, 3
                )
            ),
        )
    )
    min_family_support = int(
        max(
            1,
            int(
                _config._env_value(
                    "DECAT_RUNTIME_ATTN_FAMILY_HEATMAP_MIN_SUPPORT", None, 2
                )
            ),
        )
    )
    min_family_cats = int(
        max(
            1,
            int(
                _config._env_value(
                    "DECAT_RUNTIME_ATTN_FAMILY_HEATMAP_MIN_CATS", None, 2
                )
            ),
        )
    )
    family_topn = int(
        max(
            4,
            int(_config._env_value("DECAT_RUNTIME_ATTN_FAMILY_HEATMAP_TOPN", None, 10)),
        )
    )
    row_norm_mode = (
        str(_config._env_value("DECAT_RUNTIME_ATTN_CAT_HEATMAP_ROW_NORM", None, "max"))
        .strip()
        .lower()
    )
    if row_norm_mode not in {"max", "sum"}:
        row_norm_mode = "max"
    family_total = Counter()
    family_support = Counter()
    family_cat_support = Counter()
    for cat_name, bit_map in cat_sum.items():
        cnt_map = cat_cnt.get(cat_name, {})
        seen_family_in_cat = set()
        for bit_name, val_sum in bit_map.items():
            fam = str(bit_to_family.get(str(bit_name), "")).strip()
            if not fam:
                continue
            n = int(cnt_map.get(bit_name, 0))
            if n < min_cell_support:
                continue
            family_total[fam] += float(val_sum / max(n, 1))
            family_support[fam] += 1
            seen_family_in_cat.add(fam)
        for fam in seen_family_in_cat:
            family_cat_support[fam] += 1
    family_names = [
        fam
        for fam, _ in family_total.most_common()
        if int(family_support.get(fam, 0)) >= min_family_support
        and int(family_cat_support.get(fam, 0)) >= min_family_cats
    ][:family_topn]
    cat_order = sorted(
        [name for name in cat_n.keys() if int(cat_n.get(name, 0)) >= min_cat_samples],
        key=lambda name: (-int(cat_n[name]), str(name)),
    )
    if not family_names or not cat_order:
        return
    mat = np.zeros((len(cat_order), len(family_names)), dtype=np.float32)
    support_mat = np.zeros((len(cat_order), len(family_names)), dtype=np.float32)
    for r, cat_name in enumerate(cat_order):
        cnt_map = cat_cnt.get(cat_name, {})
        for c, fam in enumerate(family_names):
            fam_vals = []
            fam_support = 0
            for bit_name, family_name in bit_to_family.items():
                if str(family_name).strip() != fam:
                    continue
                n = int(cnt_map.get(str(bit_name), 0))
                if n < min_cell_support:
                    continue
                s = float(cat_sum.get(cat_name, {}).get(str(bit_name), 0.0))
                fam_vals.append(float(s / max(n, 1)))
                fam_support += 1
            if fam_vals:
                mat[r, c] = float(np.mean(fam_vals))
                support_mat[r, c] = float(fam_support)
    if row_norm_mode == "sum":
        row_denom = np.sum(mat, axis=1, keepdims=True)
    else:
        row_denom = np.max(mat, axis=1, keepdims=True)
    row_denom = np.where(row_denom > 1e-12, row_denom, 1.0)
    mat_norm = mat / row_denom
    mat_norm[support_mat < 1.0] = np.nan
    fig_w = max(7.6, min(12.8, 2.2 + 0.75 * len(family_names)))
    fig_h = max(6.2, min(11.5, 1.8 + 0.72 * len(cat_order)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = mpl.cm.get_cmap("magma").copy()
    cmap.set_bad(color="#E5E7EB")
    im = ax.imshow(mat_norm, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized avg family attention", fontsize=11)
    ax.set_xticks(np.arange(len(family_names)))
    ax.set_xticklabels(family_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(cat_order)))
    ax.set_yticklabels(
        [f"{cat_name}\n(n={int(cat_n[cat_name])})" for cat_name in cat_order],
        fontsize=10,
    )
    ax.set_xlabel("Motif family", fontsize=12)
    ax.set_ylabel("Category", fontsize=12)
    ax.set_title(f"{prefix} - Motif Family Attention by Category", fontsize=16, pad=14)
    fig.tight_layout()
    _plotting_predictions._savefig_with_pdf(
        fig,
        os.path.join(out_dir, "attn_family_by_category.png"),
        dpi=320,
        bbox_inches="tight",
    )
    if _runtime.context.run_output_dir and prefix:
        _plotting_predictions._savefig_with_pdf(
            fig,
            os.path.join(
                _runtime.context.run_output_dir, f"{prefix}_attn_family_by_category.png"
            ),
            dpi=320,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_attention_cls_and_category(
    model: nn.Module, test_loader: DataLoader, prefix: str
):
    out_dir = _runtime.get_run_output_dir(prefix)
    attn_model, _ = _interpretation_attribution._capture_attention_once(
        model, test_loader
    )
    if attn_model is None or not hasattr(attn_model, "attn_cache"):
        return (np.array([], dtype=int), [], {}, np.array([], dtype=int))
    try:
        self_list = attn_model.attn_cache.get("self", [])
        if not self_list:
            return (np.array([], dtype=int), [], {}, np.array([], dtype=int))
        attn = self_list[-1].mean(dim=1).mean(dim=0).detach().cpu().numpy()
        has_cls = isinstance(
            attn_model, _models_attention.ChemistryAwareAttentionTransformer
        )
        if has_cls and attn.shape[0] > 1:
            cls_row = np.asarray(attn[0, 1:], dtype=float)
        else:
            cls_row = np.asarray(attn.mean(axis=0), dtype=float)
        rank_idx = (
            np.argsort(cls_row)[::-1] if cls_row.size > 0 else np.array([], dtype=int)
        )
        topn = int(
            max(
                5,
                min(
                    20,
                    int(_config._env_value("DECAT_RUNTIME_ATTN_TOPN", None, 20)),
                    rank_idx.size if rank_idx.size else 5,
                ),
            )
        )
        top_idx = rank_idx[:topn]
        fp_labels = _data_dataset._get_fp_feature_labels(test_loader.dataset)
        token_labels = _interpretation_attribution._token_labels_from_cache(
            attn_model, cls_row.size, fp_labels
        )
        bit_map = {int(i): token_labels[i] for i in range(len(token_labels))}
        if top_idx.size > 0:
            vals = np.asarray(cls_row[top_idx], dtype=float)
            labels = [bit_map.get(int(i), f"token_{int(i)}") for i in top_idx]
            fig_h = max(6.4, min(10.8, 1.6 + top_idx.size * 0.38))
            fig, ax = plt.subplots(figsize=(6.0, fig_h))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("#FCFCFD")
            nature_fill = "#B9DCFF"
            nature_edge = "#6FA8DC"
            y_pos = np.arange(top_idx.size)
            ax.barh(
                y_pos, vals, color=nature_fill, edgecolor=nature_edge, linewidth=0.9
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("CLS attention", fontsize=11)
            ax.set_ylabel("Fingerprint token", fontsize=11)
            ax.set_title(f"Top-{top_idx.size} attention tokens", fontsize=15, pad=10)
            ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.35)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            max_label_len = max((len(str(t)) for t in labels), default=8)
            left_margin = min(0.44, max(0.24, 0.16 + max_label_len * 0.006))
            fig.subplots_adjust(left=left_margin, right=0.88)
            x_max = float(np.max(vals))
            ax.set_xlim(0.0, x_max * 1.16 if x_max > 0 else 1.0)
            if top_idx.size <= 20:
                for y, v in zip(y_pos, vals):
                    ax.text(
                        v + max(0.0015, x_max * 0.008),
                        y,
                        f"{v:.3f}",
                        va="center",
                        ha="left",
                        fontsize=8,
                        color="#374151",
                    )
            fig.tight_layout()
            _plotting_predictions._savefig_with_pdf(
                fig,
                os.path.join(out_dir, "attention_top_tokens.png"),
                dpi=320,
                bbox_inches="tight",
            )
            plt.close(fig)
        feat_names = token_labels
        return (top_idx.astype(int), feat_names, bit_map, rank_idx.astype(int))
    except Exception:
        return (np.array([], dtype=int), [], {}, np.array([], dtype=int))


def export_training_diagnostics(
    output_dir: Optional[str],
    model: nn.Module,
    test_loader: DataLoader,
    epoch_history: list,
    top_state_records: list,
    sampler_info: Optional[Dict[str, object]],
    fusion_result: Optional[Dict[str, object]],
) -> None:
    if not output_dir:
        return
    try:
        if sampler_info:
            plot_sampler_diagnostics(output_dir, sampler_info)
        if fusion_result:
            plot_fusion_diagnostics(output_dir, fusion_result)
        plot_chemistry_bias(output_dir, model, test_loader)
        if epoch_history:
            hist_csv = os.path.join(output_dir, "training_history.csv")
            pd.DataFrame(epoch_history).to_csv(hist_csv, index=False)
            try:
                df = pd.DataFrame(epoch_history)
                plt.figure(figsize=(10, 6))
                plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
                if "val_r2" in df.columns:
                    plt.plot(df["epoch"], df["val_r2"], label="Val R2")
                plt.xlabel("Epoch")
                plt.ylabel("Metric")
                plt.title("Training History")
                plt.legend()
                plt.grid(True, alpha=0.3)
                _plotting_predictions._savefig_with_pdf(
                    plt, os.path.join(output_dir, "training_history_plot.png")
                )
                plt.close()
            except Exception:
                pass
    except Exception as e:
        print(f"[DECAT] Warning: Failed to save innovation artifacts: {e}")


def plot_sampler_diagnostics(output_dir: str, info: Dict[str, object]):
    try:
        import matplotlib.pyplot as plt

        history = info.get("history", [])
        if not history:
            return
        epochs = [h["epoch"] for h in history]
        probs = np.array([h["probs"] for h in history])
        n_cat = probs.shape[1]
        plt.figure(figsize=(10, 6))
        for i in range(n_cat):
            plt.plot(epochs, probs[:, i], label=f"Cat {i}")
        plt.xlabel("Epoch")
        plt.ylabel("Sampling Probability")
        plt.title("Category Sampling Probability Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        _plotting_predictions._savefig_with_pdf(
            plt, os.path.join(output_dir, "sampler_evolution.png")
        )
        plt.close()
    except Exception:
        pass


def plot_fusion_diagnostics(output_dir: str, result: Dict[str, object]):
    try:
        import matplotlib.pyplot as plt

        global_w = result.get("global_weights", [])
        models = list(result.get("component_names", [])) or ["NN", "HGB", "RF", "ET"]
        if len(global_w) == len(models):
            plt.figure(figsize=(8, 6))
            plt.bar(models, global_w)
            plt.ylabel("Weight")
            plt.title("Global Fusion Weights")
            for i, v in enumerate(global_w):
                plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
            _plotting_predictions._savefig_with_pdf(
                plt, os.path.join(output_dir, "fusion_weights.png")
            )
            plt.close()
    except Exception:
        pass


def plot_chemistry_bias(output_dir: str, model: nn.Module, loader: DataLoader):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            attn = model.encoder.layers[0].self_attn
            if (
                hasattr(attn, "chemistry_aware_bias")
                and attn.chemistry_aware_bias is not None
            ):
                bias = attn.chemistry_aware_bias.detach().cpu().numpy()
                plt.figure(figsize=(10, 8))
                sns.heatmap(bias, cmap="RdBu_r", center=0, annot=True, fmt=".2f")
                plt.title("Chemistry-Aware Attention Bias (Layer 0)")
                plt.xlabel("Key Category")
                plt.ylabel("Query Category")
                _plotting_predictions._savefig_with_pdf(
                    plt, os.path.join(output_dir, "chem_attn_bias.png")
                )
                plt.close()
    except Exception:
        pass
