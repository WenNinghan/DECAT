"""Prediction, residual and category diagnostic plots."""

from __future__ import annotations
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import decat.chemistry as _chemistry
import decat.config as _config
import decat.evaluation as _evaluation
import decat.runtime as _runtime

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


def cleanup_attention_output_images(output_dir: str) -> None:
    if not output_dir or not os.path.isdir(output_dir):
        return
    keep_prefixes = (
        "pred_vs_true",
        "residual_hist",
        "train_test_vs_true_band",
        "train_density_test_overlay",
        "category_metrics",
        "attn_cls_by_category",
        "attn_family_by_category",
        "attention_top_tokens",
        "feature_corr_network_top20",
        "attn_bit_substructures",
        "attn_top",
        "consensus_",
        "fusion_component_metrics",
        "model_performance_reference_style",
        "decat_",
    )
    keep_exts = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() not in keep_exts:
            continue
        if any((stem.startswith(pref) for pref in keep_prefixes)):
            continue
        try:
            os.remove(path)
        except Exception:
            pass


def _bin_matrix(mat: np.ndarray, max_rows: int, max_cols: int):
    mat = np.asarray(mat, dtype=float)
    n_rows, n_cols = mat.shape
    row_bins = [np.array([i]) for i in range(n_rows)]
    col_bins = [np.array([j]) for j in range(n_cols)]
    if n_rows > max_rows:
        row_bins = np.array_split(np.arange(n_rows), max_rows)
        mat = np.vstack([mat[idx].mean(axis=0) for idx in row_bins])
    if n_cols > max_cols:
        col_bins = np.array_split(np.arange(n_cols), max_cols)
        mat = np.column_stack([mat[:, idx].mean(axis=1) for idx in col_bins])
    return (mat, row_bins, col_bins)


def _make_bin_labels(bins):
    labels = []
    for idx in bins:
        start = int(idx[0]) + 1
        end = int(idx[-1]) + 1
        labels.append(f"{start}" if start == end else f"{start}-{end}")
    return labels


def plot_pred_vs_true_and_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str
) -> None:
    out_dir = _runtime.get_run_output_dir(prefix)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 3:
        return
    x = y_true.copy()
    y = y_pred.copy()
    err = np.abs(y - x)
    if _evaluation.gaussian_kde is not None and x.size > 5:
        try:
            xy = np.vstack([x, y])
            density = _evaluation.gaussian_kde(xy)(xy)
        except Exception:
            density = np.ones_like(x, dtype=float)
    else:
        density = np.ones_like(x, dtype=float)
    order = np.argsort(density)
    x = x[order]
    y = y[order]
    err = err[order]
    density = density[order]
    cmax = float(np.nanpercentile(err, 98)) if err.size > 0 else 1.0
    if not np.isfinite(cmax) or cmax <= 1e-12:
        cmax = float(np.nanmax(err) + 1e-09) if err.size > 0 else 1.0
    norm = mpl.colors.Normalize(vmin=0.0, vmax=cmax)
    cmap = mpl.cm.get_cmap("rainbow_r")
    rgba = cmap(norm(err))
    dmin = float(np.nanmin(density)) if density.size > 0 else 0.0
    dmax = float(np.nanmax(density)) if density.size > 0 else 1.0
    dscale = (density - dmin) / max(dmax - dmin, 1e-12)
    whiten = (1.0 - dscale) * 0.72
    rgba[:, :3] = rgba[:, :3] * (1.0 - whiten[:, None]) + 1.0 * whiten[:, None]
    rgba[:, 3] = 0.92
    all_min = float(min(np.min(x), np.min(y)))
    all_max = float(max(np.max(x), np.max(y)))
    margin = 0.03 * (all_max - all_min) if all_max > all_min else 1.0
    lo = all_min - margin
    hi = all_max + margin
    bins = int(max(18, min(40, np.sqrt(x.size) * 2.2)))
    edges = np.linspace(lo, hi, bins + 1)
    bin_w = float(edges[1] - edges[0])
    grid = np.linspace(lo, hi, 500)
    fig = plt.figure(figsize=(8.8, 6.9), dpi=130)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[4.6, 1.3, 0.24],
        height_ratios=[1.15, 4.1],
        wspace=0.06,
        hspace=0.06,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cbar = fig.add_subplot(gs[:, 2])
    ax_blank = fig.add_subplot(gs[0, 1])
    ax_blank.axis("off")
    ax_main.scatter(x, y, s=26, c=rgba, edgecolors="none")
    ax_main.plot([lo, hi], [lo, hi], "--", color="red", linewidth=1.4, alpha=0.9)
    ax_main.set_xlim(lo, hi)
    ax_main.set_ylim(lo, hi)
    ax_main.set_xlabel("True logk")
    ax_main.set_ylabel("Predicted logk")
    ax_main.grid(alpha=0.2)
    ax_top.hist(y_true, bins=edges, color="#6BAED6", alpha=0.45, label="True")
    ax_top.hist(y_pred, bins=edges, color="#FDB37C", alpha=0.45, label="Pred")
    if _evaluation.gaussian_kde is not None and y_true.size > 5:
        try:
            ax_top.plot(
                grid,
                _evaluation.gaussian_kde(y_true)(grid) * y_true.size * bin_w,
                color="#1F77B4",
                linewidth=2.0,
            )
            ax_top.plot(
                grid,
                _evaluation.gaussian_kde(y_pred)(grid) * y_pred.size * bin_w,
                color="#FF7F0E",
                linewidth=2.0,
            )
        except Exception:
            pass
    ax_top.set_ylabel("Frequency")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.legend(loc="upper right", fontsize=9, frameon=True)
    ax_top.grid(alpha=0.15)
    ax_right.hist(
        y_true, bins=edges, orientation="horizontal", color="#6BAED6", alpha=0.45
    )
    ax_right.hist(
        y_pred, bins=edges, orientation="horizontal", color="#FDB37C", alpha=0.45
    )
    if _evaluation.gaussian_kde is not None and y_true.size > 5:
        try:
            ax_right.plot(
                _evaluation.gaussian_kde(y_true)(grid) * y_true.size * bin_w,
                grid,
                color="#1F77B4",
                linewidth=2.0,
            )
            ax_right.plot(
                _evaluation.gaussian_kde(y_pred)(grid) * y_pred.size * bin_w,
                grid,
                color="#FF7F0E",
                linewidth=2.0,
            )
        except Exception:
            pass
    ax_right.set_xlabel("Frequency")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.grid(alpha=0.15)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("Pred-True to y=x (hue); density controls lightness")
    fig.tight_layout()
    _savefig_with_pdf(
        fig, os.path.join(out_dir, "pred_vs_true.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    residual = y_pred - y_true
    plt.figure(figsize=(6.0, 4.0))
    plt.hist(residual, bins=30, color="#0F766E", alpha=0.85)
    plt.xlabel("Residual (Pred - True)")
    plt.ylabel("Count")
    plt.title(f"{prefix} - Residual Histogram")
    plt.tight_layout()
    _savefig_with_pdf(
        plt, os.path.join(out_dir, "residual_hist.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_train_test_vs_true_with_band(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    prefix: str,
) -> None:
    out_dir = _runtime.get_run_output_dir(prefix)
    y_train_true = np.asarray(y_train_true, dtype=float).reshape(-1)
    y_train_pred = np.asarray(y_train_pred, dtype=float).reshape(-1)
    y_test_true = np.asarray(y_test_true, dtype=float).reshape(-1)
    y_test_pred = np.asarray(y_test_pred, dtype=float).reshape(-1)
    tr_mask = np.isfinite(y_train_true) & np.isfinite(y_train_pred)
    te_mask = np.isfinite(y_test_true) & np.isfinite(y_test_pred)
    y_train_true = y_train_true[tr_mask]
    y_train_pred = y_train_pred[tr_mask]
    y_test_true = y_test_true[te_mask]
    y_test_pred = y_test_pred[te_mask]
    if y_train_true.size < 2 or y_test_true.size < 2:
        return
    r2_train = float(r2_score(y_train_true, y_train_pred))
    rmse_train = float(np.sqrt(mean_squared_error(y_train_true, y_train_pred)))
    r2_test = float(r2_score(y_test_true, y_test_pred))
    rmse_test = float(np.sqrt(mean_squared_error(y_test_true, y_test_pred)))
    band = (
        float(np.quantile(np.abs(y_test_pred - y_test_true), 0.8))
        if y_test_true.size > 0
        else 0.0
    )
    all_true = np.concatenate([y_train_true, y_test_true])
    all_pred = np.concatenate([y_train_pred, y_test_pred])
    min_v = float(min(all_true.min(), all_pred.min()))
    max_v = float(max(all_true.max(), all_pred.max()))
    margin = 0.05 * (max_v - min_v) if max_v > min_v else 1.0
    x_line = np.linspace(min_v - margin, max_v + margin, 200)
    y_line = x_line
    plt.figure(figsize=(7.5, 6.0))
    if band > 0:
        plt.fill_between(
            x_line,
            y_line - band,
            y_line + band,
            color="#F59E0B",
            alpha=0.16,
            label="~80% test band",
        )
    plt.plot(x_line, y_line, "--", color="#64748B", linewidth=1.2, label="y=x")
    plt.scatter(
        y_train_true, y_train_pred, c="#1D4ED8", s=20, alpha=0.65, label="Train"
    )
    plt.scatter(
        y_test_true, y_test_pred, c="#DC2626", s=24, alpha=0.8, marker="^", label="Test"
    )
    plt.xlabel("True logk")
    plt.ylabel("Predicted logk")
    plt.title(f"{prefix} - Train/Test Predicted vs True")
    plt.legend(
        title=f"Train R2={r2_train:.3f}, RMSE={rmse_train:.3f}\nTest  R2={r2_test:.3f}, RMSE={rmse_test:.3f}",
        loc="lower right",
        fontsize=9,
    )
    plt.tight_layout()
    _savefig_with_pdf(
        plt,
        os.path.join(out_dir, "train_test_vs_true_band.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if _runtime.context.run_output_dir and prefix:
        _savefig_with_pdf(
            plt,
            os.path.join(
                _runtime.context.run_output_dir, f"{prefix}_train_test_vs_true_band.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()


def _rgba_by_error_density(
    err: np.ndarray,
    density: np.ndarray,
    cmap_name: str = "turbo",
    lighten_max: float = 0.5,
    alpha_min: float = 0.3,
    alpha_max: float = 0.95,
):
    err = np.asarray(err, dtype=float).reshape(-1)
    density = np.asarray(density, dtype=float).reshape(-1)
    if err.size == 0:
        return (
            np.zeros((0, 4), dtype=float),
            mpl.colors.Normalize(vmin=0.0, vmax=1.0),
            mpl.cm.get_cmap(cmap_name),
        )
    cmax = float(np.nanpercentile(err, 98)) if err.size > 0 else 1.0
    if not np.isfinite(cmax) or cmax <= 1e-12:
        cmax = float(np.nanmax(err) + 1e-09) if err.size > 0 else 1.0
    norm = mpl.colors.Normalize(vmin=0.0, vmax=cmax)
    cmap = mpl.cm.get_cmap(cmap_name)
    rgba = cmap(norm(err))
    if density.size != err.size:
        density = np.ones_like(err, dtype=float)
    dmin = float(np.nanmin(density)) if density.size > 0 else 0.0
    dmax = float(np.nanmax(density)) if density.size > 0 else 1.0
    dscale = (density - dmin) / max(dmax - dmin, 1e-12)
    whiten = (1.0 - dscale) * float(lighten_max)
    rgba[:, :3] = rgba[:, :3] * (1.0 - whiten[:, None]) + 1.0 * whiten[:, None]
    rgba[:, 3] = float(alpha_min) + dscale * float(max(alpha_max - alpha_min, 1e-06))
    rgba[:, 3] = np.clip(rgba[:, 3], 0.0, 1.0)
    return (rgba, norm, cmap)


def plot_train_density_test_overlay(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    prefix: str,
) -> None:
    out_dir = _runtime.get_run_output_dir(prefix)
    y_train_true = np.asarray(y_train_true, dtype=float).reshape(-1)
    y_train_pred = np.asarray(y_train_pred, dtype=float).reshape(-1)
    y_test_true = np.asarray(y_test_true, dtype=float).reshape(-1)
    y_test_pred = np.asarray(y_test_pred, dtype=float).reshape(-1)
    tr_mask = np.isfinite(y_train_true) & np.isfinite(y_train_pred)
    te_mask = np.isfinite(y_test_true) & np.isfinite(y_test_pred)
    y_train_true = y_train_true[tr_mask]
    y_train_pred = y_train_pred[tr_mask]
    y_test_true = y_test_true[te_mask]
    y_test_pred = y_test_pred[te_mask]
    if y_train_true.size < 10 or y_test_true.size < 5:
        return
    err_test = np.abs(y_test_pred - y_test_true)
    if _evaluation.gaussian_kde is not None and y_test_true.size > 8:
        try:
            xy_test = np.vstack([y_test_true, y_test_pred])
            density_test = _evaluation.gaussian_kde(xy_test)(xy_test)
        except Exception:
            density_test = np.ones_like(y_test_true, dtype=float)
    else:
        density_test = np.ones_like(y_test_true, dtype=float)
    order = np.argsort(density_test)
    y_test_true = y_test_true[order]
    y_test_pred = y_test_pred[order]
    err_test = err_test[order]
    density_test = density_test[order]
    rgba_test, norm, cmap = _rgba_by_error_density(err_test, density_test)
    all_vals = np.concatenate(
        [y_train_true, y_train_pred, y_test_true, y_test_pred]
    ).astype(float)
    all_vals = all_vals[np.isfinite(all_vals)]
    lo = float(np.min(all_vals))
    hi = float(np.max(all_vals))
    margin = 0.03 * (hi - lo) if hi > lo else 1.0
    lo -= margin
    hi += margin
    fig = plt.figure(figsize=(10.8, 7.8), dpi=140)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[5.8, 1.7, 0.34],
        height_ratios=[1.65, 5.2],
        wspace=0.08,
        hspace=0.07,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cbar = fig.add_subplot(gs[:, 2])
    ax_main.hexbin(
        y_train_true,
        y_train_pred,
        gridsize=34,
        cmap="Greys",
        mincnt=1,
        linewidths=0.0,
        alpha=0.78,
        zorder=1,
    )
    xline = np.linspace(lo, hi, 256)
    ax_main.fill_between(
        xline, xline - 0.5, xline + 0.5, color="#e5e5e5", alpha=0.3, zorder=0
    )
    ax_main.scatter(
        y_test_true,
        y_test_pred,
        s=28,
        c=rgba_test,
        edgecolors="white",
        linewidths=0.22,
        zorder=3,
    )
    ax_main.plot(
        [lo, hi], [lo, hi], "--", color="#d62728", linewidth=1.6, alpha=0.95, zorder=2
    )
    ax_main.set_xlim(lo, hi)
    ax_main.set_ylim(lo, hi)
    ax_main.set_xlabel("True logk")
    ax_main.set_ylabel("Predicted logk")
    ax_main.grid(alpha=0.18)
    try:

        def _metrics(t, p):
            if t.size < 2:
                return (float("nan"), float("nan"))
            return (float(r2_score(t, p)), float(np.sqrt(mean_squared_error(t, p))))

        train_r2, train_rmse = _runtime.resolve_hook("metrics", _evaluation._metrics)(
            y_train_true, y_train_pred
        )
        test_r2, test_rmse = _runtime.resolve_hook("metrics", _evaluation._metrics)(
            y_test_true, y_test_pred
        )
        train_mae = float(np.mean(np.abs(y_train_pred - y_train_true)))
        test_mae = float(np.mean(np.abs(y_test_pred - y_test_true)))
        text = f"Train: n={y_train_true.size}, R²={train_r2:.3f}, RMSE={train_rmse:.3f}, MAE={train_mae:.3f}\nTest: n={y_test_true.size}, R²={test_r2:.3f}, RMSE={test_rmse:.3f}, MAE={test_mae:.3f}"
        ax_main.text(
            0.03,
            0.97,
            text,
            transform=ax_main.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor="white",
                edgecolor="#666666",
                alpha=0.92,
            ),
        )
    except Exception:
        pass
    bins = int(max(18, min(40, np.sqrt(y_train_true.size + y_test_true.size) * 1.9)))
    grid = np.linspace(lo, hi, 256)
    bin_w = (hi - lo) / max(bins, 1)
    curves = [
        (y_train_true, "#4C78A8", "Train true", "--"),
        (y_train_pred, "#F58518", "Train pred", "--"),
        (y_test_true, "#4C78A8", "Test true", "-"),
        (y_test_pred, "#F58518", "Test pred", "-"),
    ]
    hist_specs = [
        (y_train_true, "#4C78A8", 0.16),
        (y_train_pred, "#F58518", 0.16),
        (y_test_true, "#4C78A8", 0.24),
        (y_test_pred, "#F58518", 0.24),
    ]
    for arr, color, alpha in hist_specs:
        ax_top.hist(arr, bins=bins, color=color, alpha=alpha, edgecolor="none")
    for arr, color, label, style in curves:
        if _evaluation.gaussian_kde is not None and arr.size > 5:
            try:
                curve = _evaluation.gaussian_kde(arr)(grid) * arr.size * bin_w
                ax_top.plot(
                    grid,
                    curve,
                    color=color,
                    linewidth=1.8,
                    linestyle=style,
                    label=label,
                )
                continue
            except Exception:
                pass
        ax_top.hist(
            arr,
            bins=bins,
            histtype="step",
            color=color,
            linewidth=1.4,
            linestyle=style,
            label=label,
        )
    ax_top.set_ylabel("Count")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(alpha=0.14)
    ax_top.legend(loc="upper right", fontsize=8.5, frameon=True)
    for arr, color, alpha in hist_specs:
        ax_right.hist(
            arr,
            bins=bins,
            orientation="horizontal",
            color=color,
            alpha=alpha,
            edgecolor="none",
        )
    for arr, color, _, style in curves:
        if _evaluation.gaussian_kde is not None and arr.size > 5:
            try:
                curve = _evaluation.gaussian_kde(arr)(grid) * arr.size * bin_w
                ax_right.plot(curve, grid, color=color, linewidth=1.8, linestyle=style)
                continue
            except Exception:
                pass
        ax_right.hist(
            arr,
            bins=bins,
            orientation="horizontal",
            histtype="step",
            color=color,
            linewidth=1.4,
            linestyle=style,
        )
    ax_right.set_xlabel("Count")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.grid(alpha=0.14)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label(
        "Absolute prediction error on the test set\n(color and opacity modulated by local test density)"
    )
    handles = [
        Patch(facecolor="#808080", alpha=0.55, label="Training-set density"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d62728",
            markeredgecolor="white",
            markersize=6.5,
            label="Test-set samples",
        ),
        Line2D(
            [0],
            [0],
            color="#d62728",
            linestyle="--",
            linewidth=1.6,
            label="Ideal agreement (y = x)",
        ),
    ]
    ax_main.legend(handles=handles, loc="lower right", fontsize=8.8, frameon=True)
    ax_top.set_title(
        "Training-distribution-aware predictive performance on the test set",
        fontsize=13,
        pad=8,
    )
    fig.tight_layout()
    _savefig_with_pdf(
        fig,
        os.path.join(out_dir, "train_density_test_overlay.png"),
        dpi=300,
        bbox_inches="tight",
    )
    _savefig_with_pdf(
        fig,
        os.path.join(out_dir, "train_test_vs_true_band.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if _runtime.context.run_output_dir and prefix:
        _savefig_with_pdf(
            fig,
            os.path.join(
                _runtime.context.run_output_dir,
                f"{prefix}_train_density_test_overlay.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
    _savefig_with_pdf(
        fig,
        os.path.join(
            _runtime.context.run_output_dir, f"{prefix}_train_test_vs_true_band.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_category_metrics_paper(
    y_true: np.ndarray, y_pred: np.ndarray, y_cat_labels: np.ndarray, prefix: str
) -> None:
    out_dir = _runtime.get_run_output_dir(prefix)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_cat_labels = np.asarray(y_cat_labels).reshape(-1)
    if y_true.size != y_pred.size or y_true.size != y_cat_labels.size:
        return
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y_cat_labels = y_cat_labels[mask]
    if y_true.size < 20:
        return
    min_n = int(
        max(4, int(_config._env_value("DECAT_RUNTIME_CATEGORY_METRICS_MIN_N", None, 5)))
    )
    rows = []
    for c in np.unique(y_cat_labels):
        m = y_cat_labels == c
        n = int(np.sum(m))
        if n < min_n:
            continue
        yt = y_true[m]
        yp = y_pred[m]
        try:
            r2 = float(r2_score(yt, yp)) if yt.size >= 2 else float("nan")
        except Exception:
            r2 = float("nan")
        try:
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        except Exception:
            rmse = float("nan")
        rows.append(
            {
                "category": _chemistry._pretty_category_label(c),
                "n": n,
                "r2": r2,
                "rmse": rmse,
            }
        )
    if not rows:
        return
    rows.sort(
        key=lambda row: (
            float(row.get("r2", float("-inf"))),
            -float(row.get("rmse", float("inf"))),
        ),
        reverse=True,
    )
    labels = [str(row["category"]) for row in rows]
    n_vals = np.asarray([int(row["n"]) for row in rows], dtype=int)
    r2_vals = np.asarray([float(row["r2"]) for row in rows], dtype=float)
    rmse_vals = np.asarray([float(row["rmse"]) for row in rows], dtype=float)
    y_pos = np.arange(len(rows), dtype=float)
    fig_w = max(13.4, 10.4 + 0.5 * len(rows))
    fig, (ax_r2, ax_rmse) = plt.subplots(
        1,
        2,
        figsize=(fig_w, 6.9),
        sharey=True,
        gridspec_kw={"width_ratios": [1.02, 1.0], "wspace": 0.14},
    )
    fig.patch.set_facecolor("white")
    for ax in (ax_r2, ax_rmse):
        ax.set_facecolor("white")
        ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D1D5DB")
        ax.spines["bottom"].set_color("#D1D5DB")
        for yi in y_pos:
            ax.axhspan(
                yi - 0.5,
                yi + 0.5,
                color="#F8FAFC" if int(yi) % 2 == 0 else "#FFFFFF",
                zorder=0,
            )
    r2_thr = float(_config._env_value("DECAT_RUNTIME_CATEGORY_R2_TARGET", None, 0.8))
    rmse_thr = float(
        _config._env_value("DECAT_RUNTIME_CATEGORY_RMSE_TARGET", None, 1.2)
    )
    mean_r2 = float(np.nanmean(r2_vals))
    mean_rmse = float(np.nanmean(rmse_vals))
    ax_r2.axvspan(
        r2_thr,
        max(1.0, float(np.nanmax(r2_vals) + 0.03)),
        color="#ECFDF5",
        alpha=0.9,
        zorder=0,
    )
    ax_r2.axvline(r2_thr, color="#059669", linestyle="--", linewidth=1.6, alpha=0.95)
    ax_rmse.axvspan(0.0, rmse_thr, color="#EFF6FF", alpha=0.9, zorder=0)
    ax_rmse.axvline(
        rmse_thr, color="#2563EB", linestyle="--", linewidth=1.6, alpha=0.95
    )
    ax_r2.axvline(mean_r2, color="#94A3B8", linestyle=":", linewidth=1.4, alpha=0.95)
    ax_rmse.axvline(
        mean_rmse, color="#94A3B8", linestyle=":", linewidth=1.4, alpha=0.95
    )
    r2_colors = ["#0F766E" if v >= r2_thr else "#94A3B8" for v in r2_vals]
    rmse_colors = ["#2563EB" if v <= rmse_thr else "#DC2626" for v in rmse_vals]
    size_scale = 22 + 1.6 * np.sqrt(np.clip(n_vals, 1, None))
    stem_r2_start = min(r2_thr - 0.12, float(np.nanmin(r2_vals) - 0.03))
    for y, v, color in zip(y_pos, r2_vals, r2_colors):
        ax_r2.hlines(
            y, xmin=stem_r2_start, xmax=v, color="#CBD5E1", linewidth=2.6, zorder=1
        )
        ax_r2.scatter(
            [v],
            [y],
            s=float(size_scale[int(y)] * 4.6),
            color=color,
            edgecolors="white",
            linewidths=1.1,
            zorder=3,
        )
    for y, v, color in zip(y_pos, rmse_vals, rmse_colors):
        ax_rmse.hlines(y, xmin=0.0, xmax=v, color="#E5E7EB", linewidth=2.6, zorder=1)
        ax_rmse.scatter(
            [v],
            [y],
            s=float(size_scale[int(y)] * 4.6),
            color=color,
            edgecolors="white",
            linewidths=1.1,
            zorder=3,
            marker="o",
        )
    ax_r2.set_yticks(y_pos)
    ax_r2.set_yticklabels(
        [f"{lab}  (n={n})" for lab, n in zip(labels, n_vals)],
        fontsize=10.8,
        fontweight="bold",
    )
    ax_r2.invert_yaxis()
    ax_r2.set_xlabel("$R^2$", fontsize=13, fontweight="bold")
    ax_r2.set_title("Category-wise $R^2$", fontsize=15.5, fontweight="bold", pad=14)
    ax_rmse.set_xlabel("RMSE", fontsize=13, fontweight="bold")
    ax_rmse.set_title("Category-wise RMSE", fontsize=15.5, fontweight="bold", pad=14)
    ax_rmse.tick_params(axis="y", left=False, labelleft=False)
    r2_xmin = min(r2_thr - 0.12, float(np.nanmin(r2_vals) - 0.04))
    r2_xmax = max(1.0, float(np.nanmax(r2_vals) + 0.06))
    ax_r2.set_xlim(r2_xmin, r2_xmax)
    ax_rmse.set_xlim(0.0, max(rmse_thr + 0.1, float(np.nanmax(rmse_vals) + 0.1)))
    for y, v, n in zip(y_pos, r2_vals, n_vals):
        ax_r2.text(
            v + 0.008,
            y,
            f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=9.4,
            color="#334155",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.9,
            ),
        )
    for y, v in zip(y_pos, rmse_vals):
        ha = "left" if v <= ax_rmse.get_xlim()[1] * 0.82 else "right"
        dx = 0.02 if ha == "left" else -0.02
        ax_rmse.text(
            v + dx,
            y,
            f"{v:.3f}",
            va="center",
            ha=ha,
            fontsize=9.4,
            color="#334155",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.9,
            ),
        )
    ax_r2.text(
        r2_thr,
        -0.92,
        f"target = {r2_thr:.2f}",
        ha="left",
        va="center",
        fontsize=9.2,
        color="#047857",
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="#ECFDF5",
            edgecolor="#A7F3D0",
            alpha=0.95,
        ),
    )
    ax_rmse.text(
        rmse_thr,
        -0.92,
        f"target = {rmse_thr:.2f}",
        ha="right",
        va="center",
        fontsize=9.2,
        color="#1D4ED8",
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="#EFF6FF",
            edgecolor="#BFDBFE",
            alpha=0.95,
        ),
    )
    ax_r2.text(
        mean_r2,
        len(rows) - 0.15,
        f"mean {mean_r2:.3f}",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#64748B",
    )
    ax_rmse.text(
        mean_rmse,
        len(rows) - 0.15,
        f"mean {mean_rmse:.3f}",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#64748B",
    )
    fig.suptitle(
        "Performance across chemical categories",
        fontsize=19,
        fontweight="bold",
        y=0.988,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.972])
    _savefig_with_pdf(
        fig,
        os.path.join(out_dir, "category_metrics_paper.png"),
        dpi=340,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _apply_plot_defaults_if_needed() -> None:
    _config._setdefault_env("DECAT_RUNTIME_PLOT_FUSION", "0")
    _config._setdefault_env("DECAT_RUNTIME_PLOT_FEATURE_CORR_NETWORK", "0")
    _config._setdefault_env("DECAT_RUNTIME_FEATURE_CORR_PAPER", "0")
    _config._setdefault_env("DECAT_RUNTIME_PLOT_FEATURE_CORR_NETWORK20", "1")
    _config._setdefault_env("DECAT_RUNTIME_PLOT_ATTN_CLS_BY_CATEGORY", "1")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_TOPN", "20")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_CAT_HEATMAP_TOPN", "20")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_CAT_HEATMAP_MIN_CATEGORY", "8")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_CAT_HEATMAP_MIN_SUPPORT", "3")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_CAT_HEATMAP_MIN_CATS", "3")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_CAT_HEATMAP_ROW_NORM", "max")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_FAMILY_HEATMAP_TOPN", "10")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_FAMILY_HEATMAP_MIN_SUPPORT", "2")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_FAMILY_HEATMAP_MIN_CATS", "2")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_MIN_RADIUS", "2")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_SUBSTRUCT_TOPK", "30")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_SUBSTRUCT_USE_TRAINVAL", "0")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_SUBSTRUCT_MIN_ATOMS", "2")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_SUBSTRUCT_UNIQUE_TOPK", "10")
    _config._setdefault_env("DECAT_RUNTIME_ATTN_SUBSTRUCT_UNIQUE_MIN_HEAVY", "4")
    _config._setdefault_env("DECAT_RUNTIME_CMA_ENABLE", "1")
    _config._setdefault_env("DECAT_RUNTIME_CMA_TOPK", "10")
    _config._setdefault_env("DECAT_RUNTIME_CMA_CANDIDATE_TOPK", "24")
    _config._setdefault_env("DECAT_RUNTIME_CMA_MIN_SUPPORT", "5")
    _config._setdefault_env("DECAT_RUNTIME_CMA_MAX_SAMPLES", "512")
    _config._setdefault_env("DECAT_RUNTIME_CMA_BATCH_SIZE", "32")
    _config._setdefault_env("DECAT_RUNTIME_CMA_IG_STEPS", "8")
    _config._setdefault_env("DECAT_RUNTIME_CMA_BOOTSTRAP_ROUNDS", "32")
    _config._setdefault_env("DECAT_RUNTIME_CMA_BOOTSTRAP_FRAC", "0.75")
    _config._setdefault_env("DECAT_RUNTIME_CMA_STABILITY_TRUE", "1")
    _config._setdefault_env("DECAT_RUNTIME_CMA_STABILITY_SEEDS", "11,19,23,29,37")
    _config._setdefault_env("DECAT_RUNTIME_CMA_STABILITY_FOLDS", "5")
    _config._setdefault_env("DECAT_RUNTIME_CMA_STABILITY_MAX_EPOCHS", "80")
    _config._setdefault_env("DECAT_RUNTIME_CMA_STABILITY_EARLY_STOP", "20")
    _config._setdefault_env("DECAT_RUNTIME_CMA_STABILITY_BATCH_SIZE", "32")
    _config._setdefault_env("DECAT_RUNTIME_CMA_STABILITY_MIN_DELTA", "1e-4")
    _config._setdefault_env("DECAT_RUNTIME_CMA_W_ATTENTION", "0.20")
    _config._setdefault_env("DECAT_RUNTIME_CMA_W_IG", "0.20")
    _config._setdefault_env("DECAT_RUNTIME_CMA_W_DELTA", "0.30")
    _config._setdefault_env("DECAT_RUNTIME_CMA_W_STABILITY", "0.20")
    _config._setdefault_env("DECAT_RUNTIME_CMA_W_SUPPORT", "0.10")
    _config._setdefault_env("DECAT_RUNTIME_TOP_FUNCTIONAL_GROUPS", "10")
    _config._setdefault_env("DECAT_RUNTIME_EXPORT_UNIQUE_ATOM_GROUPS", "0")
    _config._setdefault_env("DECAT_RUNTIME_PLOT_CATEGORY_METRICS", "1")
    _config._setdefault_env("DECAT_RUNTIME_KEEP_CORE_IMAGES_ONLY", "1")


def _savefig_with_pdf(plt_obj, path_png, **kwargs):
    try:
        save_kwargs = {"dpi": 300, "bbox_inches": "tight"}
        save_kwargs.update(kwargs)
        plt_obj.savefig(path_png, **save_kwargs)
        pdf_path = path_png.replace(".png", ".pdf")
        pdf_kwargs = dict(save_kwargs)
        pdf_kwargs.pop("dpi", None)
        plt_obj.savefig(pdf_path, format="pdf", **pdf_kwargs)
    except Exception:
        pass
