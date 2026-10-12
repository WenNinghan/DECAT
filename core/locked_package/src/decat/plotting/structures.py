"""Functional-group and molecular motif illustrations."""

from __future__ import annotations
from rdkit.Chem import Draw
import os
from typing import Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import decat.interpretation.motifs as _interpretation_motifs
import decat.plotting.predictions as _plotting_predictions


def _plot_top_functional_groups(
    functional_rows: list, output_dir: str, artifact_tag: str = "attn", top_n: int = 10
) -> None:
    rows = list(functional_rows[: int(max(1, top_n))])
    if not rows:
        return
    labels = []
    score_vals = []
    delta_vals = []
    support_vals = []
    bit_counts = []
    for row in rows:
        fg_name = str(row.get("functional_group", "")).strip() or "Unassigned"
        fg_display = str(row.get("functional_display", "")).strip()
        labels.append(f"{fg_name} ({fg_display})" if fg_display else fg_name)
        score_vals.append(
            float(row.get("functional_score_sum", row.get("consensus_score_sum", 0.0)))
        )
        delta_vals.append(float(row.get("delta_logk", 0.0)))
        support_vals.append(int(row.get("support_molecules", 0)))
        bit_counts.append(int(row.get("bit_count", 0)))
    y_pos = np.arange(len(rows))
    fig_h = max(7.0, min(12.5, 2.0 + 0.6 * len(rows)))
    fig, (ax_score, ax_delta) = plt.subplots(
        ncols=2,
        figsize=(13.8, fig_h),
        sharey=True,
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.08},
    )
    fig.patch.set_facecolor("white")
    score_arr = np.asarray(score_vals, dtype=np.float32)
    delta_arr = np.asarray(delta_vals, dtype=np.float32)
    score_norm = mpl.colors.Normalize(
        vmin=float(score_arr.min()), vmax=float(score_arr.max() + 1e-12)
    )
    score_colors = [mpl.cm.get_cmap("YlOrRd")(score_norm(v)) for v in score_arr]
    delta_lim = float(max(np.max(np.abs(delta_arr)), 1e-06))
    delta_norm = mpl.colors.TwoSlopeNorm(vmin=-delta_lim, vcenter=0.0, vmax=delta_lim)
    delta_colors = [mpl.cm.get_cmap("coolwarm")(delta_norm(v)) for v in delta_arr]
    ax_score.set_facecolor("white")
    ax_delta.set_facecolor("white")
    ax_score.barh(
        y_pos, score_arr, color=score_colors, edgecolor="#7C2D12", linewidth=0.7
    )
    ax_score.set_yticks(y_pos)
    ax_score.set_yticklabels(labels, fontsize=10)
    ax_score.invert_yaxis()
    ax_score.set_xlabel("Functional-group evidence score")
    ax_score.set_title(f"Top-{len(rows)} functional groups", fontsize=15, pad=10)
    ax_score.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.2)
    ax_score.set_axisbelow(True)
    ax_score.spines["top"].set_visible(False)
    ax_score.spines["right"].set_visible(False)
    x_max = float(max(score_vals)) if score_vals else 1.0
    ax_score.set_xlim(0.0, x_max * 1.24 if x_max > 0 else 1.0)
    ax_delta.axvline(0.0, color="#6B7280", linewidth=1.0, alpha=0.9)
    ax_delta.barh(
        y_pos, delta_arr, color=delta_colors, edgecolor="#374151", linewidth=0.7
    )
    ax_delta.set_xlabel("Δlogk (active − inactive)")
    ax_delta.set_title("Direction of association", fontsize=15, pad=10)
    ax_delta.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.2)
    ax_delta.set_axisbelow(True)
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)
    ax_delta.spines["left"].set_visible(False)
    ax_delta.tick_params(axis="y", left=False, labelleft=False)
    ax_delta.set_xlim(-delta_lim * 1.28, delta_lim * 1.28)
    for y, score_v, delta_v, support_n, bit_n, row in zip(
        y_pos, score_arr, delta_arr, support_vals, bit_counts, rows
    ):
        ax_score.text(
            score_v + max(0.004, x_max * 0.012),
            y,
            f"support={support_n} | bits={bit_n}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#4B5563",
        )
        arrow = "↑" if delta_v > 0 else "↓" if delta_v < 0 else "→"
        ax_delta.text(
            delta_v + (0.02 * delta_lim if delta_v >= 0 else -0.02 * delta_lim),
            y,
            f"{arrow} {delta_v:+.3f}",
            va="center",
            ha="left" if delta_v >= 0 else "right",
            fontsize=8.8,
            color="#374151",
            fontweight="bold",
        )
        smarts = str(row.get("smarts", "")).strip()
        if smarts:
            ax_delta.text(
                ax_delta.get_xlim()[0]
                + 0.02 * (ax_delta.get_xlim()[1] - ax_delta.get_xlim()[0]),
                y + 0.29,
                smarts,
                va="center",
                ha="left",
                fontsize=7.5,
                color="#6B7280",
            )
    fig.suptitle(
        "Functional-group level attribution for logk",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    file_top_n = int(max(1, top_n))
    _plotting_predictions._savefig_with_pdf(
        fig,
        os.path.join(
            output_dir, f"{artifact_tag}_top{file_top_n}_functional_groups.png"
        ),
        dpi=340,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_top_functional_group_structures(
    output_dir: str, top_n: int = 10
) -> Optional[str]:
    rows = _interpretation_motifs._collect_functional_group_structure_rows(
        output_dir=output_dir, top_n=top_n
    )
    rows = _interpretation_motifs._supplement_functional_group_rows(
        rows, output_dir=output_dir, top_n=top_n
    )
    if not rows:
        return None
    n = len(rows)
    n_cols = 5
    n_rows = int(max(2, np.ceil(n / float(n_cols))))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(27.5, max(9.8, 4.9 * n_rows)))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    fig.patch.set_facecolor("white")
    for ax in axes.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_edgecolor("#E2E8F0")
    for idx, row in enumerate(rows):
        ax = axes.reshape(-1)[idx]
        fg_name = str(row.get("functional_group", "")).strip() or "Unknown"
        fg_display = str(row.get("functional_display", "")).strip()
        smarts = str(row.get("smarts", "")).strip()
        mol, highlight_atoms, shown_smi = (
            _interpretation_motifs._pick_functional_group_structure(
                functional_group=fg_name,
                example_substructures=str(row.get("example_substructures", "")),
                smarts=smarts,
            )
        )
        if mol is not None:
            try:
                img = Draw.MolToImage(
                    mol,
                    size=(900, 520),
                    highlightAtoms=list(highlight_atoms or []),
                    fitImage=True,
                )
                ax.imshow(np.asarray(img))
            except Exception:
                ax.text(
                    0.5,
                    0.55,
                    fg_name,
                    ha="center",
                    va="center",
                    fontsize=18,
                    transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5,
                0.55,
                fg_name,
                ha="center",
                va="center",
                fontsize=18,
                transform=ax.transAxes,
            )
        delta = float(row.get("delta_logk", 0.0))
        support = int(row.get("support_molecules", 0))
        bit_count = int(row.get("bit_count", 0))
        source = str(row.get("source", "")).strip()
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        delta_color = "#C0392B" if delta > 0 else "#2C5AA0" if delta < 0 else "#5B6472"
        title = f"{idx + 1}. {fg_name}"
        if fg_display:
            title += f" ({fg_display})"
        ax.text(
            0.02,
            0.98,
            title,
            ha="left",
            va="top",
            fontsize=13.0,
            fontweight="bold",
            color="#1F2937",
            transform=ax.transAxes,
        )
        ax.text(
            0.02,
            0.08,
            f"{arrow} Δlogk {delta:+.3f}",
            ha="left",
            va="bottom",
            fontsize=10.6,
            fontweight="bold",
            color=delta_color,
            transform=ax.transAxes,
        )
        ax.text(
            0.34,
            0.08,
            f"support={support} | bits={bit_count}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#475569",
            transform=ax.transAxes,
        )
        if shown_smi:
            ax.text(
                0.02,
                0.02,
                shown_smi,
                ha="left",
                va="bottom",
                fontsize=8.3,
                color="#64748B",
                transform=ax.transAxes,
            )
        if source:
            ax.text(
                0.98,
                0.02,
                source,
                ha="right",
                va="bottom",
                fontsize=8.3,
                color="#94A3B8",
                transform=ax.transAxes,
            )
    for idx in range(n, n_rows * n_cols):
        axes.reshape(-1)[idx].axis("off")
    fig.suptitle(
        "Top functional-group structures associated with logk",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.982])
    file_top_n = int(max(1, top_n))
    out_path = os.path.join(
        output_dir, f"consensus_top{file_top_n}_functional_group_structures.png"
    )
    _plotting_predictions._savefig_with_pdf(
        fig, out_path, dpi=420, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    try:
        pd.DataFrame(rows).to_csv(
            os.path.join(
                output_dir, f"consensus_top{file_top_n}_functional_group_structures.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )
    except Exception:
        pass
    return out_path


def _plot_consensus_top_motifs(
    consensus_rows: list, output_dir: str, top_n: int = 10
) -> None:
    rows = list(consensus_rows[: int(max(1, top_n))])
    if not rows:
        return
    labels = []
    vals = []
    fg_labels = []
    for row in rows:
        fg_name = str(row.get("functional_group", "")).strip()
        fg_display = str(row.get("functional_display", "")).strip()
        fg_labels.append(
            f"{fg_name} ({fg_display})"
            if fg_name and fg_display
            else fg_name if fg_name else "Unassigned"
        )
        label = str(row.get("bit_name", "")).strip()
        if not label:
            label = str(row.get("functional_group", "")).strip() or "motif"
        labels.append(label)
        vals.append(
            float(
                row.get(
                    "consensus_score",
                    row.get(
                        "consensus_score_sum", row.get("functional_score_sum", 0.0)
                    ),
                )
            )
        )
    fig_h = max(6.2, min(11.8, 1.8 + 0.5 * len(rows)))
    fig, ax = plt.subplots(figsize=(7.8, fig_h))
    cmap = mpl.cm.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=float(min(vals)), vmax=float(max(vals) + 1e-12))
    colors = [cmap(norm(v)) for v in vals]
    y_pos = np.arange(len(rows))
    ax.barh(y_pos, vals, color=colors, edgecolor="#1F2937", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{lab} | {fg}" for lab, fg in zip(labels, fg_labels)], fontsize=9
    )
    ax.invert_yaxis()
    ax.set_xlabel("Consensus motif score")
    ax.set_title(f"Top-{len(rows)} consensus motifs", fontsize=15, pad=10)
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x_max = float(max(vals)) if vals else 1.0
    ax.set_xlim(0.0, x_max * 1.18 if x_max > 0 else 1.0)
    for y, row in zip(y_pos, rows):
        score = float(
            row.get(
                "consensus_score",
                row.get("consensus_score_sum", row.get("functional_score_sum", 0.0)),
            )
        )
        if "bit_name" in row:
            txt = f"C={score:.3f} | A={float(row.get('attention_score', 0.0)):.3f} | IG={float(row.get('ig_score', 0.0)):.3f} | Δ={float(row.get('perturb_score', 0.0)):.3f} | Stab={float(row.get('stability_score', 0.0)):.2f}"
        else:
            txt = f"Score={score:.3f} | Δlogk={float(row.get('delta_logk', 0.0)):+.3f} | support={int(row.get('support_molecules', 0))} | bits={int(row.get('bit_count', 0))}"
        ax.text(
            score + max(0.003, x_max * 0.01),
            y,
            txt,
            va="center",
            ha="left",
            fontsize=8,
            color="#374151",
        )
    fig.tight_layout()
    _plotting_predictions._savefig_with_pdf(
        fig,
        os.path.join(output_dir, "consensus_top_motifs.png"),
        dpi=320,
        bbox_inches="tight",
    )
    plt.close(fig)
