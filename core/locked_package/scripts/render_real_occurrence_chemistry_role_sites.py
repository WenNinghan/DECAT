#!/usr/bin/env python
"""Render locked Top-20 local environments with chemistry-role site annotations."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from render_real_occurrence_original_artifacts_exact import (
    ASSET_ROOT,
    DEFAULT_CSV,
    load_recovered_original_functions,
)


OUTPUT_FOLDER = "key_sites_chemistry_role_bit_label_centered_v6_20260726"
ROLE_COLORS = {
    "direct_ozone_sensitive": "#E94F37",
    "system_modulator": "#5D70B8",
    "competitive_ozone_scavenger": "#168A9A",
    "low_reactivity_scaffold": "#B7BDC8",
}
ROLE_SPECS = {
    1: ("direct_ozone_sensitive", False),
    2: ("low_reactivity_scaffold", False),
    3: ("system_modulator", True),
    4: ("system_modulator", False),
    5: ("system_modulator", False),
    6: ("system_modulator", False),
    7: ("system_modulator", False),
    8: ("system_modulator", False),
    9: ("system_modulator", True),
    10: ("system_modulator", False),
    11: ("system_modulator", False),
    12: ("system_modulator", False),
    13: ("competitive_ozone_scavenger", True),
    14: ("direct_ozone_sensitive", True),
    15: ("direct_ozone_sensitive", True),
    16: ("system_modulator", False),
    17: ("system_modulator", False),
    18: ("system_modulator", True),
    19: ("direct_ozone_sensitive", True),
    20: ("direct_ozone_sensitive", True),
}
ROLE_DESCRIPTIONS = {
    "direct_ozone_sensitive": "Potential direct ozone-sensitive structural site.",
    "system_modulator": "Polar, speciation, or microenvironment-modulating site.",
    "competitive_ozone_scavenger": "Chemically reactive competitive ozone-scavenging S(IV) site.",
    "low_reactivity_scaffold": "Low-reactivity hydrophobic or structural scaffold; no atom halo.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=ASSET_ROOT / "real_occurrence_key_sites_original_artifacts")
    parser.add_argument("--dpi", type=int, default=700)
    return parser.parse_args()


def load_figure_style():
    skill_dir = os.environ.get("DECAT_FIGURE_STYLE_DIR", "")
    if skill_dir:
        sys.path.insert(0, skill_dir)
    from kernel import apply_figure_style

    return apply_figure_style


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) / 255 for index in (0, 2, 4))


def direction_symbol(direction: object) -> str:
    normalized = str(direction).strip().lower()
    if normalized == "positive":
        return "↑"
    if normalized == "negative":
        return "↓"
    return "?"


def _label_bbox_near_atom(
    molecule_image: Image.Image,
    atom_xy: tuple[float, float],
    atom_color: tuple[float, float, float],
    reference_bond_length: float,
    neighbor_xy: list[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    pixels = np.asarray(molecule_image.convert("RGBA"), dtype=np.int32)
    rgb = pixels[..., :3]
    alpha = pixels[..., 3]
    color = np.rint(np.asarray(atom_color, dtype=np.float64) * 255.0).astype(np.int32)
    distance = np.sqrt(np.sum((rgb - color) ** 2, axis=2))
    search_radius = float(max(62.0, min(112.0, reference_bond_length * 0.62)))
    yy, xx = np.ogrid[: rgb.shape[0], : rgb.shape[1]]
    nearby = (xx - float(atom_xy[0])) ** 2 + (yy - float(atom_xy[1])) ** 2 <= search_radius**2
    mask = (alpha > 0) & (distance <= 165.0) & nearby
    for neighbor_x, neighbor_y in neighbor_xy:
        vector_x = float(neighbor_x) - float(atom_xy[0])
        vector_y = float(neighbor_y) - float(atom_xy[1])
        vector_length = float(np.hypot(vector_x, vector_y))
        if vector_length <= 1e-6:
            continue
        unit_x = vector_x / vector_length
        unit_y = vector_y / vector_length
        relative_x = xx - float(atom_xy[0])
        relative_y = yy - float(atom_xy[1])
        along_bond = relative_x * unit_x + relative_y * unit_y
        perpendicular = np.abs(relative_x * unit_y - relative_y * unit_x)
        bond_pixels = (along_bond >= 9.0) & (along_bond <= vector_length * 0.84) & (perpendicular <= 30.0)
        mask &= ~bond_pixels
    if not np.any(mask):
        return None
    x_start = max(0, int(np.floor(float(atom_xy[0]) - search_radius)))
    x_stop = min(mask.shape[1], int(np.ceil(float(atom_xy[0]) + search_radius)) + 1)
    y_start = max(0, int(np.floor(float(atom_xy[1]) - search_radius)))
    y_stop = min(mask.shape[0], int(np.ceil(float(atom_xy[1]) + search_radius)) + 1)
    local_mask = mask[y_start:y_stop, x_start:x_stop]
    visited = np.zeros_like(local_mask, dtype=bool)
    components = []
    for start_y, start_x in zip(*np.where(local_mask)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        x_values = []
        y_values = []
        while stack:
            current_y, current_x = stack.pop()
            x_values.append(current_x + x_start)
            y_values.append(current_y + y_start)
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if delta_x == 0 and delta_y == 0:
                        continue
                    neighbor_y = current_y + delta_y
                    neighbor_x = current_x + delta_x
                    if (
                        0 <= neighbor_y < local_mask.shape[0]
                        and 0 <= neighbor_x < local_mask.shape[1]
                        and local_mask[neighbor_y, neighbor_x]
                        and not visited[neighbor_y, neighbor_x]
                    ):
                        visited[neighbor_y, neighbor_x] = True
                        stack.append((neighbor_y, neighbor_x))
        components.append(
            (
                float(min(x_values)),
                float(min(y_values)),
                float(max(x_values)),
                float(max(y_values)),
                len(x_values),
            )
        )
    text_components = []
    for left, top, right, bottom, count in components:
        component_width = right - left + 1.0
        component_height = bottom - top + 1.0
        if component_width >= 10.0 and component_height >= 10.0 and count >= 30:
            text_components.append((left, top, right, bottom))
    if not text_components:
        return None
    selected = list(text_components)
    for left, top, right, bottom, _ in components:
        component_width = right - left + 1.0
        component_height = bottom - top + 1.0
        if component_width >= 10.0 and component_height >= 10.0:
            continue
        component_x = (left + right) / 2.0
        component_y = (top + bottom) / 2.0
        for text_left, text_top, text_right, text_bottom in text_components:
            text_x = (text_left + text_right) / 2.0
            text_y = (text_top + text_bottom) / 2.0
            if float(np.hypot(component_x - text_x, component_y - text_y)) <= 44.0:
                selected.append((left, top, right, bottom))
                break
    x_coords = [coordinate for component in selected for coordinate in (component[0], component[2])]
    y_coords = [coordinate for component in selected for coordinate in (component[1], component[3])]
    return (
        min(x_coords),
        min(y_coords),
        max(x_coords),
        max(y_coords),
    )


def draw_label_centered_halos(
    molecule: Chem.Mol,
    *,
    highlight_atoms: list[int],
    highlight_atom_colors: dict[int, tuple[float, float, float]],
    width: int,
    height: int,
    artifacts,
):
    draw_molecule, _ = artifacts._prepare_mol_for_paper_drawing(molecule)
    valid_atoms = []
    resolved_colors = {}
    highlight_radii = {}
    for raw_index in list(highlight_atoms or []):
        try:
            atom_index = int(raw_index)
        except Exception:
            continue
        if atom_index < 0 or atom_index >= int(draw_molecule.GetNumAtoms()):
            continue
        valid_atoms.append(atom_index)
        explicit_color = highlight_atom_colors.get(atom_index)
        resolved_colors[atom_index] = (
            explicit_color
            if explicit_color is not None
            else artifacts._reference_halo_color(draw_molecule, atom_index)
        )
        try:
            highlight_radii[atom_index] = artifacts._reference_highlight_radius(
                draw_molecule.GetAtomWithIdx(atom_index)
            )
        except Exception:
            highlight_radii[atom_index] = 0.30

    drawer = rdMolDraw2D.MolDraw2DCairo(int(width), int(height))
    options = drawer.drawOptions()
    options.padding = 0.01
    options.bondLineWidth = 3.3
    options.multipleBondOffset = 0.18
    options.minFontSize = 24
    options.maxFontSize = 44
    options.annotationFontScale = 1.0
    options.additionalAtomLabelPadding = 0.10
    options.centreMoleculesBeforeDrawing = True
    options.clearBackground = False
    options.prepareMolsBeforeDrawing = False
    options.legendFontSize = 18
    options.includeAtomTags = False
    options.useComplexQueryAtomSymbols = True
    options.dummiesAreAttachments = False
    options.splitBonds = False
    options.setQueryColour((0.42, 0.45, 0.51))
    options.setVariableAttachmentColour((0.15, 0.16, 0.20))
    options.fixedBondLength = 126.0
    try:
        options.updateAtomPalette(artifacts._reference_atom_palette())
    except Exception:
        pass
    try:
        for atom in draw_molecule.GetAtoms():
            if artifacts._is_attachment_like_atom(atom):
                options.atomLabels[int(atom.GetIdx())] = "*"
    except Exception:
        pass
    drawer.SetDrawOptions(options)
    drawer.SetColour((0.24, 0.26, 0.30))

    try:
        drawer.DrawMolecule(draw_molecule)
        atom_xy = {}
        for atom in draw_molecule.GetAtoms():
            try:
                atom_index = int(atom.GetIdx())
                point = drawer.GetDrawCoords(atom_index)
                atom_xy[atom_index] = (float(point.x), float(point.y))
            except Exception:
                continue
        drawer.FinishDrawing()
        with Image.open(io.BytesIO(drawer.GetDrawingText())) as drawn:
            molecule_image = drawn.convert("RGBA")

        bond_lengths = []
        for bond in draw_molecule.GetBonds():
            start_index = int(bond.GetBeginAtomIdx())
            end_index = int(bond.GetEndAtomIdx())
            if start_index in atom_xy and end_index in atom_xy:
                x0, y0 = atom_xy[start_index]
                x1, y1 = atom_xy[end_index]
                bond_lengths.append(float(np.hypot(x1 - x0, y1 - y0)))
        reference_bond_length = (
            float(np.median(bond_lengths))
            if bond_lengths
            else float(max(28.0, min(width, height) * 0.16))
        )

        halo_image = Image.new("RGBA", (int(width), int(height)), (255, 255, 255, 0))
        halo_draw = ImageDraw.Draw(halo_image, "RGBA")
        palette = artifacts._reference_atom_palette()
        for atom_index in valid_atoms:
            if atom_index not in atom_xy:
                continue
            atom = draw_molecule.GetAtomWithIdx(atom_index)
            x_value, y_value = atom_xy[atom_index]
            base_radius = float(
                max(18.0, min(42.0, reference_bond_length * highlight_radii.get(atom_index, 0.30) * 1.18))
            )
            element_color = palette.get(int(atom.GetAtomicNum()))
            label_bbox = None
            if element_color is not None and int(atom.GetAtomicNum()) in {7, 8, 9, 15, 16, 17, 35, 53}:
                neighbor_xy = [
                    atom_xy[int(neighbor.GetIdx())]
                    for neighbor in atom.GetNeighbors()
                    if int(neighbor.GetIdx()) in atom_xy
                ]
                label_bbox = _label_bbox_near_atom(
                    molecule_image,
                    (x_value, y_value),
                    element_color,
                    reference_bond_length,
                    neighbor_xy,
                )
            if label_bbox is not None:
                left, top, right, bottom = label_bbox
                x_value = (left + right) / 2.0
                y_value = (top + bottom) / 2.0
                label_radius = float(np.hypot((right - left) / 2.0, (bottom - top) / 2.0) + 16.0)
                radius_px = max(base_radius, label_radius)
            else:
                radius_px = base_radius
            halo_color = resolved_colors.get(atom_index, artifacts._reference_halo_color(draw_molecule, atom_index))
            try:
                halo_alpha = artifacts._reference_halo_alpha(atom)
            except Exception:
                halo_alpha = 36
            if atom_index in highlight_atom_colors:
                halo_alpha = max(84, halo_alpha)
            rgba = tuple(int(round(255.0 * float(value))) for value in halo_color[:3]) + (int(halo_alpha),)
            halo_draw.ellipse(
                (
                    x_value - radius_px,
                    y_value - radius_px,
                    x_value + radius_px,
                    y_value + radius_px,
                ),
                fill=rgba,
            )

        composed = Image.new("RGBA", (int(width), int(height)), (255, 255, 255, 255))
        composed = Image.alpha_composite(composed, halo_image)
        composed = Image.alpha_composite(composed, molecule_image)
        return artifacts._normalize_rgba_content(
            np.asarray(composed),
            out_width=int(width),
            out_height=int(height),
            fit_width_ratio=0.985,
            fit_height_ratio=0.92,
            pad_px=8,
            uniform_scale=5.0,
            allow_upscale=True,
        )
    except Exception:
        return None


def atom_neighborhood(molecule: Chem.Mol, center: int, radius: int) -> set[int]:
    visited = {int(center)}
    frontier = set(visited)
    for _ in range(int(radius)):
        next_frontier = set()
        for atom_index in frontier:
            next_frontier.update(atom.GetIdx() for atom in molecule.GetAtomWithIdx(atom_index).GetNeighbors())
        frontier = next_frontier - visited
        visited.update(frontier)
        if not frontier:
            break
    return visited


def resolve_display_bit_atoms(row) -> tuple[list[int], list[int]]:
    source_molecule = Chem.MolFromSmiles(str(row["display_source_smiles"]))
    display_molecule = Chem.MolFromSmiles(str(row["display_smiles"]))
    if source_molecule is None or display_molecule is None:
        raise ValueError(f"TOP{int(row['display_rank'])}: cannot parse source or display molecule")

    center_atom = int(row["display_center_atom"])
    bit_radius = int(row["display_bit_radius"])
    display_radius = max(2, bit_radius)
    source_display_atoms = atom_neighborhood(source_molecule, center_atom, display_radius)
    source_bit_atoms = atom_neighborhood(source_molecule, center_atom, bit_radius)
    exact_matches = [
        match
        for match in source_molecule.GetSubstructMatches(display_molecule, uniquify=False)
        if set(match) == source_display_atoms
    ]
    if not exact_matches:
        raise ValueError(f"TOP{int(row['display_rank'])}: no exact atom mapping for the displayed local environment")
    source_match = exact_matches[0]
    display_bit_atoms = [
        display_index
        for display_index, source_index in enumerate(source_match)
        if source_index in source_bit_atoms
    ]
    if not display_bit_atoms:
        raise ValueError(f"TOP{int(row['display_rank'])}: the bit neighbourhood has no display atoms")
    return sorted(display_bit_atoms), sorted(source_bit_atoms)


def render_role_sites(row, output_stem, dpi, artifacts, export):
    rank = int(row["display_rank"])
    role, _ = ROLE_SPECS[rank]
    atom_indices, source_bit_atoms = resolve_display_bit_atoms(row)
    molecule = export._parse_mol(str(row["display_smiles"]))
    invalid_atoms = [index for index in atom_indices if index >= molecule.GetNumAtoms()]
    if invalid_atoms:
        raise ValueError(f"TOP{rank}: invalid highlight atom index/indices {invalid_atoms}")

    color = ROLE_COLORS[role]
    highlight_colors = (
        {index: hex_to_rgb(color) for index in atom_indices} if color is not None else {}
    )
    image = draw_label_centered_halos(
        molecule,
        highlight_atoms=atom_indices,
        highlight_atom_colors=highlight_colors,
        width=1600,
        height=980,
        artifacts=artifacts,
    )
    if image is None:
        raise RuntimeError(f"Original artifact renderer returned no image for TOP{rank}")

    figure = plt.figure(figsize=(6.6, 4.3))
    overlay = figure.add_axes([0, 0, 1, 1])
    overlay.axis("off")
    overlay.set_facecolor("white")
    overlay.text(
        0.5,
        0.965,
        f"TOP{rank}({int(row['bit'])})",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        family="Times New Roman",
        color="#111111",
    )
    molecule_axes = figure.add_axes([0.045, 0.195, 0.91, 0.68])
    molecule_axes.imshow(image)
    molecule_axes.axis("off")
    arrow, arrow_color = export._arrow_style(direction_symbol(row.get("direction_pred", "")))
    overlay.text(
        0.5,
        0.105,
        arrow,
        ha="center",
        va="center",
        fontsize=30,
        fontweight="bold",
        color=arrow_color,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    export._save_all(figure, output_stem, dpi=dpi, png_only=False)
    plt.close(figure)
    return role, atom_indices, source_bit_atoms


def write_manifest(rows, output_dir: Path):
    manifest_dir = output_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "display_rank",
        "bit",
        "display_smiles",
        "display_label",
        "display_source_name",
        "direction_pred",
        "chemical_role",
        "highlight_atoms",
        "highlight_color",
        "pH_dependent",
        "source_bit_atoms",
    ]
    with (manifest_dir / "chemistry_role_site_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (manifest_dir / "chemistry_role_palette.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "roles": {
                    role: {
                        "color": ROLE_COLORS[role],
                        "description": ROLE_DESCRIPTIONS[role],
                    }
                    for role in ROLE_COLORS
                },
                "arrow_note": "Arrow direction is the signed locked-model perturbation, not atom-level reactivity or causal reaction direction.",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def main():
    args = parse_args()
    apply_figure_style = load_figure_style()
    apply_figure_style(frame="none", font="Times New Roman", sizes=(18, 15, 12), grid=False)
    artifacts, export, _ = load_recovered_original_functions()
    frame = pd.read_csv(args.csv).sort_values("display_rank")
    if set(frame["display_rank"].astype(int)) != set(ROLE_SPECS):
        raise ValueError("The locked Top-20 rows do not exactly match the chemistry-role specification.")

    output_dir = args.output_dir / OUTPUT_FOLDER
    manifest_rows = []
    for _, series in frame.iterrows():
        row = dict(series)
        rank = int(row["display_rank"])
        stem = output_dir / f"TOP{rank:02d}_fp{int(row['bit'])}_{export._safe_slug(row['display_label'])}"
        role, atom_indices, source_bit_atoms = render_role_sites(row, stem, args.dpi, artifacts, export)
        _, pH_dependent = ROLE_SPECS[rank]
        manifest_rows.append(
            {
                "display_rank": rank,
                "bit": int(row["bit"]),
                "display_smiles": row["display_smiles"],
                "display_label": row["display_label"],
                "display_source_name": row["display_source_name"],
                "direction_pred": row["direction_pred"],
                "chemical_role": role,
                "highlight_atoms": ";".join(str(index) for index in atom_indices),
                "highlight_color": ROLE_COLORS[role] or "none",
                "pH_dependent": pH_dependent,
                "source_bit_atoms": ";".join(str(index) for index in source_bit_atoms),
            }
        )
        print(f"[OK] TOP{rank:02d}: {role}")
    write_manifest(manifest_rows, output_dir)
    print(f"[OK] Wrote chemistry-role figures to: {output_dir}")


if __name__ == "__main__":
    main()
