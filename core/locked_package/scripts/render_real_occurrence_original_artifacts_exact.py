"""Use the recovered legacy key-sites drawing functions without loading its training stack."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import io
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(
    os.environ.get(
        "DECAT_ASSET_ROOT",
        str(_PACKAGE_ROOT / "artifacts" / "legacy_rendering"),
    )
)
SOURCE_ROOT = Path(
    os.environ.get(
        "DECAT_REFERENCE_SOURCE_ROOT",
        str(_PACKAGE_ROOT / "artifacts" / "recovered_original_key_sites_runtime"),
    )
)
DEFAULT_CSV = ASSET_ROOT / "real_occurrence_top20_rebuild" / "locked_top20_real_occurrence_representatives.csv"
DEFAULT_OUTPUT = ASSET_ROOT / "real_occurrence_key_sites_original_artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render real environments with recovered legacy drawing functions.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=700)
    return parser.parse_args()


def extract_functions(path: Path, function_names: set[str], namespace: dict[str, object]) -> SimpleNamespace:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = re.match(r"^(?:async\s+def|def|class)\s+([A-Za-z_]\w*)", line)
        if match:
            starts.append((index, match.group(1)))
    definitions: dict[str, ast.AST] = {}
    for position, (start, name) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        try:
            node = ast.parse("".join(lines[start:end]), filename=str(path)).body[0]
        except SyntaxError:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[name] = node
    selected = set(function_names)
    changed = True
    while changed:
        changed = False
        for name in list(selected):
            definition = definitions.get(name)
            if definition is None:
                continue
            for child in ast.walk(definition):
                if isinstance(child, ast.Name) and child.id in definitions and child.id not in selected:
                    selected.add(child.id)
                    changed = True
    module = ast.Module(
        body=[copy.deepcopy(definitions[name]) for name in definitions if name in selected],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in selected})


def load_recovered_original_functions() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    artifacts_path = SOURCE_ROOT / "transformer_v6" / "transformer_v6" / "artifacts.py"
    export_path = SOURCE_ROOT / "export_v9_top20_atom_group_single_images.py"
    gallery_path = SOURCE_ROOT / "render_top20_atom_group_prediction_impact_gallery.py"
    for path in (artifacts_path, export_path, gallery_path):
        if not path.is_file():
            raise FileNotFoundError(f"Recovered source is missing: {path}")
    artifacts_namespace = {
        "io": io,
        "np": np,
        "Chem": Chem,
        "Image": Image,
        "ImageDraw": ImageDraw,
        "Optional": Optional,
        "rdMolDraw2D": rdMolDraw2D,
    }
    artifacts = extract_functions(
        artifacts_path,
        {"_draw_mol_paper_style", "_resolve_highlight_style_from_quality"},
        artifacts_namespace,
    )
    v5_path = ASSET_ROOT / "recovered_original_key_sites_runtime" / "transformer_v5.py"
    v5_artifacts = extract_functions(
        v5_path,
        {"_substructure_quality_metrics"},
        artifacts_namespace,
    )
    artifacts = SimpleNamespace(**vars(artifacts), **vars(v5_artifacts))
    export_namespace = {"Chem": Chem, "Path": Path, "plt": plt, "re": re}
    export = extract_functions(export_path, {"_first_bit", "_safe_slug", "_parse_mol", "_arrow_style", "_save_all"}, export_namespace)
    gallery = extract_functions(gallery_path, {"_ozone_active_site_atoms"}, {"Chem": Chem})
    return artifacts, export, gallery


def direction_symbol(direction: object) -> str:
    if str(direction).strip().lower() == "positive":
        return "↑"
    if str(direction).strip().lower() == "negative":
        return "↓"
    return "?"


def render_key_site(
    row: dict[str, object],
    output_stem: Path,
    dpi: int,
    artifacts: SimpleNamespace,
    export: SimpleNamespace,
    gallery: SimpleNamespace,
) -> list[int]:
    rank = int(row["display_rank"])
    bit = int(row["bit"])
    structure = str(row["display_smiles"])
    molecule = export._parse_mol(structure)
    original_row = dict(row)
    original_row["family_name"] = str(row.get("display_label", ""))
    original_row["family_display"] = str(row.get("display_label", ""))
    original_row["functional_display"] = str(row.get("functional_display", ""))
    key_atoms = gallery._ozone_active_site_atoms(molecule, structure, original_row)
    if not key_atoms:
        quality = artifacts._substructure_quality_metrics(structure, min_heavy_atoms=2)
        key_atoms, _ = artifacts._resolve_highlight_style_from_quality(quality)
    key_colors = {int(atom_idx): (0.60, 0.64, 0.96) for atom_idx in list(key_atoms or [])}
    image = artifacts._draw_mol_paper_style(
        molecule,
        highlight_atoms=list(key_atoms or []),
        highlight_atom_colors=key_colors,
        width=1600,
        height=980,
    )
    arrow, arrow_color = export._arrow_style(direction_symbol(row.get("direction_pred", "")))
    figure = plt.figure(figsize=(6.6, 4.3))
    overlay = figure.add_axes([0, 0, 1, 1])
    overlay.axis("off")
    overlay.set_facecolor("white")
    overlay.text(
        0.5,
        0.965,
        f"TOP{rank}({bit})",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        family="Times New Roman",
        color="#111111",
    )
    if image is not None:
        molecule_axes = figure.add_axes([0.045, 0.195, 0.91, 0.68])
        molecule_axes.imshow(image)
        molecule_axes.axis("off")
    else:
        raise RuntimeError(f"Original artifact renderer returned no image for TOP{rank}: {structure}")
    overlay.text(0.5, 0.105, arrow, ha="center", va="center", fontsize=30, fontweight="bold", color=arrow_color)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    export._save_all(figure, output_stem, dpi=dpi, png_only=False)
    plt.close(figure)
    return list(key_atoms or [])


def write_manifest(rows: list[dict[str, object]], output_dir: Path) -> None:
    manifest_dir = output_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "display_rank",
        "bit",
        "raw_hashed_fragment",
        "display_smiles",
        "display_label",
        "display_source_name",
        "display_center_atom",
        "display_bit_radius",
        "direction_pred",
        "key_site_atoms",
    ]
    with (manifest_dir / "representative_local_environment_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)
    with (manifest_dir / "renderer_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_root": str(SOURCE_ROOT),
                "entry_layout": "export_v9_top20_atom_group_single_images._render_key_site",
                "original_molecule_function": "transformer_v6.artifacts._draw_mol_paper_style",
                "original_site_function": "render_top20_atom_group_prediction_impact_gallery._ozone_active_site_atoms",
                "runtime_note": "The original drawing function bodies are loaded directly while excluding the unrelated legacy training import stack.",
            },
            handle,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    artifacts, export, gallery = load_recovered_original_functions()
    frame = pd.read_csv(args.csv).sort_values("display_rank").head(args.top_n)
    if args.preview_only:
        frame = frame.head(1)
    output_dir = args.output_dir / "key_sites"
    manifest_rows: list[dict[str, object]] = []
    for _, series in frame.iterrows():
        row = dict(series)
        rank = int(row["display_rank"])
        bit = int(row["bit"])
        stem = output_dir / f"TOP{rank:02d}_fp{bit}_{export._safe_slug(row.get('display_label', 'environment'))}"
        row["key_site_atoms"] = ";".join(str(atom_idx) for atom_idx in render_key_site(row, stem, int(args.dpi), artifacts, export, gallery))
        manifest_rows.append(row)
        print(f"[OK] TOP{rank:02d} fp{bit}")
    write_manifest(manifest_rows, args.output_dir)
    print(f"[OK] Original artifact renderer output: {args.output_dir}")


if __name__ == "__main__":
    main()
