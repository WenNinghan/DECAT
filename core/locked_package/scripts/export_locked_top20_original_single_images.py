"""Render locked seed-242 Top-20 motifs in the original V9 single-image style.

This is intentionally a compatibility renderer for
``export_v9_top20_atom_group_single_images.py``.  The old private
``transformer_v6.artifacts`` helper is not present in the archived source, so
only its RDKit drawing primitive is replaced here; canvas size, axes geometry,
typography, arrow mapping, output formats, and naming follow the original.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_ASSET_ROOT = Path(
    os.environ.get(
        "DECAT_ASSET_ROOT",
        str(_PACKAGE_ROOT / "artifacts" / "legacy_rendering"),
    )
)
DEFAULT_CSV = _ASSET_ROOT / "prediction_impact_top20_atom_groups.csv"
DEFAULT_OUT_DIR = DEFAULT_CSV.parent / "original_v9_single_image_style"

ARROW_STYLE = {
    "positive": ("↑", "#EF1D2D"),
    "negative": ("↓", "#16833A"),
}
HIGHLIGHT = (0.75, 0.78, 0.94)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export locked Top-20 motifs in original V9 style.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=700)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--molecule-style", choices=("tnr-bold", "platform-bold"), default="tnr-bold")
    parser.add_argument("--png-only", action="store_true")
    return parser.parse_args()


def safe_slug(value: object, max_length: int = 52) -> str:
    if pd.isna(value):
        return "atom_group"
    text = str(value or "").strip()
    text = text.replace("+", "plus").replace("-", "minus")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text[:max_length].strip("_") or "atom_group"


def parse_molecule(smiles: object) -> Chem.Mol | None:
    text = str(smiles or "").strip()
    if not text or text.lower() == "nan":
        return None
    molecule = Chem.MolFromSmiles(text)
    return molecule if molecule is not None else Chem.MolFromSmarts(text)


def key_site_atoms(molecule: Chem.Mol, row: dict) -> list[int]:
    """Exact compatibility port of the archived `_ozone_active_site_atoms`."""
    structure = str(row.get("canonical_substructure", "")).strip()
    label = " ".join(
        [
            str(row.get("functional_group", "")),
            str(row.get("functional_display", "")),
            structure,
        ]
    ).lower()
    active: set[int] = set()
    for atom in molecule.GetAtoms():
        atom_index = int(atom.GetIdx())
        atomic_number = int(atom.GetAtomicNum())
        if int(atom.GetFormalCharge()) < 0 or atomic_number in {7, 15, 16}:
            active.add(atom_index)
        if atomic_number == 8 and ("phenol" in label or "phenoxide" in label or "[o-]" in label or "[O-]" in structure):
            active.add(atom_index)
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        begin_index, end_index = int(begin.GetIdx()), int(end.GetIdx())
        atom_numbers = {int(begin.GetAtomicNum()), int(end.GetAtomicNum())}
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.DOUBLE and atom_numbers == {6}:
            active.update([begin_index, end_index])
        if bond_type == Chem.rdchem.BondType.DOUBLE and atom_numbers == {6, 8}:
            if any(token in label for token in ("carbonyl", "carbox", "c=o", "c(o)=o", "=o")):
                active.update([begin_index, end_index])
        if bond_type == Chem.rdchem.BondType.DOUBLE and atom_numbers == {8, 16}:
            active.update([begin_index, end_index])
    for atom_index in list(active):
        atom = molecule.GetAtomWithIdx(atom_index)
        if int(atom.GetAtomicNum()) in {7, 8, 15, 16}:
            active.update(int(neighbor.GetIdx()) for neighbor in atom.GetNeighbors()
                          if int(neighbor.GetAtomicNum()) == 6 and bool(neighbor.GetIsAromatic()))
    if not active and ("aromatic" in label or "aryl" in label):
        active.update(int(atom.GetIdx()) for atom in molecule.GetAtoms() if bool(atom.GetIsAromatic()))
    return sorted(atom_index for atom_index in active if 0 <= atom_index < molecule.GetNumAtoms())


def molecule_image(molecule: Chem.Mol, highlight_atoms: list[int], style: str) -> np.ndarray:
    drawer = Draw.MolDraw2DCairo(1600, 980)
    options = drawer.drawOptions()
    options.clearBackground = True
    if style == "platform-bold":
        options.bondLineWidth = 3.2
        options.padding = 0.08
        options.minFontSize = 22
        options.maxFontSize = 34
        options.additionalAtomLabelPadding = 0.12
    else:
        options.bondLineWidth = 4.8
        options.padding = 0.10
        options.minFontSize = 44
        options.maxFontSize = 84
        options.additionalAtomLabelPadding = 0.26
        font_path = Path(os.environ.get("DECAT_FONT_DIR", "")) / "timesbd.ttf"
        if font_path.is_file():
            options.fontFile = str(font_path)
    options.highlightRadius = 0.24
    options.atomHighlightsAreCircles = True
    options.fillHighlights = True
    if style != "platform-bold":
        options.setAtomPalette(
            {
                6: (0.38, 0.42, 0.50),
                7: (0.20, 0.35, 0.75),
                8: (0.94, 0.15, 0.21),
                9: (0.13, 0.57, 0.33),
                15: (0.95, 0.52, 0.10),
                16: (0.93, 0.63, 0.05),
                17: (0.15, 0.58, 0.37),
                35: (0.15, 0.58, 0.37),
                53: (0.45, 0.28, 0.65),
            }
        )
    colors = {atom_index: HIGHLIGHT for atom_index in highlight_atoms}
    drawer.DrawMolecule(
        molecule,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=colors,
        highlightBonds=[],
    )
    drawer.FinishDrawing()
    image = Image.open(__import__("io").BytesIO(drawer.GetDrawingText())).convert("RGB")
    return np.asarray(image)


def arrow_for(row: dict) -> tuple[str, str]:
    direction = str(row.get("direction_pred", "")).strip().lower()
    return ARROW_STYLE.get(direction, ("?", "#6B7280"))


def save_all(figure: plt.Figure, stem: Path, dpi: int, png_only: bool) -> list[str]:
    outputs: list[str] = []
    for suffix in ([".png"] if png_only else [".png", ".pdf", ".svg"]):
        path = stem.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.08, "facecolor": "white"}
        if suffix == ".png":
            kwargs["dpi"] = dpi
        figure.savefig(path, **kwargs)
        outputs.append(str(path))
    return outputs


def render_clean(row: dict, rank: int, stem: Path, dpi: int, png_only: bool, style: str, *, key_sites: bool) -> tuple[list[str], list[int]]:
    bit = int(row["bit"])
    molecule = parse_molecule(row.get("canonical_substructure"))
    sites = key_site_atoms(molecule, row) if molecule is not None else []
    displayed_sites = sites if key_sites else []
    image = molecule_image(molecule, displayed_sites, style) if molecule is not None else None
    arrow, arrow_color = arrow_for(row)

    figure = plt.figure(figsize=(6.6, 4.3))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.axis("off")
    axis.set_facecolor("white")
    axis.text(0.5, 0.965, f"TOP{rank}({bit})", ha="center", va="top", fontsize=18,
              fontweight="bold", family="Times New Roman", color="#111111")
    if image is not None:
        molecule_axis = figure.add_axes([0.045, 0.195, 0.91, 0.68])
        molecule_axis.imshow(image)
        molecule_axis.axis("off")
    else:
        axis.text(0.5, 0.54, str(row.get("canonical_substructure", "")), ha="center", va="center",
                  fontsize=18, color="#111827")
    axis.text(0.5, 0.105, arrow, ha="center", va="center", fontsize=30,
              fontweight="bold", color=arrow_color)
    outputs = save_all(figure, stem, dpi, png_only)
    plt.close(figure)
    return outputs, sites


def render_annotated(row: dict, rank: int, stem: Path, dpi: int, png_only: bool, style: str) -> list[str]:
    """Archival companion with the same structure geometry plus provenance."""
    bit = int(row["bit"])
    molecule = parse_molecule(row.get("canonical_substructure"))
    sites = key_site_atoms(molecule, row) if molecule is not None else []
    image = molecule_image(molecule, sites, style) if molecule is not None else None
    arrow, arrow_color = arrow_for(row)
    figure = plt.figure(figsize=(8.2, 5.2))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.axis("off")
    axis.set_facecolor("white")
    axis.text(0.06, 0.94, f"TOP{rank}({bit})", ha="left", va="top", fontsize=18,
              fontweight="bold", family="Times New Roman", color="#111111")
    if image is not None:
        molecule_axis = figure.add_axes([0.07, 0.19, 0.86, 0.64])
        molecule_axis.imshow(image)
        molecule_axis.axis("off")
    axis.text(0.50, 0.105, arrow, ha="center", va="center", fontsize=30,
              fontweight="bold", color=arrow_color)
    axis.text(0.06, 0.04, f"support={int(row.get('support_count', 0))} | score={float(row.get('consensus_score', 0)):.3f}",
              ha="left", va="bottom", fontsize=9, color="#475569")
    axis.text(0.94, 0.04, str(row.get("canonical_substructure", "")), ha="right", va="bottom",
              fontsize=9, color="#64748B")
    outputs = save_all(figure, stem, dpi, png_only)
    plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.csv, encoding="utf-8-sig").sort_values("consensus_score", ascending=False, kind="stable")
    rows = data.head(max(1, int(args.top_n))).to_dict(orient="records")
    if len(rows) != min(int(args.top_n), len(data)):
        raise RuntimeError("Top-N rows could not be loaded.")
    output = args.output_dir
    clean_dir = output / "clean"
    key_dir = output / "key_sites"
    annotated_dir = output / "annotated"
    for directory in (clean_dir, key_dir, annotated_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for rank, row in enumerate(rows, start=1):
        bit = int(row["bit"])
        group = row.get("functional_group")
        slug = safe_slug(group if not pd.isna(group) and str(group).strip() else row.get("canonical_substructure"))
        name = f"TOP{rank:02d}_fp{bit}_{slug}"
        clean_outputs, sites = render_clean(row, rank, clean_dir / name, args.dpi, args.png_only, args.molecule_style, key_sites=False)
        key_outputs, _ = render_clean(row, rank, key_dir / name, args.dpi, args.png_only, args.molecule_style, key_sites=True)
        annotated_outputs = render_annotated(row, rank, annotated_dir / name, args.dpi, args.png_only, args.molecule_style)
        manifest_rows.append({
            "display_rank": rank,
            "bit_name": row["bit_name"],
            "canonical_substructure": row["canonical_substructure"],
            "direction_pred": row["direction_pred"],
            "key_site_atom_indices": ";".join(map(str, sites)),
            "clean_outputs": clean_outputs,
            "key_site_outputs": key_outputs,
            "annotated_outputs": annotated_outputs,
        })
    manifest = {"source_csv": str(args.csv), "dpi": args.dpi, "molecule_style": args.molecule_style, "top_n": len(manifest_rows), "rows": manifest_rows}
    (output / "single_image_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(manifest_rows).to_csv(output / "single_image_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] Exported {len(manifest_rows)} original-style Top-20 motif image sets to: {output}")


if __name__ == "__main__":
    main()
