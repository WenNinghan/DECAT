"""Render locked Top-20 representative local environments in the legacy key-site layout.

This script preserves the locked Top-20 ranking and contribution records.  It
only renders chemically valid representative local environments recovered from
the locked test split, replacing raw hashed fragments that cannot be displayed
as standalone molecular structures.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
DEFAULT_CSV = ASSET_ROOT / "real_occurrence_top20_rebuild" / "locked_top20_real_occurrence_representatives.csv"
DEFAULT_OUTPUT = ASSET_ROOT / "real_occurrence_key_sites_reference_style"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render locked Top-20 representative local environments.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=700)
    return parser.parse_args()


def safe_slug(value: object) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "representative_environment"))
    return normalized.strip("_") or "representative_environment"


def direction_style(direction: object) -> tuple[str, str]:
    if str(direction).strip().lower() == "positive":
        return "↑", "#EF1D2D"
    if str(direction).strip().lower() == "negative":
        return "↓", "#16833A"
    return "?", "#6B7280"


def active_site_atoms(mol: Chem.Mol, structure_text: str, row: dict[str, object]) -> list[int]:
    label = " ".join(
        [
            str(row.get("functional_group", "")),
            str(row.get("functional_display", "")),
            str(row.get("display_label", "")),
            structure_text,
        ]
    ).lower()
    active: set[int] = set()
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atomic_number = atom.GetAtomicNum()
        if atom.GetFormalCharge() < 0 or atomic_number in {7, 15, 16}:
            active.add(atom_idx)
        if atomic_number == 8 and ("phenol" in label or "phenoxide" in label or "[o-]" in label):
            active.add(atom_idx)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        pair = {begin.GetAtomicNum(), end.GetAtomicNum()}
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if pair == {6}:
            active.update([begin.GetIdx(), end.GetIdx()])
        if pair == {6, 8} and any(token in label for token in ("carbonyl", "carbox", "=o")):
            active.update([begin.GetIdx(), end.GetIdx()])
        if pair == {8, 16}:
            active.update([begin.GetIdx(), end.GetIdx()])
    for atom_idx in list(active):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() not in {7, 8, 15, 16}:
            continue
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 6 and neighbor.GetIsAromatic():
                active.add(neighbor.GetIdx())
    if not active and ("aromatic" in label or "aryl" in label):
        active.update(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsAromatic())
    return sorted(active)


def draw_molecule(smiles: str, highlighted_atoms: list[int]) -> Image.Image:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid representative SMILES: {smiles}")
    molecule = rdMolDraw2D.PrepareMolForDrawing(molecule)
    drawer = rdMolDraw2D.MolDraw2DCairo(1600, 980)
    options = drawer.drawOptions()
    options.bondLineWidth = 2.6
    options.setAtomPalette(
        {
            6: (0.38, 0.42, 0.50),
            7: (0.20, 0.33, 0.78),
            8: (0.95, 0.12, 0.14),
            9: (0.12, 0.58, 0.30),
            15: (0.92, 0.48, 0.08),
            16: (0.86, 0.65, 0.05),
            17: (0.14, 0.56, 0.25),
        }
    )
    drawer.DrawMolecule(molecule)
    atom_coordinates = {
        atom_idx: drawer.GetDrawCoords(atom_idx)
        for atom_idx in highlighted_atoms
    }
    drawer.FinishDrawing()
    molecule_layer = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")
    if not highlighted_atoms:
        return molecule_layer
    foreground_data = bytearray(molecule_layer.tobytes())
    for pixel_index in range(0, len(foreground_data), 4):
        red, green, blue = foreground_data[pixel_index : pixel_index + 3]
        if red >= 250 and green >= 250 and blue >= 250:
            foreground_data[pixel_index + 3] = 0
    foreground = Image.frombytes("RGBA", molecule_layer.size, bytes(foreground_data))
    canvas = Image.new("RGBA", molecule_layer.size, "white")
    halos = Image.new("RGBA", molecule_layer.size, (255, 255, 255, 0))
    halo_draw = ImageDraw.Draw(halos)
    halo_radius = 92
    for point in atom_coordinates.values():
        halo_draw.ellipse(
            [point.x - halo_radius, point.y - halo_radius, point.x + halo_radius, point.y + halo_radius],
            fill=(184, 194, 245, 190),
        )
    return Image.alpha_composite(Image.alpha_composite(canvas, halos), foreground)


def render_single(row: dict[str, object], output_stem: Path, highlighted: bool, dpi: int) -> tuple[list[str], list[int]]:
    rank = int(row["display_rank"])
    bit = int(row["bit"])
    smiles = str(row["display_smiles"])
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid representative SMILES in TOP{rank}: {smiles}")
    sites = active_site_atoms(molecule, smiles, row) if highlighted else []
    molecule_image = draw_molecule(smiles, sites)
    arrow, arrow_color = direction_style(row.get("direction_pred", ""))

    figure = plt.figure(figsize=(6.6, 4.3))
    figure.patch.set_facecolor("white")
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
    molecule_axes = figure.add_axes([0.045, 0.195, 0.91, 0.68])
    molecule_axes.imshow(molecule_image)
    molecule_axes.axis("off")
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
    outputs: list[str] = []
    for suffix in (".png", ".pdf", ".svg"):
        output_path = output_stem.with_suffix(suffix)
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        outputs.append(str(output_path))
    plt.close(figure)
    return outputs, sites


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
    with (manifest_dir / "README.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            "Each diagram shows a representative local molecular environment in which the locked Morgan bit occurs in the locked test split.\n"
            "The ranked bits, predictive-impact values, directions, and model outputs are unchanged.\n"
            "A representative local environment is not a unique functional-group assignment or a new causal score.\n"
            "key_sites uses the legacy rule-based ozone-site halo convention; clean contains the same structures without halos.\n"
        )
    with (manifest_dir / "render_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": str(DEFAULT_CSV),
                "layout": "legacy key_sites single-image layout",
                "figure_inches": [6.6, 4.3],
                "molecule_canvas": [1600, 980],
                "highlight_color_rgb": [0.60, 0.64, 0.96],
                "highlight_bonds": False,
            },
            handle,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv).sort_values("display_rank").head(args.top_n)
    if args.preview_only:
        frame = frame.head(1)
    if frame.empty:
        raise RuntimeError(f"No rows found in {args.csv}")
    manifest_rows: list[dict[str, object]] = []
    for _, series in frame.iterrows():
        row = dict(series)
        rank = int(row["display_rank"])
        bit = int(row["bit"])
        stem = f"TOP{rank:02d}_fp{bit}_{safe_slug(row.get('display_label'))}"
        render_single(row, args.output_dir / "clean" / stem, highlighted=False, dpi=args.dpi)
        _, key_sites = render_single(row, args.output_dir / "key_sites" / stem, highlighted=True, dpi=args.dpi)
        manifest_row = {column: row.get(column, "") for column in row}
        manifest_row["key_site_atoms"] = ";".join(str(atom_idx) for atom_idx in key_sites)
        manifest_rows.append(manifest_row)
        print(f"[OK] TOP{rank:02d} fp{bit}")
    write_manifest(manifest_rows, args.output_dir)
    print(f"[OK] Rendered {len(manifest_rows)} representative environments to {args.output_dir}")


if __name__ == "__main__":
    main()
