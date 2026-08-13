from __future__ import annotations

import html
import math
import os
import re
from pathlib import Path

import cairosvg
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(
    os.environ.get(
        "DECAT_ASSET_ROOT",
        str(_PACKAGE_ROOT / "artifacts" / "legacy_rendering"),
    )
)
INPUT = ROOT / "prediction_impact_top20_atom_groups.csv"
OUTPUT = ROOT / "individual_panels_legacy_fig5_style"

PANEL_WIDTH = 1800
PANEL_HEIGHT = 1350
MOLECULE_X = 150
MOLECULE_Y = 265
MOLECULE_WIDTH = 1500
MOLECULE_HEIGHT = 760


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _group_label(row: dict) -> str:
    display = _clean(row.get("functional_display", ""))
    smiles = _clean(row.get("canonical_substructure", ""))
    if "NC(=O)" in smiles or "CNC(=O)" in smiles:
        return "-CONH-"
    if display:
        return display
    if "C(=O)" in smiles or "C=O" in smiles:
        return "C=O"
    if "S" in smiles:
        return "-S-"
    if "N" in smiles:
        return "-N-"
    if "O" in smiles:
        return "-OH"
    if "=" in smiles:
        return "C=C"
    carbon_count = smiles.count("C") + smiles.count("c")
    if carbon_count >= 3:
        return "-C-C-C-"
    return "-C-C-"


def _molecule_svg(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        molecule = Chem.MolFromSmarts(smiles)
    if molecule is None:
        raise ValueError(f"Cannot draw atom group: {smiles}")

    drawer = Draw.MolDraw2DSVG(MOLECULE_WIDTH, MOLECULE_HEIGHT)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = 2.8
    options.minFontSize = 30
    options.maxFontSize = 52
    options.padding = 0.12
    options.highlightRadius = 0.24
    options.atomHighlightsAreCircles = True
    options.fillHighlights = True
    options.setAtomPalette(
        {
            6: (0.67, 0.71, 0.79),
            7: (0.23, 0.37, 0.73),
            8: (0.95, 0.30, 0.36),
            9: (0.18, 0.62, 0.47),
            15: (0.08, 0.66, 0.76),
            16: (0.88, 0.65, 0.04),
            17: (0.18, 0.62, 0.47),
            35: (0.18, 0.62, 0.47),
            53: (0.58, 0.35, 0.72),
        }
    )
    highlight_atoms = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() not in {1, 6}
    ]
    highlight_colours = {index: (0.78, 0.80, 0.93) for index in highlight_atoms}
    drawer.DrawMolecule(molecule, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colours)
    drawer.FinishDrawing()
    drawing = drawer.GetDrawingText()
    svg_start = drawing.find("<svg")
    content_start = drawing.find(">", svg_start) + 1
    return drawing[content_start : drawing.rfind("</svg>")]


def _panel_svg(rank: int, row: dict) -> str:
    smiles = _clean(row.get("canonical_substructure", row.get("substructure_smiles", "")))
    bit_name = _clean(row.get("bit_name", ""))
    score = float(row.get("consensus_score", 0.0))
    delta = float(row.get("signed_prediction_delta_logk", 0.0))
    support = int(float(row.get("support_count", 0)))
    arrow = "↑" if delta >= 0 else "↓"
    direction_color = "#d24d62" if delta >= 0 else "#1c9d85"
    molecule = _molecule_svg(smiles)
    escaped_smiles = html.escape(smiles)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{PANEL_WIDTH}" height="{PANEL_HEIGHT}" viewBox="0 0 {PANEL_WIDTH} {PANEL_HEIGHT}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="42" y="78" font-family="Cambria, Times New Roman, serif" font-size="52" font-weight="700" fill="#1f2d45">{rank}.</text>
  <text x="42" y="140" font-family="Cambria, Times New Roman, serif" font-size="48" font-weight="700" fill="#1f2d45">{html.escape(bit_name)}</text>
  <text x="1758" y="78" text-anchor="end" font-family="Cambria, Times New Roman, serif" font-size="45" font-weight="700" fill="#314159">{html.escape(_group_label(row))}</text>
  <text x="900" y="192" text-anchor="middle" font-family="Cambria, Times New Roman, serif" font-size="24" font-weight="600" fill="#52647d">score {score:.3f} | support={support}</text>
  <text x="900" y="230" text-anchor="middle" font-family="Cambria, Times New Roman, serif" font-size="28" font-weight="700" fill="{direction_color}">{arrow} Δlogk {delta:+.3f}</text>
  <g transform="translate({MOLECULE_X},{MOLECULE_Y})">{molecule}</g>
  <text x="900" y="1245" text-anchor="middle" font-family="Cambria, Times New Roman, serif" font-size="33" font-weight="600" fill="#9aaac1">{escaped_smiles}</text>
</svg>'''


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT.glob("TOP*.*"):
        path.unlink()
    table = pd.read_csv(INPUT).sort_values("consensus_score", ascending=False).reset_index(drop=True)
    if len(table) != 20:
        raise RuntimeError(f"Expected 20 atom-group rows, got {len(table)}")
    index_rows = []
    for rank, row in enumerate(table.to_dict("records"), start=1):
        svg = _panel_svg(rank, row)
        stem = f"TOP{rank:02d}_{_clean(row['bit_name'])}"
        svg_path = OUTPUT / f"{stem}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(OUTPUT / f"{stem}.pdf"), output_width=PANEL_WIDTH, output_height=PANEL_HEIGHT)
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(OUTPUT / f"{stem}.png"), output_width=PANEL_WIDTH, output_height=PANEL_HEIGHT)
        index_rows.append(
            {
                "panel": rank,
                "bit_name": row["bit_name"],
                "substructure": row["canonical_substructure"],
                "label": _group_label(row),
                "consensus_score": row["consensus_score"],
                "signed_prediction_delta_logk": row["signed_prediction_delta_logk"],
                "direction_pred": row["direction_pred"],
            }
        )
    pd.DataFrame(index_rows).to_csv(OUTPUT / "panel_index.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
