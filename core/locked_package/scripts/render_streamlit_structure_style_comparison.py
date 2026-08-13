"""Compare the archived Streamlit RDKit structure style with a bold variant."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(
    os.environ.get(
        "DECAT_OUTPUT_PATH",
        str(_PACKAGE_ROOT / "artifacts" / "legacy_rendering" / "streamlit_structure_style_comparison.png"),
    )
)
SMILES = "Nc1nnc(c(n1)N)c2cccc(Cl)c2Cl"


def make_bold_image(molecule: Chem.Mol) -> Image.Image:
    drawer = Draw.MolDraw2DCairo(720, 495)
    options = drawer.drawOptions()
    options.bondLineWidth = 3.2
    options.minFontSize = 22
    options.maxFontSize = 34
    options.additionalAtomLabelPadding = 0.12
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")


def main() -> None:
    molecule = Chem.MolFromSmiles(SMILES)
    if molecule is None:
        raise ValueError("Reference SMILES could not be parsed.")
    original = Draw.MolToImage(molecule, size=(720, 495)).convert("RGB")
    bold = make_bold_image(molecule)
    canvas = Image.new("RGB", (1480, 620), "white")
    canvas.paste(original, (20, 105))
    canvas.paste(bold, (740, 105))
    draw = ImageDraw.Draw(canvas)
    font_dir = Path(os.environ.get("DECAT_FONT_DIR", ""))
    title_path = font_dir / "timesbd.ttf"
    label_path = font_dir / "times.ttf"
    title_font = ImageFont.truetype(str(title_path), 36) if title_path.is_file() else ImageFont.load_default()
    label_font = ImageFont.truetype(str(label_path), 26) if label_path.is_file() else ImageFont.load_default()
    draw.text((740, 20), "RDKit molecule-drawing comparison", anchor="ma", fill="#111111", font=title_font)
    draw.text((380, 70), "Streamlit original (default RDKit)", anchor="ma", fill="#334155", font=label_font)
    draw.text((1100, 70), "Recommended bold variant", anchor="ma", fill="#334155", font=label_font)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, dpi=(300, 300))
    print(OUTPUT)


if __name__ == "__main__":
    main()
