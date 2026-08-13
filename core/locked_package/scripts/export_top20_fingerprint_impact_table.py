#!/usr/bin/env python
"""Create the locked Top-20 fingerprint-impact table as an Illustrator-editable PDF."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from render_real_occurrence_original_artifacts_exact import ASSET_ROOT


SOURCE_FOLDER = (
    ASSET_ROOT
    / "real_occurrence_key_sites_original_artifacts"
    / "key_sites_chemistry_role_bit_label_centered_v6_20260726"
)
OUTPUT_FOLDER = ASSET_ROOT / "top20_fingerprint_impact_table_v1_20260726"
OUTPUT_FILENAME = "DECAT_seed242_top20_fingerprint_impact_table.pdf"
PAGE_WIDTH = 15.0 * 72.0
PAGE_HEIGHT = 14.1 * 72.0
OUTER_MARGIN = 15.0
TABLE_STROKE = HexColor("#707070")
POSITIVE_COLOR = HexColor("#EF2430")
NEGATIVE_COLOR = HexColor("#71C93E")
TITLE_FONT = "TimesNewRoman-Bold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_FOLDER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_FOLDER)
    return parser.parse_args()


def apply_figure_style() -> None:
    skill_dir = os.environ.get("DECAT_FIGURE_STYLE_DIR", "")
    if skill_dir:
        sys.path.insert(0, skill_dir)
    from kernel import apply_figure_style as configure_figure_style

    configure_figure_style(frame="none", font="Times New Roman", sizes=(18, 15, 12), grid=False)


def register_fonts() -> None:
    font_path = Path(os.environ.get("DECAT_FONT_DIR", "")) / "timesbd.ttf"
    if font_path.exists():
        pdfmetrics.registerFont(TTFont(TITLE_FONT, str(font_path)))


def find_structure_image(source_dir: Path, rank: int, bit: int) -> Path:
    matches = sorted(source_dir.glob(f"TOP{rank:02d}_fp{bit}_*.png"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one PNG for TOP{rank}({bit}), found {len(matches)}")
    return matches[0]


def crop_structure_region(image_path: Path) -> Image.Image:
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    width, height = image.size
    structure_window = image.crop(
        (
            round(width * 0.035),
            round(height * 0.135),
            round(width * 0.965),
            round(height * 0.790),
        )
    )
    pixels = np.asarray(structure_window)
    foreground = np.min(pixels, axis=2) < 242
    foreground_y, foreground_x = np.where(foreground)
    if foreground_x.size == 0:
        raise ValueError(f"No structure pixels detected in {image_path}")
    padding_x = max(36, round((foreground_x.max() - foreground_x.min() + 1) * 0.075))
    padding_y = max(36, round((foreground_y.max() - foreground_y.min() + 1) * 0.100))
    left = max(0, int(foreground_x.min()) - padding_x)
    top = max(0, int(foreground_y.min()) - padding_y)
    right = min(structure_window.width, int(foreground_x.max()) + padding_x + 1)
    bottom = min(structure_window.height, int(foreground_y.max()) + padding_y + 1)
    return structure_window.crop((left, top, right, bottom))


def draw_impact_arrow(
    pdf: canvas.Canvas,
    center_x: float,
    center_y: float,
    direction: str,
) -> None:
    positive = str(direction).strip().lower() == "positive"
    arrow_color = POSITIVE_COLOR if positive else NEGATIVE_COLOR
    head_base_y = center_y + 4.0 if positive else center_y - 4.0
    tip_y = center_y + 20.0 if positive else center_y - 20.0
    tail_y = center_y - 20.0 if positive else center_y + 20.0
    pdf.saveState()
    pdf.setStrokeColor(arrow_color)
    pdf.setFillColor(arrow_color)
    pdf.setLineCap(1)
    pdf.setLineWidth(4.1)
    pdf.line(center_x, tail_y, center_x, head_base_y)
    arrow_head = pdf.beginPath()
    arrow_head.moveTo(center_x, tip_y)
    arrow_head.lineTo(center_x - 12.0, head_base_y)
    arrow_head.lineTo(center_x + 12.0, head_base_y)
    arrow_head.close()
    pdf.drawPath(arrow_head, fill=1, stroke=0)
    pdf.restoreState()


def draw_table_grid(
    pdf: canvas.Canvas,
    table_left: float,
    table_bottom: float,
    table_width: float,
    table_height: float,
    column_width: float,
    block_height: float,
    header_height: float,
    image_height: float,
) -> None:
    pdf.saveState()
    pdf.setStrokeColor(TABLE_STROKE)
    pdf.setLineWidth(0.72)
    pdf.rect(table_left, table_bottom, table_width, table_height, stroke=1, fill=0)
    for column_index in range(1, 5):
        x_position = table_left + column_index * column_width
        pdf.line(x_position, table_bottom, x_position, table_bottom + table_height)
    for block_index in range(4):
        block_top = table_bottom + table_height - block_index * block_height
        header_bottom = block_top - header_height
        image_bottom = header_bottom - image_height
        pdf.line(table_left, header_bottom, table_left + table_width, header_bottom)
        pdf.line(table_left, image_bottom, table_left + table_width, image_bottom)
        if block_index < 3:
            pdf.line(table_left, block_top - block_height, table_left + table_width, block_top - block_height)
    pdf.restoreState()


def create_table(source_dir: Path, output_dir: Path) -> Path:
    apply_figure_style()
    register_fonts()
    manifest_path = source_dir / "manifest" / "chemistry_role_site_manifest.csv"
    manifest = pd.read_csv(manifest_path).sort_values("display_rank")
    expected_ranks = list(range(1, 21))
    actual_ranks = manifest["display_rank"].astype(int).tolist()
    if actual_ranks != expected_ranks:
        raise ValueError("The source manifest does not contain the locked Top-20 ranks in order.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    pdf = canvas.Canvas(str(output_path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT), pageCompression=1)
    pdf.setTitle("DECAT seed242 Top-20 fingerprint impact table")
    table_left = OUTER_MARGIN
    table_bottom = OUTER_MARGIN
    table_width = PAGE_WIDTH - 2.0 * OUTER_MARGIN
    table_height = PAGE_HEIGHT - 2.0 * OUTER_MARGIN
    column_width = table_width / 5.0
    block_height = table_height / 4.0
    header_height = block_height * 0.135
    image_height = block_height * 0.675
    arrow_height = block_height - header_height - image_height
    draw_table_grid(
        pdf,
        table_left,
        table_bottom,
        table_width,
        table_height,
        column_width,
        block_height,
        header_height,
        image_height,
    )

    for row_index, row in enumerate(manifest.to_dict("records")):
        rank = int(row["display_rank"])
        bit = int(row["bit"])
        block_index = row_index // 5
        column_index = row_index % 5
        cell_left = table_left + column_index * column_width
        block_top = table_bottom + table_height - block_index * block_height
        header_bottom = block_top - header_height
        image_bottom = header_bottom - image_height
        arrow_bottom = image_bottom - arrow_height
        title_y = header_bottom + header_height * 0.36
        pdf.saveState()
        pdf.setFillColor(HexColor("#111111"))
        pdf.setFont(TITLE_FONT, 14.2)
        pdf.drawCentredString(cell_left + column_width / 2.0, title_y, f"TOP{rank}({bit})")
        pdf.restoreState()

        structure = crop_structure_region(find_structure_image(source_dir, rank, bit))
        image_margin_x = 8.0
        image_margin_y = 7.0
        pdf.drawImage(
            ImageReader(structure),
            cell_left + image_margin_x,
            image_bottom + image_margin_y,
            width=column_width - 2.0 * image_margin_x,
            height=image_height - 2.0 * image_margin_y,
            preserveAspectRatio=True,
            anchor="c",
            mask="auto",
        )
        draw_impact_arrow(
            pdf,
            cell_left + column_width / 2.0,
            arrow_bottom + arrow_height / 2.0,
            str(row["direction_pred"]),
        )

    pdf.showPage()
    pdf.save()
    return output_path


def main() -> None:
    args = parse_args()
    output_path = create_table(args.source_dir, args.output_dir)
    print(f"[OK] Wrote Illustrator-editable PDF: {output_path}")


if __name__ == "__main__":
    main()
