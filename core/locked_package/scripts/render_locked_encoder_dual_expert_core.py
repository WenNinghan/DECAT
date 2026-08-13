#!/usr/bin/env python
"""Render the locked DECAT encoder and dual-expert core as editable SVG and 600-dpi PNG."""

from __future__ import annotations

import html
import math
import os
import sys
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

from render_real_occurrence_original_artifacts_exact import ASSET_ROOT


CANVAS_WIDTH = 3600
CANVAS_HEIGHT = 2550
RASTER_SCALE = 2
OUTPUT_DIRECTORY = ASSET_ROOT / "locked_seed242_model_flow_20260727" / "encoder_dual_expert_core_v2_20260727"
SVG_FILENAME = "DECAT_seed242_encoder_dual_expert_core_editable.svg"
PNG_FILENAME = "DECAT_seed242_encoder_dual_expert_core_600dpi.png"
BLUE = "#243C98"
BLUE_LIGHT = "#EDF2FF"
YELLOW = "#FFF6D8"
GRAY_LIGHT = "#F5F6F8"
PURPLE = "#F0D7F4"
GREEN = "#E8F3D9"
GREEN_STROKE = "#4D8A3D"
ORANGE = "#FFF0D8"
ORANGE_STROKE = "#DB792B"
PINK = "#FFE9EF"
PINK_STROKE = "#B5406B"
RF_FILL = "#FFF0D0"
RF_STROKE = "#B97610"
PRIOR_FILL = "#FCF8FF"
PRIOR_STROKE = "#8851A6"
BLACK = "#151515"
GRAY = "#5F6368"


def apply_figure_style() -> None:
    skill_directory = os.environ.get("DECAT_FIGURE_STYLE_DIR", "")
    if skill_directory:
        sys.path.insert(0, skill_directory)
    from kernel import apply_figure_style as configure_figure_style

    configure_figure_style(frame="none", font="Times New Roman", sizes=(18, 15, 12), grid=False)


class SvgCanvas:
    def __init__(self) -> None:
        self.elements: list[str] = []

    def rounded_rect(
        self,
        left: float,
        top: float,
        width: float,
        height: float,
        fill: str,
        stroke: str,
        stroke_width: float = 8.0,
        radius: float = 20.0,
        dashed: bool = False,
    ) -> None:
        dash = ' stroke-dasharray="22 15"' if dashed else ""
        self.elements.append(
            f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dash}/>'
        )

    def line(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        stroke: str = BLACK,
        stroke_width: float = 8.0,
        dashed: bool = False,
    ) -> None:
        dash = ' stroke-dasharray="22 15"' if dashed else ""
        self.elements.append(
            f'<line x1="{start_x}" y1="{start_y}" x2="{end_x}" y2="{end_y}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" stroke-linecap="round"{dash}/>'
        )

    def arrow(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        stroke: str = BLACK,
        stroke_width: float = 8.0,
        dashed: bool = False,
    ) -> None:
        dash = ' stroke-dasharray="22 15"' if dashed else ""
        self.elements.append(
            f'<line x1="{start_x}" y1="{start_y}" x2="{end_x}" y2="{end_y}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" stroke-linecap="round" marker-end="url(#arrowhead)"{dash}/>'
        )

    def circle(self, center_x: float, center_y: float, radius: float, fill: str, stroke: str, stroke_width: float = 6.0) -> None:
        self.elements.append(
            f'<circle cx="{center_x}" cy="{center_y}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def text(
        self,
        center_x: float,
        center_y: float,
        lines: list[str],
        font_size: float,
        color: str = BLACK,
        bold: bool = False,
        italic: bool = False,
        line_gap: float = 1.15,
    ) -> None:
        weight = "700" if bold else "400"
        style = "italic" if italic else "normal"
        offset = (len(lines) - 1) * font_size * line_gap / 2.0
        tspans = []
        for line_index, line_value in enumerate(lines):
            y_position = center_y - offset + line_index * font_size * line_gap
            tspans.append(f'<tspan x="{center_x}" y="{y_position}">{html.escape(line_value)}</tspan>')
        self.elements.append(
            f'<text text-anchor="middle" dominant-baseline="middle" font-family="Times New Roman, Times, serif" '
            f'font-size="{font_size}" font-weight="{weight}" font-style="{style}" fill="{color}">' + "".join(tspans) + "</text>"
        )

    def write(self, output_path: Path) -> None:
        header = f'''<svg xmlns="http://www.w3.org/2000/svg" width="12in" height="8.5in" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">
<defs>
  <marker id="arrowhead" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto" markerUnits="strokeWidth">
    <path d="M 0 0 L 14 7 L 0 14 z" fill="#151515"/>
  </marker>
</defs>
<rect x="0" y="0" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" fill="#FFFFFF"/>
'''
        output_path.write_text(header + "\n".join(self.elements) + "\n</svg>\n", encoding="utf-8")


class PngCanvas:
    def __init__(self) -> None:
        self.image = Image.new("RGB", (CANVAS_WIDTH * RASTER_SCALE, CANVAS_HEIGHT * RASTER_SCALE), "white")
        self.draw = ImageDraw.Draw(self.image)

    def _scale(self, value: float) -> int:
        return round(value * RASTER_SCALE)

    def _font(self, font_size: float, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
        font_name = "times.ttf"
        if bold and italic:
            font_name = "timesbi.ttf"
        elif bold:
            font_name = "timesbd.ttf"
        elif italic:
            font_name = "timesi.ttf"
        font_path = Path(os.environ.get("DECAT_FONT_DIR", "")) / font_name
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), self._scale(font_size))
        return ImageFont.load_default()

    def rounded_rect(
        self,
        left: float,
        top: float,
        width: float,
        height: float,
        fill: str,
        stroke: str,
        stroke_width: float = 8.0,
        radius: float = 20.0,
        dashed: bool = False,
    ) -> None:
        bounds = [self._scale(left), self._scale(top), self._scale(left + width), self._scale(top + height)]
        if dashed:
            self.draw.rounded_rectangle(bounds, radius=self._scale(radius), fill=ImageColor.getrgb(fill))
            self._dashed_rect(bounds, stroke, self._scale(stroke_width), self._scale(radius))
        else:
            self.draw.rounded_rectangle(
                bounds,
                radius=self._scale(radius),
                fill=ImageColor.getrgb(fill),
                outline=ImageColor.getrgb(stroke),
                width=self._scale(stroke_width),
            )

    def _dashed_rect(self, bounds: list[int], color: str, stroke_width: int, radius: int) -> None:
        left, top, right, bottom = bounds
        self._dashed_line((left + radius, top), (right - radius, top), color, stroke_width)
        self._dashed_line((right, top + radius), (right, bottom - radius), color, stroke_width)
        self._dashed_line((right - radius, bottom), (left + radius, bottom), color, stroke_width)
        self._dashed_line((left, bottom - radius), (left, top + radius), color, stroke_width)
        self.draw.arc((left, top, left + 2 * radius, top + 2 * radius), 180, 270, fill=ImageColor.getrgb(color), width=stroke_width)
        self.draw.arc((right - 2 * radius, top, right, top + 2 * radius), 270, 360, fill=ImageColor.getrgb(color), width=stroke_width)
        self.draw.arc((right - 2 * radius, bottom - 2 * radius, right, bottom), 0, 90, fill=ImageColor.getrgb(color), width=stroke_width)
        self.draw.arc((left, bottom - 2 * radius, left + 2 * radius, bottom), 90, 180, fill=ImageColor.getrgb(color), width=stroke_width)

    def _dashed_line(self, start: tuple[int, int], end: tuple[int, int], color: str, stroke_width: int) -> None:
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        total = math.hypot(delta_x, delta_y)
        if total == 0:
            return
        unit_x = delta_x / total
        unit_y = delta_y / total
        dash = self._scale(22)
        gap = self._scale(15)
        position = 0.0
        while position < total:
            next_position = min(total, position + dash)
            self.draw.line(
                [
                    (round(start[0] + unit_x * position), round(start[1] + unit_y * position)),
                    (round(start[0] + unit_x * next_position), round(start[1] + unit_y * next_position)),
                ],
                fill=ImageColor.getrgb(color),
                width=stroke_width,
            )
            position = next_position + gap

    def line(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        stroke: str = BLACK,
        stroke_width: float = 8.0,
        dashed: bool = False,
    ) -> None:
        start = (self._scale(start_x), self._scale(start_y))
        end = (self._scale(end_x), self._scale(end_y))
        if dashed:
            self._dashed_line(start, end, stroke, self._scale(stroke_width))
        else:
            self.draw.line([start, end], fill=ImageColor.getrgb(stroke), width=self._scale(stroke_width))

    def arrow(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        stroke: str = BLACK,
        stroke_width: float = 8.0,
        dashed: bool = False,
    ) -> None:
        self.line(start_x, start_y, end_x, end_y, stroke, stroke_width, dashed)
        angle = math.atan2(end_y - start_y, end_x - start_x)
        arrow_length = self._scale(20.0)
        arrow_width = self._scale(13.0)
        end_point = (self._scale(end_x), self._scale(end_y))
        left_point = (
            round(end_point[0] - arrow_length * math.cos(angle) + arrow_width * math.sin(angle)),
            round(end_point[1] - arrow_length * math.sin(angle) - arrow_width * math.cos(angle)),
        )
        right_point = (
            round(end_point[0] - arrow_length * math.cos(angle) - arrow_width * math.sin(angle)),
            round(end_point[1] - arrow_length * math.sin(angle) + arrow_width * math.cos(angle)),
        )
        self.draw.polygon([end_point, left_point, right_point], fill=ImageColor.getrgb(stroke))

    def circle(self, center_x: float, center_y: float, radius: float, fill: str, stroke: str, stroke_width: float = 6.0) -> None:
        bounds = [
            self._scale(center_x - radius),
            self._scale(center_y - radius),
            self._scale(center_x + radius),
            self._scale(center_y + radius),
        ]
        self.draw.ellipse(bounds, fill=ImageColor.getrgb(fill), outline=ImageColor.getrgb(stroke), width=self._scale(stroke_width))

    def text(
        self,
        center_x: float,
        center_y: float,
        lines: list[str],
        font_size: float,
        color: str = BLACK,
        bold: bool = False,
        italic: bool = False,
        line_gap: float = 1.15,
    ) -> None:
        font = self._font(font_size, bold=bold, italic=italic)
        line_height = self._scale(font_size * line_gap)
        total_height = line_height * len(lines)
        y_position = self._scale(center_y) - total_height // 2
        for line_value in lines:
            box = self.draw.textbbox((0, 0), line_value, font=font)
            text_width = box[2] - box[0]
            self.draw.text(
                (self._scale(center_x) - text_width // 2, y_position),
                line_value,
                fill=ImageColor.getrgb(color),
                font=font,
            )
            y_position += line_height

    def write(self, output_path: Path) -> None:
        self.image.save(output_path, dpi=(600, 600), compress_level=6)


def draw_common(canvas: SvgCanvas | PngCanvas) -> None:
    canvas.rounded_rect(760, 35, 2080, 770, "#FFFFFF", BLUE, stroke_width=10, radius=32)
    canvas.text(1800, 88, ["2× Transformer encoder"], 54, color=BLUE, bold=True)
    canvas.line(965, 250, 965, 655, stroke=BLACK, stroke_width=7)
    canvas.arrow(965, 255, 1110, 255, stroke=BLACK, stroke_width=7)
    canvas.arrow(965, 465, 1110, 465, stroke=BLACK, stroke_width=7)
    canvas.rounded_rect(1110, 205, 1230, 92, YELLOW, BLACK, stroke_width=7, radius=14)
    canvas.rounded_rect(1110, 325, 1230, 75, GRAY_LIGHT, GRAY, stroke_width=6, radius=12)
    canvas.rounded_rect(1110, 435, 1230, 92, YELLOW, BLACK, stroke_width=7, radius=14)
    canvas.rounded_rect(1110, 555, 1230, 75, GRAY_LIGHT, GRAY, stroke_width=6, radius=12)
    canvas.text(1725, 251, ["Multi-head self-attention"], 46, bold=True)
    canvas.text(1725, 363, ["Add & Post-LN"], 40)
    canvas.text(1725, 481, ["Feed-forward network"], 46, bold=True)
    canvas.text(1725, 593, ["Add & Post-LN"], 40)
    canvas.arrow(2340, 250, 2480, 250, stroke=BLACK, stroke_width=7)
    canvas.arrow(2340, 480, 2480, 480, stroke=BLACK, stroke_width=7)
    canvas.circle(2530, 250, 38, "#FFFFFF", BLACK, stroke_width=6)
    canvas.circle(2530, 480, 38, "#FFFFFF", BLACK, stroke_width=6)
    canvas.text(2530, 252, ["+"], 46, bold=True)
    canvas.text(2530, 482, ["+"], 46, bold=True)
    canvas.arrow(1800, 805, 1800, 860, stroke=BLACK, stroke_width=8)
    canvas.rounded_rect(1110, 860, 1380, 90, PURPLE, "#8B4BA0", stroke_width=7, radius=12)
    canvas.text(1800, 905, ["Attention pooling"], 44, bold=True)

    canvas.rounded_rect(140, 1080, 910, 350, GREEN, GREEN_STROKE, stroke_width=8, radius=28)
    canvas.rounded_rect(1170, 1080, 1260, 350, ORANGE, ORANGE_STROKE, stroke_width=8, radius=28)
    canvas.rounded_rect(2550, 1080, 910, 350, BLUE_LIGHT, "#3657B5", stroke_width=8, radius=28)
    canvas.text(595, 1175, ["Attention expert", "(Transformer)"], 46, bold=True)
    canvas.text(595, 1340, ["y_attn"], 56, color=GREEN_STROKE, bold=True, italic=True)
    canvas.text(1800, 1170, ["Sample-adaptive", "residual gate"], 44, bold=True)
    canvas.text(1800, 1285, ["g_res"], 56, color=ORANGE_STROKE, bold=True, italic=True)
    canvas.text(1800, 1370, ["context + density/energy", "+ expert gap"], 30, color=GRAY)
    canvas.text(3005, 1175, ["MLP expert"], 48, color=BLUE, bold=True)
    canvas.text(3005, 1340, ["y_MLP"], 56, color=BLUE, bold=True, italic=True)
    canvas.arrow(1800, 950, 595, 1080, stroke=BLACK, stroke_width=7)
    canvas.line(240, 1035, 3360, 1035, stroke=GRAY, stroke_width=6, dashed=True)
    canvas.arrow(595, 1035, 595, 1080, stroke=GRAY, stroke_width=6, dashed=True)
    canvas.arrow(1800, 1035, 1800, 1080, stroke=GRAY, stroke_width=6, dashed=True)
    canvas.arrow(3005, 1035, 3005, 1080, stroke=GRAY, stroke_width=6, dashed=True)
    canvas.text(1800, 990, ["shared fingerprint + pH/category inputs"], 28, color=GRAY)
    canvas.arrow(1050, 1255, 1170, 1255, stroke=BLACK, stroke_width=8)
    canvas.arrow(2550, 1255, 2430, 1255, stroke=BLACK, stroke_width=8)
    canvas.arrow(1800, 1430, 1800, 1610, stroke=BLACK, stroke_width=8)

    canvas.rounded_rect(1090, 1620, 1420, 180, PINK, PINK_STROKE, stroke_width=8, radius=24)
    canvas.text(1800, 1680, ["Neural correction head"], 46, color=PINK_STROKE, bold=True)
    canvas.text(1800, 1755, ["Δ_NN"], 48, color=PINK_STROKE, bold=True, italic=True)
    canvas.arrow(1800, 1800, 1800, 1870, stroke=BLACK, stroke_width=8)
    canvas.rounded_rect(650, 1870, 2300, 170, "#FFFFFF", BLACK, stroke_width=8, radius=18)
    canvas.text(1800, 1955, ["y_NN = y_attn + g_res (y_MLP − y_attn) + Δ_NN"], 50, bold=True, italic=True)
    canvas.arrow(1800, 2040, 1800, 2110, stroke=BLACK, stroke_width=8)
    canvas.rounded_rect(850, 2110, 1900, 170, RF_FILL, RF_STROKE, stroke_width=8, radius=22)
    canvas.text(1800, 2165, ["RF residual correction (validation-selected)"], 44, color=RF_STROKE, bold=True)
    canvas.text(1800, 2230, ["fingerprints + pH/category + y_NN  →  Δ_RF"], 32, color=GRAY)
    canvas.arrow(1800, 2280, 1800, 2330, stroke=BLACK, stroke_width=8)
    canvas.rounded_rect(650, 2330, 2300, 160, "#FFFFFF", BLACK, stroke_width=8, radius=18)
    canvas.text(1800, 2410, ["y_final = y_NN + Δ_RF"], 56, bold=True, italic=True)

    canvas.rounded_rect(45, 1630, 500, 450, PRIOR_FILL, PRIOR_STROKE, stroke_width=8, radius=24, dashed=True)
    canvas.text(295, 1715, ["pH-response prior"], 42, color=PRIOR_STROKE, bold=True)
    canvas.text(295, 1815, ["training-only", "L_phys on attention path"], 34, color=PRIOR_STROKE, italic=True)
    canvas.arrow(545, 1725, 595, 1430, stroke=PRIOR_STROKE, stroke_width=7, dashed=True)


def main() -> None:
    apply_figure_style()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    svg_canvas = SvgCanvas()
    draw_common(svg_canvas)
    svg_canvas.write(OUTPUT_DIRECTORY / SVG_FILENAME)
    png_canvas = PngCanvas()
    draw_common(png_canvas)
    png_canvas.write(OUTPUT_DIRECTORY / PNG_FILENAME)
    print(f"[OK] Wrote SVG: {OUTPUT_DIRECTORY / SVG_FILENAME}")
    print(f"[OK] Wrote 600-dpi PNG: {OUTPUT_DIRECTORY / PNG_FILENAME}")


if __name__ == "__main__":
    main()
