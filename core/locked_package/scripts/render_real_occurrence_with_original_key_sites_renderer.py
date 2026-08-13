"""Render locked representative environments with the recovered legacy key-sites renderer."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(
    os.environ.get(
        "DECAT_ASSET_ROOT",
        str(_PACKAGE_ROOT / "artifacts" / "legacy_rendering"),
    )
)
REFERENCE_SOURCE_ROOT = Path(
    os.environ.get(
        "DECAT_REFERENCE_SOURCE_ROOT",
        str(_PACKAGE_ROOT / "artifacts" / "recovered_original_key_sites_runtime"),
    )
)
SOURCE_ROOT = Path(
    os.environ.get(
        "DECAT_RENDERER_SOURCE_ROOT",
        str(ASSET_ROOT / "recovered_original_key_sites_runtime"),
    )
)
DEFAULT_CSV = ASSET_ROOT / "real_occurrence_top20_rebuild" / "locked_top20_real_occurrence_representatives.csv"
DEFAULT_OUTPUT = ASSET_ROOT / "real_occurrence_key_sites_original_renderer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use the recovered original key-sites renderer.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=700)
    return parser.parse_args()


def load_original_renderer():
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Recovered original renderer runtime is missing: {SOURCE_ROOT}")
    source_path = str(SOURCE_ROOT)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    import export_v9_top20_atom_group_single_images as original_export

    original_export._add_paths()
    return original_export


def direction_symbol(direction: object) -> str:
    if str(direction).strip().lower() == "positive":
        return "↑"
    if str(direction).strip().lower() == "negative":
        return "↓"
    return "?"


def original_renderer_row(source_row: dict[str, object]) -> dict[str, object]:
    row = dict(source_row)
    row["rank"] = int(row["display_rank"])
    row["member_bits"] = f"fp_{int(row['bit'])}"
    row["representative_substructure"] = str(row["display_smiles"])
    row["direction_symbol"] = direction_symbol(row.get("direction_pred", ""))
    row["family_name"] = str(row.get("display_label", ""))
    row["family_display"] = str(row.get("display_label", ""))
    row["functional_display"] = str(row.get("functional_display", ""))
    return row


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
            "Rendered through the recovered original export_v9_top20_atom_group_single_images.py _render_key_site() path.\n"
            "The original transformer_v6.artifacts._draw_mol_paper_style() function is used unchanged.\n"
            "Each structure is a representative local molecular environment in which the locked bit occurs in the locked test split.\n"
            "Ranking, directions, and model contribution records remain unchanged.\n"
        )
    with (manifest_dir / "renderer_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "original_source_root": str(SOURCE_ROOT),
                "reference_source_root": str(REFERENCE_SOURCE_ROOT),
                "entry_function": "export_v9_top20_atom_group_single_images._render_key_site",
                "molecule_renderer": "transformer_v6.artifacts._draw_mol_paper_style",
                "highlight_selector": "render_top20_atom_group_prediction_impact_gallery._ozone_active_site_atoms",
            },
            handle,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    original_export = load_original_renderer()
    frame = pd.read_csv(args.csv).sort_values("display_rank").head(args.top_n)
    if args.preview_only:
        frame = frame.head(1)
    if frame.empty:
        raise RuntimeError(f"No representative rows found in {args.csv}")
    key_site_dir = args.output_dir / "key_sites"
    manifest_rows: list[dict[str, object]] = []
    for _, series in frame.iterrows():
        source_row = dict(series)
        row = original_renderer_row(source_row)
        rank = int(row["rank"])
        bit = int(source_row["bit"])
        stem = key_site_dir / f"TOP{rank:02d}_fp{bit}_{original_export._safe_slug(source_row.get('display_label', 'environment'))}"
        _, key_atoms = original_export._render_key_site(row, stem, dpi=int(args.dpi), png_only=False)
        source_row["key_site_atoms"] = ";".join(str(atom_idx) for atom_idx in key_atoms)
        manifest_rows.append(source_row)
        print(f"[OK] TOP{rank:02d} fp{bit}: key sites {key_atoms}")
    write_manifest(manifest_rows, args.output_dir)
    print(f"[OK] Original renderer output: {args.output_dir}")


if __name__ == "__main__":
    main()
