"""Recover displayable molecular environments for the locked Top-20 Morgan bits.

The predictive-impact scores are read unchanged from the locked Top-20 CSV.
Only the diagram representative is rebuilt from actual molecules containing
each bit on the locked test split.  This avoids rendering raw hashed-query
fragments such as ``[C][C][C]`` or ``OS`` as literal chemical structures.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = Path(
    os.environ.get(
        "DECAT_ASSET_ROOT",
        str(PACKAGE / "artifacts" / "legacy_rendering"),
    )
)
DATA = PACKAGE / "data" / "反应logk指纹数据_25类_含环境无机物_Other_1626条.csv"
SPLIT = PACKAGE / "data" / "固定划分_1626_同分子异pH非均分_75-12.5-12.5.json"
INPUT = ROOT / "prediction_impact_top20_atom_groups.csv"
DEFAULT_OUTPUT = ROOT / "real_occurrence_top20_rebuild"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild Top-20 diagram structures from real Morgan-bit occurrences.")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=3147)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def atom_neighborhood(molecule: Chem.Mol, center: int, radius: int) -> list[int]:
    visited = {int(center)}
    frontier = {int(center)}
    for _ in range(int(radius)):
        next_frontier: set[int] = set()
        for atom_index in frontier:
            next_frontier.update(int(atom.GetIdx()) for atom in molecule.GetAtomWithIdx(atom_index).GetNeighbors())
        frontier = next_frontier - visited
        visited.update(frontier)
        if not frontier:
            break
    return sorted(visited)


def local_environment_smiles(molecule: Chem.Mol, center: int, bit_radius: int) -> str:
    # A display radius of two preserves a real, chemically valid neighbourhood
    # even when the hashed bit itself comes from radius zero or one.
    atom_indices = atom_neighborhood(molecule, center, max(2, int(bit_radius)))
    smiles = Chem.MolFragmentToSmiles(
        molecule,
        atomsToUse=atom_indices,
        canonical=True,
        isomericSmiles=True,
    )
    fragment = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True) if fragment is not None else ""


def valid_display_fragment(smiles: str) -> tuple[bool, dict[str, int | bool]]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return False, {}
    heavy_atoms = int(molecule.GetNumHeavyAtoms())
    radicals = int(sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()))
    metals = int(sum(atom.GetAtomicNum() > 20 and atom.GetAtomicNum() not in {35, 53} for atom in molecule.GetAtoms()))
    metrics = {
        "heavy_atoms": heavy_atoms,
        "hetero_atoms": int(sum(atom.GetAtomicNum() not in {1, 6} for atom in molecule.GetAtoms())),
        "unsaturated": bool(any(bond.GetBondTypeAsDouble() > 1.1 for bond in molecule.GetBonds())),
        "aromatic": bool(any(atom.GetIsAromatic() for atom in molecule.GetAtoms())),
        "radicals": radicals,
        "metals": metals,
    }
    return 3 <= heavy_atoms <= 18 and radicals == 0 and metals == 0, metrics


def descriptor_label(smiles: str, metrics: dict[str, int | bool]) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "Representative local environment"
    atom_numbers = {atom.GetAtomicNum() for atom in molecule.GetAtoms()}
    if 16 in atom_numbers:
        return "Sulfur-containing local environment"
    if 7 in atom_numbers and 8 in atom_numbers and any(
        {bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()} == {6, 8}
        and bond.GetBondTypeAsDouble() > 1.1
        for bond in molecule.GetBonds()
    ):
        return "N/O carbonyl local environment"
    if 8 in atom_numbers and any(
        {bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()} == {6, 8}
        and bond.GetBondTypeAsDouble() > 1.1
        for bond in molecule.GetBonds()
    ):
        return "Oxygenated carbonyl local environment"
    if 7 in atom_numbers:
        return "Nitrogen-containing local environment"
    if 8 in atom_numbers:
        return "Oxygenated local environment"
    if bool(metrics.get("aromatic")):
        return "Aromatic local environment"
    if bool(metrics.get("unsaturated")):
        return "Unsaturated carbon local environment"
    return "Aliphatic carbon local environment"


def main() -> None:
    args = parse_args()
    ranked = pd.read_csv(args.input, encoding="utf-8-sig").sort_values("consensus_score", ascending=False, kind="stable")
    ranked = ranked.head(int(args.top_n)).copy()
    bits = {int(bit) for bit in ranked["bit"].tolist()}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    test_indices = {int(index) for index in split["test_idx"]}
    data = pd.read_csv(args.data, encoding="utf-8-sig")

    occurrences: dict[int, list[dict]] = defaultdict(list)
    for data_index in sorted(test_indices):
        source = data.iloc[data_index]
        smiles = str(source["SMILES"])
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        bit_info: dict[int, list[tuple[int, int]]] = {}
        AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            int(args.radius),
            nBits=int(args.n_bits),
            bitInfo=bit_info,
        )
        for bit in bits.intersection(bit_info):
            for center, bit_radius in bit_info[bit]:
                display_smiles = local_environment_smiles(molecule, int(center), int(bit_radius))
                valid, metrics = valid_display_fragment(display_smiles)
                occurrences[bit].append(
                    {
                        "source_row": int(data_index),
                        "source_name": str(source.get("chemical compound", "")),
                        "source_smiles": smiles,
                        "center_atom": int(center),
                        "bit_radius": int(bit_radius),
                        "display_smiles": display_smiles,
                        "valid": bool(valid),
                        **metrics,
                    }
                )

    rebuilt_rows: list[dict] = []
    audit_rows: list[dict] = []
    for display_rank, (_, original) in enumerate(ranked.iterrows(), start=1):
        bit = int(original["bit"])
        candidates = occurrences.get(bit, [])
        candidate_counts = Counter(row["display_smiles"] for row in candidates if row["valid"])
        ranked_candidates: list[dict] = []
        for display_smiles, count in candidate_counts.items():
            candidate = next(row for row in candidates if row["display_smiles"] == display_smiles and row["valid"])
            ranked_candidates.append({**candidate, "occurrence_count": int(count)})
        ranked_candidates.sort(
            key=lambda row: (
                int(row["hetero_atoms"]) > 0,
                bool(row["unsaturated"]) or bool(row["aromatic"]),
                int(row["occurrence_count"]),
                int(row["hetero_atoms"]),
                int(row["heavy_atoms"]),
            ),
            reverse=True,
        )
        selected = ranked_candidates[0] if ranked_candidates else None
        audit_rows.extend({"bit": bit, **candidate} for candidate in candidates)
        rebuilt = dict(original)
        rebuilt["display_rank"] = display_rank
        rebuilt["raw_hashed_fragment"] = str(original["canonical_substructure"])
        rebuilt["display_selection_protocol"] = "locked-test real occurrence; two-hop atom neighbourhood; valid 3-18 heavy atoms"
        if selected is None:
            rebuilt.update(
                {
                    "display_smiles": "",
                    "display_label": "No valid real-occurrence environment",
                    "display_source_name": "",
                    "display_source_smiles": "",
                    "display_center_atom": "",
                    "display_bit_radius": "",
                    "display_occurrence_count": 0,
                    "display_status": "excluded_no_valid_environment",
                }
            )
        else:
            rebuilt.update(
                {
                    "display_smiles": selected["display_smiles"],
                    "display_label": descriptor_label(selected["display_smiles"], selected),
                    "display_source_name": selected["source_name"],
                    "display_source_smiles": selected["source_smiles"],
                    "display_center_atom": selected["center_atom"],
                    "display_bit_radius": selected["bit_radius"],
                    "display_occurrence_count": selected["occurrence_count"],
                    "display_status": "valid_real_occurrence",
                }
            )
        rebuilt_rows.append(rebuilt)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rebuilt_rows).to_csv(output / "locked_top20_real_occurrence_representatives.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(output / "locked_top20_real_occurrence_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "source_ranking": str(args.input),
        "data": str(args.data),
        "split": str(args.split),
        "explanation_split": "locked test split",
        "morgan_radius": int(args.radius),
        "morgan_n_bits": int(args.n_bits),
        "top_n": len(rebuilt_rows),
        "valid_display_rows": int(sum(row["display_status"] == "valid_real_occurrence" for row in rebuilt_rows)),
        "note": "Predictive-impact ranking, signed deltas, and locked model performance were not recalculated or altered.",
    }
    (output / "rebuild_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
