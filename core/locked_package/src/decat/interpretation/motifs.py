"""Fingerprint motif mapping and functional-group interpretation."""

from __future__ import annotations
import glob
import os
from collections import Counter
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, rdFMCS
import decat.chemistry as _chemistry
import decat.config as _config
import decat.data.dataset as _data_dataset
import decat.interpretation.attribution as _interpretation_attribution
import decat.runtime as _runtime


def _aggregate_functional_group_impacts(
    *,
    bit_rows: list,
    dataset: _data_dataset.FingerprintReactionDataset,
    sample_indices: np.ndarray,
    top_n: int = 10,
) -> list:
    if not bit_rows:
        return []
    sample_idx = np.asarray(sample_indices, dtype=int).reshape(-1)
    sample_idx = sample_idx[(sample_idx >= 0) & (sample_idx < len(dataset))]
    if sample_idx.size == 0:
        return []
    fp_np = np.asarray(
        getattr(dataset, "fingerprint", np.zeros((0, 0), dtype=np.float32)),
        dtype=np.float32,
    )
    y_np = np.asarray(
        getattr(dataset, "logk_raw", np.zeros((0, 1), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    if fp_np.ndim != 2 or fp_np.shape[0] == 0 or y_np.size == 0:
        return []
    group_map: Dict[str, Dict[str, object]] = {}
    for row in bit_rows:
        fg_name = str(row.get("functional_group", "")).strip()
        if not fg_name:
            continue
        item = group_map.setdefault(
            fg_name,
            {
                "functional_group": fg_name,
                "functional_display": str(row.get("functional_display", "")).strip(),
                "smarts": "",
                "bit_names": set(),
                "bits": set(),
                "substructures": Counter(),
                "functional_score_sum": 0.0,
                "attention_score_sum": 0.0,
                "ig_score_sum": 0.0,
                "perturb_score_sum": 0.0,
                "stability_score_sum": 0.0,
                "consensus_score_sum": 0.0,
                "support_count_sum": 0,
                "mol_count_sum": 0,
                "occ_count_sum": 0,
                "row_count": 0,
            },
        )
        fg_def = _chemistry._functional_group_def_by_name(fg_name)
        if fg_def is not None and (not item.get("smarts")):
            item["smarts"] = str(fg_def.smarts)
        if str(row.get("functional_display", "")).strip() and (
            not item.get("functional_display")
        ):
            item["functional_display"] = str(row.get("functional_display", "")).strip()
        bit_name = str(row.get("bit_name", "")).strip()
        bit_idx = _parse_fp_bit_index(bit_name)
        if bit_name:
            item["bit_names"].add(bit_name)
        if bit_idx is not None:
            item["bits"].add(int(bit_idx))
        sub = str(row.get("substructure_smiles", "")).strip()
        if sub and (not sub.startswith("Unknown")):
            item["substructures"][sub] += int(
                max(1, int(row.get("substructure_count", 0)))
            )
        evidence_score = float(row.get("consensus_score", 0.0))
        if evidence_score <= 0.0:
            evidence_score = (
                0.35 * float(row.get("attention_score", 0.0))
                + 0.25 * float(row.get("ig_score", 0.0))
                + 0.25 * float(row.get("perturb_score", 0.0))
                + 0.15 * float(row.get("stability_score", 0.0))
            )
        item["functional_score_sum"] += float(max(0.0, evidence_score))
        item["attention_score_sum"] += float(row.get("attention_score", 0.0))
        item["ig_score_sum"] += float(row.get("ig_score", 0.0))
        item["perturb_score_sum"] += float(row.get("perturb_score", 0.0))
        item["stability_score_sum"] += float(row.get("stability_score", 0.0))
        item["consensus_score_sum"] += float(row.get("consensus_score", 0.0))
        item["support_count_sum"] += int(row.get("support_count", 0))
        item["mol_count_sum"] += int(row.get("mol_count", 0))
        item["occ_count_sum"] += int(row.get("occ_count", 0))
        item["row_count"] += 1
    rows = []
    for fg_name, item in group_map.items():
        bits = sorted(
            (int(b) for b in item.get("bits", set()) if 0 <= int(b) < fp_np.shape[1])
        )
        if not bits:
            continue
        active_mask = (
            np.abs(fp_np[np.asarray(sample_idx, dtype=int)][:, bits]) > 1e-06
        ).any(axis=1)
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if active_mask.size != sample_idx.size:
            continue
        active_y = y_np[sample_idx][active_mask]
        inactive_y = y_np[sample_idx][~active_mask]
        delta_logk = (
            float(np.nanmean(active_y) - np.nanmean(inactive_y))
            if active_y.size > 0 and inactive_y.size > 0
            else 0.0
        )
        active_mean = float(np.nanmean(active_y)) if active_y.size > 0 else float("nan")
        inactive_mean = (
            float(np.nanmean(inactive_y)) if inactive_y.size > 0 else float("nan")
        )
        support_n = int(active_mask.sum())
        direction = (
            "positive"
            if delta_logk > 1e-06
            else "negative" if delta_logk < -1e-06 else "neutral"
        )
        examples = [
            str(smi) for smi, _ in item.get("substructures", Counter()).most_common(3)
        ]
        rows.append(
            {
                "functional_group": str(fg_name),
                "functional_display": str(item.get("functional_display", "")).strip(),
                "smarts": str(item.get("smarts", "")).strip(),
                "bit_count": int(len(item.get("bits", set()))),
                "bit_names": "; ".join(
                    sorted((str(x) for x in item.get("bit_names", set())))
                ),
                "evidence_rows": int(item.get("row_count", 0)),
                "support_molecules": int(support_n),
                "support_frac": float(support_n / max(sample_idx.size, 1)),
                "active_mean_logk": active_mean,
                "inactive_mean_logk": inactive_mean,
                "delta_logk": float(delta_logk),
                "direction": str(direction),
                "functional_score_sum": float(item.get("functional_score_sum", 0.0)),
                "attention_score_sum": float(item.get("attention_score_sum", 0.0)),
                "ig_score_sum": float(item.get("ig_score_sum", 0.0)),
                "perturb_score_sum": float(item.get("perturb_score_sum", 0.0)),
                "stability_score_sum": float(item.get("stability_score_sum", 0.0)),
                "consensus_score_sum": float(item.get("consensus_score_sum", 0.0)),
                "support_count_sum": int(item.get("support_count_sum", 0)),
                "mol_count_sum": int(item.get("mol_count_sum", 0)),
                "occ_count_sum": int(item.get("occ_count_sum", 0)),
                "example_substructures": " | ".join(examples),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row.get("functional_score_sum", 0.0)),
            abs(float(row.get("delta_logk", 0.0))),
            int(row.get("support_molecules", 0)),
            int(row.get("bit_count", 0)),
        ),
        reverse=True,
    )
    return rows[: int(max(1, top_n))]


def _load_first_matching_csv(pattern: str) -> Optional[pd.DataFrame]:
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    try:
        return pd.read_csv(matches[0])
    except Exception:
        return None


def _cleanup_legacy_functional_group_outputs(
    output_dir: str, keep_top_n: int = 10
) -> None:
    if not output_dir or not os.path.isdir(output_dir):
        return
    patterns = [
        "consensus_top*_functional_groups.png",
        "consensus_top*_functional_groups.pdf",
        "consensus_top*_functional_groups_detailed.csv",
        "consensus_top*_functional_group_structures.png",
        "consensus_top*_functional_group_structures.pdf",
        "consensus_top*_functional_group_structures.csv",
        "attn_top*_functional_groups.png",
        "attn_top*_functional_groups.pdf",
        "attn_top*_functional_groups_detailed.csv",
    ]
    keep_tokens = {
        f"top{int(keep_top_n)}_functional_groups",
        f"top{int(keep_top_n)}_functional_group_structures",
    }
    for pattern in patterns:
        for path in glob.glob(os.path.join(output_dir, pattern)):
            name = os.path.basename(path)
            if any((token in name for token in keep_tokens)):
                continue
            try:
                os.remove(path)
            except Exception:
                pass


def _collect_functional_group_structure_rows(output_dir: str, top_n: int = 10) -> list:
    rows = []
    seen = set()
    consensus_df = _load_first_matching_csv(
        os.path.join(output_dir, "consensus_top*_functional_groups_detailed.csv")
    )
    attn_df = _load_first_matching_csv(
        os.path.join(output_dir, "attn_top*_functional_groups_detailed.csv")
    )
    fallback_frames = [
        _load_first_matching_csv(
            os.path.join(output_dir, "consensus_top*_functional_groups.csv")
        ),
        _load_first_matching_csv(
            os.path.join(output_dir, "attn_top*_functional_groups.csv")
        ),
    ]

    def _append_from_df(
        df: Optional[pd.DataFrame], source_tag: str, sort_cols: list
    ) -> None:
        nonlocal rows, seen
        if df is None or df.empty:
            return
        work = df.copy()
        for col in sort_cols:
            if col not in work.columns:
                work[col] = 0.0
        work = work.sort_values(sort_cols, ascending=False)
        for _, r in work.iterrows():
            fg_name = str(r.get("functional_group", "")).strip()
            if not fg_name or fg_name in seen:
                continue
            seen.add(fg_name)
            rows.append(
                {
                    "functional_group": fg_name,
                    "functional_display": str(r.get("functional_display", "")).strip(),
                    "smarts": str(r.get("smarts", "")).strip(),
                    "delta_logk": float(
                        pd.to_numeric(r.get("delta_logk", 0.0), errors="coerce")
                        if hasattr(pd, "to_numeric")
                        else r.get("delta_logk", 0.0)
                    ),
                    "support_molecules": int(
                        pd.to_numeric(
                            r.get("support_molecules", r.get("mol_count", 0)),
                            errors="coerce",
                        )
                        if hasattr(pd, "to_numeric")
                        else r.get("support_molecules", r.get("mol_count", 0))
                    ),
                    "bit_count": int(
                        pd.to_numeric(r.get("bit_count", 0), errors="coerce")
                        if hasattr(pd, "to_numeric")
                        else r.get("bit_count", 0)
                    ),
                    "functional_score_sum": float(
                        pd.to_numeric(
                            r.get(
                                "functional_score_sum",
                                r.get("consensus_score_sum", r.get("count", 0.0)),
                            ),
                            errors="coerce",
                        )
                        if hasattr(pd, "to_numeric")
                        else r.get(
                            "functional_score_sum",
                            r.get("consensus_score_sum", r.get("count", 0.0)),
                        )
                    ),
                    "example_substructures": str(
                        r.get("example_substructures", "")
                    ).strip(),
                    "source": source_tag,
                }
            )
            if len(rows) >= int(top_n):
                return

    _append_from_df(
        consensus_df, "consensus", ["functional_score_sum", "support_molecules"]
    )
    if len(rows) < int(top_n):
        _append_from_df(
            attn_df, "attention", ["functional_score_sum", "support_molecules"]
        )
    if len(rows) < int(top_n):
        for idx, df in enumerate(fallback_frames):
            _append_from_df(df, f"fallback_{idx + 1}", ["count", "mol_count"])
            if len(rows) >= int(top_n):
                break
    return rows[: int(max(1, top_n))]


def _supplement_functional_group_rows(
    base_rows: list, output_dir: str, top_n: int = 10
) -> list:
    rows = [dict(r) for r in list(base_rows or [])]
    seen = {
        str(r.get("functional_group", "")).strip()
        for r in rows
        if str(r.get("functional_group", "")).strip()
    }
    if len(rows) >= int(top_n):
        return rows[: int(top_n)]
    attn_df = _load_first_matching_csv(
        os.path.join(output_dir, "attn_top*_functional_groups_detailed.csv")
    )
    if attn_df is None or attn_df.empty:
        attn_df = _load_first_matching_csv(
            os.path.join(output_dir, "attn_top*_functional_groups.csv")
        )
    if attn_df is None or attn_df.empty:
        return rows[: int(top_n)]
    work = attn_df.copy()
    for col in ["functional_score_sum", "support_molecules", "delta_logk", "bit_count"]:
        if col not in work.columns:
            if col == "support_molecules":
                work[col] = pd.to_numeric(
                    work.get("mol_count", 0), errors="coerce"
                ).fillna(0)
            else:
                work[col] = 0.0
    work = work.sort_values(
        ["functional_score_sum", "support_molecules"], ascending=False
    )
    for _, r in work.iterrows():
        fg_name = str(r.get("functional_group", "")).strip()
        if not fg_name or fg_name in seen:
            continue
        seen.add(fg_name)
        rows.append(
            {
                "functional_group": fg_name,
                "functional_display": str(r.get("functional_display", "")).strip(),
                "smarts": str(r.get("smarts", "")).strip(),
                "delta_logk": float(
                    pd.to_numeric(r.get("delta_logk", 0.0), errors="coerce")
                ),
                "support_molecules": int(
                    pd.to_numeric(
                        r.get("support_molecules", r.get("mol_count", 0)),
                        errors="coerce",
                    )
                ),
                "bit_count": int(pd.to_numeric(r.get("bit_count", 0), errors="coerce")),
                "functional_score_sum": float(
                    pd.to_numeric(
                        r.get("functional_score_sum", r.get("count", 0.0)),
                        errors="coerce",
                    )
                ),
                "consensus_score_sum": float(
                    pd.to_numeric(
                        r.get("functional_score_sum", r.get("count", 0.0)),
                        errors="coerce",
                    )
                ),
                "example_substructures": str(
                    r.get("example_substructures", "")
                ).strip(),
                "source": "attention",
            }
        )
        if len(rows) >= int(top_n):
            break
    if len(rows) < int(top_n):
        template_priority = [
            "Phenol",
            "Ester",
            "Thiol",
            "Carboxylate",
            "Sulfonic acid",
            "Phosphonic acid",
            "Secondary amine",
            "Ketone",
            "Alcohol",
            "Ether",
        ]
        for name in template_priority:
            if len(rows) >= int(top_n):
                break
            if name in seen:
                continue
            fg_def = _chemistry._functional_group_def_by_name(name)
            rows.append(
                {
                    "functional_group": str(name),
                    "functional_display": str(getattr(fg_def, "display", "")),
                    "smarts": str(getattr(fg_def, "smarts", "")),
                    "delta_logk": 0.0,
                    "support_molecules": 0,
                    "bit_count": 0,
                    "functional_score_sum": 0.0,
                    "consensus_score_sum": 0.0,
                    "example_substructures": str(
                        _chemistry.FG_REPRESENTATIVE_SMILES.get(name, "")
                    ),
                    "source": "template",
                }
            )
            seen.add(name)
    return rows[: int(top_n)]


def _pick_functional_group_structure(
    functional_group: str, example_substructures: str = "", smarts: str = ""
):
    fg_name = str(functional_group).strip()
    fg_def = _chemistry._functional_group_def_by_name(fg_name)
    pat = fg_def.pattern if fg_def is not None else None
    candidates = []
    rep = str(_chemistry.FG_REPRESENTATIVE_SMILES.get(fg_name, "")).strip()
    if rep:
        candidates.append(rep)
    for token in str(example_substructures).split("|"):
        tok = str(token).strip()
        if tok:
            candidates.append(tok)
    for smi in candidates:
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            mol = None
        if mol is None:
            continue
        highlight = []
        if pat is not None:
            try:
                matches = mol.GetSubstructMatches(pat, uniquify=True)
            except Exception:
                matches = []
            if matches:
                highlight = list(matches[0])
        return (mol, highlight, smi)
    if smarts:
        try:
            q = Chem.MolFromSmarts(smarts)
        except Exception:
            q = None
        if q is not None:
            return (q, [], smarts)
    return (None, [], rep or smarts)


def _substructure_quality_metrics(
    sub_smiles: str, min_heavy_atoms: int = 3
) -> Dict[str, object]:
    canonical = _chemistry._canonicalize_smiles_or_smarts(sub_smiles)
    metrics: Dict[str, object] = {
        "valid": False,
        "canonical": canonical,
        "heavy_atoms": 0,
        "hetero_non_halogen": 0,
        "halogen_atoms": 0,
        "carbon_atoms": 0,
        "aromatic_atoms": 0,
        "ring_count": 0,
        "unsat_bonds": 0,
        "disconnected": "." in str(canonical),
        "halogen_only": False,
        "all_aliphatic_carbon": False,
        "is_trivial": True,
        "fg": None,
    }
    if (
        not isinstance(canonical, str)
        or not canonical
        or canonical.startswith("Unknown")
    ):
        return metrics
    try:
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            mol = Chem.MolFromSmarts(canonical)
        if mol is None:
            return metrics
        atoms = list(mol.GetAtoms())
        heavy_atoms = int(mol.GetNumHeavyAtoms())
        halogen_atoms = int(
            sum((atom.GetAtomicNum() in {9, 17, 35, 53} for atom in atoms))
        )
        carbon_atoms = int(sum((atom.GetAtomicNum() == 6 for atom in atoms)))
        hetero_non_halogen = int(
            sum((atom.GetAtomicNum() not in {1, 6, 9, 17, 35, 53} for atom in atoms))
        )
        aromatic_atoms = int(sum((int(atom.GetIsAromatic()) for atom in atoms)))
        ring_info = mol.GetRingInfo()
        ring_count = int(ring_info.NumRings()) if ring_info is not None else 0
        unsat_bonds = int(
            sum(
                (
                    bond.GetBondTypeAsDouble() > 1.1 or bond.GetIsAromatic()
                    for bond in mol.GetBonds()
                )
            )
        )
        fg = _chemistry._detect_functional_group(canonical)
        halogen_only = heavy_atoms > 0 and heavy_atoms == halogen_atoms
        all_aliphatic_carbon = (
            heavy_atoms > 0
            and carbon_atoms == heavy_atoms
            and (aromatic_atoms == 0)
            and (hetero_non_halogen == 0)
            and (halogen_atoms == 0)
        )
        exact_trivial = canonical in _chemistry.TRIVIAL_SUBSTRUCTURE_EXACT
        is_trivial = bool(
            exact_trivial
            or metrics["disconnected"]
            or halogen_only
            or (heavy_atoms < int(min_heavy_atoms))
            or (
                all_aliphatic_carbon
                and heavy_atoms <= 4
                and (unsat_bonds == 0)
                and (ring_count == 0)
            )
            or (
                fg is None
                and hetero_non_halogen == 0
                and (halogen_atoms <= 1)
                and (heavy_atoms <= 2)
                and (aromatic_atoms == 0)
                and (unsat_bonds <= 1)
            )
        )
        metrics.update(
            {
                "valid": True,
                "canonical": canonical,
                "heavy_atoms": heavy_atoms,
                "hetero_non_halogen": hetero_non_halogen,
                "halogen_atoms": halogen_atoms,
                "carbon_atoms": carbon_atoms,
                "aromatic_atoms": aromatic_atoms,
                "ring_count": ring_count,
                "unsat_bonds": unsat_bonds,
                "halogen_only": halogen_only,
                "all_aliphatic_carbon": all_aliphatic_carbon,
                "is_trivial": is_trivial,
                "fg": fg,
            }
        )
        return metrics
    except Exception:
        return metrics


def _score_substructure_candidate(
    sub_smiles: str,
    sub_count: int,
    sub_frac: float,
    mol_count: int,
    min_heavy_atoms: int = 3,
) -> Dict[str, object]:
    quality = _substructure_quality_metrics(sub_smiles, min_heavy_atoms=min_heavy_atoms)
    fg = quality.get("fg")
    score = 0.0
    score += 8.0 if fg is not None else 0.0
    score += 1.35 * float(quality.get("heavy_atoms", 0))
    score += 1.25 * float(quality.get("hetero_non_halogen", 0))
    score += 0.55 * float(quality.get("ring_count", 0))
    score += 0.35 * float(quality.get("aromatic_atoms", 0))
    score += 0.6 * float(quality.get("unsat_bonds", 0))
    score += 3.2 * float(sub_frac)
    score += 0.45 * float(np.log1p(max(int(sub_count), 0)))
    score += 0.25 * float(np.log1p(max(int(mol_count), 0)))
    if bool(quality.get("is_trivial", True)):
        score -= 9.0
    if bool(quality.get("halogen_only", False)):
        score -= 6.0
    if (
        bool(quality.get("all_aliphatic_carbon", False))
        and int(quality.get("heavy_atoms", 0)) <= 4
    ):
        score -= 5.0
    return {
        "canonical_substructure": str(quality.get("canonical", sub_smiles)),
        "quality": quality,
        "functional_group": str(fg.get("fg_name", "")) if fg else "",
        "functional_display": str(fg.get("display", "")) if fg else "",
        "score": float(score),
        "is_reasonable": bool(
            quality.get("valid", False) and (not quality.get("is_trivial", True))
        ),
    }


def _choose_best_substructure_for_bit(
    rank: int,
    bit_name: str,
    bit: int,
    info: Dict[str, object],
    min_heavy_atoms: int = 3,
) -> Optional[Dict[str, object]]:
    rows = list(info.get("substructures", []) or [])
    if not rows:
        return None
    candidates = []
    for row in rows:
        sub = str(row.get("substructure_smiles", "Unknown"))
        cnt = int(row.get("substructure_count", 0))
        frac = float(row.get("substructure_frac", 0.0))
        scored = _score_substructure_candidate(
            sub_smiles=sub,
            sub_count=cnt,
            sub_frac=frac,
            mol_count=int(info.get("mol_count", 0)),
            min_heavy_atoms=min_heavy_atoms,
        )
        candidate = {
            "rank": int(rank),
            "bit": int(bit),
            "bit_name": str(bit_name),
            "mol_count": int(info.get("mol_count", 0)),
            "occ_count": int(info.get("occ_count", 0)),
            "substructure_smiles": sub,
            "canonical_substructure": str(scored["canonical_substructure"]),
            "substructure_count": cnt,
            "substructure_frac": frac,
            "score": float(scored["score"]),
            "functional_group": str(scored["functional_group"]),
            "functional_display": str(scored["functional_display"]),
            "is_reasonable": bool(scored["is_reasonable"]),
            "quality": dict(scored["quality"]),
        }
        candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            int(bool(row.get("is_reasonable", False))),
            float(row.get("score", float("-inf"))),
            float(row.get("substructure_frac", 0.0)),
            int(row.get("substructure_count", 0)),
            int(row.get("quality", {}).get("heavy_atoms", 0)),
        ),
        reverse=True,
    )
    return candidates[0]


def _select_unique_substructure_candidates(candidates: list, top_n: int) -> list:
    ordered = sorted(
        list(candidates),
        key=lambda row: (
            int(row.get("rank", 10**9)),
            -float(row.get("score", float("-inf"))),
            -int(row.get("quality", {}).get("heavy_atoms", 0)),
            -int(row.get("substructure_count", 0)),
        ),
    )
    selected = []
    deferred = []
    seen_sub = set()
    seen_fg = set()
    for row in ordered:
        canonical = str(row.get("canonical_substructure", ""))
        if (
            not canonical
            or canonical in seen_sub
            or (not bool(row.get("is_reasonable", False)))
        ):
            continue
        fg_name = str(row.get("functional_group", "")).strip()
        if fg_name and fg_name not in seen_fg:
            selected.append(row)
            seen_sub.add(canonical)
            seen_fg.add(fg_name)
            if len(selected) >= int(top_n):
                return selected
        else:
            deferred.append(row)
    for row in ordered:
        if len(selected) >= int(top_n):
            break
        canonical = str(row.get("canonical_substructure", ""))
        if (
            not canonical
            or canonical in seen_sub
            or (not bool(row.get("is_reasonable", False)))
        ):
            continue
        selected.append(row)
        seen_sub.add(canonical)
    if len(selected) < int(top_n):
        for row in deferred:
            if len(selected) >= int(top_n):
                break
            canonical = str(row.get("canonical_substructure", ""))
            quality = dict(row.get("quality", {}))
            if (
                not canonical
                or canonical in seen_sub
                or (not bool(quality.get("valid", False)))
            ):
                continue
            if bool(quality.get("is_trivial", True)):
                continue
            if bool(quality.get("halogen_only", False)):
                continue
            if int(quality.get("heavy_atoms", 0)) < 2:
                continue
            selected.append(row)
            seen_sub.add(canonical)
    return selected


def _parse_fp_bit_index(bit_name: str) -> Optional[int]:
    if not isinstance(bit_name, str):
        return None
    s = str(bit_name).strip()
    if not s.startswith("fp_"):
        return None
    try:
        return int(s.split("_", 1)[1])
    except Exception:
        return None


def _map_bits_to_substructures_cached(
    smiles_list: np.ndarray,
    fp_size: int,
    radius: int,
    bits: list,
    cache_tag: str = "",
    sample_indices: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, object]]:
    smiles_arr = np.asarray(smiles_list, dtype=object).reshape(-1)
    bit_list = sorted({int(b) for b in bits if b is not None})
    idx_arr = np.asarray(
        sample_indices if sample_indices is not None else np.arange(smiles_arr.size),
        dtype=int,
    ).reshape(-1)
    idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < smiles_arr.size)]
    cache_key = (
        str(cache_tag),
        int(fp_size),
        int(radius),
        tuple(bit_list),
        tuple(idx_arr.tolist()),
    )
    cached = _runtime.context.bit_substructure_cache.get(cache_key)
    if cached is not None:
        return cached
    counts_map = {b: Counter() for b in bit_list}
    raw_counts_map = {b: Counter() for b in bit_list}
    mol_count_map = {b: 0 for b in bit_list}
    occ_count_map = {b: 0 for b in bit_list}
    mols_with = {b: [] for b in bit_list}
    for idx in idx_arr.tolist():
        smi = str(smiles_arr[idx])
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            bit_info = {}
            _ = AllChem.GetMorganFingerprintAsBitVect(
                mol, int(radius), nBits=int(fp_size), bitInfo=bit_info
            )
            for b in bit_list:
                if b not in bit_info:
                    continue
                mol_count_map[b] += 1
                mols_with[b].append(mol)
                envs = bit_info[b]
                occ_count_map[b] += int(len(envs))
                for center, rad_local in envs:
                    frag = _chemistry._env_smiles_from_center(mol, center, rad_local)
                    if frag:
                        can = _chemistry._canonicalize_smiles_or_smarts(frag)
                        raw_counts_map[b][can] += 1
                        if _chemistry._is_informative_substructure(
                            can,
                            min_atoms=int(
                                _config._env_value(
                                    "DECAT_RUNTIME_ATTN_SUBSTRUCT_MIN_ATOMS", None, 2
                                )
                            ),
                        ):
                            counts_map[b][can] += 1
        except Exception:
            continue
    result: Dict[int, Dict[str, object]] = {}
    for b in bit_list:
        sub_rows = []
        if counts_map[b]:
            total = int(sum(counts_map[b].values()))
            for smi, cnt in counts_map[b].most_common():
                sub_rows.append(
                    {
                        "substructure_smiles": str(smi),
                        "substructure_count": int(cnt),
                        "substructure_frac": float(cnt / max(total, 1)),
                    }
                )
        else:
            rep = "Unknown"
            try:
                if len(mols_with[b]) >= 2:
                    mcs = rdFMCS.FindMCS(mols_with[b], timeout=5)
                    if mcs and getattr(mcs, "smartsString", ""):
                        rep = _chemistry._canonicalize_smiles_or_smarts(
                            str(mcs.smartsString)
                        )
            except Exception:
                pass
            if (
                not _chemistry._is_informative_substructure(
                    rep,
                    min_atoms=int(
                        _config._env_value(
                            "DECAT_RUNTIME_ATTN_SUBSTRUCT_MIN_ATOMS", None, 2
                        )
                    ),
                )
                and raw_counts_map[b]
            ):
                for smi, _ in raw_counts_map[b].most_common():
                    if _chemistry._is_informative_substructure(
                        smi,
                        min_atoms=int(
                            _config._env_value(
                                "DECAT_RUNTIME_ATTN_SUBSTRUCT_MIN_ATOMS", None, 2
                            )
                        ),
                    ):
                        rep = str(smi)
                        break
            if (
                not _chemistry._is_informative_substructure(
                    rep,
                    min_atoms=int(
                        _config._env_value(
                            "DECAT_RUNTIME_ATTN_SUBSTRUCT_MIN_ATOMS", None, 2
                        )
                    ),
                )
                and raw_counts_map[b]
            ):
                rep = str(raw_counts_map[b].most_common(1)[0][0])
            sub_rows.append(
                {
                    "substructure_smiles": _chemistry._canonicalize_smiles_or_smarts(
                        rep
                    ),
                    "substructure_count": int(max(mol_count_map[b], 1)),
                    "substructure_frac": 1.0 if mol_count_map[b] > 0 else 0.0,
                }
            )
        result[int(b)] = {
            "mol_count": int(mol_count_map[b]),
            "occ_count": int(occ_count_map[b]),
            "substructures": sub_rows,
        }
    _runtime.context.bit_substructure_cache[cache_key] = result
    return result


def export_attention_substructure_artifacts(
    dataset: _data_dataset.FingerprintReactionDataset,
    train_indices: np.ndarray,
    ranked_bit_names: list,
    prefix: str,
    artifact_tag: str = "attn",
    extra_bit_metrics: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    import decat.plotting.structures as _plotting_structures

    out_dir = _runtime.get_run_output_dir(prefix)
    train_idx = np.asarray(train_indices, dtype=int).reshape(-1)
    train_idx = train_idx[(train_idx >= 0) & (train_idx < len(dataset))]
    if train_idx.size == 0 or not ranked_bit_names:
        return {}
    top_n = int(
        max(
            5,
            min(
                int(_config._env_value("DECAT_RUNTIME_ATTN_SUBSTRUCT_TOPK", None, 30)),
                len(ranked_bit_names),
            ),
        )
    )
    functional_top_n = int(
        max(
            5,
            min(
                int(
                    _config._env_value("DECAT_RUNTIME_TOP_FUNCTIONAL_GROUPS", None, 10)
                ),
                top_n,
            ),
        )
    )
    unique_top_n = int(
        max(
            5,
            min(
                int(
                    _config._env_value(
                        "DECAT_RUNTIME_ATTN_SUBSTRUCT_UNIQUE_TOPK", None, 10
                    )
                ),
                top_n,
            ),
        )
    )
    polished_min_heavy = int(
        _config._env_value("DECAT_RUNTIME_ATTN_SUBSTRUCT_UNIQUE_MIN_HEAVY", None, 4)
    )
    bit_names = []
    bits = []
    seen_bits = set()
    for raw_name in ranked_bit_names:
        raw_name = str(raw_name)
        bit = _parse_fp_bit_index(raw_name)
        if bit is None or bit in seen_bits:
            continue
        seen_bits.add(bit)
        bit_names.append(raw_name)
        bits.append(int(bit))
        if len(bit_names) >= int(top_n):
            break
    if not bits:
        return {}
    sample_idx = (
        np.arange(len(dataset), dtype=int)
        if _config._env_bool("DECAT_RUNTIME_ATTN_SUBSTRUCT_USE_TRAINVAL", False)
        else train_idx
    )
    parsed = _map_bits_to_substructures_cached(
        smiles_list=getattr(dataset, "smiles", np.array([], dtype=object)),
        fp_size=int(getattr(dataset, "max_fp_bits", dataset.fingerprint_dim)),
        radius=int(_config._env_value("DECAT_RUNTIME_ATTN_MIN_RADIUS", None, 2)),
        bits=bits,
        cache_tag=str(getattr(dataset, "csv_path", _runtime.context.data_csv_path)),
        sample_indices=sample_idx,
    )
    if not parsed:
        return {}
    bit_rows = []
    sub_counter = Counter()
    fg_counter = Counter()
    fg_mol_counter = Counter()
    rep_candidates = []
    for rank, bit_name in enumerate(bit_names, start=1):
        bit = _parse_fp_bit_index(bit_name)
        if bit is None or bit not in parsed:
            continue
        info = parsed[bit]
        metric_info = dict((extra_bit_metrics or {}).get(str(bit_name), {}))
        for row in info["substructures"]:
            sub = str(row.get("substructure_smiles", "Unknown"))
            cnt = int(row.get("substructure_count", 0))
            frac = float(row.get("substructure_frac", 0.0))
            fg = _chemistry._detect_functional_group(sub)
            row_payload = {
                "rank": int(rank),
                "bit": int(bit),
                "bit_name": str(bit_name),
                "mol_count": int(info.get("mol_count", 0)),
                "occ_count": int(info.get("occ_count", 0)),
                "substructure_smiles": sub,
                "substructure_count": cnt,
                "substructure_frac": frac,
                "functional_group": str((fg or {}).get("fg_name", "")),
                "functional_display": str((fg or {}).get("display", "")),
                "functional_smarts": str((fg or {}).get("smarts", "")),
            }
            for key in (
                "attention_score",
                "ig_score",
                "perturb_score",
                "stability_score",
                "support_count",
                "support_frac",
                "consensus_score",
            ):
                if key in metric_info:
                    row_payload[key] = metric_info[key]
            bit_rows.append(row_payload)
            if sub and (not sub.startswith("Unknown")):
                sub_counter[sub] += cnt
                if fg is not None:
                    fg_name = str(fg.get("fg_name", "Unknown"))
                    fg_counter[fg_name] += cnt
                    fg_mol_counter[fg_name] += int(info.get("mol_count", 0))
        rep = _choose_best_substructure_for_bit(
            rank=rank,
            bit_name=bit_name,
            bit=int(bit),
            info=info,
            min_heavy_atoms=polished_min_heavy,
        )
        if rep is not None:
            for key, value in metric_info.items():
                rep[key] = value
            rep_candidates.append(rep)
    if bit_rows:
        pd.DataFrame(bit_rows).to_csv(
            os.path.join(out_dir, f"{artifact_tag}_bit_substructures.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    if sub_counter:
        pd.DataFrame(
            [
                {"substructure_smiles": str(smi), "count": int(cnt)}
                for smi, cnt in sub_counter.most_common(top_n)
            ]
        ).to_csv(
            os.path.join(out_dir, f"{artifact_tag}_top{top_n}_substructures.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    if fg_counter:
        fg_rows = []
        for fg_name, cnt in fg_counter.most_common():
            fg_def = next(
                (x for x in _chemistry.FG_DEFINITIONS if x.name == fg_name), None
            )
            fg_rows.append(
                {
                    "functional_group": str(fg_name),
                    "count": int(cnt),
                    "mol_count": int(fg_mol_counter.get(fg_name, 0)),
                    "smarts": str(fg_def.smarts) if fg_def is not None else "",
                }
            )
        pd.DataFrame(fg_rows).to_csv(
            os.path.join(out_dir, f"{artifact_tag}_top{top_n}_functional_groups.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    functional_rows = _aggregate_functional_group_impacts(
        bit_rows=bit_rows,
        dataset=dataset,
        sample_indices=sample_idx,
        top_n=functional_top_n,
    )
    if functional_rows:
        pd.DataFrame(functional_rows).to_csv(
            os.path.join(
                out_dir,
                f"{artifact_tag}_top{int(functional_top_n)}_functional_groups_detailed.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )
        _plotting_structures._plot_top_functional_groups(
            functional_rows, out_dir, artifact_tag=artifact_tag, top_n=functional_top_n
        )
        _cleanup_legacy_functional_group_outputs(
            out_dir, keep_top_n=int(functional_top_n)
        )
    selected_candidates = _select_unique_substructure_candidates(
        rep_candidates, unique_top_n
    )
    export_unique_atom_groups = _config._env_bool(
        "DECAT_RUNTIME_EXPORT_UNIQUE_ATOM_GROUPS", False
    )
    if selected_candidates and export_unique_atom_groups:
        selected_rows = []
        grid_mols = []
        grid_legends = []
        grid_highlights = []
        for row in selected_candidates:
            canonical = str(row.get("canonical_substructure", ""))
            mol = Chem.MolFromSmiles(canonical)
            if mol is None:
                mol = Chem.MolFromSmarts(canonical)
            if mol is None:
                continue
            fg = row.get("quality", {}).get("fg")
            display_text = str(row.get("functional_display", "")).strip()
            selected_rows.append(
                {
                    "rank": int(row.get("rank", 0)),
                    "bit": int(row.get("bit", 0)),
                    "bit_name": str(row.get("bit_name", "")),
                    "substructure_smiles": str(row.get("substructure_smiles", "")),
                    "canonical_substructure": canonical,
                    "functional_group": str(row.get("functional_group", "")),
                    "functional_display": display_text,
                    "score": float(row.get("score", 0.0)),
                    "mol_count": int(row.get("mol_count", 0)),
                    "occ_count": int(row.get("occ_count", 0)),
                    "substructure_count": int(row.get("substructure_count", 0)),
                    "substructure_frac": float(row.get("substructure_frac", 0.0)),
                    "heavy_atoms": int(row.get("quality", {}).get("heavy_atoms", 0)),
                    "hetero_non_halogen": int(
                        row.get("quality", {}).get("hetero_non_halogen", 0)
                    ),
                    "ring_count": int(row.get("quality", {}).get("ring_count", 0)),
                    "unsat_bonds": int(row.get("quality", {}).get("unsat_bonds", 0)),
                    "attention_score": float(row.get("attention_score", 0.0)),
                    "ig_score": float(row.get("ig_score", 0.0)),
                    "perturb_score": float(row.get("perturb_score", 0.0)),
                    "stability_score": float(row.get("stability_score", 0.0)),
                    "support_count": int(row.get("support_count", 0)),
                    "support_frac": float(row.get("support_frac", 0.0)),
                    "consensus_score": float(row.get("consensus_score", 0.0)),
                }
            )
            label_lines = [
                f"Rank {int(row.get('rank', 0))} | {str(row.get('bit_name', ''))}",
                canonical,
            ]
            if display_text:
                label_lines.append(display_text)
            grid_mols.append(mol)
            grid_legends.append("\n".join(label_lines))
            if isinstance(fg, dict):
                grid_highlights.append(list(fg.get("highlight_atoms", [])))
            else:
                grid_highlights.append([])
        if selected_rows:
            pd.DataFrame(selected_rows).to_csv(
                os.path.join(
                    out_dir,
                    f"{artifact_tag}_top{len(selected_rows)}_unique_atom_groups.csv",
                ),
                index=False,
                encoding="utf-8-sig",
            )
        if grid_mols:
            try:
                img = Draw.MolsToGridImage(
                    grid_mols,
                    molsPerRow=2,
                    subImgSize=(360, 280),
                    legends=grid_legends,
                    highlightAtomLists=grid_highlights,
                    legendFontSize=18,
                    useSVG=False,
                )
                img.save(
                    os.path.join(
                        out_dir,
                        f"{artifact_tag}_top{len(grid_mols)}_unique_atom_groups.png",
                    )
                )
            except Exception:
                pass
        return {
            "selected_rows": selected_rows,
            "bit_rows": bit_rows,
            "functional_rows": functional_rows,
        }
    return {
        "selected_rows": [],
        "bit_rows": bit_rows,
        "functional_rows": functional_rows,
    }


def _compute_consensus_rows_core(
    *,
    model: nn.Module,
    fingerprint_np: np.ndarray,
    numeric_np: np.ndarray,
    feature_labels: list,
    candidate_topk: int,
    final_topk: int,
    min_support: int,
    batch_size: int,
    ig_steps: int,
    stability_mode: str = "bootstrap",
    bootstrap_rounds: int = 32,
    bootstrap_frac: float = 0.75,
    stability_override: Optional[Dict[str, float]] = None,
    weight_attn: float = 0.2,
    weight_ig: float = 0.2,
    weight_delta: float = 0.3,
    weight_stab: float = 0.2,
    weight_support: float = 0.1,
) -> Tuple[list, np.ndarray, np.ndarray, list]:
    attn_mat = _interpretation_attribution._collect_attention_sample_matrix_from_arrays(
        model=model,
        fingerprint_np=fingerprint_np,
        numeric_np=numeric_np,
        feature_labels=feature_labels,
        batch_size=min(64, max(8, int(batch_size))),
    )
    ig_mat = _interpretation_attribution._collect_ig_sample_matrix_from_arrays(
        model=model,
        fingerprint_np=fingerprint_np,
        numeric_np=numeric_np,
        batch_size=min(32, max(4, int(batch_size))),
        steps=int(ig_steps),
    )
    active_mat = (np.abs(fingerprint_np) > 1e-06).astype(np.float32)
    support_count = active_mat.sum(axis=0).astype(np.float32)
    support_frac = support_count / max(float(fingerprint_np.shape[0]), 1.0)
    denom = np.where(support_count > 0, support_count, 1.0)
    attn_score = (attn_mat * active_mat).sum(axis=0) / denom
    ig_score = (ig_mat * active_mat).sum(axis=0) / denom
    support_gate = support_count >= float(min_support)
    attn_norm = _interpretation_attribution._minmax_scale_array(attn_score)
    ig_norm = _interpretation_attribution._minmax_scale_array(ig_score)
    support_norm = _interpretation_attribution._minmax_scale_array(support_frac)
    pre_score = 0.55 * attn_norm + 0.45 * ig_norm + 0.15 * support_norm
    pre_score = np.where(support_gate, pre_score, -np.inf)
    candidate_order = np.argsort(pre_score)[::-1]
    candidate_local_idx = [
        int(i) for i in candidate_order.tolist() if np.isfinite(pre_score[int(i)])
    ][: int(candidate_topk)]
    if not candidate_local_idx:
        return ([], active_mat, support_count, [])
    delta_map = _interpretation_attribution._compute_mask_delta_scores_from_arrays(
        model=model,
        fingerprint_np=fingerprint_np,
        numeric_np=numeric_np,
        candidate_local_idx=candidate_local_idx,
        batch_size=min(64, max(8, int(batch_size))),
    )
    combined_sample = 0.5 * _interpretation_attribution._row_normalize_nonnegative(
        attn_mat * active_mat
    ) + 0.5 * _interpretation_attribution._row_normalize_nonnegative(
        ig_mat * active_mat
    )
    bootstrap_map = _interpretation_attribution._compute_bootstrap_stability_scores(
        sample_score_matrix=combined_sample,
        candidate_local_idx=candidate_local_idx,
        rounds=int(bootstrap_rounds),
        frac=float(bootstrap_frac),
        topk=min(int(final_topk), max(5, int(final_topk))),
        seed=int(_config._env_value("DECAT_RUNTIME_SEED", None, 42)),
    )
    perturb_score = np.zeros((fingerprint_np.shape[1],), dtype=np.float32)
    stability_score = np.zeros((fingerprint_np.shape[1],), dtype=np.float32)
    bootstrap_score = np.zeros((fingerprint_np.shape[1],), dtype=np.float32)
    for pos, local_idx in enumerate(candidate_local_idx):
        info = delta_map.get(int(local_idx), {})
        cnt = int(info.get("delta_count", 0))
        if cnt > 0:
            perturb_score[int(local_idx)] = float(
                info.get("delta_sum", 0.0) / max(cnt, 1)
            )
        bootstrap_score[int(local_idx)] = float(bootstrap_map.get(int(local_idx), 0.0))
        if str(stability_mode).strip().lower() == "override" and isinstance(
            stability_override, dict
        ):
            stability_score[int(local_idx)] = float(
                stability_override.get(str(feature_labels[int(local_idx)]), 0.0)
            )
        elif str(stability_mode).strip().lower() == "none":
            stability_score[int(local_idx)] = 0.0
        else:
            stability_score[int(local_idx)] = float(
                bootstrap_map.get(int(local_idx), 0.0)
            )
    attn_cons = _interpretation_attribution._minmax_scale_array(
        attn_score[candidate_local_idx]
    )
    ig_cons = _interpretation_attribution._minmax_scale_array(
        ig_score[candidate_local_idx]
    )
    delta_cons = _interpretation_attribution._minmax_scale_array(
        perturb_score[candidate_local_idx]
    )
    stab_cons = _interpretation_attribution._minmax_scale_array(
        stability_score[candidate_local_idx]
    )
    supp_cons = _interpretation_attribution._minmax_scale_array(
        support_frac[candidate_local_idx]
    )
    rows = []
    for pos, local_idx in enumerate(candidate_local_idx):
        bit_name = str(feature_labels[int(local_idx)])
        rows.append(
            {
                "local_feature_idx": int(local_idx),
                "bit_name": bit_name,
                "bit": int(
                    _parse_fp_bit_index(bit_name)
                    if _parse_fp_bit_index(bit_name) is not None
                    else local_idx
                ),
                "attention_score": float(attn_score[int(local_idx)]),
                "ig_score": float(ig_score[int(local_idx)]),
                "perturb_score": float(perturb_score[int(local_idx)]),
                "stability_score": float(stability_score[int(local_idx)]),
                "bootstrap_stability_score": float(bootstrap_score[int(local_idx)]),
                "support_count": int(round(float(support_count[int(local_idx)]))),
                "support_frac": float(support_frac[int(local_idx)]),
                "consensus_score": float(
                    weight_attn * float(attn_cons[pos])
                    + weight_ig * float(ig_cons[pos])
                    + weight_delta * float(delta_cons[pos])
                    + weight_stab * float(stab_cons[pos])
                    + weight_support * float(supp_cons[pos])
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row.get("consensus_score", float("-inf"))),
            float(row.get("perturb_score", 0.0)),
            float(row.get("attention_score", 0.0)),
            float(row.get("ig_score", 0.0)),
            float(row.get("stability_score", 0.0)),
            float(row.get("support_count", 0.0)),
        ),
        reverse=True,
    )
    ranked_names = [str(row["bit_name"]) for row in rows]
    return (rows, active_mat, support_count, ranked_names)
