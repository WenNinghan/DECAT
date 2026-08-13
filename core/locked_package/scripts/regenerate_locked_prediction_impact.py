from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from torch import nn
from torch.utils.data import DataLoader


PACKAGE = Path(__file__).resolve().parents[1]
SRC = PACKAGE / "src"
ARTIFACTS = PACKAGE / "artifacts" / "run"
OUTPUT = Path(
    os.environ.get(
        "DECAT_IMPACT_OUTPUT_ROOT",
        str(PACKAGE / "artifacts" / "derived_analysis" / "prediction_impact"),
    )
)
DATA = PACKAGE / "data" / "反应logk指纹数据_25类_含环境无机物_Other_1626条.csv"
SPLIT = PACKAGE / "data" / "固定划分_1626_同分子异pH非均分_75-12.5-12.5.json"
CHECKPOINT = ARTIFACTS / "transformer_v7_best.pth"
SUMMARY = ARTIFACTS / "best_params.json"


def _set_locked_environment(summary: dict) -> None:
    payload = json.loads((ARTIFACTS / "fixed_params_run_summary.json").read_text(encoding="utf-8"))
    locked_env = payload["json_payload"]["env"]
    os.environ.update({str(key): str(value) for key, value in locked_env.items()})
    os.environ.update(
        {
            "TRANSFORMER_SEED": "242",
            "TRANSFORMER_DETERMINISTIC": "1",
            "TRANSFORMER_DEVICE": "cpu",
            "TRANSFORMER_V7_DATA_CSV": str(DATA),
            "TRANSFORMER_V7_FIXED_SPLIT_JSON": str(SPLIT),
            "TRANSFORMER_V6_FIXED_SPLIT_JSON": str(SPLIT),
            "TRANSFORMER_CMA_ENABLE": "1",
            "TRANSFORMER_CMA_MAX_SAMPLES": "200",
            "TRANSFORMER_CMA_IG_STEPS": "16",
            "TRANSFORMER_CMA_BOOTSTRAP_ROUNDS": "128",
            "TRANSFORMER_CMA_BOOTSTRAP_FRAC": "0.75",
            "TRANSFORMER_CMA_CANDIDATE_TOPK": "60",
            "TRANSFORMER_CMA_TOPK": "30",
            "TRANSFORMER_CMA_MIN_SUPPORT": "3",
            "TRANSFORMER_CMA_STABILITY_TRUE": "0",
            "TRANSFORMER_ATTN_SUBSTRUCT_TOPK": "30",
            "TRANSFORMER_ATTN_SUBSTRUCT_UNIQUE_TOPK": "20",
            "TRANSFORMER_ATTN_SUBSTRUCT_UNIQUE_MIN_HEAVY": "3",
            "TRANSFORMER_EXPORT_UNIQUE_ATOM_GROUPS": "1",
            "TRANSFORMER_TOP_FUNCTIONAL_GROUPS": "20",
            "TRANSFORMER_ATTN_SUBSTRUCT_USE_TRAINVAL": "0",
            "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        }
    )


def _build_contextual_model(module, fingerprint_dim: int, numeric_dim: int, config):
    class ContextualDualExpertRegressor(module.DualExpertRegressor):
        def __init__(self, fingerprint_dim: int, numeric_dim: int, model_config):
            super().__init__(fingerprint_dim, numeric_dim, model_config)
            hidden = max(16, int(model_config.d_model // 4))
            self.contextual_residual = nn.Sequential(
                nn.Linear(numeric_dim + 1, hidden),
                nn.GELU(),
                nn.Dropout(float(model_config.dropout)),
                nn.Linear(hidden, 1),
            )

        def forward_components(self, fingerprint: torch.Tensor, numeric: torch.Tensor):
            components = super().forward_components(fingerprint, numeric)
            context = torch.cat(
                [numeric.to(torch.float32), components["residual_gate"].unsqueeze(-1)], dim=1
            )
            adjustment = torch.tanh(self.contextual_residual(context)).squeeze(-1) * 0.15
            components["correction"] = components["correction"] + adjustment
            components["final"] = components["final"] + adjustment
            return components

    return ContextualDualExpertRegressor(fingerprint_dim, numeric_dim, config)


def _signed_mask_deltas(model, loader: DataLoader, bit_to_local: dict[str, int]) -> dict[str, float]:
    totals = {name: [0.0, 0] for name in bit_to_local}
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in loader:
            fingerprint = batch["fingerprint"].to(device)
            numeric = batch["numeric"].to(device)
            reference = model(fingerprint, numeric)
            for bit_name, local_index in bit_to_local.items():
                active = torch.abs(fingerprint[:, local_index]) > 1e-6
                if not bool(active.any()):
                    continue
                masked = fingerprint.clone()
                masked[active, local_index] = 0.0
                signed = (reference - model(masked, numeric))[active]
                totals[bit_name][0] += float(signed.sum().item())
                totals[bit_name][1] += int(signed.numel())
    return {
        name: float(total / count) if count else 0.0
        for name, (total, count) in totals.items()
    }


def _save_structure_svgs(rows: list[dict], output_dir: Path) -> None:
    structure_dir = output_dir / "individual_structure_svg"
    structure_dir.mkdir(parents=True, exist_ok=True)
    for stale_svg in structure_dir.glob("TOP*.svg"):
        stale_svg.unlink()
    for rank, row in enumerate(rows, start=1):
        smiles = str(row.get("canonical_substructure", row.get("substructure_smiles", "")))
        mol = Chem.MolFromSmiles(smiles) or Chem.MolFromSmarts(smiles)
        if mol is None:
            continue
        drawer = Draw.MolDraw2DSVG(1800, 1100)
        drawer.drawOptions().bondLineWidth = 3
        drawer.drawOptions().minFontSize = 28
        drawer.drawOptions().maxFontSize = 54
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        (structure_dir / f"TOP{rank:02d}_{row['bit_name']}.svg").write_text(
            drawer.GetDrawingText(), encoding="utf-8"
        )


def _render_top20(rows: list[dict], output_dir: Path) -> None:
    selected = rows[:20]
    _save_structure_svgs(selected, output_dir)
    figure, axes = plt.subplots(4, 5, figsize=(18, 13.6), facecolor="white")
    for axis in axes.ravel():
        axis.set_axis_off()
        axis.add_patch(
            plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#C8CDD4", linewidth=0.9, transform=axis.transAxes)
        )

    for rank, (axis, row) in enumerate(zip(axes.ravel(), selected), start=1):
        smiles = str(row.get("canonical_substructure", row.get("substructure_smiles", "")))
        mol = Chem.MolFromSmiles(smiles) or Chem.MolFromSmarts(smiles)
        if mol is not None:
            image = Draw.MolToImage(
                mol,
                size=(1800, 1100),
                kekulize=False,
                options=None,
            )
            axis.imshow(np.asarray(image), extent=(0.03, 0.97, 0.23, 0.87), aspect="auto")
        else:
            axis.text(0.5, 0.56, "Structure unavailable", ha="center", va="center", fontsize=11)

        delta = float(row.get("signed_prediction_delta_logk", 0.0))
        arrow, color = ("↑", "#D9485F") if delta >= 0 else ("↓", "#1A9E85")
        axis.text(0.04, 0.96, f"TOP{rank} ({row['bit_name']})", transform=axis.transAxes,
                  ha="left", va="top", fontsize=12.4, fontweight="bold", color="#242A33")
        axis.text(0.50, 0.105, arrow, transform=axis.transAxes, ha="center", va="center",
                  fontsize=40, fontweight="bold", color=color)

    for axis in axes.ravel()[len(selected):]:
        axis.set_visible(False)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985, wspace=0.025, hspace=0.025)
    base = output_dir / "locked_seed242_prediction_impact_top20_atom_groups"
    figure.savefig(base.with_suffix(".pdf"), dpi=400, bbox_inches=None, facecolor="white")
    figure.savefig(base.with_suffix(".png"), dpi=400, bbox_inches=None, facecolor="white")
    figure.savefig(base.with_suffix(".tif"), dpi=400, bbox_inches=None, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)


def _ranked_bit_representatives(result: dict, output_dir: Path) -> list[dict]:
    score_path = output_dir / "prediction_impact" / "consensus_motif_scores.csv"
    score_rows = pd.read_csv(score_path).to_dict("records")
    bit_rows = pd.read_csv(
        output_dir / "prediction_impact" / "consensus_bit_substructures.csv"
    ).to_dict("records")
    representatives = []
    for metric in score_rows:
        bit_name = str(metric["bit_name"])
        candidates = []
        for bit_row in bit_rows:
            if str(bit_row.get("bit_name", "")) != bit_name:
                continue
            molecule = Chem.MolFromSmiles(str(bit_row.get("substructure_smiles", "")))
            if molecule is None or molecule.GetNumHeavyAtoms() < 2:
                continue
            candidate = {**bit_row, **metric}
            candidate["canonical_substructure"] = Chem.MolToSmiles(molecule, canonical=True)
            candidate["heavy_atoms"] = int(molecule.GetNumHeavyAtoms())
            candidate["hetero_atoms"] = int(
                sum(atom.GetAtomicNum() not in (1, 6) for atom in molecule.GetAtoms())
            )
            candidate["has_unsaturation"] = bool(
                any(bond.GetBondTypeAsDouble() > 1.1 for bond in molecule.GetBonds())
            )
            candidate["has_aromaticity"] = bool(any(atom.GetIsAromatic() for atom in molecule.GetAtoms()))
            candidates.append(candidate)
        if not candidates:
            continue
        candidates.sort(
            key=lambda row: (
                int(row.get("hetero_atoms", 0)) > 0,
                bool(row.get("has_unsaturation", False)) or bool(row.get("has_aromaticity", False)),
                int(row.get("hetero_atoms", 0)),
                int(row.get("heavy_atoms", 0)),
                int(row.get("substructure_count", 0)),
                float(row.get("substructure_frac", 0.0)),
            ),
            reverse=True,
        )
        representatives.append(candidates[0])
        if len(representatives) == 20:
            break
    if len(representatives) < 20:
        raise RuntimeError(f"Only {len(representatives)} valid bit-backtracked groups were available; expected 20.")
    return representatives


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    _set_locked_environment(summary)
    sys.path.insert(0, str(SRC))
    from decat import transformer_v9_transformer_centered as decat

    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(242)
    np.random.seed(242)
    dataset = decat.FingerprintReactionDataset(str(DATA), max_fp_bits=3147, fingerprint_scale=False)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_idx = np.asarray(split["train_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)
    dataset.fit_scalers(train_idx)
    ranking = decat._get_fp_ranking(dataset, train_idx, 3147, "rf")
    selected_columns = ranking[:914]
    test_subset = decat.SelectedFeatureSubset(dataset, test_idx, selected_columns)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=0)

    config = decat.FingerprintConfig()
    config.d_model = 212
    config.dropout = 0.1
    config.batch_size = 32
    config.max_fp_tokens = 115
    config.n_layers = 2
    config.n_heads = 4
    config.norm_first = False
    config.base_numeric_dim = int(dataset.base_num_dim)
    model = _build_contextual_model(decat, len(selected_columns), dataset.num_dim, config)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    decat.OUT_DIR = str(OUTPUT)
    decat.RUN_OUTPUT_DIR = str(OUTPUT)
    result = decat.export_consensus_motif_artifacts(
        model=model,
        dataset=dataset,
        explain_loader=test_loader,
        train_indices=test_idx,
        prefix="prediction_impact",
    )
    rows = _ranked_bit_representatives(result, OUTPUT)

    bit_to_local = {f"fp_{int(original_bit)}": int(local) for local, original_bit in enumerate(selected_columns)}
    signed_scaled = _signed_mask_deltas(
        model, test_loader, {str(row["bit_name"]): bit_to_local[str(row["bit_name"])] for row in rows}
    )
    logk_scale = float(np.asarray(dataset.logk_scaler.scale_).reshape(-1)[0])
    for row in rows:
        signed_raw = float(signed_scaled.get(str(row["bit_name"]), 0.0) * logk_scale)
        row["signed_prediction_delta_scaled"] = float(signed_scaled.get(str(row["bit_name"]), 0.0))
        row["signed_prediction_delta_logk"] = signed_raw
        row["direction_pred"] = "positive" if signed_raw >= 0 else "negative"

    rows.sort(key=lambda row: float(row.get("consensus_score", 0.0)), reverse=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "prediction_impact_top20_atom_groups.csv", index=False, encoding="utf-8-sig")
    _render_top20(rows, OUTPUT)

    manifest = {
        "model": "DECAT locked seed 242, transformer-centered dual expert with contextual residual",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "data": str(DATA),
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "split": str(SPLIT),
        "split_sha256": hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
        "performance": {key: summary[key] for key in ("train_r2", "val_r2", "test_r2")},
        "explanation_set": "locked test split inputs (n=200); labels are not used in attention, IG, masking, bootstrap, or ranking",
        "score": "0.20 attention + 0.20 |IG| + 0.30 masking impact + 0.20 bootstrap stability + 0.10 support",
        "stability": "bootstrap, 128 rounds, 75% subsampling; cross-seed retraining was not run",
        "direction": "mean signed prediction(with active bit) - prediction(masked bit), expressed in logk units",
        "interpretation": "predictive-impact backtracking, not an attention ranking and not a chemical causal-effect estimate",
    }
    (OUTPUT / "prediction_impact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        "# Locked seed-242 prediction-impact backtracking\n\n"
        "The 4×5 figure ranks Morgan-bit-derived atom groups with a consensus score. "
        "Its arrows report the signed prediction shift after masking the active bit; they are not attention weights "
        "and do not claim a causal chemical effect. Individual high-definition vector structures are in `individual_structure_svg/`.\n",
        encoding="utf-8",
    )
    print(f"Export complete: {OUTPUT}")


if __name__ == "__main__":
    main()
