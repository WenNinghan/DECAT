from __future__ import annotations

import json
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT = Path(__file__).resolve().parent
BASE_CONFIG_PATH = PROJECT / "configs" / "V14_1637_validation_topk5.json"
SPLIT_PATH = PROJECT / "data" / "固定划分_1626_同分子异pH非均分_75-12.5-12.5.json"

PARAMS = {
    "fp_bits": 3147,
    "topk_features": 914,
    "d_model": 212,
    "batch_size": 32,
    "lr": 0.003,
    "weight_decay": 0.0015,
    "dropout": 0.1,
    "max_fp_tokens": 115,
    "n_layers": 2,
    "n_heads": 4,
    "model_mode": "dual",
    "fp_select": "rf",
    "enable_chem_attn_bias": True,
    "norm_first": False,
    "topk_checkpoint_ensemble": 3,
    "use_checkpoint_ensemble": True,
    "lambda_inv": 0.18,
    "lambda_dro": 0.28,
    "lambda_physics": 0.08,
    "dro_tau": 0.05,
    "generalization_penalty": 0.03,
}

# All profiles retain the same Transformer-centred dual-expert architecture,
# chemistry-aware attention bias, physics anchoring, gated correction and
# validation-fitted adaptive fusion.  They differ only in tunable capacity and
# optimisation settings.
PARAMETER_PROFILES = {
    "seed242_baseline": {},
    "category_embed12": {},
    "category_embed16": {},
    "25class_compact": {
        "fp_bits": 3147,
        "topk_features": 850,
        "d_model": 192,
        "batch_size": 32,
        "lr": 0.0022,
        "weight_decay": 0.001,
        "dropout": 0.12,
        "max_fp_tokens": 110,
        "n_layers": 2,
        "n_heads": 4,
        "norm_first": False,
        "topk_checkpoint_ensemble": 3,
        "lambda_inv": 0.12,
        "lambda_dro": 0.18,
        "lambda_physics": 0.02,
        "dro_tau": 0.03,
        "generalization_penalty": 0.02,
    },
    "25class_regularized": {
        "fp_bits": 3147,
        "topk_features": 1024,
        "d_model": 224,
        "batch_size": 32,
        "lr": 0.0018,
        "weight_decay": 0.0025,
        "dropout": 0.13,
        "max_fp_tokens": 128,
        "n_layers": 2,
        "n_heads": 4,
        "norm_first": True,
        "topk_checkpoint_ensemble": 3,
        "lambda_inv": 0.10,
        "lambda_dro": 0.16,
        "lambda_physics": 0.025,
        "dro_tau": 0.03,
        "generalization_penalty": 0.02,
    },
    "25class_wide": {
        "fp_bits": 3147,
        "topk_features": 1024,
        "d_model": 256,
        "batch_size": 32,
        "lr": 0.0015,
        "weight_decay": 0.0015,
        "dropout": 0.10,
        "max_fp_tokens": 140,
        "n_layers": 2,
        "n_heads": 4,
        "norm_first": True,
        "topk_checkpoint_ensemble": 3,
        "lambda_inv": 0.10,
        "lambda_dro": 0.16,
        "lambda_physics": 0.02,
        "dro_tau": 0.03,
        "generalization_penalty": 0.015,
    },
    "25class_stack": {},
    "25class_stack_descriptors": {},
    "25class_stack_hierarchical_descriptors": {},
    "25class_stack_radius3_descriptors": {},
    "25class_stack_structural_clusters_descriptors": {},
    "25class_transformer_centered": {},
    "25class_balanced": {},
    "25class_chem_alpha10": {},
    "25class_chem_alpha20": {},
    "25class_category_dropout10": {},
    "25class_phcubic": {},
    "25class_residual35": {},
    "25class_attention_dominant": {},
    "25class_attention_floor35": {},
    "25class_global_descriptors": {},
    "25class_hierarchical": {},
    "25class_hierarchical_descriptors": {},
    "25class_radius3": {},
    "25class_radius3_descriptors": {},
    "25class_kinetic_prior": {},
    "25class_kinetic_prior_descriptors": {},
    "25class_structural_clusters": {},
    "25class_structural_clusters_descriptors": {},
    "validation_topk5": {
        "topk_features": 1100,
        "d_model": 224,
        "batch_size": 24,
        "lr": 0.0021,
        "weight_decay": 0.001,
        "dropout": 0.065,
        "max_fp_tokens": 152,
        "norm_first": True,
        "topk_checkpoint_ensemble": 5,
        "lambda_inv": 0.12,
        "lambda_dro": 0.2,
        "lambda_physics": 2e-06,
        "dro_tau": 0.03,
        "generalization_penalty": 0.012,
    },
    "high_capacity_validation": {
        "fp_bits": 2402,
        "topk_features": 1200,
        "d_model": 300,
        "batch_size": 47,
        "lr": 0.0014863771108497273,
        "weight_decay": 0.006410193938917924,
        "dropout": 0.25,
        "max_fp_tokens": 112,
        "n_layers": 3,
        "norm_first": False,
        "topk_checkpoint_ensemble": 2,
        "lambda_inv": 0.3,
        "lambda_dro": 0.04741297383997616,
        "lambda_physics": 0.06179913141401888,
        "dro_tau": 0.014097466686165534,
        "generalization_penalty": 0.0,
    },
}

PROFILE_ENV = {
    "seed242_baseline": {"TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0"},
    "validation_topk5": {"TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0"},
    "high_capacity_validation": {"TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0"},
    "category_embed12": {"TRANSFORMER_V9_CATEGORY_EMBED_DIM": "12"},
    "category_embed16": {"TRANSFORMER_V9_CATEGORY_EMBED_DIM": "16"},
    "25class_compact": {"TRANSFORMER_V9_CATEGORY_EMBED_DIM": "8"},
    "25class_regularized": {"TRANSFORMER_V9_CATEGORY_EMBED_DIM": "16"},
    "25class_wide": {"TRANSFORMER_V9_CATEGORY_EMBED_DIM": "16"},
    "25class_stack": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V5_ENABLE_STACK_COMPONENTS": "1",
        "TRANSFORMER_V5_STACK_FIT_MODE": "oof",
        "TRANSFORMER_V5_STACK_OOF_FOLDS": "5",
        "TRANSFORMER_V5_STACK_GROUP_OOF": "1",
        "TRANSFORMER_V5_ENABLE_STACK_REFIT": "0",
    },
    "25class_stack_descriptors": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_GLOBAL_DESCRIPTORS": "1",
        "TRANSFORMER_V5_ENABLE_STACK_COMPONENTS": "1",
        "TRANSFORMER_V5_STACK_FIT_MODE": "oof",
        "TRANSFORMER_V5_STACK_OOF_FOLDS": "5",
        "TRANSFORMER_V5_STACK_GROUP_OOF": "1",
        "TRANSFORMER_V5_ENABLE_STACK_REFIT": "0",
    },
    "25class_stack_hierarchical_descriptors": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_HIERARCHICAL_CATEGORY": "1",
        "TRANSFORMER_V9_GLOBAL_DESCRIPTORS": "1",
        "TRANSFORMER_V5_ENABLE_STACK_COMPONENTS": "1",
        "TRANSFORMER_V5_STACK_FIT_MODE": "oof",
        "TRANSFORMER_V5_STACK_OOF_FOLDS": "5",
        "TRANSFORMER_V5_STACK_GROUP_OOF": "1",
        "TRANSFORMER_V5_ENABLE_STACK_REFIT": "0",
    },
    "25class_stack_radius3_descriptors": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_GLOBAL_DESCRIPTORS": "1",
        "TRANSFORMER_V9_MORGAN_RADIUS": "3",
        "TRANSFORMER_V5_ENABLE_STACK_COMPONENTS": "1",
        "TRANSFORMER_V5_STACK_FIT_MODE": "oof",
        "TRANSFORMER_V5_STACK_OOF_FOLDS": "5",
        "TRANSFORMER_V5_STACK_GROUP_OOF": "1",
        "TRANSFORMER_V5_ENABLE_STACK_REFIT": "0",
    },
    "25class_stack_structural_clusters_descriptors": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_GLOBAL_DESCRIPTORS": "1",
        "TRANSFORMER_V9_STRUCTURAL_CLUSTERS": "15",
        "TRANSFORMER_V5_ENABLE_STACK_COMPONENTS": "1",
        "TRANSFORMER_V5_STACK_FIT_MODE": "oof",
        "TRANSFORMER_V5_STACK_OOF_FOLDS": "5",
        "TRANSFORMER_V5_STACK_GROUP_OOF": "1",
        "TRANSFORMER_V5_ENABLE_STACK_REFIT": "0",
    },
    "25class_transformer_centered": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_ATTN_PH_ONLY": "1",
    },
    "25class_balanced": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V3_BALANCED_SAMPLER": "1",
    },
    "25class_chem_alpha10": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V3_CHEM_ATTN_ALPHA": "0.10",
    },
    "25class_chem_alpha20": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V3_CHEM_ATTN_ALPHA": "0.20",
    },
    "25class_category_dropout10": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_CATEGORY_DROPOUT": "0.10",
    },
    "25class_phcubic": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_PH_BASIS_DEGREE": "3",
    },
    "25class_residual35": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_RESIDUAL_MAX": "0.35",
    },
    "25class_attention_dominant": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V7_ATTENTION_GATE_LOGIT_BIAS": "0.50",
    },
    "25class_attention_floor35": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V7_ATTENTION_MIN_GATE": "0.35",
    },
    "25class_global_descriptors": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_GLOBAL_DESCRIPTORS": "1",
    },
    "25class_hierarchical": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_HIERARCHICAL_CATEGORY": "1",
    },
    "25class_hierarchical_descriptors": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_HIERARCHICAL_CATEGORY": "1",
        "TRANSFORMER_V9_GLOBAL_DESCRIPTORS": "1",
    },
    "25class_radius3": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_MORGAN_RADIUS": "3",
    },
    "25class_radius3_descriptors": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_MORGAN_RADIUS": "3",
        "TRANSFORMER_V9_GLOBAL_DESCRIPTORS": "1",
    },
    "25class_kinetic_prior": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_CATEGORY_KINETIC_PRIOR": "1",
    },
    "25class_kinetic_prior_descriptors": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_CATEGORY_KINETIC_PRIOR": "1",
        "TRANSFORMER_V9_GLOBAL_DESCRIPTORS": "1",
    },
    "25class_structural_clusters": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_STRUCTURAL_CLUSTERS": "15",
    },
    "25class_structural_clusters_descriptors": {
        "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        "TRANSFORMER_V9_STRUCTURAL_CLUSTERS": "15",
        "TRANSFORMER_V9_GLOBAL_DESCRIPTORS": "1",
    },
}


def _write_25class_training_contract(
    *,
    model,
    data_path: Path,
    split_path: Path,
    output_root: Path,
    params: dict,
    environment: dict[str, str],
    profile_name: str,
    seed: str,
    model_seed: str,
    unmask_test: bool,
    test_guided: bool,
) -> None:
    dataframe = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)
    split_payload = json.loads(split_path.read_text(encoding="utf-8-sig"))
    expected_rows = int(split_payload.get("n_samples", dataframe.shape[0]))
    required_category_columns = (
        "category27_code",
        "category27_name",
        "category27_label",
    )
    missing_columns = [column for column in required_category_columns if column not in dataframe.columns]
    if missing_columns:
        raise RuntimeError(f"25-class contract failed; missing columns: {missing_columns}")
    if dataframe.shape[0] != expected_rows:
        raise RuntimeError(
            f"25-class contract failed; split expects {expected_rows} rows, "
            f"got {dataframe.shape[0]}"
        )
    for column in required_category_columns:
        unique_count = int(dataframe[column].nunique(dropna=False))
        if unique_count != 25 or dataframe[column].isna().any():
            raise RuntimeError(
                f"25-class contract failed; {column} has {unique_count} unique values "
                f"and {int(dataframe[column].isna().sum())} missing values"
            )

    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if str(split_payload.get("source_sha256", "")).lower() != data_sha256:
        raise RuntimeError("25-class contract failed; split source SHA-256 does not match the active CSV")
    index_sets = {
        key: set(map(int, split_payload.get(key, [])))
        for key in ("train_idx", "val_idx", "test_idx")
    }
    if any(index_sets[left] & index_sets[right] for left, right in (("train_idx", "val_idx"), ("train_idx", "test_idx"), ("val_idx", "test_idx"))):
        raise RuntimeError("25-class contract failed; split indices overlap")
    if set().union(*index_sets.values()) != set(range(expected_rows)):
        raise RuntimeError(
            f"25-class contract failed; split indices do not cover exactly 0..{expected_rows - 1}"
        )

    row_hashes = pd.util.hash_pandas_object(
        dataframe.fillna("<NA>").astype(str),
        index=False,
    )
    exact_duplicate_row_count = int(row_hashes.duplicated(keep=False).sum())
    split_row_hashes = {
        key: set(row_hashes.iloc[sorted(indices)].astype("uint64").tolist())
        for key, indices in index_sets.items()
    }
    exact_row_hash_overlaps = {
        "train_val": len(split_row_hashes["train_idx"] & split_row_hashes["val_idx"]),
        "train_test": len(split_row_hashes["train_idx"] & split_row_hashes["test_idx"]),
        "val_test": len(split_row_hashes["val_idx"] & split_row_hashes["test_idx"]),
    }
    if exact_duplicate_row_count or any(exact_row_hash_overlaps.values()):
        raise RuntimeError(
            "Exact-row contract failed; "
            f"duplicate rows={exact_duplicate_row_count}, split overlaps={exact_row_hash_overlaps}"
        )

    canonical_smiles: list[str] = []
    for raw_smiles in dataframe["SMILES"].astype(str):
        cleaned_smiles = "".join(
            raw_smiles.replace("\u200b", "").replace("\ufeff", "").split()
        )
        molecule = model.Chem.MolFromSmiles(cleaned_smiles)
        if molecule is None:
            raise RuntimeError(
                f"Group contract failed; invalid SMILES cannot be canonicalized: {raw_smiles!r}"
            )
        canonical_smiles.append(
            model.Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        )
    canonical_smiles_array = np.asarray(canonical_smiles, dtype=object)
    split_groups = {
        key: set(canonical_smiles_array[sorted(indices)].tolist())
        for key, indices in index_sets.items()
    }
    group_overlaps = {
        "train_val": len(split_groups["train_idx"] & split_groups["val_idx"]),
        "train_test": len(split_groups["train_idx"] & split_groups["test_idx"]),
        "val_test": len(split_groups["val_idx"] & split_groups["test_idx"]),
    }
    ph_values = pd.to_numeric(dataframe["pH"], errors="raise").to_numpy(dtype=float)
    category_values = dataframe["category27_label"].astype(str).to_numpy(dtype=object)
    atomic_condition_keys = np.asarray(
        [
            f"{smiles}\x1f{ph.hex()}\x1f{category}"
            for smiles, ph, category in zip(
                canonical_smiles_array,
                ph_values,
                category_values,
            )
        ],
        dtype=object,
    )
    split_atomic_condition_groups = {
        key: set(atomic_condition_keys[sorted(indices)].tolist())
        for key, indices in index_sets.items()
    }
    atomic_condition_overlaps = {
        "train_val": len(
            split_atomic_condition_groups["train_idx"]
            & split_atomic_condition_groups["val_idx"]
        ),
        "train_test": len(
            split_atomic_condition_groups["train_idx"]
            & split_atomic_condition_groups["test_idx"]
        ),
        "val_test": len(
            split_atomic_condition_groups["val_idx"]
            & split_atomic_condition_groups["test_idx"]
        ),
    }
    natural_condition_group = bool(
        split_payload.get("valid_for_natural_condition_interpolation", False)
    )
    if natural_condition_group and any(atomic_condition_overlaps.values()):
        raise RuntimeError(
            "Natural condition-group contract failed; canonical SMILES + exact pH + "
            f"category27_label overlaps across splits: {atomic_condition_overlaps}"
        )
    condition_interpolation = os.environ.get("DECAT_CONDITION_INTERPOLATION", "0") == "1"
    allow_group_overlap_exploration = test_guided and os.environ.get(
        "DECAT_ALLOW_GROUP_OVERLAP_EXPLORATION", "0"
    ) == "1"
    group_overlap_allowed = (
        natural_condition_group
        or condition_interpolation
        or allow_group_overlap_exploration
    )
    if any(group_overlaps.values()) and not group_overlap_allowed:
        raise RuntimeError(
            "Unseen-molecule group contract failed; canonical SMILES overlap across splits: "
            f"{group_overlaps}"
        )

    required_innovations = {
        "transformer_centered": environment.get("TRANSFORMER_V9_TRANSFORMER_CENTERED") == "1",
        "dual_expert": params.get("model_mode") == "dual" and environment.get("TRANSFORMER_V3_ENABLE_DUAL") == "1",
        "chemistry_attention_bias": bool(params.get("enable_chem_attn_bias")) and environment.get("TRANSFORMER_V3_ENABLE_CHEM_ATTN_BIAS") == "1",
        "physics_anchoring": float(params.get("lambda_physics", 0.0)) > 0.0,
        "gated_correction": environment.get("TRANSFORMER_V3_GATE_EXTRA_FEATURES") == "1" and environment.get("TRANSFORMER_V3_ENABLE_CORRECTION_HEAD") == "1",
        "adaptive_fusion": environment.get("TRANSFORMER_V7_FIXED_ENABLE_FUSION") == "1" and environment.get("TRANSFORMER_V3_ENABLE_FUSION") == "1",
    }
    if not all(required_innovations.values()):
        raise RuntimeError(f"Innovation contract failed: {required_innovations}")

    os.environ.update(environment)
    runtime_dataset = model.FingerprintReactionDataset(str(data_path), int(params["fp_bits"]))
    if len(runtime_dataset) != expected_rows or int(runtime_dataset.primary_category_dim) != 25:
        raise RuntimeError(
            "25-class runtime contract failed; "
            f"dataset rows={len(runtime_dataset)}, primary_category_dim={runtime_dataset.primary_category_dim}"
        )
    runtime_category_columns = list(runtime_dataset.category_cols[: runtime_dataset.primary_category_dim])
    expected_category_columns = sorted(
        f"cat_{label}" for label in dataframe["category27_label"].astype(str).unique()
    )
    if sorted(runtime_category_columns) != expected_category_columns:
        raise RuntimeError("25-class runtime contract failed; encoded category labels differ from category27_label")
    exact_input_contract = profile_name == "25class_stack"
    if exact_input_contract and (
        len(runtime_dataset.category_cols) != 25 or int(runtime_dataset.num_dim) != 26
    ):
        raise RuntimeError(
            "Exact input contract failed; expected 25 category one-hot columns plus pH "
            f"(num_dim=26), got category_cols={len(runtime_dataset.category_cols)} "
            f"and num_dim={runtime_dataset.num_dim}"
        )

    contract = {
        "status": "passed",
        "data_csv": str(data_path),
        "data_sha256": data_sha256,
        "row_count": int(dataframe.shape[0]),
        "class_count": 25,
        "class_counts": {
            str(key): int(value)
            for key, value in dataframe["category27_label"].value_counts().sort_index().items()
        },
        "runtime_primary_category_dim": int(runtime_dataset.primary_category_dim),
        "runtime_numeric_dim": int(runtime_dataset.num_dim),
        "runtime_category_columns": runtime_category_columns,
        "exact_input_contract": {
            "required": exact_input_contract,
            "passed": (
                not exact_input_contract
                or (
                    len(runtime_dataset.category_cols) == 25
                    and int(runtime_dataset.num_dim) == 26
                )
            ),
            "features": "25-class one-hot + pH + molecular fingerprint",
            "other_is_single_one_hot_dimension": True,
        },
        "split_json": str(split_path),
        "split_source_sha256": split_payload["source_sha256"],
        "split_sizes": {key: len(value) for key, value in index_sets.items()},
        "exact_duplicate_row_count": exact_duplicate_row_count,
        "exact_row_hash_overlaps": exact_row_hash_overlaps,
        "canonical_smiles_group_counts": {
            key: len(value) for key, value in split_groups.items()
        },
        "canonical_smiles_group_overlaps": group_overlaps,
        "atomic_condition_group_counts": {
            key: len(value) for key, value in split_atomic_condition_groups.items()
        },
        "atomic_condition_group_overlaps": atomic_condition_overlaps,
        "natural_condition_group_contract": natural_condition_group,
        "unseen_molecule_generalization_contract": not any(group_overlaps.values()),
        "group_overlap_allowed_for_test_guided_exploration": allow_group_overlap_exploration,
        "group_overlap_allowed_for_condition_interpolation": condition_interpolation,
        "reporting_scope": (
            "natural condition-group evaluation; exact model-input groups are disjoint and "
            "different pH conditions of a molecule may naturally span splits"
            if natural_condition_group
            else (
                "condition-interpolation evaluation; exact rows are disjoint and molecules may span splits"
                if condition_interpolation
                else (
                    "development-only test-guided upper bound; not an independent blind test"
                    if allow_group_overlap_exploration
                    else "unseen-molecule group-disjoint evaluation"
                )
            )
        ),
        "profile": profile_name,
        "split_seed": int(split_payload.get("random_seed", seed)),
        "model_seed": model_seed,
        "selection_protocol": "test-guided-exploration" if test_guided else "validation-only",
        "test_unmasked_for_final_evaluation": bool(unmask_test),
        "innovations": required_innovations,
        "params": params,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "25class_training_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    seed = os.environ.get("DECAT_STRICT_SEED", "42")
    model_seed = os.environ.get("DECAT_MODEL_SEED", seed)
    split_override = os.environ.get("DECAT_SPLIT_PATH_OVERRIDE", "").strip()
    split_path = Path(split_override).resolve() if split_override else SPLIT_PATH
    if not split_path.is_file():
        raise FileNotFoundError(f"Fixed split file not found: {split_path}")
    split_metadata = json.loads(split_path.read_text(encoding="utf-8-sig"))
    natural_condition_group = bool(
        split_metadata.get("valid_for_natural_condition_interpolation", False)
    )
    data_override = os.environ.get("DECAT_DATA_PATH_OVERRIDE", "").strip()
    data_path = (
        Path(data_override).resolve()
        if data_override
        else PROJECT / "data" / "反应logk指纹数据_25类_含环境无机物_Other_1626条.csv"
    )
    if not data_path.is_file():
        raise FileNotFoundError(f"Active data file not found: {data_path}")
    profile_name = os.environ.get("DECAT_PARAMETER_PROFILE", "seed242_baseline").strip()
    if profile_name not in PARAMETER_PROFILES:
        raise ValueError(
            f"Unknown DECAT_PARAMETER_PROFILE={profile_name!r}; "
            f"available profiles: {sorted(PARAMETER_PROFILES)}"
        )
    params = {**PARAMS, **PARAMETER_PROFILES[profile_name]}
    override_text = os.environ.get("DECAT_PARAMS_JSON", "").strip()
    if override_text:
        try:
            override = json.loads(override_text)
        except json.JSONDecodeError as exc:
            raise ValueError("DECAT_PARAMS_JSON must be a JSON object.") from exc
        if not isinstance(override, dict):
            raise ValueError("DECAT_PARAMS_JSON must decode to a JSON object.")
        unknown = set(override) - set(PARAMS)
        if unknown:
            raise ValueError(f"Unsupported parameter override keys: {sorted(unknown)}")
        params.update(override)
    unmask_test = os.environ.get("DECAT_UNMASK_TEST", "0") == "1"
    # Test-guided runs are intentionally opt-in and must remain auditable.
    # They are for internal upper-bound exploration only, never for reporting
    # an independent test result.
    test_guided = os.environ.get("DECAT_TEST_GUIDED", "0") == "1"
    if test_guided and not unmask_test:
        raise ValueError("DECAT_TEST_GUIDED=1 requires DECAT_UNMASK_TEST=1.")
    objective_target = "test" if test_guided else "val"
    output_root_override = os.environ.get("DECAT_OUTPUT_ROOT_OVERRIDE", "").strip()
    output_root = (
        Path(output_root_override) / f"seed_{seed}"
        if output_root_override
        else PROJECT / "输出" / "decat_v23_blind_seed_ensemble" / f"seed_{seed}"
    )
    base_config = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    environment = {str(key): str(value) for key, value in base_config["env"].items()}
    environment.update(
        {
            "DECAT_PROJECT_DIR": str(PROJECT),
            "DECAT_OUTPUT_ROOT": str(output_root),
            "TRANSFORMER_V9_OUTPUT_ROOT": str(output_root),
            # The 25-class curation is the only active category context.  The
            # legacy 15-class file is intentionally not referenced here.
            "TRANSFORMER_V7_DATA_CSV": str(data_path),
            "TRANSFORMER_V7_FIXED_SPLIT_JSON": str(split_path),
            "TRANSFORMER_V6_FIXED_SPLIT_JSON": str(split_path),
            "TRANSFORMER_V7_MODE": "fixed_json",
            "TRANSFORMER_V7_FIXED_OBJECTIVE_TARGET": objective_target,
            # The split is locked by the JSON path. Keep model initialization
            # independent so a seed ensemble does not silently retrain seed 242.
            "TRANSFORMER_V7_FIXED_SEED": model_seed,
            "TRANSFORMER_V6_FIXED_SEED": model_seed,
            "TRANSFORMER_V7_FIXED_MAX_EPOCHS": os.environ.get("DECAT_MAX_EPOCHS", "120"),
            "TRANSFORMER_V7_FIXED_EARLY_STOP": os.environ.get("DECAT_EARLY_STOP", "36"),
            "TRANSFORMER_V7_FIXED_SKIP_ARTIFACTS": "0" if os.environ.get("DECAT_SAVE_ARTIFACTS", "0") == "1" else "1",
            "TRANSFORMER_V7_FIXED_ENABLE_FUSION": "1",
            "TRANSFORMER_SEED": model_seed,
            "TRANSFORMER_DEVICE": "cuda",
            "TRANSFORMER_DETERMINISTIC": "1",
            "TRANSFORMER_V3_CALIBRATION_TARGET": objective_target,
            "TRANSFORMER_V3_SKIP_ARTIFACTS": "0" if os.environ.get("DECAT_SAVE_ARTIFACTS", "0") == "1" else "1",
            "TRANSFORMER_V3_DEFER_BEST_EXPORT": "1",
            "TRANSFORMER_V3_ENABLE_FUSION": "1",
            "TRANSFORMER_V3_CALIBRATE_COMPONENTS": os.environ.get(
                "DECAT_CALIBRATE_COMPONENTS", "0"
            ),
            "TRANSFORMER_V3_CALIBRATE_BLEND": os.environ.get(
                "DECAT_CALIBRATE_BLEND", "0"
            ),
            "TRANSFORMER_V3_BALANCED_SAMPLER": os.environ.get("DECAT_BALANCED_SAMPLER", "0"),
        }
    )
    environment.update(PROFILE_ENV.get(profile_name, {}))
    os.environ.update(environment)
    sys.path.insert(0, str(PROJECT / "src"))
    from decat import transformer_v9_transformer_centered as model

    category_a_family = os.environ.get("DECAT_CATEGORY_A_FAMILY_OVERRIDE", "").strip()
    if category_a_family:
        model.REACTIVITY_FAMILY_BY_CATEGORY_CODE["A"] = category_a_family

    _write_25class_training_contract(
        model=model,
        data_path=data_path,
        split_path=split_path,
        output_root=output_root,
        params=params,
        environment=environment,
        profile_name=profile_name,
        seed=seed,
        model_seed=model_seed,
        unmask_test=unmask_test,
        test_guided=test_guided,
    )

    split_state: dict[str, np.ndarray] = {}
    original_build_split = model._build_split_indices
    original_subset = model.SelectedFeatureSubset
    original_metrics = model._metrics
    original_fusion = model._adaptive_fusion
    final_component = os.environ.get("DECAT_FINAL_COMPONENT", "").strip()

    def record_split(dataset):
        split = original_build_split(dataset)
        split_state["test_idx"] = np.asarray(split["test_idx"], dtype=int).copy()
        return split

    class BlindedSelectedFeatureSubset(original_subset):
        def __init__(self, base_ds, indices, selected_col_idx):
            super().__init__(base_ds, indices, selected_col_idx)
            test_idx = split_state.get("test_idx", np.asarray([], dtype=int))
            self._blind_targets = (
                self.indices.size == test_idx.size
                and np.array_equal(np.sort(self.indices), np.sort(test_idx))
            )

        def __getitem__(self, index):
            record = super().__getitem__(index)
            if self._blind_targets and not unmask_test:
                record["logk"] = torch.tensor(0.0, dtype=torch.float32)
                record["logk_raw"] = torch.tensor(0.0, dtype=torch.float32)
            return record

    def masked_metrics(y_true, y_pred):
        target = np.asarray(y_true, dtype=float).reshape(-1)
        if (not unmask_test) and target.size and np.all(target == 0.0):
            return float("nan"), float("nan")
        return original_metrics(y_true, y_pred)

    def validation_only_fusion(*args, **kwargs):
        kwargs["optimize_target"] = objective_target
        result = original_fusion(*args, **kwargs)
        if isinstance(result, dict):
            component_names = list(result.get("component_names", []) or [])
            if final_component:
                if final_component not in component_names:
                    raise RuntimeError(
                        f"Requested DECAT_FINAL_COMPONENT={final_component!r} is unavailable; "
                        f"available components: {component_names}"
                    )
                component_index = component_names.index(final_component)
                selected_train = np.asarray(
                    result["component_train_preds"][component_index], dtype=np.float32
                )
                selected_val = np.asarray(
                    result["component_val_preds"][component_index], dtype=np.float32
                )
                selected_test = np.asarray(
                    result["component_test_preds"][component_index], dtype=np.float32
                )
                result["train_pred"] = selected_train
                result["val_pred"] = selected_val
                result["test_pred"] = selected_test
                result["r2_train"], result["rmse_train"] = original_metrics(
                    kwargs["y_train"], selected_train
                )
                result["r2_val"], result["rmse_val"] = original_metrics(
                    kwargs["y_val"], selected_val
                )
                result["r2_test"], result["rmse_test"] = original_metrics(
                    kwargs["y_test"], selected_test
                )
                result["mode"] = final_component
                result["weights"] = [
                    1.0 if index == component_index else 0.0
                    for index in range(len(component_names))
                ]
            output_root = Path(os.environ["DECAT_OUTPUT_ROOT"])
            output_root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_root / "validation_predictions.npz",
                y_train=np.asarray(kwargs["y_train"], dtype=np.float32),
                pred_train=np.asarray(result["train_pred"], dtype=np.float32),
                y_val=np.asarray(kwargs["y_val"], dtype=np.float32),
                pred_val=np.asarray(result["val_pred"], dtype=np.float32),
                y_test=np.asarray(kwargs["y_test"], dtype=np.float32),
                pred_test=np.asarray(result["test_pred"], dtype=np.float32),
            )
            np.savez_compressed(
                output_root / "fusion_components.npz",
                component_names=np.asarray(result.get("component_names", []), dtype=str),
                x_train=np.asarray(kwargs["x_train"], dtype=np.float32),
                x_val=np.asarray(kwargs["x_val"], dtype=np.float32),
                x_test=np.asarray(kwargs["x_test"], dtype=np.float32),
                y_train=np.asarray(kwargs["y_train"], dtype=np.float32),
                y_val=np.asarray(kwargs["y_val"], dtype=np.float32),
                y_test=np.asarray(kwargs["y_test"], dtype=np.float32),
                component_train=np.asarray(result.get("component_train_preds", []), dtype=np.float32),
                component_val=np.asarray(result.get("component_val_preds", []), dtype=np.float32),
                component_test=np.asarray(result.get("component_test_preds", []), dtype=np.float32),
                fusion_weights=np.asarray(result.get("weights", []), dtype=np.float32),
            )
            if not unmask_test:
                result["r2_test"] = float("nan")
                result["rmse_test"] = float("nan")
        return result

    base_class = model.DualExpertRegressor

    class ContextualDualExpertRegressor(base_class):
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

    payload = {
        "params": params,
        "fixed_split_json": str(split_path),
        "env": base_config["env"],
        "selection_protocol": "test-guided" if test_guided else "validation-only",
        "final_component": final_component or "validation-selected adaptive fusion",
        "affine_component_calibration": environment.get(
            "TRANSFORMER_V3_CALIBRATE_COMPONENTS"
        ) == "1",
        "affine_blend_calibration": environment.get(
            "TRANSFORMER_V3_CALIBRATE_BLEND"
        ) == "1",
        "evaluation_scope": (
            "natural-condition-group"
            if natural_condition_group
            else (
                "condition-interpolation"
                if os.environ.get("DECAT_CONDITION_INTERPOLATION", "0") == "1"
                else "unseen-molecule-generalization"
            )
        ),
        "fixed_split_seed": int(split_metadata.get("random_seed", seed)),
        "model_initialisation_seed": model_seed,
    }
    vector = model._params_to_bo_vector(params)
    if vector is None:
        raise RuntimeError("Could not convert fixed parameter configuration.")

    model._build_split_indices = record_split
    model.SelectedFeatureSubset = BlindedSelectedFeatureSubset
    model._metrics = masked_metrics
    model._adaptive_fusion = validation_only_fusion
    model.DualExpertRegressor = ContextualDualExpertRegressor
    model._load_fixed_params_payload = lambda: (
        payload,
        (
            f"strict_{'natural_condition_group' if natural_condition_group else ('condition_interpolation' if os.environ.get('DECAT_CONDITION_INTERPOLATION', '0') == '1' else ('test_guided' if test_guided else 'blind'))}_v24_25class_"
            f"{profile_name}{'_custom' if override_text else ''}"
        ),
        list(vector),
    )
    model.main()
    if os.environ.get("DECAT_RUN_FINAL_RETRAIN", "0") == "1":
        model.run_final_retrain_on_trainval()


if __name__ == "__main__":
    main()
