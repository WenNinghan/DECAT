"""Deterministic condition-group partitions and selected-feature subsets."""

from __future__ import annotations
import json
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    import decat.data.dataset as _data_dataset


def _make_selected_subsets_for_indices(
    dataset: _data_dataset.FingerprintReactionDataset,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    fp_bits: int,
    topk_features: int,
    fp_select_method: str,
) -> Tuple[Dataset, Dataset, int]:
    from torch.utils.data import Subset
    import decat.config as _config
    import decat.data.dataset as _data_dataset
    import decat.data.features as _data_features
    import decat.runtime as _runtime

    use_topk = _config._env_bool_pair(
        "DECAT_RUNTIME_USE_TOPK", "DECAT_RUNTIME_USE_TOPK_FALLBACK", True
    )
    if use_topk:
        rank = _data_features._get_fp_ranking(
            dataset,
            np.asarray(train_idx, dtype=int),
            int(fp_bits),
            str(fp_select_method),
        )
        k = int(min(max(1, topk_features), dataset.fingerprint_dim))
        top_idx = rank[:k]
        train_subset = _runtime.resolve_hook(
            "subset_factory", _data_dataset.SelectedFeatureSubset
        )(dataset, train_idx, top_idx)
        eval_subset = _runtime.resolve_hook(
            "subset_factory", _data_dataset.SelectedFeatureSubset
        )(dataset, eval_idx, top_idx)
        return (train_subset, eval_subset, int(k))
    return (
        Subset(dataset, train_idx),
        Subset(dataset, eval_idx),
        int(dataset.fingerprint_dim),
    )


TARGET_FRACTIONS = np.asarray([0.75, 0.125, 0.125], dtype=float)
SPLIT_NAMES = ("train_idx", "val_idx", "test_idx")


def _build_split_indices(
    dataset: _data_dataset.FingerprintReactionDataset,
) -> Dict[str, np.ndarray]:
    """Load a generated local split, or build the same protocol from active inputs."""
    import decat.config as _config
    import decat.runtime as _runtime

    fixed_split_json = str(
        _config._env_value(
            "DECAT_RUNTIME_FIXED_SPLIT_JSON",
            "DECAT_RUNTIME_FIXED_SPLIT_JSON_FALLBACK",
            _config._env_value("DECAT_RUNTIME_FIXED_SPLIT_JSON_FALLBACK_2", None, ""),
        )
    ).strip()
    if fixed_split_json:
        with open(fixed_split_json, "r", encoding="utf-8-sig") as stream:
            payload = json.load(stream)
    else:
        dataframe = pd.read_csv(
            dataset.csv_path, encoding="utf-8-sig", low_memory=False
        )
        payload = build_condition_split(dataframe)
    parts = {name: np.asarray(payload.get(name, []), dtype=int) for name in SPLIT_NAMES}
    all_indices = np.concatenate(list(parts.values()))
    if (
        any(part.size == 0 for part in parts.values())
        or all_indices.size != len(dataset)
        or np.unique(all_indices).size != len(dataset)
        or not np.array_equal(np.sort(all_indices), np.arange(len(dataset)))
    ):
        raise ValueError(
            "Split members must be nonempty, disjoint and cover every active row once."
        )
    parts["train_val_idx"] = np.concatenate([parts["train_idx"], parts["val_idx"]])
    _runtime.context.split_cache = {
        "n": len(dataset),
        "seed": int(payload["random_seed"]),
        "fixed_split_json": fixed_split_json,
        **parts,
    }
    return parts


def _canonical_smiles(value: object) -> str:
    cleaned = "".join(str(value).replace("\u200b", "").replace("\ufeff", "").split())
    molecule = Chem.MolFromSmiles(cleaned)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {value!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _candidate_score(
    assignments: np.ndarray,
    group_rows: list[np.ndarray],
    group_categories: np.ndarray,
    group_target_histograms: np.ndarray,
    category_totals: np.ndarray,
    target_totals: np.ndarray,
    molecule_group_ids: dict[str, list[int]],
) -> tuple[float, dict[str, object]]:
    row_counts = np.zeros(3, dtype=float)
    category_counts = np.zeros((3, category_totals.size), dtype=float)
    target_counts = np.zeros((3, target_totals.size), dtype=float)
    for group_id, split_id in enumerate(assignments):
        row_count = group_rows[group_id].size
        row_counts[split_id] += row_count
        category_counts[split_id, group_categories[group_id]] += row_count
        target_counts[split_id] += group_target_histograms[group_id]

    total_rows = float(row_counts.sum())
    balanced_category_mask = category_totals >= 10
    size_error = float(
        np.sum(((row_counts / total_rows - TARGET_FRACTIONS) / TARGET_FRACTIONS) ** 2)
    )
    category_error = float(
        np.mean(
            (
                category_counts[:, balanced_category_mask]
                / category_totals[balanced_category_mask][None, :]
                - TARGET_FRACTIONS[:, None]
            )
            ** 2
        )
    )
    target_error = float(
        np.mean(
            (target_counts / target_totals[None, :] - TARGET_FRACTIONS[:, None]) ** 2
        )
    )
    missing_balanced = int(np.sum(category_counts[1:, balanced_category_mask] == 0))
    multi_condition_train_only = 0
    eligible_for_three_missing_one_holdout = 0
    membership_patterns: dict[str, int] = defaultdict(int)
    for group_ids in molecule_group_ids.values():
        present = sorted(set(int(assignments[group_id]) for group_id in group_ids))
        pattern = "+".join(
            SPLIT_NAMES[split_id].replace("_idx", "") for split_id in present
        )
        membership_patterns[pattern] += 1
        if len(group_ids) >= 2 and present == [0]:
            multi_condition_train_only += 1
        if len(group_ids) >= 3 and not ({1, 2} <= set(present)):
            eligible_for_three_missing_one_holdout += 1

    score = (
        1000.0 * missing_balanced
        + 50.0 * size_error
        + 20.0 * category_error
        + 2.0 * target_error
        + 0.10 * multi_condition_train_only
        + 0.02 * eligible_for_three_missing_one_holdout
    )
    return score, {
        "score": score,
        "row_counts": row_counts.astype(int).tolist(),
        "size_error": size_error,
        "category_error": category_error,
        "target_error": target_error,
        "missing_balanced_category_slots": missing_balanced,
        "multi_condition_molecules_left_train_only": multi_condition_train_only,
        "three_plus_condition_molecules_missing_val_or_test": eligible_for_three_missing_one_holdout,
        "molecule_membership_patterns": dict(sorted(membership_patterns.items())),
    }


def build_condition_split(dataframe: pd.DataFrame, *, search_seeds: int = 512) -> dict:
    """Generate the condition-interpolation protocol without stored membership.

    Rows retain their input positions. Exact reproduction requires the original
    structures, categories, float64 measurements and row order.
    """
    if int(search_seeds) != search_seeds or search_seeds < 1:
        raise ValueError("search_seeds must be a positive integer.")
    if dataframe.empty:
        raise ValueError("Cannot split an empty dataset.")
    if dataframe["category27_label"].nunique(dropna=False) != 25:
        raise RuntimeError("Expected exactly 25 category27_label values")

    canonical_smiles = dataframe["SMILES"].map(_canonical_smiles).to_numpy(dtype=object)
    ph_values = pd.to_numeric(dataframe["pH"], errors="raise").to_numpy(dtype=float)
    categories = dataframe["category27_label"].astype(str).to_numpy(dtype=object)
    category_levels = sorted(np.unique(categories).tolist())
    category_to_id = {category: index for index, category in enumerate(category_levels)}
    category_ids = np.asarray(
        [category_to_id[value] for value in categories], dtype=int
    )
    category_totals = np.bincount(category_ids, minlength=len(category_levels)).astype(
        float
    )
    target_bins = pd.qcut(
        pd.to_numeric(dataframe["logk"], errors="raise").rank(method="first"),
        q=5,
        labels=False,
    ).to_numpy(dtype=int)
    target_totals = np.bincount(target_bins, minlength=5).astype(float)

    atomic_key_to_rows: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row_index, (smiles, ph, category) in enumerate(
        zip(canonical_smiles, ph_values, categories)
    ):
        atomic_key_to_rows[(smiles, ph.hex(), category)].append(row_index)
    atomic_keys = sorted(atomic_key_to_rows)
    group_rows = [np.asarray(atomic_key_to_rows[key], dtype=int) for key in atomic_keys]
    group_categories = np.asarray(
        [category_to_id[key[2]] for key in atomic_keys], dtype=int
    )
    group_target_histograms = np.asarray(
        [np.bincount(target_bins[rows], minlength=5) for rows in group_rows],
        dtype=float,
    )
    molecule_group_ids: dict[str, list[int]] = defaultdict(list)
    for group_id, key in enumerate(atomic_keys):
        molecule_group_ids[key[0]].append(group_id)

    best: tuple[float, int, np.ndarray, dict[str, object]] | None = None
    for random_seed in range(search_seeds):
        rng = np.random.default_rng(random_seed)
        assignments = np.full(len(atomic_keys), -1, dtype=int)
        reserved_train_rows = 0
        remaining_group_ids: list[int] = []
        for molecule in sorted(molecule_group_ids):
            group_ids = molecule_group_ids[molecule]
            reserved_group = int(rng.choice(group_ids))
            assignments[reserved_group] = 0
            reserved_train_rows += group_rows[reserved_group].size
            remaining_group_ids.extend(
                group_id for group_id in group_ids if group_id != reserved_group
            )

        remaining_rows = dataframe.shape[0] - reserved_train_rows
        target_rows = TARGET_FRACTIONS * dataframe.shape[0]
        remaining_probabilities = np.asarray(
            [
                max(0.0, target_rows[0] - reserved_train_rows),
                target_rows[1],
                target_rows[2],
            ],
            dtype=float,
        )
        remaining_probabilities /= remaining_probabilities.sum()
        if remaining_rows <= 0:
            raise RuntimeError(
                "No condition groups remain after reserving training groups"
            )
        shuffled_remaining = rng.permutation(remaining_group_ids)
        sampled_splits = rng.choice(
            3,
            size=len(shuffled_remaining),
            p=remaining_probabilities,
        )
        assignments[shuffled_remaining] = sampled_splits
        score, audit = _candidate_score(
            assignments,
            group_rows,
            group_categories,
            group_target_histograms,
            category_totals,
            target_totals,
            molecule_group_ids,
        )
        candidate = (score, random_seed, assignments.copy(), audit)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    if best is None:
        raise RuntimeError("No split candidate was generated")
    _, random_seed, assignments, pre_coverage_audit = best
    coverage_moves: list[dict[str, object]] = []
    for split_id in (1, 2):
        for category_id, category_total in enumerate(category_totals):
            if category_total < 10:
                continue
            split_has_category = any(
                assignments[group_id] == split_id
                and group_categories[group_id] == category_id
                for group_id in range(len(atomic_keys))
            )
            if split_has_category:
                continue
            train_group_counts_by_molecule = defaultdict(int)
            for group_id, assigned_split in enumerate(assignments):
                if assigned_split == 0:
                    train_group_counts_by_molecule[atomic_keys[group_id][0]] += 1
            candidates = [
                group_id
                for group_id, assigned_split in enumerate(assignments)
                if assigned_split == 0 and group_categories[group_id] == category_id
            ]
            candidates.sort(
                key=lambda group_id: (
                    train_group_counts_by_molecule[atomic_keys[group_id][0]] <= 1,
                    group_rows[group_id].size,
                    atomic_keys[group_id],
                )
            )
            if not candidates:
                raise RuntimeError(
                    f"Cannot cover category {category_levels[category_id]} in {SPLIT_NAMES[split_id]}"
                )
            moved_group = candidates[0]
            molecule = atomic_keys[moved_group][0]
            preserves_seen_molecule = train_group_counts_by_molecule[molecule] > 1
            assignments[moved_group] = split_id
            coverage_moves.append(
                {
                    "category": category_levels[category_id],
                    "destination": SPLIT_NAMES[split_id],
                    "rows": int(group_rows[moved_group].size),
                    "preserves_molecule_in_training": preserves_seen_molecule,
                }
            )
    _, selection_audit = _candidate_score(
        assignments,
        group_rows,
        group_categories,
        group_target_histograms,
        category_totals,
        target_totals,
        molecule_group_ids,
    )
    selection_audit["pre_coverage_audit"] = pre_coverage_audit
    selection_audit["minimum_category_coverage_moves"] = coverage_moves
    split_indices = {
        split_name: sorted(
            int(row)
            for group_id, split_id in enumerate(assignments)
            if split_id == split_index
            for row in group_rows[group_id]
        )
        for split_index, split_name in enumerate(SPLIT_NAMES)
    }
    index_sets = {name: set(indices) for name, indices in split_indices.items()}
    if set().union(*index_sets.values()) != set(range(dataframe.shape[0])):
        raise RuntimeError("Split does not cover every row exactly once")
    if any(
        index_sets[left] & index_sets[right]
        for left, right in (
            ("train_idx", "val_idx"),
            ("train_idx", "test_idx"),
            ("val_idx", "test_idx"),
        )
    ):
        raise RuntimeError("Split row indices overlap")

    atomic_sets = {
        name: {
            atomic_keys[group_id]
            for group_id, split_id in enumerate(assignments)
            if split_id == split_index
        }
        for split_index, name in enumerate(SPLIT_NAMES)
    }
    atomic_overlaps = {
        "train_val": len(atomic_sets["train_idx"] & atomic_sets["val_idx"]),
        "train_test": len(atomic_sets["train_idx"] & atomic_sets["test_idx"]),
        "val_test": len(atomic_sets["val_idx"] & atomic_sets["test_idx"]),
    }
    if any(atomic_overlaps.values()):
        raise RuntimeError(f"Atomic model-input groups overlap: {atomic_overlaps}")

    molecule_sets = {
        name: {key[0] for key in keys} for name, keys in atomic_sets.items()
    }
    unseen_holdout_molecules = {
        "val": len(molecule_sets["val_idx"] - molecule_sets["train_idx"]),
        "test": len(molecule_sets["test_idx"] - molecule_sets["train_idx"]),
    }
    category_counts = {
        name: {
            category: int(np.sum(categories[indices] == category))
            for category in category_levels
        }
        for name, indices in split_indices.items()
    }
    payload = {
        "n_samples": int(dataframe.shape[0]),
        "split_protocol": (
            "Pre-registered non-uniform seen-molecule condition interpolation split. Atomic groups "
            "are RDKit canonical isomeric SMILES + exact pH + category27_label and never cross sets. "
            "Validation/test molecules are preferentially represented by another condition in training; "
            "minimal singleton exceptions are allowed only to cover categories with at least 10 rows. "
            "Molecules are not evenly divided across the three sets."
        ),
        "group_identifier": "RDKit canonical isomeric SMILES + exact IEEE-754 pH + category27_label",
        "group_assignment_policy": (
            "one randomly selected atomic condition group per molecule is initially reserved for training; "
            "remaining groups are assigned stochastically toward global 75/12.5/12.5 balance; "
            "then the minimum number of groups is moved to cover categories with at least 10 rows"
        ),
        "valid_for_unseen_molecule_generalization": False,
        "valid_for_natural_condition_interpolation": True,
        "valid_for_seen_molecule_condition_interpolation": True,
        "scope_warning": (
            "This split evaluates new-pH interpolation for molecules represented in training and "
            "must not be described as unseen-molecule generalization."
        ),
        "test_selection_policy": (
            "sealed before training; selected only by split size, category balance, logk-quintile "
            "balance, and soft molecule-membership preferences; never by model metrics"
        ),
        "category_balance_policy": (
            "Classes with at least 10 rows are balanced; smaller classes are assigned naturally."
        ),
        "search_seed_range": [0, search_seeds - 1],
        "random_seed": random_seed,
        **split_indices,
        "split_sizes": {name: len(indices) for name, indices in split_indices.items()},
        "atomic_group_count": len(atomic_keys),
        "atomic_group_overlaps": atomic_overlaps,
        "canonical_molecule_counts": {
            name: len(values) for name, values in molecule_sets.items()
        },
        "unseen_holdout_molecules": unseen_holdout_molecules,
        "category_counts": category_counts,
        "selection_audit": selection_audit,
        "input_contract": {
            "category_one_hot_dimensions": 25,
            "category_source": "category27_label",
            "other_label": "A",
            "other_name": "Other aqueous environmental species",
            "other_is_single_one_hot_dimension": True,
            "other_is_not_subdivided": True,
        },
    }
    return payload
