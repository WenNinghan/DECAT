"""Fixed dataset partitions and selected-feature subsets."""

from __future__ import annotations
import json
from typing import Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
import decat.config as _config
import decat.data.dataset as _data_dataset
import decat.data.features as _data_features
import decat.runtime as _runtime


def _make_selected_subsets_for_indices(
    dataset: _data_dataset.FingerprintReactionDataset,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    fp_bits: int,
    topk_features: int,
    fp_select_method: str,
) -> Tuple[Dataset, Dataset, int]:
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


def _build_split_indices(
    dataset: _data_dataset.FingerprintReactionDataset,
) -> Dict[str, np.ndarray]:
    split_seed = int(
        _config._env_value(
            "DECAT_RUNTIME_SPLIT_SEED",
            "DECAT_RUNTIME_SPLIT_SEED_FALLBACK",
            _config._env_value("DECAT_RUNTIME_SPLIT_SEED_FALLBACK_2", None, "42"),
        )
    )
    fixed_split_json = str(
        _config._env_value(
            "DECAT_RUNTIME_FIXED_SPLIT_JSON",
            "DECAT_RUNTIME_FIXED_SPLIT_JSON_FALLBACK",
            _config._env_value("DECAT_RUNTIME_FIXED_SPLIT_JSON_FALLBACK_2", None, ""),
        )
    ).strip()
    if fixed_split_json:
        try:
            with open(fixed_split_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            train_idx = np.asarray(payload.get("train_idx", []), dtype=int)
            val_idx = np.asarray(payload.get("val_idx", []), dtype=int)
            test_idx = np.asarray(payload.get("test_idx", []), dtype=int)
            all_idx = np.concatenate([train_idx, val_idx, test_idx], axis=0)
            if (
                train_idx.size > 0
                and val_idx.size > 0
                and (test_idx.size > 0)
                and (all_idx.size == len(dataset))
                and (np.unique(all_idx).size == len(dataset))
                and (int(all_idx.min()) >= 0)
                and (int(all_idx.max()) < len(dataset))
            ):
                train_val_idx = np.concatenate([train_idx, val_idx], axis=0).astype(int)
                _runtime.context.split_cache = {
                    "n": len(dataset),
                    "seed": int(split_seed),
                    "train_idx": train_idx.astype(int),
                    "val_idx": val_idx.astype(int),
                    "test_idx": test_idx.astype(int),
                    "train_val_idx": train_val_idx,
                    "fixed_split_json": fixed_split_json,
                }
                return {
                    "train_idx": np.asarray(
                        _runtime.context.split_cache["train_idx"], dtype=int
                    ),
                    "val_idx": np.asarray(
                        _runtime.context.split_cache["val_idx"], dtype=int
                    ),
                    "test_idx": np.asarray(
                        _runtime.context.split_cache["test_idx"], dtype=int
                    ),
                    "train_val_idx": np.asarray(
                        _runtime.context.split_cache["train_val_idx"], dtype=int
                    ),
                }
        except Exception:
            pass
    if _runtime.context.split_cache.get("n") == len(dataset) and int(
        _runtime.context.split_cache.get("seed", split_seed)
    ) == int(split_seed):
        return {
            "train_idx": np.asarray(
                _runtime.context.split_cache["train_idx"], dtype=int
            ),
            "val_idx": np.asarray(_runtime.context.split_cache["val_idx"], dtype=int),
            "test_idx": np.asarray(_runtime.context.split_cache["test_idx"], dtype=int),
            "train_val_idx": np.asarray(
                _runtime.context.split_cache.get(
                    "train_val_idx",
                    np.concatenate(
                        [
                            _runtime.context.split_cache["train_idx"],
                            _runtime.context.split_cache["val_idx"],
                        ],
                        axis=0,
                    ),
                ),
                dtype=int,
            ),
        }
    indices = np.arange(len(dataset), dtype=int)
    stratify_labels = None
    if _config._env_bool_pair(
        "DECAT_RUNTIME_STRATIFY", "DECAT_RUNTIME_STRATIFY_FALLBACK", False
    ):
        try:
            labels = np.asarray(dataset.category).argmax(axis=1)
            uniq, cnt = np.unique(labels, return_counts=True)
            if uniq.size > 1 and np.all(cnt >= 2):
                stratify_labels = labels
        except Exception:
            stratify_labels = None
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=int(split_seed),
        shuffle=True,
        stratify=stratify_labels,
    )
    stratify_tv = None
    if stratify_labels is not None:
        label_map = {
            int(i): int(l) for i, l in zip(indices.tolist(), stratify_labels.tolist())
        }
        try:
            labels_tv = np.asarray(
                [label_map[int(i)] for i in train_val_idx], dtype=int
            )
            uniq, cnt = np.unique(labels_tv, return_counts=True)
            if uniq.size > 1 and np.all(cnt >= 2):
                stratify_tv = labels_tv
        except Exception:
            stratify_tv = None
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.125,
        random_state=int(split_seed),
        shuffle=True,
        stratify=stratify_tv,
    )
    _runtime.context.split_cache = {
        "n": len(dataset),
        "seed": int(split_seed),
        "train_idx": np.asarray(train_idx, dtype=int),
        "val_idx": np.asarray(val_idx, dtype=int),
        "test_idx": np.asarray(test_idx, dtype=int),
        "train_val_idx": np.asarray(train_val_idx, dtype=int),
    }
    return {
        "train_idx": np.asarray(_runtime.context.split_cache["train_idx"], dtype=int),
        "val_idx": np.asarray(_runtime.context.split_cache["val_idx"], dtype=int),
        "test_idx": np.asarray(_runtime.context.split_cache["test_idx"], dtype=int),
        "train_val_idx": np.asarray(
            _runtime.context.split_cache["train_val_idx"], dtype=int
        ),
    }
