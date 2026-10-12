"""Category-balanced training sampling."""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
import decat.config as _config
import decat.data.dataset as _data_dataset


def _build_category_balanced_sampler(
    dataset: _data_dataset.FingerprintReactionDataset,
    train_subset,
    train_idx: np.ndarray,
) -> Tuple[Optional[WeightedRandomSampler], Optional[Dict[str, object]]]:
    enabled = _config._env_bool_pair(
        "DECAT_RUNTIME_BALANCED_SAMPLER",
        "DECAT_RUNTIME_BALANCED_SAMPLER_FALLBACK",
        False,
    )
    if not enabled:
        return (None, None)
    base_indices = _data_dataset._extract_subset_base_indices(train_subset, train_idx)
    if base_indices.size < 20:
        return (None, None)
    try:
        cat_all = np.asarray(dataset.category).argmax(axis=1).astype(int)
        cats = cat_all[base_indices]
    except Exception:
        return (None, None)
    uniq, cnt = np.unique(cats, return_counts=True)
    if uniq.size < 2:
        return (None, None)
    weight_by_cat = {}
    for c, n in zip(uniq.tolist(), cnt.tolist()):
        weight_by_cat[int(c)] = float(1.0 / max(np.sqrt(float(n)), 1.0))
    sample_w = np.asarray(
        [weight_by_cat[int(c)] for c in cats.tolist()], dtype=np.float64
    )
    mean_w = float(np.mean(sample_w)) if sample_w.size > 0 else 1.0
    if mean_w > 0:
        sample_w = sample_w / mean_w
    sample_w = np.clip(sample_w, 0.1, 8.0)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=int(sample_w.shape[0]),
        replacement=True,
    )
    cat_cols = list(getattr(dataset, "category_cols", []))
    stats = []
    for c, n in zip(uniq.tolist(), cnt.tolist()):
        c_int = int(c)
        if 0 <= c_int < len(cat_cols):
            label = str(cat_cols[c_int]).replace("cat_", "")
        else:
            label = str(c_int)
        stats.append(
            {
                "category_index": c_int,
                "category_label": label,
                "count": int(n),
                "sample_weight": float(weight_by_cat[c_int]),
            }
        )
    info = {
        "enabled": True,
        "n_train": int(base_indices.size),
        "n_categories": int(uniq.size),
        "stats": stats,
    }
    return (sampler, info)
