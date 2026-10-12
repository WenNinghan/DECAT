"""Online applicability-domain assessment in the selected training fingerprint space."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# Training leave-one-out nearest-neighbor distance quantiles.
AD_STRICT_DISTANCE_THRESHOLD = 0.6428571428571428
AD_BORDERLINE_DISTANCE_THRESHOLD = 0.8
PH_TRAIN_MIN = 1.0
PH_TRAIN_MAX = 12.0


def _tanimoto_dense_matrix(query_fp: np.ndarray, train_fps: np.ndarray) -> np.ndarray:
    query = np.asarray(query_fp, dtype=bool).reshape(1, -1)
    train = np.asarray(train_fps, dtype=bool)
    inter = np.logical_and(train, query).sum(axis=1).astype(np.float32)
    union = np.logical_or(train, query).sum(axis=1).astype(np.float32)
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return sim.astype(np.float32)


def evaluate_ad_for_fingerprint(
    bundle: Dict[str, object], fp_dense: np.ndarray, ph_value: Optional[float] = None
) -> Dict[str, object]:
    """Evaluate applicability using the training-space distance and pH thresholds."""
    train_fps = np.asarray(bundle["train_fp_selected_bool"], dtype=bool)
    selected_idx = np.asarray(bundle["selected_idx"], dtype=int).reshape(-1)
    fp_arr = np.asarray(fp_dense, dtype=np.float32).reshape(-1)
    if fp_arr.size < int(selected_idx.max()) + 1:
        return {
            "nearest_train_similarity": np.nan,
            "nearest_train_distance": np.nan,
            "nearest_train_compound": "",
            "structural_AD_label": "Out of AD",
            "DECAT_AD_label": "Out of AD",
            "AD_reason": "fingerprint length is incompatible with the locked DECAT feature space",
            "AD_source": "Assessed online against the training fingerprint space",
        }
    fp_selected = fp_arr[selected_idx] > 0.5
    sims = _tanimoto_dense_matrix(fp_selected.astype(np.uint8), train_fps)
    best_i = int(np.nanargmax(sims)) if sims.size else -1
    best_sim = float(sims[best_i]) if best_i >= 0 else float("nan")
    distance = float(1.0 - best_sim) if np.isfinite(best_sim) else float("nan")
    if np.isfinite(distance) and distance <= AD_STRICT_DISTANCE_THRESHOLD:
        structural_label = "In AD"
    elif np.isfinite(distance) and distance <= AD_BORDERLINE_DISTANCE_THRESHOLD:
        structural_label = "Borderline"
    else:
        structural_label = "Out of AD"

    ph_label = "within"
    if ph_value is not None:
        try:
            ph_float = float(ph_value)
            if ph_float < PH_TRAIN_MIN or ph_float > PH_TRAIN_MAX:
                ph_label = "outside"
        except Exception:
            ph_label = "missing"
    final_label = (
        "Out of AD - pH outside train range"
        if ph_label == "outside"
        else structural_label
    )
    train_names = list(bundle.get("train_compounds", []))
    train_smiles = list(bundle.get("train_smiles", []))
    return {
        "nearest_train_similarity": best_sim,
        "nearest_train_distance": distance,
        "nearest_train_index": int(best_i),
        "nearest_train_compound": (
            train_names[best_i] if 0 <= best_i < len(train_names) else ""
        ),
        "nearest_train_SMILES": (
            train_smiles[best_i] if 0 <= best_i < len(train_smiles) else ""
        ),
        "structural_AD_label": structural_label,
        "DECAT_AD_label": final_label,
        "AD_reason": (
            f"nearest Tanimoto distance={distance:.4f}; "
            f"strict<= {AD_STRICT_DISTANCE_THRESHOLD:.4f}, borderline<= {AD_BORDERLINE_DISTANCE_THRESHOLD:.4f}"
            if np.isfinite(distance)
            else "nearest Tanimoto distance is unavailable"
        ),
        "AD_source": "Assessed online against the training fingerprint space",
    }
