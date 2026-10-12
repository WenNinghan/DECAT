"""Molecular descriptors and training-only feature ranking."""

from __future__ import annotations
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
import decat.runtime as _runtime
import decat.data.dataset as _data_dataset


def _normalized_global_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Return bounded, whole-molecule descriptors for the MLP expert."""
    heavy = max(1, int(mol.GetNumHeavyAtoms()))
    aromatic = sum((1 for atom in mol.GetAtoms() if atom.GetIsAromatic()))
    hetero = sum((1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6}))
    formal_charge = sum((int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))
    values = np.asarray(
        [
            Descriptors.MolWt(mol) / 500.0,
            Crippen.MolLogP(mol) / 5.0,
            rdMolDescriptors.CalcTPSA(mol) / 150.0,
            Lipinski.NumHDonors(mol) / 5.0,
            Lipinski.NumHAcceptors(mol) / 10.0,
            Lipinski.NumRotatableBonds(mol) / 15.0,
            rdMolDescriptors.CalcNumRings(mol) / 8.0,
            rdMolDescriptors.CalcFractionCSP3(mol),
            heavy / 50.0,
            formal_charge / 3.0,
            aromatic / heavy,
            hetero / heavy,
        ],
        dtype=np.float32,
    )
    return np.clip(values, -3.0, 3.0)


def _match_feature_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    target_dim = int(max(0, target_dim))
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if target_dim == 0:
        return x.new_zeros((x.size(0), 0))
    cur_dim = int(x.size(1))
    if cur_dim == target_dim:
        return x
    if cur_dim > target_dim:
        return x[:, :target_dim]
    pad = x.new_zeros((x.size(0), target_dim - cur_dim))
    return torch.cat([x, pad], dim=1)


def _get_fp_ranking(
    dataset: _data_dataset.FingerprintReactionDataset,
    train_idx: np.ndarray,
    fp_bits: int,
    method: str,
) -> np.ndarray:
    import decat.data.dataset as _data_dataset

    method = str(method).strip().lower()
    if method not in {"rf", "f_regression"}:
        method = "rf"
    key = (int(fp_bits), method)
    if key in _runtime.context.fp_rank_cache:
        return _runtime.context.fp_rank_cache[key]
    x_train_fp = dataset.fingerprint[train_idx].astype(np.float32)
    y_train_raw = dataset.logk_raw[train_idx].reshape(-1)
    if method == "rf":
        rf = RandomForestRegressor(
            n_estimators=240, max_depth=None, random_state=42, n_jobs=-1
        )
        rf.fit(x_train_fp, y_train_raw)
        score = rf.feature_importances_
    else:
        f_vals, _ = f_regression(x_train_fp, y_train_raw)
        score = np.nan_to_num(f_vals, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    rank = np.argsort(score)[::-1].astype(int)
    _runtime.context.fp_rank_cache[key] = rank
    return rank
