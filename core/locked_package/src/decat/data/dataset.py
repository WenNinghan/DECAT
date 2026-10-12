"""Reaction records, preprocessing and selected-feature dataset views."""

from __future__ import annotations
import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import Dataset, Subset
import decat.chemistry as _chemistry
import decat.config as _config
import decat.runtime as _runtime


def _resolve_preferred_category_labels(df: pd.DataFrame) -> Optional[pd.Series]:
    resolved = pd.Series(pd.NA, index=df.index, dtype="object")
    found = False
    for col in _chemistry.CATEGORY_LABEL_PRIORITY:
        if col not in df.columns:
            continue
        found = True
        values = df[col]
        if not isinstance(values, pd.Series):
            values = pd.Series(values, index=df.index)
        values = values.astype("string").str.strip()
        values = values.mask(values.eq("")).mask(values.str.lower().eq("nan"))
        resolved = resolved.fillna(values)
    if not found:
        return None
    return resolved.fillna("Unknown").astype(str)


class FingerprintReactionDataset(Dataset):
    """Load reaction features and manage the preprocessing fitted for each split."""

    @staticmethod
    def _build_cache_key(
        csv_path: str, max_fp_bits: int
    ) -> Tuple[str, int, int, int, int]:
        abs_path = os.path.abspath(csv_path)
        try:
            stat = os.stat(abs_path)
            mtime_ns = int(stat.st_mtime_ns)
            size = int(stat.st_size)
        except Exception:
            mtime_ns = 0
            size = 0
        aggregate_duplicates = int(
            _config._env_bool_pair(
                "DECAT_RUNTIME_AGGREGATE_DUPLICATE_TARGETS",
                "DECAT_RUNTIME_AGGREGATE_DUPLICATE_TARGETS_FALLBACK",
                False,
            )
        )
        return (abs_path, int(max_fp_bits), mtime_ns, size, aggregate_duplicates)

    @staticmethod
    def _cache_store(
        cache_key: Tuple[str, int, int, int, int], payload: Dict[str, object]
    ) -> None:
        if cache_key in _runtime.context.dataset_base_cache:
            _runtime.context.dataset_base_cache[cache_key] = payload
            return
        if len(_runtime.context.dataset_base_cache) >= int(
            _runtime.DATASET_CACHE_MAX_ENTRIES
        ):
            oldest_key = next(iter(_runtime.context.dataset_base_cache.keys()))
            _runtime.context.dataset_base_cache.pop(oldest_key, None)
        _runtime.context.dataset_base_cache[cache_key] = payload

    def __init__(
        self, csv_path: str, max_fp_bits: int, fingerprint_scale: bool = False
    ):
        import decat.data.features as _data_features

        super().__init__()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        cache_key = self._build_cache_key(csv_path, int(max_fp_bits))
        cache_item = _runtime.context.dataset_base_cache.get(cache_key)
        if cache_item is None:
            df = pd.read_csv(csv_path)
            smiles_col = (
                "SMILES"
                if "SMILES" in df.columns
                else "smiles" if "smiles" in df.columns else None
            )
            if smiles_col is None:
                raise ValueError("SMILES column not found.")
            df[smiles_col] = df[smiles_col].map(_chemistry._clean_smiles_text)
            if "pH" not in df.columns:
                raise ValueError("pH column not found.")
            target_col = next(
                (c for c in ["logk", "LogK", "LOGK", "logK"] if c in df.columns), None
            )
            if target_col is None:
                raise ValueError("logk column not found.")
            preferred_cat = _resolve_preferred_category_labels(df)
            if preferred_cat is not None:
                df["__decat_category_label__"] = preferred_cat.astype(str)
                category_col = "__decat_category_label__"
            elif "category" in df.columns:
                category_col = "category"
            else:
                df["__decat_category_label__"] = "Unknown"
                category_col = "__decat_category_label__"
            valid_mask = df[[smiles_col, "pH", target_col]].notna().all(axis=1)
            keep_cols = [smiles_col, "pH", target_col, category_col]
            df_core = df.loc[valid_mask, keep_cols].copy().reset_index(drop=True)
            df_core[smiles_col] = (
                df_core[smiles_col].astype(str).str.replace("\\s+", "", regex=True)
            )
            aggregate_duplicates = _config._env_bool_pair(
                "DECAT_RUNTIME_AGGREGATE_DUPLICATE_TARGETS",
                "DECAT_RUNTIME_AGGREGATE_DUPLICATE_TARGETS_FALLBACK",
                False,
            )
            if aggregate_duplicates:
                group_cols = [smiles_col, "pH"]
                if category_col in df_core.columns:
                    group_cols.append(category_col)
                agg_payload = {target_col: "median"}
                df_core = (
                    df_core.groupby(group_cols, dropna=False, as_index=False)
                    .agg(agg_payload)
                    .reset_index(drop=True)
                )
            cat_series = df_core[category_col].fillna("Unknown").astype(str)
            x_cat = pd.get_dummies(cat_series, prefix="cat", dtype=np.uint8)
            cat_cols = x_cat.columns.tolist()
            primary_category_dim = len(cat_cols)
            if _config._env_bool_pair(
                "DECAT_RUNTIME_HIERARCHICAL_CATEGORY", None, False
            ):
                category_code = cat_series.str.split(":", n=1).str[0].str.strip()
                family_series = category_code.map(
                    _chemistry.REACTIVITY_FAMILY_BY_CATEGORY_CODE
                ).fillna("other")
                x_family = pd.get_dummies(
                    family_series, prefix="family", dtype=np.uint8
                )
                x_cat = pd.concat([x_cat, x_family], axis=1)
                cat_cols = cat_cols + x_family.columns.tolist()
            if _config._env_bool_pair(
                "DECAT_RUNTIME_CATEGORY_KINETIC_PRIOR", None, False
            ):
                for prior_col in (
                    "kinetic_category_level",
                    "kinetic_ph_response",
                    "kinetic_category_reliability",
                ):
                    x_cat[prior_col] = np.float32(0.0)
                    cat_cols.append(prior_col)
            structural_clusters = int(
                _config._env_value("DECAT_RUNTIME_STRUCTURAL_CLUSTERS", None, "0")
            )
            structural_clusters = max(0, min(32, structural_clusters))
            for cluster_idx in range(structural_clusters):
                cluster_col = f"structure_cluster_{cluster_idx:02d}"
                x_cat[cluster_col] = np.uint8(0)
                cat_cols.append(cluster_col)
            use_global_descriptors = _config._env_bool_pair(
                "DECAT_RUNTIME_GLOBAL_DESCRIPTORS", None, False
            )
            morgan_radius = int(
                _config._env_value("DECAT_RUNTIME_MORGAN_RADIUS", None, "2")
            )
            morgan_radius = max(1, min(4, morgan_radius))
            fps_list = []
            descriptor_list = []
            keep_idx = []
            for i, smi in enumerate(df_core[smiles_col].astype(str).values):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, morgan_radius, int(max_fp_bits)
                )
                arr = np.zeros((int(max_fp_bits),), dtype=np.int8)
                Chem.DataStructs.ConvertToNumpyArray(fp, arr)
                fps_list.append(arr)
                if use_global_descriptors:
                    descriptor_list.append(
                        _data_features._normalized_global_descriptors(mol)
                    )
                keep_idx.append(i)
            if not fps_list:
                raise ValueError(
                    "No valid fingerprints could be generated from SMILES."
                )
            df_core = df_core.iloc[keep_idx].reset_index(drop=True)
            x_cat = x_cat.iloc[keep_idx].reset_index(drop=True)
            if use_global_descriptors:
                descriptor_frame = pd.DataFrame(
                    np.stack(descriptor_list, axis=0),
                    columns=list(_chemistry.GLOBAL_DESCRIPTOR_COLUMNS),
                )
                x_cat = pd.concat([x_cat, descriptor_frame], axis=1)
                cat_cols = cat_cols + list(_chemistry.GLOBAL_DESCRIPTOR_COLUMNS)
            fingerprint = np.stack(fps_list, axis=0).astype(np.float32)
            self.csv_path = str(csv_path)
            self.max_fp_bits = int(max_fp_bits)
            self.smiles = df_core[smiles_col].astype(str).values
            self.ph = df_core["pH"].astype(float).values.reshape(-1, 1)
            self.logk_raw = df_core[target_col].astype(float).values.reshape(-1, 1)
            self.fingerprint = fingerprint
            self.category = x_cat.values.astype(np.float32)
            self.category_cols = cat_cols
            self.primary_category_dim = int(primary_category_dim)
            self._cache_store(
                cache_key,
                {
                    "csv_path": str(csv_path),
                    "max_fp_bits": int(max_fp_bits),
                    "smiles": np.asarray(self.smiles, dtype=object).copy(),
                    "ph": np.asarray(self.ph, dtype=np.float32).copy(),
                    "logk_raw": np.asarray(self.logk_raw, dtype=np.float32).copy(),
                    "fingerprint": np.asarray(
                        self.fingerprint, dtype=np.float32
                    ).copy(),
                    "category": np.asarray(self.category, dtype=np.float32).copy(),
                    "category_cols": list(self.category_cols),
                    "primary_category_dim": int(self.primary_category_dim),
                },
            )
        else:
            self.csv_path = str(cache_item.get("csv_path", csv_path))
            self.max_fp_bits = int(cache_item.get("max_fp_bits", max_fp_bits))
            self.smiles = np.asarray(cache_item.get("smiles", []), dtype=object).copy()
            self.ph = np.asarray(cache_item["ph"], dtype=np.float32).copy()
            self.logk_raw = np.asarray(cache_item["logk_raw"], dtype=np.float32).copy()
            self.fingerprint = np.asarray(
                cache_item["fingerprint"], dtype=np.float32
            ).copy()
            self.category = np.asarray(cache_item["category"], dtype=np.float32).copy()
            self.category_cols = list(cache_item["category_cols"])
            self.primary_category_dim = int(cache_item.get("primary_category_dim", 0))
        self.fingerprint_dim = self.fingerprint.shape[1]
        self.base_num_cols = ["pH"] + self.category_cols
        self.base_num_dim = 1 + len(self.category_cols)
        self.num_cols = self.base_num_cols
        self.num_dim = self.base_num_dim
        self.fingerprint_cols = [f"FP_{i}" for i in range(self.fingerprint_dim)]
        self.ph_scaler = RobustScaler()
        self.logk_scaler = RobustScaler()
        self.fp_scaler = StandardScaler() if fingerprint_scale else None
        self.fit_scalers(np.arange(self.ph.shape[0], dtype=int))

    def __len__(self):
        return int(self.logk_raw.shape[0])

    def fit_scalers(self, train_indices: np.ndarray):
        idx = np.asarray(train_indices, dtype=int).reshape(-1)
        if idx.size == 0:
            idx = np.arange(self.ph.shape[0], dtype=int)
        idx = idx[(idx >= 0) & (idx < self.ph.shape[0])]
        if idx.size == 0:
            idx = np.arange(self.ph.shape[0], dtype=int)
        self.ph_scaler = RobustScaler()
        self.logk_scaler = RobustScaler()
        self.ph_scaler.fit(self.ph[idx])
        self.logk_scaler.fit(self.logk_raw[idx])
        if _config._env_bool_pair("DECAT_RUNTIME_CATEGORY_KINETIC_PRIOR", None, False):
            self._fit_category_kinetic_prior(idx)
        self._fit_structural_clusters(idx)
        self.ph_scaled = self.ph_scaler.transform(self.ph).astype(np.float32)
        self.logk_scaled = self.logk_scaler.transform(self.logk_raw).astype(np.float32)
        if self.fp_scaler is not None:
            self.fp_scaler = StandardScaler()
            self.fp_scaler.fit(self.fingerprint[idx])
            self.fingerprint_scaled = self.fp_scaler.transform(self.fingerprint).astype(
                np.float32
            )
        else:
            self.fingerprint_scaled = self.fingerprint.astype(np.float32)
        return self

    def _fit_structural_clusters(self, train_indices: np.ndarray) -> None:
        cluster_cols = [
            col_idx
            for col_idx, name in enumerate(self.category_cols)
            if name.startswith("structure_cluster_")
        ]
        if not cluster_cols:
            return
        n_clusters = min(len(cluster_cols), max(1, int(train_indices.size)))
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=242,
            n_init=10,
            batch_size=256,
            reassignment_ratio=0.01,
        )
        model.fit(self.fingerprint[train_indices])
        labels = model.predict(self.fingerprint)
        self.category[:, cluster_cols] = 0.0
        for row_idx, label in enumerate(labels):
            self.category[row_idx, cluster_cols[int(label)]] = 1.0

    def _fit_category_kinetic_prior(self, train_indices: np.ndarray) -> None:
        """Cross-fit category-level kinetic context using training targets only."""
        prior_names = (
            "kinetic_category_level",
            "kinetic_ph_response",
            "kinetic_category_reliability",
        )
        try:
            prior_cols = [self.category_cols.index(name) for name in prior_names]
        except ValueError:
            return
        primary_dim = int(getattr(self, "primary_category_dim", 0))
        if primary_dim <= 0:
            return
        category_id = np.argmax(self.category[:, :primary_dim], axis=1).astype(int)
        ph = self.ph.reshape(-1).astype(float)
        target = self.logk_raw.reshape(-1).astype(float)
        train_indices = np.asarray(train_indices, dtype=int).reshape(-1)
        scale = float(np.asarray(self.logk_scaler.scale_).reshape(-1)[0])
        scale = max(1e-06, scale)

        def estimate(fit_idx: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
            fit_idx = np.asarray(fit_idx, dtype=int)
            query_idx = np.asarray(query_idx, dtype=int)
            global_mean = float(np.mean(target[fit_idx]))
            global_ph = float(np.mean(ph[fit_idx]))
            centered_ph = ph[fit_idx] - global_ph
            global_slope = float(
                np.dot(centered_ph, target[fit_idx] - global_mean)
                / (np.dot(centered_ph, centered_ph) + 2.0)
            )
            global_slope = float(np.clip(global_slope, -1.5, 1.5))
            stats = {}
            for category in np.unique(category_id[fit_idx]):
                group_idx = fit_idx[category_id[fit_idx] == category]
                count = int(group_idx.size)
                mean_ph = float(np.mean(ph[group_idx]))
                raw_mean = float(np.mean(target[group_idx]))
                shrink = count / float(count + 12.0)
                mean_y = shrink * raw_mean + (1.0 - shrink) * global_mean
                group_ph = ph[group_idx] - mean_ph
                slope = float(
                    np.dot(group_ph, target[group_idx] - raw_mean)
                    / (np.dot(group_ph, group_ph) + 2.0)
                )
                slope = float(np.clip(slope, -1.5, 1.5))
                reliability = count / float(count + 20.0)
                stats[int(category)] = (mean_y, mean_ph, slope, reliability)
            output = np.zeros((query_idx.size, 3), dtype=np.float32)
            for row, sample_idx in enumerate(query_idx):
                mean_y, mean_ph, slope, reliability = stats.get(
                    int(category_id[sample_idx]),
                    (global_mean, global_ph, global_slope, 0.0),
                )
                output[row, 0] = (mean_y - global_mean) / scale
                output[row, 1] = slope * (ph[sample_idx] - mean_ph) / scale
                output[row, 2] = reliability
            return output

        all_idx = np.arange(self.category.shape[0], dtype=int)
        prior_values = estimate(train_indices, all_idx)
        if train_indices.size >= 10:
            folds = KFold(
                n_splits=min(5, train_indices.size), shuffle=True, random_state=242
            )
            for fit_pos, hold_pos in folds.split(train_indices):
                fit_idx = train_indices[fit_pos]
                hold_idx = train_indices[hold_pos]
                prior_values[hold_idx] = estimate(fit_idx, hold_idx)
        self.category[:, prior_cols] = prior_values

    def __getitem__(self, idx):
        fp = self.fingerprint_scaled[idx].astype(np.float32)
        ph = self.ph_scaled[idx].astype(np.float32)
        cat_vec = self.category[idx].astype(np.float32)
        num_feat = np.concatenate([ph.reshape(-1), cat_vec.reshape(-1)], axis=0).astype(
            np.float32
        )
        y = self.logk_scaled[idx].astype(np.float32)
        y_raw = self.logk_raw[idx].astype(np.float32)
        return {
            "fingerprint": torch.from_numpy(fp),
            "numeric": torch.from_numpy(num_feat),
            "pH": torch.from_numpy(ph).squeeze(-1),
            "category": torch.from_numpy(cat_vec),
            "logk": torch.from_numpy(y).squeeze(-1),
            "logk_raw": torch.from_numpy(y_raw).squeeze(-1),
            "base_idx": torch.tensor(int(idx), dtype=torch.long),
        }


class SelectedFeatureSubset(Dataset):
    """Expose selected fingerprint columns for a fixed set of reaction rows."""

    def __init__(
        self,
        base_ds: FingerprintReactionDataset,
        indices: np.ndarray,
        selected_col_idx: np.ndarray,
    ):
        super().__init__()
        self.base = base_ds
        self.indices = np.asarray(indices, dtype=int)
        self.selected_col_idx = np.asarray(selected_col_idx, dtype=int)
        self.fingerprint_dim = int(self.selected_col_idx.size)
        self.feature_names = [f"fp_{int(i)}" for i in self.selected_col_idx.tolist()]

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, i):
        base_idx = int(self.indices[i])
        fp_full = self.base.fingerprint_scaled[base_idx].astype(np.float32)
        fp_sel = fp_full[self.selected_col_idx]
        ph = self.base.ph_scaled[base_idx].astype(np.float32)
        cat_vec = self.base.category[base_idx].astype(np.float32)
        num_feat = np.concatenate([ph.reshape(-1), cat_vec.reshape(-1)], axis=0).astype(
            np.float32
        )
        y = self.base.logk_scaled[base_idx].astype(np.float32)
        y_raw = self.base.logk_raw[base_idx].astype(np.float32)
        return {
            "fingerprint": torch.from_numpy(fp_sel),
            "numeric": torch.from_numpy(num_feat),
            "pH": torch.from_numpy(ph).squeeze(-1),
            "category": torch.from_numpy(cat_vec),
            "logk": torch.from_numpy(y).squeeze(-1),
            "logk_raw": torch.from_numpy(y_raw).squeeze(-1),
            "base_idx": torch.tensor(base_idx, dtype=torch.long),
        }


def _unwrap_dataset(ds):
    cur = ds
    for _ in range(8):
        if isinstance(
            cur, _runtime.resolve_hook("subset_factory", SelectedFeatureSubset)
        ):
            return cur
        if isinstance(cur, Subset) and hasattr(cur, "dataset"):
            cur = cur.dataset
            continue
        return cur
    return cur


def _get_fp_feature_labels(ds) -> Optional[list]:
    cur = _unwrap_dataset(ds)
    if isinstance(
        cur, _runtime.resolve_hook("subset_factory", SelectedFeatureSubset)
    ) and getattr(cur, "feature_names", None):
        return list(cur.feature_names)
    if hasattr(cur, "fingerprint_dim"):
        try:
            n = int(getattr(cur, "fingerprint_dim"))
            if n > 0:
                return [f"fp_{i}" for i in range(n)]
        except Exception:
            return None
    return None


def _extract_subset_base_indices(subset, fallback_indices: np.ndarray) -> np.ndarray:
    if isinstance(
        subset, _runtime.resolve_hook("subset_factory", SelectedFeatureSubset)
    ):
        return np.asarray(subset.indices, dtype=int)
    if isinstance(subset, Subset) and hasattr(subset, "indices"):
        return np.asarray(subset.indices, dtype=int)
    return np.asarray(fallback_indices, dtype=int)


def _get_base_dataset(ds):
    if isinstance(ds, _runtime.resolve_hook("subset_factory", SelectedFeatureSubset)):
        return ds.base
    if isinstance(ds, Subset) and hasattr(ds, "dataset"):
        return ds.dataset
    return ds
