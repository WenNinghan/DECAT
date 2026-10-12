"""Load trained DECAT assets and predict without a web-framework dependency."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.preprocessing import RobustScaler

from .applicability import evaluate_ad_for_fingerprint


def _load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _restore_scaler(state: dict) -> RobustScaler:
    """Restore fitted statistics without refitting on training records."""
    scaler = RobustScaler(**state["params"])
    scaler.center_ = np.asarray(state["center_"], dtype=state["center_dtype"])
    scaler.scale_ = np.asarray(state["scale_"], dtype=state["scale_dtype"])
    scaler.n_features_in_ = int(state["n_features_in_"])
    return scaler


@dataclass
class FittedPreprocessing:
    """The fitted transformations and class vocabulary needed at inference."""

    ph_scaler: RobustScaler
    logk_scaler: RobustScaler
    category_cols: list[str]
    num_dim: int

    @classmethod
    def from_state(cls, state: dict) -> "FittedPreprocessing":
        return cls(
            ph_scaler=_restore_scaler(state["ph_scaler"]),
            logk_scaler=_restore_scaler(state["logk_scaler"]),
            category_cols=list(state["category_cols"]),
            num_dim=int(state["numeric_dim"]),
        )


def _training_fingerprints(
    records: list[dict], fingerprint: dict, selected_idx: np.ndarray
) -> np.ndarray:
    """Recreate the ordered AD reference space from its public structures."""
    rows = []
    for record in records:
        mol = Chem.MolFromSmiles(record["smiles"])
        if mol is None:
            raise ValueError("An applicability reference contains invalid SMILES.")
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            int(fingerprint["radius"]),
            int(fingerprint["n_bits"]),
            useChirality=bool(fingerprint["use_chirality"]),
        )
        dense = np.zeros(int(fingerprint["n_bits"]), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, dense)
        rows.append(dense[selected_idx] > 0.5)
    return np.asarray(rows, dtype=bool)


def build_numeric_vector(
    dataset, ph_value: float, category27_info: Optional[Dict[str, object]]
) -> np.ndarray:
    """Build the locked 25-class numeric branch: [scaled pH] + one-hot class."""
    ph_arr = np.asarray([[float(ph_value)]], dtype=np.float32)
    ph_scaled = dataset.ph_scaler.transform(ph_arr).reshape(-1).astype(np.float32)
    cat_cols = list(getattr(dataset, "category_cols", []))
    cat_vec = np.zeros((len(cat_cols),), dtype=np.float32)
    info = category27_info or {}
    labels = [
        str(info.get(key, "")).strip()
        for key in (
            "category27_label",
            "label",
            "category27_code",
            "code",
            "category27_name",
            "name_en",
        )
    ]
    for value in labels:
        if not value:
            continue
        candidate = value if value.startswith("cat_") else f"cat_{value}"
        if candidate in cat_cols:
            cat_vec[cat_cols.index(candidate)] = 1.0
            return np.concatenate([ph_scaled, cat_vec], axis=0).astype(np.float32)
    raise ValueError(
        "The reaction class is not part of the locked 25-class DECAT taxonomy."
    )


def configure_runtime(package: Path, config: dict) -> None:
    base_config_path = package / config["runtime"]["base_config_path"]
    base_config = _load_json_file(str(base_config_path))
    environment = {
        str(key): str(value) for key, value in base_config.get("env", {}).items()
    }
    environment.update(
        {
            "DECAT_PROJECT_DIR": str(package),
            "DECAT_RUNTIME_MODE": "fixed_json",
            "DECAT_RUNTIME_FIXED_OBJECTIVE_TARGET": "val",
            "DECAT_RUNTIME_FIXED_SEED": "242",
            "DECAT_RUNTIME_FIXED_SEED_FALLBACK": "242",
            "DECAT_RUNTIME_SEED": "242",
            "DECAT_RUNTIME_ENABLE_FUSION": "1",
            "DECAT_RUNTIME_CALIBRATE_COMPONENTS": "0",
            "DECAT_RUNTIME_CALIBRATE_BLEND": "0",
            "DECAT_RUNTIME_CATEGORY_EMBED_DIM": "0",
        }
    )
    os.environ.update(environment)


def predict_ensemble(bundle: Dict[str, object], x: np.ndarray) -> np.ndarray:
    fp_dim = int(len(bundle["selected_idx"]))
    prediction = np.zeros(len(x), dtype=np.float64)
    device = bundle["device"]
    with torch.no_grad():
        for model, weight in zip(
            bundle["models"], bundle["ensemble_weights"], strict=True
        ):
            scaled = (
                model(
                    torch.from_numpy(x[:, :fp_dim]).to(device),
                    torch.from_numpy(x[:, fp_dim:]).to(device),
                )
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            raw = (
                bundle["dataset"]
                .logk_scaler.inverse_transform(scaled.reshape(-1, 1))
                .reshape(-1)
            )
            prediction += float(weight) * raw
    return prediction.astype(np.float32)


def load_predictor(package_root: Path, artifacts_root: Path):
    """Load the trained Top-3 ensemble and serialized residual RF."""
    package = Path(package_root).resolve()
    artifacts = Path(artifacts_root).resolve()
    config_path = package / "configs" / "decat.json"
    checkpoint_manifest_path = (
        artifacts / "top_checkpoints" / "top_checkpoint_ensemble.json"
    )
    residual_manifest_path = artifacts / "residual_rf" / "residual_rf_manifest.json"
    required_paths = [
        config_path,
        checkpoint_manifest_path,
        residual_manifest_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return None, None, f"Required inference artifacts are missing: {missing}"

    try:
        config = _load_json_file(str(config_path))
        preprocessing_path = package / config["preprocessing"]["path"]
        preprocessing = _load_json_file(str(preprocessing_path))
        ensemble = _load_json_file(str(checkpoint_manifest_path))
        states = list(ensemble.get("states", []))
        weights = np.asarray(ensemble.get("weights", []), dtype=np.float64)
        if (
            len(states) != 3
            or weights.shape != (3,)
            or not np.isclose(weights.sum(), 1.0)
        ):
            raise ValueError(
                "The inference manifest must contain three normalized checkpoint weights."
            )

        configure_runtime(package, config)
        # Model defaults read the configured runtime settings.
        from .config import FingerprintConfig
        from .models.decat import ContextualDualExpertRegressor

        dataset = FittedPreprocessing.from_state(preprocessing)
        fingerprint = preprocessing["fingerprint"]
        fp_bits = int(fingerprint["n_bits"])
        selected_idx = np.asarray(preprocessing["selected_idx"], dtype=int)
        if (
            fp_bits != int(config["params"]["fp_bits"])
            or len(selected_idx) != int(config["params"]["topk_features"])
            or dataset.num_dim != 1 + len(dataset.category_cols)
        ):
            raise ValueError(
                "Fitted preprocessing does not match the model configuration."
            )
        model_config = FingerprintConfig()
        for key, value in config["params"].items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
        model_config.base_numeric_dim = dataset.num_dim

        inference_device = torch.device("cpu")
        models = []
        for state in states:
            checkpoint_path = artifacts / "top_checkpoints" / str(state["file"])
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Model checkpoint is missing: {checkpoint_path}"
                )
            model = ContextualDualExpertRegressor(
                len(selected_idx), dataset.num_dim, model_config
            ).to(inference_device)
            payload = torch.load(
                checkpoint_path, map_location=inference_device, weights_only=True
            )
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
            models.append(model)

        residual_manifest = _load_json_file(str(residual_manifest_path))
        residual_models = []
        for filename in residual_manifest.get("files", []):
            with (artifacts / "residual_rf" / str(filename)).open("rb") as handle:
                residual_models.append(pickle.load(handle))
        if not residual_models:
            raise ValueError(
                "The inference bundle contains no serialized residual-RF model."
            )

        ad_reference = preprocessing["ad_reference"]
        train_fp_selected = _training_fingerprints(
            ad_reference, fingerprint, selected_idx
        )
        bundle = {
            "models": models,
            "ensemble_weights": weights,
            "dataset": dataset,
            "fp_bits": fp_bits,
            "selected_idx": selected_idx,
            "numeric_dim": int(dataset.num_dim),
            "residual_models": residual_models,
            "residual_feature_dim": int(len(selected_idx) + dataset.num_dim),
            "device": inference_device,
            "train_fp_selected_bool": train_fp_selected,
            "train_compounds": [record["compound"] for record in ad_reference],
            "train_smiles": [record["smiles"] for record in ad_reference],
            "params": dict(config["params"]),
            "model_label": "SEEN25-1626 Top-3 validation ensemble + serialized residual RF",
        }
        return bundle, dict(config["params"]), ""
    except Exception as error:
        return None, None, str(error)


def predict(
    bundle: Dict[str, object],
    fp_dense: np.ndarray,
    ph_value: float,
    category27_info: Optional[Dict[str, object]],
    canonical_smiles: str = "",
):
    """Predict one sample using the Top-3 ensemble and residual RF."""
    dataset = bundle["dataset"]
    selected_idx = np.asarray(bundle["selected_idx"], dtype=int).reshape(-1)
    fp_bits = int(bundle["fp_bits"])

    fp_dense = np.asarray(fp_dense, dtype=np.float32).reshape(-1)
    if fp_dense.size != fp_bits:
        raise ValueError(
            f"Fingerprint length mismatch: expected {fp_bits}, received {fp_dense.size}"
        )

    fp_selected = fp_dense[selected_idx].astype(np.float32)
    num_vec = build_numeric_vector(
        dataset, ph_value=float(ph_value), category27_info=category27_info
    )
    if num_vec.size != int(bundle["numeric_dim"]):
        raise ValueError(
            f"Numeric input dimension mismatch: expected {bundle['numeric_dim']}, received {num_vec.size}"
        )

    fusion_features = np.concatenate([fp_selected, num_vec], axis=0).astype(np.float32)
    if fusion_features.size != int(bundle["residual_feature_dim"]):
        raise ValueError("Residual input dimension does not match the trained model")
    external_x = fusion_features.reshape(1, -1)
    pred_raw_nn = float(predict_ensemble(bundle, external_x)[0])
    residual_input = np.c_[external_x, [pred_raw_nn]]
    residual = float(
        np.mean(
            [model.predict(residual_input)[0] for model in bundle["residual_models"]]
        )
    )
    pred_raw = pred_raw_nn + residual
    ad_detail = evaluate_ad_for_fingerprint(bundle, fp_dense, ph_value=ph_value)
    detail = {
        "pred_scaled": None,
        "fp_bits": fp_bits,
        "topk_features": int(selected_idx.size),
        "max_fp_tokens": int(bundle.get("params", {}).get("max_fp_tokens", 0)),
        "n_heads": int(bundle.get("params", {}).get("n_heads", 0)),
        "n_layers": int(bundle.get("params", {}).get("n_layers", 0)),
        "mode": str(bundle.get("mode", "dual")),
        "nn_prediction": pred_raw_nn,
        "rf_residual": residual,
        "model_label": str(bundle.get("model_label", "DECAT locked model")),
        **ad_detail,
    }
    return pred_raw, detail
