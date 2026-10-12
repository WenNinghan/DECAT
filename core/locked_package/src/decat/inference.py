"""Load trained DECAT assets and predict without a web-framework dependency."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .applicability import evaluate_ad_for_fingerprint
from .input_chemistry import read_csv_robust


def _load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
            "DECAT_RUNTIME_DATA_CSV": str(package / config["data"]["path"]),
            "DECAT_RUNTIME_FIXED_SPLIT_JSON": str(package / config["split"]["path"]),
            "DECAT_RUNTIME_FIXED_SPLIT_JSON_FALLBACK": str(
                package / config["split"]["path"]
            ),
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
        # Model defaults and dataset options read the configured runtime settings.
        from .config import FingerprintConfig
        from .data.dataset import FingerprintReactionDataset
        from .data.features import _get_fp_ranking
        from .models.decat import ContextualDualExpertRegressor

        split = _load_json_file(str(package / config["split"]["path"]))
        train_idx = np.asarray(split["train_idx"], dtype=int)
        fp_bits = int(config["params"]["fp_bits"])
        dataset = FingerprintReactionDataset(
            str(package / config["data"]["path"]), fp_bits
        )
        dataset.fit_scalers(train_idx)
        selected_idx = np.asarray(
            _get_fp_ranking(dataset, train_idx, fp_bits, "rf")[
                : int(config["params"]["topk_features"])
            ],
            dtype=int,
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

        train_fp_selected = np.asarray(
            dataset.fingerprint[train_idx][:, selected_idx] > 0.5, dtype=bool
        )
        df_ref = read_csv_robust(str(package / config["data"]["path"]))
        compound_col = next(
            (
                column
                for column in ["chemical compound", "chemical_compound", "compound"]
                if column in df_ref.columns
            ),
            None,
        )
        smiles_col = "SMILES" if "SMILES" in df_ref.columns else "smiles"
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
            "train_idx": train_idx,
            "train_fp_selected_bool": train_fp_selected,
            "train_compounds": (
                df_ref.iloc[train_idx][compound_col].astype(str).tolist()
                if compound_col
                else []
            ),
            "train_smiles": (
                df_ref.iloc[train_idx][smiles_col].astype(str).tolist()
                if smiles_col
                else []
            ),
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
