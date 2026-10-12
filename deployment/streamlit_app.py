"""Predict aqueous ozone reaction-rate constants with the trained DECAT ensemble."""
import os
import re
import io
import base64
import json
import pickle
import importlib.util
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from html import escape

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import quote
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Draw
import torch
import torch.nn as nn
from streamlit.runtime.scriptrunner import get_script_run_ctx

SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR.parent
DEPLOYMENT_ARTIFACTS = SCRIPT_DIR / "artifacts"
RUNTIME_PACKAGE = RELEASE_ROOT / "core" / "locked_package"
RUNTIME_CONFIG_PATH = RUNTIME_PACKAGE / "configs" / "LOCKED_SEEN25_1626_NONUNIFORM_SOTA.json"
RUNTIME_DATASET_PATH = RUNTIME_PACKAGE / "data" / "decat_features_1626.csv"
RUNTIME_SPLIT_PATH = RUNTIME_PACKAGE / "data" / "split_seed242_1626.json"
PAPER_REBUILD_PACKAGE = RUNTIME_PACKAGE
PAPER_REBUILD_CONFIG_PATH = RUNTIME_CONFIG_PATH
PAPER_REBUILD_ROOT = DEPLOYMENT_ARTIFACTS
PAPER_TOP3_MANIFEST_PATH = DEPLOYMENT_ARTIFACTS / "top_checkpoints" / "top_checkpoint_ensemble.json"
PAPER_RESIDUAL_MANIFEST_PATH = DEPLOYMENT_ARTIFACTS / "residual_rf" / "residual_rf_manifest.json"
CATEGORY_OVERRIDES_PATH = SCRIPT_DIR / "data" / "category_overrides.csv"

RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.info')
FP_SIZE = 3147
RADIUS = 2

# Training leave-one-out distance thresholds in the selected fingerprint space.
AD_STRICT_DISTANCE_THRESHOLD = 0.6428571428571428
AD_BORDERLINE_DISTANCE_THRESHOLD = 0.8
PH_TRAIN_MIN = 1.0
PH_TRAIN_MAX = 12.0

LOCKED_CATEGORY27_LABELS = {
    "A": "A: Other aqueous environmental species",
    "B": "B: Alcohol",
    "C": "C: Diol",
    "D": "D: Ether",
    "E": "E: Ketone",
    "F": "F: Aldehyde",
    "G": "G: Ester",
    "H": "H: Carboxyl",
    "I": "I: Dicarboxylic",
    "J": "J: Halogenated",
    "K": "K: Sulfide, disulfide",
    "L": "L: Oxidized sulfur compounds",
    "M": "M: Thiol",
    "N": "N: Nitrile",
    "O": "O: Nitro",
    "P": "P: Amide",
    "Q": "Q: Amine",
    "R": "R: Nitroso, nitramine",
    "S": "S: Phosphorus",
    "T": "T: Cyclo",
    "U": "U: Alkene",
    "V": "V: Residual aromatic and phenolic compounds",
    "W": "W: Six-membered N heteroaromatics",
    "Y": "Y: Urea",
    "Z": "Z: Five-membered N heteroaromatics (>=2 N)",
}
LOCKED_CATEGORY27_OPTIONS = list(LOCKED_CATEGORY27_LABELS.values())


def _load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_robust(path, **kwargs):
    """Read a CSV with common encoding fallbacks."""
    kwargs = dict(kwargs)
    kwargs.setdefault("low_memory", False)
    encodings_order = ["utf-8", "utf-8-sig", "gb18030", "gbk", "cp936", "latin1"]
    last_err = None
    for enc in encodings_order:
        try:
            df_tmp = pd.read_csv(path, encoding=enc, **kwargs)
            return df_tmp
        except UnicodeDecodeError as e:
            last_err = e
        except Exception as e:
            last_err = e

    try:
        with open(path, "rb") as f:
            raw = f.read()
        for enc in encodings_order:
            try:
                text = raw.decode(enc, errors="ignore")
                buf = io.StringIO(text)
                safe_kwargs = {k: v for k, v in kwargs.items() if k != "encoding"}
                df_tmp = pd.read_csv(buf, **safe_kwargs)
                return df_tmp
            except Exception:
                continue
    except Exception as e:
        last_err = e
    raise last_err or UnicodeDecodeError("Cannot decode the CSV file")


@st.cache_data(ttl=3600)
def get_cas_from_chemsrc(chemical_name):
    """Query Chemsrc for a CAS number."""
    if not chemical_name or chemical_name.strip() == "":
        return None
    try:
        encoded_name = quote(chemical_name.strip())
        url = f"https://www.chemsrc.com/searchResult/{encoded_name}/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        match = re.search(r"\b\d{2,7}-\d{2}-\d\b", soup.get_text())
        if match:
            return match.group()
        return None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_cas_from_pubchem(chemical_name):
    """Extract a CAS number from PubChem synonyms."""
    if not chemical_name or str(chemical_name).strip() == "":
        return None
    try:
        encoded = quote(str(chemical_name).strip())
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/cids/TXT"
        cid_resp = requests.get(cid_url, timeout=12)
        if cid_resp.status_code != 200:
            return None
        cids = [line.strip() for line in cid_resp.text.splitlines() if line.strip().isdigit()]
        if not cids:
            return None
        syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids[0]}/synonyms/JSON"
        syn_resp = requests.get(syn_url, timeout=12)
        if syn_resp.status_code != 200:
            return None
        data = syn_resp.json()
        info_list = data.get("InformationList", {}).get("Information", [])
        if not info_list:
            return None
        for syn in info_list[0].get("Synonym", []) or []:
            match = re.fullmatch(r"\d{2,7}-\d{2}-\d", str(syn).strip())
            if match:
                return match.group()
        return None
    except Exception:
        return None


def get_cas_number(chemical_name):
    """Query Chemsrc first, then fall back to PubChem."""
    cas = get_cas_from_chemsrc(chemical_name)
    if cas:
        return cas, "Chemsrc"
    cas = get_cas_from_pubchem(chemical_name)
    if cas:
        return cas, "PubChem"
    return None, ""


@st.cache_data(ttl=3600)
def get_smiles_via_api(compound_name, cas_number=None):
    """Resolve SMILES through PubChem or CACTUS using a name or CAS number."""
    name = (compound_name or "").strip()
    cas = (cas_number or "").strip()
    if name == "" and cas == "":
        return None

    def _canonicalize_smiles(smiles_text):
        if not smiles_text:
            return None
        try:
            mol = Chem.MolFromSmiles(str(smiles_text).strip())
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    def _read_pubchem_smiles_json(url):
        resp = requests.get(url, timeout=12)
        if resp.status_code != 200:
            return None
        data = resp.json()
        compounds = data.get("PC_Compounds", [])
        if not compounds:
            return None
        props = compounds[0].get("props", [])
        for prop in props:
            urn = prop.get("urn", {})
            if urn.get("label") == "SMILES" and urn.get("name") in ("Absolute", "Canonical", "Connectivity"):
                val = prop.get("value", {}).get("sval")
                cano = _canonicalize_smiles(val)
                if cano:
                    return cano
        return None

    def _read_smiles_txt(url):
        resp = requests.get(url, timeout=12)
        if resp.status_code != 200:
            return None
        txt = resp.text.strip().splitlines()
        if not txt:
            return None
        return _canonicalize_smiles(txt[0].strip())

    def _query_pubchem_name(q):
        encoded = quote(q)
        urls = [
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/TXT",
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/JSON",
        ]
        for _ in range(2):
            for u in urls:
                try:
                    smi = _read_smiles_txt(u) if u.endswith("/TXT") else _read_pubchem_smiles_json(u)
                    if smi:
                        return smi
                except Exception:
                    continue
        return None

    def _query_pubchem_cas(cas_q):
        encoded = quote(cas_q)
        urls = [
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RN/{encoded}/property/CanonicalSMILES/TXT",
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/TXT",
        ]
        for _ in range(2):
            for u in urls:
                try:
                    smi = _read_smiles_txt(u)
                    if smi:
                        return smi
                except Exception:
                    continue
        return None

    def _query_cactus(q):
        encoded = quote(q)
        url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded}/smiles"
        for _ in range(2):
            try:
                resp = requests.get(url, timeout=12)
                if resp.status_code != 200:
                    continue
                text = resp.text.strip()
                if not text or "Page not found" in text or "Status: 404" in text:
                    continue
                smi = _canonicalize_smiles(text.splitlines()[0])
                if smi:
                    return smi
            except Exception:
                continue
        return None


    if name:
        smi = _query_pubchem_name(name)
        if smi:
            return smi
    if cas:
        smi = _query_pubchem_cas(cas)
        if smi:
            return smi
    if name:
        smi = _query_cactus(name)
        if smi:
            return smi
    if cas:
        smi = _query_cactus(cas)
        if smi:
            return smi

    st.warning("SMILES lookup failed: neither PubChem nor CACTUS returned a valid structure. Please check the network, DNS, or compound name.")
    return None


def smiles_to_fingerprint(smiles, fp_size=FP_SIZE):
    """Convert SMILES to a Morgan fingerprint."""
    if pd.isna(smiles) or str(smiles).strip() == "":
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        gen_custom = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=int(fp_size))
        fp = gen_custom.GetFingerprint(mol)
        return np.array(list(fp), dtype=np.float32)
    except Exception as e:
        st.warning(f"Fingerprint generation failed: {str(e)}")
        return None


def _load_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {module_name} <- {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_v9_numeric_vector(dataset, ph_value: float, category27_info: Optional[Dict[str, object]]) -> np.ndarray:
    """Build the locked 25-class numeric branch: [scaled pH] + one-hot class."""
    ph_arr = np.asarray([[float(ph_value)]], dtype=np.float32)
    ph_scaled = dataset.ph_scaler.transform(ph_arr).reshape(-1).astype(np.float32)
    cat_cols = list(getattr(dataset, "category_cols", []))
    cat_vec = np.zeros((len(cat_cols),), dtype=np.float32)
    info = category27_info or {}
    labels = [
        str(info.get(key, "")).strip()
        for key in ("category27_label", "label", "category27_code", "code", "category27_name", "name_en")
    ]
    for value in labels:
        if not value:
            continue
        candidate = value if value.startswith("cat_") else f"cat_{value}"
        if candidate in cat_cols:
            cat_vec[cat_cols.index(candidate)] = 1.0
            return np.concatenate([ph_scaled, cat_vec], axis=0).astype(np.float32)
    raise ValueError("The reaction class is not part of the locked 25-class DECAT taxonomy.")


def _tanimoto_dense_matrix(query_fp: np.ndarray, train_fps: np.ndarray) -> np.ndarray:
    query = np.asarray(query_fp, dtype=bool).reshape(1, -1)
    train = np.asarray(train_fps, dtype=bool)
    inter = np.logical_and(train, query).sum(axis=1).astype(np.float32)
    union = np.logical_or(train, query).sum(axis=1).astype(np.float32)
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return sim.astype(np.float32)


def evaluate_ad_for_fingerprint(bundle: Dict[str, object], fp_dense: np.ndarray, ph_value: Optional[float] = None) -> Dict[str, object]:
    """Evaluate DECAT applicability-domain status using locked V9 structural thresholds."""
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
    final_label = "Out of AD - pH outside train range" if ph_label == "outside" else structural_label
    train_names = list(bundle.get("train_compounds", []))
    train_smiles = list(bundle.get("train_smiles", []))
    return {
        "nearest_train_similarity": best_sim,
        "nearest_train_distance": distance,
        "nearest_train_index": int(best_i),
        "nearest_train_compound": train_names[best_i] if 0 <= best_i < len(train_names) else "",
        "nearest_train_SMILES": train_smiles[best_i] if 0 <= best_i < len(train_smiles) else "",
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


def _configure_paper_external_runtime(package: Path, config: dict) -> None:
    base_config_path = package / config["runtime"]["base_config_path"]
    base_config = _load_json_file(str(base_config_path))
    environment = {str(key): str(value) for key, value in base_config.get("env", {}).items()}
    environment.update(
        {
            "DECAT_PROJECT_DIR": str(package),
            "TRANSFORMER_V7_DATA_CSV": str(package / config["data"]["path"]),
            "TRANSFORMER_V7_FIXED_SPLIT_JSON": str(package / config["split"]["path"]),
            "TRANSFORMER_V6_FIXED_SPLIT_JSON": str(package / config["split"]["path"]),
            "TRANSFORMER_V7_MODE": "fixed_json",
            "TRANSFORMER_V7_FIXED_OBJECTIVE_TARGET": "val",
            "TRANSFORMER_V7_FIXED_SEED": "242",
            "TRANSFORMER_V6_FIXED_SEED": "242",
            "TRANSFORMER_SEED": "242",
            "TRANSFORMER_V3_ENABLE_FUSION": "1",
            "TRANSFORMER_V3_CALIBRATE_COMPONENTS": "0",
            "TRANSFORMER_V3_CALIBRATE_BLEND": "0",
            "TRANSFORMER_V9_CATEGORY_EMBED_DIM": "0",
        }
    )
    os.environ.update(environment)


def _build_paper_contextual_model(v9_mod, fingerprint_dim: int, numeric_dim: int, model_config):
    class ContextualDualExpertRegressor(v9_mod.DualExpertRegressor):
        def __init__(self, fp_dim: int, num_dim: int, config_obj):
            super().__init__(fp_dim, num_dim, config_obj)
            hidden = max(16, int(config_obj.d_model // 4))
            self.contextual_residual = nn.Sequential(
                nn.Linear(num_dim + 1, hidden),
                nn.GELU(),
                nn.Dropout(float(config_obj.dropout)),
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

    return ContextualDualExpertRegressor(fingerprint_dim, numeric_dim, model_config)


def _paper_top3_prediction(bundle: Dict[str, object], x: np.ndarray) -> np.ndarray:
    fp_dim = int(len(bundle["selected_idx"]))
    prediction = np.zeros(len(x), dtype=np.float64)
    device = bundle["device"]
    with torch.no_grad():
        for model, weight in zip(bundle["models"], bundle["ensemble_weights"], strict=True):
            scaled = model(
                torch.from_numpy(x[:, :fp_dim]).to(device),
                torch.from_numpy(x[:, fp_dim:]).to(device),
            ).detach().cpu().numpy().reshape(-1)
            raw = bundle["dataset"].logk_scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(-1)
            prediction += float(weight) * raw
    return prediction.astype(np.float32)


@st.cache_resource
def load_decat_v9_bundle(model_path: str, params_path: str, dataset_path: str, summary_path: str = ""):
    """Load the trained Top-3 ensemble and serialized residual RF."""
    required_paths = [
        PAPER_REBUILD_CONFIG_PATH,
        PAPER_TOP3_MANIFEST_PATH,
        PAPER_RESIDUAL_MANIFEST_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return None, None, f"Required inference artifacts are missing: {missing}"

    try:
        package = PAPER_REBUILD_PACKAGE
        config = _load_json_file(str(PAPER_REBUILD_CONFIG_PATH))
        ensemble = _load_json_file(str(PAPER_TOP3_MANIFEST_PATH))
        states = list(ensemble.get("states", []))
        weights = np.asarray(ensemble.get("weights", []), dtype=np.float64)
        if len(states) != 3 or weights.shape != (3,) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("The inference manifest must contain three normalized checkpoint weights.")

        _configure_paper_external_runtime(package, config)
        runtime_path = package / "src" / "decat" / "transformer_v9_transformer_centered.py"
        v9_mod = _load_module_from_file("_streamlit_paper_seen25_runtime", str(runtime_path))
        split = _load_json_file(str(package / config["split"]["path"]))
        train_idx = np.asarray(split["train_idx"], dtype=int)
        fp_bits = int(config["params"]["fp_bits"])
        dataset = v9_mod.FingerprintReactionDataset(str(package / config["data"]["path"]), fp_bits)
        dataset.fit_scalers(train_idx)
        selected_idx = np.asarray(
            v9_mod._get_fp_ranking(dataset, train_idx, fp_bits, "rf")[: int(config["params"]["topk_features"])],
            dtype=int,
        )
        model_config = v9_mod.FingerprintConfig()
        for key, value in config["params"].items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
        model_config.base_numeric_dim = dataset.num_dim

        inference_device = torch.device("cpu")
        models = []
        for state in states:
            checkpoint_path = PAPER_REBUILD_ROOT / "top_checkpoints" / str(state["file"])
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Model checkpoint is missing: {checkpoint_path}")
            model = _build_paper_contextual_model(
                v9_mod, len(selected_idx), dataset.num_dim, model_config
            ).to(inference_device)
            payload = torch.load(checkpoint_path, map_location=inference_device, weights_only=True)
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
            models.append(model)

        residual_manifest = _load_json_file(str(PAPER_RESIDUAL_MANIFEST_PATH))
        residual_models = []
        for filename in residual_manifest.get("files", []):
            with (PAPER_REBUILD_ROOT / "residual_rf" / str(filename)).open("rb") as handle:
                residual_models.append(pickle.load(handle))
        if not residual_models:
            raise ValueError("The inference bundle contains no serialized residual-RF model.")

        train_fp_selected = np.asarray(dataset.fingerprint[train_idx][:, selected_idx] > 0.5, dtype=bool)
        df_ref = read_csv_robust(str(package / config["data"]["path"]))
        compound_col = next((column for column in ["chemical compound", "chemical_compound", "compound"] if column in df_ref.columns), None)
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
            "train_compounds": df_ref.iloc[train_idx][compound_col].astype(str).tolist() if compound_col else [],
            "train_smiles": df_ref.iloc[train_idx][smiles_col].astype(str).tolist() if smiles_col else [],
            "params": dict(config["params"]),
            "model_label": "SEEN25-1626 Top-3 validation ensemble + serialized residual RF",
        }
        return bundle, dict(config["params"]), ""
    except Exception as error:
        return None, None, str(error)


def predict_with_decat_v9(
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
        raise ValueError(f"Fingerprint length mismatch: expected {fp_bits}, received {fp_dense.size}")

    fp_selected = fp_dense[selected_idx].astype(np.float32)
    num_vec = _build_v9_numeric_vector(dataset, ph_value=float(ph_value), category27_info=category27_info)
    if num_vec.size != int(bundle["numeric_dim"]):
        raise ValueError(f"Numeric input dimension mismatch: expected {bundle['numeric_dim']}, received {num_vec.size}")

    fusion_features = np.concatenate([fp_selected, num_vec], axis=0).astype(np.float32)
    if fusion_features.size != int(bundle["residual_feature_dim"]):
        raise ValueError("Residual input dimension does not match the trained model")
    external_x = fusion_features.reshape(1, -1)
    pred_raw_nn = float(_paper_top3_prediction(bundle, external_x)[0])
    residual_input = np.c_[external_x, [pred_raw_nn]]
    residual = float(np.mean([model.predict(residual_input)[0] for model in bundle["residual_models"]]))
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


def _canonicalize_smiles_text(smiles_text: object) -> Tuple[str, str]:
    text = "" if smiles_text is None else str(smiles_text).strip()
    if not text or text.lower() == "nan":
        return "", "missing"
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return "", "invalid"
        return Chem.MolToSmiles(mol), "valid"
    except Exception:
        return "", "invalid"


def normalize_category27_input(category_input):
    """Map a user-provided hint to the locked SEEN25-1626 class taxonomy."""
    if pd.isna(category_input):
        return None
    text = str(category_input).strip()
    if text == "":
        return None
    low = text.lower()
    for code, label in LOCKED_CATEGORY27_LABELS.items():
        name = label.split(":", 1)[1].strip()
        if low in {code.lower(), label.lower(), name.lower()}:
            return locked_category_info(code, "Selected from the locked 25-class taxonomy.", "manual_input")
    return None


def locked_category_info(category_label: Optional[str], reason: str = "", source: str = ""):
    label = str(category_label or "").strip()
    if not label:
        return None
    if label.startswith("cat_"):
        label = label[4:]
    code, separator, name = label.partition(":")
    code = code.strip()
    if code not in LOCKED_CATEGORY27_LABELS:
        return None
    locked_label = LOCKED_CATEGORY27_LABELS[code]
    item = {
        "code": code,
        "name_en": locked_label.split(":", 1)[1].strip(),
        "category27_code": code,
        "category27_name": locked_label.split(":", 1)[1].strip(),
        "category27_label": locked_label,
        "label": locked_label,
    }
    if reason:
        item["category27_reason"] = str(reason)
    if source:
        item["source"] = source
    return item


@st.cache_data(show_spinner=False)
def load_category_overrides(path: str) -> Dict[str, str]:
    """Load molecule-specific class assignments used by the predictor."""
    if not path or not os.path.exists(path):
        return {}
    frame = read_csv_robust(path)
    if not {"smiles", "category27_label"}.issubset(frame.columns):
        return {}
    lookup = {}
    for _, row in frame[["smiles", "category27_label"]].dropna().iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]).strip())
        if mol is None:
            continue
        category = locked_category_info(str(row["category27_label"]).strip())
        if category:
            lookup[Chem.MolToSmiles(mol, isomericSmiles=True)] = category["category27_label"]
    return lookup


def _has_smarts(mol, pattern: str) -> bool:
    query = Chem.MolFromSmarts(pattern)
    return bool(query is not None and mol.HasSubstructMatch(query))


def _classify_locked_category_from_mol(mol) -> Tuple[str, str]:
    """Assign one of the exact 25 model classes using the training-label contract."""
    carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    if carbon_atoms == 0:
        return "A", "No carbon skeleton; assigned to aqueous environmental species."
    if _has_smarts(mol, "[PX3,PX4,PX5]"):
        return "S", "Contains a phosphorus-centered functional group."
    carboxyl_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=[OX1])[O;H1,-1]")))
    if carboxyl_count >= 2:
        return "I", "Contains two or more carboxyl/carboxylate groups."
    if carboxyl_count == 1:
        return "H", "Contains a carboxyl/carboxylate group."
    if _has_smarts(mol, "[NX3][CX3](=[OX1])[NX3]"):
        return "Y", "Contains a urea-type N-C(=O)-N motif."
    if _has_smarts(mol, "[S](=[OX1])(=[OX1])") or _has_smarts(mol, "[S](=[OX1])"):
        return "L", "Contains an oxidized sulfur functional group."
    if _has_smarts(mol, "[SX2H]"):
        return "M", "Contains a thiol group."
    if _has_smarts(mol, "[SX2,SX1]"):
        return "K", "Contains a sulfide or disulfide sulfur center."
    if _has_smarts(mol, "[N+](=O)[O-]"):
        return "O", "Contains a nitro group."
    if _has_smarts(mol, "[N]=O") or _has_smarts(mol, "[N][N](=O)"):
        return "R", "Contains a nitroso or nitramine motif."
    if _has_smarts(mol, "[C]#[N]"):
        return "N", "Contains a nitrile group."
    if _has_smarts(mol, "[NX3][CX3](=[OX1])"):
        return "P", "Contains an amide-type N-C(=O) motif."
    if _has_smarts(mol, "[CX3](=[OX1])[OX2]"):
        return "G", "Contains an ester-type C(=O)-O motif."
    if _has_smarts(mol, "[CX3H1](=[OX1])"):
        return "F", "Contains an aldehyde carbonyl."
    if _has_smarts(mol, "[CX3](=[OX1])([#6])[#6]"):
        return "E", "Contains a ketone carbonyl."

    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms = [mol.GetAtomWithIdx(index) for index in ring]
        nitrogen_count = sum(atom.GetAtomicNum() == 7 for atom in ring_atoms)
        if len(ring) == 6 and nitrogen_count >= 1 and all(atom.GetIsAromatic() for atom in ring_atoms):
            return "W", "Contains a six-membered aromatic N-heterocycle."
        if len(ring) == 5 and nitrogen_count >= 2 and all(atom.GetIsAromatic() for atom in ring_atoms):
            return "Z", "Contains a five-membered aromatic N-heterocycle with at least two nitrogens."
    if _has_smarts(mol, "[NX3;!$(N[CX3](=O));!$(N=S)]"):
        return "Q", "Contains a non-amide amine nitrogen."
    if any(atom.GetAtomicNum() in {9, 17, 35, 53} for atom in mol.GetAtoms()):
        return "J", "Contains a carbon-bound halogen without a higher-priority group."
    if _has_smarts(mol, "[CX3;!$(C=O)]-[OX2]-[CX3;!$(C=O)]"):
        return "D", "Contains an ether-type C-O-C linkage."
    hydroxyl_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H][#6]")))
    if hydroxyl_count >= 2:
        return "C", "Contains two or more alcohol-type hydroxyl groups."
    if hydroxyl_count == 1:
        return "B", "Contains one alcohol-type hydroxyl group."
    if any(bond.GetBondType() == Chem.BondType.DOUBLE and not bond.GetIsAromatic()
           and bond.GetBeginAtom().GetAtomicNum() == 6 and bond.GetEndAtom().GetAtomicNum() == 6
           for bond in mol.GetBonds()):
        return "U", "Contains a non-aromatic carbon-carbon double bond."
    if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
        return "V", "Residual aromatic or phenolic compound without a higher-priority group."
    if mol.GetRingInfo().NumRings() > 0:
        return "T", "Non-aromatic cyclic compound without a higher-priority group."
    return "A", "No higher-priority organic functional class was identified."


def classify_category27_from_smiles(smiles: Optional[str], manual_input: Optional[str] = None):
    """Assign a valid locked 25-class label from SMILES or an explicit class hint."""
    manual_info = normalize_category27_input(manual_input)
    if manual_info:
        return manual_info
    if not smiles or not str(smiles).strip():
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    override_label = load_category_overrides(str(CATEGORY_OVERRIDES_PATH)).get(canonical)
    if override_label:
        return locked_category_info(
            override_label,
            "Molecule-specific class assignment from the inference category map.",
            "category_override",
        )
    code, reason = _classify_locked_category_from_mol(mol)
    return locked_category_info(code, reason, "locked_smiles_rule")


@st.cache_data(ttl=3600)
def load_reference_dataset(dataset_path: Optional[str] = None):
    """Load the reference dataset columns used by the prediction form."""
    try:
        ref_path = dataset_path or str(RUNTIME_DATASET_PATH)
        df = read_csv_robust(ref_path)
        col_lower = {col: col.strip().lower() for col in df.columns}
        def pick(*names):
            normalized_names = [str(name).strip().lower().replace(" ", "") for name in names]
            for name in names:
                for col, low in col_lower.items():
                    if low == str(name).strip().lower() or low.replace(" ", "") in normalized_names:
                        return col
            return None
        compound_col = pick("chemical compound", "chemical_compound", "compound")
        cas_col = pick("cas", "cas number", "CAS")
        smiles_col = pick("SMILES")
        category_col = pick("category")
        category_name_col = pick("new_category_name", "family12_category_name", "category_name")
        category_code_col = pick("new_category_code", "family12_category_code", "category_code")
        category27_scheme_col = pick("category27_scheme")
        category27_code_col = pick("category27_code")
        category27_name_col = pick("category27_name")
        category27_label_col = pick("category27_label")
        category27_reason_col = pick("category27_reason")
        if not compound_col:
            st.warning("The reference dataset does not contain a 'chemical compound' column, so dataset pre-matching was skipped.")
            return None

        fp_cols = []
        for col in df.columns:
            if re.fullmatch(r"FP[_\s-]*\d+", str(col).strip(), flags=re.IGNORECASE):
                fp_cols.append(col)
        if fp_cols:
            fp_cols.sort(key=lambda x: int(re.findall(r"\d+", str(x))[0]))

        keep_cols = [
            c
            for c in [
                compound_col, cas_col, smiles_col, category_col, category_name_col, category_code_col,
                category27_scheme_col, category27_code_col, category27_name_col, category27_label_col, category27_reason_col,
            ]
            if c
        ] + fp_cols
        df = df[keep_cols]
        rename_map = {compound_col: "compound"}
        if cas_col:
            rename_map[cas_col] = "cas"
        if smiles_col:
            rename_map[smiles_col] = "smiles"
        if category_col:
            rename_map[category_col] = "category"
        if category_name_col:
            rename_map[category_name_col] = "family12_name"
        if category_code_col:
            rename_map[category_code_col] = "family12_code"
        if category27_scheme_col:
            rename_map[category27_scheme_col] = "category27_scheme"
        if category27_code_col:
            rename_map[category27_code_col] = "category27_code"
        if category27_name_col:
            rename_map[category27_name_col] = "category27_name"
        if category27_label_col:
            rename_map[category27_label_col] = "category27_label"
        if category27_reason_col:
            rename_map[category27_reason_col] = "category27_reason"
        df = df.rename(columns=rename_map)
        return df
    except Exception as e:
        st.warning(f"Reference dataset loading failed: {e}")
        return None


def find_compound_record(name, df):
    """Find a reference record matching the supplied chemical name."""
    if df is None or not name:
        return None
    name_norm = str(name).strip().lower()
    df_local = df.copy()
    df_local["compound_norm"] = df_local["compound"].astype(str).str.strip().str.lower()
    hit = df_local[df_local["compound_norm"] == name_norm]
    if hit.empty:
        return None
    row = hit.iloc[0]
    category27_info = locked_category_info(
        row.get("category27_label"),
        str(row.get("category27_reason") or "Loaded from the reference dataset."),
        "reference_dataset",
    )


    fp_pairs = []
    for col in row.index:
        m = re.fullmatch(r"fp[_\s-]*(\d+)", str(col).strip().lower())
        if not m:
            continue
        try:
            fp_idx = int(m.group(1))
            fp_val = float(row.get(col))
        except Exception:
            fp_val = 0.0
            fp_idx = int(m.group(1))
        if np.isnan(fp_val):
            fp_val = 0.0
        fp_pairs.append((fp_idx, fp_val))
    fp_vec = None
    if fp_pairs:
        fp_pairs.sort(key=lambda x: x[0])
        fp_vec = np.array([v for _, v in fp_pairs], dtype=np.float32)
        if fp_vec.size <= 0:
            fp_vec = None

    return {
        "cas": row.get("cas"),
        "smiles": row.get("smiles"),
        "fingerprint": fp_vec,
        "category27_info": category27_info,
    }


def inject_custom_styles():
    st.markdown(
        """
        <style>
            :root {
                --bg: #f5f7fa;
                --bg-2: #ffffff;
                --panel: #ffffff;
                --panel-strong: #ffffff;
                --border: rgba(0, 51, 102, 0.2);
                --text: #000000;
                --muted: #555555;
                --cyan: #0066cc;
                --teal: #0066cc;
                --gold: #ff8c00;
                --gold-soft: rgba(255, 140, 0, 0.1);
                --success: #228b22;
                --warning: #ff8c00;
                --danger: #dc143c;
                --info: #0066cc;
                --shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                --radius-xl: 8px;
                --radius-lg: 6px;
                --radius-md: 4px;
            }

            .stApp {
                background: linear-gradient(180deg, #f5f7fa 0%, #ffffff 100%);
                color: var(--text);
                font-family: "Times New Roman", Times, serif;
            }

            [data-testid="stAppViewContainer"] > .main {
                background: #ffffff;
                padding: 0;
            }

            [data-testid="stMarkdownContainer"] {
                font-family: "Times New Roman", Times, serif !important;
                font-size: 16pt !important;
            }

            [data-testid="stMarkdownContainer"] h3 {
                font-family: "Times New Roman", Times, serif !important;
                font-size: 24pt !important;
                font-weight: bold !important;
                color: #003366 !important;
                margin-bottom: 1.5rem !important;
                padding-bottom: 0.8rem !important;
                border-bottom: 4px solid #0066cc !important;
            }

            [data-testid="stMarkdownContainer"] strong {
                font-family: "Times New Roman", Times, serif !important;
                font-size: 16pt !important;
                font-weight: bold !important;
            }

            [data-testid="stMarkdownContainer"] p {
                font-family: "Times New Roman", Times, serif !important;
                font-size: 16pt !important;
                margin: 0.5rem 0 !important;
                line-height: 1.6 !important;
            }

            [data-testid="stHeader"] {
                background: transparent;
                display: none;
            }

            [data-testid="stToolbar"] {
                display: none;
            }

            #MainMenu, footer {
                visibility: hidden;
            }

            .block-container {
                padding: 1.5rem 3rem;
                max-width: 100%;
            }

            [data-testid="stSidebar"] {
                display: none;
            }

            .hero-shell {
                background: linear-gradient(135deg, #003366 0%, #004d99 100%);
                color: #ffffff;
                padding: 2rem 2rem;
                margin: -1.5rem -2rem 2rem -2rem;
                border-radius: 0;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            }

            .hero-title {
                margin: 0;
                font-size: 32pt;
                font-weight: bold;
                color: #ffffff;
                font-family: "Times New Roman", Times, serif;
                text-align: center;
                letter-spacing: 0.5px;
            }

            .hero-subtitle {
                display: none;
            }

            .section-label {
                font-size: 16pt;
                font-weight: bold;
                color: var(--text);
                margin: 1rem 0 0.5rem 0;
                font-family: "Times New Roman", Times, serif;
            }

            .grid-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
                margin-bottom: 1.5rem;
            }

            .section-box {
                background: var(--panel);
                border: 2px solid var(--border);
                border-radius: 4px;
                padding: 1.2rem;
            }

            .section-title {
                font-size: 14pt;
                font-weight: bold;
                color: var(--text);
                margin: 0 0 0.8rem 0;
                font-family: "Times New Roman", Times, serif;
            }

            .input-group {
                margin-bottom: 0.8rem;
            }

            .input-label {
                font-size: 11pt;
                font-weight: bold;
                color: var(--text);
                margin-bottom: 0.3rem;
                display: block;
                font-family: "Times New Roman", Times, serif;
            }

            .result-row {
                display: flex;
                justify-content: space-between;
                padding: 0.4rem 0;
                border-bottom: 1px solid var(--border);
                font-size: 11pt;
                font-family: "Times New Roman", Times, serif;
            }

            .result-row:last-child {
                border-bottom: none;
            }

            .result-key {
                font-weight: bold;
                color: var(--text);
            }

            .result-value {
                color: var(--muted);
                text-align: right;
                word-break: break-word;
                max-width: 60%;
            }

            .prediction-box {
                background: #e8f4f8;
                border: 3px solid var(--teal);
                border-radius: 4px;
                padding: 1.5rem;
                text-align: center;
            }

            .prediction-label {
                font-size: 12pt;
                font-weight: bold;
                color: var(--text);
                margin-bottom: 0.5rem;
                font-family: "Times New Roman", Times, serif;
            }

            .prediction-value {
                font-size: 36pt;
                font-weight: bold;
                color: var(--teal);
                margin: 0.3rem 0;
                font-family: "Times New Roman", Times, serif;
            }

            .prediction-unit {
                font-size: 12pt;
                color: var(--muted);
                font-family: "Times New Roman", Times, serif;
            }

            [data-testid="stAppViewContainer"] > .main {
                background: transparent;
            }

            [data-testid="stHeader"] {
                background: transparent;
                display: none;
            }

            [data-testid="stToolbar"] {
                display: none;
            }

            #MainMenu, footer {
                visibility: hidden;
            }

            .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
                max-width: 1200px;
            }

            [data-testid="stSidebar"] {
                display: none;
            }

            .hero-shell {
                position: relative;
                overflow: hidden;
                padding: 1.1rem 1.5rem;
                border-radius: 6px;
                border: 2px solid var(--border);
                background: #ffffff;
                box-shadow: none;
                margin-bottom: 1.15rem;
            }

            .hero-shell::before,
            .hero-shell::after {
                display: none;
            }

            @keyframes floatOrb {
                0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
                50% { transform: translate3d(10px, -12px, 0) scale(1.06); }
            }

            .hero-kicker {
                display: none;
            }

            .hero-title {
                margin: 0;
                font-size: 24pt;
                line-height: 1.2;
                letter-spacing: -0.01em;
                color: var(--text);
                font-family: "Times New Roman", Times, serif;
                font-weight: bold;
                text-align: center;
            }

            .hero-subtitle {
                display: block;
                margin-top: 0.55rem;
                color: var(--muted);
                font-size: 12.5pt;
                line-height: 1.4;
                text-align: center;
                font-family: "Times New Roman", Times, serif;
            }

            .hero-status {
                margin-top: 0.6rem;
                text-align: center;
                font-size: 11pt;
                color: var(--teal);
                font-weight: bold;
                font-family: "Times New Roman", Times, serif;
            }

            .badge-row {
                display: none;
            }

            .status-pill {
                display: none;
            }

            .status-pill strong {
                display: none;
            }

            .insight-card,
            .control-card,
            .step-card,
            .result-card,
            .taxonomy-card,
            .feature-card,
            .alert-banner,
            .workflow-card {
                display: none;
            }

            .detail-card {
                position: relative;
                overflow: hidden;
                border-radius: 4px;
                border: 2px solid var(--border);
                background: var(--panel);
                box-shadow: none;
            }

            .feature-card {
                padding: 1.5rem;
                min-height: 100px;
                transition: none;
            }

            .feature-card:hover,
            .insight-card:hover,
            .result-card:hover,
            .taxonomy-card:hover {
                transform: none;
                border-color: var(--border);
                box-shadow: none;
            }

            .feature-icon,
            .insight-label,
            .result-label,
            .step-index {
                font-size: 11pt;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                color: var(--teal);
                font-weight: bold;
                font-family: "Times New Roman", Times, serif;
            }

            .feature-title,
            .insight-title,
            .section-title,
            .result-value {
                color: var(--text);
                font-family: "Times New Roman", Times, serif;
            }

            .feature-title {
                font-size: 16pt;
                font-weight: bold;
                margin: 0.6rem 0 0.4rem;
            }

            .feature-desc,
            .insight-desc,
            .section-desc,
            .step-message,
            .taxonomy-desc,
            .result-meta {
                color: var(--muted);
                line-height: 1.5;
                font-size: 11pt;
                font-family: "Times New Roman", Times, serif;
            }

            .section-shell {
                margin: 1.2rem 0 0.8rem;
            }

            .section-kicker {
                display: none;
            }

            .section-title {
                font-size: 20pt;
                margin: 0 0 1rem 0;
                letter-spacing: 0;
                font-weight: bold;
                font-family: "Times New Roman", Times, serif;
            }

            .section-desc {
                display: none;
            }

            .control-card,
            .insight-card,
            .result-card,
            .taxonomy-card {
                display: none;
            }

            .insight-value {
                font-size: 18pt;
                font-weight: bold;
                color: var(--text);
                margin: 0.4rem 0;
                font-family: "Times New Roman", Times, serif;
            }

            .step-card {
                padding: 1.2rem;
                margin-bottom: 0.8rem;
            }

            .step-card.success { border-color: var(--success); background: #ffffff; }
            .step-card.warn { border-color: var(--warning); background: #ffffff; }
            .step-card.error { border-color: var(--danger); background: #ffffff; }
            .step-card.info { border-color: var(--info); background: #ffffff; }

            .step-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
                margin-bottom: 0.4rem;
            }

            .step-title {
                font-size: 14pt;
                font-weight: bold;
                color: var(--text);
                font-family: "Times New Roman", Times, serif;
            }

            .step-state {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.4rem 0.8rem;
                border-radius: 4px;
                font-size: 10pt;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-family: "Times New Roman", Times, serif;
            }

            .step-state.success { background: var(--success); color: #ffffff; }
            .step-state.warn { background: var(--warning); color: #ffffff; }
            .step-state.error { background: var(--danger); color: #ffffff; }
            .step-state.info { background: var(--info); color: #ffffff; }

            .result-card.main-result {
                display: block !important;
                padding: 2.5rem;
                background: linear-gradient(135deg, #e8f4f8 0%, #d4ebf7 100%);
                border: 4px solid var(--teal);
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 8px 24px rgba(0, 102, 204, 0.2);
                margin-top: 1.5rem;
            }

            .result-value {
                font-size: 56pt;
                font-weight: bold;
                line-height: 1.1;
                margin: 0.5rem 0;
                letter-spacing: 0;
                color: var(--teal);
                font-family: "Times New Roman", Times, serif;
                text-align: center;
                text-shadow: 2px 2px 4px rgba(0, 102, 204, 0.1);
            }

            .result-unit {
                color: var(--muted);
                font-size: 16pt;
                letter-spacing: 0;
                font-weight: normal;
                font-family: "Times New Roman", Times, serif;
            }

            .result-label {
                font-size: 18pt;
                font-weight: bold;
                color: #003366;
                margin-bottom: 1.15rem;
                font-family: "Times New Roman", Times, serif;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .result-grid-note {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-top: 1.2rem;
            }

            .workflow-shell {
                margin: 0.8rem 0 1rem;
            }

            .workflow-card {
                min-height: 146px;
                padding: 0.9rem 0.95rem;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }

            .workflow-step {
                font-size: 10.5pt;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-weight: bold;
                font-family: "Times New Roman", Times, serif;
            }

            .workflow-title {
                margin-top: 0.35rem;
                font-size: 13pt;
                line-height: 1.25;
                color: var(--text);
                font-weight: bold;
                font-family: "Times New Roman", Times, serif;
            }

            .workflow-detail {
                margin-top: 0.7rem;
                font-size: 10.5pt;
                line-height: 1.35;
                color: var(--muted);
                font-family: "Times New Roman", Times, serif;
            }

            .workflow-state {
                align-self: flex-start;
                margin-top: 0.85rem;
                padding: 0.32rem 0.65rem;
                border-radius: 999px;
                font-size: 9.5pt;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #ffffff;
                font-family: "Times New Roman", Times, serif;
            }

            .workflow-card.pending .workflow-state { background: #7a7a7a; }
            .workflow-card.success .workflow-state { background: var(--success); }
            .workflow-card.warn .workflow-state { background: var(--warning); }
            .workflow-card.error .workflow-state { background: var(--danger); }
            .workflow-card.info .workflow-state { background: var(--info); }

            .workflow-card.pending { border-color: rgba(0, 0, 0, 0.18); }
            .workflow-card.success { border-color: var(--success); }
            .workflow-card.warn { border-color: var(--warning); }
            .workflow-card.error { border-color: var(--danger); }
            .workflow-card.info { border-color: var(--info); }

            .detail-card {
                margin-bottom: 1.15rem;
                background: var(--panel);
                border: 2px solid var(--border);
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: var(--shadow);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }

            .detail-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 16px rgba(0, 102, 204, 0.15);
            }

            .detail-title {
                margin: 0 0 1rem 0;
                font-size: 18pt;
                line-height: 1.2;
                font-weight: bold;
                color: var(--text);
                font-family: "Times New Roman", Times, serif;
            }

            .detail-grid {
                display: grid;
                gap: 0.6rem;
            }

            .detail-row {
                display: grid;
                grid-template-columns: 140px 1fr;
                gap: 0.8rem;
                align-items: start;
                padding: 0.4rem 0;
                border-bottom: 1px solid var(--border);
            }

            .detail-row:last-child {
                border-bottom: none;
            }

            .detail-key {
                font-size: 14pt;
                line-height: 1.4;
                color: var(--text);
                font-weight: bold;
                font-family: "Times New Roman", Times, serif;
            }

            .detail-value {
                font-size: 14pt;
                line-height: 1.4;
                color: var(--muted);
                word-break: break-word;
                font-family: "Times New Roman", Times, serif;
            }

            .input-note {
                margin: 0.35rem 0 0.2rem;
                font-size: 10.5pt;
                line-height: 1.35;
                color: var(--muted);
                font-family: "Times New Roman", Times, serif;
            }

            .taxonomy-code {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem 1rem;
                border-radius: 4px;
                background: var(--gold-soft);
                color: var(--gold);
                font-weight: bold;
                margin-bottom: 1.15rem;
                font-size: 12pt;
                font-family: "Times New Roman", Times, serif;
            }

            .alert-banner {
                padding: 0.9rem 1rem;
                margin-top: 0.7rem;
            }

            .alert-banner.error {
                border-color: var(--danger);
                background: #ffffff;
            }

            .alert-banner.success {
                border-color: var(--success);
                background: #ffffff;
            }

            [data-testid="stForm"],
            [data-testid="stVerticalBlockBorderWrapper"]:has(.control-card) {
                background: transparent !important;
                border: none !important;
            }

            .stTextInput > div > div,
            .stNumberInput > div > div,
            .stSelectbox > div > div,
            .stTextArea textarea {
                background: #ffffff !important;
                border: 2px solid var(--border) !important;
                border-radius: 6px !important;
                color: var(--text) !important;
                font-family: "Times New Roman", Times, serif !important;
                font-size: 16pt !important;
                padding: 0.8rem !important;
            }

            .stTextInput input,
            .stNumberInput input {
                font-size: 17pt !important;
                font-weight: bold !important;
                line-height: 1.1 !important;
                height: auto !important;
                min-height: 0 !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                font-family: "Times New Roman", Times, serif !important;
            }

            .stTextInput [data-baseweb="input"],
            .stNumberInput [data-baseweb="input"] {
                min-height: 3.05rem !important;
                height: 3.05rem !important;
                display: flex !important;
                align-items: center !important;
                overflow: visible !important;
            }

            .stTextInput label,
            .stNumberInput label,
            .stFileUploader label,
            .stRadio label,
            .stSelectbox label {
                color: var(--text) !important;
                font-weight: bold !important;
                font-size: 18pt !important;
                font-family: "Times New Roman", Times, serif !important;
            }

            .stButton > button {
                min-height: 4rem;
                border: none;
                border-radius: 8px;
                background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
                color: #ffffff;
                font-weight: bold;
                letter-spacing: 0.5px;
                font-size: 18pt;
                box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
                transition: all 0.3s ease;
                font-family: "Times New Roman", Times, serif;
                width: 100%;
            }

            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 102, 204, 0.4);
                background: linear-gradient(135deg, #0052a3 0%, #003d7a 100%);
            }

            [data-testid="stForm"],
            [data-testid="stVerticalBlockBorderWrapper"] {
                background: transparent !important;
                border: none !important;
            }

            div[data-testid="stDataFrame"] {
                border-radius: 4px;
                overflow: hidden;
                border: 1px solid var(--border);
                font-family: "Times New Roman", Times, serif;
            }

            [data-testid="stMetric"] {
                background: var(--panel);
                border: 1px solid var(--border);
                padding: 0.8rem;
                border-radius: 4px;
            }

            [data-testid="stMetricLabel"] {
                font-size: 10pt !important;
                font-weight: bold !important;
                color: var(--muted) !important;
                font-family: "Times New Roman", Times, serif !important;
            }

            [data-testid="stMetricValue"] {
                font-size: 16pt !important;
                font-weight: bold !important;
                color: var(--text) !important;
                font-family: "Times New Roman", Times, serif !important;
            }

            .stTabs [data-baseweb="tab-list"] {
                display: none;
            }

            .stCaptionContainer, .stCaption {
                font-size: 9pt !important;
                color: var(--muted) !important;
                font-family: "Times New Roman", Times, serif !important;
            }

            .small-caption {
                font-size: 9pt;
                color: var(--muted);
                font-family: "Times New Roman", Times, serif;
            }

            @media (max-width: 900px) {
                .grid-container {
                    grid-template-columns: 1fr;
                }
            }

            @media (max-width: 900px) {
                .hero-shell {
                    padding: 1rem;
                }
                .section-title {
                    font-size: 1.42rem;
                }
                .workflow-card {
                    min-height: 128px;
                }
                .detail-row {
                    grid-template-columns: 1fr;
                    gap: 0.15rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(dataset_ready: bool):
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #003366 0%, #004d99 100%);
                    color: white;
                    padding: 1.5rem 2rem;
                    margin: -1.5rem -2rem 2rem -2rem;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
            <h1 style="font-size: 28pt;
                       font-weight: bold;
                       margin: 0;
                       font-family: 'Times New Roman', Times, serif;
                       letter-spacing: 1px;
                       white-space: nowrap;">
                Ozone Reaction Rate Constant (logk) Prediction System
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_board(step_events: List[Dict[str, str]]):
    base_steps = [
        (1, "Compound Query", "Input compound name and pH."),
        (2, "CAS Retrieval", "Resolve the registry identifier."),
        (3, "Structure Retrieval", "Obtain the canonical SMILES string."),
        (4, "Class Assignment", "Assign the reaction class automatically."),
        (5, "Fingerprint Encoding", "Build the Morgan fingerprint."),
        (6, "logk Prediction", "Return the final rate constant."),
        (7, "Prediction Range", "Assess the compound's applicability domain against the training data."),
    ]
    event_map = {int(item.get("index", 0)): item for item in step_events if item.get("index")}
    cols = st.columns(len(base_steps), gap="small")
    for col, (index, title, default_detail) in zip(cols, base_steps):
        item = event_map.get(index, {})
        tone = str(item.get("tone", "pending")).lower()
        if tone == "warning":
            tone = "warn"
        if tone not in {"pending", "success", "warn", "error", "info"}:
            tone = "info" if item else "pending"
        state = str(item.get("state", "Pending" if not item else "Ready"))
        detail = re.sub(r"\s+", " ", str(item.get("message") or default_detail)).strip()
        if len(detail) > 86:
            detail = detail[:83].rstrip() + "..."
        with col:
            st.markdown(
                f"""
                <div class="workflow-card {escape(tone)}">
                    <div>
                        <div class="workflow-step">Step {index}</div>
                        <div class="workflow-title">{escape(title)}</div>
                        <div class="workflow-detail">{escape(detail)}</div>
                    </div>
                    <div class="workflow-state">{escape(state)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_detail_card(title: str, rows: List[Tuple[str, str]]):
    st.markdown(
        f"""
        <div style="background: white;
                    border: 3px solid #003366;
                    border-radius: 8px;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <h3 style="font-size: 22pt;
                       font-weight: bold;
                       color: #003366;
                       margin: 0 0 1rem 0;
                       padding-bottom: 0.5rem;
                       border-bottom: 3px solid #0066cc;
                       font-family: 'Times New Roman', Times, serif;">
                {title}
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for key, value in rows:
        display_value = "Not available" if value is None or str(value).strip() == "" else str(value)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"<p style='font-size: 16pt; font-weight: bold; margin: 0.5rem 0; font-family: Times New Roman;'>{key}:</p>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<p style='font-size: 16pt; margin: 0.5rem 0; font-family: Times New Roman;'>{display_value}</p>", unsafe_allow_html=True)


def render_alert_banner(message: str, tone: str = "error"):
    st.markdown(
        f"""
        <div class="alert-banner {escape(tone)}">
            <div class="step-title">{escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_range_card(detail: Dict[str, object]):
    ad_label = str(detail.get("DECAT_AD_label", "Not assessed"))
    ad_color = "#dc3545"
    st.markdown(
        f"""
        <div style="background: #fff5f5;
                    border-left: 4px solid {ad_color};
                    padding: 1rem 1rem;
                    margin-top: 0.8rem;
                    margin-bottom: 0.8rem;
                    min-height: 160px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;">
            <div style="font-size: 17pt;
                        font-weight: 800;
                        color: #003366;
                        line-height: 1.2;
                        margin-bottom: 0.5rem;
                        font-family: 'Times New Roman', Times, serif;">
                Prediction Applicability Domain
            </div>
            <div style="font-size: 18pt;
                        font-weight: 800;
                        color: {ad_color};
                        font-family: 'Times New Roman', Times, serif;">
                {escape(ad_label)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def category27_reason_for_display(info: Optional[Dict[str, str]]) -> str:
    if not info:
        return "Not available."
    return str(
        info.get("category27_reason")
        or info.get("reason")
        or info.get("reason_raw")
        or "Category27 assignment completed."
    )


def category27_label_for_display(info: Optional[Dict[str, str]]) -> str:
    if not info:
        return "Not assigned"
    label = str(info.get("category27_label") or info.get("label") or "").strip()
    if label:
        return label
    code = str(info.get("category27_code") or info.get("code") or "").strip()
    name = str(info.get("category27_name") or info.get("name_en") or "").strip()
    return f"{code}: {name}".strip(": ").strip() if (code or name) else "Not assigned"


def main():
    st.set_page_config(page_title="Reaction Rate Constant Prediction Workflow", page_icon="📄", layout="wide")
    inject_custom_styles()

    reference_dataset_path = str(RUNTIME_DATASET_PATH)

    render_hero(dataset_ready=os.path.exists(reference_dataset_path))
    if not os.path.exists(reference_dataset_path):
        render_alert_banner(
            "The reference dataset is unavailable. Restore the dataset before running a prediction.",
            tone="error",
        )


    main_cols = st.columns([1.05, 1.3, 1.25], gap="large", vertical_alignment="top")

    with main_cols[0]:
        st.markdown("<h2 style='text-align: center; font-size: 22pt; font-weight: 800; color: #003366; font-family: Times New Roman; margin-bottom: 1.15rem;'>(a) Input</h2>", unsafe_allow_html=True)
        query_params = st.query_params
        auto_compound = str(query_params.get("compound", "") or "").strip()
        auto_cas = str(query_params.get("cas", "") or "").strip()
        try:
            auto_ph = float(query_params.get("ph", 7.0))
        except Exception:
            auto_ph = 7.0
        auto_run = str(query_params.get("auto_run", "") or "").strip().lower() in {"1", "true", "yes"}

        # Oxidant system info box
        st.markdown(
            """
            <div style="background: #f0f4f8;
                        border: 2px solid #003366;
                        border-radius: 6px;
                        padding: 1rem;
                        margin-bottom: 1.15rem;">
                <div style="font-size: 16pt;
                            font-weight: 800;
                            color: #003366;
                            margin-bottom: 0.5rem;
                            font-family: 'Times New Roman', Times, serif;">
                    Oxidant system
                </div>
                <div style="font-size: 15.5pt;
                            font-weight: bold;
                            color: #000;
                            font-family: 'Times New Roman', Times, serif;">
                    Ozone (O₃)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chemical_name = st.text_input("Chemical name or CAS", value=auto_compound, placeholder="e.g., acetaminophen", label_visibility="visible")
        ph_value = st.number_input("pH", min_value=0.0, max_value=14.0, value=auto_ph, step=0.1, label_visibility="visible")
        category_choice = st.selectbox(
            "Reaction class override",
            options=["Automatic assignment"] + LOCKED_CATEGORY27_OPTIONS,
            index=0,
        )
        category_input = "" if category_choice == "Automatic assignment" else category_choice

        # Add spacing before button
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("Run Prediction", type="primary", disabled=not chemical_name.strip(), use_container_width=True) or (auto_run and bool(chemical_name.strip()))
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        structure_placeholder = st.empty()

    dataset_record = None
    if chemical_name.strip() and os.path.exists(reference_dataset_path):
        try:
            ref_df = load_reference_dataset(reference_dataset_path)
            dataset_record = find_compound_record(chemical_name, ref_df)
        except Exception:
            dataset_record = None

    dataset_source = "Reference dataset" if dataset_record else ""
    preloaded_cas = dataset_record.get("cas") if dataset_record else None
    if not preloaded_cas and auto_cas:
        preloaded_cas = auto_cas
    preloaded_smiles = dataset_record.get("smiles") if dataset_record else None
    preloaded_category27 = dict(dataset_record.get("category27_info")) if (dataset_record and dataset_record.get("category27_info")) else None
    preview_category27 = preloaded_category27 or normalize_category27_input(category_input)
    preview_class_label = category27_label_for_display(preview_category27) if preview_category27 else "Automatic assignment"

    with main_cols[1]:
        st.markdown("<h2 style='text-align: center; font-size: 22pt; font-weight: 800; color: #003366; font-family: Times New Roman; margin-bottom: 1.15rem;'>(c) Processing</h2>", unsafe_allow_html=True)

        if predict_btn and chemical_name.strip():
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #228b22; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ✓ CAS Retrieval
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #555; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Querying chemical database...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #228b22; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ✓ SMILES Retrieval
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #555; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Fetching molecular structure...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #228b22; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ✓ Fingerprint Generation
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #555; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Computing Morgan fingerprint...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #228b22; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ✓ Classification
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #555; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Assigning reaction class...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #228b22; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ✓ Model Prediction
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #555; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Running Transformer model...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #228b22; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ✓ Prediction Range
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #555; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Checking applicability domain...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            prediction_range_placeholder = st.empty()
        else:
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #999999; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ○ CAS Retrieval
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #b0b0b0; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Pending...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            prediction_range_placeholder = st.empty()
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #999999; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ○ SMILES Retrieval
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #b0b0b0; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Pending...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #999999; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ○ Fingerprint Generation
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #b0b0b0; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Pending...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #999999; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ○ Classification
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #b0b0b0; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Pending...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #999999; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ○ Model Prediction
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #b0b0b0; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Pending...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-bottom: 1.05rem; min-height: 76px;">
                    <div style="font-size: 20pt; font-weight: 800; color: #999999; font-family: 'Times New Roman', Times, serif; line-height: 1.18; margin-bottom: 0.35rem;">
                        ○ Prediction Range
                    </div>
                    <div style="font-size: 16pt; font-weight: bold; color: #b0b0b0; font-family: 'Times New Roman', Times, serif; line-height: 1.35; margin-left: 1.5rem;">
                        Pending...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with main_cols[2]:
        st.markdown("<h2 style='text-align: center; font-size: 22pt; font-weight: 800; color: #003366; font-family: Times New Roman; margin-bottom: 1.15rem;'>(d) Results</h2>", unsafe_allow_html=True)

    if not predict_btn:
        # Show placeholder structure in Input column
        with structure_placeholder:
            st.markdown(
                """
                <div style="background: #f8f9fa;
                            border: 2px dashed #ccc;
                            border-radius: 6px;
                            padding: 1.5rem;
                            text-align: center;
                            margin-bottom: 1.15rem;">
                    <div style="font-size: 22pt;
                                font-weight: 800;
                                color: #003366;
                                line-height: 1.15;
                                font-family: 'Times New Roman', Times, serif;">
                        (b) Structure
                    </div>
                    <div style="font-size: 12pt;
                                color: #bbb;
                                font-family: 'Times New Roman', Times, serif;
                                margin-top: 0.5rem;">
                        Run prediction to confirm the resolved structure
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with main_cols[2]:
            st.markdown(
                """
                <div style="background: #f8f9fa;
                            border: 2px dashed #ccc;
                            border-radius: 8px;
                            padding: 2rem;
                            text-align: center;
                            margin-top: 2rem;">
                    <div style="font-size: 18pt;
                                color: #999;
                                font-family: 'Times New Roman', Times, serif;
                                margin-bottom: 0.5rem;">
                        Awaiting prediction...
                    </div>
                    <div style="font-size: 12pt;
                                color: #bbb;
                                font-family: 'Times New Roman', Times, serif;">
                        Enter compound information and click "Run Prediction"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    step_events = [
        {
            "index": 1,
            "title": "Compound query",
            "state": "Success",
            "tone": "success",
            "message": f"Received compound input '{chemical_name.strip()}' at pH {ph_value:.1f}.",
        }
    ]
    cas = preloaded_cas
    cas_source = dataset_source if cas else ""
    smiles = preloaded_smiles
    smiles_source = dataset_source if smiles else ""
    category27_info = preloaded_category27

    def render_failure_view(message: str):
        render_pipeline_board(step_events)
        result_cols = st.columns([1.05, 0.95], gap="medium")
        with result_cols[0]:
            render_detail_card(
                "Workflow Snapshot",
                [
                    ("Compound", chemical_name.strip()),
                    ("Resolved CAS", cas or "Not found"),
                    ("Resolved SMILES", smiles or "Not found"),
                    ("Reaction class", category27_label_for_display(category27_info) if category27_info else "Not assigned"),
                ],
            )
        with result_cols[1]:
            render_alert_banner(message, tone="error")

    with st.spinner("Running the prediction workflow..."):
        if cas:
            step_events.append(
                {
                    "index": 2,
                    "title": "CAS retrieval",
                    "state": "Success",
                    "tone": "success",
                    "message": f"Matched CAS from the reference dataset: {cas}.",
                }
            )
        else:
            cas, cas_source = get_cas_number(chemical_name)
            if cas:
                step_events.append(
                    {
                        "index": 2,
                        "title": "CAS retrieval",
                        "state": "Success",
                        "tone": "success",
                        "message": f"Retrieved CAS via {cas_source}: {cas}.",
                    }
                )
            else:
                step_events.append(
                    {
                        "index": 2,
                        "title": "CAS retrieval",
                        "state": "Fallback",
                        "tone": "warn",
                        "message": "No CAS record was found, so the workflow continues with direct structure retrieval.",
                    }
                )

        if smiles:
            step_events.append(
                {
                    "index": 3,
                    "title": "Structure retrieval",
                    "state": "Success",
                    "tone": "success",
                    "message": "Loaded SMILES from the reference dataset.",
                }
            )
        else:
            smiles = get_smiles_via_api(chemical_name, cas_number=cas)
            if smiles:
                smiles_source = "Online lookup"
                step_events.append(
                    {
                        "index": 3,
                        "title": "Structure retrieval",
                        "state": "Success",
                        "tone": "success",
                        "message": "Retrieved a valid SMILES string through online lookup.",
                    }
                )
            else:
                step_events.append(
                    {
                        "index": 3,
                        "title": "Structure retrieval",
                        "state": "Error",
                        "tone": "error",
                        "message": "No reliable SMILES string was found for the current compound.",
                    }
                )
                render_failure_view(
                    "The workflow stopped because no reliable SMILES string could be obtained from the reference dataset or online retrieval."
                )
                return

        # Display molecular structure in Input column
        if smiles:
            rendered_structure = False
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(320, 220))
                    img_buf = io.BytesIO()
                    img.save(img_buf, format="PNG")
                    img_b64 = base64.b64encode(img_buf.getvalue()).decode("ascii")
                    with structure_placeholder:
                        st.markdown(
                            f"""
                            <div style="background: #ffffff;
                                        border: 1px solid rgba(0, 51, 102, 0.18);
                                        border-radius: 6px;
                                        padding: 1rem 0.9rem;
                                        margin-top: 0.5rem;
                                        min-height: 244px;
                                        text-align: center;">
                                <div style="font-size: 22pt;
                                            font-weight: 800;
                                            color: #003366;
                                            font-family: 'Times New Roman', Times, serif;
                                            line-height: 1.15;
                                            margin-bottom: 0.55rem;">
                                    (b) Structure
                                </div>
                                <img src="data:image/png;base64,{img_b64}"
                                     style="display:block; width:100%; max-width:280px; height:auto; margin:0 auto;" />
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    rendered_structure = True
            except Exception:
                rendered_structure = False
            if not rendered_structure:
                with structure_placeholder:
                    render_alert_banner("Structure rendering failed. Please verify the resolved SMILES in the Results panel.", tone="warn")

        category27_info = classify_category27_from_smiles(smiles, manual_input=category_input) or category27_info
        if not category27_info:
            step_events.append(
                {
                    "index": 4,
                    "title": "Class assignment",
                    "state": "Error",
                    "tone": "error",
                    "message": "Automatic class assignment did not return a valid reaction class.",
                }
            )
            render_failure_view(
                "The workflow stopped because the reaction class could not be assigned from the current structure or the optional class hint."
            )
            return

        step_events.append(
            {
                "index": 4,
                "title": "Class assignment",
                "state": "Success",
                "tone": "success",
                "message": f"Assigned reaction class: {category27_label_for_display(category27_info)}.",
            }
        )

        target_fp_size = FP_SIZE
        try:
            params_preview = _load_json_file(str(RUNTIME_CONFIG_PATH))["params"]
            target_fp_size = int(params_preview["fp_bits"])
        except (OSError, KeyError, TypeError, ValueError):
            pass

        fp = dataset_record.get("fingerprint") if dataset_record else None
        if fp is not None and int(np.asarray(fp).reshape(-1).size) == int(target_fp_size):
            fp = np.asarray(fp, dtype=np.float32).reshape(-1)
            step_events.append(
                {
                    "index": 5,
                    "title": "Fingerprint encoding",
                    "state": "Success",
                    "tone": "success",
                    "message": f"Reused the matched {len(fp)}-bit fingerprint directly from the dataset.",
                }
            )
        else:
            if fp is not None:
                step_events.append(
                    {
                        "index": 5,
                        "title": "Fingerprint encoding",
                        "state": "Fallback",
                        "tone": "warn",
                        "message": f"The matched fingerprint length did not fit the required {target_fp_size} bits, so a new fingerprint was rebuilt.",
                    }
                )
            fp = smiles_to_fingerprint(smiles, fp_size=target_fp_size)
            if fp is None:
                step_events.append(
                    {
                        "index": 5,
                        "title": "Fingerprint encoding",
                        "state": "Error",
                        "tone": "error",
                        "message": "Fingerprint generation failed for the available structure.",
                    }
                )
                render_failure_view(
                    "The workflow stopped because RDKit could not construct a valid fingerprint from the retrieved SMILES string."
                )
                return
            if not any(event.get("index") == 5 and event.get("tone") == "warn" for event in step_events):
                step_events.append(
                    {
                        "index": 5,
                        "title": "Fingerprint encoding",
                        "state": "Success",
                        "tone": "success",
                        "message": f"Generated a {target_fp_size}-bit Morgan fingerprint from the resolved structure.",
                    }
                )

        bundle, t_params, load_err = load_decat_v9_bundle(
            model_path="",
            params_path="",
            dataset_path=reference_dataset_path,
            summary_path="",
        )
        if bundle is None:
            step_events.append(
                {
                    "index": 6,
                    "title": "logk prediction",
                    "state": "Error",
                    "tone": "error",
                    "message": f"Prediction bundle loading failed: {load_err}",
                }
            )
            render_failure_view(f"The workflow stopped because the prediction bundle could not be loaded: {load_err}")
            return

        fp_bits_required = int(bundle["fp_bits"])
        if int(np.asarray(fp).reshape(-1).size) != fp_bits_required:
            fp = smiles_to_fingerprint(smiles, fp_size=fp_bits_required)
            if fp is None:
                step_events.append(
                    {
                        "index": 6,
                        "title": "logk prediction",
                        "state": "Error",
                        "tone": "error",
                        "message": "The final model-compatible fingerprint could not be reconstructed.",
                    }
                )
                render_failure_view(
                    "The workflow stopped because the model-compatible fingerprint could not be reconstructed from the retrieved structure."
                )
                return

        logk_pred, detail = predict_with_decat_v9(
            bundle=bundle,
            fp_dense=fp,
            ph_value=ph_value,
            category27_info=category27_info,
            canonical_smiles=_canonicalize_smiles_text(smiles)[0],
        )
        step_events.append(
            {
                "index": 6,
                "title": "logk prediction",
                "state": "Success",
                "tone": "success",
                "message": "The prediction model completed inference and returned the final logk estimate.",
            }
        )
        range_source = str(detail.get("AD_source", "") or "")
        step_events.append(
            {
                "index": 7,
                "title": "prediction range",
                "state": "Online assessment",
                "tone": "success" if str(detail.get("DECAT_AD_label", "")) == "In AD" else ("warn" if "Borderline" in str(detail.get("DECAT_AD_label", "")) else "error"),
                "message": range_source or "Applicability was assessed against the training fingerprint space.",
            }
        )
        with prediction_range_placeholder.container():
            render_prediction_range_card(detail)

    class_line = category27_label_for_display(category27_info)
    assignment_basis = category27_reason_for_display(category27_info)

    # Display results in the third column
    with main_cols[2]:
        # Chemical name
        st.markdown(
            f"""
            <div style="background: #f8f9fa;
                        border-left: 4px solid #003366;
                        padding: 0.9rem 1rem;
                        margin-bottom: 0.8rem;
                        min-height: 106px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #003366;
                            line-height: 1.2;
                            margin-bottom: 0.35rem;
                            font-family: 'Times New Roman', Times, serif;">
                    Chemical
                </div>
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #000;
                            line-height: 1.25;
                            font-family: 'Times New Roman', Times, serif;">
                    {escape(chemical_name.strip())}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # CAS
        st.markdown(
            f"""
            <div style="background: #f8f9fa;
                        border-left: 4px solid #003366;
                        padding: 0.9rem 1rem;
                        margin-bottom: 0.8rem;
                        min-height: 106px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #003366;
                            line-height: 1.2;
                            margin-bottom: 0.35rem;
                            font-family: 'Times New Roman', Times, serif;">
                    CAS
                </div>
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #000;
                            line-height: 1.25;
                            font-family: 'Times New Roman', Times, serif;">
                    {escape(cas if cas else 'Not found')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # k and Logk (prominent display with red color)
        k_value = 10 ** logk_pred if logk_pred is not None else None
        st.markdown(
            f"""
            <div style="background: #fff5f5;
                        border-left: 4px solid #dc3545;
                        padding: 0.9rem 1rem;
                        margin-bottom: 0.8rem;
                        min-height: 142px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #003366;
                            line-height: 1.2;
                            margin-bottom: 0.5rem;
                            font-family: 'Times New Roman', Times, serif;">
                    Rate Constant
                </div>
                <div style="font-size: 23pt;
                            font-weight: 800;
                            color: #dc3545;
                            font-family: 'Times New Roman', Times, serif;
                            line-height: 1.22;
                            margin-bottom: 0.3rem;">
                    k = {escape(f"{k_value:.4e}" if k_value is not None else "N/A")} M⁻¹s⁻¹
                </div>
                <div style="font-size: 23pt;
                            font-weight: 800;
                            color: #dc3545;
                            line-height: 1.22;
                            font-family: 'Times New Roman', Times, serif;">
                    logk = {escape(f"{logk_pred:.4f}" if logk_pred is not None else "N/A")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Class
        st.markdown(
            f"""
            <div style="background: #f8f9fa;
                        border-left: 4px solid #003366;
                        padding: 0.9rem 1rem;
                        margin-bottom: 0.8rem;
                        min-height: 106px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #003366;
                            line-height: 1.2;
                            margin-bottom: 0.35rem;
                            font-family: 'Times New Roman', Times, serif;">
                    Class
                </div>
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #000;
                            line-height: 1.25;
                            font-family: 'Times New Roman', Times, serif;">
                    {escape(class_line)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # pH
        st.markdown(
            f"""
            <div style="background: #f8f9fa;
                        border-left: 4px solid #003366;
                        padding: 0.9rem 1rem;
                        margin-bottom: 0.8rem;
                        min-height: 106px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #003366;
                            line-height: 1.2;
                            margin-bottom: 0.35rem;
                            font-family: 'Times New Roman', Times, serif;">
                    pH
                </div>
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #000;
                            line-height: 1.25;
                            font-family: 'Times New Roman', Times, serif;">
                    {escape(f"{ph_value:.1f}")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # SMILES
        st.markdown(
            f"""
            <div style="background: #f8f9fa;
                        border-left: 4px solid #003366;
                        padding: 0.9rem 1rem;
                        margin-bottom: 0.8rem;
                        min-height: 106px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 19pt;
                            font-weight: 800;
                            color: #003366;
                            line-height: 1.2;
                            margin-bottom: 0.35rem;
                            font-family: 'Times New Roman', Times, serif;">
                    SMILES
                </div>
                <div style="font-size: 17pt;
                            font-weight: 800;
                            color: #000;
                            font-family: 'Times New Roman', Times, serif;
                            line-height: 1.25;
                            word-break: break-all;">
                    {escape(smiles or 'Not found')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _running_in_streamlit():
    """Detect if stream is launched via `streamlit run`.
    Uses runtime context to avoid AttributeError on older Streamlit versions.
    """
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if not _running_in_streamlit():
        script_path = os.path.abspath(__file__)
        print("Starting the prediction app with streamlit run...")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", script_path], check=False)
        except FileNotFoundError:
            print("Streamlit is unavailable. Install the deployment requirements first.")
        raise SystemExit(0)
    main()
