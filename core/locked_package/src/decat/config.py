"""Shared model configuration and environment option handling."""

from __future__ import annotations
import os
from typing import Dict, Optional
from skopt.space import Categorical, Integer, Real


class FingerprintConfig:
    """Model dimensions and training defaults shared by all DECAT components."""

    d_model = 256
    dropout = 0.3
    activation = "gelu"
    n_heads = 4
    n_layers = 2
    max_fp_tokens = 128
    batch_size = 32
    learning_rate = 0.0001
    weight_decay = 0.01
    max_epochs = 100
    early_stopping_patience = 80
    save_interval = 10
    min_delta = 0.0001
    scheduler_type = "plateau"
    fingerprint_scale = False
    norm_first = False
    attn_pooling = "attn"
    fp_bit_dropout = 0.0
    base_numeric_dim = 0


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_value(primary: str, fallback: Optional[str], default):
    v = os.environ.get(primary)
    if (v is None or str(v).strip() == "") and fallback:
        v = os.environ.get(fallback)
    if v is None or str(v).strip() == "":
        return default
    return v


def _env_bool_pair(
    primary: str, fallback: Optional[str], default: bool = False
) -> bool:
    v = _env_value(primary, fallback, None)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _setdefault_env(name: str, value) -> None:
    if os.environ.get(name) is None:
        os.environ[name] = str(value)


space = [
    Integer(2048, 4096, name="fp_bits"),
    Integer(192, 1200, name="topk_features"),
    Integer(128, 384, name="d_model"),
    Integer(16, 64, name="batch_size"),
    Real(1e-05, 0.003, prior="log-uniform", name="lr"),
    Real(1e-05, 0.008, prior="log-uniform", name="weight_decay"),
    Real(0.01, 0.25, prior="uniform", name="dropout"),
    Integer(48, 160, name="max_fp_tokens"),
    Integer(1, 3, name="n_layers"),
    Categorical([2, 4, 8], name="n_heads"),
    Categorical(["dual"], name="model_mode"),
    Categorical(["rf", "f_regression"], name="fp_select_method"),
    Categorical([False, True], name="enable_chem_attn_bias"),
    Categorical([False, True], name="norm_first"),
    Integer(1, 5, name="topk_ckpt_ensemble"),
    Categorical([False, True], name="use_ckpt_ensemble"),
    Real(0.0, 0.3, prior="uniform", name="lambda_inv"),
    Real(0.0, 0.4, prior="uniform", name="lambda_dro"),
    Real(0.0, 0.2, prior="uniform", name="lambda_physics"),
    Real(0.001, 0.2, prior="log-uniform", name="dro_tau"),
    Real(0.0, 0.08, prior="uniform", name="generalization_penalty"),
]


def _normalize_model_mode_value(mode: str) -> str:
    m = str(mode).strip().lower()
    if "+" in m:
        m = m.split("+", 1)[0]
    if m not in {"dual", "attn", "mlp"}:
        m = "dual"
    return m


def _params_to_bo_vector(params: Dict[str, object]) -> Optional[list]:
    try:
        n_heads = int(params.get("n_heads", 4))
        if n_heads not in {2, 4, 8}:
            n_heads = 4
        fp_select_method = (
            str(params.get("fp_select", params.get("fp_select_method", "rf")))
            .strip()
            .lower()
        )
        if fp_select_method not in {"rf", "f_regression"}:
            fp_select_method = "rf"
        vec = [
            int(params.get("fp_bits", 3147)),
            int(params.get("topk_features", params.get("topk", 914))),
            int(params.get("d_model", 212)),
            int(params.get("batch_size", 32)),
            float(params.get("lr", params.get("learning_rate", 0.003))),
            float(params.get("weight_decay", 0.0015)),
            float(params.get("dropout", 0.1)),
            int(params.get("max_fp_tokens", params.get("tok", 115))),
            int(params.get("n_layers", 2)),
            int(n_heads),
            _normalize_model_mode_value(params.get("model_mode", "dual")),
            str(fp_select_method),
            bool(params.get("enable_chem_attn_bias", True)),
            bool(params.get("norm_first", False)),
            int(params.get("topk_checkpoint_ensemble", 3)),
            bool(params.get("use_checkpoint_ensemble", True)),
            float(params.get("lambda_inv", 0.18)),
            float(params.get("lambda_dro", 0.28)),
            float(params.get("lambda_physics", 0.08)),
            float(params.get("dro_tau", 0.05)),
            float(params.get("generalization_penalty", 0.03)),
        ]
        return vec
    except Exception:
        return None


def _parse_seeds(seed_text: str) -> list:
    out = []
    for token in str(seed_text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except Exception:
            continue
    return out if out else [42]


def _dedupe_candidate_vectors(vectors: list) -> list:
    uniq = []
    seen = set()
    for vec in vectors:
        try:
            key = tuple(vec)
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        uniq.append(list(vec))
    return uniq


def _normalize_model_mode(model_mode: str) -> str:
    return _normalize_model_mode_value(model_mode)


def _params_to_vector(
    params: Dict[str, object], model_mode_override: Optional[str] = None
) -> list:
    mode = _normalize_model_mode(
        model_mode_override or params.get("model_mode", "dual")
    )
    n_heads = int(params["n_heads"])
    if n_heads not in {2, 4, 8}:
        n_heads = 4
    fp_select_method = (
        str(params.get("fp_select_method", params.get("fp_select", "rf")))
        .strip()
        .lower()
    )
    if fp_select_method not in {"rf", "f_regression"}:
        fp_select_method = "rf"
    return [
        int(params["fp_bits"]),
        int(params["topk_features"]),
        int(params["d_model"]),
        int(params["batch_size"]),
        float(params["lr"]),
        float(params["weight_decay"]),
        float(params["dropout"]),
        int(params["max_fp_tokens"]),
        int(params["n_layers"]),
        int(n_heads),
        mode,
        fp_select_method,
        bool(params.get("enable_chem_attn_bias", True)),
        bool(params.get("norm_first", False)),
        int(params.get("topk_checkpoint_ensemble", 3)),
        bool(params.get("use_checkpoint_ensemble", True)),
        float(params.get("lambda_inv", 0.18)),
        float(params.get("lambda_dro", 0.28)),
        float(params.get("lambda_physics", 0.08)),
        float(params.get("dro_tau", 0.05)),
        float(params.get("generalization_penalty", 0.03)),
    ]


def _temporary_env_set(env_updates: Dict[str, str]) -> Dict[str, Optional[str]]:
    backup: Dict[str, Optional[str]] = {}
    for k, v in env_updates.items():
        backup[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    return backup


def _temporary_env_restore(backup: Dict[str, Optional[str]]) -> None:
    for k, v in backup.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


def _extract_payload_env(payload: Optional[Dict[str, object]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    env_payload = payload.get("env") if isinstance(payload, dict) else None
    if not isinstance(env_payload, dict):
        return out
    for key, value in env_payload.items():
        key_str = str(key).strip()
        if not key_str.startswith("DECAT_RUNTIME_"):
            continue
        if value is None:
            continue
        out[key_str] = str(value)
    return out


FORMAL_DUAL_INNOV_BEST_VECTOR = [
    3147,
    1024,
    224,
    28,
    0.0025,
    0.0012,
    0.08,
    128,
    2,
    4,
    "dual",
    "rf",
    True,
    True,
    3,
    True,
    0.2,
    0.3,
    0.08,
    0.05,
    0.03,
]
