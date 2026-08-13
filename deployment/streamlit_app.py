# -*- coding: utf-8 -*-
"""
Streamlit 封装版 - 二级反应速率常数logk预测系统（超参数可自定义）
流程：化学物质名称 → CAS号查询 → SMILES获取 → 分子指纹生成 → 类别匹配 → pH选择 → logk预测
核心特性：所有ANN超参数在侧边栏可视化配置，无需修改代码
依赖：需提前安装 streamlit, pandas, requests, beautifulsoup4, rdkit-pypi, torch, scikit-learn
运行方式：streamlit run 本文件名称.py
"""
import os
import re
import io
import base64
import json
import pickle
import importlib
import importlib.util
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import quote
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Draw
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from streamlit.runtime.scriptrunner import get_script_run_ctx
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from html import escape

# The release is self-contained.  All runtime assets are resolved relative to
# this file so the platform can be copied to another machine or folder.
SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR.parent
PROJECT_DIR = RELEASE_ROOT
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

# ======================== 基础配置（固定参数，非超参数） ========================
# 禁用RDKit警告
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.info')

# 旧版ANN类别映射（名称→编码）
LEGACY_CATEGORY_MAP = {
    "烷烃 / Alkane": 0,
    "脂肪醇 / Alcohol": 1,
    "脂肪二醇 / Diol": 2,
    "醚 / Ether": 3,
    "酮 / Ketone": 4,
    "醛 / Aldehyde": 5,
    "酯 / Ester": 6,
    "羧酸 / Carboxyl": 7,
    "二元羧酸 / Dicarboxylic": 8,
    "脂肪卤代物 / Halogenated": 9,
    "硫化物 / Disulfide": 10,
    "烯烃 / Alkene": 11,
    "苯类 / Benzene": 12,
    "酚类 / Phenol": 13,
    "未知化合物": 14
}
LEGACY_CATEGORY_REVERSE = {v: k for k, v in LEGACY_CATEGORY_MAP.items()}
LEGACY_CATEGORY_CODES = set(LEGACY_CATEGORY_MAP.values())


def legacy_category_name_from_code(code: int) -> str:
    """Return display name for a legacy ANN category code."""
    return LEGACY_CATEGORY_REVERSE.get(code, "未知化合物")


def legacy_category_name_en_from_code(code: int) -> str:
    raw = legacy_category_name_from_code(code)
    if " / " in raw:
        return raw.split(" / ", 1)[1].strip()
    if "/" in raw:
        return raw.split("/", 1)[1].strip()
    return raw


FAMILY12_CLASSES = [
    {
        "code": "C1",
        "name_en": "Inorganic / organometallic",
        "name_cn": "无机物 / 有机金属化合物",
        "desc_cn": "无碳骨架，或含明显金属 / 准金属中心。",
    },
    {
        "code": "C2",
        "name_en": "Carboxylic acids / carboxylates",
        "name_cn": "羧酸 / 羧酸盐",
        "desc_cn": "以游离羧酸或羧酸盐官能团为主。",
    },
    {
        "code": "C3",
        "name_en": "Phenols / phenolates",
        "name_cn": "酚类 / 酚盐",
        "desc_cn": "含芳香羟基或酚盐结构。",
    },
    {
        "code": "C4",
        "name_en": "Organosulfur compounds",
        "name_cn": "有机硫化合物",
        "desc_cn": "含硫中心有机官能团或硫杂环骨架。",
    },
    {
        "code": "C5",
        "name_en": "Amides / ureas / imides / carbamates",
        "name_cn": "酰胺 / 尿素 / 酰亚胺 / 氨基甲酸酯",
        "desc_cn": "含酰胺型含氮羰基结构。",
    },
    {
        "code": "C6",
        "name_en": "Esters / lactones / carbonates",
        "name_cn": "酯 / 内酯 / 碳酸酯",
        "desc_cn": "含酯、内酯或碳酸酯型含氧羰基结构。",
    },
    {
        "code": "C7",
        "name_en": "Aldehydes / ketones",
        "name_cn": "醛 / 酮",
        "desc_cn": "以醛基或酮基为主，且无更高优先级官能团。",
    },
    {
        "code": "C8",
        "name_en": "Alcohols / polyols",
        "name_cn": "醇 / 多元醇",
        "desc_cn": "以醇羟基、多元醇或糖样氧化结构为主。",
    },
    {
        "code": "C9",
        "name_en": "Ethers / epoxides / acetals",
        "name_cn": "醚 / 环氧化物 / 缩醛",
        "desc_cn": "以醚、环氧或缩醛型含氧结构为主。",
    },
    {
        "code": "C10",
        "name_en": "Organonitrogen compounds",
        "name_cn": "有机含氮化合物",
        "desc_cn": "含胺、腈、偶氮、硝基或含氮杂环等结构。",
    },
    {
        "code": "C11",
        "name_en": "Aromatics / heteroaromatics",
        "name_cn": "芳香族 / 杂芳香族",
        "desc_cn": "具有芳香骨架，但无更高优先级家族官能团。",
    },
    {
        "code": "C12",
        "name_en": "Hydrocarbons / halogenated aliphatics",
        "name_cn": "烃类 / 脂肪族卤代物",
        "desc_cn": "烃类或脂肪族卤代化合物，且无更强异原子主导结构。",
    },
]
FAMILY12_CODE_TO_INFO = {item["code"]: item for item in FAMILY12_CLASSES}
FAMILY12_EN_TO_INFO = {item["name_en"]: item for item in FAMILY12_CLASSES}
FAMILY12_REASON_CN_MAP = {
    "SMILES could not be parsed reliably": "SMILES 无法被可靠解析，按无机 / 有机金属类兜底处理。",
    "No carbon skeleton or contains a metal / metalloid center": "不存在明确碳骨架，或含金属 / 准金属中心。",
    "Contains free carboxylic acid or carboxylate functionality": "检测到游离羧酸或羧酸盐官能团。",
    "Contains aromatic hydroxyl / phenolate motif": "检测到芳香羟基 / 酚盐结构。",
    "Contains sulfur-centered organic functionality or sulfur heterocycle": "检测到以硫为核心的有机官能团或硫杂环。",
    "Contains amide-like carbonyl nitrogen functionality": "检测到酰胺样含氮羰基结构。",
    "Contains ester-, lactone- or carbonate-type oxygenated carbonyl": "检测到酯、内酯或碳酸酯型含氧羰基结构。",
    "Contains aldehyde or ketone carbonyl without higher-priority family": "检测到醛或酮羰基，且无更高优先级家族结构。",
    "Dominant family is alcohol / polyol / carbohydrate-like oxygenation": "整体更符合醇 / 多元醇 / 糖样氧化家族特征。",
    "Dominant oxygenated family is ether / epoxide / acetal-like": "整体更符合醚 / 环氧 / 缩醛型含氧结构。",
    "Contains nitrogen-centered organic functionality or N-heteroaromatic scaffold": "检测到含氮官能团或含氮杂芳环骨架。",
    "Aromatic scaffold without higher-priority family-defining functionality": "为芳香 / 杂芳香骨架，且无更高优先级官能团。",
    "Hydrocarbon or halogenated aliphatic family without stronger heteroatom-defining motif": "属于烃类或脂肪族卤代物，且无更强的异原子主导结构。",
}
FAMILY12_ALIAS_TO_CODE = {
    "c1": "C1",
    "无机": "C1",
    "有机金属": "C1",
    "organometallic": "C1",
    "inorganic": "C1",
    "c2": "C2",
    "羧酸": "C2",
    "carbox": "C2",
    "c3": "C3",
    "酚": "C3",
    "phenol": "C3",
    "c4": "C4",
    "有机硫": "C4",
    "sulfur": "C4",
    "sulfur compounds": "C4",
    "c5": "C5",
    "酰胺": "C5",
    "尿素": "C5",
    "氨基甲酸酯": "C5",
    "amide": "C5",
    "urea": "C5",
    "imide": "C5",
    "carbamate": "C5",
    "c6": "C6",
    "酯": "C6",
    "内酯": "C6",
    "碳酸酯": "C6",
    "ester": "C6",
    "lactone": "C6",
    "carbonate": "C6",
    "c7": "C7",
    "醛": "C7",
    "酮": "C7",
    "aldehyde": "C7",
    "ketone": "C7",
    "c8": "C8",
    "醇": "C8",
    "多元醇": "C8",
    "alcohol": "C8",
    "polyol": "C8",
    "c9": "C9",
    "醚": "C9",
    "环氧": "C9",
    "缩醛": "C9",
    "ether": "C9",
    "epoxide": "C9",
    "acetal": "C9",
    "c10": "C10",
    "含氮": "C10",
    "有机氮": "C10",
    "nitrogen": "C10",
    "amine": "C10",
    "nitrile": "C10",
    "c11": "C11",
    "芳香": "C11",
    "杂芳香": "C11",
    "aromatic": "C11",
    "heteroaromatic": "C11",
    "c12": "C12",
    "烃": "C12",
    "卤代": "C12",
    "halogenated": "C12",
    "hydrocarbon": "C12",
}

REFERENCE27_CLASSES = [
    {"code": "A", "name_en": "Alkane"},
    {"code": "B", "name_en": "Alcohol"},
    {"code": "C", "name_en": "Diol"},
    {"code": "D", "name_en": "Ether"},
    {"code": "E", "name_en": "Ketone"},
    {"code": "F", "name_en": "Aldehyde"},
    {"code": "G", "name_en": "Ester"},
    {"code": "H", "name_en": "Carboxyl"},
    {"code": "I", "name_en": "Dicarboxylic"},
    {"code": "J", "name_en": "Halogenated"},
    {"code": "K", "name_en": "Sulfide, disulfide"},
    {"code": "L", "name_en": "Sulfoxide"},
    {"code": "M", "name_en": "Thiol"},
    {"code": "N", "name_en": "Nitrile"},
    {"code": "O", "name_en": "Nitro"},
    {"code": "P", "name_en": "Amide"},
    {"code": "Q", "name_en": "Amine"},
    {"code": "R", "name_en": "Nitroso, nitramine"},
    {"code": "S", "name_en": "Phosphorus"},
    {"code": "T", "name_en": "Cyclo"},
    {"code": "U", "name_en": "Alkene"},
    {"code": "V", "name_en": "Benzene"},
    {"code": "W", "name_en": "Pyridine"},
    {"code": "X", "name_en": "Furan"},
    {"code": "Y", "name_en": "Urea"},
    {"code": "Z", "name_en": "Imidazole"},
    {"code": "A2", "name_en": "Triazine"},
]
REFERENCE27_CODE_TO_INFO = {item["code"]: item for item in REFERENCE27_CLASSES}
REFERENCE27_NAME_TO_INFO = {item["name_en"].lower(): item for item in REFERENCE27_CLASSES}
REFERENCE27_LABEL_TO_INFO = {f"{item['code']}: {item['name_en']}".lower(): item for item in REFERENCE27_CLASSES}

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

# 分子指纹固定参数（若需修改可移至侧边栏）
FP_SIZE = 4096  # Morgan指纹长度
RADIUS = 2  # Morgan指纹半径
gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=FP_SIZE)

# 设备配置（自动识别GPU/CPU）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Self-contained release paths
# ---------------------------------------------------------------------------
DEPLOYMENT_ARTIFACTS = SCRIPT_DIR / "artifacts"
RUNTIME_PACKAGE = DEPLOYMENT_ARTIFACTS / "runtime_package"
RUNTIME_CONFIG_PATH = RUNTIME_PACKAGE / "configs" / "LOCKED_SEEN25_1626_NONUNIFORM_SOTA.json"
RUNTIME_DATASET_PATH = RUNTIME_PACKAGE / "data" / "反应logk指纹数据_25类_含环境无机物_Other_1626条.csv"
RUNTIME_SPLIT_PATH = RUNTIME_PACKAGE / "data" / "固定划分_1626_同分子异pH非均分_75-12.5-12.5.json"
LOCKED_PACKAGE_ROOT = RELEASE_ROOT / "core" / "locked_package"

DATASET_CANDIDATES = [
    os.environ.get("LOGK_DATASET_PATH", "").strip(),
    str(RUNTIME_DATASET_PATH),
    str(LOCKED_PACKAGE_ROOT / "data" / "反应logk指纹数据_25类_含环境无机物_Other_1626条.csv"),
]


def _first_existing_path(candidates):
    for p in candidates:
        if not p:
            continue
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


DATASET_PATH = _first_existing_path(DATASET_CANDIDATES) or str(RUNTIME_DATASET_PATH)

V9_LOCKED_RUN_DIR = LOCKED_PACKAGE_ROOT / "artifacts" / "run"
V9_LOCKED_MODEL_PATH = V9_LOCKED_RUN_DIR / "transformer_v7_best.pth"
V9_LOCKED_PARAMS_PATH = V9_LOCKED_RUN_DIR / "best_params.json"
V9_LOCKED_SUMMARY_PATH = V9_LOCKED_RUN_DIR / "fixed_params_run_summary.json"
V9_LOCKED_DATASET_PATH = LOCKED_PACKAGE_ROOT / "data" / "反应logk指纹数据_25类_含环境无机物_Other_1626条.csv"
V9_LOCKED_SPLIT_PATH = LOCKED_PACKAGE_ROOT / "data" / "固定划分_1626_同分子异pH非均分_75-12.5-12.5.json"
V9_LOCKED_FUSION_COMPONENTS_PATH = LOCKED_PACKAGE_ROOT / "artifacts" / "predictions" / "fusion_components.npz"
V9_LOCKED_RUNTIME_PATH = LOCKED_PACKAGE_ROOT / "src" / "decat" / "transformer_v9_transformer_centered.py"
V9_LOCKED_TEST_R2 = 0.8286396449402338
V9_LOCKED_TEST_RMSE = 1.1624043383268383

PAPER_EXTERNAL_ROOT = SCRIPT_DIR / "data"
PAPER_REBUILD_ROOT = DEPLOYMENT_ARTIFACTS
PAPER_REBUILD_PACKAGE = RUNTIME_PACKAGE
PAPER_REBUILD_CONFIG_PATH = RUNTIME_CONFIG_PATH
PAPER_TOP3_MANIFEST_PATH = DEPLOYMENT_ARTIFACTS / "top_checkpoints" / "top_checkpoint_ensemble.json"
PAPER_RESIDUAL_MANIFEST_PATH = DEPLOYMENT_ARTIFACTS / "residual_rf" / "residual_rf_manifest.json"
PAPER_FUSION_COMPONENTS_PATH = DEPLOYMENT_ARTIFACTS / "fusion_components.npz"
PAPER_EXTERNAL_CASES_PATH = SCRIPT_DIR / "data" / "external_validation_10.csv"

# Seed-242 applicability-domain route used for the complete 4,295-structure
# screen (LOO q95/q99 in the selected 914-bit space).
AD_STRICT_DISTANCE_THRESHOLD = 0.6428571428571428
AD_BORDERLINE_DISTANCE_THRESHOLD = 0.8
AD_STRICT_SIMILARITY_THRESHOLD = 1.0 - AD_STRICT_DISTANCE_THRESHOLD
AD_BORDERLINE_SIMILARITY_THRESHOLD = 1.0 - AD_BORDERLINE_DISTANCE_THRESHOLD
PH_TRAIN_MIN = 1.0
PH_TRAIN_MAX = 12.0

HIGH_THROUGHPUT_RESULT_DIR = RELEASE_ROOT / "screening" / "results"
HIGH_THROUGHPUT_INPUT_DIR = RELEASE_ROOT / "screening" / "input"
HIGH_THROUGHPUT_COMBINED_ROW_RESULTS = HIGH_THROUGHPUT_RESULT_DIR / "AD_seed242_row_level_results.csv"
HIGH_THROUGHPUT_COMBINED_UNIQUE_RESULTS = HIGH_THROUGHPUT_RESULT_DIR / "AD_seed242_unique_compound_results.csv"
HIGH_THROUGHPUT_COMBINED_METADATA = HIGH_THROUGHPUT_RESULT_DIR / "AD_seed242_summary.json"
HIGH_THROUGHPUT_TOP_BITS = HIGH_THROUGHPUT_RESULT_DIR / "seed242_selected_bit_indices.txt"


def _normalize_v6_mode(mode: str) -> str:
    mode_text = str(mode or "").strip().lower()
    if "+" in mode_text:
        mode_text = mode_text.split("+", 1)[0]
    if mode_text not in {"dual", "attn", "mlp"}:
        mode_text = "dual"
    return mode_text


def _load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_v6_payload_fields(payload: dict) -> Tuple[dict, dict, str]:
    params = payload.get("params", payload)
    payload_env = payload.get("env", {}) if isinstance(payload, dict) else {}
    dataset_path = str(
        payload.get("dataset")
        or payload.get("data_csv_path")
        or payload_env.get("TRANSFORMER_V6_DATA_CSV", "")
    ).strip()
    return params, payload_env, dataset_path


def _find_compatible_v6_checkpoint(params_path: str) -> str:
    run_root = PROJECT_DIR / "transformer_v6" / "transformer"
    if not run_root.exists():
        return ""

    try:
        target_payload = _load_json_file(params_path)
        target_params, _, _ = _extract_v6_payload_fields(target_payload)
    except Exception:
        target_params = {}

    exact_keys = [
        "fp_bits",
        "topk_features",
        "d_model",
        "max_fp_tokens",
        "n_layers",
        "n_heads",
        "enable_chem_attn_bias",
        "enable_category_fusion",
        "norm_first",
    ]
    mode_target = _normalize_v6_mode(target_params.get("model_mode", "dual"))
    best_match = ""
    best_score = -1

    run_dirs = sorted(
        [p for p in run_root.glob("运行_*") if p.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        model_path = run_dir / "transformer_v6_best.pth"
        params_candidate = run_dir / "best_params.json"
        if not (model_path.exists() and params_candidate.exists()):
            continue
        score = 0
        try:
            cand_payload = _load_json_file(str(params_candidate))
            cand_params, _, _ = _extract_v6_payload_fields(cand_payload)
            if _normalize_v6_mode(cand_params.get("model_mode", "dual")) != mode_target:
                continue
            for key in exact_keys:
                if key in target_params and str(cand_params.get(key)) == str(target_params.get(key)):
                    score += 1
        except Exception:
            score = 0
        if score > best_score:
            best_score = score
            best_match = str(model_path)
            if score >= len(exact_keys):
                break
    return best_match


def _find_latest_v6_artifacts():
    v6_root = PROJECT_DIR / "transformer_v6" / "transformer"
    if v6_root.exists():
        run_dirs = sorted(v6_root.glob("运行_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        for run_dir in run_dirs:
            model_path = run_dir / "transformer_v6_best.pth"
            params_path = run_dir / "best_params.json"
            if not (model_path.exists() and params_path.exists()):
                continue
            dataset_path = DATASET_PATH
            try:
                payload = _load_json_file(str(params_path))
                _, _, payload_dataset = _extract_v6_payload_fields(payload)
                if payload_dataset and os.path.exists(payload_dataset):
                    dataset_path = payload_dataset
            except Exception:
                dataset_path = DATASET_PATH
            return str(model_path), str(params_path), dataset_path

    recommended_params = v6_root / "fixed_recommended_params_family12_scope_filtered_tuned_ema_20260324.json"
    params_path = str(recommended_params) if recommended_params.exists() else ""
    model_path = _find_compatible_v6_checkpoint(params_path) if params_path else ""
    return model_path, params_path, DATASET_PATH


def _find_locked_v9_artifacts():
    """Return only the current locked SEEN25-1626 package artifacts."""
    return (
        str(V9_LOCKED_MODEL_PATH),
        str(V9_LOCKED_PARAMS_PATH),
        str(V9_LOCKED_DATASET_PATH),
        str(V9_LOCKED_SUMMARY_PATH),
    )


def _find_latest_v4_artifacts():
    """Find latest V4 checkpoint + params from transformer run folders."""
    run_root = PROJECT_DIR / "transformer"
    if not run_root.exists():
        return "", ""

    run_dirs = sorted(
        [p for p in run_root.glob("运行_*") if p.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for d in run_dirs:
        model_path = d / "transformer_v4_best.pth"
        params_path = d / "best_params.json"
        if model_path.exists() and params_path.exists():
            return str(model_path), str(params_path)

    # Fallbacks
    model_fallback = PROJECT_DIR / "transformer_v4_best.pth"
    params_fallback = PROJECT_DIR / "best_params_transformer_v4_innov_latest.json"
    return (str(model_fallback) if model_fallback.exists() else "", str(params_fallback) if params_fallback.exists() else "")


def _find_latest_ann_checkpoint() -> str:
    ann_root = PROJECT_DIR / "对比试验" / "运行结果" / "ANN"
    if ann_root.exists():
        for cand in sorted(ann_root.glob("运行_*/best_model.pth"), key=lambda x: x.stat().st_mtime, reverse=True):
            return str(cand)
    fallback = PROJECT_DIR / "ann_regressor.pth"
    return str(fallback) if fallback.exists() else ""

# ======================== 工具函数定义 ========================
def read_csv_robust(path, **kwargs):
    """健壮的CSV读取函数（兼容多编码）"""
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
    # 兜底方案
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
    raise last_err or UnicodeDecodeError("无法解码CSV文件")


@st.cache_data(ttl=3600)  # 缓存CAS查询结果1小时
def get_cas_from_chemsrc(chemical_name):
    """从Chemsrc查询CAS号"""
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
    """从 PubChem 同义词列表中提取 CAS 号。"""
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
    """优先 Chemsrc，失败后回退到 PubChem。"""
    cas = get_cas_from_chemsrc(chemical_name)
    if cas:
        return cas, "Chemsrc"
    cas = get_cas_from_pubchem(chemical_name)
    if cas:
        return cas, "PubChem"
    return None, ""


@st.cache_data(ttl=3600)  # 缓存SMILES查询结果1小时
def get_smiles_via_api(compound_name, cas_number=None):
    """优先PubChem查询SMILES，失败时回退到CACTUS，支持名称和CAS双通道。"""
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
        for _ in range(2):  # 轻量重试，缓解偶发网络抖动
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

    # 优先：名称（PubChem）→ CAS（PubChem）→ 名称/CAS（CACTUS）
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
    """将SMILES转换为分子指纹"""
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


def match_chemical_category(chemical_name):
    """根据化学名称匹配类别"""
    if not chemical_name:
        return 14
    name_lower = chemical_name.lower().strip()
    if "烷" in chemical_name or "alkane" in name_lower:
        return 0
    elif "醇" in chemical_name and "二醇" not in chemical_name or "alcohol" in name_lower and "diol" not in name_lower:
        return 1
    elif "二醇" in chemical_name or "diol" in name_lower:
        return 2
    elif "醚" in chemical_name or "ether" in name_lower:
        return 3
    elif "酮" in chemical_name or "ketone" in name_lower:
        return 4
    elif "醛" in chemical_name or "aldehyde" in name_lower:
        return 5
    elif "酯" in chemical_name or "ester" in name_lower:
        return 6
    elif "羧酸" in chemical_name and "二元" not in chemical_name or "carboxyl" in name_lower and "dicarboxylic" not in name_lower:
        return 7
    elif "二元羧酸" in chemical_name or "dicarboxylic" in name_lower:
        return 8
    elif "卤" in chemical_name or "halogen" in name_lower:
        return 9
    elif "硫化物" in chemical_name or "disulfide" in name_lower:
        return 10
    elif "烯" in chemical_name or "alkene" in name_lower:
        return 11
    elif "苯" in chemical_name or "benzene" in name_lower:
        return 12
    elif "酚" in chemical_name or "phenol" in name_lower:
        return 13
    else:
        return 14


# ======================== 动态超参数ANN模型定义 ========================
class DynamicANNRegressor(nn.Module):
    """根据侧边栏配置动态构建的ANN回归模型"""

    def __init__(self, input_dim, hidden_layer_sizes, dropout_rate=0.2):
        super().__init__()
        layers = []
        # 输入层 → 第一层隐藏层
        prev_dim = input_dim
        for curr_dim in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, curr_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = curr_dim
        # 输出层（回归任务，单输出）
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # 仅压掉最后一维，保留 batch 维度，避免 batch=1 时返回 0 维数组
        return self.model(x).squeeze(-1)


def _infer_architecture_from_checkpoint(checkpoint, default_input_dim, default_hidden_layer_sizes, default_dropout):
    """Infer architecture metadata from checkpoint; fallback to sidebar values."""
    input_dim = int(default_input_dim)
    hidden_layer_sizes = list(default_hidden_layer_sizes)
    dropout = float(default_dropout)

    if not isinstance(checkpoint, dict):
        return input_dim, hidden_layer_sizes, dropout

    ck_input_dim = checkpoint.get("input_dim")
    if ck_input_dim is not None:
        try:
            input_dim = int(ck_input_dim)
        except Exception:
            pass

    ck_hidden_sizes = checkpoint.get("hidden_layer_sizes")
    if isinstance(ck_hidden_sizes, (list, tuple)) and len(ck_hidden_sizes) > 0:
        try:
            hidden_layer_sizes = [int(v) for v in ck_hidden_sizes]
        except Exception:
            pass
    else:
        params = checkpoint.get("params") if isinstance(checkpoint.get("params"), dict) else {}
        params_hidden = params.get("hidden_dims") or params.get("hidden_layers")
        if isinstance(params_hidden, (list, tuple)) and len(params_hidden) > 0:
            try:
                hidden_layer_sizes = [int(v) for v in params_hidden]
            except Exception:
                pass
        ck_hidden_dim = checkpoint.get("hidden_dim")
        ck_hidden_layers = checkpoint.get("hidden_layers")
        if ck_hidden_dim is not None and ck_hidden_layers is not None:
            try:
                hidden_dim = int(ck_hidden_dim)
                hidden_layers = int(ck_hidden_layers)
                if hidden_dim > 0 and hidden_layers > 0:
                    hidden_layer_sizes = [hidden_dim] * hidden_layers
            except Exception:
                pass

    ck_dropout = checkpoint.get("dropout")
    if ck_dropout is None and isinstance(checkpoint.get("params"), dict):
        ck_dropout = checkpoint["params"].get("dropout")
    if ck_dropout is not None:
        try:
            dropout = float(ck_dropout)
        except Exception:
            pass

    return input_dim, hidden_layer_sizes, dropout


def _remap_legacy_state_dict_to_dynamic(state_dict, hidden_layer_sizes):
    """Remap hidden/output style keys to DynamicANNRegressor sequential keys."""
    if not isinstance(state_dict, dict):
        return state_dict

    target_hidden_indices = [i * 3 for i in range(len(hidden_layer_sizes))]
    output_index = len(hidden_layer_sizes) * 3

    legacy_hidden_indices = set()
    for key in state_dict.keys():
        k = key[7:] if key.startswith("module.") else key
        for prefix in ("hidden.", "model.hidden."):
            if k.startswith(prefix):
                tail = k[len(prefix):]
                parts = tail.split(".")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1] in ("weight", "bias"):
                    legacy_hidden_indices.add(int(parts[0]))
                break
    legacy_hidden_indices = sorted(legacy_hidden_indices)
    idx_map = {}
    if legacy_hidden_indices:
        for old_idx, new_idx in zip(legacy_hidden_indices, target_hidden_indices):
            idx_map[old_idx] = new_idx

    remapped = {}
    for key, val in state_dict.items():
        k = key[7:] if key.startswith("module.") else key
        new_key = k

        if k.startswith("hidden."):
            tail = k[len("hidden."):]
            parts = tail.split(".")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1] in ("weight", "bias"):
                old_idx = int(parts[0])
                mapped_idx = idx_map.get(old_idx, old_idx)
                new_key = f"model.{mapped_idx}.{parts[1]}"
        elif k.startswith("model.hidden."):
            tail = k[len("model.hidden."):]
            parts = tail.split(".")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1] in ("weight", "bias"):
                old_idx = int(parts[0])
                mapped_idx = idx_map.get(old_idx, old_idx)
                new_key = f"model.{mapped_idx}.{parts[1]}"
        elif k.startswith("output."):
            tail = k[len("output."):]
            parts = tail.split(".")
            if len(parts) >= 1 and parts[0] in ("weight", "bias"):
                new_key = f"model.{output_index}.{parts[0]}"
        elif k.startswith("model.output."):
            tail = k[len("model.output."):]
            parts = tail.split(".")
            if len(parts) >= 1 and parts[0] in ("weight", "bias"):
                new_key = f"model.{output_index}.{parts[0]}"
        elif k.startswith("net."):
            tail = k[len("net."):]
            parts = tail.split(".")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1] in ("weight", "bias"):
                new_key = f"model.{parts[0]}.{parts[1]}"

        remapped[new_key] = val
    return remapped


def _load_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{module_name} <- {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _get_family12_runtime_module():
    family12_path = PROJECT_DIR / "analyze_and_reclassify_categories_family12.py"
    if not family12_path.exists():
        raise FileNotFoundError(f"缺少 family12 分类脚本：{family12_path}")
    return _load_module_from_file("_streamlit_family12_classifier", str(family12_path))


def _get_category27_runtime_module():
    category27_path = PROJECT_DIR / "transformer_v6" / "category27_classifier.py"
    if not category27_path.exists():
        raise FileNotFoundError(f"Missing category27 classifier script: {category27_path}")
    return _load_module_from_file("_streamlit_category27_classifier", str(category27_path))


def _apply_v6_runtime_env(params: dict, payload_env: dict):
    base_mode = _normalize_v6_mode(params.get("model_mode", "dual"))
    env_updates = {
        "TRANSFORMER_V3_ENABLE_DUAL": "1" if base_mode == "dual" else "0",
        "TRANSFORMER_V3_GATE_EXTRA_FEATURES": "1" if bool(params.get("gate_extra_features", True)) else "0",
        "TRANSFORMER_V3_ENABLE_CORRECTION_HEAD": "1" if bool(params.get("enable_correction_head", True)) else "0",
        "TRANSFORMER_V3_ENABLE_FUSION": "1" if bool(params.get("enable_fusion", True)) else "0",
        "TRANSFORMER_V3_ENABLE_CATEGORY_FUSION": "1" if bool(params.get("enable_category_fusion", False)) else "0",
        "TRANSFORMER_V3_ENABLE_CHEM_ATTN_BIAS": "1" if bool(params.get("enable_chem_attn_bias", True)) else "0",
        "TRANSFORMER_V3_USE_CKPT_ENSEMBLE": "1" if bool(params.get("use_checkpoint_ensemble", True)) else "0",
        "TRANSFORMER_V3_TOPK_CHECKPOINT_ENSEMBLE": str(int(params.get("topk_checkpoint_ensemble", 3))),
        "TRANSFORMER_V3_CKPT_ENSEMBLE_WEIGHTED": "1" if bool(params.get("ckpt_ensemble_weighted", True)) else "0",
        "TRANSFORMER_V3_CKPT_ENSEMBLE_TEMP": str(float(params.get("ckpt_ensemble_temp", 0.05))),
        "TRANSFORMER_V4_LAMBDA_INV": str(float(params.get("lambda_inv", 0.2))),
        "TRANSFORMER_V4_LAMBDA_DRO": str(float(params.get("lambda_dro", 0.3))),
        "TRANSFORMER_V4_LAMBDA_PHYSICS": str(float(params.get("lambda_physics", 0.08))),
        "TRANSFORMER_V4_DRO_TAU": str(float(params.get("dro_tau", 0.05))),
        "TRANSFORMER_V3_GENERALIZATION_PENALTY": str(float(params.get("generalization_penalty", 0.03))),
        "TRANSFORMER_V6_ENABLE_EMA": "1" if bool(params.get("use_ema", False)) else "0",
        "TRANSFORMER_V6_EMA_DECAY": str(float(params.get("ema_decay", payload_env.get("TRANSFORMER_V6_EMA_DECAY", 0.998)))),
        "TRANSFORMER_V6_EMA_WARMUP_STEPS": str(int(float(params.get("ema_warmup_steps", payload_env.get("TRANSFORMER_V6_EMA_WARMUP_STEPS", 120))))),
    }
    fixed_split = str(
        payload_env.get("TRANSFORMER_V6_FIXED_SPLIT_JSON", params.get("fixed_split_json", ""))
    ).strip()
    if fixed_split and os.path.exists(fixed_split):
        env_updates["TRANSFORMER_V6_FIXED_SPLIT_JSON"] = fixed_split
    for key, value in payload_env.items():
        if value is not None and str(value).strip() != "":
            os.environ[str(key)] = str(value)
    for key, value in env_updates.items():
        os.environ[str(key)] = str(value)


def _get_v6_runtime_module():
    return importlib.import_module("transformer_v6.transformer_v6.core")


def _get_v9_runtime_module():
    return _load_module_from_file("_streamlit_decat_v9_runtime", str(V9_LOCKED_RUNTIME_PATH))


def category27_to_legacy_code(info: Optional[Dict[str, object]]) -> Optional[int]:
    """Map the 27-class display taxonomy to the 15-class input used by the locked 0.826 model."""
    if not info:
        return None
    raw_parts = [
        str(info.get(key, "")).strip()
        for key in ("category27_code", "code", "category27_name", "name_en", "category27_label", "label")
        if str(info.get(key, "")).strip()
    ]
    text = " ".join(raw_parts).lower()
    code = str(info.get("category27_code") or info.get("code") or "").strip()
    code_map = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "J": 9,
        "K": 10,
        "L": 10,
        "M": 10,
        "U": 11,
        "V": 12,
        "W": 12,
        "X": 12,
        "Z": 12,
        "A2": 12,
    }
    if code in code_map:
        return code_map[code]
    if "phenol" in text:
        return 13
    if "alkane" in text:
        return 0
    if "diol" in text:
        return 2
    if "alcohol" in text or "polyol" in text:
        return 1
    if "ether" in text:
        return 3
    if "ketone" in text:
        return 4
    if "aldehyde" in text:
        return 5
    if "ester" in text:
        return 6
    if "dicarbox" in text:
        return 8
    if "carbox" in text:
        return 7
    if "halogen" in text or "chloro" in text or "bromo" in text or "fluoro" in text:
        return 9
    if "sulf" in text or "thiol" in text:
        return 10
    if "alkene" in text:
        return 11
    if "benzene" in text or "aromatic" in text or "pyridine" in text or "furan" in text or "imidazole" in text or "triazine" in text:
        return 12
    return 14


def _build_v6_numeric_vector(dataset, ph_value: float, category27_info: Optional[Dict[str, object]]) -> np.ndarray:
    """Build V6 numeric branch: [scaled_pH] + [category27 one-hot]."""
    ph_arr = np.asarray([[float(ph_value)]], dtype=np.float32)
    ph_scaled = dataset.ph_scaler.transform(ph_arr).reshape(-1).astype(np.float32)

    cat_cols = list(getattr(dataset, "category_cols", []))
    cat_vec = np.zeros((len(cat_cols),), dtype=np.float32)
    legacy_code = None
    info = category27_info or {}
    for legacy_key in ("legacy_category_code", "category_code", "legacy_code"):
        if legacy_key in info:
            try:
                legacy_code = int(float(info.get(legacy_key)))
                break
            except Exception:
                legacy_code = None
    if legacy_code is None:
        legacy_code = category27_to_legacy_code(category27_info)
    if legacy_code is not None:
        legacy_candidates = [
            f"cat_{legacy_code}",
            f"cat_{float(legacy_code):.1f}",
            f"cat_{legacy_category_name_en_from_code(int(legacy_code))}",
        ]
        for target_col in legacy_candidates:
            if target_col in cat_cols:
                cat_vec[cat_cols.index(target_col)] = 1.0
                return np.concatenate([ph_scaled, cat_vec], axis=0).astype(np.float32)

    candidates = []
    for key in ("category27_label", "label", "category27_code", "code", "category27_name", "name_en", "name_cn"):
        value = str(info.get(key, "")).strip()
        if value:
            candidates.append(value if value.startswith("cat_") else f"cat_{value}")
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    matched = False
    for target_col in candidates:
        if target_col in cat_cols:
            cat_vec[cat_cols.index(target_col)] = 1.0
            matched = True
            break
    if not matched:
        if "cat_Unknown" in cat_cols:
            cat_vec[cat_cols.index("cat_Unknown")] = 1.0
        elif cat_cols:
            cat_vec[0] = 1.0
    return np.concatenate([ph_scaled, cat_vec], axis=0).astype(np.float32)


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


def _apply_v9_runtime_env(params: dict, payload_env: dict):
    env_updates = {
        "TRANSFORMER_V9_TRANSFORMER_CENTERED": "1",
        "TRANSFORMER_V9_RESIDUAL_MAX": "0.85",
        "TRANSFORMER_V9_RESIDUAL_MIN": "0.0",
        "TRANSFORMER_V3_ENABLE_DUAL": "1",
        "TRANSFORMER_V3_GATE_EXTRA_FEATURES": "1",
        "TRANSFORMER_V3_ENABLE_CORRECTION_HEAD": "1",
        "TRANSFORMER_V3_ENABLE_CHEM_ATTN_BIAS": "1" if bool(params.get("enable_chem_attn_bias", True)) else "0",
        "TRANSFORMER_V3_ENABLE_FUSION": "1",
        "TRANSFORMER_V3_CKPT_ENSEMBLE_WEIGHTED": "1" if bool(params.get("ckpt_ensemble_weighted", True)) else "0",
        "TRANSFORMER_V3_CKPT_ENSEMBLE_TEMP": str(float(params.get("ckpt_ensemble_temp", 0.02))),
        "TRANSFORMER_V7_ATTENTION_MIN_GATE": "0.10",
        "TRANSFORMER_V7_ATTENTION_GATE_LOGIT_BIAS": "0.10",
        "TRANSFORMER_V3_CHEM_ATTN_ALPHA": "0.42",
    }
    for key, value in payload_env.items():
        if value is not None and str(value).strip() != "":
            os.environ[str(key)] = str(value)
    for key, value in env_updates.items():
        os.environ[str(key)] = str(value)
    os.environ["TRANSFORMER_V7_FIXED_SPLIT_JSON"] = str(V9_LOCKED_SPLIT_PATH)
    os.environ["TRANSFORMER_V6_FIXED_SPLIT_JSON"] = str(V9_LOCKED_SPLIT_PATH)
    os.environ["TRANSFORMER_V7_DATA_CSV"] = str(V9_LOCKED_DATASET_PATH)
    os.environ["TRANSFORMER_V9_CATEGORY_EMBED_DIM"] = "0"
    os.environ["TRANSFORMER_SEED"] = "242"
    os.environ["TRANSFORMER_DETERMINISTIC"] = "1"


def _extract_v9_payload_fields(payload: dict) -> Tuple[dict, dict, str]:
    if "params" in payload:
        params = dict(payload.get("params", {}) or {})
    elif "best_result" in payload and isinstance(payload.get("best_result"), dict):
        params = dict(payload.get("best_result", {}).get("params", {}) or {})
    else:
        params = dict(payload or {})
    payload_env = dict(payload.get("env", {}) or {})
    dataset_path = str(
        payload.get("data_csv_path")
        or payload.get("dataset")
        or payload_env.get("TRANSFORMER_V7_DATA_CSV", "")
        or payload_env.get("TRANSFORMER_V6_DATA_CSV", "")
    ).strip()
    return params, payload_env, dataset_path


def _load_top_bit_indices(path: Path, fallback_idx: np.ndarray) -> np.ndarray:
    if path.exists():
        try:
            values = []
            for part in re.split(r"[,\s]+", path.read_text(encoding="utf-8-sig").strip()):
                if str(part).strip():
                    values.append(int(float(part)))
            if values:
                return np.asarray(values, dtype=int)
        except Exception:
            pass
    return np.asarray(fallback_idx, dtype=int)


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
            "high_throughput_library_match": False,
            "high_throughput_range_source": "Assessed online using the high-throughput prediction range rule",
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
        "high_throughput_library_match": False,
        "high_throughput_range_source": "Assessed online using the high-throughput prediction range rule",
    }


@st.cache_resource
def load_legacy_decat_v9_bundle(model_path: str, params_path: str, dataset_path: str, summary_path: str = ""):
    """Load the locked DECAT model and its validation-selected RF residual head."""
    if not os.path.exists(model_path):
        return None, None, "锁定 DECAT 模型文件不存在"
    if not os.path.exists(params_path):
        return None, None, "锁定 DECAT 参数文件不存在"
    if not os.path.exists(dataset_path):
        return None, None, "锁定 DECAT 参考数据集不存在"
    if not V9_LOCKED_FUSION_COMPONENTS_PATH.exists():
        return None, None, "锁定 DECAT 残差校正组件不存在"

    try:
        v9_mod = _get_v9_runtime_module()
        params_payload = _load_json_file(params_path)
        params, payload_env, payload_dataset = _extract_v9_payload_fields(params_payload)
        if summary_path and os.path.exists(summary_path):
            try:
                summary_payload = _load_json_file(summary_path)
                params = dict(summary_payload.get("best_result", {}).get("params", params) or params)
                payload_env.update(dict(summary_payload.get("env", {}) or {}))
            except Exception:
                pass

        _apply_v9_runtime_env(params, payload_env)
        fp_bits = int(params.get("fp_bits", 3147))
        topk_features = int(params.get("topk_features", params.get("topk", 1100)))
        topk_features = max(1, min(topk_features, fp_bits))
        fp_select = str(params.get("fp_select", params.get("fp_select_method", "rf"))).strip().lower() or "rf"

        dataset = v9_mod.FingerprintReactionDataset(dataset_path, max_fp_bits=fp_bits, fingerprint_scale=False)
        split_dict = v9_mod._build_split_indices(dataset)
        train_idx = np.asarray(split_dict["train_idx"], dtype=int)
        dataset.fit_scalers(train_idx)
        rank_idx = v9_mod._get_fp_ranking(dataset, train_idx, fp_bits, fp_select)
        selected_idx = np.asarray(rank_idx[:topk_features], dtype=int)

        cfg = v9_mod.FingerprintConfig()
        cfg.d_model = int(params.get("d_model", cfg.d_model))
        cfg.dropout = float(params.get("dropout", cfg.dropout))
        cfg.n_layers = int(params.get("n_layers", getattr(cfg, "n_layers", 2)))
        cfg.n_heads = int(params.get("n_heads", getattr(cfg, "n_heads", 4)))
        cfg.max_fp_tokens = int(params.get("max_fp_tokens", getattr(cfg, "max_fp_tokens", topk_features)))
        cfg.norm_first = bool(params.get("norm_first", getattr(cfg, "norm_first", False)))
        if cfg.n_heads <= 0:
            cfg.n_heads = 4
        if cfg.d_model % cfg.n_heads != 0:
            cfg.d_model = max(cfg.n_heads, (cfg.d_model // cfg.n_heads) * cfg.n_heads)

        numeric_dim = int(getattr(dataset, "num_dim", 1 + len(getattr(dataset, "category_cols", []))))
        cfg.base_numeric_dim = int(getattr(dataset, "base_num_dim", numeric_dim))
        class ContextualDualExpertRegressor(v9_mod.DualExpertRegressor):
            def __init__(self, fingerprint_dim: int, numeric_dim: int, config):
                super().__init__(fingerprint_dim, numeric_dim, config)
                hidden = max(16, int(config.d_model // 4))
                self.contextual_residual = nn.Sequential(
                    nn.Linear(numeric_dim + 1, hidden),
                    nn.GELU(),
                    nn.Dropout(float(config.dropout)),
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

        model = ContextualDualExpertRegressor(
            fingerprint_dim=int(selected_idx.size),
            numeric_dim=numeric_dim,
            config=cfg,
        ).to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        model.eval()

        archive = np.load(V9_LOCKED_FUSION_COMPONENTS_PATH)
        component_names = [str(value) for value in archive["component_names"].tolist()]
        if "nn" not in component_names or "nn_plus_res_rf" not in component_names:
            raise ValueError("锁定组件中缺少 nn_plus_res_rf 输出头")
        nn_component = component_names.index("nn")
        expected_feature_dim = int(selected_idx.size + numeric_dim)
        x_train = np.asarray(archive["x_train"], dtype=np.float32)
        y_train = np.asarray(archive["y_train"], dtype=np.float32).reshape(-1)
        component_train = np.asarray(archive["component_train"], dtype=np.float32)
        if x_train.shape != (train_idx.size, expected_feature_dim):
            raise ValueError("锁定残差校正特征维度与当前模型不一致")
        if component_train.shape[1] != train_idx.size:
            raise ValueError("锁定残差校正训练样本数不一致")
        residual_model = RandomForestRegressor(
            n_estimators=280,
            max_depth=12,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=90,
            n_jobs=-1,
        )
        nn_train_raw = component_train[nn_component].reshape(-1)
        residual_model.fit(
            np.concatenate([x_train, nn_train_raw.reshape(-1, 1)], axis=1),
            y_train - nn_train_raw,
        )

        train_fp_selected = np.asarray(dataset.fingerprint[train_idx][:, selected_idx] > 0.5, dtype=bool)
        df_ref = read_csv_robust(dataset_path)
        compound_col = next((c for c in ["chemical compound", "chemical_compound", "compound", "名称", "化学物质"] if c in df_ref.columns), None)
        smiles_col = "SMILES" if "SMILES" in df_ref.columns else ("smiles" if "smiles" in df_ref.columns else None)
        train_compounds = []
        train_smiles = []
        try:
            if compound_col:
                train_compounds = df_ref.iloc[train_idx][compound_col].astype(str).tolist()
            if smiles_col:
                train_smiles = df_ref.iloc[train_idx][smiles_col].astype(str).tolist()
        except Exception:
            train_compounds = []
            train_smiles = []

        bundle = {
            "model": model,
            "params": params,
            "dataset": dataset,
            "fp_bits": int(fp_bits),
            "selected_idx": selected_idx,
            "numeric_dim": int(numeric_dim),
            "mode": "dual",
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "payload_env": payload_env,
            "category_cols": list(getattr(dataset, "category_cols", [])),
            "residual_model": residual_model,
            "residual_feature_dim": expected_feature_dim,
            "train_idx": train_idx,
            "train_fp_selected_bool": train_fp_selected,
            "train_compounds": train_compounds,
            "train_smiles": train_smiles,
            "model_label": "DECAT locked model (nn_plus_res_rf)",
            "test_r2": V9_LOCKED_TEST_R2,
            "test_rmse": V9_LOCKED_TEST_RMSE,
        }
        return bundle, params, ""
    except Exception as e:
        return None, None, str(e)


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
    """Load the exact Top-3 + serialized residual-RF inference route used in the paper."""
    required_paths = [
        PAPER_REBUILD_CONFIG_PATH,
        PAPER_TOP3_MANIFEST_PATH,
        PAPER_RESIDUAL_MANIFEST_PATH,
        PAPER_FUSION_COMPONENTS_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return None, None, f"Paper external-validation inference artifacts are missing: {missing}"

    try:
        package = PAPER_REBUILD_PACKAGE
        config = _load_json_file(str(PAPER_REBUILD_CONFIG_PATH))
        ensemble = _load_json_file(str(PAPER_TOP3_MANIFEST_PATH))
        states = list(ensemble.get("states", []))
        weights = np.asarray(ensemble.get("weights", []), dtype=np.float64)
        if len(states) != 3 or weights.shape != (3,) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("The paper inference manifest must contain three normalized checkpoint weights.")

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
                raise FileNotFoundError(f"Paper checkpoint is missing: {checkpoint_path}")
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
            raise ValueError("The paper inference bundle contains no serialized residual-RF model.")

        fusion = np.load(PAPER_FUSION_COMPONENTS_PATH, allow_pickle=False)
        component_names = [str(value) for value in fusion["component_names"].tolist()]
        nn_index = component_names.index("nn")
        residual_index = component_names.index("nn_plus_res_rf")
        train_fp_selected = np.asarray(dataset.fingerprint[train_idx][:, selected_idx] > 0.5, dtype=bool)
        df_ref = read_csv_robust(str(package / config["data"]["path"]))
        compound_col = next((column for column in ["chemical compound", "chemical_compound", "compound", "名称", "化学物质"] if column in df_ref.columns), None)
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
            "test_r2": V9_LOCKED_TEST_R2,
            "test_rmse": V9_LOCKED_TEST_RMSE,
        }
        archived_x = np.asarray(fusion["x_test"], dtype=np.float32)
        expected_nn = np.asarray(fusion["component_test"][nn_index], dtype=np.float32)
        reconstructed_nn = _paper_top3_prediction(bundle, archived_x)
        if float(np.max(np.abs(reconstructed_nn - expected_nn))) > 1e-5:
            raise RuntimeError("Top-3 checkpoint ensemble does not reproduce the archived neural component.")
        expected_final = np.asarray(fusion["component_test"][residual_index], dtype=np.float32)
        correction = np.mean(
            [model.predict(np.c_[archived_x, reconstructed_nn]) for model in residual_models], axis=0
        )
        if float(np.max(np.abs(reconstructed_nn + correction - expected_final))) > 1e-5:
            raise RuntimeError("Serialized residual RF does not reproduce the archived nn_plus_res_rf component.")
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
    """Single-sample inference with the paper's Top-3 nn_plus_res_rf route."""
    dataset = bundle["dataset"]
    selected_idx = np.asarray(bundle["selected_idx"], dtype=int).reshape(-1)
    fp_bits = int(bundle["fp_bits"])

    fp_dense = np.asarray(fp_dense, dtype=np.float32).reshape(-1)
    if fp_dense.size != fp_bits:
        raise ValueError(f"指纹长度不匹配：模型需要 {fp_bits}，当前为 {fp_dense.size}")

    fp_selected = fp_dense[selected_idx].astype(np.float32)
    num_vec = _build_v9_numeric_vector(dataset, ph_value=float(ph_value), category27_info=category27_info)
    if num_vec.size != int(bundle["numeric_dim"]):
        raise ValueError(f"数值分支维度不匹配：模型需要 {bundle['numeric_dim']}，当前为 {num_vec.size}")

    fusion_features = np.concatenate([fp_selected, num_vec], axis=0).astype(np.float32)
    if fusion_features.size != int(bundle["residual_feature_dim"]):
        raise ValueError("残差校正特征维度与锁定模型不一致")
    external_x = fusion_features.reshape(1, -1)
    pred_raw_nn = float(_paper_top3_prediction(bundle, external_x)[0])
    residual_input = np.c_[external_x, [pred_raw_nn]]
    residual = float(np.mean([model.predict(residual_input)[0] for model in bundle["residual_models"]]))
    pred_raw = pred_raw_nn + residual
    ad_detail = evaluate_ad_for_fingerprint(bundle, fp_dense, ph_value=ph_value)
    library_ad_detail = lookup_high_throughput_range_record(canonical_smiles)
    if library_ad_detail:
        online_label = ad_detail.get("DECAT_AD_label", "Not assessed")
        ad_detail.update(library_ad_detail)
        ad_detail["online_DECAT_AD_label"] = online_label
        try:
            if float(ph_value) < PH_TRAIN_MIN or float(ph_value) > PH_TRAIN_MAX:
                ad_detail["DECAT_AD_label"] = "Out of AD - pH outside train range"
                ad_detail["AD_reason"] = (
                    f"Matched in the screened high-throughput library, but pH={float(ph_value):.1f} "
                    f"is outside the training range [{PH_TRAIN_MIN:.1f}, {PH_TRAIN_MAX:.1f}]."
                )
        except Exception:
            pass
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


def evaluate_ad_for_fingerprint_matrix(
    bundle: Dict[str, object],
    fp_matrix: np.ndarray,
    ph_values: Optional[np.ndarray] = None,
    chunk_size: int = 512,
) -> pd.DataFrame:
    """Vectorized nearest-neighbor Tanimoto AD screening for uploaded libraries."""
    selected_idx = np.asarray(bundle["selected_idx"], dtype=int).reshape(-1)
    train = np.asarray(bundle["train_fp_selected_bool"], dtype=np.uint8)
    train_sums = train.sum(axis=1).astype(np.float32)
    fps = np.asarray(fp_matrix, dtype=np.float32)
    if fps.ndim == 1:
        fps = fps.reshape(1, -1)
    selected = (fps[:, selected_idx] > 0.5).astype(np.uint8)
    n = int(selected.shape[0])
    best_sim = np.zeros((n,), dtype=np.float32)
    best_idx = np.full((n,), -1, dtype=int)

    train_t = train.T.astype(np.uint8)
    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        block = selected[start:end]
        inter = block @ train_t
        block_sums = block.sum(axis=1).astype(np.float32).reshape(-1, 1)
        union = block_sums + train_sums.reshape(1, -1) - inter.astype(np.float32)
        sims = np.divide(inter.astype(np.float32), union, out=np.zeros_like(union, dtype=np.float32), where=union > 0)
        local_idx = np.argmax(sims, axis=1)
        best_idx[start:end] = local_idx.astype(int)
        best_sim[start:end] = sims[np.arange(end - start), local_idx].astype(np.float32)

    distance = 1.0 - best_sim
    structural_label = np.where(
        distance <= AD_STRICT_DISTANCE_THRESHOLD,
        "In AD",
        np.where(distance <= AD_BORDERLINE_DISTANCE_THRESHOLD, "Borderline", "Out of AD"),
    )
    final_label = structural_label.astype(object)
    if ph_values is not None:
        ph_arr = pd.to_numeric(pd.Series(ph_values), errors="coerce").to_numpy(dtype=float)
        outside = np.isfinite(ph_arr) & ((ph_arr < PH_TRAIN_MIN) | (ph_arr > PH_TRAIN_MAX))
        final_label[outside] = "Out of AD - pH outside train range"

    train_names = list(bundle.get("train_compounds", []))
    train_smiles = list(bundle.get("train_smiles", []))
    nearest_names = [
        train_names[int(i)] if 0 <= int(i) < len(train_names) else ""
        for i in best_idx
    ]
    nearest_smiles = [
        train_smiles[int(i)] if 0 <= int(i) < len(train_smiles) else ""
        for i in best_idx
    ]
    reasons = [
        f"nearest Tanimoto distance={float(d):.4f}; strict<= {AD_STRICT_DISTANCE_THRESHOLD:.4f}, borderline<= {AD_BORDERLINE_DISTANCE_THRESHOLD:.4f}"
        for d in distance
    ]
    return pd.DataFrame(
        {
            "nearest_train_similarity": best_sim.astype(float),
            "nearest_train_distance": distance.astype(float),
            "nearest_train_index": best_idx.astype(int),
            "nearest_train_compound": nearest_names,
            "nearest_train_SMILES": nearest_smiles,
            "structural_AD_label": structural_label,
            "DECAT_AD_label": final_label,
            "AD_reason": reasons,
        }
    )


def predict_batch_with_decat_v9(
    bundle: Dict[str, object],
    fp_matrix: np.ndarray,
    ph_values: np.ndarray,
    category_infos: List[Optional[Dict[str, object]]],
    batch_size: int = 256,
) -> np.ndarray:
    """Batch neural inference with the locked DECAT V9 model."""
    model = bundle["model"]
    dataset = bundle["dataset"]
    selected_idx = np.asarray(bundle["selected_idx"], dtype=int).reshape(-1)
    fps = np.asarray(fp_matrix, dtype=np.float32)
    if fps.ndim == 1:
        fps = fps.reshape(1, -1)
    fp_selected = fps[:, selected_idx].astype(np.float32)
    numeric_rows = [
        _build_v9_numeric_vector(dataset, ph_value=float(ph), category27_info=info)
        for ph, info in zip(np.asarray(ph_values, dtype=float).reshape(-1), category_infos)
    ]
    numeric = np.vstack(numeric_rows).astype(np.float32)
    preds_scaled = []
    for start in range(0, fp_selected.shape[0], int(batch_size)):
        end = min(fp_selected.shape[0], start + int(batch_size))
        fp_tensor = torch.tensor(fp_selected[start:end], dtype=torch.float32, device=DEVICE)
        num_tensor = torch.tensor(numeric[start:end], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pred = model(fp_tensor, num_tensor).detach().cpu().numpy().reshape(-1, 1)
        preds_scaled.append(pred.astype(np.float32))
    pred_scaled = np.vstack(preds_scaled)
    return dataset.logk_scaler.inverse_transform(pred_scaled).reshape(-1).astype(float)


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


def _mol_scope_flags(canonical_smiles: str) -> Dict[str, object]:
    if not canonical_smiles:
        return {
            "has_carbon": False,
            "scope_ok_for_DECAT_organic_model": False,
            "scope_reason": "missing_or_invalid_smiles",
        }
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return {
            "has_carbon": False,
            "scope_ok_for_DECAT_organic_model": False,
            "scope_reason": "invalid_smiles",
        }
    atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    has_carbon = 6 in atomic_nums
    metal_like = any(num in {3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 37, 38, 39, 40, 47, 48, 49, 50, 55, 56, 57, 78, 79, 80, 81, 82} for num in atomic_nums)
    return {
        "has_carbon": bool(has_carbon),
        "scope_ok_for_DECAT_organic_model": bool(has_carbon and not metal_like),
        "scope_reason": "ok" if has_carbon and not metal_like else ("non_carbon_or_inorganic" if not has_carbon else "metal_or_counterion"),
    }


def _read_uploaded_table(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    name = str(uploaded_file.name).lower()
    try:
        uploaded_file.seek(0)
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        return pd.read_csv(uploaded_file, encoding="utf-8-sig", low_memory=False)
    except Exception:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="gb18030", low_memory=False)
        except Exception as exc:
            st.warning(f"Uploaded table could not be parsed: {exc}")
            return None


def _pick_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    normalized = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for alias in aliases:
        key = str(alias).strip().lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    for c in df.columns:
        c_low = str(c).strip().lower()
        if any(str(alias).strip().lower() in c_low for alias in aliases):
            return c
    return None


def _prepare_high_throughput_input(df: pd.DataFrame, default_ph: float) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[Optional[Dict[str, object]]]]:
    smiles_col = _pick_column(df, ["canonical_SMILES", "normalized_SMILES", "SMILES", "smiles", "raw_SMILES"])
    if smiles_col is None:
        raise ValueError("No SMILES column was found. Please include SMILES, normalized_SMILES, or canonical_SMILES.")
    name_col = _pick_column(df, ["compound_name", "chemical compound", "compound", "name", "名称", "化学物质名称"])
    cas_col = _pick_column(df, ["CAS", "CAS号", "cas"])
    ph_col = _pick_column(df, ["pH", "source_pH", "ph"])

    out = df.copy()
    out["compound_name_for_screening"] = out[name_col].astype(str) if name_col else ""
    out["CAS_for_screening"] = out[cas_col].astype(str) if cas_col else ""
    cano = []
    status = []
    for val in out[smiles_col].tolist():
        c, s = _canonicalize_smiles_text(val)
        cano.append(c)
        status.append(s)
    out["canonical_SMILES"] = cano
    out["smiles_status"] = status

    scope_rows = [_mol_scope_flags(smi) for smi in out["canonical_SMILES"].tolist()]
    for key in ["has_carbon", "scope_ok_for_DECAT_organic_model", "scope_reason"]:
        out[key] = [row[key] for row in scope_rows]

    ph_values = (
        pd.to_numeric(out[ph_col], errors="coerce").to_numpy(dtype=float)
        if ph_col
        else np.full((len(out),), np.nan, dtype=float)
    )
    ph_for_prediction = np.where(np.isfinite(ph_values), ph_values, float(default_ph))
    out["pH_for_prediction"] = ph_for_prediction

    valid_mask = (out["smiles_status"] == "valid") & out["scope_ok_for_DECAT_organic_model"].astype(bool)
    fps = []
    valid_indices = []
    category_infos = []
    for idx, smi in enumerate(out["canonical_SMILES"].tolist()):
        if not bool(valid_mask.iloc[idx]):
            continue
        fp = smiles_to_fingerprint(smi, fp_size=3147)
        if fp is None:
            out.loc[out.index[idx], "smiles_status"] = "invalid"
            continue
        info = classify_category27_from_smiles(smi, manual_input=None)
        fps.append(fp)
        valid_indices.append(idx)
        category_infos.append(info)
    fp_matrix = np.vstack(fps).astype(np.float32) if fps else np.zeros((0, 3147), dtype=np.float32)
    return out, fp_matrix, np.asarray(valid_indices, dtype=int), category_infos


@st.cache_data(ttl=3600)
def load_builtin_high_throughput_results() -> Optional[pd.DataFrame]:
    if HIGH_THROUGHPUT_COMBINED_UNIQUE_RESULTS.exists():
        return read_csv_robust(str(HIGH_THROUGHPUT_COMBINED_UNIQUE_RESULTS))
    return None


def _clean_cell_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _cell_to_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "nan", "none", "null"}:
        return False
    try:
        return bool(float(text))
    except Exception:
        return False


def _finite_float_or_nan(value: object) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else float("nan")
    except Exception:
        return float("nan")


def lookup_high_throughput_range_record(canonical_smiles: str) -> Optional[Dict[str, object]]:
    """Return prior high-throughput screening status if the compound is already in the screened library."""
    if not canonical_smiles:
        return None
    df = load_builtin_high_throughput_results()
    if df is None or df.empty or "canonical_SMILES" not in df.columns:
        return None
    target = str(canonical_smiles).strip()
    hit = df[df["canonical_SMILES"].astype(str).str.strip() == target]
    if hit.empty:
        return None
    row = hit.iloc[0].to_dict()
    # The release result is the seed-242 re-screen.  Do not fall back to the
    # legacy V9 label because its thresholds and selected-bit space differ.
    ad_label = _clean_cell_text(row.get("seed242_AD_label"))
    source_bits = []
    if _cell_to_bool(row.get("has_previous_high_throughput_source", False)):
        source_bits.append("previous pollutant library")
    if _cell_to_bool(row.get("has_ACS_EST2026_source", False)):
        source_bits.append("ACS EST external library")
    source_text = "; ".join(source_bits) if source_bits else "screened high-throughput library"
    seed_distance = _finite_float_or_nan(row.get("seed242_nearest_train_distance", np.nan))
    if not _cell_to_bool(row.get("scope_ok_for_DECAT_organic_model", True)):
        seed_structural_label = "Out of DECAT scope - inorganic/no carbon"
    elif not np.isfinite(seed_distance):
        seed_structural_label = "Not evaluated"
    elif seed_distance <= AD_STRICT_DISTANCE_THRESHOLD:
        seed_structural_label = "In AD"
    elif seed_distance <= AD_BORDERLINE_DISTANCE_THRESHOLD:
        seed_structural_label = "Borderline"
    else:
        seed_structural_label = "Out of AD"
    if np.isfinite(seed_distance):
        seed_reason = (
            f"seed-242 nearest Tanimoto distance={seed_distance:.4f}; "
            f"strict<= {AD_STRICT_DISTANCE_THRESHOLD:.4f}, "
            f"borderline<= {AD_BORDERLINE_DISTANCE_THRESHOLD:.4f}"
        )
    else:
        seed_reason = "seed-242 nearest Tanimoto distance is unavailable"
    return {
        "high_throughput_library_match": True,
        "high_throughput_range_source": f"Matched in the {source_text}",
        "high_throughput_representative_name": _clean_cell_text(row.get("representative_name", "")),
        "high_throughput_representative_CAS": _clean_cell_text(row.get("representative_CAS", "")),
        "high_throughput_n_source_rows": row.get("n_source_rows", np.nan),
        "DECAT_AD_label": ad_label,
        "structural_AD_label": seed_structural_label,
        "nearest_train_similarity": _finite_float_or_nan(
            row.get("seed242_nearest_train_similarity", np.nan)
        ),
        "nearest_train_distance": _finite_float_or_nan(
            row.get("seed242_nearest_train_distance", np.nan)
        ),
        "nearest_train_index": row.get("nearest_train_index", -1),
        "nearest_train_compound": _clean_cell_text(row.get("nearest_train_compound", "")),
        "nearest_train_SMILES": _clean_cell_text(row.get("nearest_train_SMILES", "")),
        "AD_reason": seed_reason,
    }


@st.cache_resource
def load_transformer_v6_bundle(model_path: str, params_path: str, dataset_path: str):
    """加载 V6 checkpoint、参数与训练期预处理状态。"""
    if not os.path.exists(model_path):
        return None, None, "V6模型文件不存在"
    if not os.path.exists(params_path):
        return None, None, "V6参数文件不存在"
    if not os.path.exists(dataset_path):
        return None, None, "V6参考数据集不存在（用于还原缩放器与特征筛选）"

    try:
        v6_mod = _get_v6_runtime_module()
        payload = _load_json_file(params_path)
        params, payload_env, _ = _extract_v6_payload_fields(payload)
        _apply_v6_runtime_env(params, payload_env)

        fp_bits = int(params.get("fp_bits", 3147))
        topk_features = int(params.get("topk_features", params.get("topk", fp_bits)))
        topk_features = max(1, min(topk_features, fp_bits))
        fp_select = str(params.get("fp_select", params.get("fp_select_method", "rf"))).strip().lower() or "rf"

        dataset = v6_mod.FingerprintReactionDataset(dataset_path, max_fp_bits=fp_bits, fingerprint_scale=False)
        split_dict = v6_mod._build_split_indices(dataset)
        train_idx = np.asarray(split_dict["train_idx"], dtype=int)
        dataset.fit_scalers(train_idx)
        rank_idx = v6_mod._get_fp_ranking(dataset, train_idx, fp_bits, fp_select)
        selected_idx = np.asarray(rank_idx[:topk_features], dtype=int)

        cfg = v6_mod.FingerprintConfig()
        cfg.d_model = int(params.get("d_model", cfg.d_model))
        cfg.dropout = float(params.get("dropout", cfg.dropout))
        cfg.n_layers = int(params.get("n_layers", getattr(cfg, "n_layers", 2)))
        cfg.n_heads = int(params.get("n_heads", getattr(cfg, "n_heads", 4)))
        cfg.max_fp_tokens = int(params.get("max_fp_tokens", getattr(cfg, "max_fp_tokens", topk_features)))
        cfg.norm_first = bool(params.get("norm_first", getattr(cfg, "norm_first", False)))

        if cfg.n_heads <= 0:
            cfg.n_heads = 4
        if cfg.d_model % cfg.n_heads != 0:
            cfg.d_model = max(cfg.n_heads, (cfg.d_model // cfg.n_heads) * cfg.n_heads)

        numeric_dim = int(getattr(dataset, "num_dim", 1 + len(getattr(dataset, "category_cols", []))))
        mode = _normalize_v6_mode(params.get("model_mode", "dual"))
        if mode == "dual":
            model = v6_mod.DualExpertRegressor(
                fingerprint_dim=int(topk_features),
                numeric_dim=numeric_dim,
                config=cfg,
            ).to(DEVICE)
        elif mode == "attn":
            model = v6_mod._build_attention_expert(
                fingerprint_dim=int(topk_features),
                numeric_dim=numeric_dim,
                config=cfg,
            ).to(DEVICE)
        else:
            model = v6_mod.FingerprintTransformer(
                fingerprint_dim=int(topk_features),
                numeric_dim=numeric_dim,
                config=cfg,
            ).to(DEVICE)

        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.eval()

        bundle = {
            "model": model,
            "params": params,
            "dataset": dataset,
            "fp_bits": int(fp_bits),
            "selected_idx": selected_idx,
            "numeric_dim": int(numeric_dim),
            "mode": mode,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "payload_env": payload_env,
            "category_cols": list(getattr(dataset, "category_cols", [])),
        }
        return bundle, params, ""
    except Exception as e:
        return None, None, str(e)


def predict_with_transformer_v6(bundle: Dict[str, object], fp_dense: np.ndarray, ph_value: float, category27_info: Optional[Dict[str, object]]):
    """单样本 V6 推理，并自动反标准化回原始 logk。"""
    model = bundle["model"]
    params = bundle["params"]
    dataset = bundle["dataset"]
    fp_bits = int(bundle["fp_bits"])
    selected_idx = np.asarray(bundle["selected_idx"], dtype=int).reshape(-1)

    fp_dense = np.asarray(fp_dense, dtype=np.float32).reshape(-1)
    if fp_dense.size != fp_bits:
        raise ValueError(f"指纹长度不匹配：模型需要 {fp_bits}，当前为 {fp_dense.size}")

    fp_selected = fp_dense[selected_idx].astype(np.float32)
    num_vec = _build_v6_numeric_vector(dataset, ph_value=float(ph_value), category27_info=category27_info)
    if num_vec.size != int(bundle["numeric_dim"]):
        raise ValueError(f"数值分支维度不匹配：模型需要 {bundle['numeric_dim']}，当前为 {num_vec.size}")

    fp_tensor = torch.tensor(fp_selected.reshape(1, -1), dtype=torch.float32, device=DEVICE)
    num_tensor = torch.tensor(num_vec.reshape(1, -1), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        pred_scaled = model(fp_tensor, num_tensor)
        pred_scaled_val = float(pred_scaled.detach().cpu().numpy().reshape(-1)[0])

    pred_raw = float(dataset.logk_scaler.inverse_transform(np.array([[pred_scaled_val]], dtype=np.float32)).reshape(-1)[0])
    detail = {
        "pred_scaled": pred_scaled_val,
        "fp_bits": fp_bits,
        "topk_features": int(selected_idx.size),
        "max_fp_tokens": int(params.get("max_fp_tokens", 0)),
        "n_heads": int(params.get("n_heads", 0)),
        "n_layers": int(params.get("n_layers", 0)),
        "mode": str(bundle.get("mode", "dual")),
    }
    return pred_raw, detail


def normalize_family12_input(category_input):
    """将用户输入的类别文本映射到 family12 标准。"""
    if pd.isna(category_input):
        return None
    text = str(category_input).strip()
    if text == "":
        return None
    if text in FAMILY12_CODE_TO_INFO:
        return FAMILY12_CODE_TO_INFO[text]

    low = text.lower()
    if low in FAMILY12_ALIAS_TO_CODE:
        return FAMILY12_CODE_TO_INFO[FAMILY12_ALIAS_TO_CODE[low]]

    for alias, code in FAMILY12_ALIAS_TO_CODE.items():
        if alias and (alias in low or alias in text):
            return FAMILY12_CODE_TO_INFO[code]
    for item in FAMILY12_CLASSES:
        if low == item["name_en"].lower() or text == item["name_cn"]:
            return item
    return None


def classify_family12_from_smiles(smiles: Optional[str], manual_input: Optional[str] = None):
    """优先用 SMILES 按 family12 自动分类；无法分类时回退到手工输入。"""
    if smiles and str(smiles).strip():
        try:
            family12_mod = _get_family12_runtime_module()
            features = family12_mod.parse_features(str(smiles).strip())
            code, name_en, reason = family12_mod.classify_family12(features)
            info = dict(FAMILY12_EN_TO_INFO.get(name_en, FAMILY12_CODE_TO_INFO.get(code, {})))
            if info:
                info["reason_raw"] = reason
                info["reason_cn"] = FAMILY12_REASON_CN_MAP.get(reason, reason)
                info["source"] = "smiles_rule"
                return info
        except Exception:
            pass

    manual_info = normalize_family12_input(manual_input)
    if manual_info:
        info = dict(manual_info)
        info["reason_raw"] = "manual_input_match"
        info["reason_cn"] = "未获得可用 SMILES，已根据输入的类别文本映射到 family12 标准。"
        info["source"] = "manual_input"
        return info
    return None


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
def load_external_validation_category_lookup(path: str) -> Dict[str, str]:
    """Load only archived class labels needed to reproduce published external cases."""
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
    published_label = load_external_validation_category_lookup(str(PAPER_EXTERNAL_CASES_PATH)).get(canonical)
    if published_label:
        return locked_category_info(
            published_label,
            "Archived paper external-validation class assignment.",
            "paper_external_manifest",
        )
    code, reason = _classify_locked_category_from_mol(mol)
    return locked_category_info(code, reason, "locked_smiles_rule")


def _get_v4_runtime_modules():
    """Load legacy transformer base classes + V4 innovation module for inference."""
    base_candidates = [
        PROJECT_DIR / "trans优化.py",
        PROJECT_DIR / "原版transformer.py",
    ]
    v4_script_path = PROJECT_DIR / "transformer_v4.py"
    if not v4_script_path.exists():
        raise FileNotFoundError(f"缺少V4脚本：{v4_script_path}")

    base_module = None
    last_err = None
    for cand in base_candidates:
        if not cand.exists():
            continue
        try:
            mod = _load_module_from_file("_streamlit_transformer_base", str(cand))
            required = [
                "AttentionFingerprintTransformer",
                "FingerprintConfig",
                "FingerprintReactionDataset",
                "FingerprintTrainer",
                "FingerprintTransformer",
                "SelectedFeatureSubset",
            ]
            if all(hasattr(mod, x) for x in required):
                base_module = mod
                break
            last_err = RuntimeError(f"{cand} 缺少V4依赖类")
        except Exception as e:
            last_err = e
            continue
    if base_module is None:
        raise RuntimeError(f"无法加载V4基础模块：{last_err}")

    # 1) 先载入基础类，并注册为 `transformer`，供 transformer_v4.py 的 `import transformer` 使用
    sys.modules["transformer"] = base_module

    # 2) 再载入 V4 脚本（此时其依赖将落到上面的 base_module）
    v4_module = _load_module_from_file("_streamlit_transformer_v4", str(v4_script_path))
    return base_module, v4_module


def _build_v4_numeric_vector(dataset, ph_value: float, category_code: int) -> np.ndarray:
    """Build numeric branch: [scaled_pH] + [one-hot category]."""
    ph_arr = np.asarray([[float(ph_value)]], dtype=np.float32)
    ph_scaled = dataset.ph_scaler.transform(ph_arr).reshape(-1).astype(np.float32)

    cat_cols = list(getattr(dataset, "category_cols", []))
    cat_vec = np.zeros((len(cat_cols),), dtype=np.float32)
    target_col = f"cat_{int(category_code)}"
    if target_col in cat_cols:
        cat_vec[cat_cols.index(target_col)] = 1.0
    elif "cat_Unknown" in cat_cols:
        cat_vec[cat_cols.index("cat_Unknown")] = 1.0

    return np.concatenate([ph_scaled, cat_vec], axis=0).astype(np.float32)


@st.cache_resource
def load_transformer_v4_bundle(model_path: str, params_path: str, dataset_path: str):
    """Load Transformer V4 checkpoint + params + preprocessing state for robust inference."""
    if not os.path.exists(model_path):
        return None, None, "V4模型文件不存在"
    if not os.path.exists(params_path):
        return None, None, "V4参数文件不存在"
    if not os.path.exists(dataset_path):
        return None, None, "V4参考数据集不存在（用于还原缩放与特征选择）"

    try:
        base_mod, v4_mod = _get_v4_runtime_modules()

        with open(params_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        params = payload.get("params", payload)

        fp_bits = int(params.get("fp_bits", 4096))
        topk_features = int(params.get("topk_features", params.get("topk", fp_bits)))
        topk_features = max(1, min(topk_features, fp_bits))
        fp_select = str(params.get("fp_select", "rf")).strip().lower() or "rf"

        # 与训练一致：先构建数据集（含 pH/logk scaler），再按固定 split + RF 取 Top-K 特征
        dataset = base_mod.FingerprintReactionDataset(dataset_path, max_fp_bits=fp_bits, fingerprint_scale=False)
        split_dict = v4_mod._build_split_indices(dataset)
        train_idx = np.asarray(split_dict["train_idx"], dtype=int)
        rank_idx = v4_mod._get_fp_ranking(dataset, train_idx, fp_bits, fp_select)
        selected_idx = np.asarray(rank_idx[:topk_features], dtype=int)

        cfg = base_mod.FingerprintConfig()
        cfg.d_model = int(params.get("d_model", cfg.d_model))
        cfg.dropout = float(params.get("dropout", cfg.dropout))
        cfg.n_layers = int(params.get("n_layers", getattr(cfg, "n_layers", 1)))
        cfg.n_heads = int(params.get("n_heads", getattr(cfg, "n_heads", 4)))
        cfg.max_fp_tokens = int(params.get("max_fp_tokens", getattr(cfg, "max_fp_tokens", topk_features)))

        if cfg.n_heads <= 0:
            cfg.n_heads = 4
        if cfg.d_model % cfg.n_heads != 0:
            cfg.d_model = max(cfg.n_heads, (cfg.d_model // cfg.n_heads) * cfg.n_heads)

        numeric_dim = int(getattr(dataset, "num_dim", 1 + len(getattr(dataset, "category_cols", []))))
        mode_raw = str(params.get("model_mode", "dual")).strip().lower()
        mode = mode_raw.split("+", 1)[0] if "+" in mode_raw else mode_raw
        if mode not in {"dual", "attn", "mlp"}:
            mode = "dual"

        if mode == "dual":
            model = v4_mod.DualExpertRegressor(fingerprint_dim=int(topk_features), numeric_dim=numeric_dim, config=cfg).to(DEVICE)
        elif mode == "attn":
            model = v4_mod._build_attention_expert(
                fingerprint_dim=int(topk_features),
                numeric_dim=numeric_dim,
                config=cfg,
            ).to(DEVICE)
        else:
            model = base_mod.FingerprintTransformer(
                fingerprint_dim=int(topk_features),
                numeric_dim=numeric_dim,
                config=cfg,
            ).to(DEVICE)

        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.eval()

        bundle = {
            "model": model,
            "params": params,
            "dataset": dataset,
            "fp_bits": int(fp_bits),
            "selected_idx": selected_idx,
            "numeric_dim": int(numeric_dim),
            "mode": mode,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
        return bundle, params, ""
    except Exception as e:
        return None, None, str(e)

def load_dynamic_ann_model(model_path, input_dim, hidden_layer_sizes, dropout_rate):
    """加载动态配置的ANN模型"""
    try:
        # 加载模型权重，兼容两种保存方式：仅state_dict或全字典
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        else:
            state_dict = checkpoint

        resolved_input_dim, resolved_hidden_layer_sizes, resolved_dropout = _infer_architecture_from_checkpoint(
            checkpoint=checkpoint,
            default_input_dim=input_dim,
            default_hidden_layer_sizes=hidden_layer_sizes,
            default_dropout=dropout_rate,
        )
        if resolved_hidden_layer_sizes != list(hidden_layer_sizes) or abs(resolved_dropout - float(dropout_rate)) > 1e-12:
            st.info(
                f"Architecture metadata was detected in the checkpoint and applied automatically: "
                f"hidden={resolved_hidden_layer_sizes}, dropout={resolved_dropout:.3f}"
            )
        if resolved_input_dim != int(input_dim):
            st.info(
                f"The checkpoint expects input dimension {resolved_input_dim} rather than the current concatenated dimension {input_dim}. "
                f"Features will be aligned automatically."
            )

        model = DynamicANNRegressor(
            input_dim=resolved_input_dim,
            hidden_layer_sizes=resolved_hidden_layer_sizes,
            dropout_rate=resolved_dropout,
        ).to(DEVICE)

        state_dict = _remap_legacy_state_dict_to_dynamic(state_dict, resolved_hidden_layer_sizes)
        model.load_state_dict(state_dict)
        model.eval()  # 切换到评估模式

        feature_names = None
        y_scaler_mean = None
        y_scaler_scale = None
        fp_size_ckpt = FP_SIZE
        if isinstance(checkpoint, dict):
            feat = checkpoint.get("feature_names")
            if isinstance(feat, (list, tuple)) and len(feat) > 0:
                feature_names = [str(x) for x in feat]
            if "y_scaler_mean_" in checkpoint and "y_scaler_scale_" in checkpoint:
                try:
                    y_scaler_mean = float(np.asarray(checkpoint["y_scaler_mean_"]).reshape(-1)[0])
                    y_scaler_scale = float(np.asarray(checkpoint["y_scaler_scale_"]).reshape(-1)[0])
                except Exception:
                    y_scaler_mean = None
                    y_scaler_scale = None
            if (y_scaler_mean is None or y_scaler_scale is None) and checkpoint.get("y_scaler") is not None:
                try:
                    scaler_obj = checkpoint.get("y_scaler")
                    y_scaler_mean = float(np.asarray(getattr(scaler_obj, "mean_", None)).reshape(-1)[0])
                    y_scaler_scale = float(np.asarray(getattr(scaler_obj, "scale_", None)).reshape(-1)[0])
                except Exception:
                    y_scaler_mean = None
                    y_scaler_scale = None
            try:
                fp_size_ckpt = int(checkpoint.get("fp_size", FP_SIZE))
            except Exception:
                fp_size_ckpt = FP_SIZE
            if fp_size_ckpt == FP_SIZE and feature_names:
                fp_name_count = len([name for name in feature_names if str(name).lower().startswith("fp_")])
                if fp_name_count > 0:
                    fp_size_ckpt = int(fp_name_count)

        model.expected_input_dim = int(resolved_input_dim)
        model.feature_names = feature_names
        model.y_scaler_mean = y_scaler_mean
        model.y_scaler_scale = y_scaler_scale
        model.fp_size_ckpt = fp_size_ckpt
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.info("Please confirm that the checkpoint path is correct, the file is readable, and the checkpoint is not corrupted.")
        return None


def predict_with_transformer_v4(bundle: Dict[str, object], fp_dense: np.ndarray, ph_value: float, category_code: int):
    """Run single-sample V4 inference and inverse-transform to raw logk."""
    model = bundle["model"]
    params = bundle["params"]
    dataset = bundle["dataset"]
    fp_bits = int(bundle["fp_bits"])
    selected_idx = np.asarray(bundle["selected_idx"], dtype=int).reshape(-1)

    fp_dense = np.asarray(fp_dense, dtype=np.float32).reshape(-1)
    if fp_dense.size != fp_bits:
        raise ValueError(f"指纹长度不匹配：模型需要 {fp_bits}，当前为 {fp_dense.size}")

    fp_selected = fp_dense[selected_idx].astype(np.float32)
    num_vec = _build_v4_numeric_vector(dataset, ph_value=float(ph_value), category_code=int(category_code))
    if num_vec.size != int(bundle["numeric_dim"]):
        raise ValueError(f"数值分支维度不匹配：模型需要 {bundle['numeric_dim']}，当前为 {num_vec.size}")

    fp_tensor = torch.tensor(fp_selected.reshape(1, -1), dtype=torch.float32, device=DEVICE)
    num_tensor = torch.tensor(num_vec.reshape(1, -1), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        pred_scaled = model(fp_tensor, num_tensor)
        pred_scaled_val = float(pred_scaled.detach().cpu().numpy().reshape(-1)[0])

    pred_raw = float(dataset.logk_scaler.inverse_transform(np.array([[pred_scaled_val]], dtype=np.float32)).reshape(-1)[0])
    detail = {
        "pred_scaled": pred_scaled_val,
        "fp_bits": fp_bits,
        "topk_features": int(selected_idx.size),
        "max_fp_tokens": int(params.get("max_fp_tokens", 0)),
        "n_heads": int(params.get("n_heads", 0)),
        "n_layers": int(params.get("n_layers", 0)),
    }
    return pred_raw, detail


def build_model_input_vector(fp, ph_value, category_code, model):
    """Build feature vector in model training order (supports Top-K feature_names)."""
    fp = np.asarray(fp, dtype=np.float32).reshape(-1)
    feature_names = getattr(model, "feature_names", None)
    expected_dim = int(getattr(model, "expected_input_dim", fp.size + 2))

    if not feature_names:
        full = np.concatenate([fp, np.array([ph_value, category_code], dtype=np.float32)], axis=0)
        if full.size != expected_dim:
            raise ValueError(
                f"模型期望输入维度={expected_dim}，但当前构造维度={full.size}。"
                "该模型可能使用了特征筛选，请使用包含 feature_names 的模型文件。"
            )
        return full, []

    value_by_name = {
        "pH": float(ph_value),
        "pH_processed": float(ph_value),
        "category": float(category_code),
    }

    selected = []
    missing = []
    for name in feature_names:
        if name in value_by_name:
            selected.append(value_by_name[name])
            continue
        if name.startswith("fp_"):
            try:
                idx = int(name.split("_", 1)[1])
                selected.append(float(fp[idx]) if 0 <= idx < fp.size else 0.0)
                if not (0 <= idx < fp.size):
                    missing.append(name)
            except Exception:
                selected.append(0.0)
                missing.append(name)
            continue
        selected.append(0.0)
        missing.append(name)

    x = np.asarray(selected, dtype=np.float32)
    if x.size != expected_dim:
        raise ValueError(f"按 feature_names 构造后维度={x.size}，与模型期望维度={expected_dim}不一致。")
    return x, missing


@st.cache_data(ttl=1800)
def load_local_dataset(uploaded_file):
    """加载本地CSV数据集，兼容不同编码"""
    if uploaded_file is None:
        return None
    encodings_order = ["utf-8", "utf-8-sig", "gb18030", "gbk", "cp936", "latin1"]
    for enc in encodings_order:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc, low_memory=False)
        except Exception:
            continue
    st.warning("Dataset parsing failed: the file encoding or CSV format is not supported.")
    return None


def parse_fingerprint_string(fp_str, expected_len: Optional[int] = None):
    """将字符串形式的指纹转换为numpy数组"""
    if not isinstance(fp_str, str) or fp_str.strip() == "":
        return None
    parts = [p for p in re.split(r"[,\s]+", fp_str.strip()) if p]
    if expected_len is not None and len(parts) != int(expected_len):
        return None
    try:
        return np.array([float(p) for p in parts], dtype=np.float32)
    except Exception:
        return None


def find_compound_in_dataset(chemical_name, df):
    """在本地数据集中查找化学物质，命中则返回信息"""
    if df is None or df.empty or not chemical_name:
        return None
    target = chemical_name.strip().lower()
    name_cols = ["化学物质名称", "名称", "name", "Name", "中文名称", "英文名称"]
    cas_cols = ["CAS号", "CAS", "cas"]
    smiles_cols = ["SMILES", "smiles"]
    fp_cols = ["分子指纹", "fingerprint", "Fingerprint", "fp"]
    cat_cols = ["类别编码", "category_code", "category", "类别"]
    row = None
    for col in name_cols:
        if col in df.columns:
            hit = df[df[col].astype(str).str.strip().str.lower() == target]
            if not hit.empty:
                row = hit.iloc[0]
                break
    if row is None:
        return None
    def pick_first(cols):
        for c in cols:
            if c in df.columns and pd.notna(row[c]):
                return row[c]
        return None
    cas = pick_first(cas_cols)
    smiles = pick_first(smiles_cols)
    fp_raw = pick_first(fp_cols)
    fp = parse_fingerprint_string(fp_raw) if fp_raw is not None else None
    cat = pick_first(cat_cols)
    try:
        cat_code = int(cat)
    except Exception:
        cat_code = None
    category27_info = None
    category27_code = pick_first(["category27_code", "new_category_code", "family12_category_code"])
    category27_name = pick_first(["category27_name", "new_category_name", "family12_category_name"])
    category27_label = pick_first(["category27_label"])
    category27_reason = pick_first(["category27_reason"])
    for item in [category27_label, category27_code, category27_name]:
        category27_info = normalize_category27_input(item)
        if category27_info:
            category27_info = dict(category27_info)
            category27_info["source"] = "uploaded_dataset"
            category27_info["category27_reason"] = str(category27_reason) if category27_reason is not None else "Loaded from the uploaded dataset."
            category27_info["label"] = category27_info.get("category27_label")
            break
    return {
        "cas": cas,
        "smiles": smiles,
        "fingerprint": fp,
        "category_code": cat_code,
        "category27_info": category27_info,
    }


@st.cache_data(ttl=3600)
def load_reference_dataset(dataset_path: Optional[str] = None):
    """加载本地参考数据集，仅保留核心列。"""
    try:
        ref_path = dataset_path or DATASET_PATH
        df = read_csv_robust(ref_path)
        col_lower = {col: col.strip().lower() for col in df.columns}
        def pick(*names):
            normalized_names = [str(name).strip().lower().replace(" ", "") for name in names]
            for name in names:
                for col, low in col_lower.items():
                    if low == str(name).strip().lower() or low.replace(" ", "") in normalized_names:
                        return col
            return None
        compound_col = pick("chemical compound", "chemical_compound", "compound", "化学品", "化学物质", "名称")
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


def normalize_category_input(category_input):
    """标准化类别输入（处理数字、中英文及大小写差异）。"""
    if pd.isna(category_input):
        return None
    text = str(category_input).strip()
    if text == "":
        return None
    # 纯数字或类似 "1" / "1.0" / "01"
    try:
        code_val = int(float(text))
        if code_val in LEGACY_CATEGORY_CODES:
            return code_val
    except Exception:
        pass
    category_input = text.lower()
    # 中英文名称映射
    if "烷" in category_input or "alkane" in category_input:
        return 0
    elif "醇" in category_input and "二醇" not in category_input or "alcohol" in category_input and "diol" not in category_input:
        return 1
    elif "二醇" in category_input or "diol" in category_input:
        return 2
    elif "醚" in category_input or "ether" in category_input:
        return 3
    elif "酮" in category_input or "ketone" in category_input:
        return 4
    elif "醛" in category_input or "aldehyde" in category_input:
        return 5
    elif "酯" in category_input or "ester" in category_input:
        return 6
    elif "羧酸" in category_input and "二元" not in category_input or "carboxyl" in category_input and "dicarboxylic" not in category_input:
        return 7
    elif "二元羧酸" in category_input or "dicarboxylic" in category_input:
        return 8
    elif "卤" in category_input or "halogen" in category_input:
        return 9
    elif "硫化物" in category_input or "disulfide" in category_input:
        return 10
    elif "烯" in category_input or "alkene" in category_input:
        return 11
    elif "苯" in category_input or "benzene" in category_input:
        return 12
    elif "酚" in category_input or "phenol" in category_input:
        return 13
    else:
        return 14


def find_compound_record(name, df):
    """在参考数据集中查找与化学物质名称匹配的记录。"""
    if df is None or not name:
        return None
    name_norm = str(name).strip().lower()
    df_local = df.copy()
    df_local["compound_norm"] = df_local["compound"].astype(str).str.strip().str.lower()
    hit = df_local[df_local["compound_norm"] == name_norm]
    if hit.empty:
        return None
    row = hit.iloc[0]
    category_code = normalize_category_input(row.get("category")) if "category" in row else None
    category27_info = locked_category_info(
        row.get("category27_label"),
        str(row.get("category27_reason") or "Loaded from the reference dataset."),
        "reference_dataset",
    )

    # 参考数据集中若包含 FP_1..FP_4096，则可直接使用，避免依赖外部网络查询SMILES。
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
        "category_code": category_code,
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


def render_section_heading(kicker: str, title: str, description: str):
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-kicker">{escape(kicker)}</div>
            <h2 class="section-title">{escape(title)}</h2>
            <div class="section-desc">{escape(description)}</div>
        </div>
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


def render_feature_cards(cards: List[Dict[str, str]]):
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{escape(card['kicker'])}</div>
                    <div class="feature-title">{escape(card['title'])}</div>
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
        (7, "Prediction Range", "Judge whether the compound is covered by the high-throughput prediction range."),
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


def render_insight_card(label: str, value: str, description: str):
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-label">{escape(label)}</div>
            <div class="insight-value">{escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_timeline(step_events: List[Dict[str, str]]):
    if not step_events:
        return
    for item in step_events:
        tone = item.get("tone", "info")
        st.markdown(
            f"""
            <div class="step-card {escape(tone)}">
                <div class="step-header">
                    <div>
                        <div class="step-title">{escape(item.get('title', ''))}</div>
                    </div>
                    <span class="step-state {escape(tone)}">{escape(item.get('state', tone))}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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
                High-throughput Prediction Range
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


def family12_reason_for_display(info: Optional[Dict[str, str]]) -> str:
    if not info:
        return "Not available."
    if info.get("reason_raw"):
        return str(info.get("reason_raw"))
    source = str(info.get("source", "")).strip().lower()
    if source == "uploaded_dataset":
        return "Loaded from the uploaded dataset."
    if source == "reference_dataset":
        return "Loaded from the reference dataset."
    return "Family12 assignment completed."


def family12_label_for_display(info: Optional[Dict[str, str]]) -> str:
    if not info:
        return "Not assigned"
    code = str(info.get("code", "")).strip()
    name_en = str(info.get("name_en", info.get("name_cn", ""))).strip()
    return f"{code} | {name_en}" if code else name_en


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


# ======================== Streamlit主界面构建 ========================
def main():
    st.set_page_config(page_title="Reaction Rate Constant Prediction Workflow", page_icon="📄", layout="wide")
    inject_custom_styles()

    latest_v9_model, latest_v9_params, latest_v9_dataset, latest_v9_summary = _find_locked_v9_artifacts()
    transformer_model_path = latest_v9_model
    transformer_params_path = latest_v9_params
    transformer_summary_path = latest_v9_summary
    reference_dataset_path = latest_v9_dataset or DATASET_PATH

    render_hero(dataset_ready=os.path.exists(reference_dataset_path))
    if not os.path.exists(reference_dataset_path):
        render_alert_banner(
            "The validated reference dataset is not available. Restore the dataset path before running the publication workflow.",
            tone="error",
        )

    # 横向布局：输入 -> 处理 -> 结果
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
                        Checking high-throughput prediction range...
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
        if os.path.exists(transformer_params_path):
            try:
                params_payload = _load_json_file(transformer_params_path)
                params_preview, _, _ = _extract_v9_payload_fields(params_payload)
                target_fp_size = int(params_preview.get("fp_bits", 3147))
            except Exception:
                target_fp_size = 3147

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
            model_path=transformer_model_path,
            params_path=transformer_params_path,
            dataset_path=reference_dataset_path,
            summary_path=transformer_summary_path,
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
        range_source = str(detail.get("high_throughput_range_source", "") or "")
        step_events.append(
            {
                "index": 7,
                "title": "prediction range",
                "state": "Library match" if bool(detail.get("high_throughput_library_match", False)) else "Online assessment",
                "tone": "success" if str(detail.get("DECAT_AD_label", "")) == "In AD" else ("warn" if "Borderline" in str(detail.get("DECAT_AD_label", "")) else "error"),
                "message": range_source or "The high-throughput prediction range rule was applied to this compound.",
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
    # If launched via plain python, re-launch with `streamlit run` to avoid bare-mode runtime warnings.
    if not _running_in_streamlit():
        script_path = os.path.abspath(__file__)
        print("检测到通过 python 直接运行，正在自动以 `streamlit run` 方式重新启动…")
        try:
            # inherit environment; don't raise to allow graceful fallback
            subprocess.run(["streamlit", "run", script_path], check=False)
        except FileNotFoundError:
            print("未找到 streamlit 可执行文件，请先在当前环境安装后再试。")
        raise SystemExit(0)
    main()

