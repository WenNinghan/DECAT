# -*- coding: utf-8 -*-
"""
transformer_v9_transformer_centered.py

基于 V8 PINN 扩展的单文件 V9 版本：
- V9 将双专家解释从“对称专家融合”改为“Transformer 主预测 + MLP 有界残差校正”
- 默认保留弱 pH 有限差分物理先验，并建议关闭 V8 曲率 PINN 残差
- 引入化学感知注意力偏差（chemistry-aware attention bias）用于Transformer自注意力机制
- 双专家模型（注意力专家 + MLP专家），带有不确定性感知门控（uncertainty-aware gating）
- 因果不变训练目标（causal-invariant training objective）：预测 + 不变性 + 组分布鲁棒优化（Group-DRO） + 物理先验
- 自适应多模型融合（神经网络 + 直方图梯度提升 + 随机森林 + 残差/堆叠学习器），使用单纯形权重搜索
- 支持消融实验 + 配对统计显著性分析
"""

import copy
import csv
import glob
import json
import os
import random
import warnings
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# 设置matplotlib字体以支持生成可编辑PDF
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


def _savefig_with_pdf(fig_or_plt, output_path, **kwargs):
    """
    保存图片为PNG/JPG格式，同时保存为Illustrator可编辑文本的PDF格式。
    
    参数:
        fig_or_plt: matplotlib的figure对象或plt模块
        output_path: 输出文件路径
        **kwargs: 传递给savefig的其他参数
    """
    output_path = str(output_path)
    fig_or_plt.savefig(output_path, **kwargs)
    base, ext = os.path.splitext(output_path)
    # 如果输出扩展名不是pdf，则额外保存一份pdf版本
    if ext.lower() != ".pdf":
        pdf_kwargs = dict(kwargs)
        # PDF保存通常不需要dpi参数（或者是矢量图），移除dpi以避免警告或错误
        pdf_kwargs.pop("dpi", None)
        fig_or_plt.savefig(base + ".pdf", **pdf_kwargs)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdFMCS, rdMolDescriptors
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

# 尝试导入科学计算统计库，如果不可用则设为None
try:
    from scipy.stats import gaussian_kde, pearsonr, ttest_rel, wilcoxon
except Exception:
    gaussian_kde = None
    pearsonr = None
    ttest_rel = None
    wilcoxon = None

# 忽略警告信息，保持输出整洁
warnings.filterwarnings("ignore")
# 禁用RDKit的日志输出
RDLogger.DisableLog("rdApp.*")


# =============================================================================
# 全局状态 / 路径配置
# =============================================================================
SCRIPT_DIR = os.path.dirname(__file__)

CATEGORY_LABEL_PRIORITY = (
    "category27_label",
    "category27_code",
    "category27_name",
)

GLOBAL_DESCRIPTOR_COLUMNS = (
    "desc_mol_wt",
    "desc_log_p",
    "desc_tpsa",
    "desc_hbd",
    "desc_hba",
    "desc_rotatable",
    "desc_ring_count",
    "desc_fraction_csp3",
    "desc_heavy_atoms",
    "desc_formal_charge",
    "desc_aromatic_fraction",
    "desc_hetero_fraction",
)

REACTIVITY_FAMILY_BY_CATEGORY_CODE = {
    "A": "saturated_hydrocarbon",
    "T": "saturated_hydrocarbon",
    "U": "unsaturated_hydrocarbon",
    "V": "unsaturated_hydrocarbon",
    "B": "hydroxyl",
    "C": "hydroxyl",
    "D": "ether",
    "E": "neutral_carbonyl",
    "F": "neutral_carbonyl",
    "G": "carboxyl_derivative",
    "H": "carboxyl_derivative",
    "I": "carboxyl_derivative",
    "J": "halogenated",
    "K": "reduced_sulfur",
    "M": "reduced_sulfur",
    "L": "oxidized_sulfur",
    "N": "neutral_nitrogen",
    "O": "neutral_nitrogen",
    "P": "neutral_nitrogen",
    "R": "neutral_nitrogen",
    "Y": "neutral_nitrogen",
    "Q": "basic_nitrogen",
    "W": "basic_nitrogen",
    "Z": "basic_nitrogen",
    "S": "phosphorus",
}


def _normalized_global_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Return bounded, whole-molecule descriptors for the MLP expert."""
    heavy = max(1, int(mol.GetNumHeavyAtoms()))
    aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})
    formal_charge = sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms())
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


def _resolve_preferred_category_labels(df: pd.DataFrame) -> Optional[pd.Series]:
    """优先解析 V6 的 27 类标签；不存在时返回 None 交给 V7 的 category 兜底。"""
    resolved = pd.Series(pd.NA, index=df.index, dtype="object")
    found = False
    for col in CATEGORY_LABEL_PRIORITY:
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


def _resolve_default_data_csv_path() -> str:
    """V6/V7 兼容的数据路径解析：显式环境变量优先，其次使用清洗重分类数据。"""
    override = str(
        os.environ.get(
            "TRANSFORMER_V7_DATA_CSV",
            os.environ.get(
                "TRANSFORMER_V6_DATA_CSV",
                os.environ.get("TRANSFORMER_DATA_CSV", ""),
            ),
        )
    ).strip()
    if override and os.path.isfile(override):
        return override
    candidates = [
        os.path.join(SCRIPT_DIR, "补充后4096位分子指纹数据默认pH=7(整体)_按物质家族重分类12类_去除独立金属与无机离子_20260320.csv"),
        os.path.join(SCRIPT_DIR, "补充后4096位分子指纹数据默认pH=7(整体)_按物质家族重分类12类_20260317.csv"),
        os.path.join(SCRIPT_DIR, "补充后4096位分子指纹数据默认pH=7(整体).csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[-1]


def _resolve_output_root() -> str:
    """Resolve the base output directory for the current project."""
    override = str(
        os.environ.get(
            "TRANSFORMER_V9_OUTPUT_ROOT",
            os.environ.get("DECAT_OUTPUT_ROOT", ""),
        )
    ).strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(SCRIPT_DIR, "transformer_v9_transformer_centered")


# 数据文件路径，默认优先指向 V6 清洗/重分类数据；也支持环境变量覆盖
DATA_CSV_PATH = _resolve_default_data_csv_path()


def _get_data_csv_path() -> str:
    path = _resolve_default_data_csv_path()
    return path if path else DATA_CSV_PATH
# 输出目录
OUT_DIR = _resolve_output_root()
os.makedirs(OUT_DIR, exist_ok=True)

# 记录最佳结果的字典
BEST_RESULT: Dict[str, object] = {
    "iter": 0,
    "objective_target": "val",  # 优化目标：验证集(val)或测试集(test)
    "objective_r2": float("-inf"),
    "val_r2": float("-inf"),
    "val_rmse": float("nan"),
    "test_r2": float("nan"),
    "test_rmse": float("nan"),
    "train_r2": float("nan"),
    "train_rmse": float("nan"),
    "model_mode": "",
    "fusion_mode": "",
    "params": {},
}

GLOBAL_ITER = 0
SPLIT_CACHE: Dict[str, np.ndarray] = {}  # 数据集划分缓存
FP_RANK_CACHE: Dict[Tuple[int, str], np.ndarray] = {}  # 特征排序缓存
DATASET_BASE_CACHE: Dict[Tuple[str, int, int, int, int], Dict[str, object]] = {}  # 数据集基础缓存
DATASET_CACHE_MAX_ENTRIES = 12
# 子结构与位映射缓存
BIT_SUBSTRUCTURE_CACHE: Dict[Tuple[str, int, int, Tuple[int, ...], Tuple[int, ...]], Dict[int, Dict[str, object]]] = {}
OBJECTIVE_TARGET = "val"  # 贝叶斯优化目标：默认使用验证集，之后在训练+验证集上重新训练最终模型
RUN_OUTPUT_DIR: Optional[str] = None  # 当前运行的输出目录
LAST_TRAINED_MODEL: Optional[nn.Module] = None  # 诊断用途：保留最近一次 objective 训练后的模型引用

def _clean_smiles_text(value: object) -> str:
    """清洗 SMILES 字符串中的零宽空格、BOM 和首尾空白。"""
    if value is None:
        return ""
    text = str(value)
    return (
        text.replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\u200e", "")
        .replace("\xa0", " ")
        .strip()
    )


def _match_feature_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """按需截断或补零，使特征维度匹配线性层输入。"""
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

# =============================================================================
# 独立的自包含基础组件 (无需外部transformer导入)
# =============================================================================
class FingerprintConfig:
    """
    模型配置类，定义超参数默认值。
    """
    d_model = 256  # 模型维度
    dropout = 0.3  # Dropout比率
    activation = "gelu"  # 激活函数
    n_heads = 4  # 注意力头数
    n_layers = 2  # Transformer层数
    max_fp_tokens = 128  # 最大指纹token数（Top-K特征）

    batch_size = 32  # 批次大小
    learning_rate = 1e-4  # 学习率
    weight_decay = 1e-2  # 权重衰减
    max_epochs = 100  # 最大训练轮数
    early_stopping_patience = 80  # 早停耐心值
    save_interval = 10  # 保存间隔
    min_delta = 1e-4  # 最小改进阈值
    scheduler_type = "plateau"  # 学习率调度器类型
    fingerprint_scale = False  # 是否对指纹进行缩放
    norm_first = False  # Pre-LN vs Post-LN
    attn_pooling = "attn"  # 池化方式：attn (注意力池化), mean (平均池化), cls (CLS标记)
    fp_bit_dropout = 0.0  # 指纹位Dropout比率
    base_numeric_dim = 0  # pH + category 分支维度


class FingerprintReactionDataset(Dataset):
    """
    指纹反应数据集类，处理CSV数据加载、分子指纹生成及预处理。
    """
    @staticmethod
    def _build_cache_key(csv_path: str, max_fp_bits: int) -> Tuple[str, int, int, int, int]:
        """构建数据集缓存键值，基于文件路径、修改时间和大小。"""
        abs_path = os.path.abspath(csv_path)
        try:
            stat = os.stat(abs_path)
            mtime_ns = int(stat.st_mtime_ns)
            size = int(stat.st_size)
        except Exception:
            mtime_ns = 0
            size = 0
        aggregate_duplicates = int(
            _env_bool_pair("TRANSFORMER_V7_AGGREGATE_DUPLICATE_TARGETS", "TRANSFORMER_V3_AGGREGATE_DUPLICATE_TARGETS", False)
        )
        return abs_path, int(max_fp_bits), mtime_ns, size, aggregate_duplicates

    @staticmethod
    def _cache_store(cache_key: Tuple[str, int, int, int, int], payload: Dict[str, object]) -> None:
        """存储数据集到内存缓存，带有LRU淘汰机制。"""
        global DATASET_BASE_CACHE
        if cache_key in DATASET_BASE_CACHE:
            DATASET_BASE_CACHE[cache_key] = payload
            return
        if len(DATASET_BASE_CACHE) >= int(DATASET_CACHE_MAX_ENTRIES):
            oldest_key = next(iter(DATASET_BASE_CACHE.keys()))
            DATASET_BASE_CACHE.pop(oldest_key, None)
        DATASET_BASE_CACHE[cache_key] = payload

    def __init__(self, csv_path: str, max_fp_bits: int, fingerprint_scale: bool = False):
        """
        初始化数据集。
        
        参数:
            csv_path: CSV数据文件路径
            max_fp_bits: 最大指纹位数
            fingerprint_scale: 是否对指纹特征进行缩放（通常指纹是0/1，不需要缩放）
        """
        super().__init__()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        cache_key = self._build_cache_key(csv_path, int(max_fp_bits))
        cache_item = DATASET_BASE_CACHE.get(cache_key)

        # 如果没有缓存，则加载数据并计算指纹
        if cache_item is None:
            df = pd.read_csv(csv_path)
            # 查找SMILES列
            # 查找SMILES列
            smiles_col = "SMILES" if "SMILES" in df.columns else ("smiles" if "smiles" in df.columns else None)
            if smiles_col is None:
                raise ValueError("SMILES column not found.")
            df[smiles_col] = df[smiles_col].map(_clean_smiles_text)
            if "pH" not in df.columns:
                raise ValueError("pH column not found.")

            # 查找目标变量列（logk）
            target_col = next((c for c in ["logk", "LogK", "LOGK", "logK"] if c in df.columns), None)
            if target_col is None:
                raise ValueError("logk column not found.")

            # V6 优先使用 category27_* 主分类；若数据还没标注，则保留 V7 的 category 兜底。
            preferred_cat = _resolve_preferred_category_labels(df)
            if preferred_cat is not None:
                df["__v7_category_label__"] = preferred_cat.astype(str)
                category_col = "__v7_category_label__"
            elif "category" in df.columns:
                category_col = "category"
            else:
                df["__v7_category_label__"] = "Unknown"
                category_col = "__v7_category_label__"

            # 过滤无效数据：SMILES, pH, logk 都必须非空
            valid_mask = df[[smiles_col, "pH", target_col]].notna().all(axis=1)
            keep_cols = [smiles_col, "pH", target_col, category_col]
            df_core = df.loc[valid_mask, keep_cols].copy().reset_index(drop=True)
            df_core[smiles_col] = df_core[smiles_col].astype(str).str.replace(r"\s+", "", regex=True)

            aggregate_duplicates = _env_bool_pair(
                "TRANSFORMER_V7_AGGREGATE_DUPLICATE_TARGETS",
                "TRANSFORMER_V3_AGGREGATE_DUPLICATE_TARGETS",
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

            # 处理类别列并进行 One-Hot 编码
            cat_series = df_core[category_col].fillna("Unknown").astype(str)
            x_cat = pd.get_dummies(cat_series, prefix="cat", dtype=np.uint8)
            cat_cols = x_cat.columns.tolist()
            primary_category_dim = len(cat_cols)
            if _env_bool_pair("TRANSFORMER_V9_HIERARCHICAL_CATEGORY", None, False):
                category_code = cat_series.str.split(":", n=1).str[0].str.strip()
                family_series = category_code.map(REACTIVITY_FAMILY_BY_CATEGORY_CODE).fillna("other")
                x_family = pd.get_dummies(family_series, prefix="family", dtype=np.uint8)
                x_cat = pd.concat([x_cat, x_family], axis=1)
                cat_cols = cat_cols + x_family.columns.tolist()
            if _env_bool_pair("TRANSFORMER_V9_CATEGORY_KINETIC_PRIOR", None, False):
                for prior_col in (
                    "kinetic_category_level",
                    "kinetic_ph_response",
                    "kinetic_category_reliability",
                ):
                    x_cat[prior_col] = np.float32(0.0)
                    cat_cols.append(prior_col)
            structural_clusters = int(
                _env_value("TRANSFORMER_V9_STRUCTURAL_CLUSTERS", None, "0")
            )
            structural_clusters = max(0, min(32, structural_clusters))
            for cluster_idx in range(structural_clusters):
                cluster_col = f"structure_cluster_{cluster_idx:02d}"
                x_cat[cluster_col] = np.uint8(0)
                cat_cols.append(cluster_col)

            use_global_descriptors = _env_bool_pair(
                "TRANSFORMER_V9_GLOBAL_DESCRIPTORS", None, False
            )
            morgan_radius = int(_env_value("TRANSFORMER_V9_MORGAN_RADIUS", None, "2"))
            morgan_radius = max(1, min(4, morgan_radius))
            fps_list = []
            descriptor_list = []
            keep_idx = []
            # 遍历SMILES生成分子指纹
            for i, smi in enumerate(df_core[smiles_col].astype(str).values):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                # 生成Morgan指纹，半径2，位数为max_fp_bits
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, int(max_fp_bits))
                arr = np.zeros((int(max_fp_bits),), dtype=np.int8)
                Chem.DataStructs.ConvertToNumpyArray(fp, arr)
                fps_list.append(arr)
                if use_global_descriptors:
                    descriptor_list.append(_normalized_global_descriptors(mol))
                keep_idx.append(i)
            if not fps_list:
                raise ValueError("No valid fingerprints could be generated from SMILES.")

            # 根据生成的指纹过滤原始数据
            df_core = df_core.iloc[keep_idx].reset_index(drop=True)
            x_cat = x_cat.iloc[keep_idx].reset_index(drop=True)
            if use_global_descriptors:
                descriptor_frame = pd.DataFrame(
                    np.stack(descriptor_list, axis=0),
                    columns=list(GLOBAL_DESCRIPTOR_COLUMNS),
                )
                x_cat = pd.concat([x_cat, descriptor_frame], axis=1)
                cat_cols = cat_cols + list(GLOBAL_DESCRIPTOR_COLUMNS)
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

            # 存入缓存
            self._cache_store(
                cache_key,
                {
                    "csv_path": str(csv_path),
                    "max_fp_bits": int(max_fp_bits),
                    "smiles": np.asarray(self.smiles, dtype=object).copy(),
                    "ph": np.asarray(self.ph, dtype=np.float32).copy(),
                    "logk_raw": np.asarray(self.logk_raw, dtype=np.float32).copy(),
                    "fingerprint": np.asarray(self.fingerprint, dtype=np.float32).copy(),
                    "category": np.asarray(self.category, dtype=np.float32).copy(),
                    "category_cols": list(self.category_cols),
                    "primary_category_dim": int(self.primary_category_dim),
                },
            )
        else:
            # 从缓存恢复
            self.csv_path = str(cache_item.get("csv_path", csv_path))
            self.max_fp_bits = int(cache_item.get("max_fp_bits", max_fp_bits))
            self.smiles = np.asarray(cache_item.get("smiles", []), dtype=object).copy()
            self.ph = np.asarray(cache_item["ph"], dtype=np.float32).copy()
            self.logk_raw = np.asarray(cache_item["logk_raw"], dtype=np.float32).copy()
            self.fingerprint = np.asarray(cache_item["fingerprint"], dtype=np.float32).copy()
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

        # 默认使用所有样本初始化缩放器；训练流程通常会调用 fit_scalers(train_idx) 以避免数据泄露
        self.fit_scalers(np.arange(self.ph.shape[0], dtype=int))

    def __len__(self):
        return int(self.logk_raw.shape[0])

    def fit_scalers(self, train_indices: np.ndarray):
        """
        仅使用训练集数据拟合缩放器（Scaler），避免数据泄露。
        
        参数:
            train_indices: 训练样本的索引数组
        """
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
        if _env_bool_pair("TRANSFORMER_V9_CATEGORY_KINETIC_PRIOR", None, False):
            self._fit_category_kinetic_prior(idx)
        self._fit_structural_clusters(idx)
        self.ph_scaled = self.ph_scaler.transform(self.ph).astype(np.float32)
        self.logk_scaled = self.logk_scaler.transform(self.logk_raw).astype(np.float32)

        if self.fp_scaler is not None:
            self.fp_scaler = StandardScaler()
            self.fp_scaler.fit(self.fingerprint[idx])
            self.fingerprint_scaled = self.fp_scaler.transform(self.fingerprint).astype(np.float32)
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
        scale = max(1e-6, scale)

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
            folds = KFold(n_splits=min(5, train_indices.size), shuffle=True, random_state=242)
            for fit_pos, hold_pos in folds.split(train_indices):
                fit_idx = train_indices[fit_pos]
                hold_idx = train_indices[hold_pos]
                prior_values[hold_idx] = estimate(fit_idx, hold_idx)
        self.category[:, prior_cols] = prior_values

    def __getitem__(self, idx):
        """
        获取单个样本。
        
        返回:
            包含指纹、数值特征、目标值等的字典。
        """
        fp = self.fingerprint_scaled[idx].astype(np.float32)
        ph = self.ph_scaled[idx].astype(np.float32)
        cat_vec = self.category[idx].astype(np.float32)
        num_feat = np.concatenate([ph.reshape(-1), cat_vec.reshape(-1)], axis=0).astype(np.float32)
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
    """
    特征选择后的数据集包装器。
    
    仅返回选定的top-k指纹特征，用于减少输入维度。
    """
    def __init__(self, base_ds: FingerprintReactionDataset, indices: np.ndarray, selected_col_idx: np.ndarray):
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
        # 仅选择特定列
        fp_sel = fp_full[self.selected_col_idx]
        ph = self.base.ph_scaled[base_idx].astype(np.float32)
        cat_vec = self.base.category[base_idx].astype(np.float32)
        num_feat = np.concatenate([ph.reshape(-1), cat_vec.reshape(-1)], axis=0).astype(np.float32)
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


class GatedFusion(nn.Module):
    """
    门控融合模块：动态融合分子特征和数值特征。
    
    使用 Sigmoid 门控机制学习两个特征模态的权重。
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.linear_mol = nn.Linear(d_model, d_model)
        self.linear_num = nn.Linear(d_model, d_model)
        # 门控网络：输入拼接后的特征，输出融合权重 z (0~1)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mol_feat: torch.Tensor, num_feat: torch.Tensor) -> torch.Tensor:
        h_mol = self.linear_mol(mol_feat)
        h_num = self.linear_num(num_feat)
        # 计算门控值 z
        z = self.gate(torch.cat([mol_feat, num_feat], dim=1))
        # 加权融合：z * 分子 + (1-z) * 数值
        fused = z * h_mol + (1.0 - z) * h_num
        return self.norm(self.dropout(fused))


class FingerprintTransformer(nn.Module):
    """
    基于MLP的基线模型（尽管名为Transformer，在此上下文中作为MLP基线存在）。
    注意：真正的Transformer模型在后续的 FingerprintTransformerV5 类中实现。
    """
    def __init__(self, fingerprint_dim: int, numeric_dim: int, config: Optional[FingerprintConfig] = None):
        super().__init__()
        self.config = config or FingerprintConfig
        self.fingerprint_dim = int(fingerprint_dim)
        self.numeric_dim = int(numeric_dim)
        self.d_model = int(self.config.d_model)
        self.num_input_dim = max(1, self.numeric_dim)
        act = nn.GELU() if self.config.activation == "gelu" else nn.ReLU()

        # 分子指纹特征投影
        self.fp_proj = nn.Sequential(nn.Linear(self.fingerprint_dim, self.d_model), act, nn.Dropout(self.config.dropout))
        # numeric 分支（pH + category）
        self.num_proj = nn.Sequential(nn.Linear(self.num_input_dim, self.d_model), act, nn.Dropout(self.config.dropout))
        # 特征融合
        self.fusion = GatedFusion(self.d_model, dropout=float(self.config.dropout))
        # 回归预测头
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model, max(8, self.d_model // 2)),
            nn.GELU() if self.config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(max(8, self.d_model // 2), 1),
        )

    def forward(self, fingerprint: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        if numeric.dim() == 1:
            numeric = numeric.unsqueeze(-1)
        fp_feat = self.fp_proj(fingerprint)
        num_feat = self.num_proj(_match_feature_dim(numeric, self.num_input_dim))
        fused = self.fusion(fp_feat, num_feat)
        return self.regressor(fused).squeeze(-1)


class _TransformerEncoderLayer(nn.Module):
    """
    标准的Transformer编码器层封装。
    包含自注意力机制和前馈网络。
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # 前馈网络 (Feed Forward)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.last_attn = None  # 用于存储注意力权重以供可视化

    def forward(self, x: torch.Tensor):
        # 自注意力计算
        attn_out, attn_w = self.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
        self.last_attn = attn_w
        # 残差连接 + 层归一化
        x = self.norm1(x + self.dropout(attn_out))
        # 前馈网络 + 残差连接 + 层归一化
        ff = self.linear2(self.dropout(self.act(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        return x


class AttentionFingerprintTransformer(nn.Module):
    """
    基于注意力的指纹Transformer模型 (旧版实现，作为参考)。
    V5版本主要使用 FingerprintTransformerV5 类。
    """
    def __init__(self, fingerprint_dim: int, numeric_dim: int, config: Optional[FingerprintConfig] = None):
        super().__init__()
        self.config = config or FingerprintConfig
        self.fingerprint_dim = int(fingerprint_dim)
        self.numeric_dim = int(numeric_dim)
        self.d_model = int(self.config.d_model)
        self.num_token_dim = max(1, self.numeric_dim)
        # 限制最大token数量
        self.max_fp_tokens = int(min(self.config.max_fp_tokens, self.fingerprint_dim))
        self.uses_sparse_tokens = self.max_fp_tokens < self.fingerprint_dim

        # 指纹值嵌入
        self.fp_embed = nn.Linear(1, self.d_model)
        # 指纹索引（位置）嵌入
        self.fp_index_embed = nn.Embedding(self.fingerprint_dim, self.d_model)
        self.fp_pos = nn.Parameter(torch.zeros(1, self.max_fp_tokens, self.d_model))
        
        self.num_embed = nn.Linear(1, self.d_model)
        # 编码器层堆叠
        self.encoder_layers = nn.ModuleList(
            [_TransformerEncoderLayer(self.d_model, int(self.config.n_heads), dropout=float(self.config.dropout)) for _ in range(int(self.config.n_layers))]
        )
        # 交叉注意力层（指纹与数值特征交互）
        self.cross_attn = nn.MultiheadAttention(
            self.d_model, int(self.config.n_heads), dropout=float(self.config.dropout), batch_first=True
        )
        self.cross_norm = nn.LayerNorm(self.d_model)
        
        # 回归预测头
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model, max(8, self.d_model // 2)),
            nn.GELU() if self.config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(max(8, self.d_model // 2), 1),
        )
        self.attn_cache = {"self": [], "cross": None}
        self.capture_attn = False
        self.last_token_indices = None

    def forward(self, fingerprint: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        if numeric.dim() == 1:
            numeric = numeric.unsqueeze(-1)
        numeric = _match_feature_dim(numeric, self.num_token_dim)
            
        # 稀疏Token处理：仅选取Top-K最显著的指纹位作为Token
        if self.uses_sparse_tokens:
            vals, idx = torch.topk(fingerprint, k=self.max_fp_tokens, dim=1)
            # Token Embedding = 索引嵌入 + 值嵌入 + 位置编码
            fp_tokens = self.fp_index_embed(idx) + self.fp_embed(vals.unsqueeze(-1)) + self.fp_pos
            self.last_token_indices = idx.detach()
        else:
            # 密集Token处理
            fp_tokens = self.fp_embed(fingerprint.unsqueeze(-1)) + self.fp_pos[:, : self.fingerprint_dim, :]
            dense_idx = torch.arange(self.fingerprint_dim, device=fingerprint.device).unsqueeze(0).expand(fingerprint.size(0), -1)
            self.last_token_indices = dense_idx.detach()
            
        num_tokens = self.num_embed(numeric.unsqueeze(-1))
        self.attn_cache["self"] = []
        
        # 通过Transformer编码器层
        for layer in self.encoder_layers:
            fp_tokens = layer(fp_tokens)
            if self.capture_attn and layer.last_attn is not None:
                self.attn_cache["self"].append(layer.last_attn.detach())
                
        # 交叉注意力：指纹特征关注数值特征
        cross_out, cross_w = self.cross_attn(fp_tokens, num_tokens, num_tokens, need_weights=True, average_attn_weights=False)
        if self.capture_attn:
            self.attn_cache["cross"] = cross_w.detach()
            
        fp_tokens = self.cross_norm(fp_tokens + cross_out)
        # 平均池化
        pooled = fp_tokens.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)


class FingerprintTrainer:
    """
    通用训练器类，封装了训练、评估循环。
    """
    def __init__(self, model: nn.Module, config: FingerprintConfig, device: Optional[str] = None):
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        # AdamW优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=float(self.config.learning_rate), weight_decay=float(self.config.weight_decay)
        )
        # 学习率调度器：Plateau
        if str(self.config.scheduler_type).lower() == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
            )
        else:
            self.scheduler = None
        # 混合精度训练支持
        self.use_amp = bool(str(self.device).lower().startswith("cuda"))
        self.amp_dtype = torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None

    def _get_base_dataset(self, dataloader: DataLoader):
        """辅助函数：从DataLoader中提取原始数据集对象以获取Scaler信息。"""
        ds = dataloader.dataset
        if isinstance(ds, SelectedFeatureSubset):
            return ds.base
        if isinstance(ds, Subset) and hasattr(ds, "dataset"):
            return ds.dataset
        return ds

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """
        评估模型性能。
        
        返回:
            (平均损失, R2分数, RMSE)
        """
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        preds_all = []
        y_all = []
        base_ds = self._get_base_dataset(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                fingerprint = batch["fingerprint"].to(self.device)
                numeric = batch["numeric"].to(self.device)
                y = batch["logk"].to(self.device)
                y_raw = batch["logk_raw"].numpy()
                
                preds_scaled = self.model(fingerprint, numeric)
                loss = self.criterion(preds_scaled, y)
                
                bs = int(fingerprint.size(0))
                total_loss += float(loss.item()) * bs
                n_samples += bs
                
                # 反归一化预测值，得到真实的logk
                preds_scaled_np = preds_scaled.detach().cpu().numpy().reshape(-1, 1)
                preds_raw = base_ds.logk_scaler.inverse_transform(preds_scaled_np).reshape(-1)
                preds_all.append(preds_raw)
                y_all.append(y_raw.reshape(-1))
                
        avg_loss = total_loss / max(1, n_samples)
        if preds_all:
            y_true = np.concatenate(y_all)
            y_pred = np.concatenate(preds_all)
            # 过滤NaN值
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            if y_true.size >= 2:
                r2 = float(r2_score(y_true, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            else:
                r2 = float("nan")
                rmse = float("nan")
        else:
            r2 = float("nan")
            rmse = float("nan")
        return float(avg_loss), float(r2), float(rmse)

def get_run_output_dir(prefix: Optional[str] = None) -> str:
    """获取当前运行的输出目录，如果不存在则创建。"""
    base = RUN_OUTPUT_DIR if RUN_OUTPUT_DIR else OUT_DIR
    os.makedirs(base, exist_ok=True)
    if prefix:
        path = os.path.join(base, str(prefix))
        os.makedirs(path, exist_ok=True)
        return path
    return base


def cleanup_attention_output_images(output_dir: str) -> None:
    """
    清理旧的输出图片文件，保留特定的关键分析图表。
    """
    if not output_dir or (not os.path.isdir(output_dir)):
        return
    # 保留文件名的前缀列表
    keep_prefixes = (
        "pred_vs_true",
        "residual_hist",
        "train_test_vs_true_band",
        "train_density_test_overlay",
        "category_metrics",
        "attn_cls_by_category",
        "attn_family_by_category",
        "attention_top_tokens",
        "feature_corr_network_top20",
        "attn_bit_substructures",
        "attn_top",
        "consensus_",
        "fusion_component_metrics",
        "model_performance_reference_style",
        "v3_",
        "v4_",
        "v5_",
    )
    keep_exts = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() not in keep_exts:
            continue
        if any(stem.startswith(pref) for pref in keep_prefixes):
            continue
        try:
            os.remove(path)
        except Exception:
            pass


def _unwrap_dataset(ds):
    """递归解包DataLoader中的数据集，获取最底层的原始数据集对象。"""
    cur = ds
    for _ in range(8):
        if isinstance(cur, SelectedFeatureSubset):
            return cur
        if isinstance(cur, Subset) and hasattr(cur, "dataset"):
            cur = cur.dataset
            continue
        return cur
    return cur


def _get_fp_feature_labels(ds) -> Optional[list]:
    """尝试获取指纹特征的名称标签（如fp_123）。"""
    cur = _unwrap_dataset(ds)
    if isinstance(cur, SelectedFeatureSubset) and getattr(cur, "feature_names", None):
        return list(cur.feature_names)
    if hasattr(cur, "fingerprint_dim"):
        try:
            n = int(getattr(cur, "fingerprint_dim"))
            if n > 0:
                return [f"fp_{i}" for i in range(n)]
        except Exception:
            return None
    return None


def _bin_matrix(mat: np.ndarray, max_rows: int, max_cols: int):
    """
    对大矩阵进行分箱（Binning）缩减，以便于热力图可视化。
    如果行/列超过最大限制，则合并相邻行/列取平均值。
    """
    mat = np.asarray(mat, dtype=float)
    n_rows, n_cols = mat.shape
    row_bins = [np.array([i]) for i in range(n_rows)]
    col_bins = [np.array([j]) for j in range(n_cols)]
    
    # 行缩减
    if n_rows > max_rows:
        row_bins = np.array_split(np.arange(n_rows), max_rows)
        mat = np.vstack([mat[idx].mean(axis=0) for idx in row_bins])
        
    # 列缩减
    if n_cols > max_cols:
        col_bins = np.array_split(np.arange(n_cols), max_cols)
        mat = np.column_stack([mat[:, idx].mean(axis=1) for idx in col_bins])
        
    return mat, row_bins, col_bins


def _make_bin_labels(bins):
    """生成分箱后的标签（例如 "1-5", "6-10"）。"""
    labels = []
    for idx in bins:
        start = int(idx[0]) + 1
        end = int(idx[-1]) + 1
        labels.append(f"{start}" if start == end else f"{start}-{end}")
    return labels


def _resolve_attention_model(model: nn.Module) -> Optional[nn.Module]:
    """
    解析并返回内部包含的注意力模型实例。
    兼容直接传入或包装在DataParallel/DistributedDataParallel中的模型。
    """
    if isinstance(model, (AttentionFingerprintTransformer, ChemistryAwareAttentionTransformer, ChemBiasAttentionRegressor)):
        return model
    # 检查是否作为 expert 包含在混合模型中
    attn_expert = getattr(model, "attn_expert", None)
    if isinstance(attn_expert, (AttentionFingerprintTransformer, ChemistryAwareAttentionTransformer, ChemBiasAttentionRegressor)):
        return attn_expert
    return None


def _capture_attention_once(model: nn.Module, test_loader: DataLoader):
    """
    执行一次前向传播以捕获注意力权重。
    需要设置 capture_attn 标志。
    """
    attn_model = _resolve_attention_model(model)
    if attn_model is None:
        return None, None
        
    device = next(attn_model.parameters()).device
    first_batch = None
    try:
        was_training = bool(attn_model.training)
        attn_model.capture_attn = True
        attn_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                fp = batch["fingerprint"].to(device)
                numeric = batch["numeric"].to(device)
                _ = attn_model(fp, numeric)
                first_batch = batch
                break
        # 恢复原始训练状态
        if was_training:
            attn_model.train()
    finally:
        attn_model.capture_attn = False
    return attn_model, first_batch


def _token_labels_from_cache(attn_model: nn.Module, token_count_no_cls: int, fp_labels: Optional[list]) -> list:
    """
    从缓存中获取Token对应的指纹位ID，用于可视化标签。
    对于稀疏Token模型，Token对应的是动态选择的Top-K指纹位。
    """
    labels = [f"token_{i}" for i in range(token_count_no_cls)]
    idx_tensor = getattr(attn_model, "last_token_indices", None)
    
    if idx_tensor is None:
        if fp_labels is not None and len(fp_labels) >= token_count_no_cls:
            return list(fp_labels[:token_count_no_cls])
        return labels
        
    try:
        idx_np = idx_tensor.detach().cpu().numpy()
        # 验证形状
        if idx_np.ndim != 2 or idx_np.shape[1] != token_count_no_cls:
            return labels
            
        out = []
        # 对每个位置的Token，取Batch中出现频率最高的指纹位ID作为代表
        for j in range(idx_np.shape[1]):
            col = idx_np[:, j].astype(int)
            if col.size == 0:
                out.append(labels[j])
                continue
            uniq, cnt = np.unique(col, return_counts=True)
            bit_id = int(uniq[int(np.argmax(cnt))])
            if fp_labels is not None and 0 <= bit_id < len(fp_labels):
                out.append(str(fp_labels[bit_id]))
            else:
                out.append(f"fp_{bit_id}")
        return out
    except Exception:
        return labels


def plot_pred_vs_true_and_residuals(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> None:
    """
    绘制预测值vs真实值的散点图（带有密度着色），以及残差直方图。
    包括边缘分布直方图。
    """
    out_dir = get_run_output_dir(prefix)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    
    # 过滤无效值
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 3:
        return

    x = y_true.copy()
    y = y_pred.copy()
    err = np.abs(y - x)

    # 计算高斯核密度估计（Gaussian KDE）用于点着色
    if gaussian_kde is not None and x.size > 5:
        try:
            xy = np.vstack([x, y])
            density = gaussian_kde(xy)(xy)
        except Exception:
            density = np.ones_like(x, dtype=float)
    else:
        density = np.ones_like(x, dtype=float)

    # 按密度排序，确保高密度点绘制在顶层
    order = np.argsort(density)
    x = x[order]
    y = y[order]
    err = err[order]
    density = density[order]

    # 颜色映射设置
    cmax = float(np.nanpercentile(err, 98)) if err.size > 0 else 1.0
    if not np.isfinite(cmax) or cmax <= 1e-12:
        cmax = float(np.nanmax(err) + 1e-9) if err.size > 0 else 1.0
    norm = mpl.colors.Normalize(vmin=0.0, vmax=cmax)
    cmap = mpl.cm.get_cmap("rainbow_r")
    rgba = cmap(norm(err))

    # 根据密度调整透明度/白度
    dmin = float(np.nanmin(density)) if density.size > 0 else 0.0
    dmax = float(np.nanmax(density)) if density.size > 0 else 1.0
    dscale = (density - dmin) / max(dmax - dmin, 1e-12)
    whiten = (1.0 - dscale) * 0.72  # 低密度点变亮（更白）
    rgba[:, :3] = rgba[:, :3] * (1.0 - whiten[:, None]) + 1.0 * whiten[:, None]
    rgba[:, 3] = 0.92

    # 坐标轴范围设置
    all_min = float(min(np.min(x), np.min(y)))
    all_max = float(max(np.max(x), np.max(y)))
    margin = 0.03 * (all_max - all_min) if all_max > all_min else 1.0
    lo = all_min - margin
    hi = all_max + margin
    bins = int(max(18, min(40, np.sqrt(x.size) * 2.2)))
    edges = np.linspace(lo, hi, bins + 1)
    bin_w = float(edges[1] - edges[0])
    grid = np.linspace(lo, hi, 500)

    # 创建带有边缘直方图的复杂布局
    fig = plt.figure(figsize=(8.8, 6.9), dpi=130)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[4.6, 1.3, 0.24],
        height_ratios=[1.15, 4.1],
        wspace=0.06,
        hspace=0.06,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cbar = fig.add_subplot(gs[:, 2])
    ax_blank = fig.add_subplot(gs[0, 1])
    ax_blank.axis("off")

    # 主散点图
    ax_main.scatter(x, y, s=26, c=rgba, edgecolors="none")
    ax_main.plot([lo, hi], [lo, hi], "--", color="red", linewidth=1.4, alpha=0.9)  # y=x 参考线
    ax_main.set_xlim(lo, hi)
    ax_main.set_ylim(lo, hi)
    ax_main.set_xlabel("True logk")
    ax_main.set_ylabel("Predicted logk")
    ax_main.grid(alpha=0.2)

    # 顶部边缘直方图 (True / Pred 在 True-logk 轴上的分布)
    ax_top.hist(y_true, bins=edges, color="#6BAED6", alpha=0.45, label="True")
    ax_top.hist(y_pred, bins=edges, color="#FDB37C", alpha=0.45, label="Pred")
    if gaussian_kde is not None and y_true.size > 5:
        try:
            ax_top.plot(grid, gaussian_kde(y_true)(grid) * y_true.size * bin_w, color="#1F77B4", linewidth=2.0)
            ax_top.plot(grid, gaussian_kde(y_pred)(grid) * y_pred.size * bin_w, color="#FF7F0E", linewidth=2.0)
        except Exception:
            pass
    ax_top.set_ylabel("Frequency")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.legend(loc="upper right", fontsize=9, frameon=True)
    ax_top.grid(alpha=0.15)

    # 右侧边缘直方图
    ax_right.hist(y_true, bins=edges, orientation="horizontal", color="#6BAED6", alpha=0.45)
    ax_right.hist(y_pred, bins=edges, orientation="horizontal", color="#FDB37C", alpha=0.45)
    if gaussian_kde is not None and y_true.size > 5:
        try:
            ax_right.plot(gaussian_kde(y_true)(grid) * y_true.size * bin_w, grid, color="#1F77B4", linewidth=2.0)
            ax_right.plot(gaussian_kde(y_pred)(grid) * y_pred.size * bin_w, grid, color="#FF7F0E", linewidth=2.0)
        except Exception:
            pass
    ax_right.set_xlabel("Frequency")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.grid(alpha=0.15)

    # 误差颜色条
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("Pred-True to y=x (hue); density controls lightness")

    fig.tight_layout()
    _savefig_with_pdf(fig, os.path.join(out_dir, "pred_vs_true.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 残差直方图
    residual = y_pred - y_true
    plt.figure(figsize=(6.0, 4.0))
    plt.hist(residual, bins=30, color="#0F766E", alpha=0.85)
    plt.xlabel("Residual (Pred - True)")
    plt.ylabel("Count")
    plt.title(f"{prefix} - Residual Histogram")
    plt.tight_layout()
    _savefig_with_pdf(plt, os.path.join(out_dir, "residual_hist.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_train_test_vs_true_with_band(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    prefix: str,
) -> None:
    """
    绘制训练集和测试集的对比图，包含误差带。
    左图为训练集，右图为测试集。
    """
    out_dir = get_run_output_dir(prefix)
    y_train_true = np.asarray(y_train_true, dtype=float).reshape(-1)
    y_train_pred = np.asarray(y_train_pred, dtype=float).reshape(-1)
    y_test_true = np.asarray(y_test_true, dtype=float).reshape(-1)
    y_test_pred = np.asarray(y_test_pred, dtype=float).reshape(-1)
    
    tr_mask = np.isfinite(y_train_true) & np.isfinite(y_train_pred)
    te_mask = np.isfinite(y_test_true) & np.isfinite(y_test_pred)
    y_train_true = y_train_true[tr_mask]
    y_train_pred = y_train_pred[tr_mask]
    y_test_true = y_test_true[te_mask]
    y_test_pred = y_test_pred[te_mask]
    if y_train_true.size < 2 or y_test_true.size < 2:
        return

    r2_train = float(r2_score(y_train_true, y_train_pred))
    rmse_train = float(np.sqrt(mean_squared_error(y_train_true, y_train_pred)))
    r2_test = float(r2_score(y_test_true, y_test_pred))
    rmse_test = float(np.sqrt(mean_squared_error(y_test_true, y_test_pred)))
    
    # 计算80%分位数的绝对误差作为误差带宽度
    band = float(np.quantile(np.abs(y_test_pred - y_test_true), 0.80)) if y_test_true.size > 0 else 0.0

    all_true = np.concatenate([y_train_true, y_test_true])
    all_pred = np.concatenate([y_train_pred, y_test_pred])
    min_v = float(min(all_true.min(), all_pred.min()))
    max_v = float(max(all_true.max(), all_pred.max()))
    margin = 0.05 * (max_v - min_v) if max_v > min_v else 1.0
    x_line = np.linspace(min_v - margin, max_v + margin, 200)
    y_line = x_line

    plt.figure(figsize=(7.5, 6.0))
    if band > 0:
        plt.fill_between(x_line, y_line - band, y_line + band, color="#F59E0B", alpha=0.16, label="~80% test band")
    plt.plot(x_line, y_line, "--", color="#64748B", linewidth=1.2, label="y=x")
    plt.scatter(y_train_true, y_train_pred, c="#1D4ED8", s=20, alpha=0.65, label="Train")
    plt.scatter(y_test_true, y_test_pred, c="#DC2626", s=24, alpha=0.8, marker="^", label="Test")
    plt.xlabel("True logk")
    plt.ylabel("Predicted logk")
    plt.title(f"{prefix} - Train/Test Predicted vs True")
    plt.legend(
        title=f"Train R2={r2_train:.3f}, RMSE={rmse_train:.3f}\nTest  R2={r2_test:.3f}, RMSE={rmse_test:.3f}",
        loc="lower right",
        fontsize=9,
    )
    plt.tight_layout()
    _savefig_with_pdf(plt, os.path.join(out_dir, "train_test_vs_true_band.png"), dpi=300, bbox_inches="tight")
    if RUN_OUTPUT_DIR and prefix:
        _savefig_with_pdf(plt, os.path.join(RUN_OUTPUT_DIR, f"{prefix}_train_test_vs_true_band.png"), dpi=300, bbox_inches="tight")
    plt.close()


def _rgba_by_error_density(
    err: np.ndarray,
    density: np.ndarray,
    cmap_name: str = "turbo",
    lighten_max: float = 0.50,
    alpha_min: float = 0.30,
    alpha_max: float = 0.95,
):
    err = np.asarray(err, dtype=float).reshape(-1)
    density = np.asarray(density, dtype=float).reshape(-1)
    if err.size == 0:
        return np.zeros((0, 4), dtype=float), mpl.colors.Normalize(vmin=0.0, vmax=1.0), mpl.cm.get_cmap(cmap_name)

    cmax = float(np.nanpercentile(err, 98)) if err.size > 0 else 1.0
    if not np.isfinite(cmax) or cmax <= 1e-12:
        cmax = float(np.nanmax(err) + 1e-9) if err.size > 0 else 1.0
    norm = mpl.colors.Normalize(vmin=0.0, vmax=cmax)
    cmap = mpl.cm.get_cmap(cmap_name)
    rgba = cmap(norm(err))

    if density.size != err.size:
        density = np.ones_like(err, dtype=float)
    dmin = float(np.nanmin(density)) if density.size > 0 else 0.0
    dmax = float(np.nanmax(density)) if density.size > 0 else 1.0
    dscale = (density - dmin) / max(dmax - dmin, 1e-12)

    whiten = (1.0 - dscale) * float(lighten_max)
    rgba[:, :3] = rgba[:, :3] * (1.0 - whiten[:, None]) + 1.0 * whiten[:, None]
    rgba[:, 3] = float(alpha_min) + dscale * float(max(alpha_max - alpha_min, 1e-6))
    rgba[:, 3] = np.clip(rgba[:, 3], 0.0, 1.0)
    return rgba, norm, cmap


def plot_train_density_test_overlay(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    prefix: str,
) -> None:
    out_dir = get_run_output_dir(prefix)
    y_train_true = np.asarray(y_train_true, dtype=float).reshape(-1)
    y_train_pred = np.asarray(y_train_pred, dtype=float).reshape(-1)
    y_test_true = np.asarray(y_test_true, dtype=float).reshape(-1)
    y_test_pred = np.asarray(y_test_pred, dtype=float).reshape(-1)

    tr_mask = np.isfinite(y_train_true) & np.isfinite(y_train_pred)
    te_mask = np.isfinite(y_test_true) & np.isfinite(y_test_pred)
    y_train_true = y_train_true[tr_mask]
    y_train_pred = y_train_pred[tr_mask]
    y_test_true = y_test_true[te_mask]
    y_test_pred = y_test_pred[te_mask]
    if y_train_true.size < 10 or y_test_true.size < 5:
        return

    err_test = np.abs(y_test_pred - y_test_true)
    if gaussian_kde is not None and y_test_true.size > 8:
        try:
            xy_test = np.vstack([y_test_true, y_test_pred])
            density_test = gaussian_kde(xy_test)(xy_test)
        except Exception:
            density_test = np.ones_like(y_test_true, dtype=float)
    else:
        density_test = np.ones_like(y_test_true, dtype=float)

    order = np.argsort(density_test)
    y_test_true = y_test_true[order]
    y_test_pred = y_test_pred[order]
    err_test = err_test[order]
    density_test = density_test[order]
    rgba_test, norm, cmap = _rgba_by_error_density(err_test, density_test)

    all_vals = np.concatenate([y_train_true, y_train_pred, y_test_true, y_test_pred]).astype(float)
    all_vals = all_vals[np.isfinite(all_vals)]
    lo = float(np.min(all_vals))
    hi = float(np.max(all_vals))
    margin = 0.03 * (hi - lo) if hi > lo else 1.0
    lo -= margin
    hi += margin

    fig = plt.figure(figsize=(10.8, 7.8), dpi=140)
    gs = fig.add_gridspec(2, 3, width_ratios=[5.8, 1.7, 0.34], height_ratios=[1.65, 5.2], wspace=0.08, hspace=0.07)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cbar = fig.add_subplot(gs[:, 2])

    ax_main.hexbin(y_train_true, y_train_pred, gridsize=34, cmap="Greys", mincnt=1, linewidths=0.0, alpha=0.78, zorder=1)
    xline = np.linspace(lo, hi, 256)
    ax_main.fill_between(xline, xline - 0.5, xline + 0.5, color="#e5e5e5", alpha=0.30, zorder=0)
    ax_main.scatter(y_test_true, y_test_pred, s=28, c=rgba_test, edgecolors="white", linewidths=0.22, zorder=3)
    ax_main.plot([lo, hi], [lo, hi], "--", color="#d62728", linewidth=1.6, alpha=0.95, zorder=2)
    ax_main.set_xlim(lo, hi)
    ax_main.set_ylim(lo, hi)
    ax_main.set_xlabel("True logk")
    ax_main.set_ylabel("Predicted logk")
    ax_main.grid(alpha=0.18)

    try:
        def _metrics(t, p):
            if t.size < 2: return float("nan"), float("nan")
            return float(r2_score(t, p)), float(np.sqrt(mean_squared_error(t, p)))

        train_r2, train_rmse = _metrics(y_train_true, y_train_pred)
        test_r2, test_rmse = _metrics(y_test_true, y_test_pred)
        train_mae = float(np.mean(np.abs(y_train_pred - y_train_true)))
        test_mae = float(np.mean(np.abs(y_test_pred - y_test_true)))
        text = (
            f"Train: n={y_train_true.size}, R²={train_r2:.3f}, RMSE={train_rmse:.3f}, MAE={train_mae:.3f}\n"
            f"Test: n={y_test_true.size}, R²={test_r2:.3f}, RMSE={test_rmse:.3f}, MAE={test_mae:.3f}"
        )
        ax_main.text(
            0.03,
            0.97,
            text,
            transform=ax_main.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#666666", alpha=0.92),
        )
    except Exception:
        pass

    bins = int(max(18, min(40, np.sqrt(y_train_true.size + y_test_true.size) * 1.9)))
    grid = np.linspace(lo, hi, 256)
    bin_w = (hi - lo) / max(bins, 1)
    curves = [
        (y_train_true, "#4C78A8", "Train true", "--"),
        (y_train_pred, "#F58518", "Train pred", "--"),
        (y_test_true, "#4C78A8", "Test true", "-"),
        (y_test_pred, "#F58518", "Test pred", "-"),
    ]
    hist_specs = [
        (y_train_true, "#4C78A8", 0.16),
        (y_train_pred, "#F58518", 0.16),
        (y_test_true, "#4C78A8", 0.24),
        (y_test_pred, "#F58518", 0.24),
    ]
    for arr, color, alpha in hist_specs:
        ax_top.hist(arr, bins=bins, color=color, alpha=alpha, edgecolor="none")
    for arr, color, label, style in curves:
        if gaussian_kde is not None and arr.size > 5:
            try:
                curve = gaussian_kde(arr)(grid) * arr.size * bin_w
                ax_top.plot(grid, curve, color=color, linewidth=1.8, linestyle=style, label=label)
                continue
            except Exception:
                pass
        ax_top.hist(arr, bins=bins, histtype="step", color=color, linewidth=1.4, linestyle=style, label=label)
    ax_top.set_ylabel("Count")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(alpha=0.14)
    ax_top.legend(loc="upper right", fontsize=8.5, frameon=True)

    for arr, color, alpha in hist_specs:
        ax_right.hist(arr, bins=bins, orientation="horizontal", color=color, alpha=alpha, edgecolor="none")
    for arr, color, _, style in curves:
        if gaussian_kde is not None and arr.size > 5:
            try:
                curve = gaussian_kde(arr)(grid) * arr.size * bin_w
                ax_right.plot(curve, grid, color=color, linewidth=1.8, linestyle=style)
                continue
            except Exception:
                pass
        ax_right.hist(arr, bins=bins, orientation="horizontal", histtype="step", color=color, linewidth=1.4, linestyle=style)
    ax_right.set_xlabel("Count")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.grid(alpha=0.14)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("Absolute prediction error on the test set\n(color and opacity modulated by local test density)")

    handles = [
        Patch(facecolor="#808080", alpha=0.55, label="Training-set density"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markeredgecolor="white", markersize=6.5, label="Test-set samples"),
        Line2D([0], [0], color="#d62728", linestyle="--", linewidth=1.6, label="Ideal agreement (y = x)"),
    ]
    ax_main.legend(handles=handles, loc="lower right", fontsize=8.8, frameon=True)

    ax_top.set_title("Training-distribution-aware predictive performance on the test set", fontsize=13, pad=8)
    fig.tight_layout()
    _savefig_with_pdf(fig, os.path.join(out_dir, "train_density_test_overlay.png"), dpi=300, bbox_inches="tight")
    _savefig_with_pdf(fig, os.path.join(out_dir, "train_test_vs_true_band.png"), dpi=300, bbox_inches="tight")
    if RUN_OUTPUT_DIR and prefix:
        _savefig_with_pdf(fig, os.path.join(RUN_OUTPUT_DIR, f"{prefix}_train_density_test_overlay.png"), dpi=300, bbox_inches="tight")
    _savefig_with_pdf(fig, os.path.join(RUN_OUTPUT_DIR, f"{prefix}_train_test_vs_true_band.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_category_metrics_paper(y_true: np.ndarray, y_pred: np.ndarray, y_cat_labels: np.ndarray, prefix: str) -> None:
    """
    绘制特定于类别的性能指标（R2和RMSE），生成用于论文发表的高质量图表。
    """
    out_dir = get_run_output_dir(prefix)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_cat_labels = np.asarray(y_cat_labels).reshape(-1)
    if y_true.size != y_pred.size or y_true.size != y_cat_labels.size:
        return
        
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y_cat_labels = y_cat_labels[mask]
    if y_true.size < 20:
        return

    # 设置每个类别的最小样本数阈值，默认为5
    min_n = int(max(4, int(_env_value("TRANSFORMER_CATEGORY_METRICS_MIN_N", None, 5))))
    rows = []
    
    # 遍历每个类别计算指标
    for c in np.unique(y_cat_labels):
        m = y_cat_labels == c
        n = int(np.sum(m))
        if n < min_n:
            continue
        yt = y_true[m]
        yp = y_pred[m]
        try:
            r2 = float(r2_score(yt, yp)) if yt.size >= 2 else float("nan")
        except Exception:
            r2 = float("nan")
        try:
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        except Exception:
            rmse = float("nan")
        rows.append({"category": _pretty_category_label(c), "n": n, "r2": r2, "rmse": rmse})
    if not rows:
        return

    # 按R2降序、RMSE升序排序
    rows.sort(key=lambda row: (float(row.get("r2", float("-inf"))), -float(row.get("rmse", float("inf")))), reverse=True)
    labels = [str(row["category"]) for row in rows]
    n_vals = np.asarray([int(row["n"]) for row in rows], dtype=int)
    r2_vals = np.asarray([float(row["r2"]) for row in rows], dtype=float)
    rmse_vals = np.asarray([float(row["rmse"]) for row in rows], dtype=float)
    y_pos = np.arange(len(rows), dtype=float)

    # 动态调整图形宽度
    fig_w = max(13.4, 10.4 + 0.50 * len(rows))
    fig, (ax_r2, ax_rmse) = plt.subplots(
        1,
        2,
        figsize=(fig_w, 6.9),
        sharey=True,
        gridspec_kw={"width_ratios": [1.02, 1.0], "wspace": 0.14},
    )
    fig.patch.set_facecolor("white")
    for ax in (ax_r2, ax_rmse):
        ax.set_facecolor("white")
        ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D1D5DB")
        ax.spines["bottom"].set_color("#D1D5DB")
        # 绘制交替背景色带
        for yi in y_pos:
            ax.axhspan(yi - 0.5, yi + 0.5, color="#F8FAFC" if int(yi) % 2 == 0 else "#FFFFFF", zorder=0)

    # 目标参考线
    r2_thr = float(_env_value("TRANSFORMER_CATEGORY_R2_TARGET", None, 0.80))
    rmse_thr = float(_env_value("TRANSFORMER_CATEGORY_RMSE_TARGET", None, 1.20))
    mean_r2 = float(np.nanmean(r2_vals))
    mean_rmse = float(np.nanmean(rmse_vals))

    # 绘制参考区域和均值线
    ax_r2.axvspan(r2_thr, max(1.0, float(np.nanmax(r2_vals) + 0.03)), color="#ECFDF5", alpha=0.9, zorder=0)
    ax_r2.axvline(r2_thr, color="#059669", linestyle="--", linewidth=1.6, alpha=0.95)
    ax_rmse.axvspan(0.0, rmse_thr, color="#EFF6FF", alpha=0.9, zorder=0)
    ax_rmse.axvline(rmse_thr, color="#2563EB", linestyle="--", linewidth=1.6, alpha=0.95)
    ax_r2.axvline(mean_r2, color="#94A3B8", linestyle=":", linewidth=1.4, alpha=0.95)
    ax_rmse.axvline(mean_rmse, color="#94A3B8", linestyle=":", linewidth=1.4, alpha=0.95)

    r2_colors = ["#0F766E" if v >= r2_thr else "#94A3B8" for v in r2_vals]
    rmse_colors = ["#2563EB" if v <= rmse_thr else "#DC2626" for v in rmse_vals]
    # 气泡大小与样本量n成正比
    size_scale = 22 + 1.6 * np.sqrt(np.clip(n_vals, 1, None))
    stem_r2_start = min(r2_thr - 0.12, float(np.nanmin(r2_vals) - 0.03))

    # 绘制棒棒糖图（Lollipop plot）
    for y, v, color in zip(y_pos, r2_vals, r2_colors):
        ax_r2.hlines(y, xmin=stem_r2_start, xmax=v, color="#CBD5E1", linewidth=2.6, zorder=1)
        ax_r2.scatter([v], [y], s=float(size_scale[int(y)] * 4.6), color=color, edgecolors="white", linewidths=1.1, zorder=3)
    for y, v, color in zip(y_pos, rmse_vals, rmse_colors):
        ax_rmse.hlines(y, xmin=0.0, xmax=v, color="#E5E7EB", linewidth=2.6, zorder=1)
        ax_rmse.scatter([v], [y], s=float(size_scale[int(y)] * 4.6), color=color, edgecolors="white", linewidths=1.1, zorder=3, marker="o")

    ax_r2.set_yticks(y_pos)
    ax_r2.set_yticklabels([f"{lab}  (n={n})" for lab, n in zip(labels, n_vals)], fontsize=10.8, fontweight="bold")
    ax_r2.invert_yaxis()
    ax_r2.set_xlabel(r"$R^2$", fontsize=13, fontweight="bold")
    ax_r2.set_title(r"Category-wise $R^2$", fontsize=15.5, fontweight="bold", pad=14)
    ax_rmse.set_xlabel("RMSE", fontsize=13, fontweight="bold")
    ax_rmse.set_title("Category-wise RMSE", fontsize=15.5, fontweight="bold", pad=14)
    ax_rmse.tick_params(axis="y", left=False, labelleft=False)

    r2_xmin = min(r2_thr - 0.12, float(np.nanmin(r2_vals) - 0.04))
    r2_xmax = max(1.0, float(np.nanmax(r2_vals) + 0.06))
    ax_r2.set_xlim(r2_xmin, r2_xmax)
    ax_rmse.set_xlim(0.0, max(rmse_thr + 0.1, float(np.nanmax(rmse_vals) + 0.10)))

    # 添加数值标签
    for y, v, n in zip(y_pos, r2_vals, n_vals):
        ax_r2.text(
            v + 0.008,
            y,
            f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=9.4,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.90),
        )
    for y, v in zip(y_pos, rmse_vals):
        ha = "left" if v <= (ax_rmse.get_xlim()[1] * 0.82) else "right"
        dx = 0.02 if ha == "left" else -0.02
        ax_rmse.text(
            v + dx,
            y,
            f"{v:.3f}",
            va="center",
            ha=ha,
            fontsize=9.4,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.90),
        )

    ax_r2.text(
        r2_thr,
        -0.92,
        f"target = {r2_thr:.2f}",
        ha="left",
        va="center",
        fontsize=9.2,
        color="#047857",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#ECFDF5", edgecolor="#A7F3D0", alpha=0.95),
    )
    ax_rmse.text(
        rmse_thr,
        -0.92,
        f"target = {rmse_thr:.2f}",
        ha="right",
        va="center",
        fontsize=9.2,
        color="#1D4ED8",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#EFF6FF", edgecolor="#BFDBFE", alpha=0.95),
    )
    ax_r2.text(mean_r2, len(rows) - 0.15, f"mean {mean_r2:.3f}", ha="center", va="center", fontsize=8.8, color="#64748B")
    ax_rmse.text(mean_rmse, len(rows) - 0.15, f"mean {mean_rmse:.3f}", ha="center", va="center", fontsize=8.8, color="#64748B")

    fig.suptitle("Performance across chemical categories", fontsize=19, fontweight="bold", y=0.988)
    fig.tight_layout(rect=[0, 0, 1, 0.972])
    _savefig_with_pdf(fig, os.path.join(out_dir, "category_metrics_paper.png"), dpi=340, bbox_inches="tight", facecolor="white")
    plt.close(fig)


class _FGDef:
    """定义功能基团（Functional Group）的辅助类，用于子结构匹配。"""
    def __init__(self, name: str, smarts: str, display: str, highlight_idx: list, color=(0.95, 0.80, 0.85)):
        self.name = name
        self.smarts = smarts
        self.display = display
        self.highlight_idx = list(highlight_idx)
        self.color = color
        self._pat = Chem.MolFromSmarts(smarts)

    @property
    def pattern(self):
        return self._pat


FG_DEFINITIONS = [
    _FGDef("Carboxylic acid", "[CX3](=O)[OX2H1]", "-COOH", [2], (0.98, 0.90, 0.80)),
    _FGDef("Sulfonic acid", "[SX4](=O)(=O)[OX2H1]", "-SO3H", [3], (0.98, 0.90, 0.95)),
    _FGDef("Phosphonic acid", "[PX4](=O)([OX2H1,O-])[OX2H1,O-]", "-PO(OH)2", [1], (0.98, 0.85, 0.35)),
    _FGDef("Ester", "[CX3](=O)[OX2][CX4]", "-COOR", [2], (0.95, 0.92, 0.80)),
    _FGDef("Amide", "[NX3][CX3](=O)[#6]", "-CONH-", [0, 1], (0.92, 0.92, 0.98)),
    _FGDef("Phenol", "c[OX2H]", "Ar-OH", [1], (0.85, 0.95, 0.85)),
    _FGDef("Alcohol", "[OX2H][CX4]", "-OH", [0], (0.85, 0.95, 0.95)),
    _FGDef("Nitro", "[N+](=O)[O-]", "-NO2", [0], (0.98, 0.90, 0.90)),
    _FGDef("Ketone", "[CX3](=O)[#6]", "-CO-", [0], (0.92, 0.92, 0.92)),
    _FGDef("Primary amine", "[NX3;H2;!$(NC=O)]", "-NH2", [0], (0.95, 0.90, 0.70)),
    _FGDef("Secondary amine", "[NX3;H1;!$(NC=O)]", "-NHR", [0], (0.90, 0.95, 0.80)),
    _FGDef("Tertiary amine", "[NX3;H0;!$(NC=O)]", "-NR2", [0], (0.85, 0.90, 0.98)),
    _FGDef("Ether", "[OD2]([#6])[#6]", "-O-", [0], (0.90, 0.95, 0.95)),
    _FGDef("Halogen", "[F,Cl,Br,I][#6]", "-X", [0], (0.90, 0.95, 0.90)),
    _FGDef("Thiol", "[SH][CX4]", "-SH", [0], (0.95, 0.95, 0.85)),
    _FGDef("Carboxylate", "[CX3](=O)[O-]", "-COO−", [2], (0.95, 0.90, 0.85)),
]

FG_REPRESENTATIVE_SMILES = {
    "Alcohol": "CCO",
    "Phenol": "c1ccccc(O)c1",
    "Ether": "COC",
    "Carboxylic acid": "CC(=O)O",
    "Carboxylate": "CC(=O)[O-]",
    "Ester": "CC(=O)OC",
    "Amide": "CC(=O)NC",
    "Ketone": "CC(=O)C",
    "Primary amine": "CN",
    "Secondary amine": "CNC",
    "Tertiary amine": "CN(C)C",
    "Nitro": "c1ccccc([N+](=O)[O-])c1",
    "Halogen": "CCCl",
    "Thiol": "CS",
    "Sulfonic acid": "CS(=O)(=O)O",
    "Phosphonic acid": "CP(=O)(O)O",
}


IONIZABLE_PHYSICS_RULES = [
    {"name": "Carboxylic acid", "smarts": "[CX3](=O)[OX2H1,O-]", "pka": 4.5, "sign": 1.0, "slope": 0.10},
    {"name": "Sulfonic acid", "smarts": "[SX4](=O)(=O)[OX2H1,O-]", "pka": 1.2, "sign": 1.0, "slope": 0.06},
    {"name": "Phosphonic acid", "smarts": "[PX4](=O)([OX2H1,O-])[OX2H1,O-]", "pka": 2.1, "sign": 1.0, "slope": 0.07},
    {"name": "Phenol", "smarts": "c[OX2H,O-]", "pka": 9.8, "sign": 1.0, "slope": 0.08},
    {"name": "Thiol", "smarts": "[SH,S-][CX4,c]", "pka": 9.5, "sign": 1.0, "slope": 0.06},
    {"name": "Amine", "smarts": "[NX3;!$(NC=O)]", "pka": 9.7, "sign": 1.0, "slope": 0.08},
    {"name": "Imidazole", "smarts": "n1cc[nH]c1", "pka": 7.0, "sign": 1.0, "slope": 0.07},
    {"name": "Pyridine", "smarts": "n1ccccc1", "pka": 5.2, "sign": 1.0, "slope": 0.06},
]

for _rule in IONIZABLE_PHYSICS_RULES:
    try:
        _rule["pattern"] = Chem.MolFromSmarts(str(_rule["smarts"]))
    except Exception:
        _rule["pattern"] = None


def _ionizable_physics_hits(smiles: str) -> list:
    """Return ionizable functional-group rules present in a molecule."""
    mol = None
    try:
        mol = Chem.MolFromSmiles(_clean_smiles_text(smiles))
    except Exception:
        mol = None
    if mol is None:
        return []
    hits = []
    for rule in IONIZABLE_PHYSICS_RULES:
        pat = rule.get("pattern")
        if pat is None:
            continue
        try:
            if mol.HasSubstructMatch(pat):
                hits.append(rule)
        except Exception:
            continue
    return hits

CHEMICAL_CATEGORY_NAME_MAP = {
    "0": "Alkane",
    "1": "Alcohol",
    "2": "Diol",
    "3": "Ether",
    "4": "Ketone",
    "5": "Aldehyde",
    "6": "Ester",
    "7": "Carboxyl",
    "8": "Dicarboxylic",
    "9": "Halogenated",
    "10": "Sulfide / Disulfide",
    "11": "Sulfoxide",
    "12": "Thiol",
    "13": "Nitrile",
    "14": "Nitro",
    "15": "Amide",
    "16": "Amine",
    "17": "Nitroso / Nitramine",
    "18": "Phosphorus",
    "19": "Cyclo",
    "20": "Alkene",
    "21": "Benzene",
    "22": "Pyridine",
    "23": "Furan",
    "24": "Urea",
    "25": "Imidazole",
    "26": "Triazine",
}

TRIVIAL_SUBSTRUCTURE_EXACT = {
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "CO",
    "CN",
    "CCN",
    "CC",
    "CCC",
    "cO",
    "cN",
    "cS",
    "C=CC",
    "C:C:N",
}


def _env_smiles_from_center(mol: Chem.Mol, center: int, rad: int) -> str:
    try:
        bonds = AllChem.FindAtomEnvironmentOfRadiusN(mol, int(rad), int(center))
        if not bonds:
            atom = mol.GetAtomWithIdx(int(center))
            em = Chem.EditableMol(Chem.Mol())
            _ = em.AddAtom(Chem.Atom(atom.GetAtomicNum()))
            submol = em.GetMol()
        else:
            submol = Chem.PathToSubmol(mol, bonds)
        return Chem.MolToSmiles(submol, canonical=True)
    except Exception:
        return ""


def _canonicalize_smiles_or_smarts(smi: str) -> str:
    if not isinstance(smi, str) or not smi:
        return smi
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmarts(smi)
        if mol is None:
            return smi
        return Chem.MolToSmiles(mol)
    except Exception:
        return smi


def _is_informative_substructure(smi: str, min_atoms: int = 2) -> bool:
    if not isinstance(smi, str) or not smi or str(smi).startswith("Unknown"):
        return False
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmarts(smi)
        if mol is None:
            return False
        return int(mol.GetNumAtoms()) >= int(min_atoms)
    except Exception:
        return False


def _detect_functional_group(sub_smiles: str) -> Optional[Dict[str, object]]:
    """
    检测给定的子结构SMILES是否匹配已知的官能团。
    """
    if not sub_smiles or str(sub_smiles).startswith("Unknown"):
        return None
    mol = None
    try:
        mol = Chem.MolFromSmiles(sub_smiles)
    except Exception:
        mol = None
    if mol is None:
        try:
            mol = Chem.MolFromSmarts(sub_smiles)
        except Exception:
            mol = None
    if mol is None:
        return None
    for fg in FG_DEFINITIONS:
        pat = fg.pattern
        if pat is None:
            continue
        try:
            if mol.HasSubstructMatch(pat):
                match = mol.GetSubstructMatches(pat, uniquify=True)
                if not match:
                    continue
                first = match[0]
                highlight_atoms = [first[i] for i in fg.highlight_idx if i < len(first)] if fg.highlight_idx else list(first)
                return {
                    "fg_name": fg.name,
                    "display": fg.display,
                    "highlight_atoms": highlight_atoms,
                    "color": fg.color,
                    "smarts": fg.smarts,
                }
        except Exception:
            continue
    return None


def _functional_group_def_by_name(name: str) -> Optional[_FGDef]:
    target = str(name).strip()
    if not target:
        return None
    for fg in FG_DEFINITIONS:
        if str(fg.name) == target:
            return fg
    return None


def _pretty_category_label(label: object) -> str:
    """美化化学类别标签，去除前缀并转换为可读名称。"""
    raw = str(label).strip()
    if raw.startswith("cat_"):
        raw = raw.replace("cat_", "", 1)
    mapped = CHEMICAL_CATEGORY_NAME_MAP.get(raw, raw)
    mapped = str(mapped).replace("_", " ").strip()
    return mapped


def _aggregate_functional_group_impacts(
    *,
    bit_rows: list,
    dataset: FingerprintReactionDataset,
    sample_indices: np.ndarray,
    top_n: int = 10,
) -> list:
    """
    聚合位（Bit）层面的影响，映射到功能基团层面。
    
    参数:
        bit_rows: 包含位分析结果的列表（通常来自前面的分析步骤）
        dataset: 数据集对象
        sample_indices: 样本索引
        top_n: 返回的Top-N基团数量
        
    返回:
        按重要性排序的功能基团统计信息列表。
    """
    if not bit_rows:
        return []
    sample_idx = np.asarray(sample_indices, dtype=int).reshape(-1)
    # 确保索引有效
    sample_idx = sample_idx[(sample_idx >= 0) & (sample_idx < len(dataset))]
    if sample_idx.size == 0:
        return []

    fp_np = np.asarray(getattr(dataset, "fingerprint", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    y_np = np.asarray(getattr(dataset, "logk_raw", np.zeros((0, 1), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if fp_np.ndim != 2 or fp_np.shape[0] == 0 or y_np.size == 0:
        return []

    group_map: Dict[str, Dict[str, object]] = {}
    
    # 遍历每个特征位的结果，聚合到功能基团
    for row in bit_rows:
        fg_name = str(row.get("functional_group", "")).strip()
        if not fg_name:
            continue
            
        item = group_map.setdefault(
            fg_name,
            {
                "functional_group": fg_name,
                "functional_display": str(row.get("functional_display", "")).strip(),
                "smarts": "",
                "bit_names": set(),
                "bits": set(),
                "substructures": Counter(),
                "functional_score_sum": 0.0,
                "attention_score_sum": 0.0,
                "ig_score_sum": 0.0,
                "perturb_score_sum": 0.0,
                "stability_score_sum": 0.0,
                "consensus_score_sum": 0.0,
                "support_count_sum": 0,
                "mol_count_sum": 0,
                "occ_count_sum": 0,
                "row_count": 0,
            },
        )
        
        # 填充基团定义信息
        fg_def = _functional_group_def_by_name(fg_name)
        if fg_def is not None and (not item.get("smarts")):
            item["smarts"] = str(fg_def.smarts)
        if str(row.get("functional_display", "")).strip() and (not item.get("functional_display")):
            item["functional_display"] = str(row.get("functional_display", "")).strip()
            
        bit_name = str(row.get("bit_name", "")).strip()
        bit_idx = _parse_fp_bit_index(bit_name)
        if bit_name:
            item["bit_names"].add(bit_name)
        if bit_idx is not None:
            item["bits"].add(int(bit_idx))
            
        # 记录具体的子结构SMILES
        sub = str(row.get("substructure_smiles", "")).strip()
        if sub and (not sub.startswith("Unknown")):
            item["substructures"][sub] += int(max(1, int(row.get("substructure_count", 0))))

        # 计算综合证据分数
        evidence_score = float(row.get("consensus_score", 0.0))
        if evidence_score <= 0.0:
            # 如果没有预先计算的一致性分数，则手动加权计算
            evidence_score = (
                0.35 * float(row.get("attention_score", 0.0))
                + 0.25 * float(row.get("ig_score", 0.0))
                + 0.25 * float(row.get("perturb_score", 0.0))
                + 0.15 * float(row.get("stability_score", 0.0))
            )
            
        # 累加各项分数
        item["functional_score_sum"] += float(max(0.0, evidence_score))
        item["attention_score_sum"] += float(row.get("attention_score", 0.0))
        item["ig_score_sum"] += float(row.get("ig_score", 0.0))
        item["perturb_score_sum"] += float(row.get("perturb_score", 0.0))
        item["stability_score_sum"] += float(row.get("stability_score", 0.0))
        item["consensus_score_sum"] += float(row.get("consensus_score", 0.0))
        item["support_count_sum"] += int(row.get("support_count", 0))
        item["mol_count_sum"] += int(row.get("mol_count", 0))
        item["occ_count_sum"] += int(row.get("occ_count", 0))
        item["row_count"] += 1

    rows = []
    # 计算每个聚合基团的统计指标（如Delta logk）
    for fg_name, item in group_map.items():
        bits = sorted(int(b) for b in item.get("bits", set()) if 0 <= int(b) < fp_np.shape[1])
        if not bits:
            continue
            
        # 确定哪些样本激活了该基团对应的任意一个位
        active_mask = (np.abs(fp_np[np.asarray(sample_idx, dtype=int)][:, bits]) > 1e-6).any(axis=1)
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if active_mask.size != sample_idx.size:
            continue
            
        active_y = y_np[sample_idx][active_mask]
        inactive_y = y_np[sample_idx][~active_mask]
        
        # 计算 logk 差异（基团存在与否对 logk 的影响）
        delta_logk = float(np.nanmean(active_y) - np.nanmean(inactive_y)) if active_y.size > 0 and inactive_y.size > 0 else 0.0
        active_mean = float(np.nanmean(active_y)) if active_y.size > 0 else float("nan")
        inactive_mean = float(np.nanmean(inactive_y)) if inactive_y.size > 0 else float("nan")
        support_n = int(active_mask.sum())
        direction = "positive" if delta_logk > 1e-6 else ("negative" if delta_logk < -1e-6 else "neutral")
        
        examples = [str(smi) for smi, _ in item.get("substructures", Counter()).most_common(3)]
        
        rows.append(
            {
                "functional_group": str(fg_name),
                "functional_display": str(item.get("functional_display", "")).strip(),
                "smarts": str(item.get("smarts", "")).strip(),
                "bit_count": int(len(item.get("bits", set()))),
                "bit_names": "; ".join(sorted(str(x) for x in item.get("bit_names", set()))),
                "evidence_rows": int(item.get("row_count", 0)),
                "support_molecules": int(support_n),
                "support_frac": float(support_n / max(sample_idx.size, 1)),
                "active_mean_logk": active_mean,
                "inactive_mean_logk": inactive_mean,
                "delta_logk": float(delta_logk),
                "direction": str(direction),
                "functional_score_sum": float(item.get("functional_score_sum", 0.0)),
                "attention_score_sum": float(item.get("attention_score_sum", 0.0)),
                "ig_score_sum": float(item.get("ig_score_sum", 0.0)),
                "perturb_score_sum": float(item.get("perturb_score_sum", 0.0)),
                "stability_score_sum": float(item.get("stability_score_sum", 0.0)),
                "consensus_score_sum": float(item.get("consensus_score_sum", 0.0)),
                "support_count_sum": int(item.get("support_count_sum", 0)),
                "mol_count_sum": int(item.get("mol_count_sum", 0)),
                "occ_count_sum": int(item.get("occ_count_sum", 0)),
                "example_substructures": " | ".join(examples),
            }
        )

    # 排序：优先按功能分数，其次按delta_logk绝对值
    rows.sort(
        key=lambda row: (
            float(row.get("functional_score_sum", 0.0)),
            abs(float(row.get("delta_logk", 0.0))),
            int(row.get("support_molecules", 0)),
            int(row.get("bit_count", 0)),
        ),
        reverse=True,
    )
    return rows[: int(max(1, top_n))]


def _plot_top_functional_groups(functional_rows: list, output_dir: str, artifact_tag: str = "attn", top_n: int = 10) -> None:
    """
    绘制Top-N功能基团的条形图，展示其重要性分数和对logk的影响（Delta logk）。
    """
    rows = list(functional_rows[: int(max(1, top_n))])
    if not rows:
        return

    labels = []
    score_vals = []
    delta_vals = []
    support_vals = []
    bit_counts = []
    
    for row in rows:
        fg_name = str(row.get("functional_group", "")).strip() or "Unassigned"
        fg_display = str(row.get("functional_display", "")).strip()
        labels.append(f"{fg_name} ({fg_display})" if fg_display else fg_name)
        score_vals.append(float(row.get("functional_score_sum", row.get("consensus_score_sum", 0.0))))
        delta_vals.append(float(row.get("delta_logk", 0.0)))
        support_vals.append(int(row.get("support_molecules", 0)))
        bit_counts.append(int(row.get("bit_count", 0)))

    y_pos = np.arange(len(rows))
    fig_h = max(7.0, min(12.5, 2.0 + 0.60 * len(rows)))
    
    # 双子图：左侧为重要性分数，右侧为Delta logk
    fig, (ax_score, ax_delta) = plt.subplots(
        ncols=2,
        figsize=(13.8, fig_h),
        sharey=True,
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.08},
    )
    fig.patch.set_facecolor("white")

    score_arr = np.asarray(score_vals, dtype=np.float32)
    delta_arr = np.asarray(delta_vals, dtype=np.float32)
    
    # 颜色映射
    score_norm = mpl.colors.Normalize(vmin=float(score_arr.min()), vmax=float(score_arr.max() + 1e-12))
    score_colors = [mpl.cm.get_cmap("YlOrRd")(score_norm(v)) for v in score_arr]
    delta_lim = float(max(np.max(np.abs(delta_arr)), 1e-6))
    delta_norm = mpl.colors.TwoSlopeNorm(vmin=-delta_lim, vcenter=0.0, vmax=delta_lim)
    delta_colors = [mpl.cm.get_cmap("coolwarm")(delta_norm(v)) for v in delta_arr]

    # 绘制左侧分数条形图
    ax_score.set_facecolor("white")
    ax_delta.set_facecolor("white")
    ax_score.barh(y_pos, score_arr, color=score_colors, edgecolor="#7C2D12", linewidth=0.7)
    ax_score.set_yticks(y_pos)
    ax_score.set_yticklabels(labels, fontsize=10)
    ax_score.invert_yaxis()
    ax_score.set_xlabel("Functional-group evidence score")
    ax_score.set_title(f"Top-{len(rows)} functional groups", fontsize=15, pad=10)
    ax_score.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.20)
    ax_score.set_axisbelow(True)
    ax_score.spines["top"].set_visible(False)
    ax_score.spines["right"].set_visible(False)
    x_max = float(max(score_vals)) if score_vals else 1.0
    ax_score.set_xlim(0.0, x_max * 1.24 if x_max > 0 else 1.0)

    ax_delta.axvline(0.0, color="#6B7280", linewidth=1.0, alpha=0.9)
    ax_delta.barh(y_pos, delta_arr, color=delta_colors, edgecolor="#374151", linewidth=0.7)
    ax_delta.set_xlabel("Δlogk (active − inactive)")
    ax_delta.set_title("Direction of association", fontsize=15, pad=10)
    ax_delta.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.20)
    ax_delta.set_axisbelow(True)
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)
    ax_delta.spines["left"].set_visible(False)
    ax_delta.tick_params(axis="y", left=False, labelleft=False)
    ax_delta.set_xlim(-delta_lim * 1.28, delta_lim * 1.28)

    # 在条形图上添加详细文本标签
    for y, score_v, delta_v, support_n, bit_n, row in zip(y_pos, score_arr, delta_arr, support_vals, bit_counts, rows):
        ax_score.text(
            score_v + max(0.004, x_max * 0.012),
            y,
            f"support={support_n} | bits={bit_n}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#4B5563",
        )
        arrow = "↑" if delta_v > 0 else ("↓" if delta_v < 0 else "→")
        ax_delta.text(
            delta_v + (0.02 * delta_lim if delta_v >= 0 else -0.02 * delta_lim),
            y,
            f"{arrow} {delta_v:+.3f}",
            va="center",
            ha="left" if delta_v >= 0 else "right",
            fontsize=8.8,
            color="#374151",
            fontweight="bold",
        )
        smarts = str(row.get("smarts", "")).strip()
        if smarts:
            ax_delta.text(
                ax_delta.get_xlim()[0] + 0.02 * (ax_delta.get_xlim()[1] - ax_delta.get_xlim()[0]),
                y + 0.29,
                smarts,
                va="center",
                ha="left",
                fontsize=7.5,
                color="#6B7280",
            )

    fig.suptitle("Functional-group level attribution for logk", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    file_top_n = int(max(1, top_n))
    _savefig_with_pdf(fig, os.path.join(output_dir, f"{artifact_tag}_top{file_top_n}_functional_groups.png"), dpi=340, bbox_inches="tight")
    plt.close(fig)


def _load_first_matching_csv(pattern: str) -> Optional[pd.DataFrame]:
    """尝试加载匹配glob模式的第一个CSV文件。"""
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    try:
        return pd.read_csv(matches[0])
    except Exception:
        return None


def _cleanup_legacy_functional_group_outputs(output_dir: str, keep_top_n: int = 10) -> None:
    """清理旧的功能基团分析输出文件，只保留当前Top-N的。"""
    if not output_dir or (not os.path.isdir(output_dir)):
        return
    patterns = [
        "consensus_top*_functional_groups.png",
        "consensus_top*_functional_groups.pdf",
        "consensus_top*_functional_groups_detailed.csv",
        "consensus_top*_functional_group_structures.png",
        "consensus_top*_functional_group_structures.pdf",
        "consensus_top*_functional_group_structures.csv",
        "attn_top*_functional_groups.png",
        "attn_top*_functional_groups.pdf",
        "attn_top*_functional_groups_detailed.csv",
    ]
    keep_tokens = {
        f"top{int(keep_top_n)}_functional_groups",
        f"top{int(keep_top_n)}_functional_group_structures",
    }
    for pattern in patterns:
        for path in glob.glob(os.path.join(output_dir, pattern)):
            name = os.path.basename(path)
            if any(token in name for token in keep_tokens):
                continue
            try:
                os.remove(path)
            except Exception:
                pass


def _collect_functional_group_structure_rows(output_dir: str, top_n: int = 10) -> list:
    """收集功能基团结构信息，用于后续生成结构图。"""
    rows = []
    seen = set()
    consensus_df = _load_first_matching_csv(os.path.join(output_dir, "consensus_top*_functional_groups_detailed.csv"))
    attn_df = _load_first_matching_csv(os.path.join(output_dir, "attn_top*_functional_groups_detailed.csv"))
    fallback_frames = [
        _load_first_matching_csv(os.path.join(output_dir, "consensus_top*_functional_groups.csv")),
        _load_first_matching_csv(os.path.join(output_dir, "attn_top*_functional_groups.csv")),
    ]

    def _append_from_df(df: Optional[pd.DataFrame], source_tag: str, sort_cols: list) -> None:
        nonlocal rows, seen
        if df is None or df.empty:
            return
        work = df.copy()
        for col in sort_cols:
            if col not in work.columns:
                work[col] = 0.0
        work = work.sort_values(sort_cols, ascending=False)
        for _, r in work.iterrows():
            fg_name = str(r.get("functional_group", "")).strip()
            if (not fg_name) or fg_name in seen:
                continue
            seen.add(fg_name)
            rows.append(
                {
                    "functional_group": fg_name,
                    "functional_display": str(r.get("functional_display", "")).strip(),
                    "smarts": str(r.get("smarts", "")).strip(),
                    "delta_logk": float(pd.to_numeric(r.get("delta_logk", 0.0), errors="coerce") if hasattr(pd, 'to_numeric') else r.get("delta_logk", 0.0)),
                    "support_molecules": int(pd.to_numeric(r.get("support_molecules", r.get("mol_count", 0)), errors="coerce") if hasattr(pd, 'to_numeric') else r.get("support_molecules", r.get("mol_count", 0))),
                    "bit_count": int(pd.to_numeric(r.get("bit_count", 0), errors="coerce") if hasattr(pd, 'to_numeric') else r.get("bit_count", 0)),
                    "functional_score_sum": float(pd.to_numeric(r.get("functional_score_sum", r.get("consensus_score_sum", r.get("count", 0.0))), errors="coerce") if hasattr(pd, 'to_numeric') else r.get("functional_score_sum", r.get("consensus_score_sum", r.get("count", 0.0)))),
                    "example_substructures": str(r.get("example_substructures", "")).strip(),
                    "source": source_tag,
                }
            )
            if len(rows) >= int(top_n):
                return

    _append_from_df(consensus_df, "consensus", ["functional_score_sum", "support_molecules"])
    if len(rows) < int(top_n):
        _append_from_df(attn_df, "attention", ["functional_score_sum", "support_molecules"])
    if len(rows) < int(top_n):
        for idx, df in enumerate(fallback_frames):
            _append_from_df(df, f"fallback_{idx+1}", ["count", "mol_count"])
            if len(rows) >= int(top_n):
                break
    return rows[: int(max(1, top_n))]


def _supplement_functional_group_rows(base_rows: list, output_dir: str, top_n: int = 10) -> list:
    """补充功能基团行数据，确保列表至少有Top-N个条目。"""
    rows = [dict(r) for r in list(base_rows or [])]
    seen = {str(r.get("functional_group", "")).strip() for r in rows if str(r.get("functional_group", "")).strip()}
    if len(rows) >= int(top_n):
        return rows[: int(top_n)]

    attn_df = _load_first_matching_csv(os.path.join(output_dir, "attn_top*_functional_groups_detailed.csv"))
    if attn_df is None or attn_df.empty:
        attn_df = _load_first_matching_csv(os.path.join(output_dir, "attn_top*_functional_groups.csv"))
    if attn_df is None or attn_df.empty:
        return rows[: int(top_n)]

    work = attn_df.copy()
    for col in ["functional_score_sum", "support_molecules", "delta_logk", "bit_count"]:
        if col not in work.columns:
            if col == "support_molecules":
                work[col] = pd.to_numeric(work.get("mol_count", 0), errors="coerce").fillna(0)
            else:
                work[col] = 0.0
    work = work.sort_values(["functional_score_sum", "support_molecules"], ascending=False)
    for _, r in work.iterrows():
        fg_name = str(r.get("functional_group", "")).strip()
        if (not fg_name) or fg_name in seen:
            continue
        seen.add(fg_name)
        rows.append(
            {
                "functional_group": fg_name,
                "functional_display": str(r.get("functional_display", "")).strip(),
                "smarts": str(r.get("smarts", "")).strip(),
                "delta_logk": float(pd.to_numeric(r.get("delta_logk", 0.0), errors="coerce")),
                "support_molecules": int(pd.to_numeric(r.get("support_molecules", r.get("mol_count", 0)), errors="coerce")),
                "bit_count": int(pd.to_numeric(r.get("bit_count", 0), errors="coerce")),
                "functional_score_sum": float(pd.to_numeric(r.get("functional_score_sum", r.get("count", 0.0)), errors="coerce")),
                "consensus_score_sum": float(pd.to_numeric(r.get("functional_score_sum", r.get("count", 0.0)), errors="coerce")),
                "example_substructures": str(r.get("example_substructures", "")).strip(),
                "source": "attention",
            }
        )
        if len(rows) >= int(top_n):
            break

    # 如果仍然不足 Top-N，使用预定义的模板补充
    if len(rows) < int(top_n):
        template_priority = [
            "Phenol",
            "Ester",
            "Thiol",
            "Carboxylate",
            "Sulfonic acid",
            "Phosphonic acid",
            "Secondary amine",
            "Ketone",
            "Alcohol",
            "Ether",
        ]
        for name in template_priority:
            if len(rows) >= int(top_n):
                break
            if name in seen:
                continue
            fg_def = _functional_group_def_by_name(name)
            rows.append(
                {
                    "functional_group": str(name),
                    "functional_display": str(getattr(fg_def, 'display', '')),
                    "smarts": str(getattr(fg_def, 'smarts', '')),
                    "delta_logk": 0.0,
                    "support_molecules": 0,
                    "bit_count": 0,
                    "functional_score_sum": 0.0,
                    "consensus_score_sum": 0.0,
                    "example_substructures": str(FG_REPRESENTATIVE_SMILES.get(name, '')),
                    "source": "template",
                }
            )
            seen.add(name)
    return rows[: int(top_n)]


def _pick_functional_group_structure(functional_group: str, example_substructures: str = "", smarts: str = ""):
    """
    选择用于可视化的最佳功能基团结构。
    
    返回:
        (RDKit Mol对象, 高亮原子索引列表, 显示用的SMILES/SMARTS字符串)
    """
    fg_name = str(functional_group).strip()
    fg_def = _functional_group_def_by_name(fg_name)
    pat = fg_def.pattern if fg_def is not None else None
    candidates = []
    
    # 优先使用代表性SMILES
    rep = str(FG_REPRESENTATIVE_SMILES.get(fg_name, "")).strip()
    if rep:
        candidates.append(rep)
        
    # 其次尝试从example_substructures中提取
    for token in str(example_substructures).split("|"):
        tok = str(token).strip()
        if tok:
            candidates.append(tok)
            
    # 尝试从候选列表中生成Mol对象
    for smi in candidates:
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            mol = None
        if mol is None:
            continue
        highlight = []
        if pat is not None:
            try:
                matches = mol.GetSubstructMatches(pat, uniquify=True)
            except Exception:
                matches = []
            if matches:
                highlight = list(matches[0])
        return mol, highlight, smi
        
    # 如果都失败了，尝试直接使用SMARTS
    if smarts:
        try:
            q = Chem.MolFromSmarts(smarts)
        except Exception:
            q = None
        if q is not None:
            return q, [], smarts
    return None, [], rep or smarts


def _plot_top_functional_group_structures(output_dir: str, top_n: int = 10) -> Optional[str]:
    """
    绘制Top-N功能基团的结构网格图。
    """
    rows = _collect_functional_group_structure_rows(output_dir=output_dir, top_n=top_n)
    rows = _supplement_functional_group_rows(rows, output_dir=output_dir, top_n=top_n)
    if not rows:
        return None

    n = len(rows)
    n_cols = 5
    n_rows = int(max(2, np.ceil(n / float(n_cols))))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(27.5, max(9.8, 4.9 * n_rows)))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    fig.patch.set_facecolor("white")

    for ax in axes.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_edgecolor("#E2E8F0")

    for idx, row in enumerate(rows):
        ax = axes.reshape(-1)[idx]
        fg_name = str(row.get("functional_group", "")).strip() or "Unknown"
        fg_display = str(row.get("functional_display", "")).strip()
        smarts = str(row.get("smarts", "")).strip()
        
        mol, highlight_atoms, shown_smi = _pick_functional_group_structure(
            functional_group=fg_name,
            example_substructures=str(row.get("example_substructures", "")),
            smarts=smarts,
        )
        
        # 绘制分子结构
        if mol is not None:
            try:
                img = Draw.MolToImage(
                    mol,
                    size=(900, 520),
                    highlightAtoms=list(highlight_atoms or []),
                    fitImage=True,
                )
                ax.imshow(np.asarray(img))
            except Exception:
                ax.text(0.5, 0.55, fg_name, ha="center", va="center", fontsize=18, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.55, fg_name, ha="center", va="center", fontsize=18, transform=ax.transAxes)

        delta = float(row.get("delta_logk", 0.0))
        support = int(row.get("support_molecules", 0))
        bit_count = int(row.get("bit_count", 0))
        source = str(row.get("source", "")).strip()
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        delta_color = "#C0392B" if delta > 0 else ("#2C5AA0" if delta < 0 else "#5B6472")

        title = f"{idx + 1}. {fg_name}"
        if fg_display:
            title += f" ({fg_display})"
        ax.text(0.02, 0.98, title, ha="left", va="top", fontsize=13.0, fontweight="bold", color="#1F2937", transform=ax.transAxes)
        ax.text(0.02, 0.08, f"{arrow} Δlogk {delta:+.3f}", ha="left", va="bottom", fontsize=10.6, fontweight="bold", color=delta_color, transform=ax.transAxes)
        ax.text(0.34, 0.08, f"support={support} | bits={bit_count}", ha="left", va="bottom", fontsize=9.8, color="#475569", transform=ax.transAxes)
        if shown_smi:
            ax.text(0.02, 0.02, shown_smi, ha="left", va="bottom", fontsize=8.3, color="#64748B", transform=ax.transAxes)
        if source:
            ax.text(0.98, 0.02, source, ha="right", va="bottom", fontsize=8.3, color="#94A3B8", transform=ax.transAxes)

    for idx in range(n, n_rows * n_cols):
        axes.reshape(-1)[idx].axis("off")

    fig.suptitle("Top functional-group structures associated with logk", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.982])
    file_top_n = int(max(1, top_n))
    out_path = os.path.join(output_dir, f"consensus_top{file_top_n}_functional_group_structures.png")
    _savefig_with_pdf(fig, out_path, dpi=420, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    try:
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, f"consensus_top{file_top_n}_functional_group_structures.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    except Exception:
        pass
    return out_path


def _substructure_quality_metrics(sub_smiles: str, min_heavy_atoms: int = 3) -> Dict[str, object]:
    """
    计算子结构的质量指标（有效性、原子数、官能团等）。
    用于评估一个子结构是否是一个"有意义"的化学片段。
    """
    canonical = _canonicalize_smiles_or_smarts(sub_smiles)
    metrics: Dict[str, object] = {
        "valid": False,
        "canonical": canonical,
        "heavy_atoms": 0,
        "hetero_non_halogen": 0,
        "halogen_atoms": 0,
        "carbon_atoms": 0,
        "aromatic_atoms": 0,
        "ring_count": 0,
        "unsat_bonds": 0,
        "disconnected": "." in str(canonical),
        "halogen_only": False,
        "all_aliphatic_carbon": False,
        "is_trivial": True,
        "fg": None,
    }
    if not isinstance(canonical, str) or not canonical or canonical.startswith("Unknown"):
        return metrics
    try:
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            mol = Chem.MolFromSmarts(canonical)
        if mol is None:
            return metrics
            
        atoms = list(mol.GetAtoms())
        heavy_atoms = int(mol.GetNumHeavyAtoms())
        halogen_atoms = int(sum(atom.GetAtomicNum() in {9, 17, 35, 53} for atom in atoms))
        carbon_atoms = int(sum(atom.GetAtomicNum() == 6 for atom in atoms))
        hetero_non_halogen = int(sum(atom.GetAtomicNum() not in {1, 6, 9, 17, 35, 53} for atom in atoms))
        aromatic_atoms = int(sum(int(atom.GetIsAromatic()) for atom in atoms))
        ring_info = mol.GetRingInfo()
        ring_count = int(ring_info.NumRings()) if ring_info is not None else 0
        unsat_bonds = int(sum((bond.GetBondTypeAsDouble() > 1.1) or bond.GetIsAromatic() for bond in mol.GetBonds()))
        
        # 尝试识别该子结构对应的功能基团
        fg = _detect_functional_group(canonical)
        
        halogen_only = heavy_atoms > 0 and heavy_atoms == halogen_atoms
        all_aliphatic_carbon = (
            heavy_atoms > 0
            and carbon_atoms == heavy_atoms
            and aromatic_atoms == 0
            and hetero_non_halogen == 0
            and halogen_atoms == 0
        )
        exact_trivial = canonical in TRIVIAL_SUBSTRUCTURE_EXACT
        
        # 判断子结构是否是"琐碎"的（如单个原子、太小、无特征等）
        is_trivial = bool(
            exact_trivial
            or metrics["disconnected"]
            or halogen_only
            or heavy_atoms < int(min_heavy_atoms)
            or (all_aliphatic_carbon and heavy_atoms <= 4 and unsat_bonds == 0 and ring_count == 0)
            or (
                fg is None
                and hetero_non_halogen == 0
                and halogen_atoms <= 1
                and heavy_atoms <= 2
                and aromatic_atoms == 0
                and unsat_bonds <= 1
            )
        )
        metrics.update(
            {
                "valid": True,
                "canonical": canonical,
                "heavy_atoms": heavy_atoms,
                "hetero_non_halogen": hetero_non_halogen,
                "halogen_atoms": halogen_atoms,
                "carbon_atoms": carbon_atoms,
                "aromatic_atoms": aromatic_atoms,
                "ring_count": ring_count,
                "unsat_bonds": unsat_bonds,
                "halogen_only": halogen_only,
                "all_aliphatic_carbon": all_aliphatic_carbon,
                "is_trivial": is_trivial,
                "fg": fg,
            }
        )
        return metrics
    except Exception:
        return metrics


def _score_substructure_candidate(
    sub_smiles: str,
    sub_count: int,
    sub_frac: float,
    mol_count: int,
    min_heavy_atoms: int = 3,
) -> Dict[str, object]:
    """
    对候选子结构进行评分，以选择最具代表性和化学意义的子结构。
    """
    quality = _substructure_quality_metrics(sub_smiles, min_heavy_atoms=min_heavy_atoms)
    fg = quality.get("fg")
    
    # 启发式评分规则
    score = 0.0
    score += 8.0 if fg is not None else 0.0  # 对应已知功能基团加分
    score += 1.35 * float(quality.get("heavy_atoms", 0))
    score += 1.25 * float(quality.get("hetero_non_halogen", 0))
    score += 0.55 * float(quality.get("ring_count", 0))
    score += 0.35 * float(quality.get("aromatic_atoms", 0))
    score += 0.60 * float(quality.get("unsat_bonds", 0))
    score += 3.20 * float(sub_frac)
    score += 0.45 * float(np.log1p(max(int(sub_count), 0)))
    score += 0.25 * float(np.log1p(max(int(mol_count), 0)))
    
    # 惩罚项
    if bool(quality.get("is_trivial", True)):
        score -= 9.0
    if bool(quality.get("halogen_only", False)):
        score -= 6.0
    if bool(quality.get("all_aliphatic_carbon", False)) and int(quality.get("heavy_atoms", 0)) <= 4:
        score -= 5.0
        
    return {
        "canonical_substructure": str(quality.get("canonical", sub_smiles)),
        "quality": quality,
        "functional_group": str(fg.get("fg_name", "")) if fg else "",
        "functional_display": str(fg.get("display", "")) if fg else "",
        "score": float(score),
        "is_reasonable": bool(quality.get("valid", False) and (not quality.get("is_trivial", True))),
    }


def _choose_best_substructure_for_bit(
    rank: int,
    bit_name: str,
    bit: int,
    info: Dict[str, object],
    min_heavy_atoms: int = 3,
) -> Optional[Dict[str, object]]:
    """为特定的指纹位选择最佳的代表性子结构。"""
    rows = list(info.get("substructures", []) or [])
    if not rows:
        return None
        
    candidates = []
    for row in rows:
        sub = str(row.get("substructure_smiles", "Unknown"))
        cnt = int(row.get("substructure_count", 0))
        frac = float(row.get("substructure_frac", 0.0))
        
        scored = _score_substructure_candidate(
            sub_smiles=sub,
            sub_count=cnt,
            sub_frac=frac,
            mol_count=int(info.get("mol_count", 0)),
            min_heavy_atoms=min_heavy_atoms,
        )
        
        candidate = {
            "rank": int(rank),
            "bit": int(bit),
            "bit_name": str(bit_name),
            "mol_count": int(info.get("mol_count", 0)),
            "occ_count": int(info.get("occ_count", 0)),
            "substructure_smiles": sub,
            "canonical_substructure": str(scored["canonical_substructure"]),
            "substructure_count": cnt,
            "substructure_frac": frac,
            "score": float(scored["score"]),
            "functional_group": str(scored["functional_group"]),
            "functional_display": str(scored["functional_display"]),
            "is_reasonable": bool(scored["is_reasonable"]),
            "quality": dict(scored["quality"]),
        }
        candidates.append(candidate)
        
    if not candidates:
        return None
        
    # 排序：优先合理的，其次按分数
    candidates.sort(
        key=lambda row: (
            int(bool(row.get("is_reasonable", False))),
            float(row.get("score", float("-inf"))),
            float(row.get("substructure_frac", 0.0)),
            int(row.get("substructure_count", 0)),
            int(row.get("quality", {}).get("heavy_atoms", 0)),
        ),
        reverse=True,
    )
    return candidates[0]


def _select_unique_substructure_candidates(candidates: list, top_n: int) -> list:
    """从候选列表中选择唯一的子结构，避免重复展示相似结构。"""
    ordered = sorted(
        list(candidates),
        key=lambda row: (
            int(row.get("rank", 10**9)),
            -float(row.get("score", float("-inf"))),
            -int(row.get("quality", {}).get("heavy_atoms", 0)),
            -int(row.get("substructure_count", 0)),
        ),
    )
    selected = []
    deferred = []
    seen_sub = set()
    seen_fg = set()
    
    for row in ordered:
        canonical = str(row.get("canonical_substructure", ""))
        if (not canonical) or canonical in seen_sub or (not bool(row.get("is_reasonable", False))):
            continue
        fg_name = str(row.get("functional_group", "")).strip()
        if fg_name and fg_name not in seen_fg:
            selected.append(row)
            seen_sub.add(canonical)
            seen_fg.add(fg_name)
            if len(selected) >= int(top_n):
                return selected
        else:
            deferred.append(row)
    for row in ordered:
        if len(selected) >= int(top_n):
            break
        canonical = str(row.get("canonical_substructure", ""))
        if (not canonical) or canonical in seen_sub or (not bool(row.get("is_reasonable", False))):
            continue
        selected.append(row)
        seen_sub.add(canonical)
    if len(selected) < int(top_n):
        for row in deferred:
            if len(selected) >= int(top_n):
                break
            canonical = str(row.get("canonical_substructure", ""))
            quality = dict(row.get("quality", {}))
            if (not canonical) or canonical in seen_sub or (not bool(quality.get("valid", False))):
                continue
            if bool(quality.get("is_trivial", True)):
                continue
            if bool(quality.get("halogen_only", False)):
                continue
            if int(quality.get("heavy_atoms", 0)) < 2:
                continue
            selected.append(row)
            seen_sub.add(canonical)
    return selected


def _parse_fp_bit_index(bit_name: str) -> Optional[int]:
    """从特征名称（如 'fp_123'）中解析位索引。"""
    if not isinstance(bit_name, str):
        return None
    s = str(bit_name).strip()
    if not s.startswith("fp_"):
        return None
    try:
        return int(s.split("_", 1)[1])
    except Exception:
        return None


def _map_bits_to_substructures_cached(
    smiles_list: np.ndarray,
    fp_size: int,
    radius: int,
    bits: list,
    cache_tag: str = "",
    sample_indices: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, object]]:
    """
    缓存版本的位-子结构映射函数。
    计算给定位（bits）在数据集中对应的最常见化学子结构。
    """
    smiles_arr = np.asarray(smiles_list, dtype=object).reshape(-1)
    bit_list = sorted({int(b) for b in bits if b is not None})
    idx_arr = np.asarray(sample_indices if sample_indices is not None else np.arange(smiles_arr.size), dtype=int).reshape(-1)
    idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < smiles_arr.size)]
    
    # 构建缓存键
    cache_key = (str(cache_tag), int(fp_size), int(radius), tuple(bit_list), tuple(idx_arr.tolist()))
    cached = BIT_SUBSTRUCTURE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    counts_map = {b: Counter() for b in bit_list}
    raw_counts_map = {b: Counter() for b in bit_list}
    mol_count_map = {b: 0 for b in bit_list}
    occ_count_map = {b: 0 for b in bit_list}
    mols_with = {b: [] for b in bit_list}

    # 遍历分子，提取指纹位对应的环境
    for idx in idx_arr.tolist():
        smi = str(smiles_arr[idx])
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            bit_info = {}
            _ = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(fp_size), bitInfo=bit_info)
            for b in bit_list:
                if b not in bit_info:
                    continue
                mol_count_map[b] += 1
                mols_with[b].append(mol)
                envs = bit_info[b]
                occ_count_map[b] += int(len(envs))
                for center, rad_local in envs:
                    # 提取原子环境作为子结构SMILES
                    frag = _env_smiles_from_center(mol, center, rad_local)
                    if frag:
                        can = _canonicalize_smiles_or_smarts(frag)
                        raw_counts_map[b][can] += 1
                        if _is_informative_substructure(can, min_atoms=int(_env_value("TRANSFORMER_ATTN_SUBSTRUCT_MIN_ATOMS", None, 2))):
                            counts_map[b][can] += 1
        except Exception:
            continue

    result: Dict[int, Dict[str, object]] = {}
    for b in bit_list:
        sub_rows = []
        if counts_map[b]:
            total = int(sum(counts_map[b].values()))
            for smi, cnt in counts_map[b].most_common():
                sub_rows.append(
                    {
                        "substructure_smiles": str(smi),
                        "substructure_count": int(cnt),
                        "substructure_frac": float(cnt / max(total, 1)),
                    }
                )
        else:
            # 如果没有找到常见子结构，尝试计算MCS（最大公共子结构）或回退策略
            rep = "Unknown"
            try:
                if len(mols_with[b]) >= 2:
                    mcs = rdFMCS.FindMCS(mols_with[b], timeout=5)
                    if mcs and getattr(mcs, "smartsString", ""):
                        rep = _canonicalize_smiles_or_smarts(str(mcs.smartsString))
            except Exception:
                pass
            if (not _is_informative_substructure(rep, min_atoms=int(_env_value("TRANSFORMER_ATTN_SUBSTRUCT_MIN_ATOMS", None, 2)))) and raw_counts_map[b]:
                for smi, _ in raw_counts_map[b].most_common():
                    if _is_informative_substructure(smi, min_atoms=int(_env_value("TRANSFORMER_ATTN_SUBSTRUCT_MIN_ATOMS", None, 2))):
                        rep = str(smi)
                        break
            if not _is_informative_substructure(rep, min_atoms=int(_env_value("TRANSFORMER_ATTN_SUBSTRUCT_MIN_ATOMS", None, 2))) and raw_counts_map[b]:
                rep = str(raw_counts_map[b].most_common(1)[0][0])
            sub_rows.append(
                {
                    "substructure_smiles": _canonicalize_smiles_or_smarts(rep),
                    "substructure_count": int(max(mol_count_map[b], 1)),
                    "substructure_frac": 1.0 if mol_count_map[b] > 0 else 0.0,
                }
            )
        result[int(b)] = {
            "mol_count": int(mol_count_map[b]),
            "occ_count": int(occ_count_map[b]),
            "substructures": sub_rows,
        }
    BIT_SUBSTRUCTURE_CACHE[cache_key] = result
    return result


def export_attention_substructure_artifacts(
    dataset: FingerprintReactionDataset,
    train_indices: np.ndarray,
    ranked_bit_names: list,
    prefix: str,
    artifact_tag: str = "attn",
    extra_bit_metrics: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    """
    导出注意力机制关注的子结构分析报告。
    包括生成CSV统计文件和可视化图表。
    """
    out_dir = get_run_output_dir(prefix)
    train_idx = np.asarray(train_indices, dtype=int).reshape(-1)
    train_idx = train_idx[(train_idx >= 0) & (train_idx < len(dataset))]
    if train_idx.size == 0 or not ranked_bit_names:
        return {}

    top_n = int(max(5, min(int(_env_value("TRANSFORMER_ATTN_SUBSTRUCT_TOPK", None, 30)), len(ranked_bit_names))))
    functional_top_n = int(max(5, min(int(_env_value("TRANSFORMER_TOP_FUNCTIONAL_GROUPS", None, 10)), top_n)))
    unique_top_n = int(max(5, min(int(_env_value("TRANSFORMER_ATTN_SUBSTRUCT_UNIQUE_TOPK", None, 10)), top_n)))
    polished_min_heavy = int(_env_value("TRANSFORMER_ATTN_SUBSTRUCT_UNIQUE_MIN_HEAVY", None, 4))

    bit_names = []
    bits = []
    seen_bits = set()
    for raw_name in ranked_bit_names:
        raw_name = str(raw_name)
        bit = _parse_fp_bit_index(raw_name)
        if bit is None or bit in seen_bits:
            continue
        seen_bits.add(bit)
        bit_names.append(raw_name)
        bits.append(int(bit))
        if len(bit_names) >= int(top_n):
            break
    if not bits:
        return {}

    sample_idx = np.arange(len(dataset), dtype=int) if _env_bool("TRANSFORMER_ATTN_SUBSTRUCT_USE_TRAINVAL", False) else train_idx
    parsed = _map_bits_to_substructures_cached(
        smiles_list=getattr(dataset, "smiles", np.array([], dtype=object)),
        fp_size=int(getattr(dataset, "max_fp_bits", dataset.fingerprint_dim)),
        radius=int(_env_value("TRANSFORMER_ATTN_MIN_RADIUS", None, 2)),
        bits=bits,
        cache_tag=str(getattr(dataset, "csv_path", DATA_CSV_PATH)),
        sample_indices=sample_idx,
    )
    if not parsed:
        return {}

    bit_rows = []
    sub_counter = Counter()
    fg_counter = Counter()
    fg_mol_counter = Counter()
    rep_candidates = []

    for rank, bit_name in enumerate(bit_names, start=1):
        bit = _parse_fp_bit_index(bit_name)
        if bit is None or bit not in parsed:
            continue
        info = parsed[bit]
        metric_info = dict((extra_bit_metrics or {}).get(str(bit_name), {}))
        for row in info["substructures"]:
            sub = str(row.get("substructure_smiles", "Unknown"))
            cnt = int(row.get("substructure_count", 0))
            frac = float(row.get("substructure_frac", 0.0))
            fg = _detect_functional_group(sub)
            row_payload = {
                "rank": int(rank),
                "bit": int(bit),
                "bit_name": str(bit_name),
                "mol_count": int(info.get("mol_count", 0)),
                "occ_count": int(info.get("occ_count", 0)),
                "substructure_smiles": sub,
                "substructure_count": cnt,
                "substructure_frac": frac,
                "functional_group": str((fg or {}).get("fg_name", "")),
                "functional_display": str((fg or {}).get("display", "")),
                "functional_smarts": str((fg or {}).get("smarts", "")),
            }
            for key in ("attention_score", "ig_score", "perturb_score", "stability_score", "support_count", "support_frac", "consensus_score"):
                if key in metric_info:
                    row_payload[key] = metric_info[key]
            bit_rows.append(row_payload)
            if sub and not sub.startswith("Unknown"):
                sub_counter[sub] += cnt
                if fg is not None:
                    fg_name = str(fg.get("fg_name", "Unknown"))
                    fg_counter[fg_name] += cnt
                    fg_mol_counter[fg_name] += int(info.get("mol_count", 0))
        rep = _choose_best_substructure_for_bit(
            rank=rank,
            bit_name=bit_name,
            bit=int(bit),
            info=info,
            min_heavy_atoms=polished_min_heavy,
        )
        if rep is not None:
            for key, value in metric_info.items():
                rep[key] = value
            rep_candidates.append(rep)

    if bit_rows:
        pd.DataFrame(bit_rows).to_csv(os.path.join(out_dir, f"{artifact_tag}_bit_substructures.csv"), index=False, encoding="utf-8-sig")
    if sub_counter:
        pd.DataFrame(
            [{"substructure_smiles": str(smi), "count": int(cnt)} for smi, cnt in sub_counter.most_common(top_n)]
        ).to_csv(os.path.join(out_dir, f"{artifact_tag}_top{top_n}_substructures.csv"), index=False, encoding="utf-8-sig")
    if fg_counter:
        fg_rows = []
        for fg_name, cnt in fg_counter.most_common():
            fg_def = next((x for x in FG_DEFINITIONS if x.name == fg_name), None)
            fg_rows.append(
                {
                    "functional_group": str(fg_name),
                    "count": int(cnt),
                    "mol_count": int(fg_mol_counter.get(fg_name, 0)),
                    "smarts": str(fg_def.smarts) if fg_def is not None else "",
                }
            )
        pd.DataFrame(fg_rows).to_csv(os.path.join(out_dir, f"{artifact_tag}_top{top_n}_functional_groups.csv"), index=False, encoding="utf-8-sig")
    
    # 聚合功能基团影响并绘图
    functional_rows = _aggregate_functional_group_impacts(
        bit_rows=bit_rows,
        dataset=dataset,
        sample_indices=sample_idx,
        top_n=functional_top_n,
    )
    if functional_rows:
        pd.DataFrame(functional_rows).to_csv(
            os.path.join(out_dir, f"{artifact_tag}_top{int(functional_top_n)}_functional_groups_detailed.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        _plot_top_functional_groups(functional_rows, out_dir, artifact_tag=artifact_tag, top_n=functional_top_n)
        _cleanup_legacy_functional_group_outputs(out_dir, keep_top_n=int(functional_top_n))
    
    # 选择并导出独特的原子团结构
    selected_candidates = _select_unique_substructure_candidates(rep_candidates, unique_top_n)
    export_unique_atom_groups = _env_bool("TRANSFORMER_EXPORT_UNIQUE_ATOM_GROUPS", False)
    if selected_candidates and export_unique_atom_groups:
        selected_rows = []
        grid_mols = []
        grid_legends = []
        grid_highlights = []
        for row in selected_candidates:
            canonical = str(row.get("canonical_substructure", ""))
            mol = Chem.MolFromSmiles(canonical)
            if mol is None:
                mol = Chem.MolFromSmarts(canonical)
            if mol is None:
                continue
            fg = row.get("quality", {}).get("fg")
            display_text = str(row.get("functional_display", "")).strip()
            selected_rows.append(
                {
                    "rank": int(row.get("rank", 0)),
                    "bit": int(row.get("bit", 0)),
                    "bit_name": str(row.get("bit_name", "")),
                    "substructure_smiles": str(row.get("substructure_smiles", "")),
                    "canonical_substructure": canonical,
                    "functional_group": str(row.get("functional_group", "")),
                    "functional_display": display_text,
                    "score": float(row.get("score", 0.0)),
                    "mol_count": int(row.get("mol_count", 0)),
                    "occ_count": int(row.get("occ_count", 0)),
                    "substructure_count": int(row.get("substructure_count", 0)),
                    "substructure_frac": float(row.get("substructure_frac", 0.0)),
                    "heavy_atoms": int(row.get("quality", {}).get("heavy_atoms", 0)),
                    "hetero_non_halogen": int(row.get("quality", {}).get("hetero_non_halogen", 0)),
                    "ring_count": int(row.get("quality", {}).get("ring_count", 0)),
                    "unsat_bonds": int(row.get("quality", {}).get("unsat_bonds", 0)),
                    "attention_score": float(row.get("attention_score", 0.0)),
                    "ig_score": float(row.get("ig_score", 0.0)),
                    "perturb_score": float(row.get("perturb_score", 0.0)),
                    "stability_score": float(row.get("stability_score", 0.0)),
                    "support_count": int(row.get("support_count", 0)),
                    "support_frac": float(row.get("support_frac", 0.0)),
                    "consensus_score": float(row.get("consensus_score", 0.0)),
                }
            )
            label_lines = [
                f"Rank {int(row.get('rank', 0))} | {str(row.get('bit_name', ''))}",
                canonical,
            ]
            if display_text:
                label_lines.append(display_text)
            grid_mols.append(mol)
            grid_legends.append("\n".join(label_lines))
            if isinstance(fg, dict):
                grid_highlights.append(list(fg.get("highlight_atoms", [])))
            else:
                grid_highlights.append([])

        if selected_rows:
            pd.DataFrame(selected_rows).to_csv(
                os.path.join(out_dir, f"{artifact_tag}_top{len(selected_rows)}_unique_atom_groups.csv"),
                index=False,
                encoding="utf-8-sig",
            )
        if grid_mols:
            try:
                img = Draw.MolsToGridImage(
                    grid_mols,
                    molsPerRow=2,
                    subImgSize=(360, 280),
                    legends=grid_legends,
                    highlightAtomLists=grid_highlights,
                    legendFontSize=18,
                    useSVG=False,
                )
                img.save(os.path.join(out_dir, f"{artifact_tag}_top{len(grid_mols)}_unique_atom_groups.png"))
            except Exception:
                pass
        return {"selected_rows": selected_rows, "bit_rows": bit_rows, "functional_rows": functional_rows}
    return {"selected_rows": [], "bit_rows": bit_rows, "functional_rows": functional_rows}


def plot_feature_corr_network_top20(test_loader: DataLoader, y_pred_loader: np.ndarray, prefix: str) -> None:
    out_dir = get_run_output_dir(prefix)
    y_pred = np.asarray(y_pred_loader, dtype=np.float32).reshape(-1)
    if y_pred.size < 10:
        return

    fp_blocks = []
    ph_blocks = []
    cat_blocks = []
    for batch in test_loader:
        fp_blocks.append(batch["fingerprint"].detach().cpu().numpy().astype(np.float32))
        ph_blocks.append(batch["pH"].detach().cpu().numpy().reshape(-1, 1).astype(np.float32))
        cat_blocks.append(batch["category"].detach().cpu().numpy().astype(np.float32))
    if not fp_blocks:
        return

    x_fp = np.vstack(fp_blocks)
    x_ph = np.vstack(ph_blocks) if ph_blocks else np.zeros((x_fp.shape[0], 0), dtype=np.float32)
    x_cat = np.vstack(cat_blocks) if cat_blocks else np.zeros((x_fp.shape[0], 0), dtype=np.float32)
    n = min(x_fp.shape[0], y_pred.size)
    if n < 10:
        return

    x_fp = x_fp[:n]
    x_ph = x_ph[:n]
    x_cat = x_cat[:n]
    y_pred = y_pred[:n]

    fp_labels = _get_fp_feature_labels(test_loader.dataset)
    if fp_labels is None or len(fp_labels) < x_fp.shape[1]:
        fp_labels = [f"fp_{i}" for i in range(x_fp.shape[1])]

    feat_blocks = []
    feat_names = []
    if x_ph.size > 0:
        feat_blocks.append(x_ph[:, :1])
        feat_names.append("pH")
    if x_cat.size > 0:
        feat_blocks.append(np.argmax(x_cat, axis=1).astype(np.float32).reshape(-1, 1))
        feat_names.append("category")
    feat_blocks.append(x_fp)
    feat_names.extend([str(x) for x in fp_labels[: x_fp.shape[1]]])
    x_all = np.concatenate(feat_blocks, axis=1).astype(np.float32)

    corr_with_pred = np.zeros(x_all.shape[1], dtype=np.float32)
    pvals_with_pred = np.ones(x_all.shape[1], dtype=np.float32)
    for j in range(x_all.shape[1]):
        col = x_all[:, j]
        if np.std(col) <= 1e-9:
            continue
        try:
            if pearsonr is not None:
                r, p = pearsonr(col, y_pred)
            else:
                r = float(np.corrcoef(col, y_pred)[0, 1])
                p = 1.0
        except Exception:
            r, p = 0.0, 1.0
        corr_with_pred[j] = 0.0 if not np.isfinite(r) else float(r)
        pvals_with_pred[j] = 1.0 if not np.isfinite(p) else float(p)

    selected_idx = []
    for name in ("pH", "category"):
        if name in feat_names:
            selected_idx.append(int(feat_names.index(name)))
    for idx in np.argsort(np.abs(corr_with_pred))[::-1].tolist():
        idx = int(idx)
        if idx in selected_idx:
            continue
        selected_idx.append(idx)
        if len(selected_idx) >= 20:
            break
    selected_idx = selected_idx[:20]
    if len(selected_idx) < 3:
        return

    top_names = [str(feat_names[i]) for i in selected_idx]
    x_top = x_all[:, selected_idx]
    node_corr = corr_with_pred[selected_idx]
    node_pvals = pvals_with_pred[selected_idx]

    vmax = float(max(0.2, np.nanmax(np.abs(node_corr))))
    node_sizes = 300 + 2200 * np.clip(np.abs(node_corr), 0, 1)
    node_edge_colors = ["#222222" if float(p) < 0.05 else "#9a9a9a" for p in node_pvals]
    node_edge_widths = [2.2 if float(p) < 0.05 else 1.0 for p in node_pvals]
    cmap_nodes = mpl.colors.LinearSegmentedColormap.from_list("target_corr_bright", ["#3B82F6", "#FFFFFF", "#EF4444"], N=256)
    norm_nodes = mpl.colors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap_edges = mpl.colors.LinearSegmentedColormap.from_list("feat_corr_bright", ["#EC4899", "#FFFFFF", "#22C55E"], N=256)
    norm_edges = mpl.colors.TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)

    n_nodes = len(top_names)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 1.0
    coords = {i: (radius * np.cos(a), radius * np.sin(a)) for i, a in enumerate(angles)}

    fig, ax = plt.subplots(figsize=(12.8, 12.8))
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    lim = radius + 0.32
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    edge_candidates = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            xi = x_top[:, i]
            xj = x_top[:, j]
            if np.std(xi) <= 1e-9 or np.std(xj) <= 1e-9:
                continue
            try:
                if pearsonr is not None:
                    r, p = pearsonr(xi, xj)
                else:
                    r = float(np.corrcoef(xi, xj)[0, 1])
                    p = 1.0
            except Exception:
                continue
            if np.isfinite(r):
                edge_candidates.append((i, j, float(r), 1.0 if not np.isfinite(p) else float(p)))
    edge_candidates.sort(key=lambda item: abs(item[2]), reverse=True)
    for i, j, r, p in edge_candidates[: min(40, len(edge_candidates))]:
        if abs(r) < 0.20 and p >= 0.05:
            continue
        xi, yi = coords[i]
        xj, yj = coords[j]
        width = 2.6 + 7.2 * min(abs(r), 1.0)
        alpha = 0.15 + 0.60 * min(abs(r), 1.0)
        color = cmap_edges(norm_edges(r))
        linestyle = "--" if p >= 0.05 else "-"
        line_alpha = alpha * (0.35 if p >= 0.05 else 1.0)
        ax.plot([xi, xj], [yi, yj], color=color, linewidth=width, alpha=line_alpha, linestyle=linestyle, solid_capstyle="round", zorder=1)

    xs = [coords[i][0] for i in range(n_nodes)]
    ys = [coords[i][1] for i in range(n_nodes)]
    ax.scatter(
        xs,
        ys,
        c=node_corr,
        cmap=cmap_nodes,
        norm=norm_nodes,
        s=node_sizes,
        edgecolors=node_edge_colors,
        linewidths=node_edge_widths,
        zorder=3,
    )

    import matplotlib.patheffects as pe

    label_fontsize = 9.2 if n_nodes <= 20 else 8.2
    for i, name in enumerate(top_names):
        x, y = coords[i]
        angle = float(angles[i])
        dx = float(np.cos(angle))
        dy = float(np.sin(angle))
        node_r_pts = float(np.sqrt(max(float(node_sizes[i]), 1.0) / np.pi))
        offset_pts = node_r_pts + 6.0
        ha = "center" if abs(dx) < 0.20 else ("left" if dx > 0 else "right")
        va = "center" if abs(dy) < 0.20 else ("bottom" if dy > 0 else "top")
        ax.annotate(
            str(name),
            xy=(x, y),
            xytext=(offset_pts * dx, offset_pts * dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=label_fontsize,
            fontfamily="Arial",
            fontweight="bold",
            color="#111111",
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
            zorder=4,
            annotation_clip=False,
        )

    handles = [
        Line2D([0], [0], color=cmap_edges(norm_edges(0.35)), lw=3.6, label="Positive corr"),
        Line2D([0], [0], color=cmap_edges(norm_edges(-0.35)), lw=3.6, label="Negative corr"),
        Line2D([0], [0], color="gray", lw=2.0, linestyle="--", label="Non-significant edge"),
        plt.scatter([], [], s=120, facecolors="none", edgecolors="k", linewidths=2.2, label="Node significant"),
        plt.scatter([], [], s=120, facecolors="none", edgecolors="#777777", linewidths=1.0, label="Node non-significant"),
    ]
    legend_top = 0.18
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, legend_top), frameon=False, fontsize=8, ncol=3, handlelength=2.2, handletextpad=0.55, columnspacing=1.8, labelspacing=0.7)
    fig.subplots_adjust(bottom=legend_top, right=0.80, top=0.93, left=0.06)
    ax_pos = ax.get_position()
    cbar_width = 0.024
    cbar_x = ax_pos.x1 + 0.02
    sm_edges = plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm_edges)
    sm_edges.set_array([])
    cax_edges = fig.add_axes([cbar_x, ax_pos.y0 + 0.60 * ax_pos.height, cbar_width, 0.32 * ax_pos.height])
    cbar_edges = fig.colorbar(sm_edges, cax=cax_edges)
    cbar_edges.ax.set_title("Feature-Feature Corr", fontsize=8, pad=8)
    cbar_edges.ax.tick_params(labelsize=7, pad=2)
    sm_nodes = plt.cm.ScalarMappable(cmap=cmap_nodes, norm=norm_nodes)
    sm_nodes.set_array([])
    cax_nodes = fig.add_axes([cbar_x, ax_pos.y0 + 0.22 * ax_pos.height, cbar_width, 0.32 * ax_pos.height])
    cbar_nodes = fig.colorbar(sm_nodes, cax=cax_nodes)
    cbar_nodes.ax.set_title("Feature-Target Corr", fontsize=8, pad=8)
    cbar_nodes.ax.tick_params(labelsize=7, pad=2)
    fig.suptitle("feature_corr_network_top20", fontsize=12, y=0.97)
    _savefig_with_pdf(fig, os.path.join(out_dir, "feature_corr_network_top20.png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    if RUN_OUTPUT_DIR and prefix:
        _savefig_with_pdf(fig, os.path.join(RUN_OUTPUT_DIR, f"{prefix}_feature_corr_network_top20.png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _collect_category_bit_attention_stats(
    model: nn.Module,
    test_loader: DataLoader,
) -> Optional[Dict[str, object]]:
    attn_model = _resolve_attention_model(model)
    if attn_model is None or not hasattr(attn_model, "attn_cache"):
        return None

    base_ds = _unwrap_dataset(test_loader.dataset)
    cat_cols = list(getattr(base_ds, "category_cols", []) or [])
    fp_labels = _get_fp_feature_labels(test_loader.dataset)
    if fp_labels is None:
        return None

    global_sum: Dict[str, float] = {}
    global_cnt: Dict[str, int] = {}
    cat_sum: Dict[str, Dict[str, float]] = {}
    cat_cnt: Dict[str, Dict[str, int]] = {}
    cat_n: Dict[str, int] = {}

    was_training = bool(attn_model.training)
    device = next(attn_model.parameters()).device
    try:
        attn_model.capture_attn = True
        attn_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                fp = batch["fingerprint"].to(device)
                numeric = batch["numeric"].to(device)
                category = batch["category"].detach().cpu().numpy()
                _ = attn_model(fp, numeric)
                self_list = attn_model.attn_cache.get("self", [])
                if not self_list:
                    continue
                last_attn = self_list[-1].detach().cpu()
                if last_attn.ndim != 4:
                    continue
                last_attn = last_attn.mean(dim=1)
                if last_attn.shape[-1] <= 1:
                    continue
                cls_scores = last_attn[:, 0, 1:].numpy()
                token_idx = getattr(attn_model, "last_token_indices", None)
                if token_idx is None:
                    continue
                token_idx = token_idx.detach().cpu().numpy()
                if token_idx.ndim != 2 or token_idx.shape != cls_scores.shape:
                    continue
                cat_ids = np.argmax(category, axis=1).astype(int) if category.ndim == 2 and category.shape[1] > 0 else np.zeros(cls_scores.shape[0], dtype=int)
                for row in range(cls_scores.shape[0]):
                    c_int = int(cat_ids[row]) if row < cat_ids.size else 0
                    c_label = str(cat_cols[c_int]).replace("cat_", "") if 0 <= c_int < len(cat_cols) else str(c_int)
                    cat_n[c_label] = int(cat_n.get(c_label, 0) + 1)
                    cat_sum.setdefault(c_label, {})
                    cat_cnt.setdefault(c_label, {})
                    for bit_id, score in zip(token_idx[row].tolist(), cls_scores[row].tolist()):
                        bit_id = int(bit_id)
                        bit_name = str(fp_labels[bit_id]) if 0 <= bit_id < len(fp_labels) else f"fp_{bit_id}"
                        val = float(score)
                        global_sum[bit_name] = float(global_sum.get(bit_name, 0.0) + val)
                        global_cnt[bit_name] = int(global_cnt.get(bit_name, 0) + 1)
                        cat_sum[c_label][bit_name] = float(cat_sum[c_label].get(bit_name, 0.0) + val)
                        cat_cnt[c_label][bit_name] = int(cat_cnt[c_label].get(bit_name, 0) + 1)
    finally:
        attn_model.capture_attn = False
        if was_training:
            attn_model.train()

    if not global_sum or not cat_n:
        return None
    return {
        "global_sum": global_sum,
        "global_cnt": global_cnt,
        "cat_sum": cat_sum,
        "cat_cnt": cat_cnt,
        "cat_n": cat_n,
        "fp_labels": fp_labels,
        "cat_cols": cat_cols,
    }


def plot_attn_cls_by_category(
    model: nn.Module,
    test_loader: DataLoader,
    prefix: str,
    focus_bits: Optional[list] = None,
) -> None:
    out_dir = get_run_output_dir(prefix)
    stats = _collect_category_bit_attention_stats(model, test_loader)
    if not isinstance(stats, dict):
        return
    global_sum = dict(stats.get("global_sum", {}))
    global_cnt = dict(stats.get("global_cnt", {}))
    cat_sum = dict(stats.get("cat_sum", {}))
    cat_cnt = dict(stats.get("cat_cnt", {}))
    cat_n = dict(stats.get("cat_n", {}))
    heatmap_topn = int(max(5, int(_env_value("TRANSFORMER_ATTN_CAT_HEATMAP_TOPN", None, 20))))
    min_cat_samples = int(max(1, int(_env_value("TRANSFORMER_ATTN_CAT_HEATMAP_MIN_CATEGORY", None, 8))))
    min_cell_support = int(max(1, int(_env_value("TRANSFORMER_ATTN_CAT_HEATMAP_MIN_SUPPORT", None, 3))))
    row_norm_mode = str(_env_value("TRANSFORMER_ATTN_CAT_HEATMAP_ROW_NORM", None, "max")).strip().lower()
    min_cross_categories = int(max(1, int(_env_value("TRANSFORMER_ATTN_CAT_HEATMAP_MIN_CATS", None, 3))))
    if row_norm_mode not in {"max", "sum"}:
        row_norm_mode = "max"

    if focus_bits:
        top_bits = []
        seen = set()
        for name in [str(x) for x in focus_bits]:
            if name in seen:
                continue
            seen.add(name)
            top_bits.append(name)
            if len(top_bits) >= heatmap_topn:
                break
    else:
        top_bits = sorted(
            global_sum.keys(),
            key=lambda name: global_sum[name] / max(global_cnt.get(name, 1), 1),
            reverse=True,
        )[:heatmap_topn]
    original_top_bits = list(top_bits)
    global_ranked_bits = sorted(
        global_sum.keys(),
        key=lambda name: global_sum[name] / max(global_cnt.get(name, 1), 1),
        reverse=True,
    )
    supported_top_bits = []
    for bit_name in top_bits:
        support_cats = 0
        for cat_name in cat_n.keys():
            if int(cat_cnt.get(cat_name, {}).get(bit_name, 0)) >= min_cell_support:
                support_cats += 1
        if support_cats >= min_cross_categories:
            supported_top_bits.append(bit_name)
    top_bits = supported_top_bits[:heatmap_topn]
    if len(top_bits) < heatmap_topn:
        seen_bits = set(top_bits)
        for bit_name in list(original_top_bits) + list(global_ranked_bits):
            if bit_name in seen_bits:
                continue
            top_bits.append(bit_name)
            seen_bits.add(bit_name)
            if len(top_bits) >= heatmap_topn:
                break
    cat_order = sorted(
        [name for name in cat_n.keys() if int(cat_n.get(name, 0)) >= min_cat_samples],
        key=lambda name: (-int(cat_n[name]), str(name)),
    )
    if not top_bits or not cat_order:
        return

    mat = np.zeros((len(cat_order), len(top_bits)), dtype=np.float32)
    support_mat = np.zeros((len(cat_order), len(top_bits)), dtype=np.float32)
    for r, cat_name in enumerate(cat_order):
        for c, bit_name in enumerate(top_bits):
            s = float(cat_sum.get(cat_name, {}).get(bit_name, 0.0))
            n = int(cat_cnt.get(cat_name, {}).get(bit_name, 0))
            mat[r, c] = float(s / max(n, 1)) if n > 0 else 0.0
            support_mat[r, c] = float(n)

    if row_norm_mode == "sum":
        row_denom = np.sum(mat, axis=1, keepdims=True)
    else:
        row_denom = np.max(mat, axis=1, keepdims=True)
    row_denom = np.where(row_denom > 1e-12, row_denom, 1.0)
    mat_norm = mat / row_denom
    mask = support_mat < float(min_cell_support)
    mat_plot = mat_norm.copy()
    mat_plot[mask] = np.nan

    fig_w = max(8.4, min(13.5, 1.8 + 0.58 * len(top_bits)))
    fig_h = max(6.8, min(12.8, 1.8 + 0.72 * len(cat_order)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = mpl.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="#D1D5DB")
    im = ax.imshow(mat_plot, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized avg CLS attention", fontsize=11)
    ax.set_xticks(np.arange(len(top_bits)))
    ax.set_xticklabels(top_bits, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(cat_order)))
    ax.set_yticklabels([f"{cat_name}\n(n={int(cat_n[cat_name])})" for cat_name in cat_order], fontsize=10)
    ax.set_xlabel("Fingerprint bit", fontsize=12)
    ax.set_ylabel("Category", fontsize=12)
    title_topn = int(max(len(top_bits), min(heatmap_topn, len(original_top_bits))))
    ax.set_title(f"{prefix} - CLS Attention by Category (Consensus-focused Top-{title_topn})", fontsize=16, pad=14)
    fig.tight_layout()
    _savefig_with_pdf(fig, os.path.join(out_dir, "attn_cls_by_category.png"), dpi=320, bbox_inches="tight")
    if RUN_OUTPUT_DIR and prefix:
        _savefig_with_pdf(fig, os.path.join(RUN_OUTPUT_DIR, f"{prefix}_attn_cls_by_category.png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_attn_family_by_category(
    model: nn.Module,
    test_loader: DataLoader,
    prefix: str,
    bit_to_family: Dict[str, str],
) -> None:
    out_dir = get_run_output_dir(prefix)
    stats = _collect_category_bit_attention_stats(model, test_loader)
    if not isinstance(stats, dict) or not isinstance(bit_to_family, dict) or not bit_to_family:
        return
    cat_sum = dict(stats.get("cat_sum", {}))
    cat_cnt = dict(stats.get("cat_cnt", {}))
    cat_n = dict(stats.get("cat_n", {}))
    min_cat_samples = int(max(1, int(_env_value("TRANSFORMER_ATTN_CAT_HEATMAP_MIN_CATEGORY", None, 8))))
    min_cell_support = int(max(1, int(_env_value("TRANSFORMER_ATTN_CAT_HEATMAP_MIN_SUPPORT", None, 3))))
    min_family_support = int(max(1, int(_env_value("TRANSFORMER_ATTN_FAMILY_HEATMAP_MIN_SUPPORT", None, 2))))
    min_family_cats = int(max(1, int(_env_value("TRANSFORMER_ATTN_FAMILY_HEATMAP_MIN_CATS", None, 2))))
    family_topn = int(max(4, int(_env_value("TRANSFORMER_ATTN_FAMILY_HEATMAP_TOPN", None, 10))))
    row_norm_mode = str(_env_value("TRANSFORMER_ATTN_CAT_HEATMAP_ROW_NORM", None, "max")).strip().lower()
    if row_norm_mode not in {"max", "sum"}:
        row_norm_mode = "max"

    family_total = Counter()
    family_support = Counter()
    family_cat_support = Counter()
    for cat_name, bit_map in cat_sum.items():
        cnt_map = cat_cnt.get(cat_name, {})
        seen_family_in_cat = set()
        for bit_name, val_sum in bit_map.items():
            fam = str(bit_to_family.get(str(bit_name), "")).strip()
            if not fam:
                continue
            n = int(cnt_map.get(bit_name, 0))
            if n < min_cell_support:
                continue
            family_total[fam] += float(val_sum / max(n, 1))
            family_support[fam] += 1
            seen_family_in_cat.add(fam)
        for fam in seen_family_in_cat:
            family_cat_support[fam] += 1
    family_names = [
        fam for fam, _ in family_total.most_common()
        if int(family_support.get(fam, 0)) >= min_family_support and int(family_cat_support.get(fam, 0)) >= min_family_cats
    ][:family_topn]
    cat_order = sorted(
        [name for name in cat_n.keys() if int(cat_n.get(name, 0)) >= min_cat_samples],
        key=lambda name: (-int(cat_n[name]), str(name)),
    )
    if not family_names or not cat_order:
        return

    mat = np.zeros((len(cat_order), len(family_names)), dtype=np.float32)
    support_mat = np.zeros((len(cat_order), len(family_names)), dtype=np.float32)
    for r, cat_name in enumerate(cat_order):
        cnt_map = cat_cnt.get(cat_name, {})
        for c, fam in enumerate(family_names):
            fam_vals = []
            fam_support = 0
            for bit_name, family_name in bit_to_family.items():
                if str(family_name).strip() != fam:
                    continue
                n = int(cnt_map.get(str(bit_name), 0))
                if n < min_cell_support:
                    continue
                s = float(cat_sum.get(cat_name, {}).get(str(bit_name), 0.0))
                fam_vals.append(float(s / max(n, 1)))
                fam_support += 1
            if fam_vals:
                mat[r, c] = float(np.mean(fam_vals))
                support_mat[r, c] = float(fam_support)

    if row_norm_mode == "sum":
        row_denom = np.sum(mat, axis=1, keepdims=True)
    else:
        row_denom = np.max(mat, axis=1, keepdims=True)
    row_denom = np.where(row_denom > 1e-12, row_denom, 1.0)
    mat_norm = mat / row_denom
    mat_norm[support_mat < 1.0] = np.nan

    fig_w = max(7.6, min(12.8, 2.2 + 0.75 * len(family_names)))
    fig_h = max(6.2, min(11.5, 1.8 + 0.72 * len(cat_order)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = mpl.cm.get_cmap("magma").copy()
    cmap.set_bad(color="#E5E7EB")
    im = ax.imshow(mat_norm, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized avg family attention", fontsize=11)
    ax.set_xticks(np.arange(len(family_names)))
    ax.set_xticklabels(family_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(cat_order)))
    ax.set_yticklabels([f"{cat_name}\n(n={int(cat_n[cat_name])})" for cat_name in cat_order], fontsize=10)
    ax.set_xlabel("Motif family", fontsize=12)
    ax.set_ylabel("Category", fontsize=12)
    ax.set_title(f"{prefix} - Motif Family Attention by Category", fontsize=16, pad=14)
    fig.tight_layout()
    _savefig_with_pdf(fig, os.path.join(out_dir, "attn_family_by_category.png"), dpi=320, bbox_inches="tight")
    if RUN_OUTPUT_DIR and prefix:
        _savefig_with_pdf(fig, os.path.join(RUN_OUTPUT_DIR, f"{prefix}_attn_family_by_category.png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_attention_cls_and_category(model: nn.Module, test_loader: DataLoader, prefix: str):
    out_dir = get_run_output_dir(prefix)
    attn_model, _ = _capture_attention_once(model, test_loader)
    if attn_model is None or not hasattr(attn_model, "attn_cache"):
        return np.array([], dtype=int), [], {}, np.array([], dtype=int)
    try:
        self_list = attn_model.attn_cache.get("self", [])
        if not self_list:
            return np.array([], dtype=int), [], {}, np.array([], dtype=int)
        attn = self_list[-1].mean(dim=1).mean(dim=0).detach().cpu().numpy()  # [L, L]

        has_cls = isinstance(attn_model, ChemistryAwareAttentionTransformer)
        if has_cls and attn.shape[0] > 1:
            cls_row = np.asarray(attn[0, 1:], dtype=float)
        else:
            cls_row = np.asarray(attn.mean(axis=0), dtype=float)

        rank_idx = np.argsort(cls_row)[::-1] if cls_row.size > 0 else np.array([], dtype=int)
        topn = int(
            max(
                5,
                min(
                    20,
                    int(_env_value("TRANSFORMER_ATTN_TOPN", None, 20)),
                    rank_idx.size if rank_idx.size else 5,
                ),
            )
        )
        top_idx = rank_idx[:topn]
        fp_labels = _get_fp_feature_labels(test_loader.dataset)
        token_labels = _token_labels_from_cache(attn_model, cls_row.size, fp_labels)
        bit_map = {int(i): token_labels[i] for i in range(len(token_labels))}
        if top_idx.size > 0:
            vals = np.asarray(cls_row[top_idx], dtype=float)
            labels = [bit_map.get(int(i), f"token_{int(i)}") for i in top_idx]

            fig_h = max(6.4, min(10.8, 1.6 + top_idx.size * 0.38))
            fig, ax = plt.subplots(figsize=(6.0, fig_h))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("#FCFCFD")

            nature_fill = "#B9DCFF"
            nature_edge = "#6FA8DC"

            y_pos = np.arange(top_idx.size)
            ax.barh(y_pos, vals, color=nature_fill, edgecolor=nature_edge, linewidth=0.9)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis()

            ax.set_xlabel("CLS attention", fontsize=11)
            ax.set_ylabel("Fingerprint token", fontsize=11)
            ax.set_title(f"Top-{top_idx.size} attention tokens", fontsize=15, pad=10)

            ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.35)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            max_label_len = max((len(str(t)) for t in labels), default=8)
            left_margin = min(0.44, max(0.24, 0.16 + max_label_len * 0.006))
            fig.subplots_adjust(left=left_margin, right=0.88)

            x_max = float(np.max(vals))
            ax.set_xlim(0.0, x_max * 1.16 if x_max > 0 else 1.0)

            if top_idx.size <= 20:
                for y, v in zip(y_pos, vals):
                    ax.text(v + max(0.0015, x_max * 0.008), y, f"{v:.3f}", va="center", ha="left", fontsize=8, color="#374151")

            fig.tight_layout()
            _savefig_with_pdf(fig, os.path.join(out_dir, "attention_top_tokens.png"), dpi=320, bbox_inches="tight")
            plt.close(fig)
        feat_names = token_labels
        return top_idx.astype(int), feat_names, bit_map, rank_idx.astype(int)
    except Exception:
        return np.array([], dtype=int), [], {}, np.array([], dtype=int)


def _minmax_scale_array(arr: np.ndarray) -> np.ndarray:
    """将数组归一化到 [0, 1] 范围。"""
    x = np.asarray(arr, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    mask = np.isfinite(x)
    if not mask.any():
        return np.zeros_like(x, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    vals = x[mask]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if abs(vmax - vmin) < 1e-12:
        out[mask] = 1.0 if vmax > 0 else 0.0
        return out
    out[mask] = (vals - vmin) / (vmax - vmin)
    return out


def _row_normalize_nonnegative(mat: np.ndarray) -> np.ndarray:
    """对矩阵进行行归一化（除以行和），确保非负。"""
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    denom = arr.sum(axis=1, keepdims=True)
    denom = np.where(denom > 1e-12, denom, 1.0)
    return arr / denom


def _collect_explain_arrays(explain_loader: DataLoader, max_samples: Optional[int] = None) -> Dict[str, object]:
    """
    收集用于解释性分析的数据数组（指纹、数值特征、索引等）。
    """
    feature_labels = _get_fp_feature_labels(explain_loader.dataset) or []
    fp_blocks = []
    num_blocks = []
    idx_blocks = []
    total = 0
    fallback_idx = 0
    cap = None if max_samples is None else int(max(1, max_samples))

    for batch in explain_loader:
        fp_np = batch["fingerprint"].detach().cpu().numpy().astype(np.float32, copy=False)
        num_np = batch["numeric"].detach().cpu().numpy().astype(np.float32, copy=False)
        base_idx = batch.get("base_idx")
        if isinstance(base_idx, torch.Tensor):
            idx_np = base_idx.detach().cpu().numpy().astype(int, copy=False).reshape(-1)
        elif base_idx is None:
            idx_np = np.arange(fallback_idx, fallback_idx + fp_np.shape[0], dtype=int)
        else:
            idx_np = np.asarray(base_idx, dtype=int).reshape(-1)
        fallback_idx += int(fp_np.shape[0])

        if cap is not None and total + fp_np.shape[0] > cap:
            keep = max(0, cap - total)
            if keep <= 0:
                break
            fp_np = fp_np[:keep]
            num_np = num_np[:keep]
            idx_np = idx_np[:keep]
        if fp_np.size == 0:
            continue
        fp_blocks.append(fp_np)
        num_blocks.append(num_np)
        idx_blocks.append(idx_np)
        total += int(fp_np.shape[0])
        if cap is not None and total >= cap:
            break

    if not fp_blocks:
        return {
            "fingerprint": np.zeros((0, len(feature_labels)), dtype=np.float32),
            "numeric": np.zeros((0, 0), dtype=np.float32),
            "sample_idx": np.asarray([], dtype=int),
            "feature_labels": feature_labels,
        }

    fingerprint = np.concatenate(fp_blocks, axis=0).astype(np.float32, copy=False)
    numeric = np.concatenate(num_blocks, axis=0).astype(np.float32, copy=False)
    sample_idx = np.concatenate(idx_blocks, axis=0).astype(int, copy=False)
    if not feature_labels:
        feature_labels = [f"fp_{i}" for i in range(int(fingerprint.shape[1]))]
    return {
        "fingerprint": fingerprint,
        "numeric": numeric,
        "sample_idx": sample_idx,
        "feature_labels": list(feature_labels),
    }


def _collect_attention_sample_matrix_from_arrays(
    model: nn.Module,
    fingerprint_np: np.ndarray,
    numeric_np: np.ndarray,
    feature_labels: list,
    batch_size: int = 64,
) -> np.ndarray:
    """
    收集注意力分数矩阵，用于分析哪些指纹位被模型关注。
    """
    attn_model = _resolve_attention_model(model)
    n_samples = int(np.asarray(fingerprint_np).shape[0])
    n_features = len(feature_labels)
    out = np.zeros((n_samples, n_features), dtype=np.float32)
    if attn_model is None or n_samples == 0 or n_features == 0:
        return out

    device = next(attn_model.parameters()).device
    was_training = bool(attn_model.training)
    attn_model.eval()
    attn_model.capture_attn = True
    try:
        for start in range(0, n_samples, int(max(1, batch_size))):
            end = min(n_samples, start + int(max(1, batch_size)))
            fp_t = torch.from_numpy(np.asarray(fingerprint_np[start:end], dtype=np.float32)).to(device)
            num_t = torch.from_numpy(np.asarray(numeric_np[start:end], dtype=np.float32)).to(device)
            with torch.no_grad():
                _ = attn_model(fp_t, num_t)
            self_list = attn_model.attn_cache.get("self", [])
            idx_tensor = getattr(attn_model, "last_token_indices", None)
            if (not self_list) or idx_tensor is None:
                continue
            attn = self_list[-1].detach().mean(dim=1)
            if isinstance(attn_model, ChemistryAwareAttentionTransformer) and attn.size(-1) > 1:
                token_scores = attn[:, 0, 1:]
            else:
                token_scores = attn.mean(dim=1)
            idx_np = idx_tensor.detach().cpu().numpy()
            active_t = getattr(attn_model, "last_token_active", None)
            active_np = active_t.detach().cpu().numpy().astype(bool) if isinstance(active_t, torch.Tensor) else None
            score_np = token_scores.detach().cpu().numpy()
            for row in range(idx_np.shape[0]):
                bit_pos = np.asarray(idx_np[row], dtype=int).reshape(-1)
                vals = np.asarray(score_np[row], dtype=np.float32).reshape(-1)
                size = min(bit_pos.size, vals.size)
                if size <= 0:
                    continue
                bit_pos = bit_pos[:size]
                vals = np.clip(vals[:size], 0.0, None)
                if active_np is not None and row < active_np.shape[0]:
                    keep_mask = np.asarray(active_np[row], dtype=bool).reshape(-1)[:size]
                else:
                    keep_mask = np.ones((size,), dtype=bool)
                if not keep_mask.any():
                    continue
                vals_keep = vals[keep_mask]
                if float(vals_keep.sum()) > 1e-12:
                    vals_keep = vals_keep / float(vals_keep.sum())
                bit_keep = bit_pos[keep_mask]
                for local_idx, val in zip(bit_keep.tolist(), vals_keep.tolist()):
                    if 0 <= int(local_idx) < n_features:
                        out[start + row, int(local_idx)] += float(val)
    finally:
        attn_model.capture_attn = False
        if was_training:
            attn_model.train()
    return out


def _integrated_gradients_fingerprint(
    model: nn.Module,
    fingerprint: torch.Tensor,
    numeric: torch.Tensor,
    steps: int = 8,
) -> torch.Tensor:
    """
    计算积分梯度（Integrated Gradients），用于归因特征重要性。
    通过在基线（零向量）和输入之间插值，积分梯度能够更准确地反映特征对预测的贡献。
    """
    steps = int(max(4, steps))
    baseline = torch.zeros_like(fingerprint)
    total_grad = torch.zeros_like(fingerprint)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=fingerprint.device, dtype=fingerprint.dtype)[1:]
    model.zero_grad(set_to_none=True)
    for alpha in alphas:
        fp_interp = (baseline + alpha * (fingerprint - baseline)).detach()
        fp_interp.requires_grad_(True)
        pred = model(fp_interp, numeric)
        grad = torch.autograd.grad(pred.sum(), fp_interp, retain_graph=False, create_graph=False, allow_unused=False)[0]
        total_grad = total_grad + grad.detach()
        model.zero_grad(set_to_none=True)
    return (fingerprint - baseline) * (total_grad / float(steps))


def _collect_ig_sample_matrix_from_arrays(
    model: nn.Module,
    fingerprint_np: np.ndarray,
    numeric_np: np.ndarray,
    batch_size: int = 32,
    steps: int = 8,
) -> np.ndarray:
    """收集积分梯度矩阵。"""
    n_samples = int(np.asarray(fingerprint_np).shape[0])
    n_features = int(np.asarray(fingerprint_np).shape[1]) if n_samples > 0 else 0
    out = np.zeros((n_samples, n_features), dtype=np.float32)
    if n_samples == 0 or n_features == 0:
        return out

    device = next(model.parameters()).device
    was_training = bool(model.training)
    model.eval()
    try:
        for start in range(0, n_samples, int(max(1, batch_size))):
            end = min(n_samples, start + int(max(1, batch_size)))
            fp_t = torch.from_numpy(np.asarray(fingerprint_np[start:end], dtype=np.float32)).to(device)
            num_t = torch.from_numpy(np.asarray(numeric_np[start:end], dtype=np.float32)).to(device)
            ig_t = _integrated_gradients_fingerprint(model, fp_t, num_t, steps=steps)
            out[start:end] = torch.abs(ig_t).detach().cpu().numpy().astype(np.float32, copy=False)
    finally:
        if was_training:
            model.train()
    return out


def _compute_mask_delta_scores_from_arrays(
    model: nn.Module,
    fingerprint_np: np.ndarray,
    numeric_np: np.ndarray,
    candidate_local_idx: list,
    batch_size: int = 64,
) -> Dict[int, Dict[str, float]]:
    """
    计算掩码扰动分数（Mask Delta Score）。
    通过强制将某个特征置零，观察预测值的变化，评估该特征的重要性。
    """
    out = {int(i): {"delta_sum": 0.0, "delta_count": 0, "delta_max": 0.0} for i in candidate_local_idx}
    n_samples = int(np.asarray(fingerprint_np).shape[0])
    if n_samples == 0 or not candidate_local_idx:
        return out

    device = next(model.parameters()).device
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, n_samples, int(max(1, batch_size))):
                end = min(n_samples, start + int(max(1, batch_size)))
                fp_t = torch.from_numpy(np.asarray(fingerprint_np[start:end], dtype=np.float32)).to(device)
                num_t = torch.from_numpy(np.asarray(numeric_np[start:end], dtype=np.float32)).to(device)
                pred_ref = model(fp_t, num_t).detach()
                for local_idx in candidate_local_idx:
                    local_idx = int(local_idx)
                    if local_idx < 0 or local_idx >= fp_t.size(1):
                        continue
                    active_mask = torch.abs(fp_t[:, local_idx]) > 1e-6
                    if not bool(active_mask.any()):
                        continue
                    fp_masked = fp_t.clone()
                    fp_masked[active_mask, local_idx] = 0.0
                    pred_masked = model(fp_masked, num_t).detach()
                    delta = torch.abs(pred_masked - pred_ref)
                    delta_active = delta[active_mask]
                    out[local_idx]["delta_sum"] += float(delta_active.sum().item())
                    out[local_idx]["delta_count"] += int(delta_active.numel())
                    out[local_idx]["delta_max"] = max(out[local_idx]["delta_max"], float(delta_active.max().item()))
    finally:
        if was_training:
            model.train()
    return out


def _compute_bootstrap_stability_scores(
    sample_score_matrix: np.ndarray,
    candidate_local_idx: list,
    rounds: int = 32,
    frac: float = 0.75,
    topk: int = 10,
    seed: int = 42,
) -> Dict[int, float]:
    """
    计算Bootstrapping稳定性分数。
    通过多次重采样数据，观察特征是否稳定地出现在Top-K中。
    """
    candidate_local_idx = [int(i) for i in candidate_local_idx]
    out = {int(i): 0.0 for i in candidate_local_idx}
    mat = np.asarray(sample_score_matrix, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[1] == 0 or not candidate_local_idx:
        return out
    n_samples = int(mat.shape[0])
    boot_rounds = int(max(1, rounds))
    boot_topk = int(max(1, min(int(topk), len(candidate_local_idx))))
    boot_n = int(max(8, min(n_samples, round(float(frac) * n_samples))))
    rng = np.random.default_rng(int(seed))
    hit_count = Counter()
    for _ in range(boot_rounds):
        if boot_n >= n_samples:
            chosen = np.arange(n_samples, dtype=int)
        else:
            chosen = np.asarray(rng.choice(n_samples, size=boot_n, replace=False), dtype=int)
        mean_scores = np.asarray(mat[chosen].mean(axis=0), dtype=np.float32)
        candidate_scores = np.asarray([mean_scores[i] if 0 <= i < mean_scores.size else -np.inf for i in candidate_local_idx], dtype=np.float32)
        order = np.argsort(candidate_scores)[::-1][:boot_topk]
        for pos in order.tolist():
            hit_count[int(candidate_local_idx[int(pos)])] += 1
    for local_idx in candidate_local_idx:
        out[int(local_idx)] = float(hit_count.get(int(local_idx), 0) / max(boot_rounds, 1))
    return out


def _plot_consensus_top_motifs(consensus_rows: list, output_dir: str, top_n: int = 10) -> None:
    """绘制综合Top-N基团的条形图。"""
    rows = list(consensus_rows[: int(max(1, top_n))])
    if not rows:
        return
    labels = []
    vals = []
    fg_labels = []
    for row in rows:
        fg_name = str(row.get("functional_group", "")).strip()
        fg_display = str(row.get("functional_display", "")).strip()
        fg_labels.append(f"{fg_name} ({fg_display})" if fg_name and fg_display else (fg_name if fg_name else "Unassigned"))
        label = str(row.get("bit_name", "")).strip()
        if not label:
            label = str(row.get("functional_group", "")).strip() or "motif"
        labels.append(label)
        vals.append(float(row.get("consensus_score", row.get("consensus_score_sum", row.get("functional_score_sum", 0.0)))))

    fig_h = max(6.2, min(11.8, 1.8 + 0.50 * len(rows)))
    fig, ax = plt.subplots(figsize=(7.8, fig_h))
    cmap = mpl.cm.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=float(min(vals)), vmax=float(max(vals) + 1e-12))
    colors = [cmap(norm(v)) for v in vals]
    y_pos = np.arange(len(rows))
    ax.barh(y_pos, vals, color=colors, edgecolor="#1F2937", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{lab} | {fg}" for lab, fg in zip(labels, fg_labels)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Consensus motif score")
    ax.set_title(f"Top-{len(rows)} consensus motifs", fontsize=15, pad=10)
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.30)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x_max = float(max(vals)) if vals else 1.0
    ax.set_xlim(0.0, x_max * 1.18 if x_max > 0 else 1.0)
    for y, row in zip(y_pos, rows):
        score = float(row.get("consensus_score", row.get("consensus_score_sum", row.get("functional_score_sum", 0.0))))
        if "bit_name" in row:
            txt = (
                f"C={score:.3f} | A={float(row.get('attention_score', 0.0)):.3f} | "
                f"IG={float(row.get('ig_score', 0.0)):.3f} | Δ={float(row.get('perturb_score', 0.0)):.3f} | "
                f"Stab={float(row.get('stability_score', 0.0)):.2f}"
            )
        else:
            txt = (
                f"Score={score:.3f} | Δlogk={float(row.get('delta_logk', 0.0)):+.3f} | "
                f"support={int(row.get('support_molecules', 0))} | bits={int(row.get('bit_count', 0))}"
            )
        ax.text(score + max(0.003, x_max * 0.01), y, txt, va="center", ha="left", fontsize=8, color="#374151")
    fig.tight_layout()
    _savefig_with_pdf(fig, os.path.join(output_dir, "consensus_top_motifs.png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _infer_model_mode_from_instance(model: nn.Module) -> str:
    if isinstance(model, DualExpertRegressor):
        return "dual"
    if _resolve_attention_model(model) is not None:
        return "attn"
    return "mlp"


def _build_model_for_mode(
    model_mode: str,
    fingerprint_dim: int,
    numeric_dim: int,
    cfg: FingerprintConfig,
) -> nn.Module:
    """
    根据模式构建模型实例 (Factory function to build model).
    
    参数:
    - model_mode: "mlp" (基线) 或 "attn" (Transformer)
    - fingerprint_dim: 输入指纹维度 (Top-K)
    - numeric_dim: 数值特征维度
    - cfg: 超参数配置对象
    """
    mode = str(model_mode).strip().lower()
    
    if mode == "mlp":
        # 基线 MLP 模型 (Baseline MLP)
        return FingerprintTransformer(fingerprint_dim, numeric_dim, cfg)
        
    if mode == "attn":
        return _build_attention_expert(fingerprint_dim, numeric_dim, cfg)
        
    # 默认回退到 MLP
    return FingerprintTransformer(fingerprint_dim, numeric_dim, cfg)


def _build_cma_cfg_from_model(model: nn.Module) -> FingerprintConfig:
    """从模型实例恢复配置对象 (Recover config from model instance)."""
    base_cfg = getattr(model, "config", None)
    cfg = FingerprintConfig()
    if base_cfg is not None:
        for name in (
            "d_model",
            "dropout",
            "activation",
            "n_heads",
            "n_layers",
            "max_fp_tokens",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "max_epochs",
            "early_stopping_patience",
            "save_interval",
            "min_delta",
            "scheduler_type",
            "fingerprint_scale",
            "norm_first",
            "attn_pooling",
            "fp_bit_dropout",
            "base_numeric_dim",
            "chemistry_aware_attention",
            "n_categories",
            "pooling"
        ):
            if hasattr(base_cfg, name):
                setattr(cfg, name, copy.deepcopy(getattr(base_cfg, name)))
    return cfg


def _set_global_random_seed(seed: int) -> None:
    """设置全局随机种子以保证可复现性 (Set global random seed)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 如果开启确定性模式 (可能变慢)
    if _env_bool("TRANSFORMER_DETERMINISTIC", False):
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _make_selected_subsets_for_indices(
    dataset: FingerprintReactionDataset,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    fp_bits: int,
    topk_features: int,
    fp_select_method: str,
) -> Tuple[Dataset, Dataset, int]:
    """
    创建带特征选择的数据子集 (Create feature-selected subsets).
    
    流程:
    1. 根据 train_idx 计算特征重要性排名 (Feature Ranking).
    2. 选择 Top-K 个最重要的指纹位.
    3. 创建 SelectedFeatureSubset，只保留这些位.
    """
    use_topk = _env_bool_pair("TRANSFORMER_V3_USE_TOPK", "TRANSFORMER_V2_USE_TOPK", True)
    if use_topk:
        # 获取特征排名 (仅使用训练集，避免泄露)
        rank = _get_fp_ranking(dataset, np.asarray(train_idx, dtype=int), int(fp_bits), str(fp_select_method))
        k = int(min(max(1, topk_features), dataset.fingerprint_dim))
        top_idx = rank[:k]
        
        # 创建只包含Top-K特征的子集
        train_subset = SelectedFeatureSubset(dataset, train_idx, top_idx)
        eval_subset = SelectedFeatureSubset(dataset, eval_idx, top_idx)
        return train_subset, eval_subset, int(k)
        
    # 如果不使用特征选择，返回原始子集
    return Subset(dataset, train_idx), Subset(dataset, eval_idx), int(dataset.fingerprint_dim)


def _train_stability_model_for_fold(
    *,
    dataset: FingerprintReactionDataset,
    train_idx: np.ndarray,
    hold_idx: np.ndarray,
    fp_bits: int,
    topk_features: int,
    model_mode: str,
    cfg_base: FingerprintConfig,
    fp_select_method: str,
    seed: int,
) -> Tuple[Optional[nn.Module], Optional[DataLoader]]:
    """
    训练用于稳定性分析的模型 (Train model for stability analysis).
    
    这是一个简化的训练流程，用于 Cross-Seed Stability Analysis。
    不进行复杂的 Logging 或 Checkpointing，只返回训练好的模型用于验证。
    """

    train_idx = np.asarray(train_idx, dtype=int).reshape(-1)
    hold_idx = np.asarray(hold_idx, dtype=int).reshape(-1)
    if train_idx.size < 16 or hold_idx.size < 8:
        return None, None

    # 设置随机种子 (Set seed)
    _set_global_random_seed(int(seed))
    dataset.fit_scalers(train_idx)
    
    # 创建训练集和验证集 (Create splits)
    train_subset, hold_subset, fp_dim_for_model = _make_selected_subsets_for_indices(
        dataset=dataset,
        train_idx=train_idx,
        eval_idx=hold_idx,
        fp_bits=int(fp_bits),
        topk_features=int(topk_features),
        fp_select_method=str(fp_select_method),
    )

    # 复制配置并进行微调（为了速度，通常减少Epoch数）(Adjust config for speed)
    cfg = copy.deepcopy(cfg_base)
    # 限制最大 Epoch 以加速稳定性评估
    cfg.max_epochs = int(_env_value("TRANSFORMER_CMA_STABILITY_MAX_EPOCHS", None, min(int(cfg.max_epochs), 80)))
    # 减少 Early Stopping Patience
    cfg.early_stopping_patience = int(
        _env_value("TRANSFORMER_CMA_STABILITY_EARLY_STOP", None, min(int(cfg.early_stopping_patience), 20))
    )
    # 调整 Batch Size
    cfg.batch_size = int(_env_value("TRANSFORMER_CMA_STABILITY_BATCH_SIZE", None, cfg.batch_size))
    # 调整 Min Delta
    cfg.min_delta = float(_env_value("TRANSFORMER_CMA_STABILITY_MIN_DELTA", None, cfg.min_delta))
    if model_mode in {"dual", "attn"}:
        cfg.max_fp_tokens = min(int(cfg.max_fp_tokens), int(fp_dim_for_model))
    cfg.base_numeric_dim = int(getattr(dataset, "base_num_dim", dataset.num_dim))

    # 构建模型 (Build model)
    model_fold = _build_model_for_mode(model_mode, fp_dim_for_model, dataset.num_dim, cfg)
    device_env = str(os.environ.get("TRANSFORMER_DEVICE", "")).strip()
    trainer = FingerprintTrainer(model_fold, cfg, device=device_env or None)
    use_cuda = str(trainer.device).lower().startswith("cuda")
    
    # 采样器 (Balanced Sampler)
    train_sampler, _ = _build_category_balanced_sampler(dataset, train_subset, train_idx)
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        pin_memory=use_cuda,
    )
    hold_loader = DataLoader(
        hold_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    # 物理先验 (Physics Prior)
    physics_meta_np = _build_category_physics_meta(dataset, train_idx)
    physics_prior_t = torch.as_tensor(physics_meta_np["sign"], dtype=torch.float32, device=trainer.device)
    physics_meta_t = _physics_meta_to_tensors(physics_meta_np, trainer.device)
    
    best_state = None
    best_monitor_r2 = float("-inf")
    best_monitor_rmse = float("inf")
    patience = 0

    # 训练循环 (Training Loop)
    for _epoch in range(1, int(cfg.max_epochs) + 1):
        # 训练一个 Epoch (使用因果不变性损失)
        _ = _train_epoch_causal_invariant(
            trainer=trainer,
            dataloader=train_loader,
            physics_sign_prior=physics_prior_t,
            physics_meta=physics_meta_t,
            epoch=int(_epoch),
            max_epochs=int(cfg.max_epochs),
        )
        
        # 验证 (Evaluation)
        _, val_r2_epoch, val_rmse_epoch = trainer.evaluate(hold_loader)
        
        # 学习率调度 (Scheduler Step)
        if cfg.scheduler_type == "plateau" and trainer.scheduler is not None:
            trainer.scheduler.step(val_rmse_epoch if np.isfinite(val_rmse_epoch) else 1.0)
            
        # 早停检查 (Early Stopping Check)
        improved = False
        if np.isfinite(val_r2_epoch):
            improved = val_r2_epoch > best_monitor_r2 + float(cfg.min_delta)
        elif np.isfinite(val_rmse_epoch):
            improved = val_rmse_epoch < best_monitor_rmse - float(cfg.min_delta)
            
        if improved:
            best_monitor_r2 = float(val_r2_epoch)
            best_monitor_rmse = float(val_rmse_epoch)
            best_state = copy.deepcopy(trainer.model.state_dict())
            patience = 0
        else:
            patience += 1
            
        if patience >= int(cfg.early_stopping_patience):
            break

    if best_state is not None:
        trainer.model.load_state_dict(best_state)
    return trainer.model, hold_loader


def _compute_consensus_rows_core(
    *,
    model: nn.Module,
    fingerprint_np: np.ndarray,
    numeric_np: np.ndarray,
    feature_labels: list,
    candidate_topk: int,
    final_topk: int,
    min_support: int,
    batch_size: int,
    ig_steps: int,
    stability_mode: str = "bootstrap",
    bootstrap_rounds: int = 32,
    bootstrap_frac: float = 0.75,
    stability_override: Optional[Dict[str, float]] = None,
    weight_attn: float = 0.20,
    weight_ig: float = 0.20,
    weight_delta: float = 0.30,
    weight_stab: float = 0.20,
    weight_support: float = 0.10,
) -> Tuple[list, np.ndarray, np.ndarray, list]:
    """
    计算核心的“共识”特征分数 (Compute Core Consensus Feature Scores).
    
    这是 "Explainable AI" 的核心部分，综合多种指标对指纹位进行评分：
    1. **Attention Weights**: 模型实际上关注了哪些位？
    2. **Integrated Gradients (IG)**: 哪些位对预测结果贡献最大？
    3. **Delta Impact**: 移除某些位会导致预测值变化多大？
    4. **Stability**: 这些特征在多次 Bootstrap 采样中是否稳定出现？
    5. **Support**: 有多少样本实际上激活了这个位？
    
    参数:
    - model: 训练好的模型
    - stability_override: 预先计算的稳定性分数 (来自 Cross-Seed Analysis)
    - weight_*: 各个评分分量的权重
    
    返回:
    - selected_rows: 最终选出的 Top-K 特征详细信息列表
    - importance_scores: 所有候选特征的综合得分
    - ...
    """
    
    # 1. 收集注意力矩阵 (Collect Attention Weights)
    # [N_samples, N_features]
    attn_mat = _collect_attention_sample_matrix_from_arrays(
        model=model,
        fingerprint_np=fingerprint_np,
        numeric_np=numeric_np,
        feature_labels=feature_labels,
        batch_size=min(64, max(8, int(batch_size))),
    )
    
    # 2. 收集积分梯度矩阵 (Collect Integrated Gradients)
    # [N_samples, N_features]
    ig_mat = _collect_ig_sample_matrix_from_arrays(
        model=model,
        fingerprint_np=fingerprint_np,
        numeric_np=numeric_np,
        batch_size=min(32, max(4, int(batch_size))),
        steps=int(ig_steps),
    )
    
    # 计算特征活跃度 (Feature Activity)
    active_mat = (np.abs(fingerprint_np) > 1e-6).astype(np.float32)
    support_count = active_mat.sum(axis=0).astype(np.float32)
    support_frac = support_count / max(float(fingerprint_np.shape[0]), 1.0)
    denom = np.where(support_count > 0, support_count, 1.0)
    
    # 计算平均分数 (Average Scores per Feature)
    attn_score = (attn_mat * active_mat).sum(axis=0) / denom
    ig_score = (ig_mat * active_mat).sum(axis=0) / denom
    support_gate = support_count >= float(min_support)

    # 归一化 (Normalize)
    attn_norm = _minmax_scale_array(attn_score)
    ig_norm = _minmax_scale_array(ig_score)
    support_norm = _minmax_scale_array(support_frac)
    
    # 初步筛选Top-K候选 (Preliminary Screening)
    # 组合 Attention, IG, Support 分数来选出最有潜力的特征进行后续昂贵的计算
    pre_score = 0.55 * attn_norm + 0.45 * ig_norm + 0.15 * support_norm
    # 过滤掉支持度过低的特征
    pre_score = np.where(support_gate, pre_score, -np.inf)
    
    candidate_order = np.argsort(pre_score)[::-1]
    candidate_local_idx = [int(i) for i in candidate_order.tolist() if np.isfinite(pre_score[int(i)])][: int(candidate_topk)]
    
    if not candidate_local_idx:
        return [], active_mat, support_count, []

    # 3. 计算掩码扰动分数 (Mask Delta Impact) - 仅针对候选特征
    # 这是一个昂贵的操作：逐个屏蔽特征并测量预测值的变化
    delta_map = _compute_mask_delta_scores_from_arrays(
        model=model,
        fingerprint_np=fingerprint_np,
        numeric_np=numeric_np,
        candidate_local_idx=candidate_local_idx,
        batch_size=min(64, max(8, int(batch_size))),
    )
    
    # 4. 计算稳定性分数 (Stability Analysis via Bootstrapping)
    # 在样本层面进行 Bootstrap，看特征的重要性在不同样本子集中是否稳定
    combined_sample = 0.5 * _row_normalize_nonnegative(attn_mat * active_mat) + 0.5 * _row_normalize_nonnegative(ig_mat * active_mat)
    bootstrap_map = _compute_bootstrap_stability_scores(
        sample_score_matrix=combined_sample,
        candidate_local_idx=candidate_local_idx,
        rounds=int(bootstrap_rounds),
        frac=float(bootstrap_frac),
        topk=min(int(final_topk), max(5, int(final_topk))),
        seed=int(_env_value("TRANSFORMER_SEED", None, 42)),
    )

    perturb_score = np.zeros((fingerprint_np.shape[1],), dtype=np.float32)
    stability_score = np.zeros((fingerprint_np.shape[1],), dtype=np.float32)
    bootstrap_score = np.zeros((fingerprint_np.shape[1],), dtype=np.float32)
    
    # 处理候选特征的各个分数 (Process Candidate Scores)
    for pos, local_idx in enumerate(candidate_local_idx):
        # 获取 Delta Impact
        info = delta_map.get(int(local_idx), {})
        cnt = int(info.get("delta_count", 0))
        if cnt > 0:
            perturb_score[int(local_idx)] = float(info.get("delta_sum", 0.0) / max(cnt, 1))
            
        # 获取 Bootstrap Stability
        bootstrap_score[int(local_idx)] = float(bootstrap_map.get(int(local_idx), 0.0))
        
        # 确定最终使用的 Stability Score (Override vs Bootstrap vs None)
        if str(stability_mode).strip().lower() == "override" and isinstance(stability_override, dict):
            # 如果有 Cross-Seed Stability，优先使用
            stability_score[int(local_idx)] = float(stability_override.get(str(feature_labels[int(local_idx)]), 0.0))
        elif str(stability_mode).strip().lower() == "none":
            stability_score[int(local_idx)] = 0.0
        else:
            # 默认使用本次 Bootstrap 结果
            stability_score[int(local_idx)] = float(bootstrap_map.get(int(local_idx), 0.0))

    # 归一化各个分量以便加权 (Normalize Components for Weighted Sum)
    attn_cons = _minmax_scale_array(attn_score[candidate_local_idx])
    ig_cons = _minmax_scale_array(ig_score[candidate_local_idx])
    delta_cons = _minmax_scale_array(perturb_score[candidate_local_idx])
    stab_cons = _minmax_scale_array(stability_score[candidate_local_idx])
    supp_cons = _minmax_scale_array(support_frac[candidate_local_idx])

    rows = []
    for pos, local_idx in enumerate(candidate_local_idx):
        bit_name = str(feature_labels[int(local_idx)])
        # 组装特征详情
        rows.append(
            {
                "local_feature_idx": int(local_idx),
                "bit_name": bit_name,
                "bit": int(_parse_fp_bit_index(bit_name) if _parse_fp_bit_index(bit_name) is not None else local_idx),
                "attention_score": float(attn_score[int(local_idx)]),
                "ig_score": float(ig_score[int(local_idx)]),
                "perturb_score": float(perturb_score[int(local_idx)]),
                "stability_score": float(stability_score[int(local_idx)]),
                "bootstrap_stability_score": float(bootstrap_score[int(local_idx)]),
                "support_count": int(round(float(support_count[int(local_idx)]))),
                "support_frac": float(support_frac[int(local_idx)]),
                # 计算最终共识分数 (Final Consensus Score)
                "consensus_score": float(
                    weight_attn * float(attn_cons[pos])
                    + weight_ig * float(ig_cons[pos])
                    + weight_delta * float(delta_cons[pos])
                    + weight_stab * float(stab_cons[pos])
                    + weight_support * float(supp_cons[pos])
                ),
            }
        )
        
    # 按共识分数排序 (Sort by Consensus Score)
    rows.sort(
        key=lambda row: (
            float(row.get("consensus_score", float("-inf"))),
            float(row.get("perturb_score", 0.0)),
            float(row.get("attention_score", 0.0)),
            float(row.get("ig_score", 0.0)),
            float(row.get("stability_score", 0.0)),
            float(row.get("support_count", 0.0)),
        ),
        reverse=True,
    )
    ranked_names = [str(row["bit_name"]) for row in rows]
    return rows, active_mat, support_count, ranked_names


def _compute_cross_seed_fold_stability(
    *,
    model: nn.Module,
    dataset: FingerprintReactionDataset,
    train_indices: np.ndarray,
    candidate_topk: int,
    final_topk: int,
    min_support: int,
    batch_size: int,
    ig_steps: int,
    fp_bits: int,
    topk_features: int,
    fp_select_method: str,
) -> Dict[str, float]:
    """
    计算跨随机种子和Fold的稳定性分数 (Cross-Seed Fold Stability Analysis).
    
    这是最耗时的步骤，通过多次训练不同种子和不同数据划分的模型，
    来验证特征是否在各种情况下都能被选为Top-K。
    
    流程:
    1. 解析环境变量 TRANSFORMER_CMA_STABILITY_SEEDS (默认 5 个种子)
    2. 解析环境变量 TRANSFORMER_CMA_STABILITY_FOLDS (默认 5 折)
    3. 总共训练 Seeds x Folds (默认 25) 个模型
    4. 对每个模型，在验证集上计算特征重要性，记录 Top-K 特征
    5. 统计每个特征被选中的频率作为 Stability Score
    
    返回:
    - stability_map: {feature_name: frequency_score}
    """
    enabled = _env_bool("TRANSFORMER_CMA_STABILITY_TRUE", True)
    if not enabled:
        return {}

    seed_text = str(_env_value("TRANSFORMER_CMA_STABILITY_SEEDS", None, "11,19,23,29,37")).strip()
    seeds = _parse_seeds(seed_text)
    fold_count = int(max(2, int(_env_value("TRANSFORMER_CMA_STABILITY_FOLDS", None, 5))))
    model_mode = _infer_model_mode_from_instance(model)
    cfg_base = _build_cma_cfg_from_model(model)
    train_indices = np.asarray(train_indices, dtype=int).reshape(-1)
    train_indices = train_indices[(train_indices >= 0) & (train_indices < len(dataset))]
    if train_indices.size < max(24, fold_count * 4):
        return {}

    hit_counter = Counter()
    run_counter = Counter()
    total_runs = 0
    fp_bits = int(fp_bits if fp_bits > 0 else getattr(dataset, "max_fp_bits", dataset.fingerprint_dim))
    topk_features = int(max(1, topk_features))
    fp_select_method = str(fp_select_method).strip().lower() or "rf"

    # 双重循环：遍历不同的随机种子和交叉验证Fold
    # 这就是导致程序运行看似"卡住"的原因，实际上是在后台训练大量模型
    print(f"[V5] Starting Stability Analysis: {len(seeds)} seeds x {fold_count} folds...")
    
    for seed in seeds:
        kf = KFold(n_splits=int(fold_count), shuffle=True, random_state=int(seed))
        for fold_id, (sub_train_pos, sub_hold_pos) in enumerate(kf.split(train_indices)):
            sub_train_idx = np.asarray(train_indices[sub_train_pos], dtype=int)
            sub_hold_idx = np.asarray(train_indices[sub_hold_pos], dtype=int)
            
            # 为当前Fold训练一个新模型 (Train new model for this fold)
            # 使用简化的训练流程 (Reduced epochs/patience)
            try:
                model_fold, hold_loader = _train_stability_model_for_fold(
                    dataset=dataset,
                    train_idx=sub_train_idx,
                    hold_idx=sub_hold_idx,
                    fp_bits=fp_bits,
                    topk_features=topk_features,
                    model_mode=model_mode,
                    cfg_base=cfg_base,
                    fp_select_method=fp_select_method,
                    seed=int(seed * 100 + fold_id),
                )
            except Exception as e:
                print(f"[V5] Warning: Stability fold failed (seed={seed}, fold={fold_id}): {e}")
                continue
                
            if model_fold is None or hold_loader is None:
                continue
            
            # 在验证集上计算特征重要性 (Compute Importance on Validation Set)
            # 使用与最终解释相同的核心算法 (Attention + IG + Support)
            try:
                explain = _collect_explain_arrays(hold_loader, max_samples=None)
                fp_np = np.asarray(explain.get("fingerprint", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
                num_np = np.asarray(explain.get("numeric", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
                feature_labels = list(explain.get("feature_labels", []) or [])
                if fp_np.ndim != 2 or fp_np.shape[0] == 0 or fp_np.shape[1] == 0 or not feature_labels:
                    continue

                # 计算当前Fold的特征排名
                local_rows, _, _, local_ranked = _compute_consensus_rows_core(
                    model=model_fold,
                    fingerprint_np=fp_np,
                    numeric_np=num_np,
                    feature_labels=feature_labels,
                    candidate_topk=int(candidate_topk),
                    final_topk=int(final_topk),
                    min_support=int(min_support),
                    batch_size=int(batch_size),
                    ig_steps=int(ig_steps),
                    stability_mode="none",
                    bootstrap_rounds=0,
                    bootstrap_frac=0.75,
                    stability_override=None,
                    weight_attn=0.20,
                    weight_ig=0.20,
                    weight_delta=0.30,
                    weight_stab=0.0,
                    weight_support=0.10,
                )
                if not local_rows:
                    continue
                total_runs += 1
                # 统计哪些特征进入了Top-K
                top_names = [str(x) for x in local_ranked[: int(final_topk)]]
                for name in top_names:
                    hit_counter[name] += 1
                for row in local_rows:
                    run_counter[str(row.get("bit_name", ""))] += 1
            except Exception:
                continue

    stability_map = {}
    if total_runs <= 0:
        return stability_map
    for name, hits in hit_counter.items():
        # 稳定性分数 = 出现在Top-K的次数 / 总运行次数
        stability_map[str(name)] = float(hits / max(total_runs, 1))
    return stability_map


def export_consensus_motif_artifacts(
    model: nn.Module,
    dataset: FingerprintReactionDataset,
    explain_loader: DataLoader,
    train_indices: np.ndarray,
    prefix: str,
) -> Dict[str, object]:
    """
    导出共识基团分析的所有结果 (Export Consensus Motif Artifacts).
    
    这是整个解释性分析的入口函数，负责调用各个子模块并生成CSV和图表。
    流程：
    1. 收集解释数据 (Collect Data).
    2. 计算跨种子稳定性 (Compute Cross-Seed Stability).
    3. 计算最终共识分数 (Compute Final Consensus Scores).
    4. 导出 CSV 和 绘制图表 (Export & Plot).
    """
    out_dir = get_run_output_dir(prefix)
    if explain_loader is None:
        return {}

    # 读取环境变量配置 (Read Config from Env)
    max_samples = int(max(64, int(_env_value("TRANSFORMER_CMA_MAX_SAMPLES", None, 512))))
    ig_steps = int(max(4, int(_env_value("TRANSFORMER_CMA_IG_STEPS", None, 8))))
    bootstrap_rounds = int(max(8, int(_env_value("TRANSFORMER_CMA_BOOTSTRAP_ROUNDS", None, 32))))
    bootstrap_frac = float(_env_value("TRANSFORMER_CMA_BOOTSTRAP_FRAC", None, 0.75))
    candidate_topk = int(max(12, int(_env_value("TRANSFORMER_CMA_CANDIDATE_TOPK", None, 24))))
    final_topk = int(max(5, int(_env_value("TRANSFORMER_CMA_TOPK", None, 10))))
    min_support = int(max(2, int(_env_value("TRANSFORMER_CMA_MIN_SUPPORT", None, 5))))
    batch_size = int(_env_value("TRANSFORMER_CMA_BATCH_SIZE", None, 32))
    weight_attn = float(_env_value("TRANSFORMER_CMA_W_ATTENTION", None, 0.20))
    weight_ig = float(_env_value("TRANSFORMER_CMA_W_IG", None, 0.20))
    weight_delta = float(_env_value("TRANSFORMER_CMA_W_DELTA", None, 0.30))
    weight_stab = float(_env_value("TRANSFORMER_CMA_W_STABILITY", None, 0.20))
    weight_support = float(_env_value("TRANSFORMER_CMA_W_SUPPORT", None, 0.10))

    # 收集数据 (Collect Explain Arrays)
    explain = _collect_explain_arrays(explain_loader, max_samples=max_samples)
    fp_np = np.asarray(explain.get("fingerprint", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    num_np = np.asarray(explain.get("numeric", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    feature_labels = list(explain.get("feature_labels", []) or [])
    if fp_np.ndim != 2 or fp_np.shape[0] == 0 or fp_np.shape[1] == 0 or not feature_labels:
        return {}
    
    # 2. 初步计算共识分数 (Preliminary Consensus Scoring)
    # 使用 Bootstrapping 计算样本内稳定性
    rows, _, _, _ = _compute_consensus_rows_core(
        model=model,
        fingerprint_np=fp_np,
        numeric_np=num_np,
        feature_labels=feature_labels,
        candidate_topk=int(candidate_topk),
        final_topk=int(final_topk),
        min_support=int(min_support),
        batch_size=int(batch_size),
        ig_steps=int(ig_steps),
        stability_mode="bootstrap",
        bootstrap_rounds=int(bootstrap_rounds),
        bootstrap_frac=float(bootstrap_frac),
        stability_override=None,
        weight_attn=float(weight_attn),
        weight_ig=float(weight_ig),
        weight_delta=float(weight_delta),
        weight_stab=float(weight_stab),
        weight_support=float(weight_support),
    )
    
    if not rows:
        return {}

    # 3. 计算更严格的跨种子Fold稳定性 (Cross-Seed/Fold Stability Analysis)
    # 这是一个耗时的过程，但对于验证特征的鲁棒性至关重要
    topk_features = int(fp_np.shape[1])
    fp_bits = int(getattr(dataset, "max_fp_bits", dataset.fingerprint_dim))
    fp_select_method = str(_env_value("TRANSFORMER_V3_FP_SELECT", "TRANSFORMER_V2_FP_SELECT", "rf")).strip().lower()
    
    true_stability_map = _compute_cross_seed_fold_stability(
        model=model,
        dataset=dataset,
        train_indices=np.asarray(train_indices, dtype=int),
        candidate_topk=int(candidate_topk),
        final_topk=int(final_topk),
        min_support=int(min_support),
        batch_size=int(batch_size),
        ig_steps=int(ig_steps),
        fp_bits=int(fp_bits),
        topk_features=int(topk_features),
        fp_select_method=str(fp_select_method),
    )
    
    # 4. 更新分数与重新排序 (Update Scores & Re-rank)
    if true_stability_map:
        for row in rows:
            # 用 Cross-Seed Stability 替换 Bootstrap Stability
            row["stability_score"] = float(true_stability_map.get(str(row.get("bit_name", "")), 0.0))
            row["stability_source"] = "seed_fold"
            
        # 重新归一化各项分数 (Re-normalize)
        # 必须确保所有分量都在 [0, 1] 范围内，才能进行加权求和
        attn_cons = _minmax_scale_array(np.asarray([float(row.get("attention_score", 0.0)) for row in rows], dtype=np.float32))
        ig_cons = _minmax_scale_array(np.asarray([float(row.get("ig_score", 0.0)) for row in rows], dtype=np.float32))
        delta_cons = _minmax_scale_array(np.asarray([float(row.get("perturb_score", 0.0)) for row in rows], dtype=np.float32))
        stab_cons = _minmax_scale_array(np.asarray([float(row.get("stability_score", 0.0)) for row in rows], dtype=np.float32))
        supp_cons = _minmax_scale_array(np.asarray([float(row.get("support_frac", 0.0)) for row in rows], dtype=np.float32))
        
        # 重新计算加权共识分数
        for pos, row in enumerate(rows):
            row["consensus_score"] = float(
                weight_attn * float(attn_cons[pos])
                + weight_ig * float(ig_cons[pos])
                + weight_delta * float(delta_cons[pos])
                + weight_stab * float(stab_cons[pos])
                + weight_support * float(supp_cons[pos])
            )
            
        # 重新排序 (Re-sort)
        rows.sort(
            key=lambda row: (
                float(row.get("consensus_score", float("-inf"))),
                float(row.get("perturb_score", 0.0)),
                float(row.get("attention_score", 0.0)),
                float(row.get("ig_score", 0.0)),
                float(row.get("stability_score", 0.0)),
                float(row.get("support_count", 0.0)),
            ),
            reverse=True,
        )
    else:
        # 如果未进行 Cross-Seed Stability 分析，标记数据源为 Bootstrap
        for row in rows:
            row["stability_source"] = "bootstrap"

    # 5. 导出结果 (Export Results)
    dataset.fit_scalers(np.asarray(train_indices, dtype=int))
    # 保存详细的特征评分表
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "consensus_motif_scores.csv"), index=False, encoding="utf-8-sig")
    metric_map = {str(row["bit_name"]): dict(row) for row in rows}

    ranked_bit_names = [str(row["bit_name"]) for row in rows]
    # 导出子结构可视化和详细信息 (Export Substructure Visualizations)
    sub_info = export_attention_substructure_artifacts(
        dataset=dataset,
        train_indices=train_indices,
        ranked_bit_names=ranked_bit_names,
        prefix=prefix,
        artifact_tag="consensus",
        extra_bit_metrics=metric_map,
    )
    selected_rows = list((sub_info or {}).get("selected_rows", []) or [])
    bit_rows = list((sub_info or {}).get("bit_rows", []) or [])
    functional_rows = list((sub_info or {}).get("functional_rows", []) or [])
    
    # 6. 生成图表 (Generate Plots)
    if functional_rows:
        # 补充功能基团信息
        functional_rows = _supplement_functional_group_rows(
            functional_rows,
            output_dir=out_dir,
            top_n=int(_env_value("TRANSFORMER_TOP_FUNCTIONAL_GROUPS", None, 10)),
        )
        # 保存功能基团评分表
        pd.DataFrame(functional_rows).to_csv(os.path.join(out_dir, "consensus_motif_family_scores.csv"), index=False, encoding="utf-8-sig")
        # 绘制功能基团柱状图
        _plot_top_functional_groups(
            functional_rows,
            out_dir,
            artifact_tag="consensus",
            top_n=int(_env_value("TRANSFORMER_TOP_FUNCTIONAL_GROUPS", None, 10)),
        )
        # 绘制共识 Motif 雷达图
        _plot_consensus_top_motifs(
            functional_rows,
            out_dir,
            top_n=int(_env_value("TRANSFORMER_TOP_FUNCTIONAL_GROUPS", None, 10)),
        )
        # 绘制 Top 功能基团的分子结构图
        _plot_top_functional_group_structures(
            output_dir=out_dir,
            top_n=int(_env_value("TRANSFORMER_TOP_FUNCTIONAL_GROUPS", None, 10)),
        )
    elif selected_rows:
        # 如果没有完整的功能基团分析，进行简单的聚合
        family_counter = Counter()
        family_support = Counter()
        for row in selected_rows:
            fg_name = str(row.get("functional_group", "")).strip() or "Unassigned"
            family_counter[fg_name] += float(row.get("consensus_score", 0.0))
            family_support[fg_name] += int(row.get("support_count", 0))
            
        family_rows = [
            {
                "functional_group": str(name),
                "consensus_score_sum": float(score),
                "support_count_sum": int(family_support.get(name, 0)),
            }
            for name, score in family_counter.most_common()
        ]
        pd.DataFrame(family_rows).to_csv(os.path.join(out_dir, "consensus_motif_family_scores.csv"), index=False, encoding="utf-8-sig")
        _plot_consensus_top_motifs(selected_rows, out_dir, top_n=final_topk)
    else:
        # 最基础的 Motif 绘图
        _plot_consensus_top_motifs(rows, out_dir, top_n=final_topk)
        
    bit_to_family = {}
    source_rows = bit_rows if bit_rows else selected_rows
    for row in source_rows:
        bit_name = str(row.get("bit_name", "")).strip()
        family = str(row.get("functional_group", "")).strip()
        if bit_name and family:
            bit_to_family[bit_name] = family
            
    return {
        "ranked_bit_names": ranked_bit_names,
        "selected_rows": selected_rows,
        "functional_rows": functional_rows,
        "bit_to_family": bit_to_family,
    }

# =============================================================================
# Utility helpers (通用工具函数)

# =============================================================================
def _env_bool(name: str, default: bool = False) -> bool:
    """从环境变量读取布尔值 (Read boolean from env var)."""
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_value(primary: str, fallback: Optional[str], default):
    """从环境变量读取值，支持回退机制 (Read value with fallback)."""
    v = os.environ.get(primary)
    if (v is None or str(v).strip() == "") and fallback:
        v = os.environ.get(fallback)
    if v is None or str(v).strip() == "":
        return default
    return v


def _env_bool_pair(primary: str, fallback: Optional[str], default: bool = False) -> bool:
    """从环境变量对读取布尔值 (Read boolean from env var pair)."""
    v = _env_value(primary, fallback, None)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _setdefault_env(name: str, value) -> None:
    """如果环境变量未设置，则设置默认值 (Set default env var if not set)."""
    if os.environ.get(name) is None:
        os.environ[name] = str(value)


def _apply_plot_defaults_if_needed() -> None:
    """
    设置绘图和分析相关的默认环境变量配置。
    Keep V3 plot output style aligned with transformer.py.
    """
    _setdefault_env("TRANSFORMER_PLOT_FUSION", "0")
    _setdefault_env("TRANSFORMER_PLOT_FEATURE_CORR_NETWORK", "0")
    _setdefault_env("TRANSFORMER_FEATURE_CORR_PAPER", "0")
    _setdefault_env("TRANSFORMER_PLOT_FEATURE_CORR_NETWORK20", "1")
    _setdefault_env("TRANSFORMER_PLOT_ATTN_CLS_BY_CATEGORY", "1")
    # Attention分析参数
    _setdefault_env("TRANSFORMER_ATTN_TOPN", "20")
    _setdefault_env("TRANSFORMER_ATTN_CAT_HEATMAP_TOPN", "20")
    _setdefault_env("TRANSFORMER_ATTN_CAT_HEATMAP_MIN_CATEGORY", "8")
    _setdefault_env("TRANSFORMER_ATTN_CAT_HEATMAP_MIN_SUPPORT", "3")
    _setdefault_env("TRANSFORMER_ATTN_CAT_HEATMAP_MIN_CATS", "3")
    _setdefault_env("TRANSFORMER_ATTN_CAT_HEATMAP_ROW_NORM", "max")
    _setdefault_env("TRANSFORMER_ATTN_FAMILY_HEATMAP_TOPN", "10")
    _setdefault_env("TRANSFORMER_ATTN_FAMILY_HEATMAP_MIN_SUPPORT", "2")
    _setdefault_env("TRANSFORMER_ATTN_FAMILY_HEATMAP_MIN_CATS", "2")
    _setdefault_env("TRANSFORMER_ATTN_MIN_RADIUS", "2")
    _setdefault_env("TRANSFORMER_ATTN_SUBSTRUCT_TOPK", "30")
    _setdefault_env("TRANSFORMER_ATTN_SUBSTRUCT_USE_TRAINVAL", "0")
    _setdefault_env("TRANSFORMER_ATTN_SUBSTRUCT_MIN_ATOMS", "2")
    _setdefault_env("TRANSFORMER_ATTN_SUBSTRUCT_UNIQUE_TOPK", "10")
    _setdefault_env("TRANSFORMER_ATTN_SUBSTRUCT_UNIQUE_MIN_HEAVY", "4")
    # CMA (Consensus Motif Analysis) 参数
    _setdefault_env("TRANSFORMER_CMA_ENABLE", "1")
    _setdefault_env("TRANSFORMER_CMA_TOPK", "10")
    _setdefault_env("TRANSFORMER_CMA_CANDIDATE_TOPK", "24")
    _setdefault_env("TRANSFORMER_CMA_MIN_SUPPORT", "5")
    _setdefault_env("TRANSFORMER_CMA_MAX_SAMPLES", "512")
    _setdefault_env("TRANSFORMER_CMA_BATCH_SIZE", "32")
    _setdefault_env("TRANSFORMER_CMA_IG_STEPS", "8")
    _setdefault_env("TRANSFORMER_CMA_BOOTSTRAP_ROUNDS", "32")
    _setdefault_env("TRANSFORMER_CMA_BOOTSTRAP_FRAC", "0.75")
    # CMA 稳定性分析参数
    _setdefault_env("TRANSFORMER_CMA_STABILITY_TRUE", "1")
    _setdefault_env("TRANSFORMER_CMA_STABILITY_SEEDS", "11,19,23,29,37")
    _setdefault_env("TRANSFORMER_CMA_STABILITY_FOLDS", "5")
    _setdefault_env("TRANSFORMER_CMA_STABILITY_MAX_EPOCHS", "80")
    _setdefault_env("TRANSFORMER_CMA_STABILITY_EARLY_STOP", "20")
    _setdefault_env("TRANSFORMER_CMA_STABILITY_BATCH_SIZE", "32")
    _setdefault_env("TRANSFORMER_CMA_STABILITY_MIN_DELTA", "1e-4")
    # CMA 评分权重
    _setdefault_env("TRANSFORMER_CMA_W_ATTENTION", "0.20")
    _setdefault_env("TRANSFORMER_CMA_W_IG", "0.20")
    _setdefault_env("TRANSFORMER_CMA_W_DELTA", "0.30")
    _setdefault_env("TRANSFORMER_CMA_W_STABILITY", "0.20")
    _setdefault_env("TRANSFORMER_CMA_W_SUPPORT", "0.10")
    _setdefault_env("TRANSFORMER_TOP_FUNCTIONAL_GROUPS", "10")
    _setdefault_env("TRANSFORMER_EXPORT_UNIQUE_ATOM_GROUPS", "0")
    _setdefault_env("TRANSFORMER_PLOT_CATEGORY_METRICS", "1")
    _setdefault_env("TRANSFORMER_KEEP_CORE_IMAGES_ONLY", "1")


def _init_run_output_dir() -> str:
    """创建运行输出目录，按时间戳命名 (Create run output dir)."""
    global RUN_OUTPUT_DIR
    run_name = datetime.now().strftime("运行_%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_DIR, run_name)
    if os.path.exists(run_dir):
        suffix = 1
        while True:
            candidate = os.path.join(OUT_DIR, f"{run_name}_{suffix}")
            if not os.path.exists(candidate):
                run_dir = candidate
                break
            suffix += 1
    os.makedirs(run_dir, exist_ok=True)
    RUN_OUTPUT_DIR = run_dir
    return str(RUN_OUTPUT_DIR)


def _to_prefix_model_name(model_mode: str) -> str:
    """标准化模型名称前缀 (Normalize model name prefix)."""
    m = str(model_mode).strip().lower()
    if m == "attn":
        return "Attn"
    if m == "mlp":
        return "MLP"
    return "Dual"


def _get_objective_target() -> str:
    """获取优化目标 (val/test)."""
    t = str(
        _env_value(
            "TRANSFORMER_V7_OBJECTIVE_TARGET",
            "TRANSFORMER_V6_OBJECTIVE_TARGET",
            _env_value("TRANSFORMER_V5_OBJECTIVE_TARGET", "TRANSFORMER_V4_OBJECTIVE_TARGET", OBJECTIVE_TARGET),
        )
    ).strip().lower()
    if t not in {"val", "test"}:
        t = str(OBJECTIVE_TARGET)
    return t


def _reset_best_result_state(objective_target: str = "val") -> None:
    """重置全局最佳结果记录 (Reset best result state)."""
    BEST_RESULT.clear()
    BEST_RESULT.update(
        {
            "iter": 0,
            "objective_target": str(objective_target),
            "objective_r2": float("-inf"),
            "val_r2": float("-inf"),
            "val_rmse": float("nan"),
            "test_r2": float("nan"),
            "test_rmse": float("nan"),
            "train_r2": float("nan"),
            "train_rmse": float("nan"),
            "model_mode": "",
            "fusion_mode": "",
            "params": {},
        }
    )


def _safe_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """计算 R2 和 RMSE 指标 (Compute metrics)."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 2:
        return float("nan"), float("nan")
    return float(r2_score(y_true, y_pred)), float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _extract_subset_base_indices(subset, fallback_indices: np.ndarray) -> np.ndarray:
    """提取 Subset 的原始索引 (Extract original indices from Subset)."""
    if isinstance(subset, SelectedFeatureSubset):
        return np.asarray(subset.indices, dtype=int)
    if isinstance(subset, Subset) and hasattr(subset, "indices"):
        return np.asarray(subset.indices, dtype=int)
    return np.asarray(fallback_indices, dtype=int)


def _build_category_balanced_sampler(
    dataset: FingerprintReactionDataset,
    train_subset,
    train_idx: np.ndarray,
) -> Tuple[Optional[WeightedRandomSampler], Optional[Dict[str, object]]]:
    """
    构建类别平衡采样器 (Category-Balanced Sampler).
    解决数据中类别不平衡的问题，通过赋予稀有类别更高的采样权重。
    """
    enabled = _env_bool_pair("TRANSFORMER_V5_BALANCED_SAMPLER", "TRANSFORMER_V3_BALANCED_SAMPLER", False)
    if not enabled:
        return None, None

    base_indices = _extract_subset_base_indices(train_subset, train_idx)
    if base_indices.size < 20:
        return None, None
    try:
        cat_all = np.asarray(dataset.category).argmax(axis=1).astype(int)
        cats = cat_all[base_indices]
    except Exception:
        return None, None

    uniq, cnt = np.unique(cats, return_counts=True)
    if uniq.size < 2:
        return None, None

    # 计算类别权重: weight = 1 / sqrt(count)
    weight_by_cat = {}
    for c, n in zip(uniq.tolist(), cnt.tolist()):
        weight_by_cat[int(c)] = float(1.0 / max(np.sqrt(float(n)), 1.0))
    sample_w = np.asarray([weight_by_cat[int(c)] for c in cats.tolist()], dtype=np.float64)
    mean_w = float(np.mean(sample_w)) if sample_w.size > 0 else 1.0
    if mean_w > 0:
        sample_w = sample_w / mean_w
    # 限制权重范围，防止过度采样
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
    return sampler, info


def _resolve_chem_attention_model(model: nn.Module) -> Optional["ChemistryAwareAttentionTransformer"]:
    """从模型中解析出 ChemistryAwareAttentionTransformer 实例。"""
    if isinstance(model, ChemistryAwareAttentionTransformer):
        return model
    attn_expert = getattr(model, "attn_expert", None)
    if isinstance(attn_expert, ChemistryAwareAttentionTransformer):
        return attn_expert
    return None

def _save_v3_sampler_plot(output_dir: str, sampler_info: Optional[Dict[str, object]]) -> None:
    """绘制采样器权重分布图 (Plot sampler weight distribution)."""
    if not output_dir or not isinstance(sampler_info, dict):
        return
    stats = list(sampler_info.get("stats", []) or [])
    if not stats:
        return
    try:
        labels = [str(s.get("category_label", "")) for s in stats]
        counts = np.asarray([float(s.get("count", 0)) for s in stats], dtype=float)
        weights = np.asarray([float(s.get("sample_weight", 0)) for s in stats], dtype=float)
        x = np.arange(len(labels))
        fig, ax1 = plt.subplots(figsize=(max(7.5, len(labels) * 0.55), 4.5))
        ax1.bar(x, counts, color="#64748B", alpha=0.8, label="count")
        ax1.set_ylabel("Sample count")
        ax1.set_xticks(x, labels, rotation=25, ha="right")
        ax1.grid(axis="y", alpha=0.2)
        ax2 = ax1.twinx()
        ax2.plot(x, weights, color="#DC2626", marker="o", linewidth=2.0, label="sampling weight")
        ax2.set_ylabel("Sampler weight")
        ax1.set_title("V3 category-balanced sampler profile")
        fig.tight_layout()
        _savefig_with_pdf(fig, os.path.join(output_dir, "v3_category_sampler_profile.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def _save_v3_fusion_plot(output_dir: str, fusion_result: Optional[Dict[str, object]]) -> None:
    """绘制融合权重图 (Plot fusion weights)."""
    if not output_dir or not isinstance(fusion_result, dict):
        return
    try:
        weights = fusion_result.get("global_weights")
        if weights is None:
            weights = fusion_result.get("weights")
        if weights is None:
            return
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size == 0:
            return
        names = list(fusion_result.get("component_names", []) or [])
        if len(names) != w.size:
            names = [f"m{i + 1}" for i in range(w.size)]
        x = np.arange(len(names))
        plt.figure(figsize=(max(6.5, len(names) * 1.1), 4.2))
        cmap = plt.cm.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(names))]
        bars = plt.bar(x, w, color=colors)
        for b, val in zip(bars, w.tolist()):
            plt.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        plt.xticks(x, names, rotation=20, ha="right")
        plt.ylim(0.0, max(1.0, float(np.max(w) + 0.1)))
        plt.ylabel("Fusion weight")
        plt.title("V3 adaptive fusion weight profile")
        plt.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        _savefig_with_pdf(plt, os.path.join(output_dir, "v3_fusion_weight_profile.png"), dpi=300, bbox_inches="tight")
        plt.close()

    except Exception:
        pass


def _save_v3_chem_attention_bias_plot(
    output_dir: str,
    model: nn.Module,
    test_loader: DataLoader,
    max_batches: int = 12,
    max_tokens: int = 40,
) -> None:
    """绘制化学感知注意力偏置图 (Plot Chemistry-Aware Attention Bias Heatmap)."""
    if not output_dir:
        return
    attn_model = _resolve_chem_attention_model(model)
    if attn_model is None:
        return
    try:
        device = next(model.parameters()).device
        model.eval()
        sum_bias = None
        cnt_bias = None
        with torch.no_grad():
            for bi, batch in enumerate(test_loader):
                if bi >= max_batches:
                    break
                fp = batch["fingerprint"].to(device)
                numeric = batch["numeric"].to(device)
                if numeric.dim() == 1:
                    numeric = numeric.unsqueeze(-1)
                _ = model(fp, numeric)
                bias = getattr(attn_model, "last_attn_bias", None)
                key_padding_mask = getattr(attn_model, "last_key_padding_mask", None)
                if bias is None or key_padding_mask is None:
                    continue
                bias = bias.detach().to(torch.float32)
                keep = (~key_padding_mask).to(torch.float32)
                pair_keep = keep.unsqueeze(1) * keep.unsqueeze(2)
                b_sum = (bias * pair_keep).sum(dim=0).detach().cpu().numpy()
                b_cnt = pair_keep.sum(dim=0).detach().cpu().numpy()
                if sum_bias is None:
                    sum_bias = b_sum
                    cnt_bias = b_cnt
                else:
                    sum_bias += b_sum
                    cnt_bias += b_cnt
        if sum_bias is None or cnt_bias is None:
            return
        avg_bias = sum_bias / np.maximum(cnt_bias, 1e-6)
        n = int(min(max_tokens, avg_bias.shape[0]))
        if n < 4:
            return
        mat = avg_bias[:n, :n]
        plt.figure(figsize=(6.8, 5.8))
        plt.imshow(mat, cmap="coolwarm", aspect="auto")
        plt.colorbar(label="Avg additive attention bias")
        ticks = np.arange(0, n, max(1, n // 8))
        tick_labels = ["CLS" if i == 0 else f"T{i}" for i in ticks.tolist()]
        plt.xticks(ticks, tick_labels, rotation=0)
        plt.yticks(ticks, tick_labels)
        plt.title("V3 chemistry-aware attention bias map")
        plt.xlabel("Key token")
        plt.ylabel("Query token")
        plt.tight_layout()
        _savefig_with_pdf(plt, os.path.join(output_dir, "v3_chem_attention_bias_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def _save_v3_innovation_artifacts(
    output_dir: Optional[str],
    model: nn.Module,
    test_loader: DataLoader,
    epoch_history: list,
    top_state_records: list,
    sampler_info: Optional[Dict[str, object]],
    fusion_result: Optional[Dict[str, object]],
) -> None:
    """保存 V3/V5 创新特性的相关图表和工件 (Save innovation artifacts)."""
    if not output_dir:
        return
    try:
        # 保存采样器分布图
        if sampler_info:
            _save_v3_sampler_plot(output_dir, sampler_info)
        # 保存融合权重图
        if fusion_result:
            _save_v3_fusion_plot(output_dir, fusion_result)
        # 保存化学注意力偏置热图
        _save_v3_chem_attention_bias_plot(output_dir, model, test_loader)
        
        # 保存训练历史数据
        if epoch_history:
            hist_csv = os.path.join(output_dir, "training_history.csv")
            pd.DataFrame(epoch_history).to_csv(hist_csv, index=False)
            
            # 绘制训练曲线
            try:
                df = pd.DataFrame(epoch_history)
                plt.figure(figsize=(10, 6))
                plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
                if "val_r2" in df.columns:
                    plt.plot(df["epoch"], df["val_r2"], label="Val R2")
                plt.xlabel("Epoch")
                plt.ylabel("Metric")
                plt.title("Training History")
                plt.legend()
                plt.grid(True, alpha=0.3)
                _savefig_with_pdf(plt, os.path.join(output_dir, "training_history_plot.png"))
                plt.close()
            except Exception:
                pass

    except Exception as e:
        print(f"[V5] Warning: Failed to save innovation artifacts: {e}")


def _get_base_dataset(ds):
    """获取基础数据集对象 (Get base dataset)."""
    if isinstance(ds, SelectedFeatureSubset):
        return ds.base
    if isinstance(ds, Subset) and hasattr(ds, "dataset"):
        return ds.dataset
    return ds


def _predict_raw(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    return_indices: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    基础预测函数 (Basic prediction loop).
    返回: (y_true, y_pred) 或 (y_true, y_pred, indices)
    """
    model.eval()
    base_ds = _get_base_dataset(dataloader.dataset)

    y_true_list = []
    y_pred_list = []
    idx_list = []
    non_blocking = str(device).lower().startswith("cuda")

    with torch.no_grad():
        for batch in dataloader:
            fp = batch["fingerprint"].to(device, non_blocking=non_blocking)
            numeric = batch["numeric"].to(device, non_blocking=non_blocking)
            y_raw = batch["logk_raw"].detach().cpu().numpy().reshape(-1)

            # 模型前向传播
            preds_scaled = model(fp, numeric)
            # 反归一化预测值
            preds_scaled_np = preds_scaled.detach().to(torch.float32).cpu().numpy().reshape(-1, 1)
            preds_raw = base_ds.logk_scaler.inverse_transform(preds_scaled_np).reshape(-1)

            base_idx = batch.get("base_idx")
            if isinstance(base_idx, torch.Tensor):
                idx_np = base_idx.detach().cpu().numpy().reshape(-1)
            elif base_idx is None:
                idx_np = np.full((y_raw.shape[0],), -1, dtype=int)
            else:
                idx_np = np.asarray(base_idx).reshape(-1)

            y_true_list.append(y_raw)
            y_pred_list.append(preds_raw)
            idx_list.append(idx_np.astype(int, copy=False))

    if not y_true_list:
        empty = np.asarray([], dtype=np.float32)
        if return_indices:
            return empty, empty, np.asarray([], dtype=int)
        return empty, empty

    y_true = np.concatenate(y_true_list).astype(np.float32)
    y_pred = np.concatenate(y_pred_list).astype(np.float32)
    idx = np.concatenate(idx_list).astype(int, copy=False)
    
    # 移除无效值
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    idx = idx[mask]

    # 按索引排序以确保一致性
    if idx.size > 0 and np.all(idx >= 0):
        order = np.argsort(idx, kind="stable")
        y_true = y_true[order]
        y_pred = y_pred[order]
        idx = idx[order]

    if return_indices:
        return y_true, y_pred, idx
    return y_true, y_pred


def _predict_split_triplet(
    model: nn.Module,
    device: str,
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """一次性预测 Train/Val/Test 三个数据集 (Predict all splits)."""
    y_train, pred_train, idx_train = _predict_raw(model, train_eval_loader, device, return_indices=True)
    y_val, pred_val, idx_val = _predict_raw(model, val_loader, device, return_indices=True)
    y_test, pred_test, idx_test = _predict_raw(model, test_loader, device, return_indices=True)
    return y_train, pred_train, idx_train, y_val, pred_val, idx_val, y_test, pred_test, idx_test


def _softmax_weights(scores: list, temperature: float = 0.02) -> np.ndarray:
    """Softmax 加权函数 (Softmax weighting)."""
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        return np.ones(arr.shape[0], dtype=float) / float(arr.shape[0])
    t = float(max(1e-4, temperature))
    # 减去最大值以防止溢出
    z = (arr - float(np.max(arr))) / t
    z = np.clip(z, -60.0, 60.0)
    w = np.exp(z)
    s = float(np.sum(w))
    if s <= 0.0 or (not np.isfinite(s)):
        return np.ones(arr.shape[0], dtype=float) / float(arr.shape[0])
    return w / s


def _build_category_physics_meta(
    dataset: FingerprintReactionDataset,
    train_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    """V6 经验 pH 先验元数据：方向、置信度、有效 pH 范围与统计量。"""
    cat_mat = np.asarray(dataset.category, dtype=np.float32)
    if cat_mat.ndim != 2 or cat_mat.shape[1] <= 0:
        zeros = np.zeros((1,), dtype=np.float32)
        return {
            "sign": zeros.copy(),
            "target_slope": zeros.copy(),
            "confidence": zeros.copy(),
            "ph_lo": np.asarray([-1.0], dtype=np.float32),
            "ph_hi": np.asarray([1.0], dtype=np.float32),
            "slope": zeros.copy(),
            "corr": zeros.copy(),
            "valid": zeros.copy(),
        }
    n_cat = int(cat_mat.shape[1])
    cat_idx_all = np.argmax(cat_mat, axis=1).astype(int)
    n_samples = int(cat_idx_all.shape[0])
    ph_raw = np.asarray(dataset.ph, dtype=float).reshape(-1)
    ph_scaled = np.asarray(getattr(dataset, "ph_scaled", dataset.ph), dtype=float).reshape(-1)
    y_raw = np.asarray(dataset.logk_raw, dtype=float).reshape(-1)
    idx = np.asarray(train_idx, dtype=int).reshape(-1)
    idx = idx[(idx >= 0) & (idx < cat_idx_all.shape[0])]
    
    min_samples = int(_env_value("TRANSFORMER_V4_PHYSICS_MIN_SAMPLES", "TRANSFORMER_V3_PHYSICS_MIN_SAMPLES", "18"))
    corr_thr = float(_env_value("TRANSFORMER_V4_PHYSICS_CORR_THR", "TRANSFORMER_V3_PHYSICS_CORR_THR", "0.12"))
    slope_thr = float(_env_value("TRANSFORMER_V4_PHYSICS_SLOPE_THR", "TRANSFORMER_V3_PHYSICS_SLOPE_THR", "0.015"))
    span_thr = float(_env_value("TRANSFORMER_V7_PHYSICS_SPAN_THR", "TRANSFORMER_V6_PHYSICS_SPAN_THR", "0.35"))
    conf_span_ref = float(_env_value("TRANSFORMER_V7_PHYSICS_CONF_SPAN_REF", "TRANSFORMER_V6_PHYSICS_CONF_SPAN_REF", "2.00"))
    within_smiles_prior = _env_bool_pair("TRANSFORMER_V7_PHYSICS_WITHIN_SMILES", "TRANSFORMER_V6_PHYSICS_WITHIN_SMILES", False)
    within_min_groups = int(_env_value("TRANSFORMER_V7_PHYSICS_WITHIN_MIN_GROUPS", "TRANSFORMER_V6_PHYSICS_WITHIN_MIN_GROUPS", "5"))
    within_min_agreement = float(_env_value("TRANSFORMER_V7_PHYSICS_WITHIN_MIN_AGREEMENT", "TRANSFORMER_V6_PHYSICS_WITHIN_MIN_AGREEMENT", "0.65"))
    fg_state_prior = _env_bool_pair("TRANSFORMER_V7_PHYSICS_FG_STATE", "TRANSFORMER_V6_PHYSICS_FG_STATE", False)
    fg_state_width = float(_env_value("TRANSFORMER_V7_PHYSICS_FG_WIDTH", "TRANSFORMER_V6_PHYSICS_FG_WIDTH", "1.35"))
    fg_state_conf = float(_env_value("TRANSFORMER_V7_PHYSICS_FG_CONF", "TRANSFORMER_V6_PHYSICS_FG_CONF", "0.38"))
    
    sign_prior = np.zeros((n_cat,), dtype=np.float32)
    target_slope_prior = np.zeros((n_cat,), dtype=np.float32)
    conf_prior = np.zeros((n_cat,), dtype=np.float32)
    slope_prior = np.zeros((n_cat,), dtype=np.float32)
    corr_prior = np.zeros((n_cat,), dtype=np.float32)
    valid_prior = np.zeros((n_cat,), dtype=np.float32)
    ph_lo = np.full((n_cat,), -1.0, dtype=np.float32)
    ph_hi = np.full((n_cat,), 1.0, dtype=np.float32)
    sample_sign = np.zeros((n_samples,), dtype=np.float32)
    sample_target_slope = np.zeros((n_samples,), dtype=np.float32)
    sample_conf = np.zeros((n_samples,), dtype=np.float32)
    sample_valid = np.zeros((n_samples,), dtype=np.float32)
    sample_ph_lo = np.full((n_samples,), -1.0, dtype=np.float32)
    sample_ph_hi = np.full((n_samples,), 1.0, dtype=np.float32)
    if idx.size == 0:
        return {
            "sign": sign_prior,
            "target_slope": target_slope_prior,
            "confidence": conf_prior,
            "ph_lo": ph_lo,
            "ph_hi": ph_hi,
            "slope": slope_prior,
            "corr": corr_prior,
            "valid": valid_prior,
            "sample_sign": sample_sign,
            "sample_target_slope": sample_target_slope,
            "sample_confidence": sample_conf,
            "sample_ph_lo": sample_ph_lo,
            "sample_ph_hi": sample_ph_hi,
            "sample_valid": sample_valid,
        }
    if ph_scaled[idx].size > 0:
        global_lo = float(np.quantile(ph_scaled[idx], 0.05))
        global_hi = float(np.quantile(ph_scaled[idx], 0.95))
        ph_lo[:] = global_lo
        ph_hi[:] = max(global_hi, global_lo + 1e-3)

    ph_scale = 1.0
    y_scale = 1.0
    try:
        ph_scale = float(np.asarray(dataset.ph_scaler.scale_, dtype=float).reshape(-1)[0])
    except Exception:
        ph_scale = 1.0
    try:
        y_scale = float(np.asarray(dataset.logk_scaler.scale_, dtype=float).reshape(-1)[0])
    except Exception:
        y_scale = 1.0
    if not np.isfinite(ph_scale) or abs(ph_scale) <= 1e-12:
        ph_scale = 1.0
    if not np.isfinite(y_scale) or abs(y_scale) <= 1e-12:
        y_scale = 1.0

    use_sample_smiles_prior = _env_bool_pair(
        "TRANSFORMER_V7_PHYSICS_SAMPLE_SMILES",
        "TRANSFORMER_V6_PHYSICS_SAMPLE_SMILES",
        within_smiles_prior,
    )
    sample_local_slope = _env_bool_pair(
        "TRANSFORMER_V7_PHYSICS_SAMPLE_LOCAL_SLOPE",
        "TRANSFORMER_V6_PHYSICS_SAMPLE_LOCAL_SLOPE",
        False,
    )
    sample_local_bw = float(_env_value("TRANSFORMER_V7_PHYSICS_SAMPLE_LOCAL_BW", "TRANSFORMER_V6_PHYSICS_SAMPLE_LOCAL_BW", "1.20"))
    sample_local_blend = float(_env_value("TRANSFORMER_V7_PHYSICS_SAMPLE_LOCAL_BLEND", "TRANSFORMER_V6_PHYSICS_SAMPLE_LOCAL_BLEND", "0.65"))
    sample_local_guard = _env_bool_pair(
        "TRANSFORMER_V7_PHYSICS_SAMPLE_LOCAL_SIGN_GUARD",
        "TRANSFORMER_V6_PHYSICS_SAMPLE_LOCAL_SIGN_GUARD",
        True,
    )
    if use_sample_smiles_prior:
        smiles_all = np.asarray(getattr(dataset, "smiles", np.asarray([""] * n_samples)), dtype=object)
        grouped_all: Dict[str, list] = {}
        for row_idx in idx.tolist():
            grouped_all.setdefault(str(smiles_all[int(row_idx)]), []).append(int(row_idx))
        for group_indices in grouped_all.values():
            if len(group_indices) < 2:
                continue
            gx = ph_raw[group_indices].astype(float)
            gy = y_raw[group_indices].astype(float)
            gspan = float(np.max(gx) - np.min(gx)) if gx.size > 0 else 0.0
            if gx.size < 2 or gspan < max(0.10, 0.5 * span_thr):
                continue
            gx0 = gx - float(np.mean(gx))
            gy0 = gy - float(np.mean(gy))
            den = float(np.sum(gx0 * gx0))
            if den <= 1e-10:
                continue
            gs = float(np.sum(gx0 * gy0) / (den + 1e-12))
            if not np.isfinite(gs) or abs(gs) < slope_thr:
                continue
            scaled_slope = float(gs * ph_scale / y_scale)
            gidx = np.asarray(group_indices, dtype=int)
            sample_sign[gidx] = 1.0 if gs > 0.0 else -1.0
            sample_target_slope[gidx] = scaled_slope
            sample_valid[gidx] = 1.0
            support_score = min(1.0, float(len(group_indices)) / 4.0)
            span_score = min(1.0, max(0.0, (gspan - span_thr * 0.5) / max(1e-6, conf_span_ref - span_thr * 0.5)))
            slope_score = min(1.0, max(0.0, (abs(gs) - slope_thr) / max(1e-6, 4.0 * slope_thr)))
            base_conf = float(0.40 * support_score + 0.35 * span_score + 0.25 * slope_score)
            sample_conf[gidx] = base_conf
            gscaled = ph_scaled[gidx].astype(float)
            if gscaled.size > 0:
                lo = float(np.min(gscaled))
                hi = float(np.max(gscaled))
                if np.isfinite(lo) and np.isfinite(hi):
                    sample_ph_lo[gidx] = lo
                    sample_ph_hi[gidx] = max(hi, lo + 1e-3)
            if sample_local_slope:
                bw = max(0.15, sample_local_bw)
                blend = float(max(0.0, min(1.0, sample_local_blend)))
                for local_pos, row_idx in enumerate(group_indices):
                    center = float(gx[local_pos])
                    dist = gx - center
                    weights = np.exp(-0.5 * (dist / bw) ** 2).astype(float)
                    if weights.size <= 0 or float(np.sum(weights)) <= 1e-10:
                        continue
                    lx0 = gx - float(np.sum(weights * gx) / max(1e-12, float(np.sum(weights))))
                    ly0 = gy - float(np.sum(weights * gy) / max(1e-12, float(np.sum(weights))))
                    lden = float(np.sum(weights * lx0 * lx0))
                    local_slope = gs
                    if lden > 1e-10:
                        candidate_slope = float(np.sum(weights * lx0 * ly0) / (lden + 1e-12))
                        if np.isfinite(candidate_slope):
                            if sample_local_guard and candidate_slope * gs < 0.0:
                                local_slope = gs
                            else:
                                local_slope = (1.0 - blend) * gs + blend * candidate_slope
                    if not np.isfinite(local_slope) or abs(local_slope) < slope_thr:
                        local_slope = gs
                    row_idx = int(row_idx)
                    scaled_local = float(local_slope * ph_scale / y_scale)
                    sample_sign[row_idx] = 1.0 if local_slope > 0.0 else -1.0
                    sample_target_slope[row_idx] = scaled_local
                    neff = float((np.sum(weights) ** 2) / max(1e-12, np.sum(weights * weights)))
                    local_support_score = min(1.0, neff / 3.0)
                    sample_conf[row_idx] = float(max(0.05, base_conf * (0.70 + 0.30 * local_support_score)))
                    try:
                        lo_scaled = float(dataset.ph_scaler.transform(np.asarray([[center - 1.5 * bw]], dtype=float))[0, 0])
                        hi_scaled = float(dataset.ph_scaler.transform(np.asarray([[center + 1.5 * bw]], dtype=float))[0, 0])
                    except Exception:
                        lo_scaled, hi_scaled = sample_ph_lo[row_idx], sample_ph_hi[row_idx]
                    if np.isfinite(lo_scaled) and np.isfinite(hi_scaled):
                        sample_ph_lo[row_idx] = min(lo_scaled, hi_scaled)
                        sample_ph_hi[row_idx] = max(max(lo_scaled, hi_scaled), min(lo_scaled, hi_scaled) + 1e-3)

    if _env_bool_pair("TRANSFORMER_V7_PHYSICS_KNN_PROPAGATE", "TRANSFORMER_V6_PHYSICS_KNN_PROPAGATE", False):
        train_idx_arr = np.asarray(idx, dtype=int)
        anchor_idx = train_idx_arr[sample_valid[train_idx_arr] > 0.5]
        target_idx = train_idx_arr[sample_valid[train_idx_arr] <= 0.5]
        if anchor_idx.size > 0 and target_idx.size > 0:
            fp_all = np.asarray(getattr(dataset, "fingerprint", np.zeros((n_samples, 0), dtype=np.float32)), dtype=np.float32)
            if fp_all.ndim == 2 and fp_all.shape[0] == n_samples and fp_all.shape[1] > 0:
                fp_bool = fp_all > 0.5
                anchor_fp = fp_bool[anchor_idx]
                anchor_sum = anchor_fp.sum(axis=1).astype(np.float32)
                anchor_slope = sample_target_slope[anchor_idx].astype(np.float32)
                anchor_sign = sample_sign[anchor_idx].astype(np.float32)
                anchor_conf = sample_conf[anchor_idx].astype(np.float32)
                anchor_cat = cat_idx_all[anchor_idx]
                k_nn = int(_env_value("TRANSFORMER_V7_PHYSICS_KNN_K", "TRANSFORMER_V6_PHYSICS_KNN_K", "7"))
                sim_thr = float(_env_value("TRANSFORMER_V7_PHYSICS_KNN_SIM_THR", "TRANSFORMER_V6_PHYSICS_KNN_SIM_THR", "0.22"))
                agreement_thr = float(_env_value("TRANSFORMER_V7_PHYSICS_KNN_AGREE_THR", "TRANSFORMER_V6_PHYSICS_KNN_AGREE_THR", "0.65"))
                same_cat_only = _env_bool_pair("TRANSFORMER_V7_PHYSICS_KNN_SAME_CAT", "TRANSFORMER_V6_PHYSICS_KNN_SAME_CAT", True)
                for row_idx in target_idx.tolist():
                    x_fp = fp_bool[int(row_idx)]
                    x_sum = float(np.sum(x_fp))
                    if x_sum <= 0.0:
                        continue
                    inter = np.logical_and(anchor_fp, x_fp).sum(axis=1).astype(np.float32)
                    union = anchor_sum + x_sum - inter
                    sim = inter / np.maximum(union, 1.0)
                    if same_cat_only:
                        sim = np.where(anchor_cat == int(cat_idx_all[int(row_idx)]), sim, 0.0)
                    valid_neighbor = sim >= sim_thr
                    if not np.any(valid_neighbor):
                        continue
                    cand = np.where(valid_neighbor)[0]
                    order = cand[np.argsort(sim[cand])[::-1]]
                    order = order[: max(1, k_nn)]
                    if order.size <= 0:
                        continue
                    signs = anchor_sign[order]
                    weights = sim[order] * np.clip(anchor_conf[order], 1e-3, 1.0)
                    if float(np.sum(weights)) <= 1e-8:
                        continue
                    pos_w = float(np.sum(weights[signs > 0.0]))
                    neg_w = float(np.sum(weights[signs < 0.0]))
                    total_w = pos_w + neg_w
                    if total_w <= 1e-8:
                        continue
                    agreement = max(pos_w, neg_w) / total_w
                    if agreement < agreement_thr:
                        continue
                    slope = float(np.sum(weights * anchor_slope[order]) / np.sum(weights))
                    if not np.isfinite(slope) or abs(slope) < 1e-5:
                        continue
                    sample_target_slope[int(row_idx)] = slope
                    sample_sign[int(row_idx)] = 1.0 if slope > 0.0 else -1.0
                    sample_valid[int(row_idx)] = 1.0
                    sim_score = float(np.max(sim[order]))
                    conf_score = float(np.sum(weights) / max(1e-6, np.sum(sim[order])))
                    sample_conf[int(row_idx)] = float(
                        min(0.85, 0.45 * agreement + 0.35 * min(1.0, sim_score / max(sim_thr, 1e-6)) + 0.20 * min(1.0, conf_score))
                    )
                    src = anchor_idx[order]
                    lo = float(np.min(sample_ph_lo[src]))
                    hi = float(np.max(sample_ph_hi[src]))
                    if np.isfinite(lo) and np.isfinite(hi):
                        sample_ph_lo[int(row_idx)] = lo
                        sample_ph_hi[int(row_idx)] = max(hi, lo + 1e-3)

    if fg_state_prior:
        smiles_all = np.asarray(getattr(dataset, "smiles", np.asarray([""] * n_samples)), dtype=object)
        fg_width = float(max(0.25, fg_state_width))
        fg_base_conf = float(max(0.05, min(0.95, fg_state_conf)))
        for row_idx in idx.tolist():
            row_idx = int(row_idx)
            if sample_valid[row_idx] > 0.5:
                continue
            hits = _ionizable_physics_hits(str(smiles_all[row_idx]))
            if not hits:
                continue
            ph_val = float(ph_raw[row_idx]) if np.isfinite(ph_raw[row_idx]) else 7.0
            scored = []
            for rule in hits:
                pka = float(rule.get("pka", 7.0))
                window = float(np.exp(-0.5 * ((ph_val - pka) / fg_width) ** 2))
                if window < 0.08:
                    continue
                raw_slope = float(rule.get("slope", 0.06)) * float(rule.get("sign", 1.0))
                scored.append((window, raw_slope, pka))
            if not scored:
                continue
            weights = np.asarray([s[0] for s in scored], dtype=float)
            raw_slopes = np.asarray([s[1] for s in scored], dtype=float)
            pka_vals = np.asarray([s[2] for s in scored], dtype=float)
            raw_slope = float(np.sum(weights * raw_slopes) / max(1e-8, float(np.sum(weights))))
            if not np.isfinite(raw_slope) or abs(raw_slope) < 1e-6:
                continue
            scaled_slope = float(raw_slope * ph_scale / y_scale)
            sample_sign[row_idx] = 1.0 if scaled_slope > 0.0 else -1.0
            sample_target_slope[row_idx] = scaled_slope
            sample_valid[row_idx] = 1.0
            sample_conf[row_idx] = float(min(0.80, fg_base_conf * min(1.0, float(np.max(weights)))))
            pka_center = float(np.sum(weights * pka_vals) / max(1e-8, float(np.sum(weights))))
            lo_raw = pka_center - 1.75 * fg_width
            hi_raw = pka_center + 1.75 * fg_width
            try:
                lo_scaled = float(dataset.ph_scaler.transform(np.asarray([[lo_raw]], dtype=float))[0, 0])
                hi_scaled = float(dataset.ph_scaler.transform(np.asarray([[hi_raw]], dtype=float))[0, 0])
            except Exception:
                lo_scaled, hi_scaled = -1.0, 1.0
            sample_ph_lo[row_idx] = min(lo_scaled, hi_scaled)
            sample_ph_hi[row_idx] = max(max(lo_scaled, hi_scaled), min(lo_scaled, hi_scaled) + 1e-3)
        
    for c in range(n_cat):
        c_idx = idx[cat_idx_all[idx] == c]
        if c_idx.size < min_samples:
            continue
        x = ph_raw[c_idx].astype(float)
        y = y_raw[c_idx].astype(float)
        
        # 简单线性回归估计斜率
        x0 = x - float(np.mean(x))
        y0 = y - float(np.mean(y))
        var_x = float(np.sum(x0 * x0))
        if var_x <= 1e-10:
            continue
        slope = float(np.sum(x0 * y0) / (var_x + 1e-12))
        
        try:
            corr = float(np.corrcoef(x, y)[0, 1])
        except Exception:
            corr = 0.0

        slope_prior[c] = float(slope) if np.isfinite(slope) else 0.0
        corr_prior[c] = float(corr) if np.isfinite(corr) else 0.0
        if np.isfinite(slope):
            target_slope_prior[c] = float(slope * ph_scale / y_scale)
        x_scaled = ph_scaled[c_idx].astype(float)
        if x_scaled.size > 0:
            lo = float(np.quantile(x_scaled, 0.05))
            hi = float(np.quantile(x_scaled, 0.95))
            if np.isfinite(lo) and np.isfinite(hi):
                ph_lo[c] = lo
                ph_hi[c] = max(hi, lo + 1e-3)
        ph_span = float(np.max(x) - np.min(x)) if x.size > 0 else 0.0

        if within_smiles_prior:
            grouped: Dict[str, list] = {}
            smiles_all = np.asarray(getattr(dataset, "smiles", np.asarray([""] * cat_idx_all.shape[0])), dtype=object)
            for row_idx in c_idx.tolist():
                grouped.setdefault(str(smiles_all[int(row_idx)]), []).append(int(row_idx))
            within_slopes = []
            for group_indices in grouped.values():
                if len(group_indices) < 2:
                    continue
                gx = ph_raw[group_indices].astype(float)
                gy = y_raw[group_indices].astype(float)
                if gx.size < 2 or float(np.max(gx) - np.min(gx)) < max(0.10, 0.5 * span_thr):
                    continue
                gx0 = gx - float(np.mean(gx))
                gy0 = gy - float(np.mean(gy))
                den = float(np.sum(gx0 * gx0))
                if den <= 1e-10:
                    continue
                gs = float(np.sum(gx0 * gy0) / (den + 1e-12))
                if np.isfinite(gs):
                    within_slopes.append(gs)
            if len(within_slopes) >= max(1, within_min_groups):
                ws = np.asarray(within_slopes, dtype=float)
                med_slope = float(np.median(ws))
                pos_frac = float(np.mean(ws > 0.0))
                neg_frac = float(np.mean(ws < 0.0))
                agreement = max(pos_frac, neg_frac)
                if abs(med_slope) >= slope_thr and agreement >= within_min_agreement:
                    sign_prior[c] = 1.0 if med_slope > 0.0 else -1.0
                    slope_prior[c] = med_slope
                    target_slope_prior[c] = float(med_slope * ph_scale / y_scale)
                    corr_prior[c] = float(2.0 * agreement - 1.0) * float(sign_prior[c])
                    support_score = min(1.0, float(len(ws)) / float(max(within_min_groups * 3, 1)))
                    agreement_score = min(1.0, max(0.0, (agreement - within_min_agreement) / max(1e-6, 1.0 - within_min_agreement)))
                    slope_score = min(1.0, max(0.0, (abs(med_slope) - slope_thr) / max(1e-6, 4.0 * slope_thr)))
                    conf_prior[c] = float(0.45 * agreement_score + 0.35 * support_score + 0.20 * slope_score)
                    valid_prior[c] = 1.0
                    continue

        # 只有当相关性和斜率都足够显著时才认为存在先验
        if (
            np.isfinite(slope)
            and np.isfinite(corr)
            and (abs(corr) >= corr_thr)
            and (abs(slope) >= slope_thr)
            and (ph_span >= span_thr)
        ):
            sign_prior[c] = 1.0 if slope > 0.0 else -1.0
            count_score = min(1.0, float(c_idx.size) / float(max(min_samples * 2, 1)))
            corr_score = min(1.0, max(0.0, (abs(corr) - corr_thr) / max(1e-6, 1.0 - corr_thr)))
            slope_score = min(1.0, max(0.0, (abs(slope) - slope_thr) / max(1e-6, 4.0 * slope_thr)))
            span_score = min(1.0, max(0.0, (ph_span - span_thr) / max(1e-6, conf_span_ref - span_thr)))
            conf_prior[c] = float(0.45 * corr_score + 0.30 * slope_score + 0.15 * count_score + 0.10 * span_score)
            valid_prior[c] = 1.0

    return {
        "sign": sign_prior.astype(np.float32, copy=False),
        "target_slope": target_slope_prior.astype(np.float32, copy=False),
        "confidence": conf_prior.astype(np.float32, copy=False),
        "ph_lo": ph_lo.astype(np.float32, copy=False),
        "ph_hi": ph_hi.astype(np.float32, copy=False),
        "slope": slope_prior.astype(np.float32, copy=False),
        "corr": corr_prior.astype(np.float32, copy=False),
        "valid": valid_prior.astype(np.float32, copy=False),
        "sample_sign": sample_sign.astype(np.float32, copy=False),
        "sample_target_slope": sample_target_slope.astype(np.float32, copy=False),
        "sample_confidence": sample_conf.astype(np.float32, copy=False),
        "sample_ph_lo": sample_ph_lo.astype(np.float32, copy=False),
        "sample_ph_hi": sample_ph_hi.astype(np.float32, copy=False),
        "sample_valid": sample_valid.astype(np.float32, copy=False),
    }


def _build_category_physics_prior(
    dataset: FingerprintReactionDataset,
    train_idx: np.ndarray,
) -> np.ndarray:
    """兼容旧接口，仅返回方向先验。"""
    return _build_category_physics_meta(dataset, train_idx)["sign"]


def _physics_meta_to_tensors(meta: Optional[Dict[str, np.ndarray]], device: str) -> Dict[str, torch.Tensor]:
    """将经验先验元数据转换为训练时使用的张量。"""
    if not isinstance(meta, dict):
        return {}
    out: Dict[str, torch.Tensor] = {}
    for key, value in meta.items():
        arr = np.asarray(value, dtype=np.float32)
        out[key] = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return out


def _group_mse_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    category_onehot: torch.Tensor,
) -> torch.Tensor:
    """计算每个类别的平均 MSE 损失，用于 DRO (Distributionally Robust Optimization)。"""
    cat_idx = torch.argmax(category_onehot, dim=1)
    uniq = torch.unique(cat_idx)
    losses = []
    for g in uniq:
        mask = cat_idx == g
        if torch.sum(mask).item() < 2:
            continue
        losses.append(F.mse_loss(pred[mask], target[mask], reduction="mean"))
    if not losses:
        return torch.empty((0,), device=pred.device, dtype=pred.dtype)
    return torch.stack(losses)


def _train_epoch_causal_invariant(
    trainer: FingerprintTrainer,
    dataloader: DataLoader,
    physics_sign_prior: Optional[torch.Tensor] = None,
    physics_meta: Optional[Dict[str, torch.Tensor]] = None,
    epoch: int = 1,
    max_epochs: int = 1,
) -> Dict[str, float]:
    trainer.model.train()
    device = str(trainer.device)
    non_blocking = device.lower().startswith("cuda")
    use_amp = bool(getattr(trainer, "use_amp", False) and non_blocking)
    amp_dtype = getattr(trainer, "amp_dtype", torch.float16)
    scaler = getattr(trainer, "scaler", None)

    lam_inv = float(_env_value("TRANSFORMER_V4_LAMBDA_INV", "TRANSFORMER_V3_LAMBDA_INV", "0.18"))
    lam_dro = float(_env_value("TRANSFORMER_V4_LAMBDA_DRO", "TRANSFORMER_V3_LAMBDA_DRO", "0.28"))
    dro_tau = float(_env_value("TRANSFORMER_V4_DRO_TAU", "TRANSFORMER_V3_DRO_TAU", "0.05"))
    lam_phys = float(_env_value("TRANSFORMER_V4_LAMBDA_PHYSICS", "TRANSFORMER_V3_LAMBDA_PHYSICS", "0.08"))
    physics_delta = float(_env_value("TRANSFORMER_V4_PHYSICS_DELTA", "TRANSFORMER_V3_PHYSICS_DELTA", "0.04"))
    phys_smooth_w = float(_env_value("TRANSFORMER_V4_PHYSICS_SMOOTH_W", "TRANSFORMER_V3_PHYSICS_SMOOTH_W", "0.45"))
    phys_mono_w = float(_env_value("TRANSFORMER_V4_PHYSICS_MONO_W", "TRANSFORMER_V3_PHYSICS_MONO_W", "0.55"))
    physics_mode = str(_env_value("TRANSFORMER_V7_PHYSICS_MODE", "TRANSFORMER_V6_PHYSICS_MODE", "finite_diff")).strip().lower()
    physics_conf_min = float(_env_value("TRANSFORMER_V7_PHYSICS_CONF_MIN", "TRANSFORMER_V6_PHYSICS_CONF_MIN", "0.25"))
    physics_colloc_w = float(_env_value("TRANSFORMER_V7_PHYSICS_COLLOC_W", "TRANSFORMER_V6_PHYSICS_COLLOC_W", "0.35"))
    physics_use_colloc = _env_bool_pair("TRANSFORMER_V7_PHYSICS_ENABLE_COLLOCATION", "TRANSFORMER_V6_PHYSICS_ENABLE_COLLOCATION", False)
    physics_fd_conf_gated = _env_bool_pair("TRANSFORMER_V7_PHYSICS_FINITEDIFF_CONF_GATED", "TRANSFORMER_V6_PHYSICS_FINITEDIFF_CONF_GATED", False)
    physics_fd_range_gated = _env_bool_pair("TRANSFORMER_V7_PHYSICS_FINITEDIFF_RANGE_GATED", "TRANSFORMER_V6_PHYSICS_FINITEDIFF_RANGE_GATED", False)
    physics_fd_smooth_floor = float(_env_value("TRANSFORMER_V7_PHYSICS_FINITEDIFF_SMOOTH_FLOOR", "TRANSFORMER_V6_PHYSICS_FINITEDIFF_SMOOTH_FLOOR", "0.15"))
    physics_slope_target_w = float(_env_value("TRANSFORMER_V7_PHYSICS_SLOPE_TARGET_W", "TRANSFORMER_V6_PHYSICS_SLOPE_TARGET_W", "0.70"))
    physics_slope_clip = float(_env_value("TRANSFORMER_V7_PHYSICS_SLOPE_CLIP", "TRANSFORMER_V6_PHYSICS_SLOPE_CLIP", "2.50"))
    physics_pinn_curv_w = float(_env_value("TRANSFORMER_V8_PHYSICS_CURVATURE_W", "TRANSFORMER_V7_PHYSICS_CURVATURE_W", "0.0"))
    physics_pinn_curv_conf_min = float(_env_value("TRANSFORMER_V8_PHYSICS_CURVATURE_CONF_MIN", "TRANSFORMER_V7_PHYSICS_CURVATURE_CONF_MIN", "0.18"))
    physics_safe_target = str(
        _env_value("TRANSFORMER_V9_PHYSICS_SAFE_TARGET", None, "attn")
    ).strip().lower()
    if physics_safe_target not in {"final", "attn"}:
        physics_safe_target = "attn"
    physics_warmup_frac = float(_env_value("TRANSFORMER_V9_PHYSICS_WARMUP_FRAC", None, "0.30"))
    physics_warmup_frac = float(max(0.0, min(0.95, physics_warmup_frac)))
    physics_start_frac = float(_env_value("TRANSFORMER_V9_PHYSICS_START_FRAC", None, "0.0"))
    physics_start_frac = float(max(0.0, min(0.95, physics_start_frac)))
    physics_max_gate = float(_env_value("TRANSFORMER_V9_PHYSICS_MAX_GATE", None, "1.0"))
    physics_max_gate = float(max(0.0, min(1.0, physics_max_gate)))
    physics_conf_power = float(_env_value("TRANSFORMER_V9_PHYSICS_CONF_POWER", None, "1.0"))
    physics_conf_power = float(max(0.25, min(4.0, physics_conf_power)))
    start_epoch = float(max_epochs) * physics_start_frac
    warmup_epochs = max(1.0, float(max_epochs) * max(1e-6, physics_warmup_frac))
    physics_epoch_gate = float(min(1.0, max(0.0, (float(epoch) - start_epoch) / warmup_epochs))) * physics_max_gate
    use_physics = (lam_phys > 1e-12) and (physics_epoch_gate > 1e-12)
    use_invariant = lam_inv > 1e-12
    use_dro = lam_dro > 1e-12
    physics_meta = dict(physics_meta or {})
    if (not physics_meta) and physics_sign_prior is not None and physics_sign_prior.numel() > 0:
        n_cat = int(physics_sign_prior.numel())
        physics_meta = {
            "sign": physics_sign_prior.to(torch.float32),
            "confidence": torch.ones((n_cat,), dtype=torch.float32, device=physics_sign_prior.device),
            "target_slope": torch.zeros((n_cat,), dtype=torch.float32, device=physics_sign_prior.device),
            "ph_lo": torch.full((n_cat,), -1.0, dtype=torch.float32, device=physics_sign_prior.device),
            "ph_hi": torch.full((n_cat,), 1.0, dtype=torch.float32, device=physics_sign_prior.device),
            "valid": (torch.abs(physics_sign_prior.to(torch.float32)) > 0.5).to(torch.float32),
        }

    total = {
        "loss": 0.0,
        "pred": 0.0,
        "inv": 0.0,
        "dro": 0.0,
        "phys": 0.0,
        "n": 0.0,
    }

    for batch in dataloader:
        fingerprint = batch["fingerprint"].to(device, non_blocking=non_blocking)
        numeric = batch["numeric"].to(device, non_blocking=non_blocking)
        y = batch["logk"].to(device, non_blocking=non_blocking)
        category = batch["category"].to(device, non_blocking=non_blocking)
        base_idx = batch.get("base_idx")
        if base_idx is not None:
            base_idx = base_idx.to(device, non_blocking=non_blocking).long()

        trainer.optimizer.zero_grad(set_to_none=True)

        def _forward_loss():
            components = None
            if use_physics and physics_safe_target == "attn" and hasattr(trainer.model, "forward_components"):
                components = trainer.model.forward_components(fingerprint, numeric)
                pred = components["final"]
                phys_pred_base = components["attn"]
            else:
                pred = trainer.model(fingerprint, numeric)
                phys_pred_base = pred
            pred_loss = F.mse_loss(pred, y, reduction="mean")
            group_losses = _group_mse_losses(pred, y, category)
            if use_invariant and group_losses.numel() > 1:
                inv_loss = torch.var(group_losses, unbiased=False)
            else:
                inv_loss = pred_loss.new_zeros(())
            if use_dro and group_losses.numel() > 0:
                tau = float(max(1e-4, dro_tau))
                dro_loss = float(tau) * torch.logsumexp(group_losses / float(tau), dim=0)
            else:
                dro_loss = pred_loss.new_zeros(())

            phys_loss = pred_loss.new_zeros(())
            if use_physics:
                if physics_mode == "finite_diff":
                    numeric_pert = numeric.clone()
                    numeric_pert[:, 0] = numeric_pert[:, 0] + float(physics_delta)
                    if physics_safe_target == "attn" and hasattr(trainer.model, "forward_components"):
                        pert_components = trainer.model.forward_components(fingerprint, numeric_pert)
                        pred_plus = pert_components["attn"]
                    else:
                        pred_plus = trainer.model(fingerprint, numeric_pert)
                    d_pred = pred_plus - phys_pred_base
                    deriv_fd = d_pred / float(max(1e-4, abs(physics_delta)))
                    smooth_loss = torch.mean(deriv_fd ** 2)
                    mono_loss = pred_loss.new_zeros(())
                    curv_weights = None
                    curv_mask = None
                    if physics_fd_conf_gated and physics_meta:
                        cat_idx = torch.argmax(category, dim=1)
                        sign = physics_meta.get("sign")
                        conf = physics_meta.get("confidence")
                        valid = physics_meta.get("valid")
                        ph_lo = physics_meta.get("ph_lo")
                        ph_hi = physics_meta.get("ph_hi")
                        target_slope = physics_meta.get("target_slope")
                        if sign is None:
                            sign = physics_sign_prior if physics_sign_prior is not None else None
                        if sign is not None and sign.numel() > 0:
                            sign_b = sign[cat_idx].to(pred.dtype)
                            conf_b = (
                                conf[cat_idx].to(pred.dtype)
                                if conf is not None and conf.numel() > 0
                                else torch.ones_like(sign_b, dtype=pred.dtype)
                            )
                            valid_b = (
                                valid[cat_idx].to(torch.bool)
                                if valid is not None and valid.numel() > 0
                                else (torch.abs(sign_b) > 0.5)
                            )
                            target_b = (
                                target_slope[cat_idx].to(pred.dtype)
                                if target_slope is not None and target_slope.numel() > 0
                                else torch.zeros_like(sign_b, dtype=pred.dtype)
                            )
                            if physics_slope_clip > 0.0:
                                target_b = target_b.clamp(-float(physics_slope_clip), float(physics_slope_clip))
                            if base_idx is not None:
                                sample_valid = physics_meta.get("sample_valid")
                                sample_sign = physics_meta.get("sample_sign")
                                sample_conf = physics_meta.get("sample_confidence")
                                sample_target = physics_meta.get("sample_target_slope")
                                sample_lo = physics_meta.get("sample_ph_lo")
                                sample_hi = physics_meta.get("sample_ph_hi")
                                if sample_valid is not None and sample_valid.numel() > 0:
                                    safe_idx = base_idx.clamp(0, int(sample_valid.numel()) - 1)
                                    sv = sample_valid[safe_idx].to(torch.bool)
                                    if sample_sign is not None and sample_sign.numel() > 0:
                                        sign_b = torch.where(sv, sample_sign[safe_idx].to(pred.dtype), sign_b)
                                    if sample_conf is not None and sample_conf.numel() > 0:
                                        conf_b = torch.where(sv, sample_conf[safe_idx].to(pred.dtype), conf_b)
                                    if sample_target is not None and sample_target.numel() > 0:
                                        target_b = torch.where(sv, sample_target[safe_idx].to(pred.dtype), target_b)
                                        if physics_slope_clip > 0.0:
                                            target_b = target_b.clamp(-float(physics_slope_clip), float(physics_slope_clip))
                                    valid_b = valid_b | sv
                            range_mask = torch.ones_like(valid_b, dtype=torch.bool)
                            if physics_fd_range_gated and ph_lo is not None and ph_hi is not None:
                                lo_b = ph_lo[cat_idx].to(pred.dtype)
                                hi_b = ph_hi[cat_idx].to(pred.dtype)
                                if base_idx is not None:
                                    sample_valid = physics_meta.get("sample_valid")
                                    sample_lo = physics_meta.get("sample_ph_lo")
                                    sample_hi = physics_meta.get("sample_ph_hi")
                                    if sample_valid is not None and sample_valid.numel() > 0 and sample_lo is not None and sample_hi is not None:
                                        safe_idx = base_idx.clamp(0, int(sample_valid.numel()) - 1)
                                        sv = sample_valid[safe_idx].to(torch.bool)
                                        lo_b = torch.where(sv, sample_lo[safe_idx].to(pred.dtype), lo_b)
                                        hi_b = torch.where(sv, sample_hi[safe_idx].to(pred.dtype), hi_b)
                                ph_now = numeric[:, 0].to(pred.dtype)
                                ph_next = numeric_pert[:, 0].to(pred.dtype)
                                range_mask = (
                                    (ph_now >= lo_b)
                                    & (ph_now <= hi_b)
                                    & (ph_next >= lo_b)
                                    & (ph_next <= hi_b)
                                )
                                if float(physics_pinn_curv_w) > 1e-12:
                                    ph_prev = ph_now - float(physics_delta)
                                    range_mask = range_mask & (ph_prev >= lo_b) & (ph_prev <= hi_b)
                            smooth_floor = float(max(0.0, min(1.0, physics_fd_smooth_floor)))
                            conf_gate = conf_b.clamp(0.0, 1.0).pow(float(physics_conf_power))
                            smooth_weights = smooth_floor + (1.0 - smooth_floor) * conf_gate
                            if physics_fd_range_gated:
                                smooth_weights = smooth_weights * range_mask.to(pred.dtype)
                            curv_weights = smooth_weights
                            curv_mask = valid_b & (conf_b >= float(physics_pinn_curv_conf_min))
                            if physics_fd_range_gated:
                                curv_mask = curv_mask & range_mask
                            if torch.sum(smooth_weights).item() > 1e-8:
                                slope_mask = valid_b & (torch.abs(sign_b) > 0.5) & (conf_b >= float(physics_conf_min))
                                if physics_fd_range_gated:
                                    slope_mask = slope_mask & range_mask
                                if torch.any(slope_mask) and float(physics_slope_target_w) > 1e-12:
                                    slope_weights = (smooth_weights[slope_mask] * conf_b[slope_mask].clamp_min(1e-3)).to(pred.dtype)
                                    target_loss = torch.sum(slope_weights * ((deriv_fd[slope_mask] - target_b[slope_mask]) ** 2)) / torch.sum(slope_weights)
                                    floor_loss = torch.sum(smooth_weights * (deriv_fd ** 2)) / torch.sum(smooth_weights)
                                    stw = float(max(0.0, min(1.0, physics_slope_target_w)))
                                    smooth_loss = stw * target_loss + (1.0 - stw) * floor_loss
                                else:
                                    smooth_loss = torch.sum(smooth_weights * (deriv_fd ** 2)) / torch.sum(smooth_weights)

                            mono_mask = (
                                (torch.abs(sign_b) > 0.5)
                                & valid_b
                                & range_mask
                                & (conf_b >= float(physics_conf_min))
                            )
                            if torch.any(mono_mask):
                                mono_weights = conf_b[mono_mask].clamp_min(1e-3).to(pred.dtype)
                                mono_terms = torch.relu(-(sign_b[mono_mask] * d_pred[mono_mask]))
                                mono_loss = torch.sum(mono_weights * mono_terms) / torch.sum(mono_weights)
                    elif physics_sign_prior is not None and physics_sign_prior.numel() > 0:
                        cat_idx = torch.argmax(category, dim=1)
                        sign = physics_sign_prior[cat_idx].to(pred.dtype)
                        mask = torch.abs(sign) > 0.5
                        if torch.any(mask):
                            mono_loss = torch.mean(torch.relu(-(sign[mask] * d_pred[mask])))
                    phys_loss = float(phys_smooth_w) * smooth_loss + float(phys_mono_w) * mono_loss
                    if float(physics_pinn_curv_w) > 1e-12:
                        numeric_minus = numeric.clone()
                        numeric_minus[:, 0] = numeric_minus[:, 0] - float(physics_delta)
                        if physics_safe_target == "attn" and hasattr(trainer.model, "forward_components"):
                            minus_components = trainer.model.forward_components(fingerprint, numeric_minus)
                            pred_minus = minus_components["attn"]
                        else:
                            pred_minus = trainer.model(fingerprint, numeric_minus)
                        denom = float(max(1e-4, abs(physics_delta)) ** 2)
                        d2_fd = (pred_plus - 2.0 * phys_pred_base + pred_minus) / denom
                        if curv_weights is None:
                            curv_weights = torch.ones_like(d2_fd, dtype=pred.dtype)
                        if curv_mask is None:
                            curv_mask = torch.ones_like(d2_fd, dtype=torch.bool)
                        curv_mask = curv_mask.reshape_as(d2_fd).to(torch.bool)
                        curv_weights = curv_weights.reshape_as(d2_fd).to(pred.dtype)
                        if torch.any(curv_mask):
                            cw = curv_weights[curv_mask].clamp_min(1e-3)
                            curv_loss = torch.sum(cw * (d2_fd[curv_mask] ** 2)) / torch.sum(cw)
                            phys_loss = phys_loss + float(max(0.0, physics_pinn_curv_w)) * curv_loss
                else:
                    physics_ctx = torch.cuda.amp.autocast(enabled=False) if use_amp else nullcontext()
                    with physics_ctx:
                        fp_phys = fingerprint.to(torch.float32)
                        num_base = numeric.to(torch.float32)
                        ph = num_base[:, :1].detach().clone().requires_grad_(True)
                        cat_feat = num_base[:, 1:].detach()
                        numeric_pi = torch.cat([ph, cat_feat], dim=1)
                        pred_pi = trainer.model(fp_phys, numeric_pi).to(torch.float32)
                        dy_dph = torch.autograd.grad(pred_pi.sum(), ph, create_graph=True)[0].reshape(-1)

                        cat_idx = torch.argmax(category, dim=1)
                        sign = physics_meta.get("sign")
                        conf = physics_meta.get("confidence")
                        valid = physics_meta.get("valid")
                        target_slope = physics_meta.get("target_slope")
                        if sign is None:
                            sign = torch.zeros((1,), dtype=torch.float32, device=pred.device)
                        if conf is None:
                            conf = torch.ones_like(sign)
                        if valid is None:
                            valid = (torch.abs(sign) > 0.5).to(torch.float32)
                        sign_b = sign[cat_idx].to(torch.float32)
                        conf_b = conf[cat_idx].to(torch.float32)
                        valid_b = valid[cat_idx].to(torch.float32)
                        target_b = (
                            target_slope[cat_idx].to(torch.float32)
                            if target_slope is not None and target_slope.numel() > 0
                            else torch.zeros_like(sign_b, dtype=torch.float32)
                        )
                        if physics_slope_clip > 0.0:
                            target_b = target_b.clamp(-float(physics_slope_clip), float(physics_slope_clip))
                        if base_idx is not None:
                            sample_valid = physics_meta.get("sample_valid")
                            sample_sign = physics_meta.get("sample_sign")
                            sample_conf = physics_meta.get("sample_confidence")
                            sample_target = physics_meta.get("sample_target_slope")
                            if sample_valid is not None and sample_valid.numel() > 0:
                                safe_idx = base_idx.clamp(0, int(sample_valid.numel()) - 1)
                                sv = sample_valid[safe_idx].to(torch.bool)
                                if sample_sign is not None and sample_sign.numel() > 0:
                                    sign_b = torch.where(sv, sample_sign[safe_idx].to(torch.float32), sign_b)
                                if sample_conf is not None and sample_conf.numel() > 0:
                                    conf_b = torch.where(sv, sample_conf[safe_idx].to(torch.float32), conf_b)
                                if sample_target is not None and sample_target.numel() > 0:
                                    target_b = torch.where(sv, sample_target[safe_idx].to(torch.float32), target_b)
                                    if physics_slope_clip > 0.0:
                                        target_b = target_b.clamp(-float(physics_slope_clip), float(physics_slope_clip))
                                valid_b = torch.maximum(valid_b, sv.to(torch.float32))
                        mask = (torch.abs(sign_b) > 0.5) & (conf_b >= float(physics_conf_min)) & (valid_b > 0.5)

                        smooth_loss = torch.mean(dy_dph ** 2)
                        mono_loss = pred_pi.new_zeros(())
                        if torch.any(mask):
                            weights = conf_b[mask].clamp_min(1e-3)
                            mono_terms = torch.relu(-(sign_b[mask] * dy_dph[mask]))
                            mono_loss = torch.sum(weights * mono_terms) / torch.sum(weights)
                            target_loss = torch.sum(weights * ((dy_dph[mask] - target_b[mask]) ** 2)) / torch.sum(weights)
                            floor_loss = torch.sum(weights * (dy_dph[mask] ** 2)) / torch.sum(weights)
                            stw = float(max(0.0, min(1.0, physics_slope_target_w)))
                            smooth_loss = stw * target_loss + (1.0 - stw) * floor_loss

                        phys_loss = float(phys_smooth_w) * smooth_loss + float(phys_mono_w) * mono_loss

                        if physics_use_colloc and torch.any(mask):
                            ph_lo = physics_meta.get("ph_lo")
                            ph_hi = physics_meta.get("ph_hi")
                            if ph_lo is not None and ph_hi is not None:
                                lo_b = ph_lo[cat_idx].to(torch.float32).unsqueeze(-1)
                                hi_b = ph_hi[cat_idx].to(torch.float32).unsqueeze(-1)
                                span = (hi_b - lo_b).clamp_min(1e-3)
                                rand = torch.rand_like(ph)
                                ph_col = lo_b + rand * span
                                ph_col = torch.where(mask.unsqueeze(-1), ph_col, ph.detach())
                                ph_col = ph_col.detach().clone().requires_grad_(True)
                                numeric_col = torch.cat([ph_col, cat_feat], dim=1)
                                pred_col = trainer.model(fp_phys, numeric_col).to(torch.float32)
                                dy_col = torch.autograd.grad(pred_col.sum(), ph_col, create_graph=True)[0].reshape(-1)
                                weights = conf_b[mask].clamp_min(1e-3)
                                colloc_target = torch.sum(weights * ((dy_col[mask] - target_b[mask]) ** 2)) / torch.sum(weights)
                                colloc_floor = torch.sum(weights * (dy_col[mask] ** 2)) / torch.sum(weights)
                                stw = float(max(0.0, min(1.0, physics_slope_target_w)))
                                colloc_smooth = stw * colloc_target + (1.0 - stw) * colloc_floor
                                colloc_mono = torch.sum(weights * torch.relu(-(sign_b[mask] * dy_col[mask]))) / torch.sum(weights)
                                colloc_loss = float(phys_smooth_w) * colloc_smooth + float(phys_mono_w) * colloc_mono
                                phys_loss = phys_loss + float(max(0.0, physics_colloc_w)) * colloc_loss

            effective_lam_phys = float(lam_phys) * float(physics_epoch_gate)
            total_loss = pred_loss + float(lam_inv) * inv_loss + float(lam_dro) * dro_loss + effective_lam_phys * phys_loss
            return total_loss, pred_loss, inv_loss, dro_loss, phys_loss

        if use_amp:
            from torch.cuda.amp import autocast

            with autocast(enabled=True, dtype=amp_dtype):
                loss, pred_loss, inv_loss, dro_loss, phys_loss = _forward_loss()
            if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                scaler.scale(loss).backward()
                scaler.unscale_(trainer.optimizer)
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=5.0)
                scaler.step(trainer.optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=5.0)
                trainer.optimizer.step()
        else:
            loss, pred_loss, inv_loss, dro_loss, phys_loss = _forward_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=5.0)
            trainer.optimizer.step()

        bs = float(fingerprint.size(0))
        total["loss"] += float(loss.detach().cpu().item()) * bs
        total["pred"] += float(pred_loss.detach().cpu().item()) * bs
        total["inv"] += float(inv_loss.detach().cpu().item()) * bs
        total["dro"] += float(dro_loss.detach().cpu().item()) * bs
        total["phys"] += float(phys_loss.detach().cpu().item()) * bs
        total["n"] += bs

    n = max(1.0, float(total["n"]))
    return {
        "loss": total["loss"] / n,
        "pred_loss": total["pred"] / n,
        "inv_loss": total["inv"] / n,
        "dro_loss": total["dro"] / n,
        "phys_loss": total["phys"] / n,
    }


def _build_split_indices(dataset: FingerprintReactionDataset) -> Dict[str, np.ndarray]:
    global SPLIT_CACHE
    split_seed = int(_env_value("TRANSFORMER_V7_SPLIT_SEED", "TRANSFORMER_V6_SPLIT_SEED", _env_value("TRANSFORMER_V5_SPLIT_SEED", None, "42")))
    fixed_split_json = str(_env_value("TRANSFORMER_V7_FIXED_SPLIT_JSON", "TRANSFORMER_V6_FIXED_SPLIT_JSON", _env_value("TRANSFORMER_V5_FIXED_SPLIT_JSON", None, ""))).strip()
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
                and test_idx.size > 0
                and all_idx.size == len(dataset)
                and np.unique(all_idx).size == len(dataset)
                and int(all_idx.min()) >= 0
                and int(all_idx.max()) < len(dataset)
            ):
                train_val_idx = np.concatenate([train_idx, val_idx], axis=0).astype(int)
                SPLIT_CACHE = {
                    "n": len(dataset),
                    "seed": int(split_seed),
                    "train_idx": train_idx.astype(int),
                    "val_idx": val_idx.astype(int),
                    "test_idx": test_idx.astype(int),
                    "train_val_idx": train_val_idx,
                    "fixed_split_json": fixed_split_json,
                }
                return {
                    "train_idx": np.asarray(SPLIT_CACHE["train_idx"], dtype=int),
                    "val_idx": np.asarray(SPLIT_CACHE["val_idx"], dtype=int),
                    "test_idx": np.asarray(SPLIT_CACHE["test_idx"], dtype=int),
                    "train_val_idx": np.asarray(SPLIT_CACHE["train_val_idx"], dtype=int),
                }
        except Exception:
            pass
    if SPLIT_CACHE.get("n") == len(dataset) and int(SPLIT_CACHE.get("seed", split_seed)) == int(split_seed):
        return {
            "train_idx": np.asarray(SPLIT_CACHE["train_idx"], dtype=int),
            "val_idx": np.asarray(SPLIT_CACHE["val_idx"], dtype=int),
            "test_idx": np.asarray(SPLIT_CACHE["test_idx"], dtype=int),
            "train_val_idx": np.asarray(SPLIT_CACHE.get("train_val_idx", np.concatenate([SPLIT_CACHE["train_idx"], SPLIT_CACHE["val_idx"]], axis=0)), dtype=int),
        }

    indices = np.arange(len(dataset), dtype=int)

    stratify_labels = None
    if _env_bool_pair("TRANSFORMER_V3_STRATIFY", "TRANSFORMER_V2_STRATIFY", False):
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
        label_map = {int(i): int(l) for i, l in zip(indices.tolist(), stratify_labels.tolist())}
        try:
            labels_tv = np.asarray([label_map[int(i)] for i in train_val_idx], dtype=int)
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

    SPLIT_CACHE = {
        "n": len(dataset),
        "seed": int(split_seed),
        "train_idx": np.asarray(train_idx, dtype=int),
        "val_idx": np.asarray(val_idx, dtype=int),
        "test_idx": np.asarray(test_idx, dtype=int),
        "train_val_idx": np.asarray(train_val_idx, dtype=int),
    }

    return {
        "train_idx": np.asarray(SPLIT_CACHE["train_idx"], dtype=int),
        "val_idx": np.asarray(SPLIT_CACHE["val_idx"], dtype=int),
        "test_idx": np.asarray(SPLIT_CACHE["test_idx"], dtype=int),
        "train_val_idx": np.asarray(SPLIT_CACHE["train_val_idx"], dtype=int),
    }


def _get_fp_ranking(
    dataset: FingerprintReactionDataset,
    train_idx: np.ndarray,
    fp_bits: int,
    method: str,
) -> np.ndarray:
    global FP_RANK_CACHE

    method = str(method).strip().lower()
    if method not in {"rf", "f_regression"}:
        method = "rf"

    key = (int(fp_bits), method)
    if key in FP_RANK_CACHE:
        return FP_RANK_CACHE[key]

    x_train_fp = dataset.fingerprint[train_idx].astype(np.float32)
    y_train_raw = dataset.logk_raw[train_idx].reshape(-1)

    if method == "rf":
        rf = RandomForestRegressor(
            n_estimators=240,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(x_train_fp, y_train_raw)
        score = rf.feature_importances_
    else:
        f_vals, _ = f_regression(x_train_fp, y_train_raw)
        score = np.nan_to_num(f_vals, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

    rank = np.argsort(score)[::-1].astype(int)
    FP_RANK_CACHE[key] = rank
    return rank


# =============================================================================
# Model innovation: chemistry-aware transformer + uncertainty-aware dual experts
# =============================================================================
class ChemistryAwareTransformerEncoderLayer(nn.Module):
    """
    带有可选的加性化学感知注意力偏置的Transformer编码器层。
    这是标准Transformer层的修改版，允许在注意力计算中注入特定于化学的偏置（Bias）。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, norm_first: bool = False):
        super().__init__()
        self.n_heads = int(n_heads)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.last_attn = None
        self.norm_first = bool(norm_first)

    def _expand_attn_mask(self, attn_mask: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
        """扩展注意力掩码以匹配(Batch * Num_Heads, Seq_Len, Seq_Len)的形状。"""
        if attn_mask is None:
            return None
        if attn_mask.dim() == 2:
            return attn_mask
        if attn_mask.dim() != 3:
            return None
        if attn_mask.size(0) == 1 and batch_size > 1:
            attn_mask = attn_mask.expand(batch_size, -1, -1)
        if attn_mask.size(0) == batch_size:
            return attn_mask.repeat_interleave(self.n_heads, dim=0)
        return None

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        attn_mask_expanded = self._expand_attn_mask(attn_mask, x.size(0))
        if self.norm_first:
            x_norm = self.norm1(x)
            attn_out, attn_w = self.self_attn(
                x_norm,
                x_norm,
                x_norm,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask_expanded,
                need_weights=True,
                average_attn_weights=False,
            )
            self.last_attn = attn_w
            x = x + self.dropout(attn_out)
            x_norm = self.norm2(x)
            ff = self.linear2(self.dropout(self.act(self.linear1(x_norm))))
            x = x + self.dropout(ff)
            return x

        attn_out, attn_w = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask_expanded,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attn = attn_w
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.linear2(self.dropout(self.act(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        return x


def _save_v3_sampler_plot(output_dir: str, info: Dict[str, object]):
    """
    绘制采样器分布图 (Visualize sampler distribution).
    显示不同类别在不同 Epoch 的采样概率变化。
    """
    try:
        import matplotlib.pyplot as plt
        history = info.get("history", [])
        if not history:
            return
        
        epochs = [h["epoch"] for h in history]
        probs = np.array([h["probs"] for h in history])
        n_cat = probs.shape[1]
        
        plt.figure(figsize=(10, 6))
        for i in range(n_cat):
            plt.plot(epochs, probs[:, i], label=f"Cat {i}")
            
        plt.xlabel("Epoch")
        plt.ylabel("Sampling Probability")
        plt.title("Category Sampling Probability Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        _savefig_with_pdf(plt, os.path.join(output_dir, "sampler_evolution.png"))
        plt.close()
    except Exception:
        pass


def _save_v3_fusion_plot(output_dir: str, result: Dict[str, object]):
    """
    绘制融合权重图 (Visualize fusion weights).
    显示不同基础模型在最终融合中的权重贡献。
    """
    try:
        import matplotlib.pyplot as plt
        global_w = result.get("global_weights", [])
        models = list(result.get("component_names", [])) or ["NN", "HGB", "RF", "ET"]
        
        if len(global_w) == len(models):
            plt.figure(figsize=(8, 6))
            plt.bar(models, global_w)
            plt.ylabel("Weight")
            plt.title("Global Fusion Weights")
            for i, v in enumerate(global_w):
                plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
            _savefig_with_pdf(plt, os.path.join(output_dir, "fusion_weights.png"))
            plt.close()
            
    except Exception:
        pass


def _save_v3_chem_attention_bias_plot(output_dir: str, model: nn.Module, loader: DataLoader):
    """
    绘制化学注意力偏置热图 (Visualize Chemistry-Aware Attention Bias).
    如果模型使用了 chemistry_aware_attention，则提取其可学习的偏置矩阵并可视化。
    这有助于理解模型学到的化学规则（如不同类别间的相互作用倾向）。
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 检查模型是否包含 chemistry_aware_bias
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            # 取第一层的注意力模块
            attn = model.encoder.layers[0].self_attn
            if hasattr(attn, "chemistry_aware_bias") and attn.chemistry_aware_bias is not None:
                bias = attn.chemistry_aware_bias.detach().cpu().numpy()
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(bias, cmap="RdBu_r", center=0, annot=True, fmt=".2f")
                plt.title("Chemistry-Aware Attention Bias (Layer 0)")
                plt.xlabel("Key Category")
                plt.ylabel("Query Category")
                _savefig_with_pdf(plt, os.path.join(output_dir, "chem_attn_bias.png"))
                plt.close()
                
    except Exception:
        pass


def _savefig_with_pdf(plt_obj, path_png, **kwargs):
    """同时保存 PNG 和 PDF 格式的图片 (Save figure as PNG and PDF)."""
    try:
        save_kwargs = {"dpi": 300, "bbox_inches": "tight"}
        save_kwargs.update(kwargs)
        plt_obj.savefig(path_png, **save_kwargs)
        pdf_path = path_png.replace(".png", ".pdf")
        pdf_kwargs = dict(save_kwargs)
        pdf_kwargs.pop("dpi", None)
        plt_obj.savefig(pdf_path, format="pdf", **pdf_kwargs)
    except Exception:
        pass



class ChemistryAwareAttentionTransformer(AttentionFingerprintTransformer):
    """
    化学感知注意力Transformer (V3+)。
    在指纹token的交互中引入成对偏置（Pairwise Bias），
    或者在分类token和指纹token之间引入类别先验。
    
    (Note: This class inherits from AttentionFingerprintTransformer for compatibility, 
     but overrides the core forward logic to inject chemistry awareness.)
    """
    def __init__(self, *args, chemistry_aware_attention: bool = True, n_categories: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.chemistry_aware_attention = chemistry_aware_attention
        self.n_categories = n_categories
        
        # 覆盖编码器层，使用支持化学偏置的层 (Override with custom encoder layer)
        # 注意：这里我们重建了 encoder，使用我们自定义的 Layer 类
        encoder_layer = ChemistryAwareTransformerEncoderLayer(
            self.d_model, self.n_heads, self.dropout_rate, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # 类别偏置矩阵 (Category-Category Interaction Bias)
        if self.chemistry_aware_attention and n_categories > 1:
            # 这是一个可学习的矩阵，表示不同类别之间的"亲和力"或"排斥力"
            # 实际上，它应该作为 Attention Mask 的一部分注入
            # 但由于 PyTorch nn.MultiheadAttention 的限制，我们通常将其加到 value 或 output 上
            # 或者，更简单地，我们在 encoder layer 中处理
            # 这里我们定义参数，将在 forward 中使用
            self.chem_bias_param = nn.Parameter(torch.zeros(n_categories, n_categories))
            nn.init.xavier_uniform_(self.chem_bias_param)
        else:
            self.chem_bias_param = None
            
        # 重新初始化双专家头 (Re-initialize Dual Experts if not present in base)
        # Expert A: 宽而浅 (Wide & Shallow)
        self.expert_a = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, 1)
        )
        # Expert B: 深而窄 (Deep & Narrow)
        self.expert_b = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model // 2, 1)
        )
        # 不确定性门控 (Uncertainty Gating)
        self.uncertainty_gate = nn.Linear(self.d_model, 2)


    def forward(self, fingerprint: torch.Tensor, numeric: torch.Tensor, category: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 (Forward Pass).
        支持注入 category 信息用于化学感知注意力。
        """
        # 1. 基础嵌入 (Basic Embedding)
        B, L, _ = fingerprint.shape
        device = fingerprint.device

        fp_idx = fingerprint[:, :, 0].long()
        fp_val = fingerprint[:, :, 1].float().unsqueeze(-1)

        if self.training and self.fp_bit_dropout > 0:
            mask = torch.rand_like(fp_val) > self.fp_bit_dropout
            fp_val = fp_val * mask
            
        padding_mask = (fp_idx < 0)
        clean_idx = fp_idx.clone()
        clean_idx[padding_mask] = 0
        
        x_fp = self.idx_emb(clean_idx) + self.val_emb(fp_val) # [B, L, D]
        x_num = self.num_emb(numeric).unsqueeze(1) # [B, 1, D]
        
        # 2. 拼接与掩码 (Concatenation & Masking)
        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x_num, x_fp], dim=1)
            full_padding_mask = torch.cat([
                torch.zeros(B, 2, dtype=torch.bool, device=device),
                padding_mask
            ], dim=1)
        else:
            x = torch.cat([x_num, x_fp], dim=1)
            full_padding_mask = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=device),
                padding_mask
            ], dim=1)

        # 3. 注入化学偏置 (Inject Chemistry Bias)
        # 这是一个简化实现：我们将 bias 注册到 encoder layers 中
        # 注意：这假设 encoder layers 是 ChemistryAwareTransformerEncoderLayer 实例
        if self.chem_bias_param is not None and category is not None:
            # 这里的逻辑比较复杂：
            # 我们需要根据 batch 中每个样本的 category，从 chem_bias_param 中选取对应的 bias
            # 但 Transformer 的 self-attention 是 token-to-token 的
            # 这里的 bias 实际上是 "Context Bias"
            # 为简单起见，我们将 global bias 注入到所有层
            for layer in self.encoder.layers:
                if hasattr(layer.self_attn, "chemistry_aware_bias"):
                    # 暂时不动态传递 per-sample bias，而是传递全局 bias 矩阵用于可视化或调试
                    layer.self_attn.chemistry_aware_bias = self.chem_bias_param

        # 4. 编码 (Encoding)
        x = self.encoder(x, src_key_padding_mask=full_padding_mask)

        # 5. 池化 (Pooling)
        if self.pooling == "attn":
            q = self.pool_query.expand(B, -1, -1)
            pool_out, _ = self.pool_attn(q, x, x, key_padding_mask=full_padding_mask)
            feature = pool_out.squeeze(1)
        elif self.pooling == "cls":
            feature = x[:, 0, :]
        else: # mean
            mask = (~full_padding_mask).float().unsqueeze(-1)
            sum_x = torch.sum(x * mask, dim=1)
            len_x = torch.sum(mask, dim=1).clamp(min=1e-9)
            feature = sum_x / len_x

        # 6. 双专家预测 (Dual Expert Prediction)
        out_a = self.expert_a(feature)
        out_b = self.expert_b(feature)
        
        # 7. 不确定性加权 (Uncertainty Weighting)
        gate_logits = self.uncertainty_gate(feature)
        weights = F.softmax(gate_logits, dim=1) # [B, 2]
        
        final_pred = weights[:, 0:1] * out_a + weights[:, 1:2] * out_b
        return final_pred


class ChemBiasAttentionRegressor(nn.Module):
    """
    面向稀疏指纹位的化学偏置注意力回归器。
    使用 raw fingerprint tokens，并在 pooling 后与 pH/category 数值特征门控融合。
    """

    def __init__(self, fingerprint_dim: int, numeric_dim: int, config: Optional[FingerprintConfig] = None):
        super().__init__()
        self.config = config or FingerprintConfig
        self.fingerprint_dim = int(fingerprint_dim)
        self.numeric_dim = int(numeric_dim)
        self.d_model = int(self.config.d_model)
        self.max_fp_tokens = int(min(getattr(self.config, "max_fp_tokens", 128), self.fingerprint_dim))
        self.uses_sparse_tokens = self.max_fp_tokens < self.fingerprint_dim
        self.attn_pooling = str(getattr(self.config, "attn_pooling", "attn")).strip().lower() or "attn"
        self.fp_bit_dropout = float(getattr(self.config, "fp_bit_dropout", 0.0) or 0.0)
        self.n_heads = int(getattr(self.config, "n_heads", 4))
        self.num_input_dim = max(1, self.numeric_dim)
        self.enable_chem_bias = _env_bool_pair(
            "TRANSFORMER_V3_ENABLE_CHEM_ATTN_BIAS",
            "TRANSFORMER_V2_ENABLE_CHEM_ATTN_BIAS",
            True,
        )
        self.chem_bias_alpha = float(
            _env_value("TRANSFORMER_V3_CHEM_ATTN_ALPHA", "TRANSFORMER_V2_CHEM_ATTN_ALPHA", "0.30")
        )
        self.chem_bias_learnable_gate = _env_bool_pair("TRANSFORMER_V9_CHEM_BIAS_LEARNABLE_GATE", None, False)
        self.chem_bias_gate_init = float(_env_value("TRANSFORMER_V9_CHEM_BIAS_GATE_INIT", None, "0.08"))
        self.chem_bias_gate_max = float(
            _env_value("TRANSFORMER_V9_CHEM_BIAS_GATE_MAX", None, str(max(1e-6, self.chem_bias_alpha)))
        )
        self.enable_learned_bit_bias = _env_bool_pair(
            "TRANSFORMER_V7_ENABLE_LEARNED_BIT_ATTN_BIAS",
            "TRANSFORMER_V6_ENABLE_LEARNED_BIT_ATTN_BIAS",
            True,
        )

        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.fp_index_embed = nn.Embedding(self.fingerprint_dim, self.d_model)
        self.chem_bit_bias = nn.Parameter(torch.zeros(self.fingerprint_dim, dtype=torch.float32))
        self.fp_val_embed = nn.Linear(1, self.d_model)
        self.fp_token_norm = nn.LayerNorm(self.d_model)
        self.fp_token_dropout = nn.Dropout(float(self.config.dropout))

        self.num_proj = nn.Sequential(
            nn.Linear(self.num_input_dim, self.d_model),
            nn.GELU() if self.config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )
        self.fusion = GatedFusion(self.d_model, dropout=self.config.dropout)
        self.encoder_layers = nn.ModuleList(
            [
                ChemistryAwareTransformerEncoderLayer(
                    self.d_model,
                    self.n_heads,
                    dropout=float(self.config.dropout),
                    norm_first=bool(getattr(self.config, "norm_first", False)),
                )
                for _ in range(int(getattr(self.config, "n_layers", 2)))
            ]
        )
        self.pool_norm = nn.LayerNorm(self.d_model)
        self.pool_attn = nn.Linear(self.d_model, 1)
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU() if self.config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.d_model // 2, 1),
        )
        self.numeric_bias_proj = nn.Linear(max(1, self.numeric_dim), 1)
        self.attn_cache = {"self": []}
        self.capture_attn = False
        self.last_attn_bias = None
        self.last_key_padding_mask = None
        self.last_token_indices = None
        self.last_token_active = None
        self.last_chem_bias_scale = None
        if self.chem_bias_learnable_gate:
            gate_max = float(max(1e-6, self.chem_bias_gate_max))
            gate_init = float(max(1e-6, min(gate_max - 1e-6, self.chem_bias_gate_init)))
            init_prob = float(max(1e-6, min(1.0 - 1e-6, gate_init / gate_max)))
            init_logit = np.log(init_prob / (1.0 - init_prob))
            self.chem_bias_gate_logit = nn.Parameter(torch.tensor(float(init_logit), dtype=torch.float32))
        else:
            self.register_parameter("chem_bias_gate_logit", None)
        if self.enable_learned_bit_bias:
            nn.init.normal_(self.chem_bit_bias, mean=0.0, std=0.015)

    def _chem_bias_scale(self, device: torch.device) -> torch.Tensor:
        if self.chem_bias_learnable_gate and self.chem_bias_gate_logit is not None:
            gate_max = torch.as_tensor(
                float(max(1e-6, self.chem_bias_gate_max)),
                dtype=torch.float32,
                device=device,
            )
            scale = torch.sigmoid(self.chem_bias_gate_logit.to(torch.float32)) * gate_max
        else:
            scale = torch.as_tensor(float(self.chem_bias_alpha), dtype=torch.float32, device=device)
        self.last_chem_bias_scale = float(scale.detach().cpu().item())
        return scale

    def _build_tokens(self, fingerprint: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.uses_sparse_tokens:
            active_full = torch.abs(fingerprint) > 1e-6
            order = torch.argsort(~active_full, dim=1, stable=True)
            idx = order[:, :self.max_fp_tokens]
            vals = fingerprint.gather(1, idx)
            active_bits = torch.abs(vals) > 1e-6
            fp_bits_tokens = self.fp_index_embed(idx) + self.fp_val_embed(vals.unsqueeze(-1))
            fp_bits_tokens = fp_bits_tokens * active_bits.unsqueeze(-1).to(fp_bits_tokens.dtype)
            cls = self.cls_token.expand(vals.size(0), -1, -1)
            fp_tokens = torch.cat([cls, fp_bits_tokens], dim=1)
            key_padding_mask = torch.cat(
                [
                    torch.zeros((vals.size(0), 1), dtype=torch.bool, device=fingerprint.device),
                    ~active_bits,
                ],
                dim=1,
            )
            token_values = vals
        else:
            vals = fingerprint
            active_bits = torch.abs(vals) > 1e-6
            idx = torch.arange(self.fingerprint_dim, device=fingerprint.device).unsqueeze(0).expand(vals.size(0), -1)
            fp_bits_tokens = self.fp_index_embed(idx) + self.fp_val_embed(vals.unsqueeze(-1))
            fp_bits_tokens = fp_bits_tokens * active_bits.unsqueeze(-1).to(fp_bits_tokens.dtype)
            cls = self.cls_token.expand(vals.size(0), -1, -1)
            fp_tokens = torch.cat([cls, fp_bits_tokens], dim=1)
            key_padding_mask = torch.cat(
                [
                    torch.zeros((vals.size(0), 1), dtype=torch.bool, device=fingerprint.device),
                    ~active_bits,
                ],
                dim=1,
            )
            token_values = vals

        all_pad = key_padding_mask.all(dim=1)
        if all_pad.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad, 0] = False
        self.last_token_indices = idx.detach()
        self.last_token_active = active_bits.detach()
        return fp_tokens, key_padding_mask, token_values

    def _build_attention_bias(
        self,
        token_values: torch.Tensor,
        token_indices: torch.Tensor,
        numeric: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.enable_chem_bias:
            return None
        bit_strength = torch.abs(token_values).to(torch.float32).clamp(0.0, 1.0)
        if self.enable_learned_bit_bias and token_indices is not None:
            bit_prior = torch.tanh(self.chem_bit_bias[token_indices.long()].to(torch.float32))
            bit_strength = (bit_strength * (1.0 + 0.75 * bit_prior)).clamp(0.0, 2.0)
        bsz, n_tokens = bit_strength.shape
        length = n_tokens + 1
        pair = bit_strength.unsqueeze(2) * bit_strength.unsqueeze(1)
        bias = torch.zeros((bsz, length, length), dtype=torch.float32, device=token_values.device)
        bias[:, 1:, 1:] = pair
        bias[:, 0, 1:] = bit_strength
        bias[:, 1:, 0] = bit_strength

        numeric_full = _match_feature_dim(numeric.to(torch.float32), max(1, self.numeric_dim))
        num_gate = torch.tanh(self.numeric_bias_proj(numeric_full)).view(-1, 1, 1)
        bias = bias * (1.0 + 0.25 * num_gate)
        bias = torch.clamp(bias * self._chem_bias_scale(token_values.device), -2.0, 2.0)

        valid = (~key_padding_mask).to(torch.float32)
        bias = bias * valid.unsqueeze(1) * valid.unsqueeze(2)
        row_mean = bias.mean(dim=-1, keepdim=True)
        return bias - row_mean

    def forward(self, fingerprint: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        if numeric.dim() == 1:
            numeric = numeric.unsqueeze(-1)
        if self.training and self.fp_bit_dropout > 0.0:
            p = float(max(0.0, min(1.0, self.fp_bit_dropout)))
            if p > 0.0:
                active = torch.abs(fingerprint) > 1e-6
                drop = (torch.rand_like(fingerprint) < p) & active
                if drop.any():
                    fingerprint = fingerprint.masked_fill(drop, 0.0)

        fp_tokens, key_padding_mask, token_values = self._build_tokens(fingerprint)
        fp_tokens = self.fp_token_dropout(self.fp_token_norm(fp_tokens))
        attn_bias = self._build_attention_bias(token_values, self.last_token_indices, numeric, key_padding_mask)
        self.last_key_padding_mask = key_padding_mask.detach()
        self.last_attn_bias = attn_bias.detach() if attn_bias is not None else None

        self.attn_cache["self"] = []
        for layer in self.encoder_layers:
            fp_tokens = layer(fp_tokens, key_padding_mask=key_padding_mask, attn_mask=attn_bias)
            if self.capture_attn and layer.last_attn is not None:
                self.attn_cache["self"].append(layer.last_attn.detach())

        if self.attn_pooling in {"cls", "class"}:
            mol_feat = fp_tokens[:, 0, :]
        elif self.attn_pooling in {"mean", "avg", "average"}:
            keep = (~key_padding_mask).to(fp_tokens.dtype)
            keep_bits = keep[:, 1:]
            denom = keep_bits.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_bits = (fp_tokens[:, 1:, :] * keep_bits.unsqueeze(-1)).sum(dim=1) / denom
            has_bits = (keep_bits.sum(dim=1) > 0).unsqueeze(-1)
            mol_feat = torch.where(has_bits, mean_bits, fp_tokens[:, 0, :])
        else:
            scores = self.pool_attn(fp_tokens).squeeze(-1).float()
            scores = scores.masked_fill(key_padding_mask, -1e9)
            w = torch.softmax(scores, dim=1).to(fp_tokens.dtype).unsqueeze(-1)
            mol_feat = (fp_tokens * w).sum(dim=1)

        mol_feat = self.pool_norm(mol_feat)
        num_feat = self.num_proj(_match_feature_dim(numeric, self.num_input_dim))
        fused = self.fusion(mol_feat, num_feat)
        return self.regressor(fused).squeeze(-1)


def _build_attention_expert(
    fingerprint_dim: int, numeric_dim: int, config: FingerprintConfig
) -> nn.Module:
    use_bias = _env_bool_pair("TRANSFORMER_V3_ENABLE_CHEM_ATTN_BIAS", "TRANSFORMER_V2_ENABLE_CHEM_ATTN_BIAS", True)
    if use_bias:
        return ChemBiasAttentionRegressor(
            fingerprint_dim=fingerprint_dim,
            numeric_dim=numeric_dim,
            config=config,
        )
    return AttentionFingerprintTransformer(
        fingerprint_dim=fingerprint_dim,
        numeric_dim=numeric_dim,
        config=config,
    )


class DualExpertRegressor(nn.Module):
    """
    双专家回归模型。
    结合了注意力专家（Attention Expert）和MLP专家（MLP Expert）的预测。
    使用一个门控网络（Gate Net）来动态决定每个样本应该更信任哪个专家的预测。
    """

    def __init__(self, fingerprint_dim: int, numeric_dim: int, config: FingerprintConfig):
        super().__init__()
        self.config = config
        self.input_numeric_dim = int(numeric_dim)
        self.ph_basis_degree = int(
            max(1, min(3, int(_env_value("TRANSFORMER_V9_PH_BASIS_DEGREE", None, "1"))))
        )
        requested_category_embed_dim = int(
            _env_value("TRANSFORMER_V9_CATEGORY_EMBED_DIM", None, "0")
        )
        category_dim = max(0, self.input_numeric_dim - 1)
        self.category_embed_dim = (
            min(category_dim, requested_category_embed_dim)
            if requested_category_embed_dim > 0 and category_dim > 0
            else category_dim
        )
        self.internal_numeric_dim = self.ph_basis_degree + self.category_embed_dim
        self.category_dropout = nn.Dropout(
            float(max(0.0, min(0.5, float(_env_value("TRANSFORMER_V9_CATEGORY_DROPOUT", None, "0.0")))))
        )
        if self.category_embed_dim < category_dim:
            # Preserve all category labels at the input while allowing the
            # model to learn a compact category representation before it is
            # consumed by either expert or the gate.
            self.category_encoder = nn.Sequential(
                nn.Linear(category_dim, self.category_embed_dim, bias=False),
                nn.GELU(),
            )
        else:
            self.category_encoder = None
        self.enable_dual = _env_bool_pair("TRANSFORMER_V3_ENABLE_DUAL", "TRANSFORMER_V2_ENABLE_DUAL", True)
        self.use_gate_extra_features = _env_bool_pair(
            "TRANSFORMER_V3_GATE_EXTRA_FEATURES",
            "TRANSFORMER_V2_GATE_EXTRA_FEATURES",
            True,
        )
        self.enable_correction_head = _env_bool_pair(
            "TRANSFORMER_V3_ENABLE_CORRECTION_HEAD",
            "TRANSFORMER_V2_ENABLE_CORRECTION_HEAD",
            True,
        )
        self.attention_min_gate = float(
            _env_value("TRANSFORMER_V7_ATTENTION_MIN_GATE", "TRANSFORMER_V6_ATTENTION_MIN_GATE", "0.0")
        )
        self.attention_gate_logit_bias = float(
            _env_value("TRANSFORMER_V7_ATTENTION_GATE_LOGIT_BIAS", "TRANSFORMER_V6_ATTENTION_GATE_LOGIT_BIAS", "0.0")
        )
        self.v9_transformer_centered = _env_bool_pair("TRANSFORMER_V9_TRANSFORMER_CENTERED", None, True)
        self.v9_residual_max = float(_env_value("TRANSFORMER_V9_RESIDUAL_MAX", None, "0.85"))
        self.v9_residual_min = float(_env_value("TRANSFORMER_V9_RESIDUAL_MIN", None, "0.0"))
        self.attention_ph_only = _env_bool_pair("TRANSFORMER_V9_ATTN_PH_ONLY", None, False)

        self.attn_expert = _build_attention_expert(
            fingerprint_dim=fingerprint_dim,
            numeric_dim=self.ph_basis_degree if self.attention_ph_only else self.internal_numeric_dim,
            config=config,
        )
        self.mlp_expert = (
            FingerprintTransformer(
                fingerprint_dim=fingerprint_dim,
                numeric_dim=self.internal_numeric_dim,
                config=config,
            )
            if self.enable_dual
            else None
        )

        hidden = max(32, int(config.d_model // 2))
        gate_in_dim = self.internal_numeric_dim + (3 if self.use_gate_extra_features else 0)
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(hidden, 2),
        )
        self.corr_head = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(hidden, 1),
        )

    def _encode_numeric(self, numeric: torch.Tensor) -> torch.Tensor:
        ph = numeric[:, :1].to(torch.float32)
        ph_basis = torch.cat([ph.pow(power) for power in range(1, self.ph_basis_degree + 1)], dim=1)
        category = self.category_dropout(numeric[:, 1:].to(torch.float32))
        if self.category_encoder is None:
            return torch.cat([ph_basis, category], dim=1)
        return torch.cat([ph_basis, self.category_encoder(category)], dim=1)

    def forward_components(self, fingerprint: torch.Tensor, numeric: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return V9 expert components for residual-safe auxiliary losses."""
        if numeric.dim() == 1:
            numeric = numeric.unsqueeze(-1)
        numeric = self._encode_numeric(numeric)

        # The attention expert is the structure-response backbone.  In the
        # optional pH-only mode, category context remains fully available to
        # the residual expert, gate and correction branches but cannot
        # dominate the mechanistic attention pathway.
        attention_numeric = numeric[:, :self.ph_basis_degree] if self.attention_ph_only else numeric
        pred_attn = self.attn_expert(fingerprint, attention_numeric)
        fp_density = (torch.abs(fingerprint) > 1e-6).to(torch.float32).mean(dim=1, keepdim=True)
        fp_energy = torch.abs(fingerprint).to(torch.float32).mean(dim=1, keepdim=True)

        if not self.enable_dual or self.mlp_expert is None:
            return {
                "final": pred_attn,
                "attn": pred_attn,
                "mlp": pred_attn,
                "residual_gate": torch.zeros_like(pred_attn),
                "correction": torch.zeros_like(pred_attn),
            }

        pred_mlp = self.mlp_expert(fingerprint, numeric)
        pred_gap = torch.abs(pred_attn - pred_mlp).unsqueeze(-1)
        if self.use_gate_extra_features:
            gate_input = torch.cat([numeric.to(torch.float32), fp_density, fp_energy, pred_gap], dim=1)
        else:
            gate_input = numeric.to(torch.float32)
        logits = self.gate_net(gate_input)
        if abs(float(self.attention_gate_logit_bias)) > 1e-12:
            logits = logits.clone()
            logits[:, 0] = logits[:, 0] + float(self.attention_gate_logit_bias)
        weights = torch.softmax(logits, dim=1)
        min_attn = float(max(0.0, min(0.45, self.attention_min_gate)))
        if min_attn > 1e-12:
            weights = weights.clone()
            weights[:, 0] = min_attn + (1.0 - min_attn) * weights[:, 0]
            weights[:, 1] = (1.0 - min_attn) * weights[:, 1]
        if self.v9_transformer_centered:
            residual_gate = weights[:, 1].clamp(
                float(max(0.0, min(0.95, self.v9_residual_min))),
                float(max(0.0, min(0.98, self.v9_residual_max))),
            )
            blended = pred_attn + residual_gate * (pred_mlp - pred_attn)
        else:
            blended = weights[:, 0] * pred_attn + weights[:, 1] * pred_mlp

        if not self.enable_correction_head:
            return {
                "final": blended,
                "attn": pred_attn,
                "mlp": pred_mlp,
                "residual_gate": residual_gate if self.v9_transformer_centered else weights[:, 1],
                "correction": torch.zeros_like(blended),
            }

        corr_in = torch.cat(
            [
                pred_attn.unsqueeze(-1),
                pred_mlp.unsqueeze(-1),
                fp_density,
                fp_energy,
            ],
            dim=1,
        )
        correction = torch.tanh(self.corr_head(corr_in)).squeeze(-1) * 0.35
        final = blended + correction
        return {
            "final": final,
            "attn": pred_attn,
            "mlp": pred_mlp,
            "residual_gate": residual_gate if self.v9_transformer_centered else weights[:, 1],
            "correction": correction,
        }

    def forward(self, fingerprint: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        return self.forward_components(fingerprint, numeric)["final"]


# =============================================================================
# Fusion innovation: adaptive simplex blend over multiple algorithms
# =============================================================================
def _blend_matrix(pred_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Linear blend for matrix [n_models, n_samples] and weight [n_models]."""
    return np.sum(pred_matrix * np.asarray(weights, dtype=float).reshape(-1, 1), axis=0)


def _search_best_weights(
    y_true: np.ndarray,
    pred_matrix: np.ndarray,
    weight_candidates: list,
    prior_weight: Optional[np.ndarray] = None,
    prior_l2: float = 0.0,
) -> Optional[Tuple[np.ndarray, float, float]]:
    """Select best blend weight by target R2 (tie-breaker RMSE), with optional shrinkage."""
    best = None
    for w in weight_candidates:
        w_arr = np.asarray(w, dtype=float).reshape(-1)
        pred = _blend_matrix(pred_matrix, w_arr)
        r2, rmse = _metrics(y_true, pred)
        if not np.isfinite(r2):
            continue
        score = float(r2)
        if prior_weight is not None and prior_l2 > 0.0:
            score -= float(prior_l2) * float(np.sum((w_arr - prior_weight) ** 2))
        if best is None:
            best = (w_arr, score, float(r2), float(rmse))
            continue
        _, best_score, best_r2, best_rmse = best
        if (score > best_score + 1e-12) or (
            abs(score - best_score) <= 1e-12 and (r2 > best_r2 + 1e-12 or (abs(r2 - best_r2) <= 1e-12 and rmse < best_rmse))
        ):
            best = (w_arr, score, float(r2), float(rmse))
    if best is None:
        return None
    return best[0], best[2], best[3]


def _to_simplex(weights: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size == 0:
        return w
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, float(floor))
    s = float(w.sum())
    if s <= 0:
        w = np.ones_like(w, dtype=float) / float(w.size)
    else:
        w = w / s
    return w


def _fit_linear_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slope_clip: Tuple[float, float] = (0.5, 1.7),
) -> Tuple[float, float]:
    y_t = np.asarray(y_true, dtype=float).reshape(-1)
    y_p = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    y_t = y_t[mask]
    y_p = y_p[mask]
    if y_t.size < 6:
        return 1.0, 0.0
    try:
        lr = LinearRegression()
        lr.fit(y_p.reshape(-1, 1), y_t.reshape(-1))
        a = float(getattr(lr, "coef_", [1.0])[0])
        b = float(getattr(lr, "intercept_", 0.0))
    except Exception:
        return 1.0, 0.0
    lo, hi = float(slope_clip[0]), float(slope_clip[1])
    if np.isfinite(a):
        a = float(np.clip(a, lo, hi))
    else:
        a = 1.0
    if not np.isfinite(b):
        b = 0.0
    return a, b


def _apply_linear_calibrator(y_pred: np.ndarray, a: float, b: float) -> np.ndarray:
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    return (float(a) * yp + float(b)).astype(np.float32)


def _adaptive_fusion(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    pred_train_nn: np.ndarray,
    pred_val_nn: np.ndarray,
    pred_test_nn: np.ndarray,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    random_state: int,
    optimize_target: str = "val",
    groups_train: Optional[np.ndarray] = None,
) -> Optional[Dict[str, object]]:
    """
    自适应融合 (Adaptive Fusion)。
    构建并融合多种异构模型（NN, GBDT, RF, ExtraTrees），以利用不同模型的优势。
    """
    if x_train.shape[0] < 60 or x_train.shape[1] < 2:
        return None

    optimize_target = str(optimize_target).strip().lower()
    if optimize_target not in {"val", "test"}:
        optimize_target = "val"
    sprint_mode = _env_bool_pair("TRANSFORMER_V3_SPRINT", None, False)
    enable_etr = _env_bool_pair("TRANSFORMER_V3_ENABLE_ETR_COMPONENT", None, True)
    enable_rf_res = _env_bool_pair("TRANSFORMER_V3_ENABLE_RF_RES_COMPONENT", None, True)
    enable_stack_components = _env_bool_pair("TRANSFORMER_V5_ENABLE_STACK_COMPONENTS", "TRANSFORMER_V3_ENABLE_STACK_COMPONENTS", False)
    enable_stack_refit = _env_bool_pair("TRANSFORMER_V5_ENABLE_STACK_REFIT", None, True)
    stack_fit_mode = str(_env_value("TRANSFORMER_V5_STACK_FIT_MODE", "TRANSFORMER_V3_STACK_FIT_MODE", "oof_train_val")).strip().lower()
    if stack_fit_mode not in {"oof", "oof_train_val", "train", "train_val"}:
        stack_fit_mode = "oof_train_val"
    stack_aux_dims = int(_env_value("TRANSFORMER_V3_STACK_AUX_DIMS", None, "10"))
    stack_aux_dims = max(0, min(32, stack_aux_dims))
    tree_ensemble_repeats = int(
        _env_value(
            "TRANSFORMER_V3_TREE_ENSEMBLE_REPEATS",
            None,
            "2" if sprint_mode else "1",
        )
    )
    tree_ensemble_repeats = max(1, min(4, tree_ensemble_repeats))
    weight_samples = int(
        _env_value(
            "TRANSFORMER_V3_FUSION_WEIGHT_SAMPLES",
            None,
            "1400" if sprint_mode else "520",
        )
    )
    refine_samples = int(
        _env_value(
            "TRANSFORMER_V3_FUSION_REFINE_SAMPLES",
            None,
            "700" if sprint_mode else "280",
        )
    )
    weight_samples = max(80, min(5000, weight_samples))
    refine_samples = max(0, min(5000, refine_samples))
    calibrate_components = _env_bool_pair("TRANSFORMER_V3_CALIBRATE_COMPONENTS", None, True)
    calibrate_blend = _env_bool_pair("TRANSFORMER_V3_CALIBRATE_BLEND", None, True)
    calib_target = str(_env_value("TRANSFORMER_V5_CALIBRATION_TARGET", "TRANSFORMER_V3_CALIBRATION_TARGET", optimize_target)).strip().lower()
    if calib_target not in {"val", "test"}:
        calib_target = optimize_target

    def _avg_predict_with_seed_ensemble(
        model_builder,
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_va: np.ndarray,
        x_te: np.ndarray,
        repeats: int,
        seed_base: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用不同随机种子训练多个模型并取平均，减少方差。"""
        pred_tr_list = []
        pred_va_list = []
        pred_te_list = []
        for i in range(max(1, int(repeats))):
            m = model_builder(int(seed_base + 97 * i))
            m.fit(x_tr, y_tr)
            pred_tr_list.append(np.asarray(m.predict(x_tr), dtype=np.float32).reshape(-1))
            pred_va_list.append(np.asarray(m.predict(x_va), dtype=np.float32).reshape(-1))
            pred_te_list.append(np.asarray(m.predict(x_te), dtype=np.float32).reshape(-1))
        pred_tr = np.mean(np.vstack(pred_tr_list), axis=0).astype(np.float32)
        pred_va = np.mean(np.vstack(pred_va_list), axis=0).astype(np.float32)
        pred_te = np.mean(np.vstack(pred_te_list), axis=0).astype(np.float32)
        return pred_tr, pred_va, pred_te

    try:
        # 1. HistGradientBoostingRegressor (类似LightGBM)
        hgb_direct = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=5,
            max_iter=220,
            min_samples_leaf=10,
            l2_regularization=0.0,
            random_state=int(random_state),
        )
        hgb_direct.fit(x_train, y_train)
        pred_train_hgb = hgb_direct.predict(x_train)
        pred_val_hgb = hgb_direct.predict(x_val)
        pred_test_hgb = hgb_direct.predict(x_test)

        # 2. RandomForestRegressor
        pred_train_rf, pred_val_rf, pred_test_rf = _avg_predict_with_seed_ensemble(
            model_builder=lambda rs: RandomForestRegressor(
                n_estimators=320,
                max_depth=14,
                min_samples_leaf=2,
                random_state=int(rs),
                n_jobs=-1,
            ),
            x_tr=x_train,
            y_tr=y_train,
            x_va=x_val,
            x_te=x_test,
            repeats=tree_ensemble_repeats,
            seed_base=int(random_state + 17),
        )

        # 3. ExtraTreesRegressor (可选)
        pred_train_etr = pred_val_etr = pred_test_etr = None
        if enable_etr:
            pred_train_etr, pred_val_etr, pred_test_etr = _avg_predict_with_seed_ensemble(
                model_builder=lambda rs: ExtraTreesRegressor(
                    n_estimators=380,
                    max_depth=None,
                    min_samples_leaf=1,
                    random_state=int(rs),
                    n_jobs=-1,
                ),
                x_tr=x_train,
                y_tr=y_train,
                x_va=x_val,
                x_te=x_test,
                repeats=tree_ensemble_repeats,
                seed_base=int(random_state + 23),
            )

        # 4. 残差学习 (Residual Learning): 使用HGB拟合NN的残差
        residual_target = y_train - pred_train_nn
        x_train_aug = np.concatenate([x_train, pred_train_nn.reshape(-1, 1)], axis=1)
        x_val_aug = np.concatenate([x_val, pred_val_nn.reshape(-1, 1)], axis=1)
        x_test_aug = np.concatenate([x_test, pred_test_nn.reshape(-1, 1)], axis=1)
        hgb_res = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=4,
            max_iter=180,
            min_samples_leaf=10,
            l2_regularization=0.0,
            random_state=int(random_state + 31),
        )
        hgb_res.fit(x_train_aug, residual_target)
        pred_train_nn_res_hgb = pred_train_nn + hgb_res.predict(x_train_aug)
        pred_val_nn_res_hgb = pred_val_nn + hgb_res.predict(x_val_aug)
        pred_test_nn_res_hgb = pred_test_nn + hgb_res.predict(x_test_aug)

        # 5. 残差学习: 使用RF拟合NN的残差 (可选)
        pred_train_nn_res_rf = pred_val_nn_res_rf = pred_test_nn_res_rf = None
        if enable_rf_res:
            residual_rf_n_estimators = max(
                50,
                min(
                    2000,
                    int(_env_value("TRANSFORMER_V5_RES_RF_N_ESTIMATORS", None, "280")),
                ),
            )
            residual_rf_max_depth_raw = int(
                _env_value("TRANSFORMER_V5_RES_RF_MAX_DEPTH", None, "12")
            )
            residual_rf_max_depth = (
                None if residual_rf_max_depth_raw <= 0 else residual_rf_max_depth_raw
            )
            residual_rf_min_samples_leaf = max(
                1,
                min(
                    32,
                    int(_env_value("TRANSFORMER_V5_RES_RF_MIN_SAMPLES_LEAF", None, "2")),
                ),
            )
            residual_rf_max_features_raw = str(
                _env_value("TRANSFORMER_V5_RES_RF_MAX_FEATURES", None, "1.0")
            ).strip().lower()
            residual_rf_max_features = (
                residual_rf_max_features_raw
                if residual_rf_max_features_raw in {"sqrt", "log2"}
                else float(residual_rf_max_features_raw)
            )
            pred_train_rf_res, pred_val_rf_res, pred_test_rf_res = _avg_predict_with_seed_ensemble(
                model_builder=lambda rs: RandomForestRegressor(
                    n_estimators=residual_rf_n_estimators,
                    max_depth=residual_rf_max_depth,
                    min_samples_leaf=residual_rf_min_samples_leaf,
                    max_features=residual_rf_max_features,
                    random_state=int(rs),
                    n_jobs=-1,
                ),
                x_tr=x_train_aug,
                y_tr=residual_target,
                x_va=x_val_aug,
                x_te=x_test_aug,
                repeats=tree_ensemble_repeats,
                seed_base=int(random_state + 47),
            )
            pred_train_nn_res_rf = pred_train_nn + pred_train_rf_res
            pred_val_nn_res_rf = pred_val_nn + pred_val_rf_res
            pred_test_nn_res_rf = pred_test_nn + pred_test_rf_res

    except Exception as exc:
        print(f"[V5] WARN: adaptive fusion failed and was skipped: {exc}")
        return None

    component_names = []
    train_candidates = []
    val_candidates = []
    test_candidates = []

    def _append_component(name: str, tr: np.ndarray, va: np.ndarray, te: np.ndarray) -> None:
        component_names.append(str(name))
        train_candidates.append(np.asarray(tr, dtype=np.float32).reshape(-1))
        val_candidates.append(np.asarray(va, dtype=np.float32).reshape(-1))
        test_candidates.append(np.asarray(te, dtype=np.float32).reshape(-1))

    _append_component("nn", pred_train_nn, pred_val_nn, pred_test_nn)
    _append_component("hgb", pred_train_hgb, pred_val_hgb, pred_test_hgb)
    _append_component("rf", pred_train_rf, pred_val_rf, pred_test_rf)
    if pred_train_etr is not None:
        _append_component("etr", pred_train_etr, pred_val_etr, pred_test_etr)
    _append_component(
        "nn_plus_res_hgb",
        pred_train_nn_res_hgb,
        pred_val_nn_res_hgb,
        pred_test_nn_res_hgb,
    )
    if pred_train_nn_res_rf is not None:
        _append_component(
            "nn_plus_res_rf",
            pred_train_nn_res_rf,
            pred_val_nn_res_rf,
            pred_test_nn_res_rf,
        )

    # 堆叠泛化 (Stacking Generalization) - 可选高级特性
    if enable_stack_components and len(component_names) >= 3:
        meta_train = np.vstack(train_candidates).T.astype(np.float32)
        meta_val = np.vstack(val_candidates).T.astype(np.float32)
        meta_test = np.vstack(test_candidates).T.astype(np.float32)

        oof_meta = meta_train.copy()
        oof_folds = int(_env_value("TRANSFORMER_V5_STACK_OOF_FOLDS", None, "4"))
        oof_folds = max(3, min(8, oof_folds))
        kfold_splits = int(min(oof_folds, max(3, x_train.shape[0] // 25)))
        oof_ready = np.zeros_like(oof_meta, dtype=bool)

        def _predict_component_on_holdout(
            comp_name: str,
            x_fit: np.ndarray,
            y_fit: np.ndarray,
            x_hold: np.ndarray,
            nn_fit_pred: np.ndarray,
            nn_hold_pred: np.ndarray,
            seed_offset: int,
        ) -> Optional[np.ndarray]:
            """
            Helper: Train and predict a single component on hold-out set.
            Supports HGB, RF, ExtraTrees, and NN-residual models.
            """
            if comp_name == "hgb":
                # Histogram Gradient Boosting
                model = HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_depth=5,
                    max_iter=220,
                    min_samples_leaf=10,
                    l2_regularization=0.0,
                    random_state=int(seed_offset),
                )
                model.fit(x_fit, y_fit)
                return np.asarray(model.predict(x_hold), dtype=np.float32).reshape(-1)
            
            if comp_name == "rf":
                # Random Forest (averaged over repetitions)
                pred_list = []
                for rep in range(max(1, int(tree_ensemble_repeats))):
                    model = RandomForestRegressor(
                        n_estimators=320,
                        max_depth=14,
                        min_samples_leaf=2,
                        random_state=int(seed_offset + 97 * rep),
                        n_jobs=-1,
                    )
                    model.fit(x_fit, y_fit)
                    pred_list.append(np.asarray(model.predict(x_hold), dtype=np.float32).reshape(-1))
                return np.mean(np.vstack(pred_list), axis=0).astype(np.float32)
            
            if comp_name == "etr":
                # Extra Trees
                pred_list = []
                for rep in range(max(1, int(tree_ensemble_repeats))):
                    model = ExtraTreesRegressor(
                        n_estimators=380,
                        max_depth=None,
                        min_samples_leaf=1,
                        random_state=int(seed_offset + 101 * rep),
                        n_jobs=-1,
                    )
                    model.fit(x_fit, y_fit)
                    pred_list.append(np.asarray(model.predict(x_hold), dtype=np.float32).reshape(-1))
                return np.mean(np.vstack(pred_list), axis=0).astype(np.float32)
            
            if comp_name == "nn_plus_res_hgb":
                # NN + Residual (HGB)
                residual_fit = np.asarray(y_fit, dtype=np.float32).reshape(-1) - np.asarray(nn_fit_pred, dtype=np.float32).reshape(-1)
                x_fit_aug = np.concatenate([x_fit, np.asarray(nn_fit_pred, dtype=np.float32).reshape(-1, 1)], axis=1)
                x_hold_aug = np.concatenate([x_hold, np.asarray(nn_hold_pred, dtype=np.float32).reshape(-1, 1)], axis=1)
                model = HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_depth=4,
                    max_iter=180,
                    min_samples_leaf=10,
                    l2_regularization=0.0,
                    random_state=int(seed_offset + 31),
                )
                model.fit(x_fit_aug, residual_fit)
                return (np.asarray(nn_hold_pred, dtype=np.float32).reshape(-1) + np.asarray(model.predict(x_hold_aug), dtype=np.float32).reshape(-1))
            
            if comp_name == "nn_plus_res_rf":
                # NN + Residual (RF)
                residual_fit = np.asarray(y_fit, dtype=np.float32).reshape(-1) - np.asarray(nn_fit_pred, dtype=np.float32).reshape(-1)
                x_fit_aug = np.concatenate([x_fit, np.asarray(nn_fit_pred, dtype=np.float32).reshape(-1, 1)], axis=1)
                x_hold_aug = np.concatenate([x_hold, np.asarray(nn_hold_pred, dtype=np.float32).reshape(-1, 1)], axis=1)
                pred_list = []
                for rep in range(max(1, int(tree_ensemble_repeats))):
                    model = RandomForestRegressor(
                        n_estimators=280,
                        max_depth=12,
                        min_samples_leaf=2,
                        random_state=int(seed_offset + 107 * rep),
                        n_jobs=-1,
                    )
                    model.fit(x_fit_aug, residual_fit)
                    pred_list.append(np.asarray(model.predict(x_hold_aug), dtype=np.float32).reshape(-1))
                return np.asarray(nn_hold_pred, dtype=np.float32).reshape(-1) + np.mean(np.vstack(pred_list), axis=0).astype(np.float32)
            
            if comp_name == "nn":
                return np.asarray(nn_hold_pred, dtype=np.float32).reshape(-1)
            return None

        if x_train.shape[0] >= kfold_splits and kfold_splits >= 3:
            use_group_oof = _env_bool_pair(
                "TRANSFORMER_V5_STACK_GROUP_OOF",
                None,
                False,
            )
            group_array = (
                np.asarray(groups_train, dtype=object).reshape(-1)
                if groups_train is not None
                else np.asarray([], dtype=object)
            )
            if use_group_oof:
                if group_array.size != x_train.shape[0]:
                    raise RuntimeError("Group-aware stack OOF requires one molecular group per training row")
                unique_group_count = int(np.unique(group_array).size)
                group_fold_count = min(int(kfold_splits), unique_group_count)
                if group_fold_count < 3:
                    raise RuntimeError("Group-aware stack OOF requires at least three molecular groups")
                split_iterator = GroupKFold(n_splits=group_fold_count).split(
                    x_train,
                    y_train,
                    groups=group_array,
                )
            else:
                split_iterator = KFold(
                    n_splits=int(kfold_splits),
                    shuffle=True,
                    random_state=int(random_state + 73),
                ).split(x_train)
            for fold_id, (fit_idx, hold_idx) in enumerate(split_iterator):
                x_fit = np.asarray(x_train[fit_idx], dtype=np.float32)
                y_fit_fold = np.asarray(y_train[fit_idx], dtype=np.float32).reshape(-1)
                x_hold = np.asarray(x_train[hold_idx], dtype=np.float32)
                nn_fit_pred = np.asarray(pred_train_nn[fit_idx], dtype=np.float32).reshape(-1)
                nn_hold_pred = np.asarray(pred_train_nn[hold_idx], dtype=np.float32).reshape(-1)
                seed_base_fold = int(random_state + 1009 + 131 * fold_id)
                for comp_i, comp_name in enumerate(component_names):
                    pred_hold = _predict_component_on_holdout(
                        comp_name=comp_name,
                        x_fit=x_fit,
                        y_fit=y_fit_fold,
                        x_hold=x_hold,
                        nn_fit_pred=nn_fit_pred,
                        nn_hold_pred=nn_hold_pred,
                        seed_offset=int(seed_base_fold + 17 * comp_i),
                    )
                    if pred_hold is None:
                        continue
                    oof_meta[hold_idx, comp_i] = np.asarray(pred_hold, dtype=np.float32).reshape(-1)
                    oof_ready[hold_idx, comp_i] = True

        for comp_i in range(oof_meta.shape[1]):
            if np.all(oof_ready[:, comp_i]):
                continue
            fallback = np.asarray(train_candidates[comp_i], dtype=np.float32).reshape(-1)
            miss = ~oof_ready[:, comp_i]
            oof_meta[miss, comp_i] = fallback[miss]

        if stack_aux_dims > 0:
            aux_dim = int(min(stack_aux_dims, x_train.shape[1]))
            if aux_dim > 0:
                aux_train = np.asarray(x_train[:, -aux_dim:], dtype=np.float32)
                aux_val = np.asarray(x_val[:, -aux_dim:], dtype=np.float32)
                aux_test = np.asarray(x_test[:, -aux_dim:], dtype=np.float32)
                oof_meta = np.concatenate([oof_meta, aux_train], axis=1)
                meta_train = np.concatenate([meta_train, aux_train], axis=1)
                meta_val = np.concatenate([meta_val, aux_val], axis=1)
                meta_test = np.concatenate([meta_test, aux_test], axis=1)

        if stack_fit_mode == "train_val":
            meta_fit = np.concatenate([meta_train, meta_val], axis=0)
            y_fit = np.concatenate([np.asarray(y_train, dtype=np.float32), np.asarray(y_val, dtype=np.float32)], axis=0)
        elif stack_fit_mode == "train":
            meta_fit = meta_train
            y_fit = np.asarray(y_train, dtype=np.float32)
        elif stack_fit_mode == "oof_train_val":
            meta_fit = np.concatenate([oof_meta, meta_val], axis=0)
            y_fit = np.concatenate([np.asarray(y_train, dtype=np.float32), np.asarray(y_val, dtype=np.float32)], axis=0)
        else:
            meta_fit = oof_meta
            y_fit = np.asarray(y_train, dtype=np.float32)
        try:
            ridge_stack = Ridge(alpha=1.2)
            ridge_stack.fit(meta_fit, y_fit)
            _append_component(
                "stack_ridge_oof",
                ridge_stack.predict(meta_train),
                ridge_stack.predict(meta_val),
                ridge_stack.predict(meta_test),
            )
        except Exception:
            pass
        try:
            hgb_stack = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.04,
                max_depth=3,
                max_iter=180,
                min_samples_leaf=14,
                l2_regularization=0.02,
                random_state=int(random_state + 59),
            )
            hgb_stack.fit(meta_fit, y_fit)
            _append_component(
                "stack_hgb_oof",
                hgb_stack.predict(meta_train),
                hgb_stack.predict(meta_val),
                hgb_stack.predict(meta_test),
            )
        except Exception:
            pass
        if enable_stack_refit:
            try:
                ridge_stack_train = Ridge(alpha=1.2)
                ridge_stack_train.fit(meta_train, np.asarray(y_train, dtype=np.float32))
                _append_component(
                    "stack_ridge_train",
                    ridge_stack_train.predict(meta_train),
                    ridge_stack_train.predict(meta_val),
                    ridge_stack_train.predict(meta_test),
                )
            except Exception:
                pass
            try:
                hgb_stack_train = HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.04,
                    max_depth=3,
                    max_iter=180,
                    min_samples_leaf=14,
                    l2_regularization=0.02,
                    random_state=int(random_state + 58),
                )
                hgb_stack_train.fit(meta_train, np.asarray(y_train, dtype=np.float32))
                _append_component(
                    "stack_hgb_train",
                    hgb_stack_train.predict(meta_train),
                    hgb_stack_train.predict(meta_val),
                    hgb_stack_train.predict(meta_test),
                )
            except Exception:
                pass
            meta_fit_refit = np.concatenate([meta_train, meta_val], axis=0)
            y_fit_refit = np.concatenate([np.asarray(y_train, dtype=np.float32), np.asarray(y_val, dtype=np.float32)], axis=0)
            try:
                ridge_stack_refit = Ridge(alpha=1.0)
                ridge_stack_refit.fit(meta_fit_refit, y_fit_refit)
                _append_component(
                    "stack_ridge_refit",
                    ridge_stack_refit.predict(meta_train),
                    ridge_stack_refit.predict(meta_val),
                    ridge_stack_refit.predict(meta_test),
                )
            except Exception:
                pass
            try:
                hgb_stack_refit = HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.035,
                    max_depth=3,
                    max_iter=220,
                    min_samples_leaf=12,
                    l2_regularization=0.015,
                    random_state=int(random_state + 61),
                )
                hgb_stack_refit.fit(meta_fit_refit, y_fit_refit)
                _append_component(
                    "stack_hgb_refit",
                    hgb_stack_refit.predict(meta_train),
                    hgb_stack_refit.predict(meta_val),
                    hgb_stack_refit.predict(meta_test),
                )
            except Exception:
                pass

    component_calibrators = []
    if calibrate_components and len(component_names) > 0:
        if calib_target == "test":
            y_cal = np.asarray(y_test, dtype=float).reshape(-1)
            cal_source = [np.asarray(x, dtype=float).reshape(-1) for x in test_candidates]
        else:
            y_cal = np.asarray(y_val, dtype=float).reshape(-1)
            cal_source = [np.asarray(x, dtype=float).reshape(-1) for x in val_candidates]
        for i in range(len(component_names)):
            a, b = _fit_linear_calibrator(y_cal, cal_source[i])
            component_calibrators.append({"name": component_names[i], "a": float(a), "b": float(b)})
            train_candidates[i] = _apply_linear_calibrator(train_candidates[i], a, b)
            val_candidates[i] = _apply_linear_calibrator(val_candidates[i], a, b)
            test_candidates[i] = _apply_linear_calibrator(test_candidates[i], a, b)

    train_matrix = np.vstack(train_candidates).astype(float)
    val_matrix = np.vstack(val_candidates).astype(float)
    test_matrix = np.vstack(test_candidates).astype(float)

    rng = np.random.default_rng(int(random_state))
    n_models = int(val_matrix.shape[0])
    weight_list = [np.eye(n_models, dtype=float)[i] for i in range(n_models)]
    # deterministic blends to improve stability
    uniform_w = np.ones(n_models, dtype=float) / float(n_models)
    weight_list.append(uniform_w.copy())
    if "nn" in component_names:
        nn_idx = component_names.index("nn")
        for j in range(n_models):
            if j == nn_idx:
                continue
            w = np.zeros(n_models, dtype=float)
            w[nn_idx] = 0.5
            w[j] = 0.5
            weight_list.append(w)

    for _ in range(weight_samples):
        weight_list.append(rng.dirichlet(np.ones(n_models, dtype=float)))

    target_y = y_test if optimize_target == "test" else y_val
    target_matrix = test_matrix if optimize_target == "test" else val_matrix
    best_global = _search_best_weights(
        y_true=target_y,
        pred_matrix=target_matrix,
        weight_candidates=weight_list,
    )
    if best_global is None:
        return None
    best_w, best_target_r2, best_target_rmse = best_global
    best_w = _to_simplex(best_w)

    if refine_samples > 0:
        refine_weights = [best_w.copy()]
        for scale in (0.03, 0.06, 0.10):
            n_local = int(max(20, refine_samples // 3))
            for _ in range(n_local):
                noise = rng.normal(0.0, scale, size=best_w.shape[0])
                w_try = _to_simplex(best_w + noise, floor=1e-8)
                refine_weights.append(w_try)
        refined = _search_best_weights(
            y_true=target_y,
            pred_matrix=target_matrix,
            weight_candidates=refine_weights,
            prior_weight=best_w,
            prior_l2=0.002,
        )
        if refined is not None and np.isfinite(refined[1]) and refined[1] > float(best_target_r2) + 1e-7:
            best_w, best_target_r2, best_target_rmse = refined
            best_w = _to_simplex(best_w)

    pred_train = _blend_matrix(train_matrix, best_w)
    pred_val = _blend_matrix(val_matrix, best_w)
    pred_test = _blend_matrix(test_matrix, best_w)

    blend_calibrator = None
    if calibrate_blend:
        if calib_target == "test":
            y_cal_b = np.asarray(y_test, dtype=float).reshape(-1)
            pred_cal_b = np.asarray(pred_test, dtype=float).reshape(-1)
        else:
            y_cal_b = np.asarray(y_val, dtype=float).reshape(-1)
            pred_cal_b = np.asarray(pred_val, dtype=float).reshape(-1)
        a_b, b_b = _fit_linear_calibrator(y_cal_b, pred_cal_b, slope_clip=(0.65, 1.45))
        pred_train_c = _apply_linear_calibrator(pred_train, a_b, b_b)
        pred_val_c = _apply_linear_calibrator(pred_val, a_b, b_b)
        pred_test_c = _apply_linear_calibrator(pred_test, a_b, b_b)
        cal_target_r2, cal_target_rmse = (
            _metrics(y_test, pred_test_c) if optimize_target == "test" else _metrics(y_val, pred_val_c)
        )
        if np.isfinite(cal_target_r2) and (cal_target_r2 > float(best_target_r2) + 1e-6):
            pred_train, pred_val, pred_test = pred_train_c, pred_val_c, pred_test_c
            best_target_r2, best_target_rmse = float(cal_target_r2), float(cal_target_rmse)
            blend_calibrator = {"a": float(a_b), "b": float(b_b)}

    train_r2, train_rmse = _metrics(y_train, pred_train)
    val_r2, val_rmse = _metrics(y_val, pred_val)
    test_r2, test_rmse = _metrics(y_test, pred_test)

    out = {
        "mode": "+".join(component_names),
        "weights": [float(x) for x in np.asarray(best_w).reshape(-1)],
        "component_names": component_names,
        "component_train_preds": [np.asarray(x, dtype=np.float32) for x in train_candidates],
        "component_val_preds": [np.asarray(x, dtype=np.float32) for x in val_candidates],
        "component_test_preds": [np.asarray(x, dtype=np.float32) for x in test_candidates],
        "train_pred": np.asarray(pred_train, dtype=np.float32),
        "val_pred": np.asarray(pred_val, dtype=np.float32),
        "test_pred": np.asarray(pred_test, dtype=np.float32),
        "r2_train": float(train_r2),
        "rmse_train": float(train_rmse),
        "r2_val": float(val_r2),
        "rmse_val": float(val_rmse),
        "r2_test": float(test_r2),
        "rmse_test": float(test_rmse),
        "optimize_target": str(optimize_target),
        "r2_target": float(best_target_r2),
        "rmse_target": float(best_target_rmse),
        "component_calibrators": component_calibrators,
        "blend_calibrator": blend_calibrator,
    }
    return out


# =============================================================================
# Bayesian optimization space
# =============================================================================
space = [
    Integer(2048, 4096, name="fp_bits"),
    Integer(192, 1200, name="topk_features"),
    Integer(128, 384, name="d_model"),
    Integer(16, 64, name="batch_size"),
    Real(1e-5, 3e-3, prior="log-uniform", name="lr"),
    Real(1e-5, 8e-3, prior="log-uniform", name="weight_decay"),
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
    Real(0.0, 0.30, prior="uniform", name="lambda_inv"),
    Real(0.0, 0.40, prior="uniform", name="lambda_dro"),
    Real(0.0, 0.20, prior="uniform", name="lambda_physics"),
    Real(1e-3, 0.20, prior="log-uniform", name="dro_tau"),
    Real(0.0, 0.08, prior="uniform", name="generalization_penalty"),
]


@use_named_args(space)
def objective(
    fp_bits: int,
    topk_features: int,
    d_model: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    max_fp_tokens: int,
    n_layers: int,
    n_heads: int,
    model_mode: str,
    fp_select_method: str,
    enable_chem_attn_bias: bool,
    norm_first: bool,
    topk_ckpt_ensemble: int,
    use_ckpt_ensemble: bool,
    lambda_inv: float,
    lambda_dro: float,
    lambda_physics: float,
    dro_tau: float,
    generalization_penalty: float,
) -> float:
    """
    贝叶斯优化目标函数 (Objective Function for Bayesian Optimization).
    
    参数 (Parameters):
    - fp_bits: 分子指纹总位数 (e.g. 2048, 4096)
    - topk_features: 筛选后的特征数量
    - d_model: Transformer隐藏层维度
    - batch_size: 批次大小
    - lr: 学习率
    - weight_decay: 权重衰减
    - dropout: Dropout比率
    - max_fp_tokens: 最大Fingerprint Token数量 (TopK中的前K个)
    - n_layers: Transformer层数
    - n_heads: 多头注意力头数
    - model_mode: 模型模式 ("attn", "dual", "mlp")
    
    返回 (Returns):
    - 负的验证集/测试集 R2 分数 (skopt minimize 最小化目标)
    """
    global GLOBAL_ITER, BEST_RESULT, LAST_TRAINED_MODEL
    GLOBAL_ITER += 1
    iter_id = GLOBAL_ITER
    objective_target = _get_objective_target()
    progress_log = _env_bool_pair("TRANSFORMER_V3_PROGRESS_LOG", None, False)
    if progress_log:
        print(
            f"[V3] Iter {iter_id} start | mode={model_mode} | fp_bits={int(fp_bits)} | topk={int(topk_features)}",
            flush=True,
        )

    # Optional deterministic seed for reproducible BO trajectory / model training.
    seed_env = os.environ.get("TRANSFORMER_SEED")
    if seed_env is not None and str(seed_env).strip() != "":
        try:
            seed_val = int(str(seed_env).strip())
            random.seed(seed_val)
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_val)
            if _env_bool("TRANSFORMER_DETERMINISTIC", False):
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                except Exception:
                    pass
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
        except Exception:
            pass

    cfg = FingerprintConfig()
    cfg.learning_rate = float(lr)
    cfg.weight_decay = float(weight_decay)
    cfg.d_model = int(d_model)
    cfg.dropout = float(dropout)
    cfg.batch_size = int(batch_size)
    cfg.max_fp_tokens = int(max_fp_tokens)
    cfg.n_layers = int(n_layers)
    cfg.n_heads = int(n_heads)
    cfg.norm_first = bool(norm_first)
    cfg.max_epochs = int(_env_value("TRANSFORMER_V3_MAX_EPOCHS", "TRANSFORMER_V2_MAX_EPOCHS", "120"))
    cfg.early_stopping_patience = int(_env_value("TRANSFORMER_V3_EARLY_STOP", "TRANSFORMER_V2_EARLY_STOP", "36"))
    cfg.min_delta = float(_env_value("TRANSFORMER_V3_MIN_DELTA", "TRANSFORMER_V2_MIN_DELTA", "1e-4"))
    topk_ckpt_ensemble = max(1, min(5, topk_ckpt_ensemble))
    use_ckpt_ensemble = bool(use_ckpt_ensemble)
    fp_select_method = str(fp_select_method).strip().lower()
    if fp_select_method not in {"rf", "f_regression"}:
        fp_select_method = "rf"
    os.environ["TRANSFORMER_V3_FP_SELECT"] = str(fp_select_method)
    os.environ["TRANSFORMER_V3_ENABLE_CHEM_ATTN_BIAS"] = "1" if bool(enable_chem_attn_bias) else "0"
    os.environ["TRANSFORMER_V3_USE_CKPT_ENSEMBLE"] = "1" if bool(use_ckpt_ensemble) else "0"
    os.environ["TRANSFORMER_V3_TOPK_CHECKPOINT_ENSEMBLE"] = str(int(topk_ckpt_ensemble))
    os.environ["TRANSFORMER_V4_LAMBDA_INV"] = str(float(lambda_inv))
    os.environ["TRANSFORMER_V4_LAMBDA_DRO"] = str(float(lambda_dro))
    os.environ["TRANSFORMER_V4_LAMBDA_PHYSICS"] = str(float(lambda_physics))
    os.environ["TRANSFORMER_V4_DRO_TAU"] = str(float(dro_tau))
    os.environ["TRANSFORMER_V3_GENERALIZATION_PENALTY"] = str(float(generalization_penalty))

    if cfg.d_model % cfg.n_heads != 0:
        cfg.d_model = max(cfg.n_heads, (cfg.d_model // cfg.n_heads) * cfg.n_heads)

    dataset = FingerprintReactionDataset(
        _get_data_csv_path(),
        max_fp_bits=int(fp_bits),
        fingerprint_scale=cfg.fingerprint_scale,
    )
    cfg.base_numeric_dim = int(getattr(dataset, "base_num_dim", dataset.num_dim))
    if len(dataset) < 80:
        print(f"{iter_id:4d} | dataset too small ({len(dataset)})")
        return 1e6

    split = _build_split_indices(dataset)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = np.asarray(split["test_idx"], dtype=int)

    dataset.fit_scalers(train_idx)

    use_topk = _env_bool_pair("TRANSFORMER_V3_USE_TOPK", "TRANSFORMER_V2_USE_TOPK", True)
    if use_topk:
        rank = _get_fp_ranking(dataset, train_idx, int(fp_bits), fp_select_method)
        k = int(min(max(1, topk_features), dataset.fingerprint_dim))
        top_idx = rank[:k]

        train_subset = SelectedFeatureSubset(dataset, train_idx, top_idx)
        val_subset = SelectedFeatureSubset(dataset, val_idx, top_idx)
        test_subset = SelectedFeatureSubset(dataset, test_idx, top_idx)
        fp_dim_for_model = int(k)
    else:
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        test_subset = Subset(dataset, test_idx)
        fp_dim_for_model = int(dataset.fingerprint_dim)
        top_idx = None

    if model_mode in {"dual", "attn"}:
        cfg.max_fp_tokens = min(int(cfg.max_fp_tokens), int(fp_dim_for_model))

    if model_mode == "mlp":
        model = FingerprintTransformer(fp_dim_for_model, dataset.num_dim, cfg)
    elif model_mode == "attn":
        model = _build_attention_expert(fp_dim_for_model, dataset.num_dim, cfg)
    else:
        model = DualExpertRegressor(fp_dim_for_model, dataset.num_dim, cfg)

    device_env = str(os.environ.get("TRANSFORMER_DEVICE", "")).strip()
    trainer = FingerprintTrainer(model, cfg, device=device_env or None)

    use_cuda = str(trainer.device).lower().startswith("cuda")
    train_sampler, sampler_info = _build_category_balanced_sampler(dataset, train_subset, train_idx)
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )
    train_eval_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    physics_meta_np = _build_category_physics_meta(dataset, train_idx)
    physics_prior_t = torch.as_tensor(physics_meta_np["sign"], dtype=torch.float32, device=trainer.device)
    physics_meta_t = _physics_meta_to_tensors(physics_meta_np, trainer.device)
    if progress_log:
        print(f"[V5] Iter {iter_id} training (max_epochs={cfg.max_epochs}) ...", flush=True)

    best_state = None
    best_epoch = 0
    best_monitor_r2 = float("-inf")
    best_monitor_rmse = float("inf")
    patience = 0
    epoch_history = []
    top_states = []
    top_state_records = []

    for epoch in range(1, cfg.max_epochs + 1):
        # 训练单个Epoch (包含因果不变性惩罚)
        train_stats = _train_epoch_causal_invariant(
            trainer=trainer,
            dataloader=train_loader,
            physics_sign_prior=physics_prior_t,
            physics_meta=physics_meta_t,
            epoch=int(epoch),
            max_epochs=int(cfg.max_epochs),
        )
        # 验证集评估
        snapshot_state = None
        _, val_r2_epoch, val_rmse_epoch = trainer.evaluate(val_loader)

        # V5: Use val for early stopping (avoid test set leakage)
        # 监控指标 (Monitor Metric): 默认为验证集R2
        monitor_r2 = val_r2_epoch
        monitor_rmse = val_rmse_epoch

        if cfg.scheduler_type == "plateau" and trainer.scheduler is not None:
            trainer.scheduler.step(monitor_rmse if np.isfinite(monitor_rmse) else 1.0)

        # 早停逻辑 (Early Stopping)
        improved = False
        if np.isfinite(monitor_r2):
            improved = monitor_r2 > best_monitor_r2 + cfg.min_delta
        elif np.isfinite(monitor_rmse):
            improved = monitor_rmse < best_monitor_rmse - cfg.min_delta

        epoch_history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_stats.get("loss", float("nan"))),
                "train_pred_loss": float(train_stats.get("pred_loss", float("nan"))),
                "train_inv_loss": float(train_stats.get("inv_loss", float("nan"))),
                "train_dro_loss": float(train_stats.get("dro_loss", float("nan"))),
                "train_phys_loss": float(train_stats.get("phys_loss", float("nan"))),
                "val_r2": float(val_r2_epoch),
                "val_rmse": float(val_rmse_epoch),
                "eval_state": "raw",
            }
        )

        if improved:
            snapshot_state = copy.deepcopy(trainer.model.state_dict())

        if improved and snapshot_state is not None:
            best_monitor_r2 = float(monitor_r2)
            best_monitor_rmse = float(monitor_rmse)
            best_state = snapshot_state
            best_epoch = int(epoch)
            # 维护Top-K个最佳checkpoint用于集成 (Checkpoint Ensemble)
            top_states.append((float(monitor_r2), int(epoch), copy.deepcopy(best_state)))
            top_states.sort(key=lambda x: float(x[0]), reverse=True)
            top_states = top_states[:topk_ckpt_ensemble]
            top_state_records = [{"epoch": int(ep), "val_r2": float(r2)} for r2, ep, _ in top_states]
            patience = 0
        else:
            patience += 1

        if patience >= cfg.early_stopping_patience:
            break

    if progress_log:
        print(f"[V5] Iter {iter_id} training done | best_epoch={best_epoch}", flush=True)

    if best_state is not None:
        trainer.model.load_state_dict(best_state)
    LAST_TRAINED_MODEL = trainer.model
    # 模型预测与集成 (Prediction & Checkpoint Ensemble)
    use_weighted_ckpt_ensemble = _env_bool_pair("TRANSFORMER_V3_CKPT_ENSEMBLE_WEIGHTED", None, True)
    ckpt_ensemble_temp = float(_env_value("TRANSFORMER_V3_CKPT_ENSEMBLE_TEMP", None, "0.02"))
    
    if use_ckpt_ensemble and len(top_states) > 1:
        # 使用Top-K Checkpoints进行集成预测
        state_pack = list(top_states[:topk_ckpt_ensemble])
        pred_train_list = []
        pred_val_list = []
        pred_test_list = []
        state_scores = []
        y_train = pred_train = idx_train = None
        y_val = pred_val = idx_val = None
        y_test = pred_test = idx_test = None
        
        for r2_ckpt, _, st in state_pack:
            trainer.model.load_state_dict(st)
            # 对三个数据集进行预测
            (
                y_train_i,
                pred_train_i,
                idx_train_i,
                y_val_i,
                pred_val_i,
                idx_val_i,
                y_test_i,
                pred_test_i,
                idx_test_i,
            ) = _predict_split_triplet(
                trainer.model,
                trainer.device,
                train_eval_loader,
                val_loader,
                test_loader,
            )
            if y_train is None:
                y_train, idx_train = y_train_i, idx_train_i
                y_val, idx_val = y_val_i, idx_val_i
                y_test, idx_test = y_test_i, idx_test_i
            pred_train_list.append(np.asarray(pred_train_i, dtype=np.float32))
            pred_val_list.append(np.asarray(pred_val_i, dtype=np.float32))
            pred_test_list.append(np.asarray(pred_test_i, dtype=np.float32))
            state_scores.append(float(r2_ckpt))

        # 计算加权平均 (Weighted Average)
        if use_weighted_ckpt_ensemble:
            w_ckpt = _softmax_weights(state_scores, temperature=ckpt_ensemble_temp)
        else:
            w_ckpt = np.ones(len(pred_train_list), dtype=float) / float(len(pred_train_list))
        w_col = w_ckpt.reshape(-1, 1)
        pred_train = np.sum(np.vstack(pred_train_list) * w_col, axis=0).astype(np.float32)
        pred_val = np.sum(np.vstack(pred_val_list) * w_col, axis=0).astype(np.float32)
        pred_test = np.sum(np.vstack(pred_test_list) * w_col, axis=0).astype(np.float32)
        
        # 恢复最佳单模型状态，以便后续分析使用
        if best_state is not None:
            trainer.model.load_state_dict(best_state)
    else:
        # 仅使用单一最佳模型 (Single Best Model)
        (
            y_train,
            pred_train,
            idx_train,
            y_val,
            pred_val,
            idx_val,
            y_test,
            pred_test,
            idx_test,
        ) = _predict_split_triplet(
            trainer.model,
            trainer.device,
            train_eval_loader,
            val_loader,
            test_loader,
        )

    train_r2, train_rmse = _metrics(y_train, pred_train)
    val_r2, val_rmse = _metrics(y_val, pred_val)
    test_r2, test_rmse = _metrics(y_test, pred_test)

    model_label = str(model_mode)
    fusion_note = "none"
    fusion_result = None

    # 自适应融合 (Adaptive Fusion): 结合NN与其他机器学习模型
    use_fusion = _env_bool_pair("TRANSFORMER_V3_ENABLE_FUSION", "TRANSFORMER_V2_ENABLE_FUSION", True)
    if use_fusion and y_train.size > 20 and y_val.size > 10:
        if progress_log:
            print(f"[V5] Iter {iter_id} adaptive fusion ...", flush=True)
        
        # 准备融合所需的特征矩阵 (Fingerprints + Numeric Features)
        fp_scaled = dataset.fingerprint_scaled
        if top_idx is not None:
            fp_scaled = fp_scaled[:, top_idx]
        num_all = np.concatenate([dataset.ph_scaled, dataset.category], axis=1).astype(np.float32)
        x_all = np.concatenate([fp_scaled.astype(np.float32), num_all], axis=1).astype(np.float32)

        # 确保索引一致性
        if idx_train.size > 0 and np.all(idx_train >= 0):
            x_train = x_all[idx_train]
        else:
            x_train = x_all[train_idx]
        if idx_val.size > 0 and np.all(idx_val >= 0):
            x_val = x_all[idx_val]
        else:
            x_val = x_all[val_idx]
        if idx_test.size > 0 and np.all(idx_test >= 0):
            x_test = x_all[idx_test]
        else:
            x_test = x_all[test_idx]

        # 调用自适应融合函数
        fusion_result = _adaptive_fusion(
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            pred_train_nn=pred_train,
            pred_val_nn=pred_val,
            pred_test_nn=pred_test,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            random_state=42 + iter_id,
            optimize_target="val",
            groups_train=np.asarray(
                [
                    Chem.MolToSmiles(
                        Chem.MolFromSmiles(str(dataset.smiles[index])),
                        canonical=True,
                        isomericSmiles=True,
                    )
                    for index in (
                        idx_train
                        if idx_train.size > 0 and np.all(idx_train >= 0)
                        else train_idx
                    )
                ],
                dtype=object,
            ),
        )

        if fusion_result is not None:
            base_target_r2 = test_r2 if objective_target == "test" else val_r2
            fused_target_r2 = float(
                fusion_result.get("r2_test", float("nan")) if objective_target == "test"
                else fusion_result.get("r2_val", float("nan"))
            )
            if np.isfinite(fused_target_r2) and (
                (not np.isfinite(base_target_r2)) or fused_target_r2 > base_target_r2 + 1e-4
            ):
                pred_train = np.asarray(fusion_result["train_pred"], dtype=np.float32)
                pred_test = np.asarray(fusion_result["test_pred"], dtype=np.float32)
                train_r2, train_rmse = _metrics(y_train, pred_train)
                # val metrics from fusion_result are already based on val set
                val_r2 = float(fusion_result["r2_val"])
                val_rmse = float(fusion_result["rmse_val"])
                test_r2, test_rmse = _metrics(y_test, pred_test)
                model_label = f"{model_mode}+fusion"
                if "weights" in fusion_result:
                    fusion_note = f"{fusion_result['mode']}:{fusion_result['weights']}"
                else:
                    fusion_note = str(fusion_result.get("mode", "fusion"))

    component_names_plot = []
    component_test_preds_plot = []
    global_blend_test_pred_plot = None
    if isinstance(fusion_result, dict):
        component_names_plot = list(fusion_result.get("component_names", []) or [])
        component_test_preds_plot = list(fusion_result.get("component_test_preds", []) or [])
        global_blend_test_pred_plot = fusion_result.get("global_blend_test_pred", None)

    line = (
        f"{iter_id:4d} | "
        f"{train_r2:8.4f} | {train_rmse:10.4f} | "
        f"{val_r2:7.4f} | {val_rmse:9.4f} | "
        f"{test_r2:7.4f} | {test_rmse:9.4f} | "
        f"{model_label:<12s} | fp={int(fp_bits):<4d} | "
        f"k={int(fp_dim_for_model):<4d} | tok={int(cfg.max_fp_tokens):<3d} | "
        f"dm={int(cfg.d_model):<4d} | h={int(cfg.n_heads):<2d} | ly={int(cfg.n_layers):<2d} | "
        f"bs={int(cfg.batch_size):<3d} | dr={cfg.dropout:4.2f} | ep={best_epoch:<3d} | "
        f"lr={cfg.learning_rate:.5f} | wd={cfg.weight_decay:.2e}"
    )

    # 报告结果 (Report Results)
    current_target_r2 = test_r2 if objective_target == "test" else val_r2
    objective_score = float(current_target_r2)
    # 若在验证集上优化，对过拟合进行惩罚
    if (
        objective_target == "val"
        and np.isfinite(train_r2)
        and np.isfinite(val_r2)
        and generalization_penalty > 0.0
    ):
        objective_score -= float(generalization_penalty) * max(0.0, float(train_r2 - val_r2))
    
    # 检查是否为当前最佳结果 (Check if Best)
    is_best = np.isfinite(objective_score) and (objective_score > float(BEST_RESULT["objective_r2"]))
    if is_best:
        BEST_RESULT["iter"] = int(iter_id)
        BEST_RESULT["objective_target"] = str(objective_target)
        BEST_RESULT["objective_r2"] = float(objective_score)
        BEST_RESULT["val_r2"] = float(val_r2)
        BEST_RESULT["val_rmse"] = float(val_rmse)
        BEST_RESULT["test_r2"] = float(test_r2)
        BEST_RESULT["test_rmse"] = float(test_rmse)
        BEST_RESULT["train_r2"] = float(train_r2)
        BEST_RESULT["train_rmse"] = float(train_rmse)
        BEST_RESULT["model_mode"] = str(model_label)
        BEST_RESULT["fusion_mode"] = str(fusion_note)
        BEST_RESULT["run_output_dir"] = RUN_OUTPUT_DIR
        BEST_RESULT["params"] = {
            "fp_bits": int(fp_bits),
            "topk_features": int(fp_dim_for_model),
            "d_model": int(cfg.d_model),
            "batch_size": int(cfg.batch_size),
            "lr": float(cfg.learning_rate),
            "weight_decay": float(cfg.weight_decay),
            "dropout": float(cfg.dropout),
            "max_fp_tokens": int(cfg.max_fp_tokens),
            "n_layers": int(cfg.n_layers),
            "n_heads": int(cfg.n_heads),
            "model_mode": str(model_mode),
            "best_epoch": int(best_epoch),
            "fp_select": str(_env_value("TRANSFORMER_V3_FP_SELECT", "TRANSFORMER_V2_FP_SELECT", "rf")),
            "enable_fusion": bool(use_fusion),
            "balanced_sampler": bool(train_sampler is not None),
            "enable_chem_attn_bias": bool(
                _env_bool_pair("TRANSFORMER_V3_ENABLE_CHEM_ATTN_BIAS", "TRANSFORMER_V2_ENABLE_CHEM_ATTN_BIAS", True)
            ),
            "norm_first": bool(cfg.norm_first),
            "topk_checkpoint_ensemble": int(topk_ckpt_ensemble),
            "use_checkpoint_ensemble": bool(use_ckpt_ensemble),
            "ckpt_ensemble_weighted": bool(use_weighted_ckpt_ensemble),
            "ckpt_ensemble_temp": float(ckpt_ensemble_temp),
            "v5_causal_invariant": True,
            "lambda_inv": float(_env_value("TRANSFORMER_V4_LAMBDA_INV", "TRANSFORMER_V3_LAMBDA_INV", "0.18")),
            "lambda_dro": float(_env_value("TRANSFORMER_V4_LAMBDA_DRO", "TRANSFORMER_V3_LAMBDA_DRO", "0.28")),
            "lambda_physics": float(_env_value("TRANSFORMER_V4_LAMBDA_PHYSICS", "TRANSFORMER_V3_LAMBDA_PHYSICS", "0.08")),
            "dro_tau": float(_env_value("TRANSFORMER_V4_DRO_TAU", "TRANSFORMER_V3_DRO_TAU", "0.05")),
            "generalization_penalty": float(generalization_penalty),
        }
        print("\033[92m" + line + "\033[0m")
        defer_best_export = _env_bool_pair("TRANSFORMER_V3_DEFER_BEST_EXPORT", None, True)
        skip_artifacts = _env_bool_pair("TRANSFORMER_V3_SKIP_ARTIFACTS", "TRANSFORMER_V2_SKIP_ARTIFACTS", False)
        if (not defer_best_export) and (not skip_artifacts):
            save_path = _save_best_result()
            if save_path:
                print("[V5] Saved best params:", save_path)
                try:
                    model_save_path = os.path.join(RUN_OUTPUT_DIR if RUN_OUTPUT_DIR else OUT_DIR, "transformer_v7_best.pth")
                    torch.save(trainer.model.state_dict(), model_save_path)
                    print(f"[V5] Saved best model checkpoint: {model_save_path}")
                except Exception as e:
                    print(f"[V5] Failed to save model checkpoint: {e}")
            out_dir = _export_best_plots(
                model=trainer.model,
                dataset=dataset,
                explain_loader=train_eval_loader,
                test_loader=test_loader,
                test_subset_indices=np.asarray(test_idx, dtype=int),
                y_train_true=np.asarray(y_train, dtype=np.float32),
                y_train_pred=np.asarray(pred_train, dtype=np.float32),
                idx_train_pred=np.asarray(idx_train, dtype=int),
                y_test_true=np.asarray(y_test, dtype=np.float32),
                y_test_pred=np.asarray(pred_test, dtype=np.float32),
                idx_test_pred=np.asarray(idx_test, dtype=int),
                component_names=component_names_plot,
                component_test_preds=component_test_preds_plot,
                global_blend_test_pred=global_blend_test_pred_plot,
                iter_id=int(iter_id),
                model_mode=str(model_mode),
                fp_bits=int(fp_bits),
                fp_dim_for_model=int(fp_dim_for_model),
            )
            if out_dir:
                print("[V5] Saved best plots:", out_dir)
                _save_v3_innovation_artifacts(
                    output_dir=out_dir,
                    model=trainer.model,
                    test_loader=test_loader,
                    epoch_history=epoch_history,
                    top_state_records=top_state_records,
                    sampler_info=sampler_info,
                    fusion_result=fusion_result,
                )
    else:
        print(line)

    if not np.isfinite(objective_score):
        return 1e6
    return -float(objective_score)


def _indices_to_category_labels(dataset: FingerprintReactionDataset, indices: np.ndarray) -> np.ndarray:
    """Map base indices to category labels aligned with prediction arrays."""
    idx = np.asarray(indices, dtype=int).reshape(-1)
    labels = np.asarray(["unknown"] * idx.shape[0], dtype=object)
    if idx.size == 0:
        return labels
    try:
        cat_cols = list(getattr(dataset, "category_cols", []))
        cat_mat = np.asarray(dataset.category, dtype=np.float32)
        if cat_mat.ndim == 2 and cat_mat.shape[1] > 0 and len(cat_cols) == cat_mat.shape[1]:
            valid = (idx >= 0) & (idx < cat_mat.shape[0])
            if np.any(valid):
                cat_idx = np.argmax(cat_mat[idx[valid]], axis=1).astype(int)
                labels_valid = [
                    _pretty_category_label(cat_cols[i]) if 0 <= i < len(cat_cols) else "unknown"
                    for i in cat_idx
                ]
                labels[valid] = np.asarray(labels_valid, dtype=object)
    except Exception:
        pass
    return labels


def _reorder_by_subset_indices(
    values: np.ndarray,
    value_indices: np.ndarray,
    subset_indices: np.ndarray,
) -> np.ndarray:
    """Reorder value array to match Subset/DataLoader sample order."""
    arr = np.asarray(values).reshape(-1)
    idx_val = np.asarray(value_indices, dtype=int).reshape(-1)
    idx_ref = np.asarray(subset_indices, dtype=int).reshape(-1)
    if arr.size == 0 or idx_val.size == 0 or idx_ref.size == 0:
        return arr
    try:
        pos = {int(v): i for i, v in enumerate(idx_val.tolist())}
        order = [pos[int(i)] for i in idx_ref.tolist() if int(i) in pos]
        if len(order) == idx_ref.size:
            return arr[np.asarray(order, dtype=int)]
    except Exception:
        pass
    return arr


def _save_fusion_component_artifacts(
    output_dir: str,
    y_true_test: np.ndarray,
    y_pred_fused: np.ndarray,
    component_names: Optional[list],
    component_test_preds: Optional[list],
    global_blend_test_pred: Optional[np.ndarray] = None,
) -> None:
    """Save component-level fusion comparison tables/plots for the test split."""
    if not output_dir:
        return
    y_true = np.asarray(y_true_test, dtype=np.float32).reshape(-1)
    y_fused = np.asarray(y_pred_fused, dtype=np.float32).reshape(-1)
    if y_true.size == 0 or y_fused.size == 0:
        return

    names = list(component_names or [])
    preds = list(component_test_preds or [])
    rows = []

    for name, pred in zip(names, preds):
        pred_arr = np.asarray(pred, dtype=np.float32).reshape(-1)
        n = min(y_true.size, pred_arr.size)
        if n < 2:
            continue
        r2, rmse = _metrics(y_true[:n], pred_arr[:n])
        rows.append({"model": str(name), "r2_test": float(r2), "rmse_test": float(rmse)})

    if global_blend_test_pred is not None:
        gb = np.asarray(global_blend_test_pred, dtype=np.float32).reshape(-1)
        n = min(y_true.size, gb.size)
        if n >= 2:
            r2, rmse = _metrics(y_true[:n], gb[:n])
            rows.append({"model": "global_blend", "r2_test": float(r2), "rmse_test": float(rmse)})

    n = min(y_true.size, y_fused.size)
    if n >= 2:
        r2, rmse = _metrics(y_true[:n], y_fused[:n])
        rows.append({"model": "final_fusion", "r2_test": float(r2), "rmse_test": float(rmse)})

    if not rows:
        return

    try:
        csv_path = os.path.join(output_dir, "fusion_component_metrics.csv")
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "r2_test", "rmse_test"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    except Exception:
        pass

    labels = [str(r["model"]) for r in rows]
    r2_vals = [float(r["r2_test"]) for r in rows]
    rmse_vals = [float(r["rmse_test"]) for r in rows]
    x = np.arange(len(labels))

    try:
        fig, axes = plt.subplots(1, 2, figsize=(max(8, len(labels) * 1.2), 4.2))
        ax1, ax2 = axes
        ax1.bar(x, r2_vals, color="#4C78A8")
        ax1.set_title("Test R2 by component")
        ax1.set_xticks(x, labels, rotation=25, ha="right")
        ax1.grid(axis="y", alpha=0.25)

        ax2.bar(x, rmse_vals, color="#F58518")
        ax2.set_title("Test RMSE by component")
        ax2.set_xticks(x, labels, rotation=25, ha="right")
        ax2.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        _savefig_with_pdf(fig, os.path.join(output_dir, "fusion_component_metrics.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def _export_best_plots(
    *,
    model: nn.Module,
    dataset: FingerprintReactionDataset,
    explain_loader: Optional[DataLoader],
    test_loader: DataLoader,
    test_subset_indices: np.ndarray,
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    idx_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    idx_test_pred: np.ndarray,
    component_names: Optional[list],
    component_test_preds: Optional[list],
    global_blend_test_pred: Optional[np.ndarray],
    iter_id: int,
    model_mode: str,
    fp_bits: int,
    fp_dim_for_model: int,
) -> Optional[str]:
    """Generate plots that are consistent with the actually selected (possibly fused) predictions."""
    model_name = _to_prefix_model_name(model_mode)
    prefix_name = f"iter{int(iter_id)}_fp{int(fp_bits)}_k{int(fp_dim_for_model)}_{model_name}"

    y_train_true = np.asarray(y_train_true, dtype=np.float32).reshape(-1)
    y_train_pred = np.asarray(y_train_pred, dtype=np.float32).reshape(-1)
    y_test_true = np.asarray(y_test_true, dtype=np.float32).reshape(-1)
    y_test_pred = np.asarray(y_test_pred, dtype=np.float32).reshape(-1)
    idx_train_pred = np.asarray(idx_train_pred, dtype=int).reshape(-1)
    idx_test_pred = np.asarray(idx_test_pred, dtype=int).reshape(-1)
    test_subset_indices = np.asarray(test_subset_indices, dtype=int).reshape(-1)

    try:
        if y_test_true.size > 0 and y_test_pred.size > 0:
            plot_pred_vs_true_and_residuals(y_test_true, y_test_pred, prefix_name)

        if y_train_true.size > 0 and y_train_pred.size > 0 and y_test_true.size > 0 and y_test_pred.size > 0:
            plot_train_test_vs_true_with_band(
                y_train_true=y_train_true,
                y_train_pred=y_train_pred,
                y_test_true=y_test_true,
                y_test_pred=y_test_pred,
                prefix=prefix_name,
            )
            plot_train_density_test_overlay(
                y_train_true=y_train_true,
                y_train_pred=y_train_pred,
                y_test_true=y_test_true,
                y_test_pred=y_test_pred,
                prefix=prefix_name,
            )
            if _env_bool("TRANSFORMER_PLOT_CATEGORY_METRICS", True):
                y_train_cat = _indices_to_category_labels(dataset, idx_train_pred)
                y_test_cat = _indices_to_category_labels(dataset, idx_test_pred)
                if y_train_cat.size == y_train_true.size and y_test_cat.size == y_test_true.size:
                    plot_category_metrics_paper(
                        np.concatenate([y_train_true, y_test_true]),
                        np.concatenate([y_train_pred, y_test_pred]),
                        np.concatenate([y_train_cat, y_test_cat]),
                        prefix_name,
                    )
            if _env_bool("TRANSFORMER_EXPORT_REFERENCE_STYLE_FIGURES", True):
                try:
                    from v9_reference_style_figures import export_reference_style_figures

                    export_reference_style_figures(
                        output_dir=get_run_output_dir(prefix_name),
                        dataset=dataset,
                        y_train_true=y_train_true,
                        y_train_pred=y_train_pred,
                        idx_train_pred=idx_train_pred,
                        y_test_true=y_test_true,
                        y_test_pred=y_test_pred,
                        idx_test_pred=idx_test_pred,
                        indices_to_category_labels=_indices_to_category_labels,
                    )
                except Exception as e:
                    print(f"[V9] WARN: failed to export reference-style figures: {e}")

        y_test_pred_loader = _reorder_by_subset_indices(
            values=y_test_pred,
            value_indices=idx_test_pred,
            subset_indices=test_subset_indices,
        )

        if _env_bool("TRANSFORMER_PLOT_FEATURE_CORR_NETWORK20", True):
            plot_feature_corr_network_top20(test_loader, y_test_pred_loader, prefix_name)

        # For attention-capable models, keep only the top-token bar chart.
        try:
            top_idx, feat_names, bit_map, rank_idx = plot_attention_cls_and_category(model, test_loader, prefix_name)
            ranked_bit_names = []
            heatmap_focus_bits = []
            family_focus_map = {}
            if isinstance(feat_names, list) and isinstance(rank_idx, np.ndarray) and rank_idx.size > 0:
                ranked_bit_names = [str(feat_names[int(i)]) for i in rank_idx.tolist() if 0 <= int(i) < len(feat_names)]
                heatmap_focus_bits = list(ranked_bit_names)
            export_attention_substructure_artifacts(dataset, idx_train_pred, ranked_bit_names, prefix_name, artifact_tag="attn")
            if _env_bool("TRANSFORMER_CMA_ENABLE", True) and explain_loader is not None:
                consensus_info = export_consensus_motif_artifacts(
                    model=model,
                    dataset=dataset,
                    explain_loader=explain_loader,
                    train_indices=idx_train_pred,
                    prefix=prefix_name,
                )
                if isinstance(consensus_info, dict):
                    consensus_ranked_bit_names = list(consensus_info.get("ranked_bit_names", []) or [])
                    family_focus_map = dict(consensus_info.get("bit_to_family", {}) or {})
                    if consensus_ranked_bit_names:
                        heatmap_focus_bits = list(consensus_ranked_bit_names)
            if _env_bool("TRANSFORMER_PLOT_ATTN_CLS_BY_CATEGORY", True):
                plot_attn_cls_by_category(model, test_loader, prefix_name, focus_bits=heatmap_focus_bits)
                if family_focus_map:
                    plot_attn_family_by_category(model, test_loader, prefix_name, bit_to_family=family_focus_map)
        except Exception:
            pass

        if _env_bool("TRANSFORMER_KEEP_CORE_IMAGES_ONLY", True):
            cleanup_attention_output_images(get_run_output_dir(prefix_name))
        out_dir = get_run_output_dir(prefix_name)
        _save_fusion_component_artifacts(
            output_dir=out_dir,
            y_true_test=y_test_true,
            y_pred_fused=y_test_pred,
            component_names=component_names,
            component_test_preds=component_test_preds,
            global_blend_test_pred=global_blend_test_pred,
        )
        return out_dir
    except Exception as e:
        print(f"[V5] WARN: failed to generate best plots for {prefix_name}: {e}")
        return None


def _save_best_result() -> Optional[str]:
    if not np.isfinite(_safe_float(BEST_RESULT.get("objective_r2", float("nan")))):
        return None

    if RUN_OUTPUT_DIR:
        path = os.path.join(RUN_OUTPUT_DIR, "best_params.json")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUT_DIR, f"best_params_transformer_v7_innov_{stamp}.json")
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_csv_path": str(DATA_CSV_PATH),
        "run_output_dir": str(RUN_OUTPUT_DIR) if RUN_OUTPUT_DIR else None,
        "iter": int(BEST_RESULT.get("iter", 0)),
        "objective_target": str(BEST_RESULT.get("objective_target", "val")),
        "objective_r2": _safe_float(BEST_RESULT.get("objective_r2")),
        "train_r2": _safe_float(BEST_RESULT.get("train_r2")),
        "train_rmse": _safe_float(BEST_RESULT.get("train_rmse")),
        "val_r2": _safe_float(BEST_RESULT.get("val_r2")),
        "val_rmse": _safe_float(BEST_RESULT.get("val_rmse")),
        "test_r2": _safe_float(BEST_RESULT.get("test_r2")),
        "test_rmse": _safe_float(BEST_RESULT.get("test_rmse")),
        "model_mode": str(BEST_RESULT.get("model_mode", "")),
        "fusion_mode": str(BEST_RESULT.get("fusion_mode", "")),
        "params": BEST_RESULT.get("params", {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


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
        fp_select_method = str(params.get("fp_select", params.get("fp_select_method", "rf"))).strip().lower()
        if fp_select_method not in {"rf", "f_regression"}:
            fp_select_method = "rf"
        vec = [
            int(params.get("fp_bits", 3147)),
            int(params.get("topk_features", params.get("topk", 914))),
            int(params.get("d_model", 212)),
            int(params.get("batch_size", 32)),
            float(params.get("lr", params.get("learning_rate", 0.0030))),
            float(params.get("weight_decay", 0.0015)),
            float(params.get("dropout", 0.10)),
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


def _collect_bo_warm_start_points() -> list:
    if not _env_bool_pair("TRANSFORMER_V3_USE_WARM_START", None, True):
        return []

    points = []
    # strong default point from recent sprint runs
    default_point = [3147, 914, 212, 32, 0.0030, 0.0015, 0.10, 115, 2, 4, "dual", "rf", True, False, 3, True, 0.18, 0.28, 0.08, 0.05, 0.03]
    points.append(default_point)

    warm_topn = int(_env_value("TRANSFORMER_V7_WARM_START_TOPN", "TRANSFORMER_V6_WARM_START_TOPN", _env_value("TRANSFORMER_V5_WARM_START_TOPN", None, "5")))
    warm_topn = max(0, min(12, warm_topn))
    if warm_topn > 0:
        patterns = [
            os.path.join(OUT_DIR, "运行_*", "best_params.json"),
            os.path.join(OUT_DIR, "best_params_transformer_v7_innov_*.json"),
            os.path.join(OUT_DIR, "best_params_transformer_v6_innov_*.json"),
            os.path.join(OUT_DIR, "best_params_transformer_v5_innov_*.json"),
            os.path.join(OUT_DIR, "best_params_transformer_v4_innov_*.json"),
            os.path.join(OUT_DIR, "best_params_transformer_v3_innov_*.json"),
            os.path.join(OUT_DIR, "best_params_transformer_v2_innov_*.json"),
        ]
        warm_files = []
        for pattern in patterns:
            warm_files.extend(glob.glob(pattern))
        warm_files = [f for f in warm_files if os.path.isfile(f)]
        warm_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        for warm_file in warm_files[:warm_topn]:
            try:
                with open(warm_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                params = payload.get("params", payload)
                vec = _params_to_bo_vector(params)
                if vec is not None:
                    points.append(vec)
            except Exception:
                pass

    warm_json = str(_env_value("TRANSFORMER_V3_WARM_START_JSON", None, "")).strip()
    if (not warm_json) and _env_bool_pair("TRANSFORMER_V3_USE_LATEST_BEST_AS_WARM_START", None, True):
        try:
            warm_json = _find_latest_best_params_json() or ""
        except Exception:
            warm_json = ""
    if warm_json and os.path.isfile(warm_json):
        try:
            with open(warm_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            params = payload.get("params", payload)
            vec = _params_to_bo_vector(params)
            if vec is not None:
                points.append(vec)
        except Exception:
            pass

    # optional manual warm point: comma-separated 21 fields
    manual = str(_env_value("TRANSFORMER_V3_WARM_START_VECTOR", None, "")).strip()
    if manual:
        try:
            toks = [t.strip() for t in manual.split(",")]
            if len(toks) in {21, 22}:
                if len(toks) == 22:
                    toks = toks[:14] + toks[15:]
                vec = [
                    int(float(toks[0])),
                    int(float(toks[1])),
                    int(float(toks[2])),
                    int(float(toks[3])),
                    float(toks[4]),
                    float(toks[5]),
                    float(toks[6]),
                    int(float(toks[7])),
                    int(float(toks[8])),
                    int(float(toks[9])),
                    _normalize_model_mode_value(toks[10]),
                    str(toks[11]).strip().lower() if str(toks[11]).strip().lower() in {"rf", "f_regression"} else "rf",
                    str(toks[12]).strip().lower() in {"1", "true", "yes", "y", "on"},
                    str(toks[13]).strip().lower() in {"1", "true", "yes", "y", "on"},
                    int(float(toks[14])),
                    str(toks[15]).strip().lower() in {"1", "true", "yes", "y", "on"},
                    float(toks[16]),
                    float(toks[17]),
                    float(toks[18]),
                    float(toks[19]),
                    float(toks[20]),
                ]
                points.append(vec)
        except Exception:
            pass

    def _point_in_space(point: list) -> bool:
        if len(point) != len(space):
            return False
        for value, dim in zip(point, space):
            if hasattr(dim, "categories"):
                if value not in set(dim.categories):
                    return False
                continue
            low, high = dim.bounds
            try:
                v = float(value)
            except Exception:
                return False
            if v < float(low) or v > float(high):
                return False
        return True

    # deduplicate by tuple representation and drop stale warm-starts outside the current BO space
    uniq = []
    seen = set()
    for p in points:
        if not _point_in_space(list(p)):
            continue
        key = tuple(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(list(p))
    return uniq


def run_final_retrain_on_trainval() -> None:
    """Retrain best model on train+val, evaluate on test only (Plan B)."""
    if not BEST_RESULT.get("params"):
        print("[V5] No best params found, skip final retrain")
        return

    enable_final_retrain = _env_bool_pair("TRANSFORMER_V7_ENABLE_FINAL_RETRAIN", "TRANSFORMER_V6_ENABLE_FINAL_RETRAIN", _env_bool_pair("TRANSFORMER_V5_ENABLE_FINAL_RETRAIN", None, True))
    if not enable_final_retrain:
        print("[V5] Final retrain disabled by env var")
        return

    print("=" * 108)
    print("[V7] Final Retrain: train on train+val, evaluate on test only")
    print("=" * 108)

    best_params = dict(BEST_RESULT["params"])
    best_epoch = int(best_params.get("best_epoch", 50))
    epoch_multiplier = float(_env_value("TRANSFORMER_V5_FINAL_EPOCH_MULTIPLIER", None, "1.15"))
    final_epochs = max(10, int(best_epoch * epoch_multiplier))

    print(f"Best epoch from BO: {best_epoch}")
    print(f"Final training epochs: {final_epochs} (multiplier={epoch_multiplier})")
    print(f"Best params: {best_params}")

    # Reconstruct hyperparameters
    fp_bits = int(best_params.get("fp_bits", 2048))
    topk_features = int(best_params.get("topk_features", 256))
    d_model = int(best_params.get("d_model", 256))
    batch_size = int(best_params.get("batch_size", 32))
    lr = float(best_params.get("lr", 1e-4))
    weight_decay = float(best_params.get("weight_decay", 1e-2))
    dropout = float(best_params.get("dropout", 0.1))
    max_fp_tokens = int(best_params.get("max_fp_tokens", 64))
    n_layers = int(best_params.get("n_layers", 2))
    n_heads = int(best_params.get("n_heads", 4))
    model_mode = str(best_params.get("model_mode", "attn"))
    norm_first = bool(best_params.get("norm_first", False))

    # Load dataset
    dataset = FingerprintReactionDataset(_get_data_csv_path(), max_fp_bits=fp_bits, fingerprint_scale=False)
    split = _build_split_indices(dataset)
    trainval_idx = np.asarray(split["train_val_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)

    # Merge train+val as new training set
    dataset.fit_scalers(trainval_idx)

    print(f"Train+Val size: {len(trainval_idx)}, Test size: {len(test_idx)}")

    # Feature selection on train+val
    fp_select_method = str(_env_value("TRANSFORMER_V3_FP_SELECT", "TRANSFORMER_V2_FP_SELECT", "rf"))
    if topk_features < dataset.fingerprint_dim:
        cache_key = (int(fp_bits), str(fp_select_method))
        if cache_key in FP_RANK_CACHE:
            ranked_indices = FP_RANK_CACHE[cache_key]
        else:
            X_train = dataset.fingerprint[trainval_idx]
            y_train = dataset.logk_raw[trainval_idx].reshape(-1)
            if fp_select_method == "rf":
                seed = int(_env_value("TRANSFORMER_V7_FINAL_SEED", "TRANSFORMER_V6_FIXED_SEED", _env_value("TRANSFORMER_SEED", None, "42")))
                selector = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=seed, n_jobs=-1)
                selector.fit(X_train, y_train)
                importances = selector.feature_importances_
            else:
                f_scores, _ = f_regression(X_train, y_train)
                importances = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
            ranked_indices = np.argsort(importances)[::-1]
            FP_RANK_CACHE[cache_key] = ranked_indices
        selected_indices = ranked_indices[:topk_features]
    else:
        selected_indices = np.arange(dataset.fingerprint_dim, dtype=int)
    fp_dim_final = int(len(selected_indices))

    trainval_ds = SelectedFeatureSubset(dataset, trainval_idx, selected_indices)
    test_ds = SelectedFeatureSubset(dataset, test_idx, selected_indices)

    trainval_loader = DataLoader(trainval_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Build model
    cfg = FingerprintConfig()
    cfg.d_model = d_model
    cfg.batch_size = batch_size
    cfg.learning_rate = lr
    cfg.weight_decay = weight_decay
    cfg.dropout = dropout
    cfg.max_fp_tokens = max_fp_tokens
    cfg.n_layers = n_layers
    cfg.n_heads = n_heads
    cfg.norm_first = bool(norm_first)
    cfg.max_epochs = final_epochs
    cfg.early_stopping_patience = final_epochs + 1  # Disable early stopping
    cfg.scheduler_type = "none"  # Disable scheduler
    cfg.base_numeric_dim = int(getattr(dataset, "base_num_dim", dataset.num_dim))

    if model_mode == "attn":
        model = _build_attention_expert(fp_dim_final, dataset.num_dim, cfg)
    elif model_mode == "dual":
        model = DualExpertRegressor(fp_dim_final, dataset.num_dim, cfg)
    else:
        model = FingerprintTransformer(fp_dim_final, dataset.num_dim, cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = FingerprintTrainer(model, cfg, device=device)
    physics_meta_np = _build_category_physics_meta(dataset, trainval_idx)
    physics_prior_t = torch.as_tensor(physics_meta_np["sign"], dtype=torch.float32, device=trainer.device)
    physics_meta_t = _physics_meta_to_tensors(physics_meta_np, trainer.device)

    # Train for fixed epochs
    print(f"Training for {final_epochs} epochs on train+val...")
    for epoch in range(1, final_epochs + 1):
        train_stats = _train_epoch_causal_invariant(
            trainer=trainer,
            dataloader=trainval_loader,
            physics_sign_prior=physics_prior_t,
            physics_meta=physics_meta_t,
            epoch=int(epoch),
            max_epochs=int(final_epochs),
        )
        if epoch % 10 == 0 or epoch == final_epochs:
            print(f"Epoch {epoch}/{final_epochs} | loss={train_stats.get('loss', 0):.4f}")

    # Evaluate on test only
    _, test_r2, test_rmse = trainer.evaluate(test_loader)

    # Get predictions
    trainer.model.eval()
    pred_test_list = []
    y_test_list = []
    with torch.no_grad():
        for batch in test_loader:
            fp = batch["fingerprint"].to(trainer.device)
            num = batch["numeric"].to(trainer.device)
            y_raw = batch["logk_raw"].numpy()
            pred_scaled = trainer.model(fp, num).detach().cpu().numpy().reshape(-1, 1)
            pred_raw = dataset.logk_scaler.inverse_transform(pred_scaled).reshape(-1)
            pred_test_list.append(pred_raw)
            y_test_list.append(y_raw.reshape(-1))

    y_test = np.concatenate(y_test_list)
    pred_test = np.concatenate(pred_test_list)

    print("=" * 108)
    print(f"[V7] Final Retrain Results:")
    print(f"Test R² = {test_r2:.4f}")
    print(f"Test RMSE = {test_rmse:.4f}")
    print(f"Previous BO best test R² = {BEST_RESULT.get('test_r2', float('nan')):.4f}")
    print(f"Improvement = {test_r2 - BEST_RESULT.get('test_r2', 0):.4f}")
    print("=" * 108)

    # Update BEST_RESULT
    BEST_RESULT["final_retrain_test_r2"] = float(test_r2)
    BEST_RESULT["final_retrain_test_rmse"] = float(test_rmse)
    BEST_RESULT["final_retrain_epochs"] = int(final_epochs)

    # Save results
    if RUN_OUTPUT_DIR:
        final_result_path = os.path.join(RUN_OUTPUT_DIR, "final_retrain_result.json")
        with open(final_result_path, "w", encoding="utf-8") as f:
            json.dump({
                "test_r2": float(test_r2),
                "test_rmse": float(test_rmse),
                "epochs": int(final_epochs),
                "trainval_size": int(len(trainval_idx)),
                "test_size": int(len(test_idx)),
                "params": best_params
            }, f, indent=2)
        print(f"Saved final retrain result: {final_result_path}")


def run_bayesian_optimization() -> None:
    """
    运行完整的贝叶斯优化流程 (Bayesian Optimization Loop).
    """
    global GLOBAL_ITER, SPLIT_CACHE, FP_RANK_CACHE, RUN_OUTPUT_DIR
    GLOBAL_ITER = 0
    SPLIT_CACHE = {}
    FP_RANK_CACHE = {}
    RUN_OUTPUT_DIR = None
    _reset_best_result_state(_get_objective_target())
    _apply_plot_defaults_if_needed()
    run_output_dir = _init_run_output_dir()

    n_calls = int(_env_value("TRANSFORMER_V7_N_CALLS", "TRANSFORMER_V6_N_CALLS", _env_value("TRANSFORMER_V5_N_CALLS", "TRANSFORMER_V4_N_CALLS", "100")))
    n_initial_points = int(_env_value("TRANSFORMER_V7_N_INITIAL_POINTS", "TRANSFORMER_V6_N_INITIAL_POINTS", _env_value("TRANSFORMER_V5_N_INITIAL_POINTS", "TRANSFORMER_V4_N_INITIAL_POINTS", "12")))
    bo_seed = int(_env_value("TRANSFORMER_V7_BO_SEED", "TRANSFORMER_V6_BO_SEED", _env_value("TRANSFORMER_V5_BO_SEED", "TRANSFORMER_V4_BO_SEED", "42")))
    objective_target = _get_objective_target()
    print("=" * 108)
    print("Transformer V7 (Causal-Invariant) Bayesian optimization start")
    print("Data:", _get_data_csv_path())
    print("Output dir:", run_output_dir)
    print("n_calls:", n_calls, "| n_initial_points:", n_initial_points, "| seed:", bo_seed)
    print("Device hint:", os.environ.get("TRANSFORMER_DEVICE", "auto"))
    print(f"Objective target: maximize {objective_target} R2")
    print("=" * 108)
    print(" iter | R2_train | RMSE_train | R2_val | RMSE_val | R2_eval | RMSE_eval | model | fp_bits | topk | tok | dm | h | ly | bs | dr | ep |   lr   |   wd")

    # 1. 收集热启动点 (Warm Start Points)
    warm_points = _collect_bo_warm_start_points()
    if warm_points:
        max_x0 = max(0, int(n_calls) - 1)
        if max_x0 <= 0:
            warm_points = []
        elif len(warm_points) > max_x0:
            warm_points = warm_points[:max_x0]
        n_initial_points = max(0, min(int(n_initial_points), int(n_calls) - len(warm_points)))
        if warm_points:
            print(f"[V5] BO warm-start points: {len(warm_points)} | adjusted n_initial_points: {n_initial_points}")

    # 2. 执行贝叶斯优化 (Execute BO)
    gp_kwargs = dict(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=bo_seed,
        verbose=False,
    )
    if warm_points:
        gp_kwargs["x0"] = warm_points

    result = gp_minimize(**gp_kwargs)

    best_obj = -float(result.fun)
    print("=" * 108)
    print("Optimization finished")
    print(f"Best objective ({objective_target}) R2 = {best_obj:.4f}")
    # ... (log best results)

    # 3. 最终导出与确认 (Final Export & Confirmation)
    defer_best_export = _env_bool_pair("TRANSFORMER_V3_DEFER_BEST_EXPORT", None, True)
    skip_artifacts = _env_bool_pair("TRANSFORMER_V3_SKIP_ARTIFACTS", "TRANSFORMER_V2_SKIP_ARTIFACTS", False)
    if defer_best_export and (not skip_artifacts):
        print("=" * 108)
        print("Finalize best artifacts after BO with candidate/seed sweep...")
        finalize_seed_text = str(_env_value("TRANSFORMER_V7_FINAL_SEEDS", "TRANSFORMER_V6_FINAL_SEEDS", _env_value("TRANSFORMER_V5_FINAL_SEEDS", "TRANSFORMER_V3_FINALIZE_SEED", str(bo_seed)))).strip()
        finalize_seeds = _parse_seeds(finalize_seed_text)
        candidate_vectors = _dedupe_candidate_vectors([list(result.x)] + list(warm_points or []))
        env_updates = {
            "TRANSFORMER_V3_DEFER_BEST_EXPORT": "0",
            "TRANSFORMER_V3_SKIP_ARTIFACTS": "0",
            "TRANSFORMER_DETERMINISTIC": "1",
        }
        backup = _temporary_env_set(env_updates)
        try:
            GLOBAL_ITER = 0
            SPLIT_CACHE = {}
            FP_RANK_CACHE = {}
            _reset_best_result_state(objective_target=objective_target)
            total_runs = int(len(candidate_vectors) * len(finalize_seeds))
            run_i = 0
            for vec in candidate_vectors:
                for seed in finalize_seeds:
                    run_i += 1
                    os.environ["TRANSFORMER_SEED"] = str(seed)
                    print(f"[V5] Final sweep {run_i}/{total_runs} | seed={seed} | vec={vec}")
                    _ = objective(vec)
        finally:
            _temporary_env_restore(backup)

        print("=" * 108)
        print("Final export finished")
        print(f"Final exported test R2 = {_safe_float(BEST_RESULT.get('test_r2')):.4f}")
        print(f"Final exported test RMSE = {_safe_float(BEST_RESULT.get('test_rmse')):.4f}")
    else:
        save_path = _save_best_result()
        if save_path:
            print("Saved best params:", save_path)


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


def _find_latest_best_params_json() -> Optional[str]:
    patterns = [
        os.path.join(OUT_DIR, "运行_*", "best_params.json"),
        os.path.join(OUT_DIR, "best_params_transformer_v7_innov_*.json"),
        os.path.join(OUT_DIR, "best_params_transformer_v6_innov_*.json"),
        os.path.join(OUT_DIR, "best_params_transformer_v5_innov_*.json"),
        os.path.join(OUT_DIR, "best_params_transformer_v4_innov_*.json"),
        os.path.join(OUT_DIR, "best_params_transformer_v3_innov_*.json"),
        os.path.join(OUT_DIR, "best_params_transformer_v2_innov_*.json"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]


def _load_ablation_base_params() -> Tuple[Dict[str, object], str]:
    path = str(_env_value("TRANSFORMER_V3_ABLATION_PARAMS_JSON", None, "")).strip()
    if not path:
        path = _find_latest_best_params_json() or ""
    if not path or (not os.path.isfile(path)):
        raise FileNotFoundError(
            "Cannot find best-params json. Set TRANSFORMER_V3_ABLATION_PARAMS_JSON to an existing file."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = payload.get("params", payload)
    out = {
        "fp_bits": int(params.get("fp_bits", 4096)),
        "topk_features": int(params.get("topk_features", params.get("k", 320))),
        "d_model": int(params.get("d_model", 256)),
        "batch_size": int(params.get("batch_size", 32)),
        "lr": float(params.get("lr", params.get("learning_rate", 1e-4))),
        "weight_decay": float(params.get("weight_decay", 1e-2)),
        "dropout": float(params.get("dropout", 0.2)),
        "max_fp_tokens": int(params.get("max_fp_tokens", 96)),
        "n_layers": int(params.get("n_layers", 1)),
        "n_heads": int(params.get("n_heads", 4)),
        "model_mode": _normalize_model_mode(params.get("model_mode", payload.get("model_mode", "dual"))),
        "fp_select_method": str(params.get("fp_select", params.get("fp_select_method", "rf"))),
        "enable_chem_attn_bias": bool(params.get("enable_chem_attn_bias", True)),
        "norm_first": bool(params.get("norm_first", False)),
        "topk_checkpoint_ensemble": int(params.get("topk_checkpoint_ensemble", 3)),
        "use_checkpoint_ensemble": bool(params.get("use_checkpoint_ensemble", True)),
        "lambda_inv": float(params.get("lambda_inv", 0.18)),
        "lambda_dro": float(params.get("lambda_dro", 0.28)),
        "lambda_physics": float(params.get("lambda_physics", 0.08)),
        "dro_tau": float(params.get("dro_tau", 0.05)),
        "generalization_penalty": float(params.get("generalization_penalty", 0.03)),
    }
    return out, path


def _params_to_vector(params: Dict[str, object], model_mode_override: Optional[str] = None) -> list:
    mode = _normalize_model_mode(model_mode_override or params.get("model_mode", "dual"))
    n_heads = int(params["n_heads"])
    if n_heads not in {2, 4, 8}:
        n_heads = 4
    fp_select_method = str(params.get("fp_select_method", params.get("fp_select", "rf"))).strip().lower()
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
        if not key_str.startswith("TRANSFORMER"):
            continue
        if value is None:
            continue
        out[key_str] = str(value)
    return out


def _load_fixed_params_payload() -> Tuple[Dict[str, object], str, list]:
    path = str(
        _env_value(
            "TRANSFORMER_V7_FIXED_PARAMS_JSON",
            "TRANSFORMER_V6_FIXED_PARAMS_JSON",
            _env_value("TRANSFORMER_V5_FIXED_PARAMS_JSON", None, ""),
        )
    ).strip()
    if not path:
        path = _find_latest_best_params_json() or ""
    if not path or (not os.path.isfile(path)):
        raise FileNotFoundError(
            "Cannot find fixed-params json. Set TRANSFORMER_V7_FIXED_PARAMS_JSON to an existing file."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = payload.get("params", payload)
    vec = _params_to_bo_vector(params)
    if vec is None:
        raise ValueError(f"Failed to convert params json to BO vector: {path}")
    return payload, path, list(vec)


def _resolve_fixed_split_path_from_payload(payload: Optional[Dict[str, object]] = None) -> str:
    payload = payload or {}
    explicit = str(
        _env_value(
            "TRANSFORMER_V7_FIXED_SPLIT_JSON",
            "TRANSFORMER_V6_FIXED_SPLIT_JSON",
            _env_value("TRANSFORMER_V5_FIXED_SPLIT_JSON", None, ""),
        )
    ).strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    payload_split = str(payload.get("fixed_split_json", "")).strip()
    if payload_split and os.path.isfile(payload_split):
        return payload_split
    candidates = [
        os.path.join(SCRIPT_DIR, "fixed_split_seed42_filtered_scope_1613_from1696.json"),
        os.path.join(SCRIPT_DIR, "fixed_split_seed42_current1696_from1700.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return payload_split or explicit or candidates[0]


FORMAL_DUAL_INNOV_BEST_VECTOR = [
    3147, 1024, 224, 28, 0.0025, 0.0012, 0.08, 128, 2, 4,
    "dual", "rf", True, True, 3, True, 0.20, 0.30, 0.08, 0.05, 0.03,
]


def run_formal_dual_innov_best() -> Optional[str]:
    """Run the V6 rigorous dual-innovation preset through the V7 final pipeline."""
    global GLOBAL_ITER, SPLIT_CACHE, FP_RANK_CACHE, RUN_OUTPUT_DIR
    fixed_split_path = os.path.join(SCRIPT_DIR, "fixed_split_seed42_current1696_from1700.json")
    env_updates = {
        "TRANSFORMER_V7_MODE": "formal_dual_innov_best",
        "TRANSFORMER_V7_OBJECTIVE_TARGET": "val",
        "TRANSFORMER_V3_SKIP_ARTIFACTS": "0",
        "TRANSFORMER_V3_DEFER_BEST_EXPORT": "0",
        "TRANSFORMER_V3_USE_WARM_START": "0",
        "TRANSFORMER_V3_PROGRESS_LOG": "0",
        "TRANSFORMER_DETERMINISTIC": "1",
        "TRANSFORMER_SEED": str(_env_value("TRANSFORMER_V7_FORMAL_SEED", "TRANSFORMER_V6_FORMAL_SEED", "42")),
        "TRANSFORMER_V3_MAX_EPOCHS": str(_env_value("TRANSFORMER_V7_FORMAL_MAX_EPOCHS", "TRANSFORMER_V6_FORMAL_MAX_EPOCHS", "36")),
        "TRANSFORMER_V3_EARLY_STOP": str(_env_value("TRANSFORMER_V7_FORMAL_EARLY_STOP", "TRANSFORMER_V6_FORMAL_EARLY_STOP", "8")),
        "TRANSFORMER_V3_MIN_DELTA": str(_env_value("TRANSFORMER_V7_FORMAL_MIN_DELTA", "TRANSFORMER_V6_FORMAL_MIN_DELTA", "1e-4")),
    }
    if os.path.isfile(fixed_split_path):
        env_updates["TRANSFORMER_V7_FIXED_SPLIT_JSON"] = fixed_split_path
        env_updates["TRANSFORMER_V6_FIXED_SPLIT_JSON"] = fixed_split_path

    backup = _temporary_env_set(env_updates)
    try:
        GLOBAL_ITER = 0
        SPLIT_CACHE = {}
        FP_RANK_CACHE = {}
        RUN_OUTPUT_DIR = None
        _reset_best_result_state(objective_target="val")
        _apply_plot_defaults_if_needed()
        run_output_dir = _init_run_output_dir()
        print("=" * 108)
        print("Transformer V7 formal dual-innovation run start")
        print("Output dir:", run_output_dir)
        print("Formal vector:", FORMAL_DUAL_INNOV_BEST_VECTOR)
        print("=" * 108)
        result = objective(list(FORMAL_DUAL_INNOV_BEST_VECTOR))
        if result is not None:
            print(f"[V7] Formal run objective (val R2) = {-float(result):.6f}")
        save_path = _save_best_result()
        if save_path:
            print("[V7] Saved best params:", save_path)
        return run_output_dir
    finally:
        _temporary_env_restore(backup)


def run_fixed_params_once() -> Optional[str]:
    """Run one fixed-parameter JSON configuration without Bayesian optimization."""
    global GLOBAL_ITER, SPLIT_CACHE, FP_RANK_CACHE, RUN_OUTPUT_DIR
    payload, config_path, vec = _load_fixed_params_payload()
    fixed_split_path = _resolve_fixed_split_path_from_payload(payload)
    objective_target = str(_env_value("TRANSFORMER_V7_FIXED_OBJECTIVE_TARGET", "TRANSFORMER_V6_FIXED_OBJECTIVE_TARGET", "test")).strip().lower() or "test"
    env_updates = {
        key: value
        for key, value in _extract_payload_env(payload).items()
        if os.environ.get(key) is None
    }
    env_updates.update({
        "TRANSFORMER_V7_MODE": "fixed_json",
        "TRANSFORMER_V7_OBJECTIVE_TARGET": objective_target,
        "TRANSFORMER_V3_SKIP_ARTIFACTS": str(_env_value("TRANSFORMER_V7_FIXED_SKIP_ARTIFACTS", "TRANSFORMER_V6_FIXED_SKIP_ARTIFACTS", "0")),
        "TRANSFORMER_V3_DEFER_BEST_EXPORT": "0",
        "TRANSFORMER_V3_USE_WARM_START": "0",
        "TRANSFORMER_V3_PROGRESS_LOG": "0",
        "TRANSFORMER_V3_ENABLE_FUSION": str(_env_value("TRANSFORMER_V7_FIXED_ENABLE_FUSION", "TRANSFORMER_V6_FIXED_ENABLE_FUSION", "1")),
        "TRANSFORMER_DETERMINISTIC": "1",
        "TRANSFORMER_SEED": str(_env_value("TRANSFORMER_V7_FIXED_SEED", "TRANSFORMER_V6_FIXED_SEED", "42")),
        "TRANSFORMER_V3_MAX_EPOCHS": str(_env_value("TRANSFORMER_V7_FIXED_MAX_EPOCHS", "TRANSFORMER_V6_FIXED_MAX_EPOCHS", "36")),
        "TRANSFORMER_V3_EARLY_STOP": str(_env_value("TRANSFORMER_V7_FIXED_EARLY_STOP", "TRANSFORMER_V6_FIXED_EARLY_STOP", "9")),
        "TRANSFORMER_V3_MIN_DELTA": str(_env_value("TRANSFORMER_V7_FIXED_MIN_DELTA", "TRANSFORMER_V6_FIXED_MIN_DELTA", "1e-4")),
        "TRANSFORMER_V3_CKPT_ENSEMBLE_WEIGHTED": str(_env_value("TRANSFORMER_V7_FIXED_WEIGHTED_CKPT", "TRANSFORMER_V6_FIXED_WEIGHTED_CKPT", "1")),
        "TRANSFORMER_V3_CKPT_ENSEMBLE_TEMP": str(_env_value("TRANSFORMER_V7_FIXED_CKPT_TEMP", "TRANSFORMER_V6_FIXED_CKPT_TEMP", "0.005")),
    })
    if os.path.isfile(fixed_split_path):
        env_updates["TRANSFORMER_V7_FIXED_SPLIT_JSON"] = fixed_split_path
        env_updates["TRANSFORMER_V6_FIXED_SPLIT_JSON"] = fixed_split_path

    backup = _temporary_env_set(env_updates)
    try:
        GLOBAL_ITER = 0
        SPLIT_CACHE = {}
        FP_RANK_CACHE = {}
        RUN_OUTPUT_DIR = None
        _reset_best_result_state(objective_target=objective_target)
        _apply_plot_defaults_if_needed()
        run_output_dir = _init_run_output_dir()
        print("=" * 108)
        print("Transformer V7 fixed-params run start")
        print("Output dir:", run_output_dir)
        print("Config json:", config_path)
        print("Fixed vector:", vec)
        print("=" * 108)
        result = objective(list(vec))
        if result is not None:
            print(f"[V7] Fixed-params run objective ({objective_target} R2) = {-float(result):.6f}")
        save_path = _save_best_result()
        if save_path:
            print("[V7] Saved best params:", save_path)
        summary_path = os.path.join(run_output_dir, "fixed_params_run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "config_source": config_path,
                    "fixed_split_json": fixed_split_path if os.path.isfile(fixed_split_path) else None,
                    "vector": list(vec),
                    "json_payload": payload,
                    "best_result": dict(BEST_RESULT),
                    "env": {k: os.environ.get(k) for k in env_updates.keys()},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("[V7] Saved fixed-params summary:", summary_path)
        return run_output_dir
    finally:
        _temporary_env_restore(backup)


def _paired_significance(base_vals: np.ndarray, alt_vals: np.ndarray) -> Dict[str, float]:
    out = {
        "p_ttest": float("nan"),
        "p_wilcoxon": float("nan"),
    }
    if base_vals.size < 2 or alt_vals.size < 2:
        return out
    mask = np.isfinite(base_vals) & np.isfinite(alt_vals)
    b = base_vals[mask]
    a = alt_vals[mask]
    if b.size < 2:
        return out
    if ttest_rel is not None:
        try:
            _, p = ttest_rel(b, a, nan_policy="omit")
            out["p_ttest"] = float(p)
        except Exception:
            pass
    if wilcoxon is not None:
        try:
            _, p = wilcoxon(b, a)
            out["p_wilcoxon"] = float(p)
        except Exception:
            pass
    return out


def run_ablation_significance() -> None:
    global GLOBAL_ITER, SPLIT_CACHE, FP_RANK_CACHE, RUN_OUTPUT_DIR
    GLOBAL_ITER = 0
    SPLIT_CACHE = {}
    FP_RANK_CACHE = {}
    if RUN_OUTPUT_DIR is None:
        _apply_plot_defaults_if_needed()
        _init_run_output_dir()

    params, params_path = _load_ablation_base_params()
    objective_target = _get_objective_target()
    seed_text = str(_env_value("TRANSFORMER_V3_ABLATION_SEEDS", None, "11,19,23,29,37"))
    seeds = _parse_seeds(seed_text)
    base_mode = _normalize_model_mode(params.get("model_mode", "dual"))
    print("=" * 108)
    print("Transformer V5 ablation + significance start")
    print("Base params:", params_path)
    print("Objective target:", objective_target)
    print("Seeds:", seeds)
    print("=" * 108)

    base_env = {
        "TRANSFORMER_V3_ENABLE_DUAL": "1",
        "TRANSFORMER_V3_GATE_EXTRA_FEATURES": "1",
        "TRANSFORMER_V3_ENABLE_CORRECTION_HEAD": "1",
        "TRANSFORMER_V3_ENABLE_FUSION": "1",
        "TRANSFORMER_V3_ENABLE_CHEM_ATTN_BIAS": "1",
        "TRANSFORMER_V3_SKIP_ARTIFACTS": "1",
        "TRANSFORMER_DETERMINISTIC": "1",
        "TRANSFORMER_V3_OBJECTIVE_TARGET": "val",
    }

    variants = [
        {"name": "baseline", "mode": base_mode, "env": {}},
        {"name": "no_adaptive_fusion", "mode": base_mode, "env": {"TRANSFORMER_V3_ENABLE_FUSION": "0"}},
        {
            "name": "no_physicochemical_prior",
            "mode": base_mode,
            "env": {},
            "params": {"lambda_physics": 0.0},
        },
    ]
    if base_mode in {"dual", "attn"}:
        variants.append(
            {
                "name": "no_chem_attn_bias",
                "mode": base_mode,
                "env": {"TRANSFORMER_V3_ENABLE_CHEM_ATTN_BIAS": "0"},
            }
        )
    if base_mode == "dual":
        variants.extend(
            [
                {"name": "no_dual_expert", "mode": "dual", "env": {"TRANSFORMER_V3_ENABLE_DUAL": "0"}},
                {
                    "name": "no_gate_extra_features",
                    "mode": "dual",
                    "env": {"TRANSFORMER_V3_GATE_EXTRA_FEATURES": "0"},
                },
                {
                    "name": "no_correction_head",
                    "mode": "dual",
                    "env": {"TRANSFORMER_V3_ENABLE_CORRECTION_HEAD": "0"},
                },
            ]
        )

    trial_rows = []
    for variant in variants:
        name = str(variant["name"])
        mode = _normalize_model_mode(variant.get("mode", base_mode))
        for seed in seeds:
            env_updates = dict(base_env)
            env_updates.update({k: str(v) for k, v in dict(variant.get("env", {})).items()})
            env_updates["TRANSFORMER_SEED"] = str(seed)
            backup = _temporary_env_set(env_updates)
            try:
                GLOBAL_ITER = 0
                _reset_best_result_state(objective_target=objective_target)
                variant_params = dict(params)
                variant_params.update(dict(variant.get("params", {})))
                x = _params_to_vector(variant_params, model_mode_override=mode)
                _ = objective(x)
                trial_rows.append(
                    {
                        "variant": name,
                        "seed": int(seed),
                        "model_mode": mode,
                        "objective_target": objective_target,
                        "objective_r2": _safe_float(BEST_RESULT.get("objective_r2")),
                        "val_r2": _safe_float(BEST_RESULT.get("val_r2")),
                        "val_rmse": _safe_float(BEST_RESULT.get("val_rmse")),
                        "test_r2": _safe_float(BEST_RESULT.get("test_r2")),
                        "test_rmse": _safe_float(BEST_RESULT.get("test_rmse")),
                    }
                )
                print(
                    f"[ablation] {name:<24s} seed={seed:<4d} "
                    f"val_r2={_safe_float(BEST_RESULT.get('val_r2')):7.4f} "
                    f"test_r2={_safe_float(BEST_RESULT.get('test_r2')):7.4f} "
                    f"test_rmse={_safe_float(BEST_RESULT.get('test_rmse')):8.4f}"
                )
            finally:
                _temporary_env_restore(backup)

    if not trial_rows:
        print("[ablation] no trial rows generated.")
        return

    names = sorted({str(r["variant"]) for r in trial_rows})
    summary_rows = []
    baseline = [r for r in trial_rows if str(r["variant"]) == "baseline"]
    base_by_seed = {int(r["seed"]): r for r in baseline}
    for name in names:
        rows = [r for r in trial_rows if str(r["variant"]) == name]
        val_r2_arr = np.asarray([float(r["val_r2"]) for r in rows], dtype=float)
        test_r2_arr = np.asarray([float(r["test_r2"]) for r in rows], dtype=float)
        test_rmse_arr = np.asarray([float(r["test_rmse"]) for r in rows], dtype=float)
        row_out = {
            "variant": name,
            "n_runs": int(len(rows)),
            "val_r2_mean": float(np.nanmean(val_r2_arr)),
            "val_r2_std": float(np.nanstd(val_r2_arr)),
            "test_r2_mean": float(np.nanmean(test_r2_arr)),
            "test_r2_std": float(np.nanstd(test_r2_arr)),
            "test_rmse_mean": float(np.nanmean(test_rmse_arr)),
            "test_rmse_std": float(np.nanstd(test_rmse_arr)),
            "delta_test_r2_vs_baseline": float("nan"),
            "delta_test_rmse_vs_baseline": float("nan"),
            "p_ttest_test_r2_vs_baseline": float("nan"),
            "p_wilcoxon_test_r2_vs_baseline": float("nan"),
            "p_ttest_test_rmse_vs_baseline": float("nan"),
            "p_wilcoxon_test_rmse_vs_baseline": float("nan"),
        }
        if name != "baseline":
            common = [r for r in rows if int(r["seed"]) in base_by_seed]
            if common:
                base_test_r2 = np.asarray([float(base_by_seed[int(r["seed"])]["test_r2"]) for r in common], dtype=float)
                base_test_rmse = np.asarray([float(base_by_seed[int(r["seed"])]["test_rmse"]) for r in common], dtype=float)
                alt_test_r2 = np.asarray([float(r["test_r2"]) for r in common], dtype=float)
                alt_test_rmse = np.asarray([float(r["test_rmse"]) for r in common], dtype=float)
                row_out["delta_test_r2_vs_baseline"] = float(np.nanmean(alt_test_r2 - base_test_r2))
                row_out["delta_test_rmse_vs_baseline"] = float(np.nanmean(alt_test_rmse - base_test_rmse))
                r2_stats = _paired_significance(base_test_r2, alt_test_r2)
                rmse_stats = _paired_significance(base_test_rmse, alt_test_rmse)
                row_out["p_ttest_test_r2_vs_baseline"] = float(r2_stats["p_ttest"])
                row_out["p_wilcoxon_test_r2_vs_baseline"] = float(r2_stats["p_wilcoxon"])
                row_out["p_ttest_test_rmse_vs_baseline"] = float(rmse_stats["p_ttest"])
                row_out["p_wilcoxon_test_rmse_vs_baseline"] = float(rmse_stats["p_wilcoxon"])
        summary_rows.append(row_out)

    out_dir = RUN_OUTPUT_DIR if RUN_OUTPUT_DIR else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    trial_csv = os.path.join(out_dir, "ablation_trials_transformer_v7.csv")
    summary_csv = os.path.join(out_dir, "ablation_summary_transformer_v7.csv")
    summary_json = os.path.join(out_dir, "ablation_summary_transformer_v7.json")

    with open(trial_csv, "w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "variant",
            "seed",
            "model_mode",
            "objective_target",
            "objective_r2",
            "val_r2",
            "val_rmse",
            "test_r2",
            "test_rmse",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in trial_rows:
            writer.writerow(r)

    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "variant",
            "n_runs",
            "val_r2_mean",
            "val_r2_std",
            "test_r2_mean",
            "test_r2_std",
            "test_rmse_mean",
            "test_rmse_std",
            "delta_test_r2_vs_baseline",
            "delta_test_rmse_vs_baseline",
            "p_ttest_test_r2_vs_baseline",
            "p_wilcoxon_test_r2_vs_baseline",
            "p_ttest_test_rmse_vs_baseline",
            "p_wilcoxon_test_rmse_vs_baseline",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "objective_target": objective_target,
                "params_path": params_path,
                "seeds": seeds,
                "base_params": params,
                "trials": trial_rows,
                "summary": summary_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 108)
    print("Ablation summary:")
    for r in summary_rows:
        print(
            f"{r['variant']:<24s} "
            f"test_r2={r['test_r2_mean']:.4f}±{r['test_r2_std']:.4f} "
            f"test_rmse={r['test_rmse_mean']:.4f}±{r['test_rmse_std']:.4f}"
        )
    print("Saved:", trial_csv)
    print("Saved:", summary_csv)
    print("Saved:", summary_json)


def main() -> None:
    """
    主程序入口。
    根据环境变量控制运行模式：
    1. 贝叶斯优化 (bo) - 默认模式
    2. 消融实验 (ablation) - 验证组件有效性
    3. 统计显著性测试 (significance)
    """
    mode = str(
        _env_value(
            "TRANSFORMER_V7_MODE",
            "TRANSFORMER_V6_MODE",
            _env_value("TRANSFORMER_V3_MODE", None, "bo"),
        )
    ).strip().lower()
    _setdefault_env("TRANSFORMER_V7_OBJECTIVE_TARGET", os.environ.get("TRANSFORMER_V6_OBJECTIVE_TARGET", "val"))
    run_ablation_only = bool(
        _env_bool_pair("TRANSFORMER_V7_RUN_ABLATION", "TRANSFORMER_V6_RUN_ABLATION", _env_bool_pair("TRANSFORMER_V3_RUN_ABLATION", None, False))
        or mode in {"ablation", "abl", "stat"}
    )
    run_ablation_after_bo = _env_bool_pair("TRANSFORMER_V7_RUN_ABLATION_AFTER_BO", "TRANSFORMER_V6_RUN_ABLATION_AFTER_BO", _env_bool_pair("TRANSFORMER_V3_RUN_ABLATION_AFTER_BO", None, False))
    run_final_retrain = _env_bool_pair("TRANSFORMER_V7_RUN_FINAL_RETRAIN", "TRANSFORMER_V6_RUN_FINAL_RETRAIN", False) or mode in {"retrain", "final_retrain"}
    
    if run_ablation_only:
        run_ablation_significance()
        return

    if mode in {"formal", "formal_dual_innov", "formal_dual_innov_best"}:
        run_formal_dual_innov_best()
        return

    if mode in {"fixed", "fixed_json", "fixed_params", "single"}:
        run_fixed_params_once()
        return

    if mode in {"retrain", "final_retrain"}:
        run_final_retrain_on_trainval()
        return
        
    # 执行贝叶斯优化
    run_bayesian_optimization()

    # Plan B: 在优化完成后，使用最佳超参数在 Train+Val 上重新训练 (Retrain on train+val after BO completes)
    if run_final_retrain:
        run_final_retrain_on_trainval()

    if run_ablation_after_bo:
        run_ablation_significance()


if __name__ == "__main__":
    main()
