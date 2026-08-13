# -*- coding: utf-8 -*-
"""
DECAT SEEN25-1626 seed-242 高通量适用域（AD）重算脚本

用锁定路线（SEEN25-1626 · seed=242）替换旧 V9 路线（1696 条 · seed-42 · top-1100）
重新计算高通量外部库（4,295 个唯一结构）的适用域分类。

方法口径（与 SI Text S8 一致，仅替换模型侧输入）：
  1. Morgan 指纹 radius=2, nBits=3147，由标准化 SMILES 重新生成（与锁定模型一致，
     不使用 CSV 中的 4096 位列）。
  2. 特征筛选：RandomForestRegressor(n_estimators=240, random_state=42) 仅在
     训练集（1240 条，固定划分 train_idx）上拟合，保留 top-914 指纹位。
  3. AD 距离 = 1 - 最近邻 Tanimoto 相似度（在 top-914 位空间内计算）。
  4. 阈值 = 训练集 LOO 最近邻距离分布的 95 / 99 百分位。
  5. pH 超出训练集覆盖范围的结构单独标记；无机/无碳结构沿用原筛选的排除标签。

运行：
  python run_ad_screening_seed242.py

输出（写入本脚本所在目录）：
  AD_seed242_unique_compound_results.csv   4,295 个唯一结构的新 AD 标签
  AD_seed242_summary.json                  阈值、计数、参数汇总
  AD_seed242_report_CN.md                  中文报告（新旧对比）
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR.parent
LOCKED_PKG = RELEASE_ROOT / "core" / "locked_package"
DATA_CSV = LOCKED_PKG / "data" / "反应logk指纹数据_25类_含环境无机物_Other_1626条.csv"
SPLIT_JSON = LOCKED_PKG / "data" / "固定划分_1626_同分子异pH非均分_75-12.5-12.5.json"
INPUT_DIR = RELEASE_ROOT / "screening" / "input"
UNIQUE_CSV = INPUT_DIR / "unique_compound_library.csv"
ROW_CSV = INPUT_DIR / "row_level_external_library.csv"

OUT_DIR = SCRIPT_DIR.parent / "screening" / "results"

FP_BITS = 3147
TOP_K = 914
RF_ESTIMATORS = 240
RF_SEED = 42  # 与模型源码 _fingerprint_ranking 一致（feature selection 内部固定 42）


def smiles_to_fp_bits(smiles: str):
    """标准化 SMILES -> Morgan(r=2, 3147) 的置位 bit 列表；失败返回 None。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
    return list(fp.GetOnBits())


def main():
    t0 = time.time()
    print("[1/6] 读取锁定数据与固定划分 ...")
    df = pd.read_csv(DATA_CSV)
    with open(SPLIT_JSON, encoding="utf-8") as f:
        split = json.load(f)
    train_idx = np.array(split["train_idx"], dtype=int)
    assert len(df) == 1626 and len(train_idx) == 1240

    train_df = df.iloc[train_idx].reset_index(drop=True)
    y_train = train_df["logk"].to_numpy(dtype=float)
    train_pH = pd.to_numeric(train_df["pH"], errors="coerce")
    pH_min, pH_max = float(train_pH.min()), float(train_pH.max())
    print(f"    训练集 n=1240, pH 范围 [{pH_min}, {pH_max}]")

    print("[2/6] 由 SMILES 重新生成训练集 Morgan 指纹 (r=2, 3147 bits) ...")
    train_bits = []
    for i, smi in enumerate(train_df["SMILES"]):
        bits = smiles_to_fp_bits(str(smi))
        if bits is None:
            raise ValueError(f"训练集 SMILES 无法解析: 行 {i} {smi}")
        train_bits.append(bits)

    X_train = np.zeros((len(train_bits), FP_BITS), dtype=np.uint8)
    for i, bits in enumerate(train_bits):
        X_train[i, bits] = 1

    print(f"[3/6] 训练集内 RF 排序并选取 top-{TOP_K} 指纹位 ...")
    rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS, max_depth=None,
                               random_state=RF_SEED, n_jobs=-1)
    rf.fit(X_train.astype(np.float32), y_train)
    rank = np.argsort(rf.feature_importances_)[::-1]
    sel_idx = np.sort(rank[:TOP_K]).astype(int)

    train_sel = X_train[:, sel_idx].astype(bool)  # (1240, 914)

    print("[4/6] 训练集 LOO 最近邻距离分布 -> q95/q99 阈值 ...")
    # 分块计算 1240 x 1240 Tanimoto（选中位空间），避免一次性大矩阵
    n_train = train_sel.shape[0]
    train_counts = train_sel.sum(axis=1)
    loo_nn_sim = np.full(n_train, -1.0)
    chunk = 128
    for s in range(0, n_train, chunk):
        e = min(s + chunk, n_train)
        blk = train_sel[s:e]  # (b, 914)
        inter = np.logical_and(blk[:, None, :], train_sel[None, :, :]).sum(axis=2)
        union = blk.sum(axis=1)[:, None] + train_counts[None, :] - inter
        sim = inter / np.maximum(union, 1)
        for r in range(e - s):
            sim[r, s + r] = -1.0  # 排除自身
        loo_nn_sim[s:e] = sim.max(axis=1)
        del inter, union, sim
    loo_nn_dist = 1.0 - loo_nn_sim
    q95 = float(np.percentile(loo_nn_dist, 95))
    q99 = float(np.percentile(loo_nn_dist, 99))
    print(f"    strict(q95)={q95:.4f}  borderline(q99)={q99:.4f}")

    print("[5/6] 重算 4,295 个唯一外部结构的 AD 标签 ...")
    uniq = pd.read_csv(UNIQUE_CSV)
    assert len(uniq) == 4295, f"唯一结构数异常: {len(uniq)}"

    new_sim = np.full(len(uniq), np.nan)
    new_dist = np.full(len(uniq), np.nan)
    for i, smi in enumerate(uniq["canonical_SMILES"]):
        bits = smiles_to_fp_bits(str(smi))
        if bits is None:
            continue
        q = np.zeros(FP_BITS, dtype=bool)
        q[bits] = True
        q_sel = q[sel_idx]
        inter = np.logical_and(train_sel, q_sel).sum(axis=1)
        union = train_sel.sum(axis=1) + q_sel.sum() - inter
        sim = (inter / np.maximum(union, 1)).max()
        new_sim[i] = sim
        new_dist[i] = 1.0 - sim
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/4295 ...")

    # 最近邻训练化合物信息
    uniq["seed242_nearest_train_similarity"] = new_sim
    uniq["seed242_nearest_train_distance"] = new_dist

    # 新标签：先沿用无机/无碳排除，再按距离分类，再 pH 标记
    labels = []
    for i in range(len(uniq)):
        scope_ok = bool(uniq.loc[i, "scope_ok_for_DECAT_organic_model"])
        d = new_dist[i]
        if not scope_ok:
            labels.append("Out of DECAT scope - inorganic/no carbon")
            continue
        if np.isnan(d):
            labels.append("Not evaluated - invalid SMILES")
            continue
        ph_lo, ph_hi = uniq.loc[i, "pH_min"], uniq.loc[i, "pH_max"]
        ph_out = False
        if pd.notna(ph_lo) and pd.notna(ph_hi):
            if (float(ph_hi) < pH_min) or (float(ph_lo) > pH_max):
                ph_out = True
        if ph_out:
            labels.append("Out of AD - pH outside train range")
        elif d <= q95:
            labels.append("In AD")
        elif d <= q99:
            labels.append("Borderline")
        else:
            labels.append("Out of AD")
    uniq["seed242_AD_label"] = labels

    unique_counts = uniq["seed242_AD_label"].value_counts().to_dict()
    in_ad = int(unique_counts.get("In AD", 0))
    print(f"    In AD: {in_ad}/4295 = {in_ad / 4295 * 100:.1f}%")

    print("[6/6] 重算 11,654 行级标签并汇总 ...")
    rows = pd.read_csv(ROW_CSV)
    label_map = dict(zip(uniq["canonical_SMILES"], uniq["seed242_AD_label"]))
    dist_map = dict(zip(uniq["canonical_SMILES"], uniq["seed242_nearest_train_distance"]))
    rows["seed242_nearest_train_distance"] = rows["canonical_SMILES"].map(dist_map)
    rows["seed242_AD_label"] = rows["canonical_SMILES"].map(label_map)
    # 行级 pH 覆盖：若行本身有 pH 且超出训练范围，改标 pH 出界（结构标签非无机排除时）
    scope_ok_map = dict(zip(uniq["canonical_SMILES"], uniq["scope_ok_for_DECAT_organic_model"]))
    rows["_scope_ok"] = rows["canonical_SMILES"].map(scope_ok_map)
    row_pH = pd.to_numeric(rows["source_pH"], errors="coerce")
    ph_out_rows = row_pH.notna() & ((row_pH < pH_min) | (row_pH > pH_max)) & rows["_scope_ok"].astype(bool)
    rows.loc[ph_out_rows, "seed242_AD_label"] = "Out of AD - pH outside train range"
    rows.drop(columns=["_scope_ok"], inplace=True)
    row_counts = rows["seed242_AD_label"].value_counts().to_dict()

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    out_unique = os.path.join(OUT_DIR, "AD_seed242_unique_compound_results.csv")
    out_rows = os.path.join(OUT_DIR, "AD_seed242_row_level_results.csv")
    out_json = os.path.join(OUT_DIR, "AD_seed242_summary.json")
    uniq.to_csv(out_unique, index=False, encoding="utf-8-sig")
    rows.to_csv(out_rows, index=False, encoding="utf-8-sig")

    old_unique_counts = {"In AD": 3468, "Borderline": 676, "Out of AD": 61,
                         "Out of DECAT scope - inorganic/no carbon": 82,
                         "Out of AD - pH outside train range": 8}
    summary = {
        "route": "SEEN25-1626 seed-242 (locked) vs old V9 (1696, seed-42, top-1100)",
        "train_n": 1240,
        "fp_bits": FP_BITS,
        "topk_features": TOP_K,
        "rf": {"n_estimators": RF_ESTIMATORS, "random_state": RF_SEED},
        "train_pH_range": [pH_min, pH_max],
        "strict_distance_q95": q95,
        "borderline_distance_q99": q99,
        "unique_total": 4295,
        "unique_counts_seed242": {k: int(v) for k, v in unique_counts.items()},
        "unique_counts_old_v9": old_unique_counts,
        "in_ad_percent_seed242": round(in_ad / 4295 * 100, 2),
        "row_total": int(len(rows)),
        "row_counts_seed242": {k: int(v) for k, v in row_counts.items()},
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 中文报告
    def pct(n, total):
        return f"{n / total * 100:.1f}%"

    report = ["# DECAT SEEN25-1626 seed-242 高通量适用域（AD）重算报告", ""]
    report.append(f"- 路线：SEEN25-1626 · seed=242（锁定）｜训练集 1,240 条｜Morgan r=2 / 3,147 bits / top-914")
    report.append(f"- 阈值：strict distance ≤ {q95:.4f}（LOO q95）；borderline ≤ {q99:.4f}（LOO q99）")
    report.append(f"- 训练集 pH 覆盖：[{pH_min}, {pH_max}]")
    report.append("")
    report.append("## 唯一结构（4,295）新旧对比")
    report.append("")
    report.append("| 标签 | 旧 V9（1696/top-1100） | 新（1626/top-914） |")
    report.append("|---|---:|---:|")
    for k in ["In AD", "Borderline", "Out of AD",
              "Out of DECAT scope - inorganic/no carbon",
              "Out of AD - pH outside train range"]:
        old_n = old_unique_counts.get(k, 0)
        new_n = int(unique_counts.get(k, 0))
        report.append(f"| {k} | {old_n} ({pct(old_n, 4295)}) | {new_n} ({pct(new_n, 4295)}) |")
    report.append("")
    report.append("## 行级（11,654）新标签分布")
    report.append("")
    report.append("| 标签 | count | percent |")
    report.append("|---|---:|---:|")
    for k, v in sorted(row_counts.items(), key=lambda kv: -kv[1]):
        report.append(f"| {k} | {int(v)} | {pct(int(v), len(rows))} |")
    report.append("")
    with open(os.path.join(OUT_DIR, "AD_seed242_report_CN.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("完成！输出目录:", OUT_DIR)
    print(json.dumps(summary["unique_counts_seed242"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
