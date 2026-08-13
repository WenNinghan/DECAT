# DECAT 1626 · 25 类锁定可复现实验包

**包名：** `DECAT_SEEN25_1626_locked_package_20260723`  
**锁定结果：** Test R² = **0.828640**，Val R² = **0.830382**，Train R² = **0.978636**  
**选模：** 仅验证集（`validation-only`）；测试集仅最终揭盲报数  
**最终输出头：** `nn_plus_res_rf`（Transformer 中心双专家主预测 + RF 残差校正）

---

## 1. 目录结构

```text
DECAT_SEEN25_1626_locked_package_20260723/
├── README.md
├── requirements.txt
├── environment.yml                 # 可选 conda 环境
├── run_decat_v21_blind_validation_fixed.py
├── configs/
│   ├── LOCKED_SEEN25_1626_NONUNIFORM_SOTA.json   # 锁定配置 + 期望指标 + SHA256
│   └── V14_1637_validation_topk5.json            # 运行时 base env
├── data/
│   ├── 反应logk指纹数据_25类_含环境无机物_Other_1626条.csv
│   └── 固定划分_1626_同分子异pH非均分_75-12.5-12.5.json
├── src/decat/
│   └── transformer_v9_transformer_centered.py    # 模型源码
├── scripts/
│   ├── run_locked_seen25_1626.ps1                # 一键复现
│   └── _repro_launch.py
├── artifacts/
│   ├── run/                    # 原锁定运行 summary / checkpoint / 图
│   └── predictions/            # validation_predictions.npz, fusion_components.npz
├── metrics/
│   ├── PERFORMANCE_REPORT.md   # 详细性能报告
│   ├── detailed_performance.json
│   ├── performance_overview.csv
│   ├── component_performance.csv
│   └── predictions_{train,validation,test}.csv
└── docs/
    └── PERFORMANCE_REPORT.md
```

---

## 2. 环境

推荐：

```powershell
python
```

依赖见 `requirements.txt`（至少：`torch numpy pandas scikit-learn scipy rdkit matplotlib`）。

检查文件完整性（不训练）：

```powershell
cd <本包路径>
pwsh -File .\scripts\run_locked_seen25_1626.ps1 -VerifyOnly
```

---

## 3. 完整复现训练（约 GPU 数分钟–十几分钟）

```powershell
cd <本包路径>
pwsh -File .\scripts\run_locked_seen25_1626.ps1
```

自定义 Python / 输出目录：

```powershell
pwsh -File .\scripts\run_locked_seen25_1626.ps1 `
  -Python python `
  -OutputRoot ".\outputs\my_repro"
```

脚本会：

1. 校验数据 / 划分 / 源码 / runner 的 SHA-256  
2. 按锁定超参 + `DECAT_FINAL_COMPONENT=nn_plus_res_rf` 训练  
3. `DECAT_UNMASK_TEST=1` 报 Test（配置已在验证集上选定，不再调参）  
4. 将 Val/Test R²、RMSE 与参考值比对（容差见配置 `expected_result`）

---

## 4. 不重训：直接查看本次锁定结果

| 文件 | 内容 |
|------|------|
| `metrics/PERFORMANCE_REPORT.md` | 中文详细性能 |
| `metrics/detailed_performance.json` | 机器可读全量指标 |
| `metrics/performance_overview.csv` | Train/Val/Test 总表 |
| `metrics/component_performance.csv` | 各融合候选 R² |
| `metrics/predictions_*.csv` | 逐条真值 / 预测 / 残差 |
| `artifacts/run/fixed_params_run_summary.json` | 原始运行摘要 |
| `artifacts/run/transformer_v7_best.pth` | 权重（参考） |
| `artifacts/predictions/*.npz` | 预测与组件数组 |

---

## 5. 协议说明（写论文用）

- **样本**：一条实验记录（SMILES + pH + 25 类标签 + logk）  
- **划分**：原子组 = RDKit canonical SMILES + 精确 pH + `category27_label`，**原子组不跨集**  
- **允许**：同一分子不同 pH 出现在不同集合（记录级 holdout）  
- **禁止**：测试标签参与选参 / 早停 / 融合选择  
- **预测**：端到端模型前向，**不是**读训练集插值  

---

## 6. 参考指标（本包锁定）

| Split | n | R² | RMSE | MAE |
|-------|---|-----|------|-----|
| Train | 1240 | 0.978636 | 0.410047 | 0.204703 |
| Validation | 186 | 0.830382 | 1.107434 | 0.703577 |
| **Test** | **200** | **0.828640** | **1.162404** | **0.737111** |

Seed = 242；best_epoch = 33；fp_bits = 3147；topk = 914。
