# DECAT SEEN25-1626 seed-242 高通量适用域（AD）重算报告

- 路线：SEEN25-1626 · seed=242（锁定）｜训练集 1,240 条｜Morgan r=2 / 3,147 bits / top-914
- 阈值：strict distance ≤ 0.6429（LOO q95）；borderline ≤ 0.8000（LOO q99）
- 训练集 pH 覆盖：[1.0, 12.0]

## 唯一结构（4,295）新旧对比

| 标签 | 旧 V9（1696/top-1100） | 新（1626/top-914） |
|---|---:|---:|
| In AD | 3468 (80.7%) | 3423 (79.7%) |
| Borderline | 676 (15.7%) | 699 (16.3%) |
| Out of AD | 61 (1.4%) | 51 (1.2%) |
| Out of DECAT scope - inorganic/no carbon | 82 (1.9%) | 82 (1.9%) |
| Out of AD - pH outside train range | 8 (0.2%) | 40 (0.9%) |

## 行级（11,654）新标签分布

| 标签 | count | percent |
|---|---:|---:|
| In AD | 9764 | 83.8% |
| Borderline | 1017 | 8.7% |
| Out of AD - pH outside train range | 309 | 2.7% |
| Out of DECAT scope - inorganic/no carbon | 235 | 2.0% |
| Out of AD | 64 | 0.5% |
