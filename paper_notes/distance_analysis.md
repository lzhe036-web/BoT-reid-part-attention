# Baseline-Control 与 C2-L03 距离分布分析

更新时间：2026-07-19。

## 1. 证据状态与数据位置

Market1501 和 DukeMTMC-reID 的真实距离分析均已完成，不再是“待实验”。原始 CSV、JSON、日志、checkpoint 和图片保存在项目外的证据目录，本文件只整理结果，没有修改或移动原始产物。

证据根目录：

`D:\thesis_reid\thesis_evidence\distance_analysis\2026-07-18`

完整结果目录：

- Market1501：`analysis_results\market_epoch120_control`
- DukeMTMC-reID：`analysis_results\duke_epoch120_control`

用户提供的源压缩包：

| 压缩包 | SHA256 |
|---|---|
| `D:\Downloads\distance_evidence_server_78514386_20260718.tar.gz` | `0add8d833c60c425d48bfbe8d3f1cf829943525a46f9f8087adea78339f45f5b` |
| `D:\Downloads\market_c2_l03_evidence_server_ltmxmapcaz_20260718.tar.gz` | `61ff271800b7157351fffa5f9a54403caf5a87b65f1407292125d12711885624` |

两份压缩包与证据根目录 `raw_archives` 中已有副本的 SHA256 完全一致，且内容已经解压和分析，因此未向仓库再次复制约 1.25 GB 的重复文件。

## 2. 分析协议

两组模型在每个数据集内使用完全相同的样本顺序和 pair 索引。

| 项目 | Market1501 | DukeMTMC-reID |
|---|---:|---:|
| 样本范围 | query + gallery | query + gallery |
| 样本数 | 19,281（3,368 + 15,913） | 19,889（2,228 + 17,661） |
| 特征 | BNNeck-after，2048 维 | 相同 |
| normalization | 显式 L2 | 相同 |
| 距离 | squared Euclidean | 相同 |
| re-ranking | 关闭 | 关闭 |
| camera-mean debias | 关闭 | 关闭 |
| same-ID pair | 全部无序 pair | 全部无序 pair |
| different-ID pair | 均匀无放回抽样 200,000 对 | 相同 |
| 抽样 seed | 42 | 42 |
| distance chunk | 4,096 对 | 4,096 对 |

三类 pair 定义为：

\[
\begin{aligned}
\text{same-ID same-camera}:&\quad y_i=y_j,\ c_i=c_j,\\
\text{same-ID different-camera}:&\quad y_i=y_j,\ c_i\ne c_j,\\
\text{different-ID}:&\quad y_i\ne y_j.
\end{aligned}
\]

质量检查均已通过：

- 排除 self-pair；
- 只保留 \(i<j\)，不重复统计 `(i,j)` 与 `(j,i)`；
- 三类 pair 互斥且分类与 pid/camid 逐行一致；
- Baseline 与 C2-L03 的 `samples.csv` 顺序相同；
- Baseline 与 C2-L03 复用同一份 `pair_indices.csv`；
- `pair_indices.csv` 与 `pair_distances.csv` 逐行对齐；
- 四个 checkpoint 均无 missing/skipped 参数；
- 从 CSV 独立重算的统计与 JSON 一致，最大绝对误差约为 `4.3e-10`。

需要特别区分两个特征空间：

- C2 训练辅助项使用 BNNeck-before 的全局与 Part Attention 融合特征，默认不显式 L2 normalize；
- 本分析使用实际非重排检索协议中的 BNNeck-after、L2-normalized 特征。

因此，本节检验的是“C2 训练后检索嵌入空间的距离结构”，不能表述为 C2 直接优化了 post-BN、L2-normalized 的 squared Euclidean 距离。

## 3. 样本、pair 与 checkpoint 哈希

| 数据集 | sample order SHA256 | pair index SHA256 |
|---|---|---|
| Market1501 | `ede26d3a28aece193741f618d26bd5b3ceecacce8ed359589322030a01d14461` | `193dd9ccfd552c6fcf7c402e2d31ad469ee51633cafc30142e2f3a835b522084` |
| DukeMTMC-reID | `be498c3a413de371c27b610728b7a4c2465307e6c6e74ee5184b4e92df2a7f2e` | `1e0ffcd68de323d8c0f56d69737a673d0cdda11db1c0b4c423ccae59e96a234b` |

| 数据集/模型 | checkpoint | epoch | SHA256 |
|---|---|---:|---|
| Market Baseline-Control | `market_baseline\resnet50_checkpoint_22320.pt` | 120 | `171aea42cb8df1464461887c78352205d61319341cd4ea93afc9dc5e9fc34edf` |
| Market C2-L03 | `market_c2_l03\resnet50_checkpoint_22320.pt` | 120 | `2008541aa5738c8bc3a440504d0bbb055b646f5d8e97a25e113f2d2e463d497b` |
| Duke Baseline-Control | `duke_baseline\resnet50_checkpoint_28920.pt` | 120 | `ae0713f27dbc80684c4840d0d0138748dcb2f03228d9fe7de4cc9c6bf36f7a55` |
| Duke C2-L03 | `duke_c2_l03\resnet50_checkpoint_28920.pt` | 120 | `d5ded542fd82bb96068106a7769cb9290c5939bde4862fb60e8ffbb89d5f2e0f` |

Duke 论文主表中的 Baseline 报告点为 epoch 80，但本节为避免 checkpoint epoch 差异，使用 Baseline 与 C2-L03 均为 epoch 120 的对照。对应检索指标为 Baseline `86.4/76.3`，C2-L03 `88.4/78.7`（Rank-1/mAP）。不能把这里的 Baseline checkpoint 称为“最佳指标 checkpoint”。

## 4. Market1501 结果

### 4.1 完整描述统计

下表中 std 为总体标准差。

| pair 类型 | 模型 | count | mean | std | median | q25 | q75 | q05 | q95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| same-ID same-camera | Baseline | 717,000 | 1.698396 | 0.348022 | 1.783479 | 1.603586 | 1.914227 | 0.896448 | 2.064273 |
| same-ID same-camera | C2-L03 | 717,000 | 1.701136 | 0.365142 | 1.794443 | 1.613689 | 1.923648 | 0.810956 | 2.071109 |
| same-ID different-camera | Baseline | 3,408,766 | 1.773591 | 0.270782 | 1.825205 | 1.675943 | 1.945302 | 1.233907 | 2.090542 |
| same-ID different-camera | C2-L03 | 3,408,766 | 1.772773 | 0.291417 | 1.832293 | 1.682981 | 1.951205 | 1.181387 | 2.095465 |
| different-ID | Baseline | 200,000 | 1.988320 | 0.141430 | 2.001366 | 1.911494 | 2.082514 | 1.741437 | 2.191031 |
| different-ID | C2-L03 | 200,000 | 1.989686 | 0.151847 | 2.008878 | 1.915006 | 2.088958 | 1.718618 | 2.195377 |

### 4.2 C2-L03 相对 Baseline 的变化

| pair 类型 | Δmean | Δstd | Δmedian | Δq25 | Δq75 | Δq05 | Δq95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| same-ID same-camera | +0.002740（+0.161%） | +0.017119（+4.919%） | +0.010963（+0.615%） | +0.010103 | +0.009421 | −0.085492 | +0.006836 |
| same-ID different-camera | **−0.000818（−0.046%）** | **+0.020636（+7.621%）** | **+0.007088（+0.388%）** | +0.007038 | +0.005903 | −0.052520 | +0.004923 |
| different-ID | +0.001366（+0.069%） | +0.010416（+7.365%） | +0.007511（+0.375%） | +0.003512 | +0.006444 | −0.022820 | +0.004346 |

Market 的 `same-ID different-camera` 均值只下降 `0.000818`，相对变化为 `−0.046%`；但 median、q25、q75 和 q95 均略微上升，std 增加 `7.621%`。q05 向左移动说明低距离尾部有所变化，但不能代表主体分布整体收缩。

### 4.3 类间间隔

这里的间隔定义为：

\[
\text{gap}=
d(\text{different-ID})-
d(\text{same-ID different-camera}).
\]

| 统计量 | Baseline | C2-L03 | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| mean gap | 0.214728 | 0.216913 | +0.002185 | +1.017% |
| median gap | 0.176162 | 0.176585 | +0.000424 | +0.241% |

Market 的类间间隔略有扩大，different-ID 均值也未出现不利左移，但变化幅度很小。

### 4.4 Market 结论

不能写“C2-L03 在 Market1501 上整体拉近了跨摄像头同身份距离”。与数据一致的表述是：

> 在当前 Market1501 epoch-120 checkpoint 和统一检索嵌入协议下，C2-L03 的跨摄像头同身份距离均值仅下降 0.046%，而中位数及多数主要分位数略有上升，离散度增加。类间间隔略有扩大，但幅度很小。因此，当前全 pair 距离分布不足以支持“跨摄像头同身份检索嵌入被整体明显拉近”的机制解释。

Market 的 Rank-1/mAP 从 `94.4/85.5` 提高到 `95.0/87.8`，说明检索指标与“全体跨摄像头 pair 整体左移”并不等价。当前证据只能报告这种不一致，不能选择性使用极小的均值下降来宣称机制已得到验证。

## 5. DukeMTMC-reID 结果

### 5.1 完整描述统计

| pair 类型 | 模型 | count | mean | std | median | q25 | q75 | q05 | q95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| same-ID same-camera | Baseline | 325,738 | 0.925329 | 0.385137 | 0.923258 | 0.626278 | 1.204942 | 0.314469 | 1.572626 |
| same-ID same-camera | C2-L03 | 325,738 | 0.883940 | 0.382157 | 0.876288 | 0.587737 | 1.147529 | 0.291622 | 1.546491 |
| same-ID different-camera | Baseline | 179,162 | 1.167991 | 0.375483 | 1.132603 | 0.880371 | 1.435743 | 0.607502 | 1.833666 |
| same-ID different-camera | C2-L03 | 179,162 | 1.081971 | 0.384484 | 1.039084 | 0.782012 | 1.357593 | 0.518360 | 1.768922 |
| different-ID | Baseline | 200,000 | 1.975909 | 0.139658 | 1.989833 | 1.899474 | 2.069762 | 1.730713 | 2.174437 |
| different-ID | C2-L03 | 200,000 | 1.965717 | 0.147499 | 1.984430 | 1.889030 | 2.064860 | 1.701981 | 2.167414 |

### 5.2 C2-L03 相对 Baseline 的变化

| pair 类型 | Δmean | Δstd | Δmedian | Δq25 | Δq75 | Δq05 | Δq95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| same-ID same-camera | −0.041390（−4.473%） | −0.002981（−0.774%） | −0.046970（−5.087%） | −0.038542 | −0.057412 | −0.022847 | −0.026135 |
| same-ID different-camera | **−0.086020（−7.365%）** | +0.009002（+2.397%） | **−0.093520（−8.257%）** | −0.098360 | −0.078150 | −0.089142 | −0.064744 |
| different-ID | −0.010192（−0.516%） | +0.007841（+5.615%） | −0.005403（−0.272%） | −0.010444 | −0.004902 | −0.028732 | −0.007023 |

Duke 的 `same-ID different-camera` mean、median、q05、q25、q75 和 q95 全部下降。mean 下降 `7.365%`，median 下降 `8.257%`，不是由少数异常值单独造成的。

different-ID mean 也下降 `0.516%`，但远小于跨摄像头同身份距离的下降；同时其 std 增加 `5.615%`、q05 下降 `1.660%`，表明低距离尾部有所扩展，不能写成“异身份结构的所有方面均改善”。

### 5.3 类间间隔

| 统计量 | Baseline | C2-L03 | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| mean gap | 0.807918 | 0.883746 | +0.075828 | +9.386% |
| median gap | 0.857230 | 0.945346 | +0.088117 | +10.279% |

跨摄像头同身份距离的下降幅度大于 different-ID 距离的下降，mean 和 median separation gap 均扩大。

### 5.4 Duke 结论

可以谨慎写：

> 在当前 DukeMTMC-reID epoch-120 checkpoint 和统一检索嵌入协议下，C2-L03 的跨摄像头同身份距离在均值、中位数和全部主要分位数上均下降，其中均值下降 7.365%，中位数下降 8.257%。different-ID 均值也下降 0.516%，但幅度较小，mean/median separation gap 分别扩大 9.386% 和 10.279%。因此，当前 Duke 证据支持 C2-L03 拉近跨摄像头同身份检索嵌入，并未在总体间隔指标上观察到类间分离被破坏。

该结论只适用于当前 checkpoint 和分析协议。由于 pair 并非相互独立、训练 seed 未记录且没有多 seed 重复实验，不能将描述统计写成统计显著性或普遍规律。

## 6. 跨数据集综合判断

| 问题 | Market1501 | DukeMTMC-reID |
|---|---|---|
| 跨摄像头同身份 mean 是否下降 | 是，−0.046%，幅度极小 | 是，−7.365% |
| median 与主体分位数是否同向下降 | 否 | 是 |
| different-ID mean 是否下降 | 否，+0.069% | 是，−0.516% |
| mean separation gap | +1.017% | +9.386% |
| 是否支持“整体拉近” | **不支持** | **在当前协议下支持** |

综合表述：

> 距离机制证据具有数据集差异。DukeMTMC-reID 的主体分布与间隔变化支持 C2 对跨摄像头类内结构的改善；Market1501 虽取得更高 Rank-1/mAP，但全 pair 检索距离并未呈现一致的整体收缩。论文应同时报告两者，并将“C2 的作用必然表现为所有跨摄像头正 pair 的全局左移”保留为未被普遍验证的假设。

## 7. 机器可读结果与图表

每个完整结果目录包含：

- `metadata.json`
- `samples.csv`
- `pair_indices.csv`
- `pair_distances.csv`
- `distance_summary.csv`
- `distance_summary.json`
- `distance_summary.md`
- `separation_gap_summary.csv`
- `separation_gap_summary.json`
- `distance_histogram.png`
- `distance_boxplot.png`

Market 图表：

- `D:\thesis_reid\thesis_evidence\distance_analysis\2026-07-18\analysis_results\market_epoch120_control\distance_histogram.png`
- `D:\thesis_reid\thesis_evidence\distance_analysis\2026-07-18\analysis_results\market_epoch120_control\distance_boxplot.png`

Duke 图表：

- `D:\thesis_reid\thesis_evidence\distance_analysis\2026-07-18\analysis_results\duke_epoch120_control\distance_histogram.png`
- `D:\thesis_reid\thesis_evidence\distance_analysis\2026-07-18\analysis_results\duke_epoch120_control\distance_boxplot.png`

图中已注明 dataset、两组 checkpoint、BNNeck-after、L2 normalization、squared Euclidean、same-ID 全量 pair，以及 different-ID 均匀无放回抽样 `n=200000, seed=42`。

## 8. 证据边界与后续工作

已经完成：

- 两数据集样本顺序、pair 索引和 checkpoint 哈希核验；
- self 排除、\(i<j\)、pair 去重和三类互斥核验；
- 全量统计复算；
- 原始日志、checkpoint、CSV/JSON 和图表归档。

仍缺少：

1. 固定训练 seed 的多次独立重复实验；
2. 对不同 checkpoint epoch 的敏感性检查；
3. 与训练空间直接对应的 BNNeck-before、未显式 L2 特征分析；
4. hard-positive、近邻或检索排序局部 pair 的补充分析，用于解释 Market 上“检索指标改善但全 pair 分布不整体左移”的现象；
5. 配对样本具有相关性时更合适的不确定性估计，不能把数百万 pair 直接视为独立重复。
