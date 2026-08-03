# Baseline-Control 与 C2-L03 距离分布分析

更新时间：2026-07-22。

## 1. 证据状态与数据位置

Market1501 和 DukeMTMC-reID 的真实距离分析均已完成，不再是“待实验”。原始 CSV、JSON、日志、checkpoint 和图片保存在项目外的证据目录，本文件只整理结果，没有修改或移动原始产物。

证据根目录：

`D:\thesis_reid\thesis_evidence\distance_analysis\2026-07-18`

完整结果目录：

- Market1501 正式结果：`analysis_results\market_epoch120_person_only_v2`
- Market1501 旧结果：`analysis_results\market_epoch120_control`（含 pid=0，已标记为 superseded，仅保留审计）
- DukeMTMC-reID：`analysis_results\duke_epoch120_control`

论文统一表位于 `paper_notes/c2_l03_final_evidence/UNIFIED_TABLES.md`，机器可读
正式距离表为 `distance_distribution_formal.csv`。Market person-only v2 的
证据 ID 为 `EV-ANALYSIS-MKT-DIST-PIDGT0`，分析执行 E3、论文结论 E2；Duke
证据 ID 为 `EV-ANALYSIS-DUK-DIST-E120`，分析与结论均为 E2。

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
| PID 过滤 | `pid>0`，特征提取和 pair 生成前执行 | 无额外过滤；样本中无非正 pid |
| 样本数 | 16,483（3,368 + 13,115；排除 2,798 张 pid=0） | 19,889（2,228 + 17,661） |
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
- Market metadata 还确认全部分析 pid 为正，过滤在特征提取和 pair 生成前执行；
- 从 CSV 独立重算的统计与 JSON 一致。

需要特别区分两个特征空间：

- C2 训练辅助项使用 BNNeck-before 的全局与 Part Attention 融合特征，默认不显式 L2 normalize；
- 本分析使用实际非重排检索协议中的 BNNeck-after、L2-normalized 特征。

因此，本节检验的是“C2 训练后检索嵌入空间的距离结构”，不能表述为 C2 直接优化了 post-BN、L2-normalized 的 squared Euclidean 距离。

Market 的第一版输出因 loader 保留 `pid=0` background 而失效；正式修正版使用
`--pid-filter positive-only`，在特征提取和 pair 生成前排除全部 2,798 张
`pid=0` gallery。旧目录保留为 E0/superseded 审计记录，不再进入论文定量结论。

## 3. 样本、pair 与 checkpoint 哈希

| 数据集 | sample order SHA256 | pair index SHA256 |
|---|---|---|
| Market1501（person-only v2） | `c923b061a62243a08c7adc66a040302bb9662cbdfe92b0d350dfe5f5baa47fad` | `bd4093c0a557b55f43e6d2342c8ce9bb1cb2a85a0f5cb2aeade70dccedba464e` |
| DukeMTMC-reID | `be498c3a413de371c27b610728b7a4c2465307e6c6e74ee5184b4e92df2a7f2e` | `1e0ffcd68de323d8c0f56d69737a673d0cdda11db1c0b4c423ccae59e96a234b` |

| 数据集/模型 | checkpoint | epoch | SHA256 |
|---|---|---:|---|
| Market Baseline-Control | `market_baseline\resnet50_checkpoint_22320.pt` | 120 | `171aea42cb8df1464461887c78352205d61319341cd4ea93afc9dc5e9fc34edf` |
| Market C2-L03 | `market_c2_l03\resnet50_checkpoint_22320.pt` | 120 | `2008541aa5738c8bc3a440504d0bbb055b646f5d8e97a25e113f2d2e463d497b` |
| Duke Baseline-Control | `duke_baseline\resnet50_checkpoint_28920.pt` | 120 | `ae0713f27dbc80684c4840d0d0138748dcb2f03228d9fe7de4cc9c6bf36f7a55` |
| Duke C2-L03 | `duke_c2_l03\resnet50_checkpoint_28920.pt` | 120 | `d5ded542fd82bb96068106a7769cb9290c5939bde4862fb60e8ffbb89d5f2e0f` |

Duke 论文主表中的 Baseline 报告点为 epoch 80，但本节为避免 checkpoint epoch 差异，使用 Baseline 与 C2-L03 均为 epoch 120 的对照。对应检索指标为 Baseline `86.4/76.3`，C2-L03 `88.4/78.7`（Rank-1/mAP）。不能把这里的 Baseline checkpoint 称为“最佳指标 checkpoint”。

## 4. Market1501 person-only v2 正式结果

### 4.1 样本过滤与完整描述统计

Market `bounding_box_test` 共 19,732 张图像；loader 先排除 3,819 张
`pid=-1` junk，再由分析脚本在特征提取前排除 2,798 张 `pid=0` background。
最终使用 3,368 张 query 与 13,115 张 gallery，共 16,483 张 `pid>0` 样本。
下表中 std 为总体标准差。

| pair 类型 | 模型 | count | mean | std | median | q25 | q75 | q05 | q95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| same-ID same-camera | Baseline | 45,776 | 0.744444 | 0.337128 | 0.721556 | 0.503314 | 0.957050 | 0.213158 | 1.338907 |
| same-ID same-camera | C2-L03 | 45,776 | 0.655998 | 0.318625 | 0.616974 | 0.425832 | 0.846842 | 0.191510 | 1.236588 |
| same-ID different-camera | Baseline | 166,987 | 0.980708 | 0.279031 | 0.954839 | 0.777812 | 1.157930 | 0.568262 | 1.476210 |
| same-ID different-camera | C2-L03 | 166,987 | 0.856452 | 0.283329 | 0.822299 | 0.647833 | 1.029423 | 0.456323 | 1.372043 |
| different-ID | Baseline | 200,000 | 1.986254 | 0.145640 | 2.000804 | 1.908711 | 2.083568 | 1.729285 | 2.192818 |
| different-ID | C2-L03 | 200,000 | 1.988719 | 0.156476 | 2.009332 | 1.914379 | 2.090418 | 1.706774 | 2.198465 |

### 4.2 C2-L03 相对 Baseline 的变化

| pair 类型 | Δmean | Δstd | Δmedian | Δq25 | Δq75 | Δq05 | Δq95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| same-ID same-camera | −0.088446（−11.881%） | −0.018504（−5.489%） | −0.104582（−14.494%） | −0.077482 | −0.110208 | −0.021648 | −0.102319 |
| same-ID different-camera | **−0.124255（−12.670%）** | +0.004298（+1.540%） | **−0.132540（−13.881%）** | −0.129980 | −0.128507 | −0.111939 | −0.104167 |
| different-ID | +0.002464（+0.124%） | +0.010836（+7.441%） | +0.008528（+0.426%） | +0.005669 | +0.006850 | −0.022511 | +0.005647 |

Market 的 `same-ID different-camera` mean、median、q05、q25、q75 和 q95
全部下降，说明主体和主要分位数均发生左移；std 小幅增加，故不写成“所有 pair
都变近”。different-ID mean 与 median 略有上升，未观察到总体类间距离同步收缩；
但 q05 略降，仍需保留低距离尾部风险。

### 4.3 类间间隔

这里的间隔定义为：

\[
\text{gap}=
d(\text{different-ID})-
d(\text{same-ID different-camera}).
\]

| 统计量 | Baseline | C2-L03 | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| mean gap | 1.005547 | 1.132267 | +0.126720 | +12.602% |
| median gap | 1.045965 | 1.187034 | +0.141069 | +13.487% |

Market 的 mean 与 median separation gap 均扩大。

### 4.4 Market 结论

> 在当前 Market1501 person-only epoch-120 checkpoint 和统一分析协议下，
> C2-L03 的跨摄像头同身份距离均值、中位数及主要分位数均下降，其中均值下降
> 12.670%，中位数下降 13.881%；different-ID 均值未下降，mean/median
> separation gap 分别扩大 12.602% 和 13.487%。该结果支持当前检索嵌入空间中
> 更有利的距离结构，但不构成统计显著性或普遍规律证明。

旧 `market_epoch120_control` 中 94.8431% 的 same-ID pair 为 pid0-pid0，已由
`SUPERSEDED.md` 和证据等级 E0 标记为废弃；旧数值仅保留协议审计价值，不得与
person-only v2 合并或择优使用。

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
| 跨摄像头同身份统计是否具备人物语义 | 是；person-only v2 全部 `pid>0` | 是 |
| same-ID different-camera mean/median 变化 | −12.670% / −13.881% | −7.365% / −8.257% |
| different-ID mean 变化 | +0.124% | −0.516% |
| mean separation gap | +12.602% | +9.386% |
| 是否支持更有利的距离结构 | **当前 checkpoint 下支持** | **当前 checkpoint 下支持** |

综合表述：

> C2-L03 改善了当前 Market1501 与 DukeMTMC-reID 运行的检索指标，并在两个
> 当前 epoch-120 检索特征空间中观察到更有利的跨摄像头距离结构变化。该观察
> 来自各一组 checkpoint 的描述统计，不支持“C2 一定拉近所有数据集、所有
> checkpoint 或所有跨摄像头样本距离”。

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

Market 正式图表：

- `D:\thesis_reid\thesis_evidence\distance_analysis\2026-07-18\analysis_results\market_epoch120_person_only_v2\distance_histogram.png`
- `D:\thesis_reid\thesis_evidence\distance_analysis\2026-07-18\analysis_results\market_epoch120_person_only_v2\distance_boxplot.png`

Duke 图表：

- `D:\thesis_reid\thesis_evidence\distance_analysis\2026-07-18\analysis_results\duke_epoch120_control\distance_histogram.png`
- `D:\thesis_reid\thesis_evidence\distance_analysis\2026-07-18\analysis_results\duke_epoch120_control\distance_boxplot.png`

图中已注明 dataset、两组 checkpoint、BNNeck-after、L2 normalization、squared Euclidean、same-ID 全量 pair，以及 different-ID 均匀无放回抽样 `n=200000, seed=42`。Market 图还对应 `pid>0` 的 person-only 协议。

分析工具已由提交 `1d5f48ddd85a3e0bdb3396e86be22d9eeaebb9f9` 纳入版本管理；Market 修正版 metadata 记录的 Windows 工作树文件 SHA256 为 `5a48cd50769b50972e52c27ed93e744eb0723c13145a736b77b4a06e27c2a493`。

## 8. 证据边界与后续工作

已经完成：

- 两数据集样本顺序、pair 索引和 checkpoint 哈希核验；
- Market 在特征提取和 pair 生成前排除 `pid<=0`，并完整重生成数据、统计和图片；
- self 排除、\(i<j\)、pair 去重和三类互斥核验；
- 全量统计复算；
- 原始日志、checkpoint、CSV/JSON 和图表归档。

仍缺少：

1. 固定训练 seed 的多次独立重复实验；
2. 对不同 checkpoint epoch 的敏感性检查；
3. 与训练空间直接对应的 BNNeck-before、未显式 L2 特征分析；
4. hard-positive、近邻或检索排序局部 pair 的补充分析；
5. 配对样本具有相关性时更合适的不确定性估计，不能把数十万 pair 直接视为独立重复。
