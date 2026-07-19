# C2 创新点、证据边界与实验设计

> 证据快照：2026-07-19。本文档区分“代码可核验”“Git 已登记”“原始日志/
> checkpoint 已核验”“归档实验记录”和“待核实”，不把配置文件名当作实验结果。

## 1. 论文主方法与统一命名

当前论文主方法为 **C2（Cross-Camera Positive Only）**，最终配置记为
**C2-L03**：

```yaml
MODEL:
  PART_ATTENTION: True
  PART_ATTENTION_PARTS: 6
  CAMERA_AWARE_TRIPLET: False
  CROSS_CAMERA_POSITIVE_ONLY: True
  CROSS_CAMERA_POSITIVE_LAMBDA: 0.3
  CROSS_CAMERA_POSITIVE_MODE: "mean"
```

- C2 / Cross-camera positive only：只在新增辅助项中约束同身份、不同摄像头
  的正样本，不在该辅助项中增加负样本。
- C2-L03：C2 的 `mean` 聚合版本，辅助损失权重 \(\lambda=0.3\)，Part
  Attention 开启且 \(K=6\)。
- Baseline-Control：与 C2-L03 使用相同 Part Attention \(K=6\) 和训练/
  测试设置，只关闭 C2；它不是关闭 Part Attention 的“裸 BoT”。
- 完整 CAAT：代码类名 `CameraAwareTripletLoss`，作为消融。
- S2 / same-camera positive only：仅在独立 S2 分支存在，作为消融。

完整 CAAT 与 S2 均不是最终主方法。

## 2. C2 的代码定义与公式

### 2.1 训练特征口径

`modeling/baseline.py` 在训练阶段返回 `cls_score, global_feat`。启用 Part
Attention 时：

\[
g_i=\operatorname{GAP}(F_i)+\operatorname{PartAttention}_{K=6}(F_i).
\]

这里的 \(g_i\) 是 **BNNeck-before** 特征。普通 Triplet 和 C2 都接收该
`global_feat`；`CrossCameraPositiveLoss` 调用时没有传入
`normalize_feature=True`，因此训练辅助项**没有显式 L2 归一化**，距离为代码
`euclidean_dist()` 计算的欧氏距离，而不是平方欧氏距离。

这与第 6 节的检索距离分析不同：距离分析使用 **BNNeck-after + L2
normalize + squared Euclidean**。因此距离分析验证的是实际检索嵌入空间，
不是对训练时 C2 数值的直接复算。

### 2.2 有效 anchor 与辅助损失

对 batch 中 anchor \(i\)，跨摄像头同身份正样本集合为：

\[
\mathcal P_i^{cc}
=\{j\mid j\ne i,\ y_j=y_i,\ c_j\ne c_i\}.
\]

由于 \(c_j\ne c_i\)，self-pair 自然被排除。有效 anchor 集合为：

\[
\mathcal A
=\{i\mid |\mathcal P_i^{cc}|>0\}.
\]

C2-L03 使用 `mean` 聚合：

\[
L_{\mathrm{C2}}
=
\frac{1}{|\mathcal A|}
\sum_{i\in\mathcal A}
\left(
\frac{1}{|\mathcal P_i^{cc}|}
\sum_{j\in\mathcal P_i^{cc}}
\lVert g_i-g_j\rVert_2
\right).
\]

当 batch 中没有有效 anchor 时，代码返回与计算图相连的零损失。训练日志中的
`cross_camera_positive_count` 是 \(|\mathcal A|\)，即有效 anchor 数，不是
正样本 pair 数。

在当前 `softmax_triplet` 配置下，总损失为：

\[
L
=L_{\mathrm{id}}+L_{\mathrm{tri}}
+\lambda L_{\mathrm{C2}},
\qquad \lambda=0.3.
\]

必须区分两个数值恰好都为 0.3 的参数：

| 参数 | 配置项 | 含义 |
|---|---|---|
| C2 辅助损失权重 \(\lambda=0.3\) | `MODEL.CROSS_CAMERA_POSITIVE_LAMBDA` | 控制 \(L_{\mathrm{C2}}\) 在总损失中的权重 |
| 普通 Triplet margin \(m=0.3\) | `SOLVER.MARGIN` | 控制普通 Triplet 的间隔 |

C2 自身没有 margin。“Positive Only”只描述新增辅助项；原有普通 Triplet
仍包含 hardest positive 和 hardest negative。

### 2.3 与完整 CAAT、S2 的准确区别

完整 CAAT 对每个有效 anchor 选择跨摄像头 hardest positive，并在所有异身份
样本中选择 hardest negative：

\[
L_{\mathrm{CAAT}}
=\frac{1}{|\mathcal A'|}
\sum_{i\in\mathcal A'}
\left[
\max_{p\in\mathcal P_i^{cc}}d(g_i,g_p)
-\min_{n:y_n\ne y_i}d(g_i,g_n)
+m_{\mathrm{aux}}
\right]_+.
\]

当前登记的 CAAT-L05 使用辅助权重 0.5、辅助 margin 0.3。S2 则把正样本
范围改为“同身份、同摄像头、排除 self”，采用 mean 聚合且不增加辅助负样本。

| 方法 | 辅助正样本范围 | 聚合 | 新增辅助负样本 | 论文定位 |
|---|---|---|---|---|
| 完整 CAAT | 同 ID、不同 camera | hardest | hardest negative | 消融 |
| C2-L03 | 同 ID、不同 camera | mean | 无 | 主方法 |
| S2 | 同 ID、同 camera、排除 self | mean | 无 | 消融 |

## 3. 可写入论文的贡献点

以下内容可作为“方法贡献”组织；是否构成严格意义上的学术新颖性，仍需结合
完整相关工作检索，不能仅由本仓库证明。

### 3.1 面向跨摄像头检索目标的正样本显式约束

C2 同时使用 pid 和 camid 构造监督集合，把新增优化项集中在跨摄像头同身份
一致性上。相较完整 CAAT，C2 不重复引入辅助负样本，目标更简单，也便于通过
正样本范围消融解释。

### 3.2 正样本范围与辅助负样本的可解释消融

完整 CAAT、C2 与 S2 共同回答：

1. 跨摄像头条件是否必要；
2. 额外相机感知 hardest negative 是否必要；
3. mean 与 hard 正样本聚合有何差异；
4. 收益是否来自针对跨摄像头类内结构的约束，而不是一般类内收紧。

### 3.3 从“性能、嵌入结构、监督机会”三层验证

- Market1501 和 DukeMTMC-reID 的 Rank-1/mAP 检验最终检索性能；
- 配对 checkpoint 的距离分布检验检索嵌入结构；
- 离线 sampler 覆盖率说明 C2 在多少 anchor 上有机会生效。

三类证据不能互相替代。尤其是 Market 距离结果并不支持“跨摄像头距离整体
收缩”，因此论文不能把 C2 的机制写成已在所有数据集上得到一致验证。

### 3.4 Part Attention 的定位

Part Attention \(K=6\) 是主方法与公平控制组共同使用的特征骨架。它可以作为
前期研究成果和方法基础介绍，但 C2-L03 与 Baseline-Control 的性能差值不能
归因于 Part Attention。

## 4. 主结果与证据等级

### 4.1 Market1501

| 实验 | epoch | Rank-1 | Rank-5 | Rank-10 | mAP | re-ranking | 证据 |
|---|---:|---:|---:|---:|---:|---|---|
| Baseline-Control | 120 | 94.4% | 98.1% | 98.9% | 85.5% | no | 原始日志与 checkpoint 已核验 |
| **C2-L03** | **120** | **95.0%** | **98.5%** | **99.1%** | **87.8%** | **no** | **原始日志与 checkpoint 已核验** |

C2-L03 相对控制组的当前单次运行差值为 Rank-1 **+0.6** 个百分点、mAP
**+2.3** 个百分点。C2-L03 的运行时间为 0:52:35，登记训练 commit 为
`7b8195d4a02b536b27ab4d6ac80652091db7468f`，epoch 120 checkpoint 为
`resnet50_checkpoint_22320.pt`，SHA256 为
`2008541aa5738c8bc3a440504d0bbb055b646f5d8e97a25e113f2d2e463d497b`。
因此 95.0%/87.8% 已不再只是“用户确认数值”。

Market 控制组 epoch 120 checkpoint 的 SHA256 为
`171aea42cb8df1464461887c78352205d61319341cd4ea93afc9dc5e9fc34edf`。
两组固定训练 seed 均未记录，也没有多 seed 统计，故只能写“当前实验取得”
和“当前单次运行提高”，不能写“稳定提升”“统计显著”。

### 4.2 \(\lambda\) 敏感性

同一归档服务器实验记录给出的 Market1501 序列为：

| 配置 | \(\lambda\) | epoch | Rank-1 | mAP | 运行时间 | 当前证据 |
|---|---:|---:|---:|---:|---:|---|
| C2-L01 | 0.1 | 120 | 94.7% | 87.5% | 0:52:25 | 归档实验表；原始日志/checkpoint 未随当前证据归档 |
| **C2-L03** | **0.3** | **120** | **95.0%** | **87.8%** | **0:52:35** | **原始日志/checkpoint 已核验** |
| C2-L05 | 0.5 | 120 | 94.8% | 87.5% | 0:51:46 | 归档实验表；原始日志/checkpoint 未随当前证据归档 |
| C2-L10 | 1.0 | 120 | 94.6% | 87.0% | 0:51:47 | 归档实验表；原始日志/checkpoint 未随当前证据归档 |

该归档序列登记训练 commit 为 `7b8195d`，seed 未记录。在这组单次序列内，
\(\lambda=0.3\) 同时取得最高 Rank-1 和 mAP，因此作为当前最终配置。早期独立
C2 \(\lambda=0.5\) 运行的 95.0%/87.7% 及其重复运行 95.0%/87.3% 属于另一组
Git 登记实验，不能与上表的 C2-L05 94.8%/87.5% 混为同一次运行。

### 4.3 DukeMTMC-reID 跨数据集验证

| 实验 | 选用 epoch | Rank-1 | Rank-5 | Rank-10 | mAP | re-ranking | 证据 |
|---|---:|---:|---:|---:|---:|---|---|
| Baseline-Control | 80 | 86.7% | 94.0% | 95.8% | 75.7% | no | Git 登记、原始日志与 checkpoint 已核验 |
| **C2-L03** | **120** | **88.4%** | **95.2%** | **96.8%** | **78.7%** | **no** | **Git 登记、原始日志与 checkpoint 已核验** |

按各自登记 checkpoint，C2-L03 的 Rank-1/mAP 差值为 **+1.7/+3.0** 个
百分点。两组运行时间分别为 0:47:46 和 0:53:21，训练 commit 为
`3ce47246d67c1c43befd651ea44082216167478f`，结果登记 commit 为
`6f49104f6c9413b91df515c8a934eaa42d7a2ff3`。

Duke 主表按各自登记 checkpoint 汇报，因此 Baseline 取 epoch 80、C2-L03
取 epoch 120。用于距离分析的公平配对则固定为 epoch 120：Baseline 为
86.4%/76.3%，C2-L03 为 88.4%/78.7%，对应 Rank-1/mAP 差值
+2.0/+2.4 个百分点。两种口径必须分表说明。

原始产物现已归档，Duke 不再属于“只有 Git 登记、缺少日志/checkpoint”的
状态；但训练 seed 仍未显式记录，也没有多 seed 重复。GPU 型号 RTX 4090
来自归档实验表，原始训练日志本身只记录使用 1 块 GPU。

### 4.4 CAAT、S2 与历史 C2 消融

| 方法 | \(\lambda\) | Rank-1 | mAP | 证据与定位 |
|---|---:|---:|---:|---|
| 完整 CAAT-L05 | 0.5 | 94.2% | 85.4% | Git `73a44c0` 登记；消融 |
| S2-SCPO | 0.5 | 94.4% | 86.8% | Git `a4a42b3` 登记；消融 |
| C2-CCPO | 0.5 | 95.0% | 87.7% | Git `9b81850` 登记；历史 C2 |
| C2-CCPO-Repeat | 0.5 | 95.0% | 87.3% | Git `9b81850` 登记；无固定 seed 的历史重复 |
| **C2-L03** | **0.3** | **95.0%** | **87.8%** | **原始日志/checkpoint 已核验；当前主方法** |

CAAT、S2 和历史 C2 \(\lambda=0.5\) 的原始日志/checkpoint 尚未随当前证据
归档，表中只能标作 Git 已登记结果。

## 5. 公平对照与实验设计

### 5.1 主对比

| 组别 | Part Attention | 普通 Triplet | C2 | \(\lambda\) |
|---|---:|---:|---:|---:|
| Baseline-Control | ✓，K=6 | ✓ | × | 0 |
| C2-L03 | ✓，K=6 | ✓ | ✓，mean | 0.3 |

Market 与 Duke 的主对比应保持 backbone、输入尺寸、ImageNet 预训练、增强、
sampler、batch size、每身份样本数、optimizer、epoch、学习率策略、测试特征
和 L2 normalization 一致；re-ranking 均关闭。除 C2 开关、有效辅助权重和
隔离的输出目录外，其余设置对齐。

### 5.2 消融应回答的问题

1. `mean` 与 `hard` 的跨摄像头正样本聚合差异；
2. 跨摄像头正样本条件相对 same-camera 条件是否更符合检索目标；
3. 完整 CAAT 的额外辅助负样本是否必要；
4. \(\lambda\) 从 0.1、0.3、0.5 到 1.0 的敏感性；
5. C2 的收益是否伴随跨摄像头类内结构改善；
6. 结论能否在 Market 和 Duke 上同时成立。

## 6. 距离分布机制分析

### 6.1 已核验协议

- 样本：同一数据集的 query+gallery，Baseline 与 C2-L03 样本顺序一致。
- 特征：推理实际输出的 BNNeck-after 特征，再执行 L2 normalize。
- 距离：squared Euclidean；关闭 re-ranking 和 camera-mean debias。
- pair：只统计 \(i<j\)，排除 self 和反向重复；三类 mask 互斥。
- different-ID：从全部候选 pair 中以 seed=42 均匀无放回抽样 200,000 对。
- 两模型复用完全相同的 pair 索引；距离分块大小为 4096。
- 标准差为总体标准差（ddof=0），分位数使用 NumPy 默认线性插值。

| 数据集 | 样本数 | sample hash | pair hash |
|---|---:|---|---|
| Market1501 | 19,281 | `ede26d3a28aece193741f618d26bd5b3ceecacce8ed359589322030a01d14461` | `193dd9ccfd552c6fcf7c402e2d31ad469ee51633cafc30142e2f3a835b522084` |
| DukeMTMC-reID | 19,889 | `be498c3a413de371c27b610728b7a4c2465307e6c6e74ee5184b4e92df2a7f2e` | `1e0ffcd68de323d8c0f56d69737a673d0cdda11db1c0b4c423ccae59e96a234b` |

| 数据集 | 距离分析 checkpoint | SHA256 |
|---|---|---|
| Market1501 | Baseline epoch 120 | `171aea42cb8df1464461887c78352205d61319341cd4ea93afc9dc5e9fc34edf` |
| Market1501 | C2-L03 epoch 120 | `2008541aa5738c8bc3a440504d0bbb055b646f5d8e97a25e113f2d2e463d497b` |
| DukeMTMC-reID | Baseline epoch 120 | `ae0713f27dbc80684c4840d0d0138748dcb2f03228d9fe7de4cc9c6bf36f7a55` |
| DukeMTMC-reID | C2-L03 epoch 120 | `d5ded542fd82bb96068106a7769cb9290c5939bde4862fb60e8ffbb89d5f2e0f` |

两组分析的 `identical_sample_order`、`identical_pair_indices`、
`self_pairs_excluded`、`unordered_pairs_unique` 和
`pair_types_mutually_exclusive` 检查均通过，checkpoint 加载无 missing/
skipped 参数。

### 6.2 Market1501

| pair 类别 | count | Baseline mean±std | C2-L03 mean±std | Baseline median | C2-L03 median |
|---|---:|---:|---:|---:|---:|
| same-ID same-camera | 717,000 | 1.698396 ± 0.348022 | 1.701136 ± 0.365142 | 1.783479 | 1.794443 |
| same-ID different-camera | 3,408,766 | 1.773591 ± 0.270782 | 1.772773 ± 0.291417 | 1.825205 | 1.832293 |
| different-ID | 200,000 | 1.988320 ± 0.141430 | 1.989686 ± 0.151847 | 2.001366 | 2.008878 |

| pair 类别 | Baseline q05/q25/q75/q95 | C2-L03 q05/q25/q75/q95 |
|---|---|---|
| same-ID same-camera | 0.896448 / 1.603586 / 1.914227 / 2.064273 | 0.810956 / 1.613689 / 1.923648 / 2.071109 |
| same-ID different-camera | 1.233907 / 1.675943 / 1.945302 / 2.090542 | 1.181387 / 1.682981 / 1.951205 / 2.095465 |
| different-ID | 1.741437 / 1.911494 / 2.082514 / 2.191031 | 1.718618 / 1.915006 / 2.088958 / 2.195377 |

Market 的 same-ID different-camera 均值仅下降 0.000818（-0.046%），但
median 上升 0.007088（+0.388%），q25、q75、q95 均上升，std 从
0.270782 增至 0.291417。`different-ID − same-ID different-camera` 的均值
间隔由 0.214728 增至 0.216913（+0.002185，+1.017%），median 间隔仅由
0.176162 增至 0.176585（+0.000424，+0.241%）。

因此，**Market 当前全 pair 描述统计不支持“C2 使跨摄像头同身份距离整体
收缩”**。微小均值下降主要与低分位变化并存，而中位数和主要高分位没有同向
下降；不能只挑选均值来证明预期机制。不同身份距离没有整体不利左移，间隔
略增，但幅度很小。

### 6.3 DukeMTMC-reID（epoch 120 配对）

| pair 类别 | count | Baseline mean±std | C2-L03 mean±std | Baseline median | C2-L03 median |
|---|---:|---:|---:|---:|---:|
| same-ID same-camera | 325,738 | 0.925329 ± 0.385137 | 0.883940 ± 0.382157 | 0.923258 | 0.876288 |
| same-ID different-camera | 179,162 | 1.167991 ± 0.375483 | 1.081971 ± 0.384484 | 1.132603 | 1.039084 |
| different-ID | 200,000 | 1.975909 ± 0.139658 | 1.965717 ± 0.147499 | 1.989833 | 1.984430 |

| pair 类别 | Baseline q05/q25/q75/q95 | C2-L03 q05/q25/q75/q95 |
|---|---|---|
| same-ID same-camera | 0.314469 / 0.626278 / 1.204942 / 1.572626 | 0.291622 / 0.587737 / 1.147529 / 1.546491 |
| same-ID different-camera | 0.607502 / 0.880371 / 1.435743 / 1.833666 | 0.518360 / 0.782012 / 1.357593 / 1.768922 |
| different-ID | 1.730713 / 1.899474 / 2.069762 / 2.174437 | 1.701981 / 1.889030 / 2.064860 / 2.167414 |

Duke 的 same-ID different-camera 均值下降 0.086020（-7.365%），median
下降 0.093520（-8.257%），q05、q25、q75、q95 均下降。different-ID
均值也下降 0.010192（-0.516%），但幅度远小于跨摄像头同身份距离。
`different-ID − same-ID different-camera` 的均值间隔由 0.807918 增至
0.883746（+0.075828，+9.386%），median 间隔由 0.857230 增至 0.945346
（+0.088117，+10.279%）。

因此允许写：

> 在当前 DukeMTMC-reID epoch 120 配对 checkpoint、BNNeck-after + L2
> 与平方欧氏距离协议下，C2-L03 缩小了跨摄像头同身份样本的检索嵌入距离；
> 尽管不同身份距离也有轻微下降，跨摄像头类内距离下降幅度更大，净分离间隔
> 在描述统计上扩大。

该结论只适用于当前 checkpoint 和分析协议。pair 之间并非相互独立，且没有
多 seed 重复，因此不能表述为统计显著或普遍规律。

### 6.4 跨数据集机制结论

检索指标在 Market 和 Duke 上都向好，但距离分布证据并不完全一致：

- Market：没有观察到跨摄像头同身份距离的整体收缩；
- Duke：在 epoch 120 公平配对下观察到均值、中位数和全部主要分位数下降，
  且类间间隔扩大。

论文应如实报告这种差异。当前结果支持“C2 能够改善检索性能，并在 Duke 的
当前检索空间中呈现预期结构变化”，不支持“C2 在所有数据集上都通过全局拉近
跨摄像头同身份距离起效”。

## 7. Batch 内有效 cross-camera positive 覆盖率

固定 seed=42、batch size=64、每身份 \(K=4\)（每 batch 16 个 pid），离线
连续模拟 10 个 epoch：

| 数据集 | 训练图像 | batch 数 | 总 anchors | 有效 anchors | 加权有效比例 |
|---|---:|---:|---:|---:|---:|
| Market1501 | 12,936 | 1,834 | 117,376 | 114,852 | **97.8496%** |
| DukeMTMC-reID | 16,522 | 2,259 | 144,576 | 135,980 | **94.0543%** |

| 数据集 | batch 比例均值±总体 std | min/median/max | 零有效 batch | 每 batch 唯一 pid | 唯一 camera 均值（范围） |
|---|---:|---:|---:|---:|---:|
| Market1501 | 97.8496% ± 3.6160% | 81.25% / 100% / 100% | 0/1,834 | 16 | 5.9771（5–6） |
| DukeMTMC-reID | 94.0543% ± 7.0234% | 43.75% / 93.75% / 100% | 0/2,259 | 16 | 7.8601（5–8） |

附加的“有序 cross-camera positive pair 占全部同 ID 有序 positive pair”
比例为 Market 77.2196%、Duke 67.0614%。论文主口径是有效 anchor 覆盖率，
不能把 pair 比例和 anchor 比例混用。

允许写：

> 在固定 64/4 PK 采样协议的 10 个离线模拟 epoch 中，Market1501 和
> DukeMTMC-reID 分别有 97.85% 和 94.05% 的 anchor 至少拥有一个同身份、
> 不同摄像头正样本，且没有出现完全缺少有效 anchor 的 batch，说明当前
> sampler 为 C2 提供了较高比例的监督机会。

同时必须说明：

- 这是离线 sampler 覆盖率，不是历史训练 batch 的精确回放；
- 它反映“有机会接受监督”，不是模型准确率；
- 它不能单独证明 C2 提高了检索性能；
- 机制解释必须与 Rank-1/mAP 和距离分布联合讨论。

## 8. 证据位置与仍待补齐项

距离与原始实验产物位于：

`D:\thesis_reid\thesis_evidence\distance_analysis\2026-07-18\`

其中 Market 和 Duke 的完整距离结果分别位于：

- `analysis_results\market_epoch120_control\`
- `analysis_results\duke_epoch120_control\`

当前仍缺少或需要补强：

1. Market 与 Duke 的显式固定训练 seed，以及至少多 seed 重复实验的均值和
   标准差；现有结果不能支持统计显著性。
2. C2-L01、C2-L05、C2-L10 的原始日志/checkpoint；当前只有归档服务器实验
   表，不能把这些行提升为“原始产物已核验”。
3. 完整 CAAT、S2 和早期 C2 \(\lambda=0.5\) 的原始日志/checkpoint；当前
   只有 Git 登记。
4. Market 距离机制的进一步预注册分析，例如 hardest-positive、query-gallery
   检索局部邻域或训练特征空间分析；在补充前必须保留当前不支持整体收缩的
   结论，不能选择性解释。
5. Part Attention/C2 的参数量、FLOPs、训练与推理开销。
6. 注意力可视化、检索成功/失败案例，以及相关工作对“创新性”的系统比较。
