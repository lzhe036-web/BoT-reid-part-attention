# 研究生论文目录草案

## 建议题目

> 基于跨摄像头正样本显式优化的行人重识别方法研究

备选：

> 融合局部特征与跨摄像头正样本约束的行人重识别研究

当前主方法统一称为 C2，主配置为 C2-L03。完整 CAAT 和 Same-Camera
Positive Only 只在消融实验出现。

## 摘要

- 背景：跨摄像头视角、光照和风格变化造成类内差异。
- 问题：普通 Triplet 没有显式区分同 camera 与跨 camera 正样本。
- 基础：BoT + Part Attention 特征骨架。
- 方法：C2 在普通 ID/Triplet 目标之外，仅对同 ID、不同 camera 的正样本
  增加均值距离约束；不引入新的辅助负样本项。
- 主配置：C2-L03，辅助权重 λ=0.3。
- 当前已核验结果：
  - Market1501：C2-L03 在 epoch 120 取得 Rank-1 95.0%、mAP 87.8%，
    原始训练日志和 checkpoint 已在外部证据包中核验。
  - DukeMTMC-reID：C2-L03 在 epoch 120 取得 Rank-1 88.4%、mAP
    78.7%，原始训练日志和 checkpoint 已核验；登记的 Baseline-Control
    为 Rank-1 86.7%、mAP 75.7%。
- 机制分析：
  - 固定 64/4 PK 采样协议下，Market 和 Duke 的离线有效 anchor 覆盖率
    分别为 97.8496% 和 94.0543%。
  - Market 的全体无序 pair 描述统计不支持“跨摄像头同身份距离整体明显
    拉近”；Duke 的 matched-epoch-120 分析则观察到跨摄像头同身份距离
    下降且与异身份均值距离的间隔扩大。
- 证据边界：训练 seed 未记录，尚无固定 seed 重复实验；上述变化只能按
  当前 checkpoint 与分析协议陈述，不写“统计显著”或“普遍有效”。

关键词：行人重识别；跨摄像头匹配；正样本约束；度量学习；局部注意力。

## 第 1 章 绪论

### 1.1 研究背景与意义

- 行人重识别的任务和应用。
- 多摄像头检索与单摄像头分类的差异。
- 视角、光照、遮挡、分辨率和相机风格变化。

### 1.2 国内外研究现状

1. 全局特征方法。
2. 局部/人体部件方法。
3. 注意力机制。
4. 分类损失与度量学习。
5. 困难样本挖掘。
6. camera-aware、camera-invariant 与域适应。

文献事实需从原论文核验，不能只引用项目 README。

### 1.3 现有方法的不足

- 全局池化可能弱化局部判别区域。
- 普通 Triplet 的正样本选择不区分 camera 条件。
- 复杂的正负样本联合约束不容易解释收益来源。
- PK 采样提供同身份样本，但不保证每个 anchor 都有跨 camera 正样本。

### 1.4 研究内容

- 建立 BoT + Part Attention 公平实验基础。
- 提出 C2 跨摄像头正样本辅助约束。
- 通过 λ 敏感性确定 C2-L03。
- 用完整 CAAT 与 S2 构建机制消融。
- 在 Market 和 Duke 进行主对比。
- 通过距离分布和有效 anchor 覆盖率分析机制。

### 1.5 主要贡献

1. 构建 pid/camid 联合掩码的跨摄像头正样本显式优化方法。
2. 设计 CAAT/C2/S2 可解释消融，分离跨 camera 条件和辅助负样本的作用。
3. 从检索距离分布与采样监督覆盖率两个层面分析方法机制，并如实报告
   Market 与 Duke 在全局距离分布上的差异，限定机制结论的适用范围。

### 1.6 论文结构

概述第 2—6 章。

## 第 2 章 相关理论与关键技术

### 2.1 行人重识别问题定义

- 训练集、query、gallery。
- pid、camid 和标准检索协议。
- 排除同 pid 同 camid gallery 的规则。

### 2.2 卷积特征与 ResNet

- 残差网络。
- 全局平均池化。
- ImageNet 预训练与迁移学习。

### 2.3 BoT Strong Baseline

- Last stride、Random Erasing、Label Smoothing、BNNeck、Warm-up。
- ID Loss、Triplet Loss、Center Loss。
- 训练时 ID 分类分支使用 BNNeck-after 特征，普通 Triplet 及本文 C2
  使用 BNNeck-before 的融合特征；测试协议输出 BNNeck-after 特征。

### 2.4 局部特征与 Part Attention

- 水平区域划分和局部池化。
- 区域 softmax 权重。
- 全局/局部残差式融合。
- 本文中它是 Baseline-Control 与 C2 共享的特征骨架。

### 2.5 三元组损失与样本挖掘

- Batch-hard。
- hardest positive/negative。
- margin 与辅助损失权重的区别。

### 2.6 跨摄像头类内差异

- camera 风格与视角变化。
- camid 监督的合理使用。
- 训练期相机感知约束与测试期去偏的区别。

### 2.7 CMC、Rank-k 与 mAP

- 指标定义。
- re-ranking 单独报告。

## 第 3 章 C2 跨摄像头正样本显式优化方法

### 3.1 问题定义与动机

定义训练 batch
\(\mathcal B=\{(x_i,y_i,c_i)\}_{i=1}^{B}\)，其中 \(y_i\) 为身份标签、
\(c_i\) 为摄像头标签。说明普通 batch-hard Triplet 将同身份正样本放在
同一集合中，并未显式区分 same-camera 与 cross-camera positives；因而
新增目标应回答“哪些 anchor 真正拥有跨摄像头正样本”以及“约束是否改变
检索嵌入中的跨摄像头类内结构”。

### 3.2 总体结构

```text
Image → ResNet-50 → Global + Part Attention → fused feature (BN-before)
                                                   ├→ ordinary Triplet
                                                   ├→ C2 auxiliary loss
                                                   └→ BNNeck → ID classifier
                                                              └→ test embedding
```

### 3.3 统一特征骨架

- ResNet-50、Last stride=1。
- K=6 Part Attention。
- Baseline-Control 与 C2-L03 共用相同 Part Attention 特征骨架，方法
  对比不把局部模块差异混入 C2 收益。
- 训练阶段，模型向损失函数返回融合后的 BNNeck-before 全局特征；
  ordinary Triplet 与 C2 都在该空间计算，ID 分类使用 BNNeck-after 特征。
- 检索阶段按主测试协议输出 BNNeck-after 特征，再执行 L2 normalize；
  论文不得把训练损失空间和最终检索嵌入空间混写。

### 3.4 C2 正样本集合与有效 anchor

\[
\mathcal P_i^{cc}=\{j\mid j\ne i,\ y_j=y_i,\ c_j\ne c_i\},
\qquad
\mathcal V=\{i\mid |\mathcal P_i^{cc}|>0\}.
\]

- 只有 \(i\in\mathcal V\) 的 anchor 参与 C2；没有跨 camera positive 的
  anchor 不进入该辅助项。
- `cross_camera_positive_count` 对应 \(|\mathcal V|\)，即有效 anchor
  数，而不是跨摄像头正样本 pair 数。
- 当前主配置使用 mean 聚合：

\[
L_{\mathrm{C2}}
=\frac{1}{|\mathcal V|}
\sum_{i\in\mathcal V}
\frac{1}{|\mathcal P_i^{cc}|}
\sum_{j\in\mathcal P_i^{cc}} d(f_i,f_j),
\]

其中 \(f_i\) 是训练阶段的 BNNeck-before 融合特征。若
\(|\mathcal V|=0\)，该 batch 的 C2 项为零。

### 3.5 联合损失与超参数

\[
L=L_{\mathrm{id}}+L_{\mathrm{tri}}+\lambda L_{\mathrm{C2}},
\qquad \lambda=0.3.
\]

- `MODEL.CROSS_CAMERA_POSITIVE_LAMBDA=0.3` 是辅助损失权重 λ。
- `SOLVER.MARGIN=0.3` 是 ordinary Triplet 的 margin。
- 两者数值恰好相同但作用不同，公式、配置表和正文必须分列说明。

### 3.6 与完整 CAAT、S2 的区别

| 方法 | 辅助正样本范围 | 聚合 | 额外辅助负样本 | 论文定位 |
|---|---|---|---|---|
| C2 | same-ID different-camera | mean | 无 | 主方法 |
| 完整 CAAT | same-ID different-camera | hardest | hardest negative | 消融 |
| S2 | same-ID same-camera | mean | 无 | 消融 |

- C2 与 CAAT 的对比回答“额外辅助负样本和 hard 聚合是否必要”。
- C2 与 S2 的对比回答“跨摄像头条件是否必要”。
- CAAT 和 S2 均保留为研究演进与机制消融，不写成最终主方法。

### 3.7 训练、推理与公平控制

- RandomIdentitySampler：batch 64、每身份 4 张。
- Adam、120 epoch、学习率节点 40/70。
- 训练：ordinary Triplet 与 C2 使用 BNNeck-before 特征，C2 的有效
  anchor 由同 batch 的 pid/camid 掩码确定。
- 推理：BNNeck-after、L2 normalize、squared Euclidean、no re-ranking；
  不使用 camera-mean debias。
- Baseline-Control 与 C2-L03 除 C2 开关和有效辅助权重外，保持 backbone、
  Part Attention、输入、增强、sampler、优化器、学习率策略和测试协议一致。

### 3.8 方法复杂度

- C2 只增加训练期 batch 内 pair 距离计算。
- 推理模型结构与公平控制组一致。
- 参数量、FLOPs、训练耗时需要实测。

## 第 4 章 实验与分析

### 4.1 数据集

- Market1501。
- DukeMTMC-reID。
- 官方样本数和协议需引用数据集原始文献。

### 4.2 实验设置与证据等级

- 软件、GPU、输入尺寸、增强、采样、优化器、epoch。
- 证据分为“代码可核验”“Git 已登记”“原始日志/checkpoint 已核验”
  “用户确认”和“待核实/待实验”。
- Market C2-L03、Market Baseline-Control、Duke C2-L03 和 Duke
  Baseline-Control 的原始日志与 checkpoint 已在外部证据包中核验；距离
  分析的 metadata、sample hash、pair hash 和 checkpoint SHA256 也已核对。
- 训练 seed 未记录，且没有固定 seed 重复运行，所有主指标和距离变化均按
  当前单次运行/当前 checkpoint 描述。
- GPU 型号可引用已归档实验登记，但应注明原始训练日志只记录了单 GPU，
  未保存直接的设备查询输出。

### 4.3 Market1501 主结果

| 方法 | checkpoint epoch | Rank-1 | Rank-5 | Rank-10 | mAP | 证据 |
|---|---:|---:|---:|---:|---:|---|
| Baseline-Control | 120 | 94.4% | 98.1% | 98.9% | 85.5% | 原始日志/checkpoint 已核验 |
| C2-L03 | 120 | **95.0%** | **98.5%** | **99.1%** | **87.8%** | 原始日志/checkpoint 已核验 |

- 当前单次运行差值：Rank-1 +0.6、mAP +2.3 个百分点。
- C2-L03 原始训练记录对应 commit `7b8195d`，运行时长约 52 分 35 秒；
  checkpoint SHA256 为
  `2008541aa5738c8bc3a440504d0bbb055b646f5d8e97a25e113f2d2e463d497b`。
- Baseline epoch-120 checkpoint SHA256 为
  `171aea42cb8df1464461887c78352205d61319341cd4ea93afc9dc5e9fc34edf`。
- 上述结果不再仅标为“用户确认”，但因 seed 和重复实验缺失，不推导稳定性
  或统计显著性。

### 4.4 Duke 跨数据集验证

| 方法 | 登记 checkpoint epoch | Rank-1 | Rank-5 | Rank-10 | mAP | 证据 |
|---|---:|---:|---:|---:|---:|---|
| Duke-Baseline-Control | 80 | 86.7% | 94.0% | 95.8% | 75.7% | 原始日志/checkpoint 已核验 |
| Duke-C2-L03 | 120 | **88.4%** | **95.2%** | **96.8%** | **78.7%** | 原始日志/checkpoint 已核验 |

- 两组仅 C2 开关不同。
- 按当前登记 checkpoint，Rank-1、Rank-5、Rank-10 和 mAP 差值分别为
  +1.7、+1.2、+1.0 和 +3.0 个百分点。
- 距离机制分析为控制 epoch，使用 Baseline epoch 120（86.4%/76.3%）
  与 C2-L03 epoch 120（88.4%/78.7%），不要把 Baseline epoch 80 的主表
  数值代入 matched-epoch 距离分析。
- Duke 训练记录对应 commit `3ce4724`；Baseline/C2 epoch-120 checkpoint
  SHA256 分别为
  `ae0713f27dbc80684c4840d0d0138748dcb2f03228d9fe7de4cc9c6bf36f7a55`
  和
  `d5ded542fd82bb96068106a7769cb9290c5939bde4862fb60e8ffbb89d5f2e0f`。
- 原始产物已核验不等于重复性已验证；seed 和多次独立运行仍缺失。

### 4.5 λ 敏感性

| 配置 | λ | Rank-1 | mAP | 当前证据 |
|---|---:|---:|---:|---|
| C2-L01 | 0.1 | 94.7% | 87.5% | 服务器归档实验记录 |
| C2-L03 | 0.3 | **95.0%** | **87.8%** | 原始日志/checkpoint 已核验 |
| C2-L05 | 0.5 | 94.8% | 87.5% | 服务器归档实验记录 |
| C2-L10 | 1.0 | 94.6% | 87.0% | 服务器归档实验记录 |

- 同一归档序列中 λ=0.3 数值最高，因此选为当前主配置。
- L01/L05/L10 尚未逐一核验原始日志和 checkpoint；该表用于当前实验整理，
  不能替代固定 seed 重复实验，也不能证明 λ=0.3 在随机波动下必然最优。

### 4.6 CAAT 与 S2 消融

| 方法 | 新增正样本 | 聚合 | 新增负样本 | Rank-1 | mAP | 当前证据 |
|---|---|---|---|---:|---:|---|
| 完整 CAAT-L05 | 跨 camera | hardest | hardest negative | 94.2% | 85.4% | Git/文档登记，原始产物待核实 |
| S2 | 同 camera | mean | 无 | 94.4% | 86.8% | Git/文档登记，原始产物待核实 |
| C2-L03 | 跨 camera | mean | 无 | **95.0%** | **87.8%** | 原始日志/checkpoint 已核验 |

- 完整 CAAT 和 S2 仅用于解释正样本范围、聚合方式及额外辅助负样本的作用，
  不作为最终方法。
- 三行证据等级不同；在补齐 CAAT/S2 原始日志、checkpoint、配置哈希和
  seed 前，消融讨论只作当前登记结果的描述性比较。

### 4.7 距离分布分析

- 协议：query+gallery、BNNeck-after、L2 normalize、squared Euclidean、
  no re-ranking、no camera-mean debias、排除 self-pair、只取 \(i<j\)。
- Baseline 与 C2-L03 使用相同样本顺序和相同无序 pair；sample hash、
  pair hash、checkpoint SHA256 和三类互斥性均已核验。different-ID 以
  seed=42 均匀无放回抽取 200,000 对。

三类 pair 数：

| 数据集 | same-ID same-camera | same-ID different-camera | different-ID |
|---|---:|---:|---:|
| Market1501 | 717,000 | 3,408,766 | 200,000 |
| DukeMTMC-reID | 325,738 | 179,162 | 200,000 |

跨摄像头同身份距离与分离间隔的核心描述统计：

| 数据集与统计量 | Baseline | C2-L03 | C2 − Baseline |
|---|---:|---:|---:|
| Market cross-camera mean | 1.773591 | 1.772773 | -0.000818（-0.046%） |
| Market cross-camera median | 1.825205 | 1.832293 | +0.007088（+0.388%） |
| Market different-ID mean | 1.988320 | 1.989686 | +0.001366（+0.069%） |
| Market mean gap | 0.214728 | 0.216913 | +0.002185（+1.017%） |
| Duke cross-camera mean | 1.167991 | 1.081971 | -0.086020（-7.365%） |
| Duke cross-camera median | 1.132603 | 1.039084 | -0.093520（-8.257%） |
| Duke different-ID mean | 1.975909 | 1.965717 | -0.010192（-0.516%） |
| Duke mean gap | 0.807918 | 0.883746 | +0.075828（+9.386%） |

- Market：cross-camera mean 仅微降，而 median、q25、q75、q95 上升且
  标准差增大，因此不支持“跨摄像头同身份距离整体明显拉近”。different-ID
  mean 略升，mean gap 略增；检索指标改善与“全体正样本整体左移”并不等价。
- Duke：matched epoch 120 下，cross-camera mean、median 以及
  q05/q25/q75/q95 均下降；different-ID mean 也下降，但幅度较小，mean
  gap 扩大 0.075828，median gap 扩大 0.088117（+10.279%）。因此只能写：
  “在当前 Duke checkpoint 和分析协议下，C2-L03 缩小了跨摄像头同身份
  检索嵌入距离，并扩大了描述统计上的类间间隔。”
- pair 并非相互独立，且没有跨 seed 重复；不把 pair 数量大写成统计显著，
  也不把 Duke 结论外推为所有数据集上的普遍机制。
- 完整的 count、std、分位数、hash、直方图和箱线图在
  `paper_notes/distance_analysis.md` 展开。

### 4.8 有效 cross-camera anchor 覆盖率

- 协议：RandomIdentitySampler，batch size=64、K=4、seed=42，各离线
  模拟 10 个完整 epoch。

| 数据集 | 总 anchor | 有效 anchor | 加权有效率 | batch 比例均值 ± 总体标准差 | min/median/max | 零有效 batch |
|---|---:|---:|---:|---:|---:|---:|
| Market1501 | 117,376 | 114,852 | 97.8496% | 97.8496% ± 3.6160% | 81.25%/100%/100% | 0/1,834 |
| DukeMTMC-reID | 144,576 | 135,980 | 94.0543% | 94.0543% ± 7.0234% | 43.75%/93.75%/100% | 0/2,259 |

- 两数据集每 batch 的唯一 pid 均固定为 16；Market 唯一 camid 数均值
  5.9771（范围 5—6），Duke 为 7.8601（范围 5—8）。
- 有序 cross-camera positive pair 比例分别为 Market 77.2196%、Duke
  67.0614%，该指标与“有效 anchor 比例”定义不同，需分开呈现。
- 这是离线 sampler 的监督机会覆盖率，不是历史训练 batch 的精确回放，
  不是模型准确率，也不能单独证明 C2 带来性能提升；解释时须与 Rank-1/mAP
  和距离分布共同使用。

### 4.9 可视化与失败案例

- 已生成的三类距离归一化直方图与箱线图。
- 待补：注意力权重、Top-k 检索和 t-SNE/UMAP。
- 遮挡、相似服装和跨 camera 风格变化。

### 4.10 复杂度

- 参数量、FLOPs、显存、训练和推理耗时。

### 4.11 结果讨论与局限

- 训练 seed 未记录，主结果和机制分析均缺少固定 seed 重复实验。
- Market 的全体 pair 分布不支持“C2 整体拉近跨摄像头同身份距离”，而
  Duke 支持；机制具有数据集/当前 checkpoint 依赖，需进一步复验。
- L01/L05/L10、CAAT 和 S2 尚缺逐项原始日志/checkpoint 证据。
- 固定水平条带的姿态错位问题。
- C2 依赖 camid。
- 当前距离分析针对 BNNeck-after 检索空间，而 C2 在 BNNeck-before
  训练空间优化；两空间之间的机制传递仍需进一步分析。

## 第 5 章 实验复现与工程实现（可选）

### 5.1 配置和分支管理

- 一实验一配置、一输出目录。
- commit/config/log/checkpoint 绑定。
- 原始证据包、归档 SHA256、checkpoint SHA256 与分析结果分层保存。

### 5.2 AutoDL 运行流程

- 数据目录。
- 训练脚本。
- 自动解析 Rank-1/5/10 和 mAP。

### 5.3 机制分析工具

- 距离分布工具。
- batch 覆盖率工具。
- 合成数据单元测试。
- sample hash 与 pair hash 用于保证 Baseline/C2 比较使用相同样本和 pair。

如学校更强调算法，可将本章压缩为第 4 章复现小节和附录。

## 第 6 章 总结与展望

### 6.1 总结

- 总结 C2 方法、Market/Duke 原始日志核验结果和机制分析框架。
- 报告 C2-L03 在 Market 的 95.0% Rank-1/87.8% mAP，以及在 Duke
  的 88.4%/78.7%；同时说明这些是缺少重复 seed 的单次实验结果。
- 机制结论分数据集陈述：Market 未观察到跨摄像头同身份全局距离整体明显
  收缩；Duke matched epoch 120 观察到收缩且分离间隔扩大。
- 覆盖率结论只说明 C2 在当前 PK 采样中有较高的潜在生效机会。

### 6.2 局限性

- seed 未记录、缺少重复实验与置信区间。
- λ 其余档位及 CAAT/S2 原始产物尚未逐项归档核验。
- 训练 BNNeck-before 空间与检索 BNNeck-after 空间之间的机制差异。
- camid 依赖、固定局部划分和跨数据集机制一致性不足。

### 6.3 展望

- camera-balanced sampler。
- 姿态/人体解析局部对齐。
- 可学习辅助权重。
- 无 camid、伪 camid 和跨域 ReID。
- 更复杂数据集上的验证。
- 固定多 seed 重复实验，并增加训练空间/检索空间的对应距离分析。

## 附录

- 完整 YAML。
- commit/branch/config/seed 对照。
- 距离分析 metadata、checkpoint SHA256、sample/pair hash、完整分位数和图表。
- batch coverage CSV/JSON。
- 失败实验和负结果。
