# BoT-ReID 研究时间线与分支演进

更新时间：2026-07-19。当前分支为
`exp/c2-l03-duke-validation`，HEAD 为
`6f49104f6c9413b91df515c8a934eaa42d7a2ff3`。

本文档依据当前仓库全部本地/远程分支、`git log --all`、历史配置与实验登记，
以及 2026-07-18 归档的原始训练日志、checkpoint 和距离分析结果整理。它记录的是
研究演进和证据强度，不把并行探索改写成一条连续单调提升的路线。

## 1. 证据口径与外部证据位置

### 1.1 证据等级

- **代码可核验**：当前工作树或指定 Git 提交中的实现、配置、调用链可直接检查。
- **Git 已登记**：指标或实验说明已经提交到某分支的 `EXPERIMENTS.md`，但不等同于
  已取得对应原始日志和 checkpoint。
- **原始日志/checkpoint 已核验**：已定位原始训练日志和模型权重，并核对日志指标、
  checkpoint epoch、文件哈希或分析元数据。
- **归档实验记录**：结果存在于外部证据包内归档的实验表，但该实验本身的原始日志或
  checkpoint 尚未逐项取得。其强度高于口头转述，低于原始产物核验。
- **用户确认**：由用户明确提供，但仍缺少可追溯原始产物。若后来找到原始产物，应提升
  证据等级，不再继续标为“仅用户确认”。
- **待核实/待实验**：缺少必要证据，或方案尚未实际运行。此类项目不能填写推测指标。

“Git 已登记”“归档实验记录”或单次“原始日志/checkpoint 已核验”均不能替代固定
seed 的多次独立重复实验，也不能据此声称统计显著性。

### 1.2 2026-07-18 外部证据包

原始下载位于 `D:\Downloads`，同内容副本已按证据类型整理到：

`D:\thesis_reid\thesis_evidence\distance_analysis\2026-07-18`

| 归档文件 | SHA256 | 状态 |
|---|---|---|
| `distance_evidence_server_78514386_20260718.tar.gz` | `0add8d833c60c425d48bfbe8d3f1cf829943525a46f9f8087adea78339f45f5b` | 下载目录与证据目录副本一致，已解压 |
| `market_c2_l03_evidence_server_ltmxmapcaz_20260718.tar.gz` | `61ff271800b7157351fffa5f9a54403caf5a87b65f1407292125d12711885624` | 下载目录与证据目录副本一致，已解压 |

证据目录中的 `extracted/` 保存原始日志、配置、checkpoint 和服务器端实验登记，
`analysis_results/` 保存距离分析元数据、统计表、样本/样本对索引和图片。为避免在
仓库内重复保存约 1.25 GiB 原始数据，论文材料只记录其路径和哈希。

## 2. Git 分支与提交拓扑

### 2.1 当前可见分支

| 路线 | 本地端点 | 远程端点 | 说明 |
|---|---|---|---|
| BoT 本地主线 | `main @ a626838` | `origin/main @ 23849fc` | 本地主线停在数据路径配置；远程已含 Part Attention |
| Part Attention K=6 | `part-attention-k6 @ 23849fc` | 与 `origin/main` 同提交 | 轻量局部特征骨架 |
| AutoDL 主线 | `main-autodl @ 2b06f88` | `origin/main-autodl @ 2b06f88` | 训练入口与实验记录 |
| Part Attention τ | `part-attention-tau-sensitivity @ c4e766b` | 同名远程 `c4e766b` | 注意力温度敏感性 |
| Adaptive Hard Triplet | `exp/adaptive-hard-triplet-loss @ 700e4b8` | 同名远程 `700e4b8` | 并行困难样本探索 |
| Normalized Weighted Loss | `exp/normalized-weighted-loss @ 3d5c71c` | 同名远程 `3d5c71c` | Adaptive Hard 的归一化变体 |
| BNNeck camera debias | `exp-bnneck-camera-debias @ d998f4b` | `origin/exp-bnneck-camera-debias @ a98e3d6` | 本地比远程少两个结果提交 |
| 完整 CAAT | `exp/camera-aware-triplet-loss @ 610a9ca` | 同名远程 `610a9ca` | `CameraAwareTripletLoss` |
| Hierarchical Camera-Aware | `exp/hierarchical-camera-aware-loss @ ddce037` | 同名远程 `ddce037` | CAAT 的并行层级化探索 |
| 完整 CAAT λ | `exp/camera-aware-triplet-lambda-sensitivity @ 61da0df` | 同名远程 `73a44c0` | 本地未含结果登记提交 |
| C2 对照/重复 | `exp/cross-camera-positive-only @ d98fb00` | 同名远程 `9b81850` | 本地未含结果登记提交 |
| S2 消融 | `exp/same-camera-positive-only @ f1f1692` | 同名远程 `a4a42b3` | 本地未含结果登记提交 |
| C2 λ 敏感性 | `exp/cross-camera-positive-lambda-sensitivity @ 7b8195d` | 同名远程 `7b8195d` | C2-L03 的训练代码/配置版本 |
| Duke 验证 | `exp/c2-l03-duke-validation @ 6f49104` | 同名远程 `6f49104` | 当前分支和当前 HEAD |

`origin/HEAD` 指向 `origin/main`。`refs/stash` 下的 `7a47abe`、`3ccaeb7` 和
`08828ee` 是 2026-07-13 的暂存引用，不作为独立实验分支或论文方法阶段。

### 2.2 主要拓扑

```text
0722e55  BoT baseline 迁移
└─ a626838  Market1501 本地启动
   ├─ 23849fc  Part Attention K=6（origin/main）
   ├─ 2e7d330 → b1cac4e → 2b06f88  Part Attention + AutoDL
   │  ├─ 8ed6b5f → 8e0d1ea → c4e766b  Part Attention τ 敏感性
   │  ├─ 8e3ed83 → d998f4b → 5e59d5e → a98e3d6  camera mean debias
   │  └─ a3cd05d → 6557988 → 610a9ca  Camera-Aware Triplet
   │     ├─ ddce037  Hierarchical Camera-Aware
   │     └─ 0eeb467  C2 / Cross-Camera Positive Only
   │        └─ 76fc7aa  C2 对照与复现实验结构
   │           ├─ d98fb00 → 9b81850  C2 baseline/λ=.5/repeat
   │           ├─ f7131ab → f1f1692 → a4a42b3  S2 消融
   │           └─ ce10042 → 7b8195d  C2 λ 敏感性
   │              └─ 3ce4724 → 6f49104  Duke C2-L03 验证
   └─ 700e4b8  Adaptive Hard Triplet
      └─ 39bd62d → 3d5c71c  Normalized Weighted Loss
```

完整 CAAT 的 λ 敏感性从 `610a9ca` 分出，路线为
`61da0df → 73a44c0`。`23849fc` 与 `2e7d330` 是同一基础上的平行
Part Attention 提交，不是两次连续算法改进。

## 3. 研究时间线

### 2026-06-24：建立 BoT 强基线

- `0722e55`：迁入 BoT/Strong Baseline 工程。
- 基础能力包括 ResNet/SENet/IBN、BNNeck、ID Loss、Triplet、Center Loss、
  Random Erasing、warm-up、CMC/mAP 和 re-ranking。
- `a626838`：配置 Market1501 本地路径并验证训练入口。
- 证据状态：**代码可核验、Git 已登记**。

证据边界：README 中的公开成绩属于原始 BoT 项目，不是本项目重新运行得到的成绩。

### 2026-06-24：引入 Part Attention K=6

- `23849fc` / `2e7d330`：沿特征图高度划分 6 个水平区域，分别池化并学习
  softmax 权重。
- 加权局部特征与全局 GAP 特征相加，保持 2048 维输出。
- 后续 Baseline-Control 与 C2 均开启该模块，因此它是公平对照的统一特征骨架，
  不是只加在 C2 上的额外优势。
- 证据状态：**代码可核验、Git 已登记**。

### 2026-06-25：Adaptive Hard Triplet 并行探索

- `700e4b8`：依据 `gap = dist_an - dist_ap` 和温度函数对 batch-hard 样本加权。
- `39bd62d → 3d5c71c`：将加权均值改为按权重和归一化。
- 该路线直接从 baseline 分出，不属于当前 C2 主线。
- 证据状态：**代码可核验、Git 已登记**；当前论文不把它列为最终方法。

### 2026-07-01：建立 AutoDL 与实验追踪流程

- `b1cac4e`、`2b06f88`：加入 AutoDL 配置、运行脚本、`EXPERIMENTS.md`
  和输出忽略规则。
- 开始以 commit、config、output 和 log 关联实验。
- 证据状态：**代码可核验、Git 已登记**。

### 2026-07-03：注意力与困难样本敏感性

- `8ed6b5f`：准备 Part Attention 的 τ=0.1/0.2/0.5 方案。
- `8e0d1ea`：修正 Part Attention 参数传递。
- `c4e766b`：增加实验日志解析。
- 这些提交与 Adaptive Hard 路线均为并行探索，不能写成 C2 的连续前置增益。
- 证据状态：**代码可核验、Git 已登记**；原始实验产物需按具体实验另行核实。

### 2026-07-06—11：BNNeck 相机均值去偏探索

- `8e3ed83 → d998f4b → 5e59d5e → a98e3d6`。
- 测试阶段减去每个 camera 的均值特征后重新归一化。
- 这是推理后处理，不属于 C2 训练方法；当前 C2 主结果和本次距离分析均关闭
  camera-mean debias。
- 如论文讨论此路线，需单独说明利用测试集 camera 统计量带来的协议问题。
- 证据状态：**代码可核验、Git 已登记**；不纳入当前主方法。

### 2026-07-09：提出完整 Camera-Aware Triplet（CAAT）

- `610a9ca`：形成 `CameraAwareTripletLoss`。
- 对每个有效 anchor 选择跨摄像头同 ID hardest positive，同时加入异 ID
  hardest negative；使用独立的辅助 margin 与辅助权重。
- 基础 ID Loss 与普通 Triplet 仍保留。
- 当前定位：用于回答“辅助 hardest negative 是否必要”的消融方法，不作为最终主方法。
- 证据状态：**代码可核验、Git 已登记**。

### 2026-07-09：提出 C2 / Cross-Camera Positive Only

- `0eeb467`：形成 `CrossCameraPositiveLoss` 和
  `MODEL.CROSS_CAMERA_POSITIVE_ONLY=True` 的方法开关。
- 新增辅助项只约束相同 ID、不同 camera 的正样本，不在该辅助项中加入负样本；
  原有 ID Loss 和普通 Triplet 继续保留。
- 支持 `mean` 与 `hard` 聚合，当前主方法 C2-L03 使用 `mean`。
- 证据状态：**代码可核验、Git 已登记**。

### 2026-07-11：完整 CAAT 的 λ 消融

- 配置提交：`61da0df`；结果登记提交：`73a44c0`。
- Git 登记的 Market1501 结果如下：

| 配置 | 辅助权重 λ | Rank-1 | mAP | 证据 |
|---|---:|---:|---:|---|
| CAAT-L01 | 0.1 | 94.2% | 85.5% | Git 已登记 |
| CAAT-L03 | 0.3 | 94.2% | 85.5% | Git 已登记 |
| CAAT-L05 | 0.5 | 94.2% | 85.4% | Git 已登记 |
| CAAT-L10 | 1.0 | 94.1% | 84.8% | Git 已登记 |

原始日志、checkpoint、固定 seed 和重复运行尚未归档。因此该表只能作为完整
CAAT 的历史消融登记，不能写成统计显著结论。

### 2026-07-13—18：C2 公平对照、重复实验和 S2 消融

- `76fc7aa`：建立 C2 公平控制、重复实验和可复现记录结构。
- `9b81850`：Git 登记 Baseline-Control、C2 λ=0.5 和重复运行结果。
- `a4a42b3`：Git 登记 Same-Camera Positive Only（S2）结果。

| 实验 | 约束 | 辅助权重 λ | Rank-1 | mAP | 证据 |
|---|---|---:|---:|---:|---|
| C2-Baseline-Control | 关闭 C2，保留 Part Attention | 0 | 94.4% | 85.5% | Git 已登记；其 epoch120 原始日志/checkpoint 后续已核验 |
| C2-CCPO-Market | 跨 camera 同 ID positive only | 0.5 | 95.0% | 87.7% | Git 已登记 |
| C2-CCPO-Repeat | 同上，重复运行 | 0.5 | 95.0% | 87.3% | Git 已登记 |
| S2-SCPO-Market | 同 camera 同 ID positive only | 0.5 | 94.4% | 86.8% | Git 已登记 |

C2 λ=0.5 的两次登记结果提供了历史重复线索，但固定 seed、两次运行的完整原始日志
和 checkpoint 尚未归档；不能据此声称统计显著。S2 只作为检验“跨 camera 条件是否
必要”的消融，不是最终方法。

### 2026-07-13—18：C2 λ 敏感性与 C2-L03 确立

- `ce10042 → 7b8195d`：准备并修正 C2 λ 敏感性配置和实验记录。
- 外部证据包中的归档实验表记录了同一序列：

| 配置 | 辅助权重 λ | Rank-1 | mAP | 运行时间 | 证据 |
|---|---:|---:|---:|---:|---|
| C2-L01 | 0.1 | 94.7% | 87.5% | 00:52:25 | 归档实验记录；原始产物待核实 |
| **C2-L03** | **0.3** | **95.0%** | **87.8%** | **00:52:35** | **原始日志/checkpoint 已核验** |
| C2-L05 | 0.5 | 94.8% | 87.5% | 00:51:46 | 归档实验记录；原始产物待核实 |
| C2-L10 | 1.0 | 94.6% | 87.0% | 00:51:47 | 归档实验记录；原始产物待核实 |

C2-L03 的 Market1501 原始证据已核验：

- 方法：C2，`mode=mean`，Part Attention 开启且 `K=6`。
- 配置：`configs/softmax_triplet_cross_camera_positive_lambda03_autodl.yml`。
- 训练版本：
  `7b8195d4a02b536b27ab4d6ac80652091db7468f`。
- epoch 120：Rank-1 **95.0%**、Rank-5 **98.5%**、Rank-10 **99.1%**、
  mAP **87.8%**。
- 原始日志：
  `extracted/server_ltmxmapcaz/market_c2_l03/log.txt`。
- epoch120 checkpoint：
  `resnet50_checkpoint_22320.pt`，SHA256
  `2008541aa5738c8bc3a440504d0bbb055b646f5d8e97a25e113f2d2e463d497b`。
- 运行时间：00:52:35。
- seed：日志和归档记录均未记录，仍为**待核实**。
- GPU：归档实验表登记为 RTX 4090；原始训练日志本身只记录使用 1 张 GPU，
  因而硬件型号属于**归档实验记录**，不是日志内直接核验字段。

Market Baseline-Control 的 epoch120 原始日志/checkpoint 也已核验：Rank-1 94.4%、
Rank-5 98.1%、Rank-10 98.9%、mAP 85.5%，checkpoint SHA256 为
`171aea42cb8df1464461887c78352205d61319341cd4ea93afc9dc5e9fc34edf`。

因此，95.0% Rank-1 / 87.8% mAP 已从早期“用户确认”提升为
**原始日志/checkpoint 已核验**，并作为当前 Market1501 主结果。仍因 seed 缺失且
没有固定 seed 的多次独立重复，不能写“稳定提升”或“统计显著”。

必须区分两个数值均为 0.3 的参数：

- `MODEL.CROSS_CAMERA_POSITIVE_LAMBDA=0.3`：C2 辅助损失权重 \(\lambda\)；
- `SOLVER.MARGIN=0.3`：普通 Triplet margin。

### 2026-07-18：DukeMTMC-reID 跨数据集验证

- `3ce4724`：加入 Duke Baseline-Control 和 Duke C2-L03 配置/入口。
- `6f49104`：登记 Duke 结果。
- 两组使用相同 backbone、Part Attention K=6、输入尺寸、增强、sampler、
  batch 组成、优化器、epoch、学习率策略和测试协议；re-ranking 均关闭。除 C2
  开关、有效辅助权重和独立输出目录外，其余关键条件一致。
- 训练版本：`3ce47246d67c1c43befd651ea44082216167478f`。

原始训练日志和 checkpoint 已从证据包核验，不再仅是 Git 登记：

| 实验 | 选取 epoch | Rank-1 | Rank-5 | Rank-10 | mAP | 运行时间 | 证据 |
|---|---:|---:|---:|---:|---:|---:|---|
| Duke-Baseline-Control | 80 | 86.7% | 94.0% | 95.8% | 75.7% | 00:47:46 | 原始日志/checkpoint 已核验 |
| Duke-C2-L03 | 120 | 88.4% | 95.2% | 96.8% | 78.7% | 00:53:21 | 原始日志/checkpoint 已核验 |

按各自登记/报告的选取 epoch，C2-L03 相对控制组为 Rank-1 **+1.7**、mAP
**+3.0** 个百分点。在当前单次实验中，正向差值不只出现在 Market1501；但 seed
未记录且没有多 seed 重复，不能据此声称统计显著或普遍有效。

用于后续距离机制分析的是严格对齐的 epoch120 checkpoint，而不是把 baseline 的
epoch80 与 C2 的 epoch120 混合比较：

| 实验 | epoch120 Rank-1 | epoch120 mAP | checkpoint SHA256 |
|---|---:|---:|---|
| Duke-Baseline-Control | 86.4% | 76.3% | `ae0713f27dbc80684c4840d0d0138748dcb2f03228d9fe7de4cc9c6bf36f7a55` |
| Duke-C2-L03 | 88.4% | 78.7% | `d5ded542fd82bb96068106a7769cb9290c5939bde4862fb60e8ffbb89d5f2e0f` |

### 2026-07-18—19：距离分布机制分析

完整结果位于：

- Market：
  `analysis_results/market_epoch120_control`
- Duke：
  `analysis_results/duke_epoch120_control`

两组比较均使用 query+gallery、BNNeck-after 检索特征、显式 L2 normalization、
squared Euclidean distance；关闭 re-ranking 和 camera-mean debias。排除 self-pair，
只统计 `i<j` 的无向样本对，三类 pair 互斥。different-ID 固定 seed=42、
无放回抽样 200,000 对。

| 数据集 | sample hash | pair hash | Baseline/C2 协议 |
|---|---|---|---|
| Market1501 | `ede26d3a28aece193741f618d26bd5b3ceecacce8ed359589322030a01d14461` | `193dd9ccfd552c6fcf7c402e2d31ad469ee51633cafc30142e2f3a835b522084` | 一致 |
| DukeMTMC-reID | `be498c3a413de371c27b610728b7a4c2465307e6c6e74ee5184b4e92df2a7f2e` | `1e0ffcd68de323d8c0f56d69737a673d0cdda11db1c0b4c423ccae59e96a234b` | 一致 |

关键结果：

| 数据集 | same-ID different-camera 均值变化 | 中位数变化 | different-ID 均值变化 | 均值间隔变化 | 结论 |
|---|---:|---:|---:|---:|---|
| Market1501 | 1.773591 → 1.772773（−0.000818，−0.046%） | 1.825205 → 1.832293（+0.007088） | 1.988320 → 1.989686（+0.001366） | 0.214728 → 0.216913（+0.002185） | 不支持“整体明显拉近”；均值微降，但中位数及主要高分位上升、分布变宽 |
| DukeMTMC-reID | 1.167991 → 1.081971（−0.086020，−7.365%） | 1.132603 → 1.039084（−0.093520，−8.257%） | 1.975909 → 1.965717（−0.010192） | 0.807918 → 0.883746（+0.075828） | 当前 epoch120 checkpoint 和协议下支持跨摄像头同 ID 距离下降，且净间隔扩大 |

Market 的 cross-camera 均值仅有极小下降，而 median、q25、q75、q95 均上升，
标准差也增大；直方图和箱线图整体高度重叠。因此不能选择性引用均值并写成
“C2 在 Market 上整体拉近了跨摄像头同身份特征”。Market 的检索指标提升与
全体无向 pair 的全局距离分布并不构成同一个结论。

Duke 的 cross-camera 均值、中位数以及 q05/q25/q75/q95 均下降；different-ID
均值也下降约 0.516%，但幅度远小于 cross-camera 同 ID 的约 7.365%，均值间隔扩大
约 9.386%，中位数间隔扩大约 10.279%。因此可以限定为：

> 在当前 Duke epoch120 checkpoint 和既定分析协议下，C2-L03 缩小了跨摄像头
> 同身份样本的检索嵌入距离，同时描述统计上的类间净间隔扩大。

这仍是单组 checkpoint 的描述统计，不是独立样本显著性检验。`duke_best_reported`
目录对应的额外分析没有形成完整结果，论文只采用已完成的
`duke_epoch120_control`，避免混入未完成产物。

### 2026-07-18：batch 内有效 cross-camera positive 覆盖率

离线模拟 `RandomIdentitySampler`，固定 seed=42、10 个 epoch。有效 anchor 定义为：
batch 内至少存在一个相同 pid、不同 camid 且不是自身的样本。

| 数据集 | 总 anchor | 有效 anchor | 加权有效比例 | 每 batch 均值 ± 总体标准差 | min/median/max | 零有效 batch |
|---|---:|---:|---:|---:|---:|---:|
| Market1501 | 117,376 | 114,852 | 97.8496% | 97.8496% ± 3.6160% | 81.25% / 100% / 100% | 0/1,834 |
| DukeMTMC-reID | 144,576 | 135,980 | 94.0543% | 94.0543% ± 7.0234% | 43.75% / 93.75% / 100% | 0/2,259 |

两组每 batch 均固定 16 个 pid。Market 每 batch 唯一 camera 数均值为 5.9771
（范围 5—6），Duke 为 7.8601（范围 5—8）。有序 cross-camera positive pair
比例分别为 77.2196% 和 67.0614%。

证据边界：这是固定 sampler 规则下的**监督机会覆盖率**，不是历史训练 batch 的
精确回放，也不是模型准确率或性能因果证明。它只能说明 C2 辅助项在多少 anchor
上有机会生效，机制解释必须与距离分析和 Rank-1/mAP 共同讨论。

## 4. 当前研究主线

1. 以 BoT 为基础，使用 Part Attention K=6 构建主方法和控制组共同的特征骨架。
2. 指出普通 Triplet 不显式区分同摄像头与跨摄像头正样本这一问题。
3. 提出 C2，仅在新增辅助项中优化跨摄像头同身份一致性，不重复加入辅助负样本。
4. 通过归档 λ 序列选择 C2-L03（辅助权重 \(\lambda=0.3\)）作为当前配置；
   其中 L03 已有原始日志/checkpoint，其余 λ 点仍以归档实验记录为证。
5. 将完整 CAAT 用作“额外 hardest negative 是否必要”的消融，将 S2 用作
   “跨 camera 条件是否必要”的消融；二者都不作为最终方法。
6. 在 Market1501 报告当前主结果 95.0% Rank-1 / 87.8% mAP，并用 Duke 的
   88.4% / 78.7% 进行跨数据集验证。
7. 用 batch 覆盖率说明 C2 监督机会较高，但明确它不能证明模型收益。
8. 用距离分布检验机制：Duke 的 matched-epoch 结果支持预期，Market 的全局
   pair 分布不支持“整体明显拉近”。论文应保留这一跨数据集差异，而不是为迎合
   预期选择性报告。

## 5. 当前仍缺少的关键证据

- Market C2-L03 和 Duke 两组训练均未记录可核验 seed，尚无固定 seed 的多次独立重复。
- C2-L01、L05、L10 当前只有归档实验表，仍缺逐项原始日志和 checkpoint。
- 完整 CAAT λ 系列、S2 及 C2 λ=0.5 重复实验仍缺完整原始产物归档。
- RTX 4090 型号来自归档实验记录，原始日志内没有直接的设备型号快照。
- 当前距离结论基于各数据集一组 Baseline/C2 epoch120 checkpoint，未覆盖多 seed
  和训练阶段变化；Market 与 Duke 的机制表现不同，不能泛化为普遍规律。
- pair 数量很大但样本对并非相互独立；当前只报告描述统计，不据此宣称统计显著。
