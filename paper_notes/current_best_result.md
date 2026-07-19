# 当前最优结果：C2-L03

更新时间：2026-07-19。

## 1. 结论

C2-L03 是当前论文主方法，也是当前已归档实验中的 Market1501 最优配置。原先仅由用户确认的 `Rank-1=95.0%`、`mAP=87.8%`，现已在外部证据包中找到对应原始训练日志、配置和 epoch-120 checkpoint，因此证据状态升级为“原始日志/checkpoint 已核验”。固定 seed 和多次独立重复实验仍然缺失，不能据此声称统计显著或稳定提升。

| 字段 | 当前值 | 证据状态 |
|---|---|---|
| 方法 | C2 / Cross-Camera Positive Only | 代码可核验 |
| 项目配置 | `configs/softmax_triplet_cross_camera_positive_lambda03_autodl.yml` | 当前仓库可核验 |
| 训练 commit | `7b8195d4a02b536b27ab4d6ac80652091db7468f` | 原始运行记录已核验 |
| 数据集 | Market1501 | config、log 已核验 |
| Part Attention | 开启，K=6 | config 已核验 |
| C2 聚合模式 | `mean` | config、代码已核验 |
| C2 辅助损失权重 | `MODEL.CROSS_CAMERA_POSITIVE_LAMBDA=0.3` | config 已核验 |
| 普通 Triplet margin | `SOLVER.MARGIN=0.3` | config 已核验；与上项不是同一参数 |
| 测试特征 | BNNeck-after，L2 normalize | config、评估代码已核验 |
| re-ranking | 关闭 | config、log 已核验 |
| 结果 epoch | 120 | 原始 log、checkpoint 已核验 |
| Rank-1 / Rank-5 / Rank-10 | **95.0% / 98.5% / 99.1%** | 原始 log 已核验 |
| mAP | **87.8%** | 原始 log 已核验 |
| 运行时间 | 0:52:35 | 原始运行记录已核验 |
| GPU | RTX 4090 | 归档实验记录；原始 log 只记录单 GPU，未输出型号 |
| seed | 待核实 | 原始 config、log 未记录固定 seed |

## 2. 原始证据

证据根目录：

`D:\thesis_reid\thesis_evidence\distance_analysis\2026-07-18`

Market C2-L03 原始产物：

- 配置：`extracted\server_ltmxmapcaz\market_c2_l03\config.yml`
- 日志：`extracted\server_ltmxmapcaz\market_c2_l03\log.txt`
- checkpoint：`extracted\server_ltmxmapcaz\market_c2_l03\resnet50_checkpoint_22320.pt`
- checkpoint SHA256：`2008541aa5738c8bc3a440504d0bbb055b646f5d8e97a25e113f2d2e463d497b`

源压缩包位于 `D:\Downloads\market_c2_l03_evidence_server_ltmxmapcaz_20260718.tar.gz`，SHA256 为：

`61ff271800b7157351fffa5f9a54403caf5a87b65f1407292125d12711885624`

下载目录中的压缩包与证据根目录 `raw_archives` 中的副本哈希一致，故不再向项目目录重复复制大文件。

## 3. 与公平控制组的同 epoch 对比

两组均使用 BoT + Part Attention K=6，区别是是否启用 C2 及其有效辅助权重。下表使用同为 epoch 120 的 checkpoint：

| 方法 | C2 | Rank-1 | Rank-5 | Rank-10 | mAP |
|---|---:|---:|---:|---:|---:|
| Market Baseline-Control | 关闭 | 94.4% | 98.1% | 98.9% | 85.5% |
| **Market C2-L03** | 开启，λ=0.3 | **95.0%** | **98.5%** | **99.1%** | **87.8%** |
| 差值 | — | **+0.6 pp** | **+0.4 pp** | **+0.2 pp** | **+2.3 pp** |

控制组 checkpoint：

`extracted\server_78514386\market_baseline\resnet50_checkpoint_22320.pt`

SHA256：

`171aea42cb8df1464461887c78352205d61319341cd4ea93afc9dc5e9fc34edf`

## 4. 方法与特征空间边界

C2 辅助项在训练阶段接收的是 BNNeck-before 的全局与 Part Attention 融合特征，代码默认不对该特征显式执行 L2 normalization。距离分布分析则使用测试协议中的 BNNeck-after、L2-normalized 检索特征。论文中应将两者分别称为“训练损失特征空间”和“检索嵌入空间”，不能写成 C2 直接优化了 post-BN、L2-normalized 的距离。

完整 CAAT 和 same-camera positive only（S2）均只作为消融：

- 完整 CAAT：跨摄像头 hardest positive，并在辅助项中加入 hardest negative；
- C2：跨摄像头正样本 `mean` 聚合，不在新增辅助项中加入负样本；
- S2：同摄像头同身份正样本 `mean` 聚合并排除 self。

## 5. 可直接用于论文的谨慎表述

> 在当前单次已归档实验中，C2-L03 在 Market1501 上取得 Rank-1 95.0% 和 mAP 87.8%，相对同 epoch 的 Baseline-Control 分别提高 0.6 和 2.3 个百分点。该结果已有原始日志与 checkpoint 支持；由于运行未记录固定 seed，且尚未完成多 seed 重复实验，本文不将该差异解释为统计显著性证据。

不应写：

- “λ=0.3 是 Triplet margin”；
- “多次实验均稳定达到 95.0%/87.8%”；
- “结果具有统计显著性”；
- “Market 距离分布已经证明 C2 整体拉近跨摄像头同身份样本”。

## 6. 仍缺少的证据

1. 固定 seed 及至少 3 次独立重复实验的均值和标准差；
2. 原始运行环境快照，包括 CUDA、PyTorch、驱动和依赖版本；
3. 原始日志中可直接核验的 GPU 型号；
4. 与该次运行绑定的完整命令行和数据目录清单。
