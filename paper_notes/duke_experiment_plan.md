# DukeMTMC-reID 跨数据集验证记录

更新时间：2026-07-22。

## 1. 当前状态

Duke Baseline-Control 与 Duke C2-L03 已完成配置、运行、Git 结果登记和原始产物归档，无需重新创建配置或重复训练。

- 配置与运行入口提交：`3ce47246d67c1c43befd651ea44082216167478f`
- 结果登记提交：`6f49104f6c9413b91df515c8a934eaa42d7a2ff3`
- 当前结果证据：Git 已登记，且外部证据包中的原始 log、config 和 checkpoint 已核验
- 固定 seed：`not_recorded`
- re-ranking：两组均关闭

正式证据 ID 为 `EV-TRAIN-DUK-BASELINE` 与 `EV-TRAIN-DUK-C2-L03`；两行均为
单次运行 `n=1`、证据等级 E2。

Duke 必须在论文中单独成表，不与 Market 主结果拼成同一数据集表。当前结论固定为：
C2-L03 相对 Duke Baseline-Control 的 Rank-1 与 mAP 分别提高 **1.7** 和
**3.0** 个百分点，说明当前正向差值并非只出现在 Market1501；该表述不扩展为
“所有数据集均有效”。

证据根目录：

`D:\thesis_reid\thesis_evidence\distance_analysis\2026-07-18`

源压缩包：

`D:\Downloads\distance_evidence_server_78514386_20260718.tar.gz`

压缩包 SHA256：

`0add8d833c60c425d48bfbe8d3f1cf829943525a46f9f8087adea78339f45f5b`

下载目录中的压缩包与证据根目录 `raw_archives` 中的副本哈希相同，因此没有向项目目录重复复制。

## 2. 公平性核验

两份解析后的完整配置已逐项比较。除 C2 开关和独立 `OUTPUT_DIR` 外，其余训练与测试条件一致。

| 设置 | Duke Baseline-Control | Duke C2-L03 |
|---|---|---|
| Backbone | ResNet-50，last stride=1 | 相同 |
| ImageNet 预训练 | 相同预训练方式和权重设置 | 相同 |
| Part Attention | 开启，K=6 | 相同 |
| 输入尺寸 | 256×128 | 相同 |
| 数据增强 | 相同 | 相同 |
| sampler | RandomIdentitySampler，batch=64，K=4 | 相同 |
| optimizer / epoch | Adam / 120 | 相同 |
| 学习率策略 | 相同 warm-up、base LR 和 step | 相同 |
| 测试特征 | BNNeck-after，L2 normalize | 相同 |
| re-ranking | 关闭 | 关闭 |
| C2 开关 | 关闭 | 开启 |
| C2 聚合 | 不生效 | `mean` |
| C2 有效辅助权重 | 0 | 0.3 |
| `OUTPUT_DIR` | `duke_baseline_control` | `duke_c2_l03` |

说明：Baseline 配置中可能保留默认的 `CROSS_CAMERA_POSITIVE_LAMBDA=0.3` 字段，但因为 C2 开关关闭，该权重不参与损失，故有效辅助权重为 0。不能仅凭字段值把 Baseline 写成启用了 C2。

这里的 Baseline-Control 是“BoT + Part Attention K=6 + 普通 ID/Triplet，关闭 C2 和完整 CAAT”，不是未加入 Part Attention 的裸 BoT。

## 3. 论文主表结果

Git 登记和原始日志一致。主表沿用项目实验记录选定的报告点：

| 实验 | 报告 epoch | Rank-1 | Rank-5 | Rank-10 | mAP | 运行时间 | re-ranking |
|---|---:|---:|---:|---:|---:|---:|---|
| Duke Baseline-Control | 80 | 86.7% | 94.0% | 95.8% | 75.7% | 0:47:46 | no |
| **Duke C2-L03** | **120** | **88.4%** | **95.2%** | **96.8%** | **78.7%** | **0:53:21** | **no** |
| 差值 | — | **+1.7 pp** | **+1.2 pp** | **+1.0 pp** | **+3.0 pp** | — | — |

原始日志还记录了下列中间评估点：

| 方法 | epoch 40：R1/mAP | epoch 80：R1/mAP | epoch 120：R1/mAP |
|---|---:|---:|---:|
| Baseline-Control | 80.2% / 67.1% | 86.7% / 75.7% | 86.4% / 76.3% |
| C2-L03 | 84.2% / 73.0% | 88.0% / 78.3% | 88.4% / 78.7% |

可以用于论文的谨慎表述：

> 在训练与测试配置对齐的 DukeMTMC-reID 单次归档实验中，C2-L03 相对 Baseline-Control 的 Rank-1 和 mAP 分别提高 1.7 和 3.0 个百分点。该方向与 Market1501 主结果一致，说明当前正向差值并非只出现在 Market1501；由于两组均未记录固定 seed，也未完成多 seed 重复实验，本文不将该差异表述为统计显著或普遍有效。

## 4. 原始产物与哈希

Baseline：

- config：`extracted\server_78514386\duke_baseline\config.yml`
- log：`extracted\server_78514386\duke_baseline\log.txt`
- epoch-80 checkpoint：`extracted\server_78514386\duke_baseline\resnet50_checkpoint_19280.pt`
- epoch-80 checkpoint SHA256：`1cc8fb04b05e72d43020d42e69977c63f7e35a34cb79916048fdbb9d475dd180`
- epoch-120 checkpoint：`extracted\server_78514386\duke_baseline\resnet50_checkpoint_28920.pt`
- epoch-120 checkpoint SHA256：`ae0713f27dbc80684c4840d0d0138748dcb2f03228d9fe7de4cc9c6bf36f7a55`

C2-L03：

- config：`extracted\server_78514386\duke_c2_l03\config.yml`
- log：`extracted\server_78514386\duke_c2_l03\log.txt`
- epoch-120 checkpoint：`extracted\server_78514386\duke_c2_l03\resnet50_checkpoint_28920.pt`
- epoch-120 checkpoint SHA256：`d5ded542fd82bb96068106a7769cb9290c5939bde4862fb60e8ffbb89d5f2e0f`

四次距离分析中的 checkpoint 加载均未出现 missing 或 skipped 参数。

## 5. 与距离机制分析的 checkpoint 对齐

论文主表的 Baseline 报告点是 epoch 80，但当前公平距离分布分析使用两组同为 epoch 120 的 checkpoint，必须在图表和正文中明确区分：

| 方法 | 距离分析 checkpoint | 对应 Rank-1 | 对应 mAP |
|---|---:|---:|---:|
| Duke Baseline-Control | epoch 120 | 86.4% | 76.3% |
| Duke C2-L03 | epoch 120 | 88.4% | 78.7% |
| 同 epoch 差值 | — | +2.0 pp | +2.4 pp |

不能把 epoch-120 Baseline 的距离分布标成“Baseline 最佳指标 checkpoint”。一个名为 `duke_best_reported.run.log` 的文件只包含数据集加载阶段信息，没有形成完整分析输出，不能作为有效距离结果。

## 6. 仍缺少的证据

1. 两组运行的固定 seed；
2. 至少 3 个固定 seed 的独立重复实验及均值、标准差；
3. 原始运行环境快照，包括 CUDA、PyTorch、驱动和依赖版本；
4. 原始 log 中可直接核验的 GPU 型号。RTX 4090 目前来自归档实验记录，而非训练 log 的设备型号输出。

距离与覆盖率分析工具已由提交
`1d5f48ddd85a3e0bdb3396e86be22d9eeaebb9f9` 纳入版本管理；这不改变 Duke
训练所对应的 `3ce4724` 训练版本，也不能将后来工作区新增的 seed 42 回填到历史
Duke 运行。
