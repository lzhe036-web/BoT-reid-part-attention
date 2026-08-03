# Cross-camera positive 有效 anchor 覆盖率

更新时间：2026-07-22。

## 1. 定义

对 batch 内 anchor \(i\)，若至少存在一个样本 \(j\) 满足

\[
j\ne i,\qquad y_j=y_i,\qquad c_j\ne c_i,
\]

则 \(i\) 是有效 cross-camera-positive anchor。

训练代码中的 `cross_camera_positive_count` 表示有效 anchor 数，不是正样本 pair 数。论文主口径使用：

\[
r_{\mathrm{anchor}}
=
\frac{\sum_b n^{\mathrm{valid}}_b}
{\sum_b n^{\mathrm{anchor}}_b}.
\]

另外单独报告有序 cross-camera positive pair 占全部同身份有序 positive pair 的比例，但二者不能混用。

## 2. 统计协议

- sampler：RandomIdentitySampler；
- batch size：64；
- 每个身份样本数 K：4，即每 batch 16 个身份；
- 随机 seed：42；
- 连续离线模拟：10 个 epoch；
- Market1501 训练图像：12,936 张；
- DukeMTMC-reID 训练图像：16,522 张；
- 不读取图像、模型或 checkpoint；
- Baseline-Control 与 C2-L03 的 sampler 参数一致，因此同一数据集只需统计一次。

这是对当前采样协议的离线模拟，不是历史训练 batch 的精确回放。它测量“C2 在多少 anchor 上有监督机会”，不是模型准确率，也不直接证明 C2 能提升检索性能。

## 3. Anchor 覆盖率

| 数据集 | batch 数 | 总 anchors | 有效 anchors | 无效 anchors | 加权有效比例 |
|---|---:|---:|---:|---:|---:|
| Market1501 | 1,834 | 117,376 | 114,852 | 2,524 | **97.8496%** |
| DukeMTMC-reID | 2,259 | 144,576 | 135,980 | 8,596 | **94.0543%** |

| 数据集 | batch 比例均值 | 总体 std | min | median | max | 零有效 batch |
|---|---:|---:|---:|---:|---:|---:|
| Market1501 | 97.8496% | 3.6160% | 81.25% | 100% | 100% | 0/1,834（0%） |
| DukeMTMC-reID | 94.0543% | 7.0234% | 43.75% | 93.75% | 100% | 0/2,259（0%） |

由于所有 batch 都是 64 个 anchor，加权有效比例与 batch 比例算术均值相同。这一相等由固定 batch size 导致，不应推广到可变 batch size 情形。

| 证据 ID | 数据集 | n | 训练 seed | 分析 seed | 结论等级 |
|---|---|---:|---|---:|---|
| EV-ANALYSIS-MKT-COVERAGE | Market1501 | 1 | not_recorded | 42 | E2 |
| EV-ANALYSIS-DUK-COVERAGE | DukeMTMC-reID | 1 | not_recorded | 42 | E2 |

## 4. Batch 组成

| 数据集 | 唯一 pid / batch | 唯一 camid 均值 | camid 总体 std | camid min | camid median | camid max |
|---|---:|---:|---:|---:|---:|---:|
| Market1501 | 16（固定） | 5.9771 | 0.1496 | 5 | 6 | 6 |
| DukeMTMC-reID | 16（固定） | 7.8601 | 0.3946 | 5 | 8 | 8 |

逐 batch CSV 已使用 `pid_camera_counts` 重新核验：

- 每行有效 anchor 数和比例计算无误；
- 所有比例均在 `[0,1]`；
- 每个 epoch 汇总与总 JSON 一致；
- 两个数据集均未发现零有效 anchor 的 batch。

## 5. Pair 级旁证

下表使用有序 pair，排除 self：

| 数据集 | cross-camera 同身份有序 pair | 全部同身份有序 positive pair | 比例 |
|---|---:|---:|---:|
| Market1501 | 271,912 | 352,128 | 77.2196% |
| DukeMTMC-reID | 290,864 | 433,728 | 67.0614% |

Pair 比例小于 anchor 覆盖率并不矛盾：一个 anchor 只要拥有至少一个跨摄像头同身份正样本就被计为有效，而 pair 比例衡量所有同身份正样本关系中有多少跨摄像头。

## 6. 论文解释

可使用：

> 在固定 64/4 PK 采样协议的 10 个离线模拟 epoch 中，Market1501 和 DukeMTMC-reID 分别有 97.85% 和 94.05% 的 anchor 至少拥有一个同身份、不同摄像头正样本，且两数据集均未出现完全缺少有效 anchor 的 batch。这说明当前 sampler 为 C2 提供了较高比例的潜在监督机会。

必须同时说明：

- 这是采样监督机会覆盖率，不是模型准确率；
- 它不是历史训练 batch 的精确回放；
- 高覆盖率说明损失有机会生效，不等于损失一定产生了正确梯度或性能收益；
- 最终机制解释应同时结合 Rank-1/mAP 和检索距离分布；
- 当前 Market person-only v2 与 Duke 距离结果均观察到跨摄像头同身份距离下降，
  但都只是一组 checkpoint 下的描述统计，不能据此推广为普遍规律。

## 7. 机器可读证据

- Market1501：`paper_notes/analysis_results/batch_coverage_market1501/`
- DukeMTMC-reID：`paper_notes/analysis_results/batch_coverage_dukemtmc/`

各目录包含逐 batch CSV、逐 epoch CSV、JSON 和 Markdown 摘要。现有文件均保留原样，本次仅更新论文说明。

论文机器可读正式表为
`paper_notes/c2_l03_final_evidence/anchor_coverage_formal.csv`；上述目录中的明细
CSV/JSON 是其底层审计依据。

分析工具 `tools/analyze_cross_camera_batch_coverage.py` 已由提交
`1d5f48ddd85a3e0bdb3396e86be22d9eeaebb9f9` 纳入版本管理。落盘
`summary.json` 没有保存原始 shell 命令全文，因此论文材料中的复现命令是依据
config、dataset、data root、seed、epoch、batch size 和 K 等元数据重建的命令，
不是历史命令的逐字回放。

## 8. 证据边界

已经核验：

- JSON、逐 epoch CSV 与逐 batch CSV 数值一致；
- 有效 anchor 定义与训练代码一致；
- 比例范围、batch 组成和 pair 统计无异常。

仍可补充：

1. 固定训练 seed 后，将真实训练过程中的 batch 覆盖率同步写入日志；
2. 在不同 K、batch size 或 camera-balanced sampler 下比较覆盖率；
3. 统计有效 anchor 的正样本数量分布，而不仅是是否至少存在一个；
4. 联合梯度大小或实际 C2 loss 值分析“有监督机会”与“产生有效优化”的差异。

当前工作区后来新增的训练 seed 和确定性 sampler 修改没有参与这份历史离线统计，
也不能用于回填既有训练运行的 seed。这里的 seed=42 始终只表示分析模拟 seed。
