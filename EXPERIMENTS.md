# Experiments

> **用途边界（2026-07-22）**：本文件保留历史运行登记格式，不再作为论文表格或当前证据状态的真源。论文统一表见 `paper_notes/c2_l03_final_evidence/UNIFIED_TABLES.md`，机器表见同目录 `performance_results.csv`、`distance_distribution_formal.csv`、`anchor_coverage_formal.csv`，E0—E4 规则见 `TABLE_SCHEMA.md`，详细等级见 `evidence_registry.tsv`。下文历史行中的“待核实/用户确认/not_found”等旧措辞不得覆盖正式台账。

> 证据口径：表中“Git 已登记”表示指标存在于指定结果提交的
> `EXPERIMENTS.md`，但原始 AutoDL 日志/checkpoint 当前不在本仓库；
> “用户确认”表示数值由用户提供，相关运行 commit、日志、checkpoint、
> seed、best epoch 和运行时间仍待归档。不得把这两类证据写成重复实验的
> 统计显著性结论。

## Current Main Result

| 实验 | 数据集 | 方法 | auxiliary λ | Triplet margin | Rank-1 | mAP | re-ranking | 证据 |
|---|---|---|---:|---:|---:|---:|---|---|
| C2-L03 | Market1501 | Cross-camera positive only, mean, Part Attention K=6 | 0.3 | 0.3 | **95.0%** | **87.8%** | no | E2：config、原始日志、epoch-120 checkpoint 与 SHA256 已核验；training seed=not_recorded，n=1 |

这里的 auxiliary λ 是 `MODEL.CROSS_CAMERA_POSITIVE_LAMBDA`；Triplet
margin 是 `SOLVER.MARGIN`。两者数值恰好都为 0.3，但定义不同。

## Legacy Initial Placeholders

以下 E001/E002 是早期建立实验管理时留下的占位，不作为当前主结果证据。

| 实验编号 | 日期 | commit id | 分支 | config 文件 | seed | GPU | 数据集 | 运行时间 | best epoch | Rank-1 | mAP | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E001 | 待填写 | 待填写 | main-autodl | configs/softmax_triplet_part_attention_autodl.yml | 待填写 | 待填写 | Market1501 | 待填写 | 待填写 | 待填写 | 待填写 | BoT + Part Attention, K=6, AutoDL |

## 字段填写说明

1. commit id：运行前执行 `git rev-parse --short HEAD` 获取。
2. 分支：运行前执行 `git branch --show-current` 获取。
3. config 文件：本次训练使用的 yml 文件。
4. seed：如果当前代码没有显式设置 seed，就填写“未固定”。
5. GPU：运行 `nvidia-smi` 查看，例如 RTX 4090。
6. 运行时间：记录训练开始和结束时间，或使用 `time` 命令。
7. best epoch：从训练日志中查看最佳 Rank-1/mAP 对应 epoch。
8. Rank-1、Rank-5、Rank-10 和 mAP：从同一个 best epoch 的评估块填写；
   历史日志缺失时保持“待核实”。
9. re-ranking：从实际测试配置填写；主表中的 `no` 表示关闭重排。
10. 备注：记录是否使用 Part Attention、K 值、是否修改 batch size 等。

## Camera-Aware Triplet Loss Experiments

| 实验编号 | 日期 | commit id | 分支 | 实验类型 | 数据集 | config 文件 | OUTPUT_DIR | 日志路径 | GPU | seed | lambda | 运行时间 | best epoch | Rank-1 | Rank-5 | Rank-10 | mAP | re-ranking | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CAT001 | 2026-07-09 | 610a9ca | exp/camera-aware-triplet-loss | 完整 CAAT / Camera-Aware Triplet | Market1501 | configs/softmax_triplet_camera_aware_autodl.yml | /root/autodl-tmp/experiments/BoT/camera_aware_triplet_market1501 | /root/autodl-tmp/experiments/BoT/camera_aware_triplet_market1501/log.txt | NVIDIA GeForce RTX 4080（工作树记录） | not_recorded | 0.5 | not_recorded（日志跨度约 01:09:42） | 120 | 94.3% | 98.2% | 98.9% | 85.5% | no | 辅助 margin=0.3；原始 log 和三轮 checkpoint 已归档；结果仅存在于未提交工作树记录，无独立 result commit；不得替换受控系列 CAAT-L05。 |

## Cross-Camera Positive Only Experiments

| 实验编号 | 日期 | commit id | 分支 | 实验类型 | 数据集 | config 文件 | OUTPUT_DIR | 日志路径 | GPU | seed | lambda | 运行时间 | best epoch | Rank-1 | Rank-5 | Rank-10 | mAP | re-ranking | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2-CCPO-Market | 2026-07-13 | d98fb00 | exp/cross-camera-positive-only | Cross-camera positive only | Market1501 | configs/softmax_triplet_cross_camera_positive_only_autodl.yml | /root/autodl-tmp/experiments/BoT/cross_camera_positive_only_market1501 | /root/autodl-tmp/experiments/BoT/cross_camera_positive_only_market1501/log.txt | NVIDIA GeForce RTX 4090 | 待核实 | 0.5 | 0:44:30 | 120 | 95.0% | 待核实 | 待核实 | 87.7% | no | Git 已登记于结果提交 9b81850；原始日志/checkpoint 当前未归档。 |
| C2-CCPO-Repeat | 2026-07-13 | d98fb00 | exp/cross-camera-positive-only | Cross-camera positive only | Market1501 | configs/softmax_triplet_cross_camera_positive_only_repeat_autodl.yml | /root/autodl-tmp/experiments/BoT/cross_camera_positive_only_repeat_market1501 | /root/autodl-tmp/experiments/BoT/cross_camera_positive_only_repeat_market1501/log.txt | NVIDIA GeForce RTX 4090 | 待核实 | 0.5 | 0:44:06 | 80 | 95.0% | 待核实 | 待核实 | 87.3% | no | Git 已登记于结果提交 9b81850；无固定 seed，不能作为统计显著性证据。 |

## C2 Baseline-Control Experiments

| 实验编号 | 日期 | commit id | 分支 | 实验类型 | 数据集 | config 文件 | OUTPUT_DIR | 日志路径 | GPU | seed | lambda | 运行时间 | best epoch | Rank-1 | Rank-5 | Rank-10 | mAP | re-ranking | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2-Baseline-Control | 2026-07-13 | d98fb00 | exp/cross-camera-positive-only | Baseline control（BoT + Part Attention） | Market1501 | configs/softmax_triplet_c2_baseline_control_autodl.yml | /root/autodl-tmp/experiments/BoT/c2_baseline_control_market1501 | /root/autodl-tmp/experiments/BoT/c2_baseline_control_market1501/log.txt | NVIDIA GeForce RTX 4090 | 待核实 | 0 (disabled) | 0:38:47 | 120 | 94.4% | 待核实 | 待核实 | 85.5% | no | Git 已登记于结果提交 9b81850；这是关闭 C2 的公平控制组，不是关闭 Part Attention 的裸 BoT。 |

## C2 Lambda Sensitivity Experiments

| 实验编号 | 日期 | commit id | 分支 | 实验类型 | 数据集 | config 文件 | OUTPUT_DIR | 日志路径 | GPU | seed | lambda | 运行时间 | best epoch | Rank-1 | Rank-5 | Rank-10 | mAP | re-ranking | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2-L01 | 待核实 | 待核实 | exp/cross-camera-positive-lambda-sensitivity | Cross-camera positive only | Market1501 | configs/softmax_triplet_cross_camera_positive_lambda01_autodl.yml | /root/autodl-tmp/experiments/BoT/cross_camera_positive_lambda01_market1501 | 待核实 | 待核实 | 待核实 | 0.1 | 待核实 | 待核实 | 待实验/待归档 | 待核实 | 待核实 | 待实验/待归档 | no | λ 消融；不能由配置推断结果。 |
| C2-L03 | 待核实 | 待核实 | exp/cross-camera-positive-lambda-sensitivity | Cross-camera positive only（当前主方法） | Market1501 | configs/softmax_triplet_cross_camera_positive_lambda03_autodl.yml | /root/autodl-tmp/experiments/BoT/cross_camera_positive_lambda03_market1501 | 待核实 | 待核实 | 待核实 | 0.3 | 待核实 | 待核实 | 95.0% | 待核实 | 待核实 | 87.8% | no | 当前 Market 最优；数值由用户确认。运行 commit、原始日志、checkpoint、seed、best epoch 和运行时间待归档。λ 是 C2 辅助损失权重，不是 Triplet margin。 |
| C2-L05 | 待核实 | 待核实 | exp/cross-camera-positive-lambda-sensitivity | Cross-camera positive only | Market1501 | configs/softmax_triplet_cross_camera_positive_lambda05_autodl.yml | /root/autodl-tmp/experiments/BoT/cross_camera_positive_lambda05_market1501 | 待核实 | 待核实 | 待核实 | 0.5 | 待核实 | 待核实 | 待实验/待归档 | 待核实 | 待核实 | 待实验/待归档 | no | λ 消融；独立 C2 λ=0.5 运行见 C2-CCPO-Market，但不能自动视为本配置的同一次运行。 |
| C2-L10 | 待核实 | 待核实 | exp/cross-camera-positive-lambda-sensitivity | Cross-camera positive only | Market1501 | configs/softmax_triplet_cross_camera_positive_lambda10_autodl.yml | /root/autodl-tmp/experiments/BoT/cross_camera_positive_lambda10_market1501 | 待核实 | 待核实 | 待核实 | 1.0 | 待核实 | 待核实 | 待实验/待归档 | 待核实 | 待核实 | 待实验/待归档 | no | λ 消融；不能由配置推断结果。 |

## C2-L03 Multi-Granularity Part Experiments

| 实验编号 | 日期 | 训练 commit | 分支 | 实验类型 | 数据集 | config 文件 | OUTPUT_DIR | 日志路径 | GPU | seed | lambda | 训练时间 | best epoch | Rank-1 | Rank-5 | Rank-10 | mAP | re-ranking | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2-MGP-K246-S42 | 2026-08-05 | f7e9e46 | exp/c2-l03-multi-granularity-local-feature | C2-L03 + Global + Multi-Granularity Part K={2,4,6} | Market1501 | configs/softmax_triplet_c2_l03_multi_granularity_part_autodl.yml | /root/autodl-tmp/experiments/BoT/c2_l03_multi_granularity_part_market1501 | /root/autodl-tmp/experiments/BoT/c2_l03_multi_granularity_part_market1501/log.txt | NVIDIA GeForce RTX 4090 | 42 | 0.3 | 0:46:58 | 120 | 94.8% | 98.4% | 98.9% | 87.5% | no | Global 与 K=2/4/6 局部分支 concat 融合，每个局部分支 256 维；未启用 fixed-index PCC 或软对齐；正式台账保留精确指标（Rank-1 94.7743475%，mAP 87.5333545%）及证据哈希；finalization commit 40293f4；n=1，外部归档尚未记录，不作稳定性或显著性结论。 |

机器可读真源位于 `experiment_records/c2_l03_multi_granularity_part/`。本次恢复仅修复并登记已完成训练的证据，没有重新训练；训练运行时间为 2818.077 秒，总流程时间为 2836.141 秒。

## Duke Validation Experiments

Market1501 current best: C2-L03, lambda=0.3, Rank-1=95.0, mAP=87.8（用户确认；原始运行证据待归档）。

This section compares the baseline control with C2-L03 on DukeMTMC-reID. The
results were registered by commit `6f49104`; the external AutoDL logs and
checkpoints are not present in this checkout, and the seed/Rank-5/Rank-10 remain
unverified.

| 实验编号 | 日期 | commit id | 分支 | 实验类型 | 数据集 | config 文件 | OUTPUT_DIR | 日志路径 | GPU | seed | lambda | 运行时间 | best epoch | Rank-1 | Rank-5 | Rank-10 | mAP | re-ranking | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Duke-Baseline-Control | 2026-07-18 | 3ce4724 | exp/c2-l03-duke-validation | Baseline control（BoT + Part Attention） | DukeMTMC-reID | configs/softmax_triplet_c2_l03_duke_baseline_autodl.yml | /root/autodl-tmp/experiments/BoT/duke_baseline_control | /root/autodl-tmp/experiments/BoT/duke_baseline_control/log.txt | NVIDIA GeForce RTX 4090 | 待核实 | 0 (disabled) | 0:47:46 | 80 | 86.7% | 待核实 | 待核实 | 75.7% | no | Git 已登记于 6f49104；与 C2-L03 的设置除 C2 开关和 OUTPUT_DIR 外一致。原始日志/checkpoint未归档。 |
| Duke-C2-L03 | 2026-07-18 | 3ce4724 | exp/c2-l03-duke-validation | Cross-camera positive only | DukeMTMC-reID | configs/softmax_triplet_c2_l03_duke_autodl.yml | /root/autodl-tmp/experiments/BoT/duke_c2_l03 | /root/autodl-tmp/experiments/BoT/duke_c2_l03/log.txt | NVIDIA GeForce RTX 4090 | 待核实 | 0.3 | 0:53:21 | 120 | 88.4% | 待核实 | 待核实 | 78.7% | no | Git 已登记于 6f49104；相对登记控制组 Rank-1 +1.7、mAP +3.0 个百分点。seed 与原始日志/checkpoint待归档。 |

## Cross-Branch Ablation Results

以下结果来自独立远程分支的 Git 实验登记，用于论文消融，不作为主方法。
当前 checkout 没有相应原始日志/checkpoint；Rank-5/Rank-10 和 seed 待核实。

| 实验编号 | 日期 | 运行代码 commit | 结果提交 | 分支 | 实验类型 | 数据集 | lambda | auxiliary margin | best epoch | Rank-1 | Rank-5 | Rank-10 | mAP | re-ranking | 证据说明 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| CAT-L01 / CAAT-L01 | 2026-07-11 | 61da0df | 73a44c0 | exp/camera-aware-triplet-lambda-sensitivity | 完整 CAAT：跨相机 hardest positive + hardest negative | Market1501 | 0.1 | 0.3 | 120 | 94.2% | 98.1% | 98.9% | 85.5% | no | 原始 log 和三轮 checkpoint 已归档；seed not_recorded；单次运行。 |
| CAT-L03 / CAAT-L03 | 2026-07-11 | 61da0df | 73a44c0 | exp/camera-aware-triplet-lambda-sensitivity | 完整 CAAT：跨相机 hardest positive + hardest negative | Market1501 | 0.3 | 0.3 | 120 | 94.2% | 98.4% | 98.9% | 85.5% | no | 原始 log 和三轮 checkpoint 已归档；seed not_recorded；单次运行。 |
| CAT-L05 / CAAT-L05 | 2026-07-11 | 61da0df | 73a44c0 | exp/camera-aware-triplet-lambda-sensitivity | 完整 CAAT：跨相机 hardest positive + hardest negative | Market1501 | 0.5 | 0.3 | 120 | 94.2% | 98.0% | 98.8% | 85.4% | no | 论文完整 CAAT 消融主行；原始 log 和三轮 checkpoint 已归档；seed not_recorded；单次运行。 |
| CAT-L10 / CAAT-L10 | 2026-07-11 | 61da0df | 73a44c0 | exp/camera-aware-triplet-lambda-sensitivity | 完整 CAAT：跨相机 hardest positive + hardest negative | Market1501 | 1.0 | 0.3 | 120 | 94.1% | 98.3% | 99.0% | 84.8% | no | 原始 log 和三轮 checkpoint 已归档；seed not_recorded；单次运行。 |
| S2-SCPO-Market | 2026-07-14 | f1f1692 | a4a42b3 | exp/same-camera-positive-only | Same-camera positive only | Market1501 | 0.5 | 不适用 | 120 | 94.4% | 待核实 | 待核实 | 86.8% | no | Git 已登记；实现只存在于独立 S2 分支。 |

## Cross-Camera Batch Coverage Analysis

以下是固定 seed=42、按当前 RandomIdentitySampler 规则离线模拟 10 个 epoch
的监督机会覆盖率；它不使用模型或 checkpoint，不能单独证明 C2 有效。

| 数据集 | 训练图像 | batch/K | epoch | 总 anchors | 有效 anchors | 加权有效 anchor 比例 | 每 batch 比例均值±总体标准差 | min/median/max | 零有效 batch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Market1501 | 12936 | 64/4 | 10 | 117376 | 114852 | 97.8496% | 97.8496% ± 3.6160% | 81.25% / 100% / 100% | 0/1834 |
| DukeMTMC-reID | 16522 | 64/4 | 10 | 144576 | 135980 | 94.0543% | 94.0543% ± 7.0234% | 43.75% / 93.75% / 100% | 0/2259 |

生成工具：`tools/analyze_cross_camera_batch_coverage.py`。训练日志中的
`cross_camera_positive_count` 是“至少拥有一个跨相机同 ID 正样本的有效
anchor 数”，不是 positive pair 数。

## Distance Distribution Analysis

正式距离表见 `paper_notes/c2_l03_final_evidence/UNIFIED_TABLES.md` 表 6。
Market 已按 `pid>0` 重做为 `market_epoch120_person_only_v2`；旧的含 `pid=0`
结果为 E0、`superseded`。Market 与 Duke 当前机制结论均按 E2 使用，只作
固定 checkpoint 和协议下的描述，不作统计显著性或普遍机制声明。
