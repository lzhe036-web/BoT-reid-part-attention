# Experiments

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
8. Rank-1 和 mAP：从测试/评估输出中填写。
9. 备注：记录是否使用 Part Attention、K 值、是否 reranking、是否修改 batch size 等。

## Camera-Aware Triplet Loss Experiments

| 实验编号 | 日期 | commit id | 分支 | config 文件 | seed | GPU | 数据集 | 运行时间 | best epoch | Rank-1 | mAP | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CAT001 | 待填写 | 待填写 | exp/camera-aware-triplet-loss | configs/softmax_triplet_camera_aware_autodl.yml | 待填写 | 待填写 | Market1501 | 待填写 | 待填写 | 待填写 | 待填写 | Camera-aware triplet loss, lambda=0.5, margin=0.3, AutoDL |

## Cross-Camera Positive Only Experiments

| 实验编号 | 日期 | commit id | 分支 | 实验类型 | 数据集 | config 文件 | OUTPUT_DIR | 日志路径 | GPU | seed | lambda | 运行时间 | best epoch | Rank-1 | mAP | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2-CCPO-Market | 2026-07-09 | 0eeb467 | exp/cross-camera-positive-only | Cross-camera positive only | Market1501 | configs/softmax_triplet_cross_camera_positive_only_autodl.yml | /root/autodl-tmp/experiments/BoT/cross_camera_positive_only_market1501 | 待填写 | 待填写 | 待填写 | 0.5 | 待填写 | 待填写 | 待填写 | 待填写 | Cross-camera positive only, no extra hard negative, Market1501, AutoDL。 |
| C2-CCPO-Repeat | 待填写 | 待填写 | exp/cross-camera-positive-only | Cross-camera positive only | Market1501 | configs/softmax_triplet_cross_camera_positive_only_repeat_autodl.yml | /root/autodl-tmp/experiments/BoT/cross_camera_positive_only_repeat_market1501 | 待填写 | 待填写 | 待填写 | 0.5 | 待填写 | 待填写 | 待填写 | 待填写 | C2 repeat run to verify stability, cross-camera positive only, no extra hard negative. |

## C2 Baseline-Control Experiments

| 实验编号 | 日期 | commit id | 分支 | 实验类型 | 数据集 | config 文件 | OUTPUT_DIR | 日志路径 | GPU | seed | lambda | 运行时间 | best epoch | Rank-1 | mAP | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2-Baseline-Control | 待填写 | 待填写 | exp/cross-camera-positive-only | Baseline control | Market1501 | configs/softmax_triplet_c2_baseline_control_autodl.yml | /root/autodl-tmp/experiments/BoT/c2_baseline_control_market1501 | 待填写 | 待填写 | 待填写 | 0 (disabled) | 待填写 | 待填写 | 待填写 | 待填写 | Baseline control under C2 setting, cross-camera positive loss disabled, other settings aligned with C2. |

## C2 Lambda Sensitivity Experiments

| 实验编号 | 日期 | commit id | 分支 | 实验类型 | 数据集 | config 文件 | OUTPUT_DIR | 日志路径 | GPU | seed | lambda | 运行时间 | best epoch | Rank-1 | mAP | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2-L01 | 待填写 | 待填写 | exp/cross-camera-positive-lambda-sensitivity | Cross-camera positive only | Market1501 | configs/softmax_triplet_cross_camera_positive_lambda01_autodl.yml | /root/autodl-tmp/experiments/BoT/cross_camera_positive_lambda01_market1501 | 待填写 | 待填写 | 待填写 | 0.1 | 待填写 | 待填写 | 待填写 | 待填写 | C2 lambda sensitivity, no extra hard negative. |
| C2-L03 | 待填写 | 待填写 | exp/cross-camera-positive-lambda-sensitivity | Cross-camera positive only | Market1501 | configs/softmax_triplet_cross_camera_positive_lambda03_autodl.yml | /root/autodl-tmp/experiments/BoT/cross_camera_positive_lambda03_market1501 | 待填写 | 待填写 | 待填写 | 0.3 | 待填写 | 待填写 | 待填写 | 待填写 | C2 lambda sensitivity, no extra hard negative. |
| C2-L05 | 待填写 | 待填写 | exp/cross-camera-positive-lambda-sensitivity | Cross-camera positive only | Market1501 | configs/softmax_triplet_cross_camera_positive_lambda05_autodl.yml | /root/autodl-tmp/experiments/BoT/cross_camera_positive_lambda05_market1501 | 待填写 | 待填写 | 待填写 | 0.5 | 待填写 | 待填写 | 待填写 | 待填写 | C2 lambda sensitivity, no extra hard negative. |
| C2-L10 | 待填写 | 待填写 | exp/cross-camera-positive-lambda-sensitivity | Cross-camera positive only | Market1501 | configs/softmax_triplet_cross_camera_positive_lambda10_autodl.yml | /root/autodl-tmp/experiments/BoT/cross_camera_positive_lambda10_market1501 | 待填写 | 待填写 | 待填写 | 1.0 | 待填写 | 待填写 | 待填写 | 待填写 | C2 lambda sensitivity, no extra hard negative. |

<!-- AUTO-EXPERIMENT-RESULTS:START -->
## Automated Formal Experiment Runs

This section is generated from `experiment_records/tables/main_results.csv`.
Historical experiment rows outside this section are never rewritten.

| experiment_id | run_id | run_kind | date | commit | branch | parent_branch | parent_commit | method | method_family | method_variant | dataset | config | output_dir | log_path | log_sha256 | GPU | seed | lambda | cross_camera_positive_lambda | pcc_lambda | pcc_enabled | pcc_parts | pcc_mode | alignment_strategy | alignment_mode | alignment_temperature | gating_mode | gating_temperature | multigranular_feature_signature_sha256 | baseline | valid_pcc_pair_count | mean_fixed_index_part_distance | hard_alignment_loss | valid_alignment_pair_count | mean_hard_path_cost | mean_path_absolute_offset | soft_alignment_loss | mean_soft_path_cost | runtime_seconds | best_epoch | Rank-1 | Rank-5 | Rank-10 | mAP | checkpoint | checkpoint_sha256 | status | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
<!-- AUTO-EXPERIMENT-RESULTS:END -->
