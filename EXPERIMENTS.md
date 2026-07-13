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
| C2-CCPO-Market | 2026-07-13 | d98fb00 | exp/cross-camera-positive-only | Cross-camera positive only | Market1501 | configs/softmax_triplet_cross_camera_positive_only_autodl.yml | /root/autodl-tmp/experiments/BoT/cross_camera_positive_only_market1501 | /root/autodl-tmp/experiments/BoT/cross_camera_positive_only_market1501/log.txt | NVIDIA GeForce RTX 4090 | 待填写 | 0.5 | 0:44:30 | 120 | 95.0% | 87.7% | Cross-camera positive only, no extra hard negative, Market1501, AutoDL |
| C2-CCPO-Repeat | 2026-07-13 | d98fb00 | exp/cross-camera-positive-only | Cross-camera positive only | Market1501 | configs/softmax_triplet_cross_camera_positive_only_repeat_autodl.yml | /root/autodl-tmp/experiments/BoT/cross_camera_positive_only_repeat_market1501 | /root/autodl-tmp/experiments/BoT/cross_camera_positive_only_repeat_market1501/log.txt | NVIDIA GeForce RTX 4090 | 待填写 | 0.5 | 0:44:06 | 80 | 95.0% | 87.3% | C2 repeat run to verify stability, cross-camera positive only, no extra hard negative. |

## C2 Baseline-Control Experiments

| 实验编号 | 日期 | commit id | 分支 | 实验类型 | 数据集 | config 文件 | OUTPUT_DIR | 日志路径 | GPU | seed | lambda | 运行时间 | best epoch | Rank-1 | mAP | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2-Baseline-Control | 2026-07-13 | d98fb00 | exp/cross-camera-positive-only | Baseline control | Market1501 | configs/softmax_triplet_c2_baseline_control_autodl.yml | /root/autodl-tmp/experiments/BoT/c2_baseline_control_market1501 | /root/autodl-tmp/experiments/BoT/c2_baseline_control_market1501/log.txt | NVIDIA GeForce RTX 4090 | 待填写 | 0 (disabled) | 0:38:47 | 120 | 94.4% | 85.5% | Baseline control under C2 setting, cross-camera positive loss disabled, other settings aligned with C2. |
