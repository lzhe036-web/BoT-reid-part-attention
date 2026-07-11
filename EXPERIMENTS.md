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

## Camera-Aware Triplet Lambda Sensitivity Experiments

| 实验编号 | 日期 | commit id | 分支 | config 文件 | seed | GPU | 数据集 | lambda | margin | 运行时间 | best epoch | Rank-1 | mAP | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CAT-L01 | 待填写 | 待填写 | exp/camera-aware-triplet-lambda-sensitivity | configs/softmax_triplet_camera_aware_lambda01_autodl.yml | 待填写 | 待填写 | Market1501 | 0.1 | 0.3 | 待填写 | 待填写 | 待填写 | 待填写 | BoT + L_camera_triplet, camera-aware hard triplet |
| CAT-L03 | 待填写 | 待填写 | exp/camera-aware-triplet-lambda-sensitivity | configs/softmax_triplet_camera_aware_lambda03_autodl.yml | 待填写 | 待填写 | Market1501 | 0.3 | 0.3 | 待填写 | 待填写 | 待填写 | 待填写 | BoT + L_camera_triplet, camera-aware hard triplet |
| CAT-L05 | 待填写 | 待填写 | exp/camera-aware-triplet-lambda-sensitivity | configs/softmax_triplet_camera_aware_lambda05_autodl.yml | 待填写 | 待填写 | Market1501 | 0.5 | 0.3 | 待填写 | 待填写 | 待填写 | 待填写 | BoT + L_camera_triplet, camera-aware hard triplet |
| CAT-L10 | 待填写 | 待填写 | exp/camera-aware-triplet-lambda-sensitivity | configs/softmax_triplet_camera_aware_lambda10_autodl.yml | 待填写 | 待填写 | Market1501 | 1.0 | 0.3 | 待填写 | 待填写 | 待填写 | 待填写 | BoT + L_camera_triplet, camera-aware hard triplet |
