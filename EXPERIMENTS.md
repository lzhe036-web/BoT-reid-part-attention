# Experiments

| 实验编号 | 日期 | commit id | 分支 | config 文件 | seed | GPU | 数据集 | 运行时间 | best epoch | Rank-1 | mAP | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E001 | 待填写 | 待填写 | main-autodl | configs/softmax_triplet_part_attention_autodl.yml | 待填写 | 待填写 | Market1501 | 待填写 | 待填写 | 待填写 | 待填写 | BoT + Part Attention, K=6, AutoDL |
| E002 | 待运行/待填写 | 2b06f88 | exp-bnneck-camera-debias | configs/softmax_triplet_with_center.yml | 待填写 | 待填写 | Market1501 | 待运行/待填写 | 待运行/待填写 | 待运行/待填写 | 待运行/待填写 | TEST.CAMERA_MEAN_DEBIAS=True；BNNeck after feature；测试阶段按 camera mean 去偏置；未运行 |

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

## Camera Bias Debias Validation Experiments

| 实验编号 | 日期 | commit id | 分支 | 训练集 | 测试集 | checkpoint | camera debias | config 文件 | Rank-1 | mAP | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D1-Market-Debias-Off | 2026-07-11 18:43:12 | d998f4b | exp-bnneck-camera-debias | Market1501 | Market1501 | /root/autodl-tmp/experiments/BoT/softmax_triplet_part_attention_market1501/resnet50_checkpoint_22320.pt | False | configs/test_market1501_debias_off_autodl.yml | 85.0% | 69.3% | D1 Market1501 same-domain, camera debias off, joint query-gallery camera mean protocol |
| D1-Market-Debias-On | 2026-07-11 18:44:41 | d998f4b | exp-bnneck-camera-debias | Market1501 | Market1501 | /root/autodl-tmp/experiments/BoT/softmax_triplet_part_attention_market1501/resnet50_checkpoint_22320.pt | True | configs/test_market1501_debias_on_autodl.yml | 87.2% | 73.1% | D1 Market1501 same-domain, camera debias on, joint query-gallery camera mean protocol |
| D2-Duke2Market-Debias-Off | 2026-07-11 18:46:09 | d998f4b | exp-bnneck-camera-debias | DukeMTMC-reID | Market1501 | /root/autodl-tmp/experiments/BoT/softmax_triplet_part_attention_dukemtmc-reid/resnet50_checkpoint_22320.pt | False | configs/test_duke2market_debias_off_autodl.yml | 84.3% | 67.3% | D2 DukeMTMC-reID to Market1501 cross-domain, camera debias off, joint query-gallery camera mean protocol |
| D2-Duke2Market-Debias-On | 2026-07-11 18:47:40 | d998f4b | exp-bnneck-camera-debias | DukeMTMC-reID | Market1501 | /root/autodl-tmp/experiments/BoT/softmax_triplet_part_attention_dukemtmc-reid/resnet50_checkpoint_22320.pt | True | configs/test_duke2market_debias_on_autodl.yml | 87.6% | 71.8% | D2 DukeMTMC-reID to Market1501 cross-domain, camera debias on, joint query-gallery camera mean protocol |
