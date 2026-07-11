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
| D1-Market-Debias-Off | 待填写 | 待填写 | exp-bnneck-camera-debias | Market1501 | Market1501 | 待填写 | False | configs/test_market1501_debias_off_autodl.yml | 待填写 | 待填写 | D1 同域基线；统计协议为 joint query-gallery camera mean debias |
| D1-Market-Debias-On | 待填写 | 待填写 | exp-bnneck-camera-debias | Market1501 | Market1501 | 待填写 | True | configs/test_market1501_debias_on_autodl.yml | 待填写 | 待填写 | D1 同域去偏；joint query-gallery camera mean debias |
| D2-Duke2Market-Debias-Off | 待填写 | 待填写 | exp-bnneck-camera-debias | DukeMTMC-reID | Market1501 | 待填写 | False | configs/test_duke2market_debias_off_autodl.yml | 待填写 | 待填写 | D2 跨域基线；统计协议为 joint query-gallery camera mean debias |
| D2-Duke2Market-Debias-On | 待填写 | 待填写 | exp-bnneck-camera-debias | DukeMTMC-reID | Market1501 | 待填写 | True | configs/test_duke2market_debias_on_autodl.yml | 待填写 | 待填写 | D2 跨域去偏；joint query-gallery camera mean debias |
