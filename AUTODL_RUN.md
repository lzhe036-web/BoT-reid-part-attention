# AutoDL Run Guide

## Paths

- Project directory: `/root/autodl-tmp/BoT-reid`
- Dataset directory: `/root/autodl-tmp/datasets`
- Pretrained weight: `/root/autodl-tmp/pretrained/resnet50-19c8e357.pth`
- Training output directory: `/root/autodl-tmp/experiments/BoT/softmax_triplet_part_attention_market1501`

## Run Training

Manual command:

```bash
python tools/train.py --config_file configs/softmax_triplet_part_attention_autodl.yml
```

Script command:

```bash
bash scripts/train_part_attention_autodl.sh
```

## Run In tmux

Create a tmux session:

```bash
tmux new -s bot_part_attention
```

Start training:

```bash
cd /root/autodl-tmp/BoT-reid
bash scripts/train_part_attention_autodl.sh
```

Detach from tmux:

```bash
Ctrl-b d
```

Reattach later:

```bash
tmux attach -t bot_part_attention
```

## GitHub Notes

Do not upload datasets, pretrained weights, experiment outputs, logs, or model checkpoint files to GitHub.

Keep these paths and files local to AutoDL or external storage:

- `data/`
- `datasets/`
- `pretrained/`
- `experiments/`
- `log/`
- `logs/`
- `saved-models/`
- `*.pth`
- `*.pt`
- `*.pkl`

## Camera Bias Debias Validation D1 / D2

### 实验目的与方法

验证测试期 camera mean debias 是否能缓解 camera bias。对 BNNeck 后特征 `f`，按 `camid` 求均值 `m_cam`，再执行：

```text
f' = normalize(f - m_cam)
```

当前统计协议是 **joint query-gallery camera mean debias**：先合并 query 和 gallery 特征，再按 `camid` 计算 camera mean。该方法仅为测试期后处理，不修改训练 loss 或模型结构。

- D1：Market1501 → Market1501，对比 Camera Debias `False` / `True`，判断同域性能是否持平或小涨。
- D2：DukeMTMC-reID → Market1501，对比 Camera Debias `False` / `True`，判断跨域性能是否明显改善。D2 配置中的测试集仍为 Market1501；Duke 来源由 Duke checkpoint 和实验记录中的训练集字段表示。

### AutoDL 准备

建议 clone 到独立目录：

```bash
cd /root/autodl-tmp
git clone -b exp-bnneck-camera-debias https://github.com/lzhe036-web/BoT-reid-part-attention.git BoT-reid-camera-bias
cd /root/autodl-tmp/BoT-reid-camera-bias
```

预期数据路径：

```text
/root/autodl-tmp/datasets/market1501
/root/autodl-tmp/datasets/dukemtmc-reid
```

如果实际数据根目录为 `/root/autodl-tmp/data`，可建立软链接：

```bash
cd /root/autodl-tmp
rm -f datasets
ln -s data datasets
```

D1 需要 Market1501 训练得到的 checkpoint，脚本变量为 `MARKET_CKPT`。D2 需要 DukeMTMC-reID 训练得到的 checkpoint，脚本变量为 `DUKE_CKPT`。请确保 checkpoint 的模型结构与配置匹配。

默认路径不存在时，可在运行时指定：

```bash
MARKET_CKPT=/path/to/market_checkpoint.pt bash scripts/test_camera_bias_d1_market_autodl.sh
DUKE_CKPT=/path/to/duke_checkpoint.pt bash scripts/test_camera_bias_d2_duke2market_autodl.sh
```

### 运行与日志

```bash
bash scripts/test_camera_bias_d1_market_autodl.sh
bash scripts/test_camera_bias_d2_duke2market_autodl.sh
bash scripts/test_camera_bias_d1_d2_autodl.sh
```

后台运行全部实验：

```bash
nohup bash scripts/test_camera_bias_d1_d2_autodl.sh > camera_bias_d1_d2.out 2>&1 &
tail -f camera_bias_d1_d2.out
```

测试成功后，脚本自动更新：

```text
/root/autodl-tmp/BoT-reid-camera-bias/EXPERIMENTS.md
```

D1/D2 四份配置设置 `TEST.AUTO_RECORD=False`，因此 `tools/test.py` 不写通用 `Auto Test Records`；各脚本只通过 `append_camera_bias_result.py` 更新专用的 `Camera Bias Debias Validation Experiments` 表格，避免重复记录。其他配置默认 `TEST.AUTO_RECORD=True`，原有自动记录行为不变。

如需将结果保存到 GitHub，请人工确认记录后执行：

```bash
git add EXPERIMENTS.md
git commit -m "record camera bias debias validation results"
git push
```
