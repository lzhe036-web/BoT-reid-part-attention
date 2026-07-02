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

## Tau Sensitivity Experiments

Run the K=6 part attention tau sensitivity experiments with:

```bash
bash scripts/train_part_attention_k6_tau01_autodl.sh
bash scripts/train_part_attention_k6_tau02_autodl.sh
bash scripts/train_part_attention_k6_tau05_autodl.sh
```

## 自动整理实验结果

训练结束后，可以手动运行结果整理脚本。这个脚本不会训练模型，只会从 config、git 信息和 `log.txt` 中尽量解析实验结果，并追加或更新 `EXPERIMENTS.md`。

推荐先使用 `--dry-run` 检查解析结果和即将写入的 markdown 行：

tau=0.1：

```bash
python scripts/append_experiment_result.py --config configs/softmax_triplet_part_attention_k6_tau01_autodl.yml --experiment-id T001 --note "BoT + Part Attention, K=6, tau=0.1, AutoDL"
```

tau=0.2：

```bash
python scripts/append_experiment_result.py --config configs/softmax_triplet_part_attention_k6_tau02_autodl.yml --experiment-id T002 --note "BoT + Part Attention, K=6, tau=0.2, AutoDL"
```

tau=0.5：

```bash
python scripts/append_experiment_result.py --config configs/softmax_triplet_part_attention_k6_tau05_autodl.yml --experiment-id T003 --note "BoT + Part Attention, K=6, tau=0.5, AutoDL"
```

脚本会从 `log.txt` 中尽量解析 Rank-1 和 mAP。如果解析不到，会填写“待填写”，需要人工补充。

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
