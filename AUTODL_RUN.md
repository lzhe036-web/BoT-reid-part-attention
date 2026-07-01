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
