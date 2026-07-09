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

## Camera-Aware Triplet Loss

1. Branch:

```bash
exp/camera-aware-triplet-loss
```

2. Output locations:

Training results:

```bash
/root/autodl-tmp/experiments/BoT/camera_aware_triplet_market1501
```

Training log:

```bash
/root/autodl-tmp/experiments/BoT/camera_aware_triplet_market1501/log.txt
```

Model checkpoints:

```bash
/root/autodl-tmp/experiments/BoT/camera_aware_triplet_market1501/*.pt
```

Automatic experiment record:

```bash
/root/autodl-tmp/BoT-reid/EXPERIMENTS.md
```

3. Check required paths before running:

```bash
ls /root/autodl-tmp/datasets/market1501
ls /root/autodl-tmp/pretrained
```

4. Record basic information before training:

```bash
bash scripts/record_experiment_info.sh configs/softmax_triplet_camera_aware_autodl.yml
```

5. Start training:

```bash
bash scripts/train_camera_aware_triplet_autodl.sh
```

The script runs training first. If training succeeds, it automatically updates:

```bash
/root/autodl-tmp/BoT-reid/EXPERIMENTS.md
```

If training fails, `EXPERIMENTS.md` is not updated.

6. Run in tmux:

```bash
tmux new -s cat
cd /root/autodl-tmp/BoT-reid
bash scripts/train_camera_aware_triplet_autodl.sh
```

7. View Rank-1 and mAP:

```bash
grep -i "mAP" /root/autodl-tmp/experiments/BoT/camera_aware_triplet_market1501/log.txt
grep -i "Rank" /root/autodl-tmp/experiments/BoT/camera_aware_triplet_market1501/log.txt
```

8. Manually update experiment results, if needed:

```bash
python scripts/append_experiment_result.py --config configs/softmax_triplet_camera_aware_autodl.yml --experiment-id CAT001 --note "Camera-aware triplet loss, lambda=0.5, margin=0.3, AutoDL" --dry-run
```

After confirming the dry-run output, remove `--dry-run` and use `--mode update`.

9. Save the experiment record to GitHub:

The training script does not run `git commit` or `git push`. To save the updated experiment record to GitHub, run:

```bash
git add EXPERIMENTS.md
git commit -m "record camera aware triplet experiment result"
git push
```
