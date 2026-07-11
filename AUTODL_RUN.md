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

## Camera-Aware Triplet Lambda Sensitivity

1. Branch: `exp/camera-aware-triplet-lambda-sensitivity`.

2. Only `CAMERA_AWARE_TRIPLET_LAMBDA` is adjusted on top of BoT + L_camera_triplet. L_camera_triplet uses cross-camera positives and hard negative mining. No hierarchical difficulty, hard negative weighting, pseudo labels, memory bank, or HNIE is added. This is not a cross-camera-positive-only experiment.

3. Configs:

- `configs/softmax_triplet_camera_aware_lambda01_autodl.yml`
- `configs/softmax_triplet_camera_aware_lambda03_autodl.yml`
- `configs/softmax_triplet_camera_aware_lambda05_autodl.yml`
- `configs/softmax_triplet_camera_aware_lambda10_autodl.yml`

4. Output directories:

- `/root/autodl-tmp/experiments/BoT/camera_aware_triplet_lambda01_market1501`
- `/root/autodl-tmp/experiments/BoT/camera_aware_triplet_lambda03_market1501`
- `/root/autodl-tmp/experiments/BoT/camera_aware_triplet_lambda05_market1501`
- `/root/autodl-tmp/experiments/BoT/camera_aware_triplet_lambda10_market1501`

5. These independent directories do not overwrite the existing `camera_aware_triplet_market1501`, `cross_camera_positive_only_market1501`, `normalized_weighted_loss_market1501`, `softmax_triplet_part_attention_k6_tau01_market1501`, or `softmax_triplet_part_attention_k6_tau02_market1501` directories under `/root/autodl-tmp/experiments/BoT/`.

6. Recommended AutoDL clone directory:

```bash
cd /root/autodl-tmp
git clone -b exp/camera-aware-triplet-lambda-sensitivity https://github.com/lzhe036-web/BoT-reid-part-attention.git BoT-reid-cat-lambda
cd /root/autodl-tmp/BoT-reid-cat-lambda
```

7. Dataset path: `/root/autodl-tmp/datasets/market1501`. If the actual directory is `data`:

```bash
cd /root/autodl-tmp
rm -f datasets
ln -s data datasets
```

8. Pretrained weight: `/root/autodl-tmp/pretrained/resnet50-19c8e357.pth`.

9. Individual runs:

```bash
bash scripts/train_camera_aware_triplet_lambda01_autodl.sh
bash scripts/train_camera_aware_triplet_lambda03_autodl.sh
bash scripts/train_camera_aware_triplet_lambda05_autodl.sh
bash scripts/train_camera_aware_triplet_lambda10_autodl.sh
```

10. Sequential run:

```bash
bash scripts/train_camera_aware_triplet_lambda_sensitivity_autodl.sh
```

11. nohup example and log:

```bash
cd /root/autodl-tmp/BoT-reid-cat-lambda
nohup bash scripts/train_camera_aware_triplet_lambda_sensitivity_autodl.sh > cat_lambda_sensitivity.out 2>&1 &
tail -f cat_lambda_sensitivity.out
```

12. Each successful run automatically updates `/root/autodl-tmp/BoT-reid-cat-lambda/EXPERIMENTS.md`.

13. Save completed records to GitHub manually:

```bash
git add EXPERIMENTS.md
git commit -m "record camera aware triplet lambda sensitivity results"
git push
```
