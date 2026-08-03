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

## Cross-Camera Positive Only

1. Branch:

```bash
exp/cross-camera-positive-only
```

2. Experiment meaning:

Only same-pid and different-camid cross-camera positives are used as the extra auxiliary constraint. No extra hard negative mining is used. No extra hard negative weighting is used. The original BoT triplet loss is still kept.

3. Output directory:

```bash
/root/autodl-tmp/experiments/BoT/cross_camera_positive_only_market1501
```

4. Training log:

```bash
/root/autodl-tmp/experiments/BoT/cross_camera_positive_only_market1501/log.txt
```

5. Checkpoints:

```bash
/root/autodl-tmp/experiments/BoT/cross_camera_positive_only_market1501/*.pt
```

6. Automatic experiment record:

```bash
/root/autodl-tmp/BoT-reid/EXPERIMENTS.md
```

7. Check required paths before running:

```bash
ls /root/autodl-tmp/datasets/market1501
ls /root/autodl-tmp/pretrained
```

8. Start training:

```bash
bash scripts/train_cross_camera_positive_only_autodl.sh
```

9. Run in tmux:

```bash
tmux new -s ccpo
cd /root/autodl-tmp/BoT-reid
bash scripts/train_cross_camera_positive_only_autodl.sh
```

10. Save the experiment record to GitHub:

After training ends successfully, the script automatically updates:

```bash
/root/autodl-tmp/BoT-reid/EXPERIMENTS.md
```

To save the updated experiment record to GitHub, run:

```bash
git add EXPERIMENTS.md
git commit -m "record cross camera positive only experiment result"
git push
```

## C2 Ablations and Reproduction

- Baseline control: `bash scripts/train_c2_baseline_control_autodl.sh`
- C2 repeat: `bash scripts/train_cross_camera_positive_only_repeat_autodl.sh`

Each successful training script updates `EXPERIMENTS.md` with `--mode update`; use `--dry-run` to preview records without writing.

## C2 Lambda Sensitivity

Run the C2 lambda sensitivity sequence only from branch
`exp/cross-camera-positive-lambda-sensitivity`. Before running, either switch an
existing clone to that branch:

```bash
git switch exp/cross-camera-positive-lambda-sensitivity
bash scripts/train_cross_camera_positive_lambda_sensitivity_autodl.sh
```

Or clone the lambda branch directly:

```bash
git clone --branch exp/cross-camera-positive-lambda-sensitivity --single-branch https://github.com/lzhe036-web/BoT-reid-part-attention.git BoT-reid-c2-lambda
cd BoT-reid-c2-lambda
bash scripts/train_cross_camera_positive_lambda_sensitivity_autodl.sh
```

The sequence runs lambda values 0.1, 0.3, 0.5, and 1.0 in order. Each successful
training script updates `EXPERIMENTS.md` with `--mode update`.

## Duke C2-L03 Validation

Run this validation from the correct branch:

```bash
git switch exp/c2-l03-duke-validation
```

The DukeMTMC-reID dataset registration name must be `dukemtmc`.

Expected AutoDL dataset structure:

```text
/root/autodl-tmp/datasets/dukemtmc-reid/DukeMTMC-reID/bounding_box_train
```

If the dataset is instead located at `/root/autodl-tmp/datasets/DukeMTMC-reID`,
create the expected link:

```bash
cd /root/autodl-tmp/datasets
mkdir -p dukemtmc-reid
ln -s ../DukeMTMC-reID dukemtmc-reid/DukeMTMC-reID
```

Run either experiment separately:

```bash
bash scripts/train_duke_baseline_control_autodl.sh
bash scripts/train_duke_c2_l03_autodl.sh
```

Or run the complete validation sequence:

```bash
bash scripts/train_duke_c2_l03_validation_autodl.sh
```

The validation sequence runs `Duke-Baseline-Control` first and `Duke-C2-L03`
second. It uses `set -e`, so it stops immediately if either experiment fails.
Each successful experiment automatically updates the Duke validation section in
`EXPERIMENTS.md`.

Registered results (commit `6f49104`):

- Duke-Baseline-Control: Rank-1 86.7%, mAP 75.7%.
- Duke-C2-L03: Rank-1 88.4%, mAP 78.7%.

The corresponding AutoDL logs and checkpoints are not stored in this repository.
The seed and Rank-5/Rank-10 values remain to be recovered from the original
experiment environment.

## C2 Mechanism Analysis

### Cross-camera-positive batch coverage

This analysis uses only training metadata and reproduces the
`RandomIdentitySampler` grouping rule. It measures supervision opportunity, not
model performance. Use a new output directory for every run; the tool refuses
to overwrite a non-empty directory:

```bash
python tools/analyze_cross_camera_batch_coverage.py \
  --config-file configs/softmax_triplet_c2_baseline_control_autodl.yml \
  --compare-config-file configs/softmax_triplet_cross_camera_positive_lambda03_autodl.yml \
  --dataset market1501 \
  --data-root /root/autodl-tmp/datasets \
  --output-dir /root/autodl-tmp/analysis/market_batch_coverage \
  --seed 42 \
  --epochs 10
```

For Duke, replace the two configs and dataset:

```bash
python tools/analyze_cross_camera_batch_coverage.py \
  --config-file configs/softmax_triplet_c2_l03_duke_baseline_autodl.yml \
  --compare-config-file configs/softmax_triplet_c2_l03_duke_autodl.yml \
  --dataset dukemtmc \
  --data-root /root/autodl-tmp/datasets \
  --output-dir /root/autodl-tmp/analysis/duke_batch_coverage \
  --seed 42 \
  --epochs 10
```

### Baseline vs C2-L03 distance distributions

Run only after both aligned checkpoints are available:

```bash
python tools/analyze_distance_distributions.py \
  --baseline-config-file configs/softmax_triplet_c2_baseline_control_autodl.yml \
  --baseline-weight /path/to/baseline_checkpoint.pth \
  --c2-config-file configs/softmax_triplet_cross_camera_positive_lambda03_autodl.yml \
  --c2-weight /path/to/c2_l03_checkpoint.pth \
  --dataset market1501 \
  --data-root /root/autodl-tmp/datasets \
  --output-dir /root/autodl-tmp/analysis/market_distance_distribution \
  --seed 42 \
  --max-different-id-pairs 200000
```

The script uses the same query+gallery sample order and unordered pair indices
for both models. It analyzes L2-normalized BNNeck-after retrieval features using
squared Euclidean distance. It does not use re-ranking or camera-mean debiasing.
It also verifies the full resolved configs differ only in the C2 switch,
auxiliary lambda, and output path; requires C2 lambda=0.3/mode=mean; and refuses
to overwrite a non-empty output directory.
