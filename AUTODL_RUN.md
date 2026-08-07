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

## C2-L03 Independent Formal Experiment

Run the single C2-L03 formal experiment only from branch `C2L03`. Before
running, either switch an existing clone to that branch:

```bash
git switch C2L03
bash scripts/train_cross_camera_positive_lambda03_autodl.sh
```

Or clone the formal branch directly:

```bash
git clone --branch C2L03 --single-branch https://github.com/lzhe036-web/BoT-reid-part-attention.git BoT-reid-c2-l03
cd BoT-reid-c2-l03
bash scripts/train_cross_camera_positive_lambda03_autodl.sh
```

Each invocation starts exactly one training task with
`configs/softmax_triplet_cross_camera_positive_lambda03_autodl.yml`,
`MODEL.CROSS_CAMERA_POSITIVE_LAMBDA=0.3`, and the independent output directory
`/root/autodl-tmp/experiments/BoT/C2L03_market1501`.

The shell script calls `tools/run_experiment.py`, which performs a clean Git /
branch preflight and then starts the unchanged `tools/train.py` exactly once.
After the training subprocess exits with code 0, the bypass finalizer:

1. parses `log.txt` and derives `validation_history.jsonl`;
2. selects the best validation epoch using Rank-1, then mAP;
3. maps `resnet50_checkpoint_<global_iteration>.pt` through the logged
   `iterations_per_epoch` instead of treating the suffix as an epoch;
4. hashes the log, selected checkpoint, configs, and generated artifacts;
5. runs independent distance, controlled anchor-coverage, and efficiency
   analyses without changing the training process;
6. updates CSV sources, regenerates Markdown tables, and updates only the
   generated section of `EXPERIMENTS.md` by `run_id`.

The run is written to `experiment_records/runs/<run_id>/`. Missing or
conflicting strong evidence leaves it `failed` or `incomplete` and prevents a
success row. In particular, this legacy branch does not emit an applied seed;
the recorder reports `missing_evidence` rather than injecting a seed or copying
a historical value. Historical lambda-sensitivity rows are never overwritten.
