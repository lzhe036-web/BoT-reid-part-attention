# AutoDL Run Notes

## Normalized Weighted Loss

分支名：

```bash
exp/normalized-weighted-loss
```

运行命令：

```bash
bash scripts/train_normalized_weighted_loss_autodl.sh
```

正式训练前记录信息：

```bash
bash scripts/record_experiment_info.sh configs/softmax_triplet_normalized_weighted_loss_autodl.yml
```

训练结束后整理结果：

```bash
python scripts/append_experiment_result.py --config configs/softmax_triplet_normalized_weighted_loss_autodl.yml --experiment-id NWL001 --note "Normalized weighted loss, AutoDL" --dry-run
```

确认无误后去掉 `--dry-run`。
