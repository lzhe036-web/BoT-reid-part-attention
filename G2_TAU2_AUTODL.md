# G2 门控温度 τg=2.0：AutoDL 正式运行与交付包

本分支直接继承 `codex/g2-global-local-gating-tau0p5` 的提交
`d724a6536e4a819c5d2932412e90b7dea224041b`。唯一算法变量为
`MODEL.MULTI_GRANULARITY_GATING_TAU: 0.5 -> 2.0`；独立输出路径、run ID 与
证据文件名不属于算法变量。

## 正式训练前的严格核验

```bash
git branch --show-current
git status --porcelain=v1 --untracked-files=all
git merge-base --is-ancestor d724a6536e4a819c5d2932412e90b7dea224041b HEAD
python tools/verify_g2_tau0p5_protocol.py \
  --baseline-config configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_tau0p5_autodl.yml \
  --candidate-config configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_tau2_autodl.yml \
  --expected-baseline-tau 0.5 --expected-candidate-tau 2.0
```

仅当上列命令成功、Market1501 与 ImageNet 预训练权重均存在时，才可启动：

```bash
nohup bash scripts/train_g2_global_local_tau2_seed42_autodl.sh \
  > /root/autodl-tmp/g2_tau2_runner.log 2>&1 &
tail -f /root/autodl-tmp/g2_tau2_runner.log
```

该 runner 拒绝错误分支、脏工作树、错误父系、缺失数据/权重和已有输出。训练完成后，
finalizer 和 recovery 会自动生成且登记 source/resolved config、SHA256、训练/console log、
checkpoint、环境、Rank-1/5/10、mAP、选定 epoch 和三路门控证据。若任何收尾步骤失败，
checkpoint 会被保留，runner 以非零状态退出。

## 版本必须提交的五类结果

正式 τg=2.0 run 已登记后，执行下面的后处理。它绝不重新训练；仅使用已经冻结的
G1/G2 样本列表和三份经核验的已有门控 TSV，在**相同图像**上提取实际 τg=2.0 checkpoint
的门控值。

```bash
export G1_G2_ANALYSIS_DIR=/root/autodl-tmp/analysis_outputs/g1_vs_g2_dynamic_gating_seed42
export TAU0P5_COMPARISON_DIR=/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_global_local_tau0p5_seed42_market1501/g1_g2_tau0p5_fixed_sample_comparison
export G1_RUN_MANIFEST=/root/autodl-tmp/BoT-reid-c2-l03-dynamic-gating-formal-v2/experiment_records/runs/<G1-run-id>/run_manifest.json
export G2_RUN_MANIFEST=/root/autodl-tmp/BoT-reid-g2-global-local/experiment_records/runs/<G2-run-id>/run_manifest.json
export TAU0P5_RUN_MANIFEST=/root/autodl-tmp/BoT-reid-g2-global-local-gating-tau0p5/experiment_records/runs/<tau0p5-run-id>/run_manifest.json
export TAU2_RUN_MANIFEST=/root/autodl-tmp/BoT-reid-g2-global-local-tau2/experiment_records/runs/<tau2-run-id>/run_manifest.json

bash scripts/complete_g2_tau2_comparison_autodl.sh
```

输出目录为：

```text
/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_global_local_tau2_seed42_market1501/g1_g2_tau0p5_tau2_fixed_sample_comparison/
```

其内容由程序生成，满足方案筛选所需的所有结果项：

| 类别 | 机器生成文件 |
| --- | --- |
| 检索指标 | `retrieval_metrics.csv`：G1、G2 τ=1.0、τ=0.5、τ=2.0 的 Rank-1、Rank-5、Rank-10、mAP；τ=2.0 相对直接 τ=0.5 基线的百分点差值。 |
| 门控统计 | `gate_statistics.csv`：每版本的 p2/p4/p6 均值及 K2/K4/K6 dominant ratio。 |
| 分布图 | `figures/gating_probability_distributions.png` 和 `.pdf`：四版本同一冻结样本、同一 `[0,1]` 横轴、同一分箱的 p 值直方图。 |
| 样本图 | `figures/k4_dominant_samples.*`、`figures/k6_dominant_samples.*`：τ=2.0 实际 K4/K6 主导样本各至多六个，同时列出同一图像在 G1/G2/τ=.5 的 dominant K。 |
| 可复核原始统计 | `per_sample_gating.csv`、`g2_tau2_fixed_gating_samples.tsv`、`run_manifest.json`、`config_resolved.yml` 和 `README.md`。 |

概率使用 `p2+p4+p6=1`，而 scaled-softmax 的实际融合权重为 `w=3p`；图表明确画的是
概率 `p`。K4 或 K6 若少于 4 张或为零，脚本只输出实际数量（及零行 CSV/空类图），不会拿
其他样本凑数。未经 formal run 和证据核验，目录不会出现预填的指标或门控结论。
