# G2 门控温度 τg=0.5：AutoDL 正式运行

本候选从正式 G2 记录提交 `5a9a2a33e19f3d9dfc3afaf356dc8b9ee3f0f737`
演进而来。唯一算法变量是
`MODEL.MULTI_GRANULARITY_GATING_TAU: 1.0 -> 0.5`；独立 `OUTPUT_DIR`
不属于算法变量。

## 正式训练

在 AutoDL 克隆或获取分支后，先在仓库根目录执行：

```bash
git branch --show-current
git status --porcelain=v1 --untracked-files=all
git merge-base --is-ancestor 5a9a2a33e19f3d9dfc3afaf356dc8b9ee3f0f737 HEAD
python tools/verify_g2_tau0p5_protocol.py \
  --baseline-config configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_autodl.yml \
  --candidate-config configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_tau0p5_autodl.yml
```

三项检查均成功后启动：

```bash
nohup bash scripts/train_g2_global_local_tau0p5_seed42_autodl.sh \
  > /root/autodl-tmp/g2_tau0p5_runner.log 2>&1 &
tail -f /root/autodl-tmp/g2_tau0p5_runner.log
```

该脚本拒绝非目标分支、脏工作树、缺少 Market1501/预训练权重、复用输出目录或复用 console log；成功后会依次完成训练、checkpoint 选择、门控分析和统一台账登记。它不会将 checkpoint 加入 Git。

## 冻结样本的三版本比较

训练登记成功后，令下列三处路径指向已存在的正式证据。`G1_G2_ANALYSIS_DIR`
必须是先前 G1-vs-G2 分析产生的目录，其中包含冻结的 256 query + 256 gallery 样本；不能用另一个随机样本集替代。

```bash
export G1_G2_ANALYSIS_DIR=/root/autodl-tmp/analysis_outputs/g1_vs_g2_dynamic_gating_seed42
export G1_RUN_MANIFEST=/root/autodl-tmp/BoT-reid-c2-l03-dynamic-gating-formal-v2/experiment_records/runs/<G1-run-id>/run_manifest.json
export G2_RUN_MANIFEST=/root/autodl-tmp/BoT-reid-g2-global-local/experiment_records/runs/<G2-run-id>/run_manifest.json
export TAU_RUN_MANIFEST=/root/autodl-tmp/BoT-reid-g2-global-local-gating-tau0p5/experiment_records/runs/<tau-run-id>/run_manifest.json

bash scripts/complete_g2_tau0p5_comparison_autodl.sh
```

它只对新 checkpoint 进行推理，固定样本列表、G1/G2 门控 TSV、三份 run manifest
都必须可验证。输出目录默认是：

```text
/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_global_local_tau0p5_seed42_market1501/g1_g2_tau0p5_fixed_sample_comparison/
```

其中包括 `retrieval_metrics.csv`、`gate_statistics.csv`、`per_sample_gating.csv`、
`config_resolved.yml`、`run_manifest.json`、分布 PNG/PDF、K4/K6 主导样本 PNG/PDF
及对应 CSV。若 K6 主导样本数量为零，脚本会输出空类说明和零行 CSV，而不会替换成其他样本。
