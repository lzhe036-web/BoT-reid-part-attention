# G2 E1：静态拼接 + 门控残差（α=0.5）

本分支直接基于原始 G2 正式提交
`5a9a2a33e19f3d9dfc3afaf356dc8b9ee3f0f737`，而非任意 τ 变体。唯一算法变化是：

```text
原始 G2: concat(g, w2*z2, w4*z4, w6*z6)
E1:      concat(g, (1+0.5*w2)*z2, (1+0.5*w4)*z4, (1+0.5*w6)*z6)
```

`alpha=0.5` 固定且不学习；`tau_g=1.0` 保持不变。输出仍为 2816 维，接入原有 BNNeck、
分类头和损失路径。

## AutoDL：smoke 先行

```bash
git branch --show-current
git status --porcelain=v1 --untracked-files=all
git merge-base --is-ancestor 5a9a2a33e19f3d9dfc3afaf356dc8b9ee3f0f737 HEAD

python tools/verify_g2_e1_static_dynamic_protocol.py \
  --baseline-config configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_autodl.yml \
  --candidate-config configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_e1_static_dynamic_alpha0p5_autodl.yml

bash scripts/test_g2_e1_static_dynamic_alpha0p5_1epoch_autodl.sh
```

该一轮 smoke 使用真实训练数据、评估、checkpoint 和逐样本门控导出，但不是正式结果。
成功后再启动正式训练：

```bash
nohup bash scripts/train_g2_e1_static_dynamic_alpha0p5_seed42_autodl.sh \
  > /root/autodl-tmp/g2_e1_static_dynamic_alpha0p5_runner.log 2>&1 &
tail -f /root/autodl-tmp/g2_e1_static_dynamic_alpha0p5_runner.log
```

正式 runner 强制 clean worktree、原始 G2 父系、Market1501、ImageNet 权重、smoke evidence
和独立输出路径；完成后自动保存配置、日志、环境、checkpoint SHA256、指标和 E1 证据，并登记
统一台账。不会将 checkpoint 加入 Git。

## 正式三版本完整 query 对比

在 G1、原始 G2、E1 三个正式 checkpoint 均可用后设置实际路径。工具会在同一完整
Market1501 query 集重新提取全部三个模型的门控值；不复用不同随机候选集。

```bash
export G1_CONFIG=/root/autodl-tmp/BoT-reid-c2-l03-dynamic-gating-formal-v2/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_autodl.yml
export G1_CHECKPOINT=/root/autodl-tmp/experiments/BoT/<G1-output>/resnet50_checkpoint_<iteration>.pt
export G1_RESULT=/root/autodl-tmp/BoT-reid-c2-l03-dynamic-gating-formal-v2/experiment_records/runs/<G1-run-id>/run_manifest.json

export G2_CONFIG=/root/autodl-tmp/BoT-reid-g2-global-local/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_autodl.yml
export G2_CHECKPOINT=/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_global_local_tau1_seed42_market1501/resnet50_checkpoint_<iteration>.pt
export G2_RESULT=/root/autodl-tmp/BoT-reid-g2-global-local/experiment_records/runs/<G2-run-id>/run_manifest.json

export E1_RUN_MANIFEST="$(grep -rl '"experiment_id": "C2-L03-MGDG-G2-E1-STATIC-DYNAMIC-A05-T1-S42"' experiment_records/runs/*/run_manifest.json | head -n 1)"
test -n "$E1_RUN_MANIFEST" && test -f "$E1_RUN_MANIFEST"

bash scripts/complete_g2_e1_static_dynamic_comparison_autodl.sh
```

输出目录：

```text
/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_e1_static_dynamic_alpha0p5_tau1_seed42_market1501/g1_g2_e1_static_dynamic_complete_query_comparison/
```

其中由程序生成 `retrieval_metrics.csv`、`gate_statistics.csv`、
`per_sample_gating.csv`、`sample_selection.csv`、`config_resolved.yml`、`run_manifest.json`、
相对 G2 的代码/配置差异、三版本概率分布 PNG/PDF、E1 系数分布 PNG/PDF，以及 K4/K6
主导样本 PNG/PDF/CSV。`p`、`w=3p`、`residual=0.5w` 和 `c=1+0.5w` 分列保存，c 不会被
标作概率；零样本类别会保留为空类观测。
