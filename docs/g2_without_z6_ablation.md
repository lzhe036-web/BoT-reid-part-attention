# G2-without-z6 消融

- 基线：G2-local-only，`Gate([z2,z4,z6]) -> [w2,w4,w6]`。
- 新方案：G2-without-z6，`Gate([z2,z4]) -> [w2,w4]`。
- 唯一结构差异：z6 仍由局部特征头计算，但不进入 gate、没有权重、也不进入本方案的加权局部融合；因此最终检索 descriptor 为 2560 维，而非 G2-local-only 的 2816 维。
- 固定协议：Market1501、Seed=42、tau=1.0、C2-L03 损失和其余训练/评估配置与 G2-local-only 一致。

正式 Rank-1、mAP 与门控统计必须由 AutoDL 真实运行和自动 recovery 登记产生；本分支不包含任何预填指标。

## AutoDL 执行顺序

以下命令只在本分支推送后执行。`<commit>` 应替换为本分支发布时报告的
完整提交 SHA；不要用该文档预填任何结果。

```bash
cd /root/autodl-tmp
git clone --branch codex/g2-without-z6 --single-branch \
  https://github.com/lzhe036-web/BoT-reid-part-attention.git \
  BoT-reid-g2-without-z6
cd BoT-reid-g2-without-z6
git rev-parse HEAD
git status --porcelain=v1 --untracked-files=all
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from config import cfg; c=cfg.clone(); c.merge_from_file('configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_without_z6_autodl.yml'); print(c.MODEL.MULTI_GRANULARITY_GATING_INPUT, c.OUTPUT_DIR)"
```

先完成独立 smoke：

```bash
bash scripts/test_g2_without_z6_gating_1epoch_autodl.sh
```

确认 smoke 输出目录无误且 Git 工作树仍为空后，再启动正式训练：

```bash
bash scripts/train_g2_without_z6_seed42_autodl.sh
```

正式脚本会将控制台输出保存为
`/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z6_tau1_seed42_market1501.console.log`，
并在成功训练后运行 finalizer 和幂等 recovery。若需在中断后仅重新登记已完成
且未经修改的机器产物，可显式运行：

```bash
python tools/recover_g2_without_z6_experiment.py \
  --output-dir /root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z6_tau1_seed42_market1501 \
  --console-log /root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z6_tau1_seed42_market1501.console.log
```

登记完成后从机器生成的台账和正式结果检查指标、最佳 checkpoint 与两路门控证据：

```bash
python -c "import json; p='/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z6_tau1_seed42_market1501/g2_without_z6_formal_result.json'; r=json.load(open(p)); print(r['metrics'], r['selected_checkpoint'], r['gate_outputs'])"
grep -F 'C2-L03-MGDG-G2-WITHOUT-Z6-T1-S42' experiment_records/runs.csv
git status --short
git add experiment_records EXPERIMENTS.md
git commit -m 'docs: record G2 without z6 formal evidence'
git push origin codex/g2-without-z6
```

不得编辑 `runs.csv`、`EXPERIMENTS.md`、指标 JSON 或门控统计来补写结果；它们必须
由 recovery 从原始训练输出、checkpoint 和分析文件生成。
