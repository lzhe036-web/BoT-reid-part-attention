# G2-without-z4 消融

- 基线：G2-local-only，`Gate([z2,z4,z6]) -> [w2,w4,w6]`。
- 新方案：G2-without-z4，`Gate([z2,z6]) -> [w2,w6]`。
- 唯一结构差异：z4 仍由局部特征头计算，但不进入 gate、没有门控权重、也不进入本方案的动态局部融合。项目原有的 `concat` 融合协议保持不变，因此检索描述符为 `concat(g, w2*z2, w6*z6)`，共 2560 维。
- 固定协议：Market1501、Seed=42、tau=1.0、C2-L03 损失和其他训练/评估配置均与 G2-local-only 一致。

正式 Rank-1、mAP 与门控统计只能由 AutoDL 真实运行、finalizer 和 recovery 自动生成。本分支不包含预填指标。

## AutoDL 执行顺序

```bash
cd /root/autodl-tmp
git clone --branch codex/g2-without-z4 --single-branch https://github.com/lzhe036-web/BoT-reid-part-attention.git BoT-reid-g2-without-z4
cd BoT-reid-g2-without-z4
EXPECTED_COMMIT=<在最终报告中替换为完整提交SHA>
test "$(git rev-parse HEAD)" = "${EXPECTED_COMMIT}" || { echo "Commit mismatch"; exit 1; }
test -z "$(git status --porcelain=v1 --untracked-files=all)" || { echo "Dirty worktree"; exit 1; }
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
python -c "from config import cfg; c=cfg.clone(); c.merge_from_file('configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_without_z4_autodl.yml'); print(c.MODEL.MULTI_GRANULARITY_GATING_INPUT, c.OUTPUT_DIR, c.SEED)"
python -c "from modeling.baseline import MultiGranularityDynamicGate; g=MultiGranularityDynamicGate(2048,3,gating_input='concat_z2_z6',local_feature_dim=256); print(g.controller.in_features,g.controller.out_features)"
```

先执行 smoke：

```bash
bash scripts/test_g2_without_z4_gating_1epoch_autodl.sh
```

再执行正式训练（脚本会保存 console log，随后自动 finalization 和 recovery）：

```bash
bash scripts/train_g2_without_z4_seed42_autodl.sh
tail -f /root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z4_tau1_seed42_market1501.console.log
```

若训练完成而登记中断，仅恢复已有原始证据：

```bash
python tools/recover_g2_without_z4_experiment.py --output-dir /root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z4_tau1_seed42_market1501 --console-log /root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z4_tau1_seed42_market1501.console.log
```

检查机器生成结果、两路权重与证据哈希：

```bash
python -c "import json; p='/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z4_tau1_seed42_market1501/g2_without_z4_formal_result.json'; r=json.load(open(p)); print(r['metrics'], r['selected_checkpoint'], r['gate_outputs'])"
grep -F 'C2-L03-MGDG-G2-WITHOUT-Z4-T1-S42' experiment_records/runs.csv
sha256sum /root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z4_tau1_seed42_market1501/g2_without_z4_formal_result.json
git add experiment_records EXPERIMENTS.md
git commit -m 'docs: record G2 without z4 formal evidence'
git push origin codex/g2-without-z4
```

不得手工编辑结果、门控统计或实验表格；它们必须由 recovery 从真实训练产物生成。
