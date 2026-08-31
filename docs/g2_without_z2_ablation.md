# G2-without-z2 消融实验

- 基线：G2-local-only，`Gate([z2,z4,z6]) -> [w2,w4,w6]`。
- 新方案：G2-without-z2，`Gate([z4,z6]) -> [w4,w6]`。
- 唯一结构差异：z2 仍由多粒度局部特征头计算，但不进入控制器、不获得门控权重、也不参与动态局部融合。为保持 G2-local-only 的既有 `concat` 描述符协议，检索描述符为 `concat(g, w4*z4, w6*z6)`，共 2560 维；其中动态局部融合只含 `w4*z4` 和 `w6*z6`。
- 固定协议：Market1501、Seed=42、tau=1.0、C2-L03 损失及其余训练/评估设置均与 G2-local-only 相同。

正式 Rank-1、mAP、selected epoch 和门控统计只能由 AutoDL 真实训练、finalizer 与 recovery 自动生成。本分支不预填任何指标或门控数值。

## AutoDL 执行顺序

将 `<EXPECTED_COMMIT>` 替换为本分支最终提交的完整 SHA；最终交付报告会提供该值。

```bash
cd /root/autodl-tmp
git clone --branch codex/g2-without-z2 --single-branch https://github.com/lzhe036-web/BoT-reid-part-attention.git BoT-reid-g2-without-z2
cd BoT-reid-g2-without-z2
EXPECTED_COMMIT=<EXPECTED_COMMIT>
test "$(git branch --show-current)" = "codex/g2-without-z2" || { echo "Branch mismatch"; exit 1; }
test "$(git rev-parse HEAD)" = "${EXPECTED_COMMIT}" || { echo "Commit mismatch"; exit 1; }
test -z "$(git status --porcelain=v1 --untracked-files=all)" || { echo "Dirty worktree"; exit 1; }
python -c "import sys,torch; print(sys.version); print('torch=',torch.__version__,'cuda=',torch.version.cuda,'gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
python -c "from config import cfg; c=cfg.clone(); c.merge_from_file('configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_without_z2_autodl.yml'); print(c.MODEL.MULTI_GRANULARITY_GATING_INPUT,c.OUTPUT_DIR,c.SEED)"
python -c "import torch; from modeling.baseline import MultiGranularityDynamicGate; gate=MultiGranularityDynamicGate(2048,3,gating_input='concat_z4_z6',local_feature_dim=256); g=torch.zeros(2,2048); z=(torch.zeros(2,256),torch.zeros(2,256),torch.zeros(2,256)); _,p,w=gate(g,z); print('input=',gate.controller_input(g,z).shape,'output=',p.shape,'sum=',p.sum(1),'weights=',w.shape)"
```

先执行一轮 smoke（它使用独立目录且不覆盖正式输出）：

```bash
bash scripts/test_g2_without_z2_gating_1epoch_autodl.sh
```

再执行 Seed=42 的正式训练。脚本会保存完整 console log，并在训练返回 0 后依次执行 finalizer 和 recovery：

```bash
bash scripts/train_g2_without_z2_seed42_autodl.sh
tail -f /root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z2_tau1_seed42_market1501.console.log
```

若训练已成功完成、但收尾或登记中断，只恢复已有原始证据，不重新训练：

```bash
python tools/recover_g2_without_z2_experiment.py \
  --output-dir /root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z2_tau1_seed42_market1501 \
  --console-log /root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z2_tau1_seed42_market1501.console.log
```

检查机器生成的结果、两路门控证据和哈希：

```bash
python -c "import json; p='/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z2_tau1_seed42_market1501/g2_without_z2_formal_result.json'; r=json.load(open(p)); print(r['metrics'],r['selected_checkpoint'],r['gate_outputs'])"
python -c "import csv; p='/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z2_tau1_seed42_market1501/g2_without_z2_gating_analysis/g2_without_z2_gate_test_weight_summary.csv'; print(list(csv.DictReader(open(p))))"
grep -F 'C2-L03-MGDG-G2-WITHOUT-Z2-T1-S42' experiment_records/runs.csv
sha256sum /root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z2_tau1_seed42_market1501/g2_without_z2_formal_result.json \
  /root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z2_tau1_seed42_market1501/config_resolved.yml \
  /root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z2_tau1_seed42_market1501/resnet50_checkpoint_*.pt
```

仅在 recovery 已成功生成机器台账后，提交并推送该台账：

```bash
git add experiment_records EXPERIMENTS.md
git commit -m 'docs: record G2 without z2 formal evidence'
git push origin codex/g2-without-z2
```

不得手工编辑指标、门控统计或实验表格；所有正式记录必须由 recovery 从训练期原始文件生成。
