# G1 与当前正式 G2 的可复现 Dynamic Gating 分析

本分析固定比较两项正式实验：

- G1：`Gate(g) -> [w2,w4,w6]`，训练 commit
  `352127379e2c4bd7475fddb79fb8e3754cb8a2b8`，checkpoint SHA256
  `e57dd34a1b8d10ef6f544d55d4f627656815704ee8303a1c20e3ae735d4ca6aa`；
- G2：`Gate([g,z2,z4,z6]) -> [w2,w4,w6]`，训练 commit
  `fa4e7f88f7ab645e9ba6b9a8e6cffdd9056b36c8`，checkpoint SHA256
  `49a766fb520cca5dfe9121f272994185db9fddee45709c9d61446c6781dc7d45`。

`tools/analyze_g1_vs_g2_gating.py` 在推理前严格核验 formal/success、
Market1501、Seed=42、temperature=1、`[2,4,6]`、scaled-softmax、selected
epoch=120、gate input 与 checkpoint SHA256。任何不一致都会 fail-closed。
若历史 `run_manifest.json` 将 `dataset` 或 `selected_epoch` 写为 JSON `null`，
工具仅在已校验 SHA256 的 source/resolved config（数据集）或同一 manifest 的
`metrics`/`selected_checkpoint.epoch`（epoch）提供一致原始证据时恢复读取；
恢复来源会写入最终 `analysis_manifest.json`。若这些原始证据也缺失，仍拒绝分析。
同样地，历史 JSON 中仅允许将 `[2,4,6]` 编码为规范十进制字符串
`["2","4","6"]`；浮点、前导零、布尔值或顺序变化仍会被拒绝。

概率和实际融合权重分别记录：`p2+p4+p6=1`；G1、G2 均为 scaled-softmax，
故 `w=3p`、`w2+w4+w6=3`。训练末轮的 epoch JSONL 仅有聚合矩，未存储原始
训练样本权重时，median/q25/q75/bootstrap CI 必须为 `not_recorded`，不会推断。

## AutoDL 两阶段运行

请使用纯 URL：`https://github.com/lzhe036-web/BoT-reid-part-attention.git`。

第一阶段冻结候选集并生成不含任何门控信息的盲标注 contact sheet：

```bash
cd /root/autodl-tmp/BoT-reid-g2-global-local
git fetch origin '+refs/heads/codex/g1-vs-g2-gating-analysis:refs/remotes/origin/codex/g1-vs-g2-gating-analysis'
git switch codex/g1-vs-g2-gating-analysis 2>/dev/null || git switch -c codex/g1-vs-g2-gating-analysis origin/codex/g1-vs-g2-gating-analysis
git merge --ff-only origin/codex/g1-vs-g2-gating-analysis
git rev-parse HEAD
test -z "$(git status --porcelain=v1 --untracked-files=all)" || { echo 'dirty worktree'; exit 1; }

G1_RUN=/root/autodl-tmp/BoT-reid-c2-l03-dynamic-gating-formal-v2/experiment_records/runs/C2-L03-MGDG-T1-S42-20260815T124921Z-352127379e-914121de
G1_OUTPUT=/root/autodl-tmp/experiments/BoT/c2_l03_multi_granularity_dynamic_gating_tau1_seed42_market1501_r2
G1_CHECKPOINT="$G1_OUTPUT/resnet50_checkpoint_22320.pt"
G2_RUN=/root/autodl-tmp/BoT-reid-g2-global-local/experiment_records/runs/C2-L03-MGDG-G2-GL-T1-S42-fa4e7f88f7-49a766fb52
G2_OUTPUT=/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_global_local_tau1_seed42_market1501
G2_CHECKPOINT="$G2_OUTPUT/resnet50_checkpoint_22320.pt"
DATASET_ROOT=/root/autodl-tmp/datasets
OUT=/root/autodl-tmp/analysis_outputs/g1_vs_g2_dynamic_gating_seed42

test -f "$G1_RUN/run_manifest.json" && test -f "$G2_RUN/run_manifest.json"
test -f "$G1_CHECKPOINT" && test -f "$G2_CHECKPOINT"
test -d "$DATASET_ROOT/market1501/query" || test -d "$DATASET_ROOT/Market-1501-v15.09/query" || test -d "$DATASET_ROOT/query"
echo 'e57dd34a1b8d10ef6f544d55d4f627656815704ee8303a1c20e3ae735d4ca6aa  '"$G1_CHECKPOINT" | sha256sum -c -
echo '49a766fb520cca5dfe9121f272994185db9fddee45709c9d61446c6781dc7d45  '"$G2_CHECKPOINT" | sha256sum -c -
python -m unittest tests.test_analyze_g1_vs_g2_gating -v

python tools/analyze_g1_vs_g2_gating.py \
  --g1-run "$G1_RUN" --g1-output "$G1_OUTPUT" --g1-checkpoint "$G1_CHECKPOINT" \
  --g2-run "$G2_RUN" --g2-output "$G2_OUTPUT" --g2-checkpoint "$G2_CHECKPOINT" \
  --dataset-root "$DATASET_ROOT" --output-dir "$OUT" \
  --query-limit 256 --gallery-limit 256 --bootstrap-seed 42 --bootstrap-replicates 1000 \
  --prepare-annotations-only
```

只看 `$OUT/figures/blind_annotation/` 的图像，人工填写
`$OUT/manifests/image_type_annotations.tsv`。这是多标签盲标注，不能根据文件名、
权重、dominant K 或检索结果改写类别。每个将要解释的类别必须有至少 5 张固定样本。

第二阶段在已完成盲标注后抽取 G1/G2 的同一固定样本门控、生成统计和图片，并打包：

```bash
ARCHIVE=/root/autodl-tmp/g1_vs_g2_dynamic_gating_seed42.tar.gz
python tools/analyze_g1_vs_g2_gating.py \
  --g1-run "$G1_RUN" --g1-output "$G1_OUTPUT" --g1-checkpoint "$G1_CHECKPOINT" \
  --g2-run "$G2_RUN" --g2-output "$G2_OUTPUT" --g2-checkpoint "$G2_CHECKPOINT" \
  --dataset-root "$DATASET_ROOT" --output-dir "$OUT" \
  --query-limit 256 --gallery-limit 256 --bootstrap-seed 42 --bootstrap-replicates 1000 \
  --resume --archive-path "$ARCHIVE"

sha256sum "$ARCHIVE"
du -h "$ARCHIVE"
find "$OUT" -maxdepth 3 -type f | sort
```

压缩包只包含分析输出、派生图与 manifests；不包含 Market1501 原图、checkpoint、
训练缓存或训练日志副本。若字段缺失或类别样本不足，工具会中止或明确写
`not_recorded`/`insufficient_blind_annotated_fixed_samples`，不制造结论。
