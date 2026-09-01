# 当前正式 G2 与全部 Dynamic Gating 版本的统一分析

## 统计基线

本分析的唯一统计 baseline 是 `G2-global-local`，而不是任何版本的 Git
父分支：

- 分支：`codex/g2-global-local-gating`
- 正式训练 commit：`fa4e7f88f7ab645e9ba6b9a8e6cffdd9056b36c8`
- 后续实验记录 branch tip：`5a9a2a33e19f3d9dfc3afaf356dc8b9ee3f0f737`
- selected checkpoint SHA256：`49a766fb520cca5dfe9121f272994185db9fddee45709c9d61446c6781dc7d45`
- Seed：42；selected epoch：120

每个比较对象独立与此 baseline 比较：G1、G2-local-only、G2-without-z6、
G2-without-z4 和 G2-without-z2。工具拒绝将 comparator 彼此当作 baseline。

## 证据与归一化规则

`tools/analyze_all_gating_variants_vs_g2.py` 从各自分支的
`experiment_records/runs.csv` 精确查找指定 `experiment_id` 的 `formal/success`
记录；不会使用 inherited row、smoke 或未完成结果。它还必须用记录的 SHA256
在本机或 `--artifact-root` 下找到 source/resolved config、checkpoint、
`gating_samples.tsv` 和 `dynamic_gating_summary.json`，才会读取门控样本。

三路 G1/G2/G2-local-only 的代码契约为 native applied weight
`w=3p`，所以 `sum(w)=3`。两路消融的分支代码契约为 `w=p`，所以 active
weights 的和为 1。工具同时输出 native applied weights 和归一化 probabilities
`p`；跨两路与三路的坍缩解释以 normalized entropy 等 probability 指标为主。
结构删除的 scale 永远记为 `excluded` / `N/A`，绝不填零。

固定样本清单在读取 TSV 前从 Market1501 query 与 gallery 生成。稳定身份为
`split|relative_path|pid|camid`，排序键为
`SHA256("g2-baseline-all-variants-v1|" + stable_sample_key)`。若任意版本缺少
一个冻结样本，工具 fail-closed，拒绝生成跨版本门控图和统计。

## 输出

默认目录：`analysis_outputs/g2_baseline_all_variants/`。

- `evidence_inventory.{csv,md,json}`：六个版本的 formal/原始证据审计；
- `tables/performance_vs_g2.*`：全部差值相对同一 G2；
- `tables/gating_weight_statistics.*`：native weight 与 normalised probability；
- `tables/gating_collapse_comparison_vs_g2.*`：每个 comparator 相对 G2 的坍缩指标；
- `tables/paired_samples_g2_vs_*.csv`：图的机器可读配对源数据，仅在完整配对后生成；
- `manifests/fixed_candidate_samples.tsv`：冻结候选清单（提供 `--market-root` 时）；
- `manifests/image_type_annotations.tsv`：盲标注模板，Market 类别不是官方标签；
- `figures/g2_vs_*/`：完整配对后输出 PNG/PDF（300 DPI），否则只保留状态说明；
- `missing_formal_evidence_commands.md`：未完成正式实验的命令，绝不用 smoke 补数。

## AutoDL：审计与分析

GitHub URL 必须使用纯 URL：

```bash
cd /root/autodl-tmp
git clone https://github.com/lzhe036-web/BoT-reid-part-attention.git BoT-reid-g2-analysis
cd BoT-reid-g2-analysis
git fetch origin codex/g2-baseline-all-variants-analysis \
  codex/g2-global-local-gating exp/c2-l03-multi-granularity-dynamic-gating \
  codex/g2-local-only codex/g2-without-z6 codex/g2-without-z4 codex/g2-without-z2
git switch --track origin/codex/g2-baseline-all-variants-analysis
git rev-parse HEAD
test -z "$(git status --porcelain=v1 --untracked-files=all)" || { echo 'Dirty worktree'; exit 1; }
python tools/analyze_all_gating_variants_vs_g2.py \
  --repo-root . \
  --output-dir analysis_outputs/g2_baseline_all_variants \
  --artifact-root /root/autodl-tmp/g2_evidence_packages \
  --artifact-root /root/autodl-tmp/experiments
```

上述第一轮只做台账审计和性能表；它不会因缺少原始门控文件而伪造统计。若 Market
数据与所有原始门控 TSV 已经按 SHA256 归档，可在冻结候选清单后再次运行：

```bash
python tools/analyze_all_gating_variants_vs_g2.py \
  --repo-root . \
  --output-dir analysis_outputs/g2_baseline_all_variants \
  --artifact-root /root/autodl-tmp/g2_evidence_packages \
  --artifact-root /root/autodl-tmp/experiments \
  --market-root /root/autodl-tmp/datasets \
  --fixed-sample-limit 256
find analysis_outputs/g2_baseline_all_variants -type f -print0 | sort -z | xargs -0 sha256sum \
  > analysis_outputs/g2_baseline_all_variants/SHA256SUMS.external
```

如果 `analysis_status.json` 报告 `fixed_sample_coverage_incomplete`，必须先对
G2 及每个 comparator 用同一冻结 manifest 重新提取门控 TSV；不得截取各自旧 TSV
的交集，也不得更换 manifest 以适配某一版本。

## AutoDL：缺失正式运行

具体命令由工具写入
`analysis_outputs/g2_baseline_all_variants/missing_formal_evidence_commands.md`。
在每个对应分支上按顺序运行 smoke gate，再运行正式 runner；正式 runner 的自动
finalizer/registry 必须产出 `formal/success`、selected checkpoint、metrics 与原始
门控证据。然后 fetch 回分析分支并重新运行上节的审计命令。

当前文档和工具不启动训练、不修改任何正式模型分支，也不手工写入 Rank-1、mAP 或
门控权重。
