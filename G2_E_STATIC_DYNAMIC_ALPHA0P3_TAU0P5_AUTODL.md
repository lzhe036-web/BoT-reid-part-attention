# G2-E-alpha0p3-tau0p5

直接基线为 `codex/g2-global-local-gating-tau0p5` 的
`d724a6536e4a819c5d2932412e90b7dea224041b`。本分支只将 G2-A 的纯动态局部融合改为：

```text
z_final = concat(g, (1 + 0.3*w2)z2, (1 + 0.3*w4)z4, (1 + 0.3*w6)z6)
p = softmax(Linear(concat(g,z2,z4,z6)) / 0.5)
w = 3p
```

`g` 仍为 2048 维，`z2/z4/z6` 各为 256 维，检索描述子仍为 2816 维。`p`、`w` 与最终系数 `c=1+0.3w` 会分别记录，不能混用。该分支不含 hidden-256 控制器、delta46 输入或 α=0.5 变体。

## AutoDL：获取与验证

```bash
set -euo pipefail
cd /root/autodl-tmp
git clone --branch codex/g2-e-static-dynamic-alpha0p3-tau0p5 --single-branch \
  https://github.com/lzhe036-web/BoT-reid-part-attention.git BoT-reid-g2-e-alpha0p3
cd BoT-reid-g2-e-alpha0p3
git branch --show-current
git rev-parse HEAD
test "$(git merge-base HEAD d724a6536e4a819c5d2932412e90b7dea224041b)" = d724a6536e4a819c5d2932412e90b7dea224041b
test -z "$(git status --porcelain=v1 --untracked-files=all)"
export OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
python -m unittest tests.test_g2_e_static_dynamic_alpha0p3_tau0p5 \
  tests.test_g2_e_static_dynamic_evidence_tools \
  tests.test_g2_e_static_dynamic_autodl_scripts \
  tests.test_compare_g1_g2_g2a_g2e_gating -v
python tools/verify_g2_e_static_dynamic_alpha0p3_tau0p5_protocol.py \
  --baseline-config configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_tau0p5_autodl.yml \
  --candidate-config configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_e_static_dynamic_alpha0p3_tau0p5_autodl.yml
```

若 GitHub 网络不可用，使用本分支发布时提供的 Git bundle；不要把网页 Markdown 链接（如 `[https://…](https://…)`）粘进 `git clone`。

## AutoDL：smoke、正式训练与封包

```bash
# 先运行一次真实一 epoch smoke；输出目录不得已有内容。
bash scripts/test_g2_e_static_dynamic_alpha0p3_tau0p5_1epoch_autodl.sh

# 正式训练：120 epochs、Market1501、seed=42、tau_g=.5。此脚本自动 finalization、
# G2-E 独立封包、SHA256 归档和统一实验台账恢复登记。
bash scripts/train_g2_e_static_dynamic_alpha0p3_tau0p5_seed42_autodl.sh
```

正式输出目录：

```text
/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_e_static_dynamic_alpha0p3_tau0p5_seed42_market1501
```

如训练已经完成但压缩包尚未生成，只能对存在且通过哈希校验的独立证据包执行：

```bash
OUT=/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_e_static_dynamic_alpha0p3_tau0p5_seed42_market1501
python tools/package_g2_e_static_dynamic_alpha0p3_tau0p5_result.py --output-dir "$OUT"
bash scripts/export_g2_e_static_dynamic_alpha0p3_tau0p5_result_autodl.sh "$OUT"
PKG="$OUT/g2_e_static_dynamic_alpha0p3_tau0p5_result_package"
(cd "$PKG" && sha256sum -c SHA256SUMS.txt)
```

该包只在已存在的 `g2_e...formal_result.json`、checkpoint、hash-bound gating evidence 均通过后创建；不会补写或猜测指标。

## 四版本共同样本比较（训练后）

为避免混用不同 256 图像集合，比较脚本强制四个 TSV 的 `stable_sample_key` **集合与顺序完全一致**。先把已核验的 G1、原始 G2、G2-A 与本次 G2-E 的逐样本 gate TSV 和对应 formal result JSON 填入变量：

```bash
python tools/compare_g1_g2_g2a_g2e_gating.py \
  --g1-samples "$G1_SAMPLES" --g1-result "$G1_RESULT" \
  --g2-samples "$G2_SAMPLES" --g2-result "$G2_RESULT" \
  --g2a-samples "$G2A_SAMPLES" --g2a-result "$G2A_RESULT" \
  --g2e-samples "$OUT/g2_e_static_dynamic_alpha0p3_tau0p5_analysis/per_sample_gating.tsv" \
  --g2e-result "$OUT/g2_e_static_dynamic_alpha0p3_tau0p5_formal_result.json" \
  --output-dir /root/autodl-tmp/analysis_outputs/g1_g2_g2a_g2e_shared_samples
```

该工具输出四版本 `w=3p` 的公共分布、G2-E 独立的 `c` 分布、严格样本清单、以及相对 G2-A 的百分点变化。K4/K6 样本图和其图片已经由单版本 G2-E 包生成；若某类实际为 0，图中会明确写为 0，绝不拿其它类替代。
