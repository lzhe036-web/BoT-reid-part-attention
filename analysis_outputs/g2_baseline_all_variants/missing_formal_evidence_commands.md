# 缺少正式证据的 AutoDL 命令

这些命令不会使用 smoke 指标替代正式结果。执行前请确认分支 SHA 并保持工作树干净。

## G1

```bash
git fetch origin exp/c2-l03-multi-granularity-dynamic-gating
git switch exp/c2-l03-multi-granularity-dynamic-gating
test -z "$(git status --porcelain=v1 --untracked-files=all)" || { echo 'Dirty worktree'; exit 1; }
bash scripts/test_c2_l03_multi_granularity_dynamic_gating_1epoch.sh
bash scripts/train_c2_l03_multi_granularity_dynamic_gating_autodl.sh
```

## G2-local-only

```bash
git fetch origin codex/g2-local-only
git switch codex/g2-local-only
test -z "$(git status --porcelain=v1 --untracked-files=all)" || { echo 'Dirty worktree'; exit 1; }
bash scripts/test_g2_local_only_gating_1epoch_autodl.sh
bash scripts/train_g2_local_only_seed42_autodl.sh
```

## G2-without-z4

```bash
git fetch origin codex/g2-without-z4
git switch codex/g2-without-z4
test -z "$(git status --porcelain=v1 --untracked-files=all)" || { echo 'Dirty worktree'; exit 1; }
bash scripts/test_g2_without_z4_gating_1epoch_autodl.sh
bash scripts/train_g2_without_z4_seed42_autodl.sh
```

## G2-without-z2

```bash
git fetch origin codex/g2-without-z2
git switch codex/g2-without-z2
test -z "$(git status --porcelain=v1 --untracked-files=all)" || { echo 'Dirty worktree'; exit 1; }
bash scripts/test_g2_without_z2_gating_1epoch_autodl.sh
bash scripts/train_g2_without_z2_seed42_autodl.sh
```
