# G2-D1：差异输入门控消融

直接基线为 `codex/g2-global-local-gating-tau0p5` 的提交 `d724a6536e4a819c5d2932412e90b7dea224041b`（G2-A）。本分支只改变控制器输入：

```text
[g, z2, z4, z6]  ->  [g, z2, z4, z6, abs(z4-z6)]
Linear(2816, 3)  ->  Linear(3072, 3)
```

其中 `z2/z4/z6` 均为已完成原始投影和均值聚合的 256 维局部描述子，`delta46=abs(z4-z6)` 未被门控、未 detach、未额外归一化，仅输入线性控制器。门控仍为 `p=softmax(logits/0.5)`、`w=3p`，最终检索描述子仍为 `concat(g,w2*z2,w4*z4,w6*z6)` 的 2816 维向量；差异项没有第四个权重，也不进入检索特征。

固定协议：Market1501、seed=42、K=[2,4,6]、120 epochs、`tau_g=0.5`、无 re-ranking。相对于原始 G2（tau=1.0）同时改变了温度和输入，因此差异项效果只能主要相对 G2-A 解释。

正式结果仅由 `scripts/train_g2_diff46_seed42_autodl.sh` 的全量训练、机器选择 checkpoint、finalizer、recovery 与 package 链路生成。本仓库不含任何 G2-D1 Rank-1、mAP、epoch 或门控数值。
