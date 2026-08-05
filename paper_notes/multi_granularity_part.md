# 子创新点 2-A：多粒度局部特征表示

## 方法定义

给定一次 ResNet50 backbone 前向得到的共享特征图

\[
F_i \in \mathbb{R}^{C \times H \times W}, \qquad C=2048,
\]

全局分支使用全局平均池化：

\[
g_i=\operatorname{GAP}(F_i) \in \mathbb{R}^{2048}.
\]

局部分支使用固定粒度集合 \(S=\{2,4,6\}\)。对尺度 \(s\) 的第 \(k\) 个水平区域，边界严格定义为

\[
\operatorname{start}_{s,k}=\left\lfloor\frac{Hk}{s}\right\rfloor,
\qquad
\operatorname{end}_{s,k}=\left\lfloor\frac{H(k+1)}{s}\right\rfloor.
\]

这些区域连续、不重叠且完整覆盖特征图高度。程序在 \(H<\max(S)\) 时直接报错，防止空条带。区域向量经过尺度专属非线性投影：

\[
p_i^{(s,k)}=\phi_s\!\left(\operatorname{GAP}
\left(F_i[:,\operatorname{start}_{s,k}:\operatorname{end}_{s,k},:]\right)\right),
\]

其中

\[
\phi_s=\operatorname{ReLU}\circ\operatorname{LayerNorm}\circ
\operatorname{Linear}_{2048\rightarrow256}.
\]

同一尺度的所有 part 共享一个 \(\phi_s\)，不同尺度的投影参数相互独立。第一版只做尺度内均值汇总：

\[
z_i^s=\frac{1}{s}\sum_{k=0}^{s-1}p_i^{(s,k)}
\in\mathbb{R}^{256}.
\]

最终在 BNNeck 之前拼接：

\[
f_i^{\mathrm{preBN}}=
[g_i;z_i^2;z_i^4;z_i^6]
\in\mathbb{R}^{2048+3\times256}
=\mathbb{R}^{2816}.
\]

训练时 classifier 接收 2816 维 BNNeck 输出，模型仍返回
`(cls_score, fused_pre_bn)`；因此原 ID loss、triplet loss 和 C2
跨摄像头正样本损失全部作用于新的融合描述符，未增加局部 classifier、局部损失、额外 triplet 或 hard mining。推理仍只返回一个 tensor，并保持
`TEST.NECK_FEAT=before/after` 的原语义。

## 数据流

```text
input
  └─ ResNet50 backbone（每次 forward 仅一次）→ F [B,2048,H,W]
       ├─ Global GAP                         → g   [B,2048]
       ├─ K=2：2 个条带 → shared φ_2 → mean → z_2 [B,256]
       ├─ K=4：4 个条带 → shared φ_4 → mean → z_4 [B,256]
       └─ K=6：6 个条带 → shared φ_6 → mean → z_6 [B,256]
  concat(g,z_2,z_4,z_6) → fused_pre_bn [B,2816]
  BNNeck(2816) → classifier(2816,num_classes)
```

## 配置项

```yaml
MODEL:
  PART_ATTENTION: False
  MULTI_GRANULARITY_PART: True
  MULTI_GRANULARITY_PART_SCALES: [2, 4, 6]
  MULTI_GRANULARITY_PART_DIM: 256
  MULTI_GRANULARITY_PART_AGGREGATION: "mean"
  MULTI_GRANULARITY_PART_FUSION: "concat"
```

全局默认值中 `MULTI_GRANULARITY_PART=False`，所以旧配置的结构、维度和接口不变。旧 `PART_ATTENTION` 与新开关互斥，同时开启会抛出
`ValueError`。本实验保持 `IF_WITH_CENTER: 'no'`。

## 与 Market1501 C2-L03 的公平对比

直接基线为
`configs/softmax_triplet_cross_camera_positive_lambda03_autodl.yml`。新配置除关闭旧固定 K=6 Part Attention、开启多粒度分支以及使用独立
`OUTPUT_DIR` 外，保持相同的 Market1501 数据、256×128 输入、增强、
`softmax_triplet` sampler、每身份 4 张、Adam、学习率 0.00035、120 epoch、batch size 64、seed 42、ImageNet backbone 初始化和 C2 设置：

- `CROSS_CAMERA_POSITIVE_ONLY=True`
- `CROSS_CAMERA_POSITIVE_LAMBDA=0.3`
- `CROSS_CAMERA_POSITIVE_MODE="mean"`

新结构不能严格加载旧 C2 完整 checkpoint；正式训练仍使用相同 ImageNet backbone 权重初始化，不通过忽略 missing/unexpected keys 绕过结构差异。

## 参数量与显存

测量命令：

```bash
python tools/profile_multi_granularity_part.py \
  --device auto --batch-size 64 \
  --input-height 256 --input-width 128 --dtype float32
```

本地实测条件：batch size 64，输入 256×128，FP32，NVIDIA GeForce RTX
4060 Laptop GPU。两种模型均随机初始化，不下载预训练权重；forward+backward
使用原 `make_loss()` 的 ID、triplet 和 C2-L03 损失在合成 PK batch 上测量。

| 指标 | 原始 C2-L03 | C2-L03 + K={2,4,6} | 增量 |
|---|---:|---:|---:|
| 最终特征维度 | 2048 | 2816 | +768 |
| 总参数量 | 25,052,225 | 27,202,880 | +2,150,655（+8.5847%） |
| 可训练参数量 | 25,050,177 | 27,200,064 | +2,149,887（+8.5823%） |
| forward 峰值 allocated | 535.819 MiB | 562.653 MiB | +26.834 MiB |
| forward 峰值 reserved | 658.000 MiB | 684.000 MiB | +26.000 MiB |
| forward+backward 峰值 allocated | 4142.479 MiB | 4179.506 MiB | +37.027 MiB |
| forward+backward 峰值 reserved | 4920.000 MiB | 4994.000 MiB | +74.000 MiB |

CUDA allocator 峰值会受 PyTorch/CUDA 版本和 GPU 环境影响，论文最终表格应在正式训练设备上用同一入口复测并报告环境；以上数值只代表所列本地实测条件。

## 当前局限与归因边界

尺度内均值汇总不保留 part 顺序，当前尚未实现局部软对齐、动态规划对齐、可学习 part 权重或跨尺度注意力。拼接还把描述符从 2048 维扩大到
2816 维，并同步增加 classifier、BNNeck 参数与显存。因此论文必须同时报告描述符维度、参数量和显存，不能把可能的全部性能提升直接归因于多粒度结构本身。
