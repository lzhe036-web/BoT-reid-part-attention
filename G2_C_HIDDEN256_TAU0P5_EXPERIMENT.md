# G2-C：hidden=256 MLP 控制器，τg=0.5

本实验从原始 G2 提交 `5a9a2a33e19f3d9dfc3afaf356dc8b9ee3f0f737` 直接创建，固定 `SEED=42`。相对原始 G2 只有两项算法变化：

1. 控制器由 `Linear(2816, 3)` 改为 `Linear(2816, 256) -> ReLU -> Linear(256, 3)`；
2. `MODEL.MULTI_GRANULARITY_GATING_TAU` 由 `1.0` 改为 `0.5`。

门控输入仍为 `[g,z2,z4,z6]`，其中维度为 `2048+256+256+256=2816`；概率和权重仍为 `p=softmax(logits/0.5)`、`w=3p`，描述子仍为 `concat(g,w2*z2,w4*z4,w6*z6)`（2816 维）。没有引入 dropout、额外损失、静态残差或其他融合变动。

默认配置仍使用线性控制器，保留历史 G1/G2 的 `controller.weight`、`controller.bias` 键。G2-C 的 MLP 首层采用 Kaiming/ReLU 初始化、bias=0，输出层 weight/bias=0，所以初始 `p=(1/3,1/3,1/3)`、`w=(1,1,1)`；额外随机初始化被隔离，同时保留原线性控制器的随机数推进，以保证相同 seed 下共享模块初始化与原始 G2 一致。

## AutoDL 顺序

在一个干净的 G2-C 分支工作树中执行：

```bash
python tools/verify_g2_c_hidden256_tau0p5_protocol.py \
  --base-config configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_autodl.yml \
  --candidate-config configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_c_hidden256_tau0p5_autodl.yml

bash scripts/test_g2_c_hidden256_tau0p5_1epoch_autodl.sh
bash scripts/train_g2_c_hidden256_tau0p5_seed42_autodl.sh
bash scripts/export_g2_c_hidden256_tau0p5_result_autodl.sh
```

正式脚本拒绝脏工作树、错误分支、非直接 G2 父提交、复用输出目录/console log、缺失 Market 数据或 ImageNet 权重。训练、最终选择、门控分析、证据恢复都从机器文件生成；失败不会伪造指标。最终单版本包位于正式输出目录的 `g2_c_hidden256_tau0p5_result_package/`，下载包位于 `/root/autodl-tmp/exports/`。

门控样本统计是 SHA256 固定顺序选取的最多 256 张 query+gallery 图像；完整检索指标来自同一 selected checkpoint 的完整验证记录。MLP 无直接 `2816→3` 输入系数矩阵，因此“输入块系数范数”明确输出 `not_applicable`，不会将 MLP 层权重冒充为分支权重。

## 与 G1/G2/G2-A 的比较

正式训练完成后，只能使用经 SHA256 核验且采用同一冻结图片清单的 G1、原始 G2、G2-A（线性，τg=0.5）与 G2-C 逐样本门控数据做比较。不能将 256 样本、512 样本和完整 query 的统计静默混合。G2-C 单版本交付不会等待跨版本数据，但跨版本图和结论应另行生成并保留共同样本清单。
