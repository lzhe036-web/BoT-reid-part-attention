# G2-E：α=0.1 / 0.5 独立对照实验

两个分支都直接从 `codex/g2-e-static-dynamic-alpha0p3-tau0p5` 的 `63761021a40693694f037d850066deb2237a5c41` 创建。唯一算法配置差异为 `MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA`；输出路径另行隔离。τg=0.5、Linear(2816,3)、零初始化、静态拼接加门控残差、2816 维描述子及训练协议均保持基线。CROSS_CAMERA_POSITIVE_LAMBDA 仍为 0.3。

```text
p = softmax(controller(concat(g,z2,z4,z6))/0.5)
w = 3*p
z_final = concat(g,(1+alpha*w2)*z2,(1+alpha*w4)*z4,(1+alpha*w6)*z6)
```

本次代码交付没有运行 AutoDL 正式训练。新版本的正式检索指标、checkpoint 与结果压缩包须由下面的训练产生；测试中使用的合成证据不属于实验结果。现有 α=0.3 历史结果已经校验，其实际训练 SHA 为 `dbdcdf524137b93d0cdbe6216d6851953308c23f`；`6376102` 是本次代码基线，不冒充历史训练提交。

## 准备两个独立目录

将交付文件 `g2_e_alpha_sweep_delivery.zip` 上传到 `/root/autodl-tmp`。使用 α=0.3 实验原有的 Python/CUDA 环境、Market1501 数据及 ImageNet 预训练文件。不要从 α=0.3 训练 checkpoint 微调。

```bash
cd /root/autodl-tmp
unzip -n g2_e_alpha_sweep_delivery.zip
cd g2_e_alpha_sweep_delivery
bash prepare_autodl.sh
```

准备脚本验证 bundle 校验和、分支提交及目标目录，分别 clone 两个分支。目标目录已存在时只接受同一提交且工作区干净的目录，不覆盖。

检查原有环境和数据路径：

```bash
python -c 'import torch, torchvision, ignite, yacs, numpy, matplotlib, PIL; print("torch",torch.__version__,"CUDA",torch.version.cuda,"available",torch.cuda.is_available()); assert torch.cuda.is_available()'
test -d /root/autodl-tmp/datasets
test -f /root/autodl-tmp/pretrained/resnet50-19c8e357.pth
```

历史 α=0.3 环境记录为 Python 3.10.8、torch 2.1.2+cu118、torchvision 0.16.2+cu118、ignite 0.4.13、numpy 1.26.4。优先复用这个环境，并保留每次运行自动生成的实际环境记录。

## 同一 GPU 顺序运行

```bash
cd /root/autodl-tmp/g2_e_alpha_sweep_delivery
bash run_all_autodl.sh
```

脚本依次执行 α=0.1 的 smoke 和正式训练，再执行 α=0.5 的 smoke 和正式训练，最后自动生成六版本对照。任一步失败即停下并报告；不要重新运行训练来修复后处理。两个独立 AutoDL 实例也可以分别运行下面各自的两条命令。

```bash
cd /root/autodl-tmp/BoT-reid-g2-e-static-dynamic-alpha0p1-tau0p5
bash scripts/test_g2_e_static_dynamic_alpha0p1_tau0p5_1epoch_autodl.sh
bash scripts/train_g2_e_static_dynamic_alpha0p1_tau0p5_seed42_autodl.sh
```

```bash
cd /root/autodl-tmp/BoT-reid-g2-e-static-dynamic-alpha0p5-tau0p5
bash scripts/test_g2_e_static_dynamic_alpha0p5_tau0p5_1epoch_autodl.sh
bash scripts/train_g2_e_static_dynamic_alpha0p5_tau0p5_seed42_autodl.sh
```

正式训练为 seed=42、120 epochs，沿用基线所有训练参数。每个训练入口检查完整解析配置、工作区身份和独立 smoke 的配置、checkpoint、32 个样本及训练提交。smoke 不写入正式 registry。正式 checkpoint 选择：最高 Rank-1 → 最高 mAP → 最早 epoch；不强制最后一轮。

## 输出及自动登记

正式输出分别为：

```text
/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_e_static_dynamic_alpha0p1_tau0p5_seed42_market1501
/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_e_static_dynamic_alpha0p5_tau0p5_seed42_market1501
```

smoke 目录在对应路径后增加 `_smoke`；console log 在正式路径后增加 `.console.log`。两个 YAML 和脚本都位于各自仓库的 configs/scripts 中，名称包含对应 alpha。

实际顺序：训练及全量评估 → checkpoint 选择及原始门控证据 → `experiment_records/runs.csv`、`experiment_records/tables/main_results.csv`、对应 run_manifest、仓库根目录 **EXPERIMENTS.md** → 绘图、打包和压缩。不会另外创建 experiment.md。

EXPERIMENTS.md 保留历史内容，正式表与新增的固定样本统计区分别标明训练期统计和 query/gallery 固定样本统计。记录分支、直接基线、实际训练 SHA、alpha/tau、维度与公式、选中 epoch/checkpoint/SHA256、四项检索指标、p/w/alpha*w/c 均值、dominant ratio、样本范围、原始来源与图表路径、打包状态和下载位置。

experiment_id 为 `C2-L03-MGDG-G2-E-SD-A01-T0P5-S42` 或 `C2-L03-MGDG-G2-E-SD-A05-T0P5-S42`；run_id 由 experiment_id、实际训练 SHA 前 10 位和 checkpoint SHA 前 10 位组成。重复补录不增加行，两个 alpha 不覆盖。更新使用原子文件写入，登记表沿用现有事务写入。

checkpoint 使用独立 `.metadata.json` 记录训练身份和包含 alpha/tau/融合方式的特征签名；不会改变 state_dict 结构。正式恢复和分析必须验证该身份；相同形状不能绕过校验。

## 不重训补录、分析、重新打包

以下循环在对应仓库执行两个独立补录；每个 output 必须已经完成正式训练。

```bash
for token in 0p1 0p5; do
  alpha="${token/p/.}"
  repo="/root/autodl-tmp/BoT-reid-g2-e-static-dynamic-alpha${token}-tau0p5"
  stem="g2_e_static_dynamic_alpha${token}_tau0p5"
  out="/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_${stem}_seed42_market1501"
  (
    set -euo pipefail
    cd "$repo"
    python tools/finish_g2_e_alpha_experiment.py --alpha "$alpha" \
      --config-file "configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_${stem}_autodl.yml" \
      --output-dir "$out" --console-log "${out}.console.log" --register-only
  ) || exit "$?"
done
```

`--register-only` 会先进行必要的选中 checkpoint 原始数据导出；已经有效的 formal result 会复用。独立 `tools/recover_g2_e_static_dynamic_alpha0p1_tau0p5_experiment.py` 和 `...alpha0p5...` 也可使用相同 config/output/console 参数对已有正式证据补录。

完成补录后重新绘图、打包、压缩：

```bash
cd /root/autodl-tmp/BoT-reid-g2-e-static-dynamic-alpha0p1-tau0p5
bash scripts/export_g2_e_static_dynamic_alpha0p1_tau0p5_result_autodl.sh

cd /root/autodl-tmp/BoT-reid-g2-e-static-dynamic-alpha0p5-tau0p5
bash scripts/export_g2_e_static_dynamic_alpha0p5_tau0p5_result_autodl.sh
```

每次重新打包都创建时间戳与随机后缀的新目录。绘图/压缩失败不会删除正式表中的指标；对应 output 的 `delivery_status.json` 和 EXPERIMENTS.md 会记录失败阶段。绘图依赖延迟加载，缺少绘图库也不阻止已通过校验的指标登记。

如需额外原始数据分析，不覆盖正式证据目录，可使用以下独立入口；`weight` 应从 formal_result 的 selected_checkpoint.path 获取：

```bash
python tools/analyze_g2_e_static_dynamic_alpha0p1_tau0p5.py --help
python tools/analyze_g2_e_static_dynamic_alpha0p5_tau0p5.py --help
```

以上两个分析命令分别属于各自分支。日常修复直接使用 finish/export，它们自动绑定选中 checkpoint。不要手改 formal_result、原始统计或 checkpoint 身份文件。

## 下载与离线重绘

每版压缩包位于 `/root/autodl-tmp/exports/g2_e_alpha0p1_tau0p5_<时间戳>-<标识>.tar.gz` 或 alpha0p5 对应名称，并带同名 `.sha256`。成功时终端、EXPERIMENTS.md 和 delivery_status.json 均输出确切路径。

包内包括四项全量检索指标、p/w/residual/c 原始数据和统计、样本清单、配置/训练身份、checkpoint 签名、PNG/PDF、K4/K6 主导展示图片、脚本、相对 alpha0p3 的代码差异、EXPERIMENTS.md 登记快照、registry 证据和 SHA256SUMS。包内 EXPERIMENTS.md 是封包时登记快照；封包成功后的路径与压缩包哈希还写回原仓库台账，避免压缩包自引用哈希。

```bash
cd /root/autodl-tmp/exports
sha256sum -c <实际包名>.tar.gz.sha256
```

下载并解压后，设置实际包路径，图另存包外（不改变已校验包）：

```bash
PKG=/实际解压目录/g2_e_alpha0p1_tau0p5_实际标识
python "$PKG/scripts/plot_g2_e_alpha_results.py" --package-dir "$PKG" --output-dir "${PKG}_replot"
```

离线重绘只需 numpy、matplotlib、Pillow，不需要 PyTorch、完整数据集或训练 checkpoint。K4/K6 各取稳定 key 顺序的前六例，不足则显示实际数量，为零则明确标零。

## 共同样本对照

交付包包含已核验的 G1/G2/G2-A 和 alpha0p3 历史证据；不重训这些版本。新版本均按原规则选择 256 张 query+gallery 样本。比较严格核对 stable_sample_key、split、pid、camid；不静默取交集。

两个新版本导出成功后：

```bash
cd /root/autodl-tmp/g2_e_alpha_sweep_delivery
python compare_after_training.py
```

输出包含各版 Rank-1/5/10/mAP、alpha0.1/0.3/0.5 相对 alpha0.3 的百分点差、各版 w 公共直方图、三种 alpha 的 w 和 c 独立图、相同分箱坐标、原始样本 CSV 和来源清单。跨温度/融合结构的差异不归因于 alpha。单 seed 结果不作显著性或稳定性结论。

底层比较命令为 `python tools/compare_g2_e_alpha_sweep.py --sources 来源配置.json --output-dir 新目录`；JSON 包含 `sources: [{label, package_dir, alpha}]`，历史 G1/G2/G2-A 不填 alpha。缺失包明确列出，已有证据损坏或样本不匹配则报错。缺少 alpha0.1/0.5 时只能生成部分对照，不能宣称三种 alpha 已比较完成。

## 本地验证

新增集成测试覆盖：完整解析配置、1.1/1.5 倍率、全局不变与2816维、同 seed 参数一致、梯度、checkpoint身份、延迟绘图及选中epoch、真实登记与表更新、双 alpha 去重、故障保留、完整打包入口与离线重绘、坏证据拒绝。合成 fixture 仅存在临时测试目录，不进入实际实验台账。

每个分支的测试命令：

```bash
python -m unittest tests.test_g2_e_alpha_delivery tests.test_g2_e_static_dynamic_alpha0p3_tau0p5 tests.test_g2_e_static_dynamic_evidence_tools tests.test_dynamic_gating tests.test_recover_g2_global_local_experiment -v
```

具体代码 SHA 与验证结果见交付目录的 release_manifest.json 和 validation.json。
