# Bag of Tricks and A Strong ReID Baseline

## 项目维护说明

### 一、项目基本情况

本项目是基于 **Bag of Tricks and A Strong Baseline for Deep Person Re-identification** 的行人重识别实验项目。

主要用途：

1. 复现 BoT baseline。
2. 在 BoT 基础上加入 Part Attention。
3. 做 AutoDL 运行适配。
4. 做不同 loss / camera-aware / cross-camera positive / tau 敏感性等实验。
5. 所有实验都要求记录 commit id、config、seed、GPU、运行时间、best epoch、Rank-1、mAP、备注。

### 二、重要目录说明

| 路径 | 说明 |
|---|---|
| `config/defaults.py` | 管理默认配置。新增实验开关、loss 权重、路径默认值等应优先放在这里，并保持默认值不影响旧实验。 |
| `configs/` | 保存不同实验的 yml 配置。一个实验应有一个独立 config，便于复现。 |
| `data/` | 数据加载相关代码。不要把真实数据集提交到 GitHub。 |
| `data/datasets/` | Market1501、DukeMTMC-reID 等数据集解析逻辑，包括 `pid`、`camid` 解析。 |
| `data/collate_batch.py` | batch 拼接逻辑。训练中需要 `camid` 时，应确认这里返回 `img, pid, camid`。 |
| `engine/trainer.py` | 负责训练过程、loss 调用、日志输出、验证触发和 checkpoint 保存。 |
| `layers/triplet_loss.py` | 保存 `TripletLoss`、`CameraAwareTripletLoss`、`CrossCameraPositiveLoss` 等 loss 实现。修改新 loss 时不要破坏原 `TripletLoss`。 |
| `layers/__init__.py` | loss 构建入口，负责根据 config 组合 id loss、triplet loss 和实验 loss。 |
| `modeling/baseline.py` | BoT baseline 主干模型定义。实验中通常不要随意改 backbone。 |
| `scripts/` | 保存 AutoDL 启动脚本和实验记录脚本，例如训练脚本、`record_experiment_info.sh`、`append_experiment_result.py`。 |
| `EXPERIMENTS.md` | 保存实验记录，至少记录 commit id、config、seed、GPU、运行时间、best epoch、Rank-1、mAP、备注。 |
| `AUTODL_RUN.md` | 保存 AutoDL 运行说明、路径、tmux 命令、训练入口和结果整理方式。 |

### 三、当前主要分支说明

| 分支 | 用途 |
|---|---|
| `main` | 原始 BoT / 本地基础版本。 |
| `main-autodl` | BoT + Part Attention K=6 + AutoDL 可运行稳定版本。 |
| `part-attention-tau-sensitivity` | Part Attention 的 tau 敏感性实验版本，包含 tau=0.1、0.2、0.5 配置。 |
| `exp/normalized-weighted-loss` | 归一化 weighted loss 实验版本。 |
| `exp/camera-aware-triplet-loss` | Camera-aware hard triplet loss 实验版本，使用 same pid + different camid 的 cross-camera positive，并包含 hard negative mining。 |
| `exp/hierarchical-camera-aware-loss` | Hierarchical camera-aware loss 实验版本，包含 easy / boundary / hard anchor、hard negative weighting 等增强逻辑。 |
| `exp/cross-camera-positive-only` | 只使用 cross-camera positive 的消融实验版本，不额外使用 hard negative mining / hard negative weighting。 |

### 四、实验记录规则

以后所有实验必须自动记录到 `EXPERIMENTS.md` 或对应实验记录文档。

记录字段至少包括：

- commit id
- config 文件
- seed
- GPU
- 运行时间
- best epoch
- Rank-1
- mAP
- 备注

说明：

1. 训练前可以用 `scripts/record_experiment_info.sh` 查看基础信息。
2. 训练结束后应由 `scripts/append_experiment_result.py` 自动解析 `log.txt` 并更新 `EXPERIMENTS.md`。
3. 如果 Rank-1、mAP、best epoch 无法解析，填写“待填写”，不能编造。
4. 训练脚本应在训练成功结束后自动调用 `append_experiment_result.py`。
5. 自动更新 `EXPERIMENTS.md` 后，如需保存到 GitHub，需要手动执行 `git add`、`git commit`、`git push`。

### 五、AutoDL 基本路径说明

项目目录：

```bash
/root/autodl-tmp/BoT-reid
```

数据集目录：

```bash
/root/autodl-tmp/datasets/market1501
```

如果实际目录叫 `data`，可以使用软链接：

```bash
cd /root/autodl-tmp
rm -f datasets
ln -s data datasets
```

预训练权重：

```bash
/root/autodl-tmp/pretrained/resnet50-19c8e357.pth
```

实验输出：

```bash
/root/autodl-tmp/experiments/BoT/
```

### 六、AutoDL 常用命令

拉取指定分支：

```bash
git clone -b <branch_name> https://github.com/lzhe036-web/BoT-reid-part-attention.git BoT-reid
```

查看分支和最近提交：

```bash
git branch
git log --oneline -5
```

检查数据集：

```bash
ls /root/autodl-tmp/datasets/market1501
```

检查权重：

```bash
ls /root/autodl-tmp/pretrained
```

tmux 启动：

```bash
tmux new -s <name>
cd /root/autodl-tmp/BoT-reid
bash scripts/<train_script>.sh
```

断开 tmux：

```text
Ctrl + B，然后按 D
```

重新进入：

```bash
tmux attach -t <name>
```

### 七、不同实验运行入口

常见脚本如下。如果某些脚本在当前分支不存在，则以当前分支实际文件为准。

- `scripts/train_part_attention_autodl.sh`
- `scripts/train_part_attention_k6_tau01_autodl.sh`
- `scripts/train_part_attention_k6_tau02_autodl.sh`
- `scripts/train_part_attention_k6_tau05_autodl.sh`
- `scripts/train_normalized_weighted_loss_autodl.sh`
- `scripts/train_camera_aware_triplet_autodl.sh`
- `scripts/train_hierarchical_camera_aware_autodl.sh`
- `scripts/train_cross_camera_positive_only_autodl.sh`
- `scripts/train_c2_baseline_control_autodl.sh`
- `scripts/train_cross_camera_positive_only_repeat_autodl.sh`

### 八、维护注意事项

1. 不要把数据集、预训练权重、实验输出、checkpoint 上传 GitHub。
2. 不要在稳定分支上直接改实验代码，应新建实验分支。
3. 每次实验前确认 `git status` 干净。
4. 每次实验结果必须能追溯到 commit id 和 config 文件。
5. 修改 loss 时不要破坏原 `TripletLoss`。
6. 修改 trainer 时要保证旧配置仍可运行。
7. AutoDL 上训练结束后，`EXPERIMENTS.md` 如果被自动更新，需要手动 commit 回 GitHub。
8. 模型 checkpoint 建议压缩下载保存，不上传 GitHub。

### 九、当前项目维护原则

1. 一个实验一个分支。
2. 一个实验一个 config。
3. 一个实验一个 `OUTPUT_DIR`。
4. 一个实验一条 `EXPERIMENTS.md` 记录。
5. 训练脚本负责跑实验和自动记录。
6. 代码改动必须可复现、可回退、可解释。

---

Bag of Tricks and A Strong Baseline for Deep Person Re-identification. CVPRW2019, Oral.

A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification. IEEE Transactions on Multimedia (Accepted).

[[Journal Version(TMM)]](https://ieeexplore.ieee.org/document/8930088)
[[PDF]](http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)
[[Slides]](https://drive.google.com/open?id=1h9SgdJenvfoNp9PTUxPiz5_K5HFCho-V)
[[Poster]](https://drive.google.com/open?id=1izZYAwylBsrldxSMqHCH432P6hnyh1vR)

### News! Based on the strong baseline, we won 3rd place on AICity Challenge 2020. [[PDF]](https://arxiv.org/pdf/2004.10547.pdf) [[Code]](https://github.com/heshuting555/AICITY2020_DMT_VehicleReID)

### News! Our journal version has been accepted by IEEE Transactions on Multimedia.

### We are very grateful for your contribution to our project and hope that this project can help your research or work.

The codes are expanded on a [ReID-baseline](https://github.com/L1aoXingyu/reid_baseline) , which is open sourced by our co-first author [Xingyu Liao](https://github.com/L1aoXingyu).

Another re-implement is developed by python2.7 and pytorch0.4. [[link]](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks)

A tiny repo with simple re-implement. [[link]](https://github.com/lulujianjie/person-reid-tiny-baseline)

Our baseline also achieves great performance on __Vehicle ReID__ task! [[link]](https://github.com/DTennant/reid_baseline_with_syncbn)

With Ranked List loss(CVPR2019)[[link]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper.pdf), our baseline can achieve better performance. [[link]](https://github.com/Qidian213/Ranked_Person_ReID)


```
@InProceedings{Luo_2019_CVPR_Workshops,
author = {Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
title = {Bag of Tricks and a Strong Baseline for Deep Person Re-Identification},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}

@ARTICLE{Luo_2019_Strong_TMM, 
author={H. {Luo} and W. {Jiang} and Y. {Gu} and F. {Liu} and X. {Liao} and S. {Lai} and J. {Gu}}, 
journal={IEEE Transactions on Multimedia}, 
title={A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification}, 
year={2019}, 
pages={1-1}, 
doi={10.1109/TMM.2019.2958756}, 
ISSN={1941-0077}, 
}
```

## Authors
- [Hao Luo](https://github.com/michuanhaohao)
- [Youzhi Gu](https://github.com/shaoniangu)
- [Xingyu Liao](https://github.com/L1aoXingyu)
- [Shenqi Lai](https://github.com/xiaolai-sqlai)

We support
- [x] easy dataset preparation
- [x] end-to-end training and evaluation
- [x] high modular management
- [x] speed up inference [[link]](https://github.com/DTennant/reid_baseline_with_syncbn)
- [x] support multi-gpus training [[link]](https://github.com/DTennant/reid_baseline_with_syncbn)

Bag of tricks
- Warm up learning rate
- Random erasing augmentation
- Label smoothing
- Last stride
- BNNeck
- Center loss

## TODO list
In the future, we will
- [] support more datasets
- [] support more models
- [] explore more tricks


## Pipeline
<div align=center>
<img src='imgs/pipeline.jpg' width='800'>
</div>

## Results (rank1/mAP)
| Model | Market1501 | DukeMTMC-reID |
| --- | -- | -- |
| Standard baseline | 87.7 (74.0) |  79.7 (63.8) |
| +Warmup | 88.7 (75.2) |  80.6(65.1) |
| +Random erasing augmentation | 91.3 (79.3) |  81.5 (68.3) |
| +Label smoothing | 91.4 (80.3) |  82.4 (69.3) |
| +Last stride=1 | 92.0 (81.7) | 82.6 (70.6) |
| +BNNeck | 94.1 (85.7) | 86.2 (75.9) |
| +Center loss | 94.5 (85.9) | 86.4 (76.4) |
| +Reranking | 95.4 (94.2) | 90.3 (89.1) |

| Backbone | Market1501 | DukeMTMC-reID |
| --- | -- | -- |
| ResNet18 | 91.7 (77.8) |  82.5 (68.8) |
| ResNet34 | 92.7 (82.7) |  86.4(73.6) |
| ResNet50 | 94.5 (85.9) | 86.4 (76.4) |
| ResNet101 | 94.5 (87.1) |  87.6 (77.6) |
| ResNet152 | 80.9 (59.0) | 87.5 (78.0) |
| SeResNet50 | 94.4 (86.3) | 86.4 (76.5) |
| SeResNet101 | 94.6 (87.3) | 87.5 (78.0) |
| SeResNeXt50 | 94.9 (87.6) | 88.0 (78.3) |
| SeResNeXt101 | 95.0 (88.0) | 88.4 (79.0) |
| IBN-Net50-a | 95.0 (88.2) | 90.1 (79.1) |

[model(Market1501)](https://drive.google.com/open?id=1hn0sXLZ5yJcxtmuY-ItQfYD7hBtHwt7A)

[model(DukeMTMC-reID)](https://drive.google.com/open?id=1LARvQe-gUbflbanidUM0keKmHoKTpLUj)

## Get Started
The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

1. `cd` to folder where you want to download this repo

2. Run `git clone https://github.com/michuanhaohao/reid-strong-baseline.git`

3. Install dependencies:
    - [pytorch>=0.4](https://pytorch.org/)
    - torchvision
    - [ignite=0.1.2](https://github.com/pytorch/ignite) (Note: V0.2.0 may result in an error)
    - [yacs](https://github.com/rbgirshick/yacs)

4. Prepare dataset

    Create a directory to store reid datasets under this repo or outside this repo. Remember to set your path to the root of the dataset in `config/defaults.py` for all training and testing or set in every single config file in `configs/` or set in every single command.

    You can create a directory to store reid datasets under this repo via

    ```bash
    cd reid-strong-baseline
    mkdir data
    ```

    （1）Market1501

    * Download dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html
    * Extract dataset and rename to `market1501`. The data structure would like:

    ```bash
    data
        market1501 # this folder contains 6 files.
            bounding_box_test/
            bounding_box_train/
            ......
    ```
    （2）DukeMTMC-reID

    * Download dataset to `data/` from https://github.com/layumi/DukeMTMC-reID_evaluation#download-dataset
    * Extract dataset and rename to `dukemtmc-reid`. The data structure would like:

    ```bash
    data
        dukemtmc-reid
        	DukeMTMC-reID # this folder contains 8 files.
            	bounding_box_test/
            	bounding_box_train/
            	......
    ```

5. Prepare pretrained model if you don't have

    （1）ResNet

    ```python
    from torchvision import models
    models.resnet50(pretrained=True)
    ```
    （2）Senet

    ```python
    import torch.utils.model_zoo as model_zoo
    model_zoo.load_url('the pth you want to download (specific urls are listed in  ./modeling/backbones/senet.py)')
    ```
    Then it will automatically download model in `~/.torch/models/`, you should set this path in `config/defaults.py` for all training or set in every single training config file in `configs/` or set in every single command.

    （3）ResNet_IBN_a

    You can download the ImageNet pre-trained weights from here [[link]](https://drive.google.com/open?id=1_r4wp14hEMkABVow58Xr4mPg7gvgOMto)

    （4）Load your self-trained model
    If you want to continue your train process based on your self-trained model, you can change the configuration `PRETRAIN_CHOICE` from 'imagenet' to 'self' and set the `PRETRAIN_PATH` to your self-trained model. We offer `Experiment-pretrain_choice-all_tricks-tri_center-market.sh` as an example. 

6. If you want to know the detailed configurations and their meaning, please refer to `config/defaults.py`. If you want to set your own parameters, you can follow our method: create a new yml file, then set your own parameters.  Add `--config_file='configs/your yml file'` int the commands described below, then our code will merge your configuration. automatically.

## Train
You can run these commands in  `.sh ` files for training different datasets of differernt loss.  You can also directly run code `sh *.sh` to run our demo after your custom modification.

1. Market1501, cross entropy loss + triplet loss

```bash
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('market1501')" OUTPUT_DIR "('your path to save checkpoints and logs')"
```

2. DukeMTMC-reID, cross entropy loss + triplet loss + center loss


```bash
python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('dukemtmc')" OUTPUT_DIR "('your path to save checkpoints and logs')"
```

## Test
You can test your model's performance directly by running these commands in `.sh ` files after your custom modification. You can also change the configuration to determine which feature of BNNeck is used and whether the feature is normalized (equivalent to use Cosine distance or Euclidean distance) for testing.

Please replace the data path of the model and set the `PRETRAIN_CHOICE` as 'self' to avoid time consuming on loading ImageNet pretrained model.

1. Test with Euclidean distance using feature before BN without re-ranking,.

```bash
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('market1501')" TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('no')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('your path to trained checkpoints')"
```
2. Test with Cosine distance using feature after BN without re-ranking,.

```bash
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('market1501')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('your path to trained checkpoints')"
```
3. Test with Cosine distance using feature after BN with re-ranking

```bash
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('dukemtmc')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.RE_RANKING "('yes')" TEST.WEIGHT "('your path to trained checkpoints')"
```

