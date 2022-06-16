# 目录

<!-- TOC -->

- [目录](#目录)
- [DeepLabV3+描述](#deeplabv3+描述)
    - [描述](#描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [GPU处理器环境运行](#gpu处理器环境运行)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [GPU处理器环境运行](#gpu处理器环境运行-1)
        - [结果](#结果-1)
            - [训练准确率](#训练准确率)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法-2)
        - [结果](#结果-2)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# DeepLabV3+描述

## 描述

DeepLab是一系列图像语义分割模型，DeepLabv3+通过encoder-decoder进行多尺度信息的融合，同时保留了原来的空洞卷积和ASSP层，
其骨干网络使用了Resnet模型，提高了语义分割的健壮性和运行速率。

有关网络详细信息，请参阅[论文][1]
`Chen, Liang-Chieh, et al. "Encoder-decoder with atrous separable convolution for semantic image segmentation." Proceedings of the European conference on computer vision (ECCV). 2018.`

[1]: https://arxiv.org/abs/1802.02611

# 模型架构

以ResNet-101为骨干，通过encoder-decoder进行多尺度信息的融合，使用空洞卷积进行密集特征提取。

# 数据集

Pascal VOC数据集和语义边界数据集（Semantic Boundaries Dataset，SBD）

- 下载分段数据集。

- 准备训练数据清单文件。清单文件用于保存图片和标注对的相对路径。如下：

     ```text
     VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg VOCdevkit/VOC2012/SegmentationClassGray/2007_000032.png
     VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg VOCdevkit/VOC2012/SegmentationClassGray/2007_000039.png
     VOCdevkit/VOC2012/JPEGImages/2007_000063.jpg VOCdevkit/VOC2012/SegmentationClassGray/2007_000063.png
     VOCdevkit/VOC2012/JPEGImages/2007_000068.jpg VOCdevkit/VOC2012/SegmentationClassGray/2007_000068.png
     ......
     ```

你也可以通过运行脚本：`python get_dataset_list.py --data_root=/PATH/TO/DATA` 来自动生成数据清单文件。

- 配置并运行get_dataset_mindrecord.sh，将数据集转换为MindRecords。scripts/get_dataset_mindrecord.sh中的参数：

     ```
     --data_root                 训练数据的根路径
     --data_lst                  训练数据列表（如上准备）
     --dst_path                  MindRecord所在路径
     --num_shards                MindRecord的分片数
     --shuffle                   是否混洗
     ```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件
    - 准备GPU搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)
- 安装requirements.txt中的python包。
- 生成config json文件用于8卡训练。

     ```
     # 从项目根目录进入
     cd src/tools/
     python3 get_multicards_json.py 10.111.*.*
     # 10.111.*.*为计算机IP地址
     ```

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- GPU处理器环境运行

按照以下训练步骤进行8卡训练：

1.使用VOCaug数据集训练s16，微调ResNet-101预训练模型。脚本如下：

```bash
bash run_distribute_train_s16_r1_gpu.sh /PATH/TO/MINDRECORD_NAME /PATH/TO/PRETRAIN_MODEL
```

2.使用VOCaug数据集训练s8，微调上一步的模型。脚本如下：

```bash
bash run_distribute_train_s8_r1_gpu.sh /PATH/TO/MINDRECORD_NAME /PATH/TO/PRETRAIN_MODEL
```

3.使用VOCtrain数据集训练s8，微调上一步的模型。脚本如下：

```bash
bash run_distribute_train_s8_r2_gpu.sh /PATH/TO/MINDRECORD_NAME /PATH/TO/PRETRAIN_MODEL
```

评估步骤如下：

1.使用voc val数据集评估s16。评估脚本如下：

```bash
bash run_eval_s16_gpu.sh /PATH/TO/DATA /PATH/TO/DATA_lst.txt /PATH/TO/PRETRAIN_MODEL DEVICE_ID
```

2.使用voc val数据集评估多尺度s16。评估脚本如下：

```bash
bash run_eval_s16_multiscale_gpu.sh /PATH/TO/DATA /PATH/TO/DATA_lst.txt /PATH/TO/PRETRAIN_MODEL DEVICE_ID
```

3.使用voc val数据集评估多尺度和翻转s16。评估脚本如下：

```bash
bash run_eval_s16_multiscale_flip_gpu.sh /PATH/TO/DATA /PATH/TO/DATA_lst.txt /PATH/TO/PRETRAIN_MODEL DEVICE_ID
```

4.使用voc val数据集评估s8。评估脚本如下：

```bash
bash run_eval_s8_gpu.sh /PATH/TO/DATA /PATH/TO/DATA_lst.txt /PATH/TO/PRETRAIN_MODEL DEVICE_ID
```

5.使用voc val数据集评估多尺度s8。评估脚本如下：

```bash
bash run_eval_s8_multiscale_gpu.sh /PATH/TO/DATA /PATH/TO/DATA_lst.txt /PATH/TO/PRETRAIN_MODEL DEVICE_ID
```

6.使用voc val数据集评估多尺度和翻转s8。评估脚本如下：

```bash
bash run_eval_s8_multiscale_flip_gpu.sh /PATH/TO/DATA /PATH/TO/DATA_lst.txt /PATH/TO/PRETRAIN_MODEL DEVICE_ID
```

# 脚本说明

## 脚本及样例代码

```shell
.
└──deeplabv3plus
  ├── script
    ├── get_dataset_mindrecord.sh                 # 将原始数据转换为MindRecord数据集
    ├── run_distribute_train_s16_r1_gpu.sh            # 使用s16结构的VOCaug数据集启动GPU分布式训练（8卡）
    ├── run_distribute_train_s8_r1_gpu.sh             # 使用s8结构的VOCaug数据集启动GPU分布式训练（8卡）
    ├── run_distribute_train_s8_r2_gpu.sh             # 使用s8结构的VOCtrain数据集启动GPU分布式训练（8卡）
    ├── run_eval_s16_gpu.sh                           # 使用s16结构启动GPU评估
    ├── run_eval_s16_multiscale_gpu.sh                # 使用多尺度s16结构启动GPU评估
    ├── run_eval_s16_multiscale_filp_gpu.sh           # 使用多尺度和翻转s16结构启动GPU评估
    ├── run_eval_s8_gpu.sh                            # 使用s8结构启动GPU评估
    ├── run_eval_s8_multiscale_gpu.sh                 # 使用多尺度s8结构启动GPU评估
    ├── run_eval_s8_multiscale_filp_gpu.sh            # 使用多尺度和翻转s8结构启动GPU评估
  ├── src
    ├── tools
        ├── get_dataset_list.py               # 获取数据清单文件
        ├── get_dataset_mindrecord.py         # 获取MindRecord文件
        ├── get_multicards_json.py            # 获取rank table文件
        ├── get_pretrained_model.py           # 获取resnet预训练模型
    ├── dataset.py                            # 数据预处理
    ├── deeplab_v3plus.py                     # DeepLabV3+网络结构
    ├── learning_rates.py                     # 生成学习率
    ├── loss.py                               # DeepLabV3+的损失定义
  ├── eval.py                                 # 评估网络
  ├── train.py                                # 训练网络
  ├──requirements.txt                         # requirements文件
  └──README.md
```

## 脚本参数

默认配置

```bash
"data_file":"/PATH/TO/MINDRECORD_NAME"            # 数据集路径
"device_target":Ascend                            # 训练后端类型
"train_epochs":300                                # 总轮次数
"batch_size":32                                   # 输入张量的批次大小
"crop_size":513                                   # 裁剪大小
"base_lr":0.08                                    # 初始学习率
"lr_type":cos                                     # 用于生成学习率的衰减模式
"min_scale":0.5                                   # 数据增强的最小尺度
"max_scale":2.0                                   # 数据增强的最大尺度
"ignore_label":255                                # 忽略标签
"num_classes":21                                  # 类别数
"model":DeepLabV3plus_s16                         # 选择模型
"ckpt_pre_trained":"/PATH/TO/PRETRAIN_MODEL"      # 加载预训练检查点的路径
"is_distributed":                                 # 分布式训练，设置该参数为True
"save_steps":410                                  # 用于保存的迭代间隙
"freeze_bn":                                      # 设置该参数freeze_bn为True
"keep_checkpoint_max":200                         # 用于保存的最大检查点
```

#### 训练准确率

| **网络** | OS=16 | OS=8 | MS |翻转| mIOU |论文中的mIOU |
| :----------: | :-----: | :----: | :----: | :-----: | :-----: | :-------------: |
| deeplab_v3+ | √     |      |      |       | 79.78 | 78.85    |
| deeplab_v3+ | √     |     | √    |       | 80.59 |80.09   |
| deeplab_v3+ | √     |     | √    | √     | 80.76 | 80.22        |
| deeplab_v3+ |       | √    |      |       | 79.56 | 79.35    |
| deeplab_v3+ |       | √    | √    |       | 80.43 |80.43   |
| deeplab_v3+ |       | √    | √    | √     | 80.69 | 80.57        |

注意：OS指输出步长（output stride）， MS指多尺度（multiscale）。

# 模型描述

## 性能

### 评估性能

| 参数 | GPU |
| -------------------------- | -------------------------------------- |
| 模型版本 | DeepLabV3+ |
| 资源 | NV SMX2 V100-32G|
| 上传日期 | 2021-08-23|
| MindSpore版本 | 1.4.0|
| 数据集 |  PASCAL VOC2012 + SBD |
| 训练参数 | epoch = 300, batch_size = 16 (s16_r1)  epoch = 800, batch_size = 8 (s8_r1)  epoch = 300, batch_size = 8 (s8_r2) |
| 优化器 | Momentum |
| 损失函数 | Softmax交叉熵 |
| 输出 | 概率 |
| 损失 | 0.003395824|
| 性能 | 1080 ms/step（单卡，s16）|  
| 微调检查点 | 454M （.ckpt文件）|
| 脚本 | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/deeplabv3plus) |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
