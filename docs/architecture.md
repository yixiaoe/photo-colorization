# 项目架构说明

**更新日期：** 2026/05/09

---

## 整体结构

```
photo-colorization/
├── InstColorization-master(reference)/  # 参考实现（只读，不可修改）
├── code/                                # 项目代码（唯一开发区）
├── opsx/                                # 计划/进度文档
└── paper/                               # 论文 PDF
```

---

## 三阶段总体规划

| Phase | 方法 | 核心技术 | Mask R-CNN | Attention |
|-------|------|---------|-----------|-----------|
| Phase 1 | Zhang et al. 2016 | 全局 CNN + ab 量化分类 | 否 | 否 |
| Phase 2 | Su et al. CVPR 2020 | 双分支 + 融合权重（在 Phase 1 骨干上叠加） | 是（torchvision） | 是（融合权重） |
| Phase 3 | Exemplar Bonus | Cross-Attention 色彩迁移（叠加于任意方法） | 同上 | 是（额外） |

---

## code/ 目录架构

```
code/
├── train.py                     # 训练主入口（--method / --stage）
├── test.py                      # 推理主入口（--method / --exemplar）
├── options/
│   ├── base_options.py          # 基础参数（dataset、name、fineSize 等）
│   └── train_options.py         # 训练/推理参数（method、stage、exemplar）
├── models/
│   ├── __init__.py              # 按 --method 动态加载模型
│   ├── base_model.py            # 基类（save/load/scheduler）
│   ├── zhang2016_model.py       # Phase 1：全图上色训练/推理逻辑
│   ├── inst2020_model.py        # Phase 2：三阶段训练/融合推理逻辑
│   └── networks.py              # 所有网络结构定义（见下方说明）
├── datasets/
│   └── colorization_dataset.py  # 统一 Dataset（支持 zhang2016 / inst2020）
├── util/
│   ├── util.py                  # Lab/RGB 转换、313-bin 量化、color 工具
│   └── visualizer.py            # TensorBoard 可视化
└── scripts/
    ├── train_phase1.sh          # Phase 1 单阶段训练
    ├── train_phase2.sh          # Phase 2 三阶段训练编排
    ├── test.sh                  # 推理（支持所有方法组合）
    └── setup.sh                 # 环境验证脚本
```

---

## networks.py 中的网络结构

| 类名 | 所属 Phase | 说明 |
|------|-----------|------|
| `Zhang2016Generator` | Phase 1 | 全局 CNN，L → 313 ab bins |
| `SIGGRAPHGenerator` | Phase 2 骨干 | 继承/复用 Zhang2016Generator |
| `FusionGenerator` | Phase 2 | 3-conv 融合权重预测 |
| `WeightGenerator` | Phase 2 | 实例与全图权重融合 |
| `ExemplarAttention` | Phase 3 | Cross-Attention 色彩迁移模块 |

---

## 服务器环境（已确定）

| 项目 | 配置 |
|------|------|
| 镜像 | PyTorch 2.0.0 + Python 3.8 + CUDA 11.8 |
| GPU | RTX 4090 (24GB) × 1 |
| CPU | 16 vCPU Intel Xeon Gold 6430 |
| 内存 | 120 GB |
| 存储 | 系统盘 30GB + 数据盘 50GB SSD |

**Detectron2 不需要安装**，Mask R-CNN 使用 `torchvision.models.detection.maskrcnn_resnet50_fpn`（PyTorch 2.0 内置）。

---

## 数据集约定

- 使用 **ImageNet-Mini** 或 **CIFAR-10**
- 彩色图 → CIE Lab 空间：L 通道为模型输入，ab 通道为预测目标
- 无需外部标注，无需预计算 bbox npz 文件

---

## Phase 1 数据流

```
彩色图 → Lab 转换 → L 通道（输入）
                              │
                    Zhang2016Generator
                              │
                    313 ab bins 概率图
                              │
                    annealed-mean 解码（T=0.38）
                              │
                         ab 通道 → RGB 输出
```

---

## Phase 2 数据流

```
彩色图 → Lab 转换 → L 通道（输入）
          │
          ├── 全图分支（SIGGRAPHGenerator）─────────────┐
          │                                              ▼
          └── 实例分支（torchvision Mask R-CNN → crop）  FusionGenerator
               └── SIGGRAPHGenerator（256×256 crop）  ──┘
                                                          │
                                               逐像素融合权重（WeightGenerator）
                                                          │
                                                     ab 通道 → RGB 输出
```

---

## Phase 3 数据流（Bonus，叠加于 Phase 1 或 Phase 2）

```
灰度图（L）          参考风格图
    │                    │
特征提取         Color Palette 提取
    │                    │（色彩 patch embedding）
    └────── Cross-Attention ──────┘
                   │
             上色输出（RGB）
```

---

## Phase 2 训练三阶段

| Stage | 输入 | 训练目标 | 初始权重 |
|-------|------|---------|---------|
| `full` | 全图 L 通道 | 全图上色分支 netG | Phase 1 权重或随机初始化 |
| `instance` | 实例 crop L 通道 | 实例分支 netG | `full` 阶段权重 |
| `fusion` | 全图 + bbox | 融合权重 netGF | `full` + `instance` 权重 |

---

## 关键设计原则

1. **Phase 2 骨干复用 Phase 1**：`SIGGRAPHGenerator` 直接加载 Phase 1 训练好的权重，不从头训练
2. **Mask R-CNN 不是硬前置**：仅 Phase 2 instance/fusion 阶段使用，在线调用，无需离线预计算
3. **Phase 3 按需叠加**：通过 `--exemplar` flag 激活，不影响 Phase 1/2 的主逻辑
