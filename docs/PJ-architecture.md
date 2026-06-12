# 项目架构说明

**更新日期：** 2026/05/18

---

## 整体结构

```
photo-colorization/
├── InstColorization-master(reference)/  # 参考实现（只读，不可修改）
├── code/                                # 项目代码（唯一开发区）
├── docs/                                # 计划/进度文档
└── paper/                               # 论文 PDF
```

---

## 三阶段总体规划

| Phase | 方法标识 | 核心技术 | Mask R-CNN | Attention |
|-------|---------|---------|-----------|-----------|
| Phase 1 | `cnn_color` | 全局 CNN + ab 软编码分类（Zhang et al. 2016） | 否 | 否 |
| Phase 2 | `inst_fusion` | 双分支 + FiLM 语义调制 + 融合权重（Su et al. CVPR 2020 + 创新） | 是（torchvision） | 是（融合权重） |
| Phase 3 | `text_color` | CLIP 文本 Adapter 叠加于冻结 Phase 2，per-instance prompt 颜色控制 | 是（推理时 + ab_cap 后处理） | 否（FiLM 调制） |

---

## code/ 目录架构

```
code/
├── train.py                       # 训练主入口（--method / --stage）
├── test.py                        # 推理主入口（--method / --prompt）
├── options/
│   ├── base_options.py            # 基础参数（dataset、name、fineSize 等）
│   └── train_options.py           # 训练/推理参数（method、stage、prompt、ab_cap）
├── models/
│   ├── __init__.py                # 按 --method 动态加载模型
│   ├── base_model.py              # 基类（save/load/scheduler）
│   ├── cnn_color_model.py         # Phase 1：全图上色训练/推理逻辑
│   ├── inst_fusion_model.py       # Phase 2：三阶段训练/融合推理逻辑
│   ├── text_color_model.py        # Phase 3：双前向训练 + adapter ckpt I/O
│   ├── text_color_networks.py     # Phase 3：TextAdapter + TextColorPipeline（组合 Phase 2）
│   └── networks.py                # Phase 1/2 网络结构定义
├── datasets/
│   └── colorization_dataset.py   # 统一 Dataset（支持 cnn_color / inst_fusion）
├── util/
│   ├── util.py                    # Lab/RGB 转换、313-bin 量化、color 工具
│   └── visualizer.py              # TensorBoard 可视化
└── scripts/
    ├── train_phase1.sh            # Phase 1 单阶段训练
    ├── train_phase2.sh            # Phase 2 三阶段训练编排
    ├── train_phase3.sh            # Phase 3 单 adapter 训练
    ├── build_phase3_jsonl.py      # COCO → JSONL（HSV 颜色词 + 负色采样）
    ├── cache_clip_embeddings.py   # 离线缓存所有训练 caption 的 CLIP 文本嵌入
    ├── test.sh                    # 推理（支持所有方法组合）
    └── setup.sh                   # 环境验证脚本
```

---

## networks.py 中的网络结构

| 类名 | 所属 Phase | 说明 |
|------|-----------|------|
| `CnnColorGenerator` | Phase 1 | 全局 CNN，L → 313 ab bins（Zhang 2016） |
| `InstanceGenerator` | Phase 2 骨干 | 全图/实例共用架构，U-Net skip decoder |
| `FiLMLayer` | Phase 2 | 逐通道 scale+shift 条件调制 |
| `FiLMInstanceGenerator` | Phase 2 实例分支 | InstanceGenerator + conv4~7 FiLM 调制 |
| `WeightGenerator` | Phase 2 融合 | 逐层 softmax 加权融合全图与实例特征 |
| `FusionPipeline` | Phase 2 | 调度全图/实例/融合的完整推理流程 |
| `TextAdapter` | Phase 3 | 每层独立 2 层 MLP（512→GELU→2C）生成 FiLM gamma/beta，最后一层 zero-init |
| `TextColorPipeline` | Phase 3 | 组合冻结 `FiLMInstanceGenerator` + `FusionPipeline`，在实例 5 个 conv 层注入 TextAdapter（背景不调制） |

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

- Phase 1/2 Stage-full：**ImageNet-Mini** 或 **CIFAR-10**，无需标注
- Phase 2 Stage-instance：**COCO2017**，使用 GT bbox + GT label 裁剪实例
- Phase 2 Stage-fusion：COCO2017 全图，在线 Mask R-CNN 检测
- 彩色图 → CIE Lab 空间：L 通道为模型输入，ab 通道为预测目标
- 无需预计算 bbox npz 文件，bbox 全部在线生成

---

## Phase 1 数据流

```
彩色图 → Lab 转换 → L 通道（输入）
                              │
                    CnnColorGenerator
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
          ├── Mask R-CNN → {bbox, label} × top-8
          │
          ├── 全图分支 InstanceGenerator（冻结）─────────────────────┐
          │     └── 逐层输出特征                                      │
          │                                                           ▼
          └── 实例分支 FiLMInstanceGenerator（冻结）          WeightGenerator × N层
               └── 裁剪实例 + label → FiLM 调制（conv4~7）    （逐层 softmax 加权融合）
               └── 逐层输出特征 ────────────────────────────────────┘
                                                                      │
                                                                 ab 通道 → RGB 输出
```

---

## Phase 3 数据流（CLIP 文本控制，叠加于冻结 Phase 2）

```
灰度图 L                              用户 prompt:
   │                                  inst:0=a red dog
   │                                  inst:1=a yellow shirt
   │                                  (bg prompt 接受但忽略，背景不调制)
   │                                     ↓
   │                                  CLIP ViT-B/32 文本塔（冻结）
   │                                  → text_emb (N, 512)
   │
   ├──→ Mask R-CNN → {bbox, label, mask} × N 实例
   │
   ├──→ FiLMInstanceGenerator（冻结）
   │      实例分支 feature_map per layer
   │           ↓
   │      TextAdapter（可训，2.89M）
   │           5 层 FiLM 调制：conv6_3 / conv7_3 / conv8_3 / conv9_3 / conv10_2
   │           ↓
   │      modulated inst_feats
   │
   └──→ FusionPipeline（冻结）
          backbone 处理全图 bg 特征
          WeightGenerator 用 modulated inst_feats 逐层融合
          model_class → out_class (313)
              ↓
          annealed-mean(T=0.38) → ab
              ↓
          ab magnitude cap（默认 0.45，保留色相、缩放半径）
              ↓
          L + ab → lab2rgb → RGB
```

**关键性质：** Phase 2 全部权重冻结，TextAdapter MLP 最后一层 zero-init 保证训练前 pipeline 输出与 Phase 2 baseline bit-equal。Phase 3 是纯加法扩展，不破坏 Phase 1/2 已建立的能力。

---

## Phase 2 训练三阶段

| Stage | 输入 | 可训练部分 | 初始权重 | 数据集 |
|-------|------|----------|---------|--------|
| `full` | 全图 L | InstanceGenerator | Phase 1 权重迁移 | ImageNet-Mini |
| `instance` | 实例 crop L + label | FiLMInstanceGenerator（FiLM 层随机初始化） | full 权重 | COCO2017 |
| `fusion` | 全图 + bbox + label | WeightGenerator × N层 + output_conv | full + instance 权重（冻结） | COCO2017 |

---

## 关键设计原则

1. **Phase 2 骨干复用 Phase 1**：`InstFusionGenerator` 直接加载 Phase 1 训练好的权重，不从头训练
2. **Mask R-CNN 不是硬前置**：仅 Phase 2 instance/fusion 阶段使用，在线调用，无需离线预计算
3. **Phase 3 通过 composition 叠加**：`TextColorPipeline` 持有冻结 Phase 2 对象，`networks.py` / `inst_fusion_model.py` / Phase 2 ckpt 全部零修改
