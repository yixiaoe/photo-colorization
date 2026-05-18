# Phase 2 实施方案：FiLM 条件嵌入的实例感知上色

**日期：** 2026/05/18  
**状态：** 待执行  
**核心思路：** 在 Su2020 双分支融合框架上，将 Mask R-CNN 的语义标签通过 FiLM 机制注入实例分支，使实例网络"知道自己在给什么物体上色"

---

## 1. 动机

上色是语义驱动的任务。Su2020 的实例网络通过裁剪物体隐式聚焦语义，但不知道裁出的是什么。同一灰度轮廓，"金毛犬"应涂金棕，"灰猫"应涂灰色。引入 FiLM 条件嵌入后，类别标签作为调制信号深度渗透到特征提取中，无需改变输入输出格式，轻量且与融合框架完全兼容。

---

## 2. 整体架构

```
输入灰度图
    │
    ├── Mask R-CNN（冻结）→ {bbox, label, score} × top-8
    │
    ├── 全图网络 InstanceGenerator（冻结，加载 stage-full 权重）
    │     └── 逐层前向，每层输出特征供融合
    │
    ├── 实例网络 FiLMInstanceGenerator（冻结，加载 stage-instance 权重）
    │     └── 输入：裁剪实例 L 通道 + class_label
    │     └── conv4~conv7 后插入 FiLM 调制
    │
    └── 融合模块 WeightGenerator × N层（可训练）
          └── 逐层对全图特征和实例特征做 softmax 加权融合
          └── 最终输出头 → ab 通道
```

---

## 3. 网络组件

### InstanceGenerator（全图/实例共用架构）

与 Su2020 一致，两分支共享架构但权重不同。

- Encoder: conv1(64)→conv2(128,s2)→conv3(256,s2)→conv4~7(512,dilation)
- Decoder: conv8_up(256,×2)+skip→conv9_up(128,×2)+skip→conv10(128)+skip
- 输出头：model_class(128→313)、model_out(128→2, Tanh)
- 输入：仅 L 通道（1ch），不使用 hint

### FiLMInstanceGenerator（实例分支）

继承 InstanceGenerator，在 conv4~conv7 后插入 FiLM 层：

- `label_embedding`：nn.Embedding(91, 64)
- `FiLMLayer` × 4：每层 embed_dim→(γ, β)，对特征图做 `feat × (1+γ) + β`
- 使用 residual 形式（1+γ）确保初始化时接近恒等映射
- FiLM 新增参数：~270K（4 × FiLMLayer + Embedding）

### WeightGenerator（融合权重，Su2020 原版）

- 对全图特征和各实例特征分别预测权重图
- 所有权重图拼接后 softmax，做加权求和
- 每个 WeightGenerator 约 3K 参数，13 层合计 ~40K

---

## 4. 训练策略（三阶段）

### Stage 1：全图网络训练

| 项目 | 内容 |
|------|------|
| 网络 | InstanceGenerator（无 FiLM） |
| 数据 | ImageNet-Mini，全图 |
| 损失 | CE（313-bin，rebalanced）+ Huber（ab 回归），λ=1:10 |
| 初始化 | Phase 1 匹配层权重迁移 |
| 超参 | lr=1e-4，Adam，100+100 epoch |
| 产出 | `checkpoints/inst_full/net_G.pth` |

### Stage 2：实例网络训练（含 FiLM）

| 项目 | 内容 |
|------|------|
| 网络 | FiLMInstanceGenerator |
| 数据 | **COCO2017**，GT bbox + GT label 裁剪实例 |
| 损失 | CE + Huber（同 Stage 1） |
| 初始化 | InstanceGenerator 部分加载 Stage 1 权重；FiLM 层随机初始化 |
| 超参 | lr=5e-5（主干）/ 1e-4（FiLM 层），Adam，100+100 epoch |
| 产出 | `checkpoints/inst_instance/net_G.pth` |

**关键细节：**
- 训练用 GT label，不用 Mask R-CNN 预测（避免噪声）
- 测试时用 Mask R-CNN 输出（全自动推理）

### Stage 3：融合模块训练

| 项目 | 内容 |
|------|------|
| 网络 | FusionPipeline（仅 WeightGenerator + output_conv 可训练） |
| 数据 | COCO2017 全图，在线 Mask R-CNN 检测 |
| 损失 | Huber（融合 ab vs gt_ab） |
| 冻结 | 全图网络 + 实例网络全部冻结 |
| 超参 | lr=2e-5，Adam，60+60 epoch |
| 产出 | `checkpoints/inst_fusion/net_fusion.pth` |

---

## 5. 数据集

| Stage | Dataset | 新增字段 |
|-------|---------|---------|
| full | ColorizationDataset（复用现有） | — |
| instance | InstanceDataset（新增） | `gray`、`ab_gt`、`class_label` |
| fusion | FusionDataset（扩展现有） | `gray_full`、`instance_crops`、`bboxes`、`class_labels` |

COCO 下载命令（服务器）：
```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

---

## 6. FiLM 注入位置依据

| 层 | 特征语义 | 是否插入 FiLM |
|----|---------|-------------|
| conv1-3 | 低级（边缘、纹理） | 否，语义无关 |
| **conv4-7** | **高级（物体部件、类别）** | **是** |
| conv8-10 | 解码（上采样恢复） | 否，保留空间信息 |

---

## 7. 损失函数

- **Stage 1/2**：`L = CE（rebalanced）+ 10 × Huber(ab)`
- **Stage 3**：`L = Huber(fused_ab, gt_ab)`

---

## 8. 推理流程

1. Mask R-CNN 检测实例 → top-8 bbox + label
2. 无实例时回退到全图网络直接输出
3. 各实例裁剪 → FiLMInstanceGenerator 前向（获取特征）
4. FusionPipeline 逐层融合全图特征与实例特征
5. 输出头 → ab 通道 → 与 L 合并 → RGB

---

## 9. 文件改动清单

| 文件 | 改动 |
|------|------|
| `models/networks.py` | 新增：InstanceGenerator、FiLMLayer、FiLMInstanceGenerator、WeightGenerator、FusionPipeline |
| `models/inst_fusion_model.py` | 实现：三阶段训练路由、initialize/forward/backward/visuals |
| `data_process/colorization_dataset.py` | 扩展：InstanceDataset（COCO）、FusionDataset（含 class_label） |
| `options/train_options.py` | 新增：FiLM 相关参数（num_classes、embed_dim） |
| `scripts/train_phase2.sh` | 新增：三阶段训练脚本 |
| `test.py` | 新增：Phase 2 推理分支 |

---

## 10. 潜在风险

| 风险 | 缓解措施 |
|------|---------|
| FiLM 层训练不稳定 | Residual FiLM（1+γ）；监控 γ/β 分布 |
| Mask R-CNN 灰度检测性能差 | 训练用 GT，测试用检测器；对比两者结果 |
| 融合参数太少学不动 | WeightGenerator 通道 16→32 可调 |
| 标签粒度不足（如 "bird" 含多色） | 接受并在报告中分析 |

---

## 11. 里程碑

| 阶段 | 任务 | 验收标准 |
|------|------|---------|
| W1 | FiLMLayer + FiLMInstanceGenerator 实现 | 前向无报错，形状正确 |
| W2 | WeightGenerator + FusionPipeline 实现 | 融合管线烟雾测试通过 |
| W3 | InstanceDataset + FusionDataset 实现 | 数据加载形状验证 |
| W4 | Stage 1 训练（全图） | 收敛，PSNR > 24dB |
| W5-6 | Stage 2 训练（实例+FiLM） | 实例级 PSNR 优于 Stage 1 |
| W7 | Stage 3 训练（融合） | 融合后 PSNR 优于 Stage 1 全图 |
| W8 | 消融实验 + 评估 | 指标表格完整 |
