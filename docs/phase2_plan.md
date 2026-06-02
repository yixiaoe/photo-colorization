# Phase 2 实施报告：FiLM 条件嵌入的实例感知上色

**日期：** 2026/06/02  
**状态：** 训练完成（RTX 5090，约 5 天）  
**核心思路：** 在 Su2020 双分支融合框架上，将 Mask R-CNN 的语义标签通过 FiLM 机制注入实例分支，使实例网络"知道自己在给什么物体上色"

---

## 1. 最终架构

```
输入灰度图 L
    │
    ├── 全图网络 InstanceGenerator（冻结）
    │     └── 逐层前向，每层输出特征供融合
    │
    ├── Mask R-CNN → {bbox, label} × top-8
    │
    ├── 实例网络 FiLMInstanceGenerator（冻结）
    │     └── 输入：裁剪实例 L + class_label
    │     └── conv4~conv7 后插入 FiLM 调制
    │
    └── FusionPipeline（WeightGenerator ×13 + output heads 可训练）
          └── 逐层 softmax 加权融合全图与实例特征
          └── conv8_3 → model_class(256→313) 分类头（新增）
          └── conv10_2 → output_conv(128→2, Tanh) 回归头
          └── 推理用分类头 + annealed-mean 解码
```

**关键架构变更：** 原 FusionPipeline 仅有回归头（Tanh 2ch），融合结果灰暗。新增分类头（313-bin）后推理使用 annealed-mean 解码，色彩鲜艳度大幅提升。

---

## 2. 实际训练参数

| 项目 | Stage 1 (full) | Stage 2 (instance) | Stage 3 (fusion) |
|------|----------------|-------------------|-----------------|
| 网络 | InstanceGenerator | FiLMInstanceGenerator | FusionPipeline |
| 数据 | **COCO2017** 全图 | COCO2017 GT bbox+label | COCO2017 + bbox cache |
| 损失 | CE + **3×**Huber | CE + **3×**Huber | CE（新增）+ **5×**Huber |
| 初始化 | 随机 | 骨干加载 Stage 1；FiLM 随机 | 骨干加载 Stage 1；实例加载 Stage 2 |
| 学习率 | **3e-5** | 骨干**1e-5** / FiLM **3e-5** | **2e-5** |
| 训练次数 | v1(Huber×10)→v2(Huber×3)→v3(纯CE, 跳过) | v2(Huber×3, 40 epoch) | v4-v6 多次调参, 最终 10+10 epoch |
| 最终 epoch | **50+50=100**（v2 版本） | **20+20=40** | **10+10=20** |
| 产出 | `inst_fusion_full/net_G.pth` | `inst_fusion_instance/net_G.pth` | `inst_fusion_fusion/20_net_G.pth` |

### 损失函数演变

训练过程中经历了多轮调参，最终确定的 loss 配置：
- **Stage 1/2**：`L = CE(rebalanced) + 3 × Huber(ab)` — 分类头驱动鲜艳度，Huber 约束空间一致性
- **Stage 3**：`L = CE(rebalanced) + 3 × Huber(fused_ab)` — CE 主导色彩，Huber×3 防止异常色块

---

## 3. 推理流程

1. Mask R-CNN 检测实例 → top-8 bbox + label
2. 无实例时回退到骨干网络回归头输出
3. 各实例裁剪 → FiLMInstanceGenerator 前向（获取特征）
4. FusionPipeline 逐层融合全图与实例特征
5. 分类头 → annealed-mean 解码(T=0.38) → ab → 与 L 合并 → RGB

---

## 4. 已修复问题

| 问题 | 修复 |
|------|------|
| Huber×10 导致色彩灰暗 | 降到 3×（Stage 1/2）或 3×（Stage 3） |
| Fusion 仅有回归头 → 灰暗 | **新增分类头 + annealed-mean 解码** |
| BN running stats NaN 污染 | forward NaN 时回滚 BN stats |
| Fusion 骨干 BN 漂移 | train() 中冻结 BN 为 eval 模式 |
| 初始 Huber×10 梯度爆炸 | 学习率从 1e-4 降到 3e-5 |
