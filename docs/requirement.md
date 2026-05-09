# 项目要求与功能规格

**更新日期：** 2026/05/09

---

## 数据集

- **ImageNet-Mini** 或 **CIFAR-10**
- 彩色图 → 灰度图（输入）+ 原彩色图（标签），无需外部标注

---

## 评测指标

1. **主观色彩合理性**：物体（草地、天空、人脸）颜色是否符合现实逻辑
2. **色调分布对比**：预测色彩分布与真实色彩分布的直方图相似度（巴氏距离或 KL 散度）

---

## 三阶段计划与功能规格

### Phase 1：Zhang et al. 2016（基线）

**论文：** `Colorful_Image_Colorizasion_2016_paper.pdf`

- 输入：L 通道（灰度）→ 输出：313 个 ab 色彩 bin 概率分布
- 损失：多项交叉熵 + 类别重平衡权重（抑制灰色偏向）
- 推理：annealed-mean 解码（温度参数 T=0.38）
- **无需 Mask R-CNN，无需 Attention**
- 验收：能对 ImageNet-Mini/CIFAR-10 完整跑通训练与推理

**运行命令：**
```bash
python train.py --method zhang2016 --dataset imagenet_mini
python test.py  --method zhang2016 --input <img>
```

---

### Phase 2：Su et al. CVPR 2020（进阶）

**论文：** `Aware_Image_Colorization_CVPR_2020_paper.pdf`

- 在 Phase 1 骨干（`SIGGRAPHGenerator`）上叠加双分支结构
- **Mask R-CNN**（`torchvision` 内置）：检测实例 → crop 256×256
- **融合权重（Attention）**：3-conv 预测逐像素权重，融合实例分支与全图分支特征
- 三阶段训练：`full → instance → fusion`
- 验收：视觉效果优于 Phase 1，尤其多物体场景颜色更准确

**运行命令：**
```bash
python train.py --method inst2020 --stage full     --dataset imagenet_mini
python train.py --method inst2020 --stage instance --dataset imagenet_mini
python train.py --method inst2020 --stage fusion   --dataset imagenet_mini
python test.py  --method inst2020 --input <img>
```

---

### Phase 3：Exemplar-based 上色（Bonus）

- 网络额外输入一张参考风格图（`--exemplar --ref_img <path>`）
- 从参考图提取 Color Palette（色彩 patch embedding）
- **Cross-Attention**：将参考色彩迁移到灰度图对应语义区域
- 可叠加于 Phase 1 或 Phase 2 之上
- 验收：切换不同参考图，输出颜色风格随之变化

**运行命令：**
```bash
python test.py --method zhang2016 --exemplar --ref_img <ref>  --input <img>
python test.py --method inst2020  --exemplar --ref_img <ref>  --input <img>
```

---

## 各 Phase 技术组件对比

| 组件 | Phase 1 | Phase 2 | Phase 3 |
|------|---------|---------|---------|
| 全局 CNN（Zhang2016Generator） | ✓ | ✓（骨干复用） | ✓ |
| ab 量化分类（313 bins） | ✓ | ✓ | ✓ |
| Mask R-CNN | ✗ | ✓ | 同上 |
| 融合 Attention（FusionGenerator） | ✗ | ✓ | 同上 |
| Cross-Attention（ExemplarAttention） | ✗ | ✗ | ✓ |

---

## 两篇论文技术要点速查

### Zhang et al. 2016
- CIE Lab 空间，L → ab
- ab 量化为 313 个色彩 bin，按经验分布重加权缓解灰色偏向
- 推理用 annealed-mean（温度 T）从分布采样颜色
- 损失：多项交叉熵

### Su et al. 2020
- 以 Zhang 2016 为骨干，增加实例分支
- Mask R-CNN 检测实例 → crop 256×256 → 独立上色 → 融合
- 融合：`f̃ = f_full ⊙ W_F + Σ f_inst_i ⊙ W_i`（逐像素加权求和）
- 损失：Smooth-L1
