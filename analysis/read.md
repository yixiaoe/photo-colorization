# 上色项目结果分析 · Phase 1 & Phase 2

> 这是讨论"要做哪些分析、怎么做"的地方，不是最终报告。
> 所有分析代码只放在 `analysis/` 下，通过 `sys.path` 引用 `code/`，**绝对不修改原始代码任何文件**。

---

## 0. 目录结构约定

```
analysis/
├── read.md                  ← 本文件（计划讨论）
├── _utils.py                ← 本分析专用工具函数（LPIPS、colorfulness 等）
├── run_phase2.py            ← Phase 2 分析入口
├── run_phase1.py            ← Phase 1 分析入口（待权重到位）
│
├── data/                    ← 约 50 张 COCO val2017 GT 彩色图（原图）
│   ├── 000000016502.jpg
│   ├── 000000482800.jpg
│   └── ...（共约 50 张，从本地已有 30 张扩充）
│
├── phase1/                  ← Phase 1 跑出的所有结果
│   ├── vis/                 ← 三联对比图
│   ├── metrics.csv          ← 逐图指标
│   ├── metrics_summary.txt  ← 均值±标准差
│   ├── ab_scatter.png
│   ├── colorfulness.png
│   └── error_heatmap/
│
└── phase2/                  ← Phase 2 跑出的所有结果
    ├── vis/
    ├── metrics.csv
    ├── metrics_summary.txt
    ├── ab_scatter.png
    ├── colorfulness.png
    ├── error_heatmap/
    ├── fusion_weights/      ← fusion 权重热力图（待 fusion 权重到位）
    └── film_ablation.csv    ← FiLM 消融（待 instance 权重到位）
```

**代码隔离原则**：每个脚本头部统一写：
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
```
之后 `from models.networks import ...` 均正常工作，原始 `code/` 目录下任何文件不会被改动。

---

## 1. 当前权重状况（已确认）

| 路径 | 属于哪个阶段 | 能做什么 |
|------|------------|---------|
| `code/checkpoints/inst_full/best_net_G.pth` | Phase 2 stage-full（全图，无 FiLM） | ✅ 现在就能推理 |
| `code/checkpoints/inst_full/30_net_G.pth` | Phase 2 stage-full epoch 30 | ✅ 可用于收敛曲线 |
| `code/checkpoints/inst_full/80_net_G.pth` | Phase 2 stage-full epoch 80 | ✅ 可用于收敛曲线 |

**缺失（需从训练服务器拷回）**：

| 缺少的文件 | 影响哪些分析 |
|-----------|------------|
| `checkpoints/cnn_color/best_net_G.pth` | Phase 1 所有结果 |
| `checkpoints/inst_instance/best_net_G.pth` | FiLM 消融实验 |
| `checkpoints/inst_fusion/net_fusion.pth` | Phase 2 完整 fusion 推理 |
| `checkpoints/*/logs/` | TensorBoard 收敛曲线 |

---

## 2. 数据集口径（已确认）

### 训练数据

- Phase 1 (`cnn_color`)：**ImageNet-Mini**
- Phase 2 stage-full：**ImageNet-Mini**
- Phase 2 stage-instance/fusion：**COCO train2017**

### 评测数据

- 我们的评测脚本（`eval_val50.py`）用的是 **COCO val2017**
- 本地已有 30 张 COCO val2017 图（`code/results/phase3_coco_val30_.../images/`），image ID 格式（000000016502 等）确认属于 val2017
- 计划扩充至 **50 张**，放在 `analysis/data/`
- 正式对比数字需要完整 5000 张，需要服务器

### 为什么用 COCO val2017

Su 2020 (Table 1/2) 的 COCO-Stuff validation split 底层图片就是 COCO val2017（COCO-Stuff 是在 COCO val2017 图片上加了 stuff 语义标注）。因此我们在 COCO val2017 上算的指标可以**直接和 Su 2020 论文数字对比**，口径一致。

---

## 3. 量化指标（关键：两篇论文用的指标完全不同）

### Zhang 2016 的指标（**没有 PSNR/SSIM**）

论文第 9 页 Table 1 明确使用三个指标：

| 指标 | 含义 | 论文数字（full model） |
|------|------|----------------------|
| **AuC** | ab 空间 L2 距离的累计分布面积（0-150 阈值扫描），rebalanced 变体按颜色频率重加权 | non-rebal 89.5% / rebal 67.3% |
| **AMT Turing Test** | Amazon Mechanical Turk 真人分辨真假，被骗率（Ground Truth = 50%） | **32.3%** |
| **VGG Top-1** | 把上色图喂给预训练 VGG-16 的分类准确率恢复比例 | 56.0% |

论文第 4 页明确写道 PSNR（MSE）的问题：
> *"the averaging effect favors grayish, desaturated results"*

Zhang 2016 **主动拒绝了 PSNR**，因为 PSNR 奖励"灰色平均"，和上色任务的目标相悖。

### Su 2020 的指标

论文 Table 1（全图级别）和 Table 2（实例级别），在 COCO-Stuff val 上：

| 指标 | 含义 | Su 2020 数字（COCO-Stuff val，finetuned） |
|------|------|----------------------------------------|
| **LPIPS ↓** | 深度感知相似度，越低越接近真实 | 全图 0.110 / 实例 0.095 |
| **PSNR ↑** | 峰值信噪比 (dB) | 全图 28.592 / 实例 29.522 |
| **SSIM ↑** | 结构相似度 [0,1] | 全图 0.944 / 实例 0.938 |

Su 2020 同时报告了**全图级别**和**实例级别**（GT bbox crop 内的指标）两个维度。

### 我们的指标方案

| 指标 | 现有？ | 操作 |
|------|-------|------|
| PSNR | ✅ `util/metrics.compute_psnr` | 直接用，对齐 Su 2020 |
| SSIM | ✅ `util/metrics.compute_ssim` | 直接用，对齐 Su 2020 |
| **LPIPS** | ❌ 完全没有 | **必须在 `_utils.py` 里加**；`pip install lpips`，10 行代码 |
| Bhattacharyya 距离 (BD) | ✅ `test.py:bhattacharyya_distance` | 我们自己加的，两篇论文都没用；保留作为色彩分布分析用，但**不能用来和论文数字对标** |
| AuC | ❌ | 仅 Zhang 2016 用；若想和它对齐可以加，优先级低 |
| AMT / VGG | ❌ | 需要真人众包或 VGG 推理，成本高；本次不做 |

**最终指标组合（和 Su 2020 对齐）**：LPIPS + PSNR + SSIM，全图级别 + 实例级别各报一次。

---

## 4. 可视化计划（六类，按优先级）

### V1. 三联对比条（P0，最基础）

**内容**：每行一张图，三格并排：`[灰度输入] | [模型预测彩色] | [GT 彩色]`

**用途**：最直观地展示"上没上对颜色"，每篇上色论文都有这个图。

**挑图原则**：
- 成功案例：语义颜色明确的（草地、天空、香蕉、交通标志）
- 失败案例：多物体复杂背景
- 两类各拼一张大图（5-8 行），**不要只展示好结果**

输出到：`phase{N}/vis/triplet_good.png` 和 `triplet_fail.png`

---

### V2. ab 散点密度图（P0，证明"不偏灰"）

**内容**：横轴 a，纵轴 b，所有图的所有像素打点；GT 点云一个颜色，预测点云另一个颜色，叠加在同一坐标系。

**用途**：直接回应 Zhang 2016 说的"MSE 损失导致偏灰"问题。
- 好模型：两个点云形状接近，都铺满整个 ab 空间
- 偏灰模型：预测点云塌缩在原点（0,0）附近

同时画 32×32 的 2D ab 直方图（热力图），方便看密度分布。

输出到：`phase{N}/ab_scatter.png`

---

### V3. 逐像素误差热力图（P1，定位失败区域）

**内容**：对每张图，把每像素的 ab L2 误差归一化后渲染成红色热力图，叠加在灰度图上。误差大的地方显示为红色/橙色。

**用途**：
- 展示"模型在哪里出错"（通常是物体边界、颜色多变的区域）
- Phase 2 vs Phase 1 对比时，Phase 2 在实例边界处误差应该更小

选 3-5 张典型图，每图四格：`[灰度] | [Phase1预测] | [Phase2预测] | [误差热力图对比]`

输出到：`phase2/error_heatmap/`

---

### V4. Colorfulness 箱线图（P1，色彩饱和度对比）

**内容**：对每张图算 Colorfulness 分数（Hasler & Süsstrunk 2003 公式：`σ_rg + σ_yb + 0.3*(μ_rg²+μ_yb²)^0.5`），分三组：`[GT] [Phase1] [Phase2]`，画箱线图并排。

**用途**：一眼看出哪个模型颜色更鲜艳，有没有出现"全图棕褐色"的退化情况。可以量化为"预测 Colorfulness / GT Colorfulness 的比值"，越接近 1 越好。

输出到：`phase{N}/colorfulness.png`

---

### V5. 融合权重热力图（P2，Phase 2 专属，待 fusion 权重）

**内容**：把 `WeightGenerator` 每一层的融合权重可视化，叠加在原图上，展示"模型在关注哪个实例区域"。

**用途**：Su 2020 论文 Figure 6 就是这个图，展示实例感知机制是否真正起作用。

挑有多个明显实例的图（如多辆车、多个人），展示不同层的权重分布变化。

输出到：`phase2/fusion_weights/`

⚠️ 需要 `inst_fusion` 权重，目前本地没有。

---

### V6. FiLM 消融对比（P2，我们最核心的创新验证，待 instance 权重）

**内容**：同一张图，同一个物体的 mask 区域：
```
[目标 mask 区域] | [无FiLM 的预测颜色] | [有FiLM 的预测颜色] | [GT]
```

**挑图原则**：专选语义颜色明确的物体——香蕉（黄）、停车标志（红）、草地（绿）、天空（蓝）。这类物体有正确答案，FiLM 的提升最清晰。

同时给出 mask 内 ab Huber 的数字（有 FiLM vs 无 FiLM）。

输出到：`phase2/film_ablation.csv` + `phase2/film_ablation.png`

⚠️ 需要 `inst_instance` 权重，目前本地没有。

---

## 5. data/ 文件夹说明

`analysis/data/` 放约 50 张 COCO val2017 的原始 GT 彩色图，作为所有分析的统一输入。

**来源**：本地 `code/results/phase3_coco_val30_.../images/` 已有 30 张，从中选 + 从服务器补至 50 张。

**选图标准**：
- 覆盖多种类别：单体（香蕉/交通标志）+ 多实例（多车/多人）+ 复杂背景
- 图像尺寸统一 resize 到 256×256 再推理（和训练一致）
- 图片本身是原始彩色图（GT），推理时临时转灰度作为输入

**这个文件夹只放图片，不放权重、不放代码。**

---

## 6. 执行计划

### 第一批（现在就能做，用 `data/` 50 张 + `inst_full/best`）

- [ ] 往 `data/` 里放 50 张 COCO val2017 图
- [ ] 写 `_utils.py`（LPIPS、Colorfulness、ab scatter 工具）
- [ ] 写 `run_phase2.py`（调用 `inst_full/best`，输出 V1/V2/V3/V4）
- [ ] 生成 `phase2/metrics.csv` 和 `phase2/metrics_summary.txt`

### 第二批（需要从服务器拷回权重）

从服务器需要拷的文件：
```
checkpoints/cnn_color/best_net_G.pth
checkpoints/inst_instance/best_net_G.pth
checkpoints/inst_fusion/net_fusion.pth
checkpoints/*/logs/（TensorBoard 日志）
```

- [ ] 补写 `run_phase1.py`（Phase 1 全套分析）
- [ ] Phase 1 vs Phase 2 三路对比表（full/instance/fusion）
- [ ] V5 融合权重热力图
- [ ] V6 FiLM 消融（有无 FiLM 的 mask 内 Huber + 可视化）
- [ ] 扩大到 500 张出正式对比数字

---

## 7. 论文对比表模板（TBD 等结果到位后填）

| 维度 | Zhang 2016 | Su 2020 | 我们 Phase 1 | 我们 Phase 2 |
|------|-----------|---------|-------------|-------------|
| 骨干结构 | 无 skip，H/4 输出 | skip + 全分辨率 | 同 Zhang 2016 | skip + 全分辨率 |
| 损失函数 | CE (rebalanced) | Smooth-L1 | CE (rebalanced) | **CE + Huber (λ=3)** |
| 语义感知 | 无 | 实例裁剪（不知类别） | 无 | **FiLM 注入类别标签** |
| LPIPS↓ (COCO val) | 0.238* | 0.110 | TBD | TBD |
| PSNR↑ (COCO val) | 21.791* | 28.592 | TBD | TBD |
| SSIM↑ (COCO val) | 0.892* | 0.944 | TBD | TBD |
| instance LPIPS↓ | 0.219* | 0.095 | — | TBD |
| instance PSNR↑ | 0.213* | 29.522 | — | TBD |
| Colorfulness 比值 | 偏低 | 略好 | TBD | TBD |
| Bhattacharyya 距离↓ | 较高 | 略低 | TBD | TBD |

*Su 2020 Table 1 中引用的 Zhang et al. [38] 数字（即 Zhang 2016），在 ImageNet 训练、无 finetune 条件下。

---

## 8. 备忘：重要实现细节

- **Phase 1 (`CnnColorGenerator`) 和 Phase 2 (`InstanceGenerator`) 不能共用权重**，网络结构不同，强行 load 会报 size mismatch（已验证）。
- **PSNR/SSIM 在 RGB [0,1] 空间算**，不在 Lab 空间算。
- **LPIPS 在 RGB [-1,1] 空间算**（lpips 库默认），注意 normalize。
- Bhattacharyya 距离在归一化后的 ab 通道 2D 直方图（32×32 bin，范围 [-1,1]）上算。
- `eval_val50.py` 只能跑 `InstanceGenerator`（stage-full），不含 FiLM 也不含 fusion。Phase 2 完整推理要走 `test.py --method inst_fusion`。
- `inst_full/best_net_G.pth` 在本地可正常推理，已验证：输出 `out_class (1,313,64,64)`，`out_reg (1,2,256,256)`。
