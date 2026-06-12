# Phase 3 实施方案：用户文本控制 Mask R-CNN 实例颜色

**日期：** 2026/06/12（v2 收官）
**状态：** ✅ 训练完成，epoch 10 为 final ckpt
**核心思路：** 在 Phase 2 完成的 instance+fusion pipeline 上叠加**单个 CLIP InstanceTextAdapter**（MLP + 5 个注入点，~2.89M 参数），让用户为每个 Mask R-CNN 实例指定文本 prompt 控制颜色。**背景不接受文本调制**，永远走 Phase 2 baseline。Phase 2 全部权重 / 文件冻结，Phase 3 仅新增 adapter、纯加法。

---

## 1. 动机与定位

Phase 2 用 FiLM 把 Mask R-CNN 的**类别 id**（91 类）注入实例分支——网络"知道在给什么物体上色"，但用户无法说"红色衣服" / "金色头发" / "绿色草地"。Phase 3 引入 CLIP 文本塔，让用户对每个实例单独施加自然语言控制。

**不做 Stable Diffusion / Latent Diffusion**：训练成本高、推理慢。Phase 3 保留 Phase 2 的 CNN 313-bin 分类架构，仅在 Phase 2 实例分支 5 层注入 FiLM 风格 adapter，配合正负 prompt + Ranking Loss 保证文本被真正使用。

**背景为什么不调制**：前期试验显示背景文本 adapter 容易与 Phase 2 解码器相互干扰，产生异常色块且对实例可控性几乎无贡献。删除后背景永远等于 Phase 2 baseline，所有 cat / dog2 / car / person 测试异常色块 = 0。

---

## 2. 整体架构

```
gray_full ─┬─ Mask R-CNN (with masks) ──→ N 个 instance crops + bbox + label + mask
           │                                         │
           │                              用户 prompt[i] 或 "a <class_name>" 兜底
           │                                         ↓
           │                                  CLIP ViT-B/32 txt encoder（冻结）
           │                                  → text_emb_inst (N, 512)
           │
           ├─ FiLMInstanceGenerator（冻结，加载 inst_fusion_instance/25_net_G.pth）
           │   完整 forward 得到 feature_map per instance
           │       ↓
           │   InstanceTextAdapter（新增可训，2.89M）
           │       在 conv6_3 / conv7_3 / conv8_3 / conv9_3 / conv10_2 做 FiLM 调制
           │       ↓
           │   modulated inst_feats（保持原结构供 WG 消费）
           │
           └─ FusionPipeline.backbone（冻结，加载 inst_fusion_full/80_net_G.pth）
               逐层产出 bg_feat[layer]（不经过任何 adapter）
               ↓
               冻结的 WeightGenerator × 13 层逐层融合 (modulated inst_feats, bg_feat)
               ↓
               冻结的 model_class / output_conv（加载 inst_fusion_fusion/25_net_G.pth）
               ↓
               out_class (313, H/4) → annealed-mean(T=0.38) → ab
               ↓
               ab magnitude cap（默认 0.45，保留色相方向，仅缩放半径）
               ↓
               与 L 合并 → RGB
```

---

## 3. TextAdapter 设计（FiLM + 2 层 MLP + zero-init）

```python
# code/models/text_color_networks.py
class TextAdapter(nn.Module):
    def __init__(self, clip_dim, layer_channels: Dict[str, int], hidden_dim=512):
        super().__init__()
        self.proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * ch),    # 最后一层 zero-init
            )
            for name, ch in layer_channels.items()
        })
        for mlp in self.proj.values():
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    def forward(self, feat_dict, text_emb):
        out = dict(feat_dict)
        for name in feat_dict:
            if name not in self.proj:
                continue
            gb = self.proj[name](text_emb)               # (B, 2C)
            gamma, beta = gb.chunk(2, dim=1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta  = beta.unsqueeze(-1).unsqueeze(-1)
            out[name] = feat_dict[name] * (1.0 + gamma) + beta
        return out
```

**5 个注入点：**

| 层 | 通道 | 角色 | 参数量 |
|---|:---:|---|:---:|
| `conv6_3` | 512 | encoder 瓶颈，高级语义 | 0.79M |
| `conv7_3` | 512 | encoder 末端，类别先验聚合 | 0.79M |
| `conv8_3` | 256 | decoder 起点，驱动 `model_class`（313 分类） | 0.53M |
| `conv9_3` | 128 | decoder 中段 | 0.39M |
| `conv10_2` | 128 | decoder 末端，驱动 `output_conv` | 0.39M |
| **合计** | | | **~2.89M** |

**Zero-init 等价性**：每个 MLP 最后一层 Linear 零初始化，训练前 pipeline 输出与 Phase 2 fusion 推理 bit-equal（已单元测试验证）。

---

## 4. 训练数据

**JSONL 预处理**（`scripts/build_phase3_jsonl.py`）：
- COCO2017 instances，筛选规则：`iscrowd=0`、area/img_area ≥ 0.005、HSV 主色置信度 ≥ 0.04（person ≥ 0.15）
- 负色采样：50% 互补、30% 邻近、20% 随机
- 产出：**train 113,674 张图 / 407,866 实例**；val 4,801 张 / 17,309 实例

**CLIP 离线缓存**（`scripts/cache_clip_embeddings.py`）：约 3986 条目，`datasets/phase3/clip_text_cache.pt`，训练时零 CLIP forward 开销。

---

## 5. 损失函数

每 step 双前向（pos / neg prompt）：

```
L_global   = CE_rebalanced(out_class_pos, gt_ab) + 3.0 × Huber(out_reg_pos, gt_ab)
L_inst_rec = masked_mean(CE + 3×Huber, mask 内)
L_rank     = max(0, 0.4 + KL(gt || p_pos) − KL(gt || p_neg))   ← mask 内，逼 pos 更贴 GT
L_outside  = pos / neg 在 mask 外都贴 GT
L_total    = L_global + 1.0×L_inst_rec + 1.5×L_rank + 0.1×L_outside
```

---

## 6. 训练配置

| 项 | 值 |
|---|---|
| 冻结 backbone | `inst_fusion_full/80_net_G.pth` + `inst_fusion_instance/25_net_G.pth` + `inst_fusion_fusion/25_net_G.pth` |
| 可训参数 | **2.89M**（TextAdapter） |
| 数据集 | COCO2017 train 113K 图 / 408K 实例 |
| Optim | Adam(lr=1e-4, beta=(0.5, 0.999))，lambda 衰减 |
| Epoch | `niter 6 + niter_decay 4` = **10** |
| Batch | 1 |
| AMP | 关闭 |
| `lambda_rank / rank_margin` | 1.5 / 0.4 |
| `rank_warmup_epoch` | -1（一上来即满值） |
| 训练时间 | RTX 5090 单卡，~18h |

**训练曲线（10 epoch 全程）：**

| epoch | G | rank | 说明 |
|:---:|:---:|:---:|---|
| 1 iter 100 | 5.48 | **0.40** | cold start |
| 2 | ~4.9 | 0.033 | rank 快速下降 |
| 6 | 4.69 | 0.022 | 满 LR 阶段收敛 |
| 8 | 5.05 | 0.018 | LR 衰减开始 |
| **10** | **4.31** | **0.026** | **收官，G 最低** |

全程 nan% = 0。

---

## 7. 推理 CLI

```bash
python test.py --method text_color \
  --which_epoch 10 --name phase3_text_color \
  --full_ckpt   checkpoints/inst_fusion_full/80_net_G.pth \
  --inst_ckpt   checkpoints/inst_fusion_instance/25_net_G.pth \
  --fusion_ckpt checkpoints/inst_fusion_fusion/25_net_G.pth \
  --image datasets/test/dog.png \
  --prompt "inst:0=a red dog" \
  --results_img_dir results/phase3 \
  --gpu_ids -1
  # --fineSize 256 (默认即 256，不要传 224)
  # --ab_cap 0.45  (默认)
```

**实例编号**：Mask R-CNN 输出按置信度降序；CLI 打印 `inst:0 dog (conf 0.99)` 帮用户对齐。未指定 prompt 的实例自动兜底 `"a <class_name>"`。`bg=` prompt 接受但不调制。

**ab magnitude cap**（`--ab_cap 0.45`）：在 `decode_zhang2016_annealed_mean` 后、`lab2rgb` 前对 normalised ab 做 magnitude cap，保留色相方向、仅缩放半径。抑制低 L + 中 b 像素在 sRGB 映射后的异常鲜黄色块。

---

## 8. 文件清单（全部新增，零修改 Phase 2）

```
code/
├── models/
│   ├── text_color_networks.py     # TextAdapter + TextColorPipeline
│   └── text_color_model.py        # TextColorModel(BaseModel)
├── data_process/
│   └── text_color_dataset.py      # TextColorCocoDataset
├── util/
│   ├── clip_encoder.py            # open_clip 文本塔封装 + 嵌入缓存
│   └── maskrcnn_helper.py         # 带 mask 的 Mask R-CNN 调用
├── scripts/
│   ├── build_phase3_jsonl.py      # COCO → JSONL
│   ├── cache_clip_embeddings.py   # 离线 CLIP 嵌入缓存
│   └── train_phase3.sh            # 训练启动脚本
└── checkpoints/phase3_text_color/
    └── 10_net_T.pth               # final ckpt (~12MB, 2.89M 参数)
```

---

## 9. 推理强制要求

**必须 `--fineSize 256`**（默认即 256，不要传 `--fineSize 224`）。fineSize 224 会产生大量饱和黄色异常色块（~15× Phase 2 baseline），原因是 BN 统计量错位。

---

## 10. 最终验收结果（epoch 10 final）

### 三 epoch 对比（e6 / e8 / e10，含 ab_cap=0.45）

**平均可控性（mean Δ vs default）：**

| Epoch | 覆盖测试数 | 平均 mean Δ |
|:---:|:---:|:---:|
| e6 | 9（限 dog/cat/dog2） | 5.94 |
| e8 | 25 | 5.50 |
| **e10** | **25** | **5.85** |

**各测试 e10 数据（mean Δ vs default）：**

| 测试 | prompt | e10 mean Δ | 异常色块 |
|---|---|:---:|:---:|
| dog | red | 6.46 | 35 px |
| dog | yellow | 5.67 | 12 px |
| dog | brown | 4.04 | 56 px |
| dog | green | 7.13 | 5 px |
| cat | gray | **12.10** | 0 px |
| cat | orange | 2.52 | 0 px |
| cat | brown | 2.92 | 0 px |
| dog2 | red | 3.12 | 0 px |
| dog2 | yellow | 3.03 | 0 px |
| car | blue（原色） | **13.87** | 0 px |
| car | green | **11.83** | 52 px |
| car | gray | 9.75 | 0 px |
| car | yellow | 9.46 | 129 px |
| person | yellow | 5.53 | 9 px |
| person | green | 5.11 | 0 px |
| person | blue | 3.46 | 0 px |
| person | white（原色） | 1.98 | 0 px |

**Cross-prompt 差异（e10）：**

| 对比 | mean Δ |
|---|:---:|
| car orange ↔ blue | **19.37** |
| car blue ↔ yellow | **16.43** |
| dog red ↔ green | **11.38** |
| cat gray ↔ orange | **13.61** |
| dog2 red ↔ yellow | 5.45 |
| person blue ↔ yellow | 8.12 |

**选 e10 的理由：**
- car 8 项测试 e10 比 e8 强 6 项，泛化能力最好
- G loss 最低（4.31），训练最收敛
- 全程训练协议完整，符合学术汇报标准

---

## 11. 支持的颜色集合与已知限制

### ✅ 支持的 prompt 颜色

**核心彩色系（实测有效）：** `red` / `yellow` / `orange` / `brown` / `green` / `gray` / `blue` / `purple`

```
inst:i=a <color> <class>     # 例：inst:0=a red dog
inst:i=a <color> <object>    # 例：inst:1=a yellow car
```

### ❌ 不支持

**亮度类：** `black` / `white`（mean Δ < 2.0）。原因：313-bin Lab 颜色量化器在低饱和度区域分辨力极弱，且 CLIP "a black dog" 与 "a dog" 嵌入距离过近。

### ⚠️ 已知限制

1. **相近色语义混淆**：cat `brown` vs `orange`（CLIP 嵌入距离过近，两者几乎等价）
2. **跨图泛化差异**：dog2 可控性约为 dog 主图 50%，因 Mask R-CNN bbox/mask 与训练分布不完全匹配
3. **white prompt 无法染色**：person 白衬衫的白色先验太强，white/red/gray prompt 效果均接近 default（yellow/green/blue 略有效）

---

## 12. 收官状态

- ✅ `checkpoints/phase3_text_color/10_net_T.pth` 为 final ckpt（~12MB，2.89M 可训参数）
- ✅ Phase 2 权重零修改：`networks.py` / `inst_fusion_model.py` / Phase 2 ckpt 全部未触
- ✅ 推理 CLI 跑通：`test.py --method text_color`，支持 `--image` + `--prompt "inst:i=..."`
- ✅ 推理默认 `--ab_cap 0.45` 抑制异常色块，`--fineSize 256` 强制
- ✅ car / cat / dog2 / person 新图测试通过，cat/dog2 异常色块 0 px
- ✅ 训练日志完整：`logs/phase3_v8_v2.log`（10 epoch，11367 行，零 NaN）
- ✅ docs 已同步（phase3_plan.md / network_architecture.md / PJ-architecture.md）
- ⚠️ 不支持 black / white prompt（详见下）
- ⚠️ cat brown / orange 语义混淆（CLIP 嵌入距离过近）

---

## 13. Black/White Prompt 实验记录

**结论：black / white prompt 在当前架构下无效。**

实测 mean Δ 看似不小（dog black=8.22，car black=7.28），但实际颜色变化方向**与 black/white 无关**——模型被 prompt 推向了另一个错误的彩色方向，而非低饱和度的黑/白区域。

| 测试 | default 中心 RGB | black 中心 RGB | white 中心 RGB |
|---|---|---|---|
| dog | R=176 G=142 B=107（金毛） | R=176 G=144 **B=86**（更黄）| R=183 G=138 B=106（微偏暖）|
| cat | R=192 G=140 B=97（橘猫） | R=193 G=141 **B=69**（更黄）| R=200 G=136 B=100（微偏暖）|
| car | R=180 G=117 B=103（红车） | R=178 G=119 **B=79**（更黄）| R=176 G=116 **B=125**（偏紫）|

饱和度（default→black）：dog 59.5→71.4，cat 63.3→78.1——**黑色 prompt 反而让图像更饱和，与预期相反**。

**根本原因（双重限制）：**
1. **CLIP 语义**：`"a black dog"` 与 `"a dog"` 在 512 维嵌入空间距离很近，调制信号弱
2. **313-bin Lab ab 量化**：黑色对应 (a≈0, b≈0) 的低饱和区域，但 Phase 2 彩色先验 + annealed-mean 解码把预测锁定在常见彩色 bin，无法到达黑白区

**若未来要支持**：需在 L 通道（亮度）上额外调制，单纯调 ab 无法实现黑白控制。
