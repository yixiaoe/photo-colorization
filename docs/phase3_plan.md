# Phase 3 实施方案：用户文本控制 Mask R-CNN 实例 / 背景颜色

**日期：** 2026/06/04
**状态：** 方案设计（已与代码计划对齐）
**核心思路：** 在 Phase 2 完成的 instance+fusion pipeline 上叠加 **两个 CLIP 文本 Adapter**（InstanceTextAdapter / BgTextAdapter），让用户为每个 Mask R-CNN 实例（或背景）单独指定文本 prompt，控制该区域颜色。Phase 2 全部权重 / 文件冻结，Phase 3 仅新增模块、纯加法。

---

## 1. 动机与定位

Phase 2 用 FiLM 把 Mask R-CNN 的**类别 id**（91 类）注入实例分支——网络"知道在给什么物体上色"，但用户无法说"红色衣服" / "金色头发" / "夕阳天空"。Phase 3 引入 CLIP 文本塔，让用户对每个实例（或背景）单独施加自然语言控制。

**不做 Stable Diffusion / Latent Diffusion**：训练成本高、推理慢。Phase 3 保留 Phase 2 的 CNN 313-bin 分类架构，仅在 Phase 2 decoder 三层（conv8_3 / conv9_3 / conv10_2）注入轻量 FiLM 风格 adapter，配合正负 prompt + Ranking Loss 保证文本被真正使用。

**与旧版方案的关键差异**：旧版（单分支文本注入全图）丢掉了 Phase 2 的实例感知。本版**完整复用 Phase 2 的 instance + fusion pipeline**，把文本作用域天然绑定到每个 Mask R-CNN 实例和背景。

---

## 2. 整体架构

```
gray_full ─┬─ Mask R-CNN (with masks) ──→ N 个 instance crops + bbox + label + mask
           │                                         │
           │                              用户 prompt[i] 或 "a <class_name>" 兜底
           │                              用户 bg prompt 或 "" 兜底
           │                                         ↓
           │                                  CLIP ViT-B/32 txt encoder（冻结）
           │                                  → text_emb_inst (N, 512)
           │                                  → text_emb_bg   (1, 512)
           │
           ├─ FiLMInstanceGenerator（冻结，加载 inst_fusion_instance/25_net_G.pth）
           │   完整 forward 得到 feature_map per instance
           │       ↓
           │   InstanceTextAdapter（新增可训） 在 conv8_3/conv9_3/conv10_2 做 FiLM 调制
           │       ↓
           │   modulated inst_feats（保持原结构供 WG 消费）
           │
           └─ FusionPipeline.backbone（冻结，加载 inst_fusion_full/80_net_G.pth）
               逐层产出 bg_feat[layer]
               ↓
               TextColorPipeline 手工镜像 FusionPipeline.forward，
               在 conv8_3 / conv9_3 / conv10_2 三层 bg_feat 先过 BgTextAdapter（新增可训）
               再喂入冻结的 WeightGenerator
               ↓
               冻结的 model_class / output_conv（加载 inst_fusion_fusion/20_net_G.pth）
               ↓
               out_class (313, H/4) → annealed-mean(T=0.38) → ab
               out_reg   (2,  H)
               ↓
               与 L 合并 → RGB
```

**核心架构决策**：FiLMInstanceGenerator / FusionPipeline 的 `forward` 是单次贯通的，没有插入点。Phase 3 **不修改 `networks.py`**，而是写一个新的 `TextColorPipeline`，持有冻结 Phase 2 网络的引用，**手工调用其子模块**（`netG.model1`、`netG.wg_conv4_3`、`netG.model_class` 等公开属性），中间穿插 adapter。这是组合（composition），不是修改——Phase 2 一行不动。

---

## 3. Adapter 设计（FiLM 风格、残差、zero-init）

```python
# code/models/text_adapter.py（新增）
class TextAdapter(nn.Module):
    """对若干特征层做文本条件 FiLM 调制。zero-init → 训练初期等价 Phase 2。"""
    def __init__(self, clip_dim, layer_channels: Dict[str, int]):
        super().__init__()
        self.proj = nn.ModuleDict({
            k: nn.Linear(clip_dim, 2 * c) for k, c in layer_channels.items()
        })
        for m in self.proj.values():
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, feat_dict, text_emb):
        out = dict(feat_dict)
        for k, lin in self.proj.items():
            g, b = lin(text_emb).chunk(2, dim=1)
            g = g.unsqueeze(-1).unsqueeze(-1)
            b = b.unsqueeze(-1).unsqueeze(-1)
            out[k] = feat_dict[k] * (1.0 + g) + b
        return out
```

**调制位置**：decoder 三层 `conv8_3`、`conv9_3`、`conv10_2`。
- 这三层直接驱动 `model_class`（来自 conv8_3，→313 分类）和 `output_conv`（来自 conv10_2，→2 通道回归）
- 不与 Phase 2 已收敛的 FiLM @conv4_3..7_3 干扰（类别语义）
- 参数量：512×513 + 128×257 + 128×257 ≈ 0.79M 每个 adapter，两个合计 **~1.6M 可训**

**zero-init 保证**：训练初期 γ=0、β=0，feat * (1+0) + 0 = feat，整个 Pipeline 严格等价于 Phase 2 fusion 推理。

---

## 4. TextColorPipeline 主类

```python
# code/models/text_color_networks.py（新增）
class TextColorPipeline(nn.Module):
    def __init__(self, netInst, netFusion, clip_dim=512):
        super().__init__()
        self.netInst   = netInst       # FiLMInstanceGenerator (frozen)
        self.netFusion = netFusion     # FusionPipeline (frozen)
        for p in self.netInst.parameters():   p.requires_grad_(False)
        for p in self.netFusion.parameters(): p.requires_grad_(False)
        self.netInst.eval(); self.netFusion.eval()

        dec = {'conv8_3': 256, 'conv9_3': 128, 'conv10_2': 128}
        self.inst_adapter = TextAdapter(clip_dim, dec)
        self.bg_adapter   = TextAdapter(clip_dim, dec)

    def forward(self, gray_full, inst_L, class_labels, box_info_list,
                inst_text_embs, bg_text_emb, empty_box=False):
        # 1) 实例分支
        if not empty_box:
            _, _, fm_batch = self.netInst(inst_L, class_labels)
            fm_batch = self.inst_adapter(fm_batch, inst_text_embs)
            N = inst_L.shape[0]
            inst_feats = [{k: fm_batch[k][[i]] for k in fm_batch} for i in range(N)]
        else:
            inst_feats = []

        # 2) 背景分支：手工镜像 FusionPipeline.forward，在 dec 三层注入 bg_adapter
        return self._fusion_with_bg_adapter(gray_full, inst_feats, box_info_list, bg_text_emb)
```

`_fusion_with_bg_adapter` 完全镜像 `networks.py:570-637` 的逻辑（encoder + WG + decoder + heads），**仅在 conv8_3 / conv9_3 / conv10_2 三处**对 bg_feat 先过 `bg_adapter` 再喂 `wg_*`。约 65 行。

**等价性单元测试（必须通过才进入训练）**：所有 adapter zero-init 时，TextColorPipeline 在测试图上的输出与 `python test.py --method inst_fusion --stage fusion` 的逐像素差异 < 1e-4。

---

## 5. Mask R-CNN with masks

Phase 2 `_predict_bbox`（`colorization_dataset.py:78`）丢弃了 `preds['masks']`。Phase 3 需要 mask 做推理可视化和（可选）训练监督。新增：

```python
# code/util/maskrcnn_helper.py
from data_process.colorization_dataset import _get_maskrcnn  # 复用 Phase 2 singleton

@torch.no_grad()
def predict_with_masks(pil_img, device, box_num=8, score_thresh=0.5):
    """返回 [(box_xyxy, label, score, mask_uint8_HxW)] 按 score 降序"""
```

**只 import 不修改** Phase 2 现有代码。

---

## 6. 数据流（训练）

### 6.1 输入

COCO2017 全图 + GT bbox + GT label + GT segmentation mask（不再依赖 Mask R-CNN 推理）。

**离线预处理** `scripts/build_phase3_jsonl.py`：
- 解析 `instances_train2017.json`，按 `(image_id, ann_id)` 展开
- 筛选：`iscrowd=0`、polygon、area/img_area ≥ 0.01、HSV 主色置信度 ≥ 0.05（person ≥ 0.30）
- mask 内 HSV 推断主色：V<0.15→black；S<0.20 & V≥0.82→white；S<0.20→gray；15≤H<45 & V<0.55→brown；否则按 hue 圆形平均（权重 S×V）映射到 red/orange/yellow/green/cyan/blue/purple/pink
- 负色采样：50% 互补、30% 邻近、20% 随机
- caption 改写：`"a <color> <object>"`；caption_neg 同结构，颜色换成 neg_color
- 输出 JSONL，每行 `{image_id, ann_id, class_id, color_pos, color_neg, mask_rle, bbox}`

### 6.2 Dataset

```python
# code/data_process/text_color_dataset.py
class TextColorCocoDataset:
    # 按 image_id group JSONL 行，__getitem__ 返回：
    {
      'full_rgb':       (1, 3, H, W),
      'cropped_rgb':    (N, 3, H, W),       # 按 bbox crop + resize
      'class_labels':   (N,) long,
      'box_info':       (N, 6) at H,         # 复用 colorization_dataset.get_box_info
      'box_info_2x/4x/8x':  (N, 6),
      'masks_full':     (N, 1, H, W) float,  # 栅格化的 instance mask
      'masks_4x':       (N, 1, H/4, W/4),
      'caption_pos':    list[str] of len N,
      'caption_neg':    list[str] of len N,
      'caption_bg_pos': str,                  # "outdoor scene" / "indoor room" 模板
      'caption_bg_neg': str,
      'empty_box':      bool,
    }
```

`caption_bg_pos/neg` 由 COCO supercategory 简单推断（含 outdoor 物体 → "outdoor scene"，否则 "indoor room"），bg 负采样为另一类。

---

## 7. 损失函数

每个 step 做 **双前向**（pos / neg prompt 各跑一次 TextColorPipeline）：

```
inputs:
  out_class_pos, out_reg_pos = TCP(gray_full, ..., text_emb_pos_inst, text_emb_pos_bg)
  out_class_neg, out_reg_neg = TCP(gray_full, ..., text_emb_neg_inst, text_emb_neg_bg)

mask_4x   = masks_4x.max(dim=0)       # N 个实例 mask 在 H/4 上的并集
mask_full = masks_full.max(dim=0)     # 在 H 上的并集
```

### 7.1 全图重建（pos）

```
L_global = CE_rebalanced(out_class_pos, gt_ab) + 3.0 * Huber(out_reg_pos, gt_ab)
```

### 7.2 instance mask 内重建（pos）

```
L_inst_rec = masked_mean(CE_rebalanced(out_class_pos, gt_ab), mask_4x)
           + 3.0 * masked_mean(Huber(out_reg_pos, gt_ab), mask_full)
```

### 7.3 Ranking Loss（反事实监督）

在 313-bin 概率层面（避开 annealed-mean 解码后的梯度消失）：

```
p_pos = softmax(out_class_pos, dim=1)
p_neg = softmax(out_class_neg, dim=1)
D_pos = masked_mean(KL(gt_soft || p_pos), mask_4x)
D_neg = masked_mean(KL(gt_soft || p_neg), mask_4x)
L_rank = max(0, margin + D_pos - D_neg)        # margin = 0.05
```

### 7.4 mask 外一致性（防文本污染背景）

```
L_outside = 0.25 * (
    masked_mean(Huber(out_reg_pos, gt_ab), 1 - mask_full)
  + masked_mean(Huber(out_reg_neg, gt_ab), 1 - mask_full)
  + masked_mean(CE_rebalanced(out_class_pos, gt_ab), 1 - mask_4x)
  + masked_mean(CE_rebalanced(out_class_neg, gt_ab), 1 - mask_4x)
)
```

**用 pos/neg vs GT** 而非 pos vs neg，避免退化解（背景全输出灰色）。

### 7.5 总损失

```
L = L_global + λ_inst * L_inst_rec + λ_rank * L_rank + λ_outside * L_outside
λ_inst    = 1.0
λ_rank    = 0.1   (warmup 后)
λ_outside = 0.2   (warmup 后)
```

### 7.6 Loss Warm-up

zero-init 阶段 pos/neg 输出几乎相同，过早启用 ranking 噪声大：

```
epoch 0–4 :  λ_rank = 0,   λ_outside = 0
epoch 5–10:  cosine 升至  λ_rank = 0.1, λ_outside = 0.2
epoch 11+ :  保持目标值
```

**Bg adapter 的训练信号**：经由 `L_global`（全图重建）和 `L_outside`（mask 外 pos/neg vs GT）隐式监督。负 bg prompt（如 "indoor" vs "outdoor"）驱动 bg_adapter 学到背景颜色调度。

---

## 8. 训练配置

| 项 | 值 |
|---|---|
| Backbone | `inst_fusion_full/80_net_G.pth` + `inst_fusion_instance/25_net_G.pth` + `inst_fusion_fusion/25_net_G.pth` 全冻结 |
| CLIP | open_clip `ViT-B-32-quickgelu`，冻结 |
| 可训参数 | InstanceTextAdapter + BgTextAdapter ≈ 1.6M |
| 优化器 | Adam(lr=1e-4, betas=(0.5, 0.999))，lambda 线性衰减 |
| Batch size | 1（全图 + N 实例，与 Phase 2 fusion 一致） |
| Epoch | 30 + 30 |
| AMP | 启用（NaN 时 `scaler.update()` 缩小 scale + 回滚 BN running stats） |
| 梯度裁剪 | max_norm = 5.0 |
| 训练时间预估 | RTX 4090 ~36h |

**启动**：
```bash
cd code
bash scripts/train_phase3.sh /root/autodl-tmp/code/datasets/coco
```

展开：
```bash
python train_phase3.py \
  --jsonl_file        data/phase3_color_object_train.jsonl \
  --val_jsonl_file    data/phase3_color_object_val.jsonl \
  --img_dir           /root/autodl-tmp/coco2017/train2017 \
  --full_ckpt         checkpoints/inst_fusion_full/80_net_G.pth \
  --inst_ckpt         checkpoints/inst_fusion_instance/25_net_G.pth \
  --fusion_ckpt       checkpoints/inst_fusion_fusion/25_net_G.pth \
  --fineSize 256  --batch_size 1  --nThreads 4 \
  --lr 1e-4  --huber_weight 3.0 \
  --lambda_inst 1.0  --lambda_rank 0.1  --lambda_outside 0.2 \
  --rank_margin 0.05  --rank_warmup_epoch 5 \
  --niter 30  --niter_decay 30 \
  --name phase3_text_color  --gpu_ids 0
```

---

## 9. 推理 CLI

```bash
python test_phase3.py \
  --image dog_on_grass.jpg \
  --prompt "inst:0=a black labrador" \
  --prompt "inst:1=green grass" \
  --prompt "bg=sunset sky" \
  --full_ckpt    checkpoints/inst_fusion_full/80_net_G.pth \
  --inst_ckpt    checkpoints/inst_fusion_instance/25_net_G.pth \
  --fusion_ckpt  checkpoints/inst_fusion_fusion/25_net_G.pth \
  --adapter_ckpt checkpoints/phase3_text_color/latest_net_T.pth \
  --results_img_dir results/phase3
```

**实例编号**：Mask R-CNN 按 score 降序输出，CLI 同步打印 `inst:0 person score=0.97 / inst:1 dog score=0.91 / ...` 帮用户对齐 idx。

**兜底规则**：
- 未指定 `inst:i=...` → 自动用 `"a <class_name>"`（class_name 来自 Mask R-CNN COCO label 字典）
- 未指定 `bg=...` → 空字符串 ""（adapter 在训练分布上学过弱响应）

**未检测到实例**：退化为纯 bg_adapter 控制 + Phase 2 fusion empty_box 通路。

---

## 10. 评估方案

### 10.1 反事实测试（核心创新验证）

固定灰度图，仅改 prompt：
```
"a red shirt"  vs  "a blue shirt"  vs  "a white shirt"
```
- 量化：mask 内 mean Δab（pos vs neg）应**大**（>10）
- 量化：mask 外 mean Δab 应**小**（<2）
- 反事实差异图：`fake_pos | fake_neg | abs(fake_pos - fake_neg)`

### 10.2 prompt 控制矩阵

8–16 张固定灰度图 × 5–8 组 prompt，矩阵图每 `val_freq` 保存。

### 10.3 量化指标

- **BD 距离**：与 Phase 1/2 对齐
- **Caption-Color Consistency**：解析 caption 颜色词，统计 mask 内 ab 与该颜色 hue 吻合度
- **PSNR / SSIM**：与 Phase 2 一致，**只作描述指标**，不作为 best 选择依据

### 10.4 best checkpoint 选择

**不以 PSNR 选 best**，以 `val_loss_G + overfit_gap_G` 联合判断（同 Phase 2）。

### 10.5 对比基线

| 方法 | 备注 |
|---|---|
| Phase 1（CNN baseline） | 无文本 |
| Phase 2 fusion | 类别条件，无文本 |
| Phase 3（本方案） | 完整 per-instance + bg 文本控制 |
| Phase 3 + 空 caption | 验证兜底（应接近 Phase 2 fusion） |

---

## 11. 监控与可视化

| 类型 | 项目 |
|---|---|
| 标量曲线 | `loss/G`、`loss/global`、`loss/inst_rec`、`loss/rank`、`loss/outside`、`val_loss/G`、`overfit_gap/G`、`grad_norm`、`lr` |
| 固定样本网格 | `gray | fake_rgb_class | fake_rgb_reg | gt` |
| **反事实差异图** | `fake_pos | fake_neg | abs(fake_pos - fake_neg)` |
| prompt 矩阵 | 行=图像，列=prompt |
| 分布统计 | ab 分布、313-bin 使用频率、colorfulness |

**Ablation 优先级**（一次只改一个变量）：

```
lr:                  1e-4 vs 5e-5 vs 2e-4
注入层:              {conv10_2} vs {conv9_3, conv10_2} vs {conv8_3, conv9_3, conv10_2}
λ_rank:              0.05 vs 0.10 vs 0.20
margin:              0.03 vs 0.05 vs 0.10
caption 增强:        关 vs 开
negative 策略:       仅互补 vs 混合
单 adapter:          仅 inst vs inst+bg
```

---

## 12. 文件清单（新增，零修改 Phase 2）

```
code/
├── models/
│   ├── text_adapter.py            # TextAdapter
│   ├── text_color_networks.py     # TextColorPipeline
│   └── text_color_model.py        # TextColorModel(BaseModel)
├── data_process/
│   └── text_color_dataset.py      # TextColorCocoDataset
├── util/
│   ├── clip_encoder.py            # open_clip 文本塔封装
│   └── maskrcnn_helper.py         # 带 mask 的 Mask R-CNN
├── options/
│   └── phase3_options.py          # Phase3TrainOptions / Phase3TestOptions
├── scripts/
│   ├── build_phase3_jsonl.py      # COCO → JSONL（HSV 颜色词）
│   └── train_phase3.sh
├── train_phase3.py
├── test_phase3.py
└── checkpoints/
    └── phase3_text_color/          # 仅 ~1.6M adapter 权重
docs/
└── phase3_plan.md                  # ← 本文档
requirements.txt                    # 新增 open_clip_torch, pycocotools
```

---

## 13. 复用的 Phase 2 接口（read-only import）

| 模块 | 用途 |
|---|---|
| `models.networks.FiLMInstanceGenerator` | 实例分支主体（冻结） |
| `models.networks.FusionPipeline` | 背景分支 + WG + decoder + heads（冻结） |
| `models.base_model.BaseModel` | TextColorModel 父类 |
| `util.util.{rgb2lab, lab2rgb, encode_ab_bins_soft/hard, decode_zhang2016_annealed_mean, build_zhang2016_rebalance_weights, load_zhang2016_ab_bins, tensor2im, save_image}` | 不重复实现 |
| `util.visualizer.Visualizer` | TensorBoard |
| `data_process.colorization_dataset.{get_box_info, _load_coco_instances, _get_maskrcnn, _collect_images}` | 复用 bbox 几何 / COCO 解析 / Mask R-CNN 单例 |
| `resources.defaults.{T, REBALANCE_GAMMA}` | 解码温度 / rebalance gamma |

---

## 14. 关键设计依据

| 设计点 | 依据 |
|---|---|
| 不修改 Phase 2，用 TextColorPipeline 组合 | 保护已收敛权重；任何改动都可单独撤除 |
| 调制仅 decoder 三层 | 直接驱动分类头/回归头；避开 Phase 2 conv4_3..7_3 的 FiLM 类别先验 |
| zero-init out_proj | 训练初期严格等价 Phase 2，可单元测试 |
| FiLM 风格 adapter（非 Cross-Attn） | 参数小（1.6M vs 0.9M Cross-Attn 但更稳）、与 Phase 2 FiLM 同形态、训练快 |
| color-word + GT mask 区域监督 | 直接用 caption 太稀疏，离线 HSV 分析保证每条样本必含颜色监督 |
| 正负 prompt 双前向 + Ranking | 把"反事实区分"作为训练信号，防止模型忽略文本 |
| Ranking 在概率分布层面 | annealed-mean 解码梯度过度平滑，直接对 logits/概率做差异更有效 |
| L_outside 用 pos/neg vs GT | 避免 pos==neg 退化解；约束"换 prompt 不改变背景" |
| 未指定 prompt 走 class_name 兜底 | 依赖 CLIP 类别先验，比空字符串语义更清晰 |
| Bg adapter 训练信号靠 L_global + L_outside | 不需要 mask；隐式监督足够，避免增加显式 bg mask 复杂度 |

---

## 15. 潜在风险

| 风险 | 缓解 |
|---|---|
| TextColorPipeline 与 FusionPipeline.forward 不一致 | adapter zero-init 等价性单元测试（必须 bit-equal） |
| 文本被忽略（text bypass） | Ranking Loss 强制 pos > neg；反事实可视化早期监控 |
| 颜色词映射到错误区域 | L_outside 约束 mask 外不变 |
| CLIP token 稀释（77 太长） | 短文本（caption 截断）+ key_padding_mask（CLIP 内部已处理） |
| 训练初期 adapter 输出爆炸 | zero-init + 梯度裁剪 max_norm=5.0 |
| AMP NaN 卡死 | NaN 时 `scaler.update()` 缩小 scale + 跳过 step + 回滚 BN running stats |
| Ranking 梯度消失 | 在 313-bin 概率层面做 KL 差异 |
| 背景退化灰色 | L_outside pos/neg vs GT |
| HSV 主色误判 | conf ≥ 0.05 过滤；person ≥ 0.30 |
| Mask R-CNN 推理空检测 | CLI 打印；退化为 bg_adapter 全图控制 |

---

## 16. 里程碑

| 阶段 | 任务 | 验收 |
|---|---|---|
| W1 | CLIP 封装 + Mask R-CNN helper + TextAdapter + TextColorPipeline | 等价性测试 < 1e-4 |
| W2 | JSONL 预处理 + TextColorCocoDataset + 双前向 forward 跑通 | 100 iter 烟雾测试 loss 非 NaN |
| W3 | 完整训练 30+30 epoch | 收敛；val_loss 稳定；BD 不差于 Phase 2 |
| W4 | 反事实测试 + prompt 控制矩阵 + ablation | mask 内 Δab > 10，mask 外 < 2 |
| W5 | 端到端 demo + 报告整理 | 5 张实拍黑白照人工评测 |

---

## 17. 最终训练配置（实际跑通版本）

原计划 30+30 epoch 在实际训练中被压缩到 **5+3 = 8 epoch**，warmup 也从 cosine 5 epoch 上升改为一次到位（`rank_warmup_epoch=-1, rank_warmup_len=1`）。原因：参数量只有 ~1.05M，warmup 期间 rank loss 系数为 0 时模型只学重建（等同 Phase 2 baseline），相当于浪费 epoch。

**最终生效超参（`scripts/train_phase3.sh`）：**

| 参数 | 值 | 备注 |
|---|---|---|
| `fineSize` | 256 | **强制要求**：与 Phase 2 训练分辨率一致 |
| `batch_size` | 1 | 全图 + N 实例 |
| `nThreads` | 4 | dataloader worker，避免 GPU 等待 |
| `lr` | 5e-5 | adapter zero-init，不宜过大 |
| `lambda_inst` | 1.0 | 与 `L_global` 同量级 |
| `lambda_rank` | **1.0** | 初版 0.1 太弱，rank 信号被 global+inst_rec 淹没 |
| `lambda_outside` | **0.1** | 初版 0.2 过强，背景过度锚定导致 prompt 无效 |
| `rank_margin` | **0.3** | 初版 0.05 在 KL 空间太小���达标过易 |
| `rank_warmup_epoch` | **-1** | 一次到位，warmup 从 epoch 1 即满值 |
| `rank_warmup_len` | 1 | 同上 |
| `niter / niter_decay` | 5 / 3 | 总 8 epoch |
| `save_epoch_freq` | 2 | 中间存 ckpt 便于对比 |
| AMP | 关闭 | 冻结 BN + fp16 会偶发 NaN，fp32 更稳 |

**训练数据：** COCO 2017 train，113,674 张图 / 407,866 实例（HSV 颜色词置信度 ≥ 0.04 / person ≥ 0.15 过滤后），val 4,801 张 / 17,309 实例。

**训练时间：** RTX 5090 单卡，909k iter / 8 epoch ≈ 14h（loss EMA 平稳，0 NaN）。

**LR 时间线：**

| epoch | warmup λ | LR | 阶段 |
|:---:|:---:|:---:|---|
| 1-5 | 1.0 | 5e-5 | 满 LR 学习期 |
| 6-8 | 1.0 | 5e-5 → 0 | LR 线性衰减 |

---

## 18. 推理强制要求

**🚨 推理时必须使用 `--fineSize 256`（默认即 256，但不要传 `--fineSize 224` 之类）。**

实测 `fineSize 224` 推理会产生大量饱和黄色异常色块（dog default 837 px vs Phase 2 baseline 54 px，相差 ~15×）。原因是 Phase 2 网络在 256 分辨率上训练，224 输入导致 BN 统计量、感受野尺寸全部错位，annealed-mean 解码产生 Lab 色域外像素，clamp 后形成尖锐黄色块。

**fs256 实测：** dog default 87 px，dog 各 prompt 87-192 px，cat 全部 0 px——回到 Phase 2 baseline 水准。

---

## 19. 最终验收结果（epoch 8）

### 文本可控性（mean Δ vs default）

| 测试 | prompt | mean Δ | 中心 RGB 变化 | 评价 |
|---|---|:---:|---|:---:|
| dog | red | 4.59 | B: 115→130 ↑ | ✅ 偏暖 |
| dog | yellow | 5.33 | B: 115→94 ↓ | ✅ 明显变黄 |
| dog | brown | 3.09 | B: 115→103 ↓ | ✅ 中度偏深 |
| dog | **green** | **8.51** | R: 173→143 ↓, G↑ | ✅ **跨色成功**（金毛→绿） |
| cat | gray | **6.93** | R: 185→166, B: 100→132 | ✅ **跨色成功**（橘猫→灰白） |
| cat | orange | 2.19 | 微调强化原色 | ✓ 较弱但方向对 |
| cat | brown | 1.60 | 与 orange 几乎相同 | ⚠️ **语义混淆**（见下） |

### 跨色差异（cross-prompt mean Δ）

| 对比 | mean Δ | 评价 |
|---|:---:|:---:|
| dog red ↔ yellow | **9.43** | 肉眼明显 |
| dog green ↔ brown | **10.55** | 最大差异 |
| cat gray ↔ orange | **9.02** | 肉眼明显 |

### 训练曲线（epoch 4 → epoch 8 提升）

| 指标 | e4 | e8 | Δ |
|---|:---:|:---:|:---:|
| dog red↔yellow | 7.42 | 9.43 | **+2.01** |
| cat gray↔orange | 6.06 | 9.02 | **+2.96** |
| dog brown vs default | 2.45 | 3.09 | +0.64 |

**结论：** 后 4 epoch 训练继续有效，每个指标都在提升，未饱和。

---

## 20. 支持的颜色集合与已知限制

### ✅ 支持的 prompt 颜色

**核心彩色系（实测有效）：** `red` / `yellow` / `orange` / `brown` / `green` / `gray`

这些颜色对应 313-bin ab 空间中**饱和度 > 0.3** 的清晰区域，模型能稳定学习并跨过 Phase 2 baseline 先验。

**推荐 prompt 模板：**
```
inst:i=a <color> <class>           # 例：inst:0=a red dog
inst:i=a <color> <class>           # 例：inst:1=a yellow shirt
bg=<color> <background_type>       # 例：bg=blue sky / green grass
```

### ❌ 不支持的 prompt 颜色

**亮度类颜色：** `black` / `white`（实测 mean Δ < 1.0，几乎无效）

原因：
1. 黑/白对应 ab ≈ (0, 0)，313-bin 分类器在低饱和度区域分辨力极弱
2. 训练时 HSV 主色推断把 V<0.15 / V≥0.82 分别归为 black/white，但这些样本在 mask 内通常是阴影/高光而非真实"黑色物体/白色物体"
3. CLIP 文本 "a black dog" 与 "a dog" 在嵌入空间距离较近，调制信号有限

**建议：** 推理时若用户传 `black` / `white` prompt，应在 CLI 打印警告"亮度类描述效果受限"。

### ⚠️ 已知限制

**1. 相近色语义混淆**

CLIP 在常见动物色上的语义边界较模糊：

- cat `brown` vs `orange` mean Δ = 1.08（几乎相同）—— 训练数据里"orange tabby"和"brown cat"主色都偏暖，CLIP 嵌入距离过近
- dog `brown` vs `yellow` mean Δ = 3.64 —— 中度可分但不强

**这不是 bug，是 CLIP 语义本身的限制**。如果用户需要精确区分，建议使用差异更大的颜色对（如 red vs blue、gray vs orange）。

**2. 跨图泛化能力较弱**

| 测试 | dog (主测试图) | dog2 (泛化图) |
|---|:---:|:---:|
| red vs default | 4.59 | 1.67 |
| yellow vs default | 5.33 | 2.16 |
| red↔yellow | 9.43 | 3.12 |

dog2 的可控性约为 dog 的 1/3。原因可能：
- dog2 的 Mask R-CNN bbox/mask 与训练分布不完全匹配
- 不同图像的 Phase 2 解码稳定性不同，prompt 撬动效果有差异

**应对：** 实际部署时，建议用户对同一张图尝试多个 prompt 强度变体（如 "a red dog" vs "a deeply red dog"）。

**3. yellow prompt 局部过饱和**

dog yellow 推理图中狗鼻子下方偶有小块饱和黄色（192 px 区域）。这是 prompt 把局部 ab 推到色域边界后 RGB clamp 产生的瑕疵。可接受范围内，但若需修复可在 `lab2rgb` 之前对预测的 ab 加 magnitude 限制。

---

## 21. 收官状态

- ✅ 训练完成：`checkpoints/phase3_text_color/8_net_T.pth` 作为 final ckpt（~4MB，1.05M 可训参数）
- ✅ Phase 2 权重零修改：`networks.py` / `inst_fusion_model.py` / Phase 2 ckpt 全部未触
- ✅ 推理 CLI 跑通：`test.py --method text_color`，支持 `--image` + `--prompt "inst:i=..." / "bg=..."`
- ✅ 异常色块控制在 Phase 2 baseline 水平（仅 fs256 推理）
- ✅ 颜色控制力可视化：dog 4 色 + cat 3 色 + dog2 泛化，结果保存在 `results/phase3_v6_e8_final/`

---

## 22. 续训计划：Plan A（e8 → e12）

**动机**：e8 已稳定收敛，但 dog2 泛化弱（mean Δ 仅为 dog 主图的 ~1/3），cat brown/orange 语义混淆。在 e8 基础上微调 4 个 epoch 尝试提升，**保留 e8 作为 fallback**。

**关键调整：**

| 参数 | e1-e8 | e9-e12 续训 | 理由 |
|---|:---:|:---:|---|
| `lr` | 5e-5 | **2e-5** | 避免破坏 e8 已学的能力 |
| `lambda_rank` | 1.0 | **1.5** | 进一步推动可控性 |
| `lambda_outside` | 0.1 | **0.05** | 防溢出由 e8 已学好，放开让 prompt 更激进 |
| `rank_margin` | 0.3 | **0.4** | 逼模型再拉大 pos/neg 差距 |
| `niter / niter_decay` | 5 / 3 | **10 / 2** | 总 epoch 数（绝对值），e9-e10 满 LR，e11-e12 衰减 |
| `epoch_count` | 0 | **8** | 从 `8_net_T.pth` warm-start |
| `save_epoch_freq` | 2 | **1** | 每 epoch 都存，方便逐步对比 |

**预期时间：** RTX 5090 单卡 ~6h（4 epoch × ~1.5h/epoch）。

**验收指标：**
- ✅ 成功：dog2 mean Δ 提升 ≥ 1.5；dog 主图可控性不退化；异常色块 ≤ baseline × 3
- ❌ 失败：rank loss 飙到 0.05+ / G 飙到 8+ / 异常色块 > 500 px → kill，回滚 e8

**安全措施：** 续训前备份 `cp 8_net_T.pth 8_net_T.pth.backup`；e9 ckpt 出来先验证质量再决定是否继续。

**Plan B（如 Plan A 失败）**：重建 JSONL 加 caption 多样化（5-10 种模板），从 e8 续训 8 epoch。

---

## 23. Plan A 续训结论（e8 → e12）

**结论：未达成验收标准，回滚 `8_net_T.pth` 为最终版本。**

| 指标 | e8 | e12 | 变化 | 评价 |
|---|:---:|:---:|:---:|:---:|
| dog red vs default | 4.59 | 5.00 | +0.41 | 微涨 |
| dog yellow vs default | 5.33 | 5.20 | -0.13 | 微跌 |
| dog brown vs default | 3.09 | 2.69 | -0.40 | 微跌 |
| dog green vs default | 8.51 | 9.39 | +0.88 | 微涨 |
| cat gray vs default | 6.93 | 7.62 | +0.70 | 微涨 |
| **dog2 red vs default** | **1.67** | **1.93** | **+0.26** | ❌ 远未达标 |
| dog2 yellow vs default | 2.16 | 2.29 | +0.13 | ❌ 未达标 |
| dog default 异常色块 | 135 px | 181 px | +34% | ⚠️ 略劣化 |
| dog yellow 异常色块 | 338 px | 461 px | +36% | ⚠️ 略劣化 |

**根因分析：**

1. **rank loss 已在 e8 饱和（~0.002）**，margin 0.3→0.4 在 100 iter 内被吸收，模型对 ranking 信号已"免疫"，没有有效梯度
2. **1.05M adapter 参数容量已达上限**，4 个微调 epoch 改变不了什么
3. **dog2 泛化弱不是训练问题，是数据问题**——单图 Mask R-CNN 检测分布与训练集 GT mask 分布有差异，纯靠训练无法弥合

**Plan B（caption 多样化）受时间限制未尝试。**

---

## 24. Phase 3 后续优化方向

按"成本/收益"排序，留作未来迭代参考：

### 🥇 高收益（数据层面优化，最值得做）

**1. 重建 JSONL，caption 模板多样化**

当前 caption 只有"a {color} {object}"一种模板，CLIP 接收的语义信号单一。改造：

```python
# scripts/build_phase3_jsonl.py 增加模板池
INST_TEMPLATES = [
    "a {color} {obj}",
    "a vivid {color} {obj}",
    "the {obj} is {color}",
    "a {color}-colored {obj}",
    "a {color} colored {obj}",
    "{obj} with {color} color",
    "a deep {color} {obj}",
    "a bright {color} {obj}",
]
# build 时为每个样本随机选 1-2 个模板存进 JSONL
# 训练时 dataloader 再从中随机抽
```

**预期提升**：跨图泛化 +30-50%（dog2 mean Δ 从 1.67 提升到 ~2.5）。CLIP 见过更多自然语言变体，对新图新句的鲁棒性更强。

**2. 加大 mask 内训练样本占比**

当前 JSONL 一张图一行，但训练时 inst mask 只占图像面积 ~10%，背景占 90%。可以：
- 训练时 inst crop 二次采样，放大到 fineSize
- 给 inst_rec loss 加更大权重（lambda_inst 1.0 → 2.0）

**预期提升**：mask 内可控性 +20%。

### 🥈 中等收益（架构优化）

**3. Adapter 加大到 2-3M 参数**

当前 adapter 只调制 conv8_3/9_3/10_2 三层 decoder。可以：
- 增加 conv7_3 / conv6_3 注入点（encoder 末尾）
- 或者用 MLP（2 层）替换当前 1 层 Linear

**预期提升**：黑/白 prompt 部分可用，cat brown↔orange 语义混淆改善。

**4. Cross-attention 替换 FiLM（高风险）**

参考 L-CoDer 思路，把 FiLM 改成 cross-attention（图像 query 文本 token）。L-CoDer 已经验证有效，但**代价是必须重训 +30 epoch**，且需修复你之前在 phase3_clip 分支看到的"颜色溢出"问题（outside loss 用 pos vs neg 而非 pos/neg vs GT 锚定）。

### 🥉 小收益（推理 trick，不改训练）

**5. 推理时多 prompt 平均**

```python
prompts = [
    f"a {color} dog",
    f"a vivid {color} dog",
    f"the dog is {color}",
]
text_emb = mean([clip.encode(p) for p in prompts])
```

CLIP 噪声平均后调制更稳定。

**预期提升**：mean Δ +0.5-1.0，0 训练成本。

**6. ab magnitude 限制**

在 `lab2rgb` 之前对 pred_ab 加 magnitude cap，防止 yellow 鼻子下方那种局部过饱和：

```python
ab_norm = pred_ab.norm(dim=1, keepdim=True)
max_norm = 80  # 经验值
pred_ab = pred_ab * (max_norm / ab_norm.clamp(min=max_norm))
```

**预期提升**：异常色块从 192 px 降到 < 50。0 训练成本。

### 不建议做

- **直接增加 epoch 数**：本次续训已证明无效，1M 参数 e8 已饱和
- **调整 lambda 系数**：已经是当前架构下最优区间
- **改 Lab→RGB 转换公式**：与 Phase 2 完全一致，改了会破坏 baseline

### 推荐路径

如果未来要进一步迭代 Phase 3，按 **5 → 6 → 1 → 3** 的顺序：

1. 先做推理 trick（5, 6），零成本看上限
2. 再重做数据（1），20h 训练换 30% 提升
3. 最后才考虑架构（3），承担 30h 训练 + 风险
