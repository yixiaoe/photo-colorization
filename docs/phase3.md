# Phase 3: CLIP 文本引导上色

## 结论

Phase 3 不是 Exemplar/reference image 上色，而是 CLIP 文本引导上色。它依赖一个已经训练好的 Phase 2 `InstanceGenerator` 全图骨干：先用 ImageNet-Mini 训练 `stage=full`，固定使用 `checkpoints/inst_full/80_net_G.pth`，再冻结该骨干，在解码器中加入文本 Cross-Attention。

训练前需要注意当前代码状态：

- Phase 2 已改为不使用 PSNR 选择 best，而是用 `val_loss_G` 与 `overfit_gap_G` 判断是否过拟合。
- Phase 2 的 PSNR/SSIM 只作为固定抽样验证图的描述指标。
- 当前本地 `code/checkpoints/inst_full` 已精简为 `30_net_G.pth`、`80_net_G.pth`、`best_net_G.pth`。
- 当前本地 `code/results/jiandu/inst_full` 已精简为 `loss_history.csv`、`nan_events.csv`、`epoch_030`、`epoch_060`。
- Phase 3 当前采用 `color-word + object + instance mask` 的区域监督。COCO caption 只作为句子底稿，object/mask 来自 COCO instance annotation。
- Phase 3 冻结骨干 checkpoint 必须使用 `80_net_G.pth`，不使用 `best_net_G.pth`。
- Phase 3 预处理默认排除 `person`，训练应使用 `phase3_color_object_no_person_*.jsonl`，不要继续使用上一轮可能包含 `person` 的旧 JSONL。

## Phase 2 全图骨干检查

`InstanceGenerator` 可以作为 Phase 3 的全图骨干使用。它的接口满足后续文本注入需要：

- 输入：`L` 通道灰度图，形状 `(N, 1, H, W)`。
- 输出一：`out_class`，313-bin ab 分类 logits，形状 `(N, 313, H/4, W/4)`。
- 输出二：`out_reg`，ab 回归结果，形状 `(N, 2, H, W)`。
- 输出三：`feature_map`，包含 `conv8_3`、`conv9_3` 等中间特征，便于后续注入文本条件。

三个输出的作用不同：

1. `out_class`
   - 表示每个低分辨率位置属于 313 个 ab 颜色 bin 的概率分布。
   - 训练时参与 rebalanced cross-entropy，缓解灰色/低饱和颜色占多数的问题。
   - 推理时通过 annealed-mean 解码为 ab，再上采样到原图大小。
   - 优点是颜色更稳定，符合 Zhang 2016 的分类式上色思路。

2. `out_reg`
   - 表示直接回归出的连续 ab 通道，分辨率与输入一致。
   - 训练时参与 Huber loss，约束空间连续性和局部细节。
   - 推理时可直接与 `L` 通道拼接成 Lab，再转 RGB。
   - 优点是分辨率高、局部边界更直接；缺点是回归容易平均化，颜色可能更保守。

3. `feature_map`
   - 不直接作为最终图片输出。
   - 保存 encoder/decoder 中间特征，用于 Phase 2 fusion、调试可视化，以及 Phase 3 文本注入点选择。
   - Phase 3 实际使用的是内部 `conv8_3` 和 `conv9_3` 特征；它们分别对应 `H/4`、`H/2` 尺度。

最终推理需要同时保存两种上色结果：

```text
out_class -> annealed-mean 解码 -> fake_rgb
out_reg   -> 直接拼接 Lab      -> fake_rgb_reg
```

这样可以比较分类分支和回归分支的差异：`fake_rgb` 观察颜色分布是否合理，`fake_rgb_reg` 观察连续 ab 回归是否稳定。

已做 smoke check：

```bash
cd code
python -c "import torch; from models.networks import InstanceGenerator; net=InstanceGenerator().eval(); x=torch.randn(1,1,256,256); y_cls,y_reg,fm=net(x); print(tuple(y_cls.shape), tuple(y_reg.shape), len(fm))"
```

期望输出：

```text
(1, 313, 64, 64) (1, 2, 256, 256) 13
```

反向传播也可以正常走通：

```bash
cd code
python -c "import torch; from models.networks import InstanceGenerator; net=InstanceGenerator(); x=torch.randn(2,1,128,128); y_cls,y_reg,fm=net(x); loss=y_cls.mean()+y_reg.mean(); loss.backward(); print(tuple(y_cls.shape), tuple(y_reg.shape), bool(next(net.parameters()).grad is not None))"
```

期望输出：

```text
(2, 313, 32, 32) (2, 2, 128, 128) True
```

## Phase 2: ImageNet-Mini 全图骨干训练

目标是只训练 `InstanceGenerator` 的 full-image backbone，不训练 instance/fusion 分支。这个阶段不需要 COCO bbox，也不需要 captions。

数据流：

```text
ImageNet-Mini RGB 图像
  -> ColorizationDataset 读取与增强
  -> RGB 转 Lab
  -> L 通道输入 InstanceGenerator
  -> 输出 313-bin 分类 + ab 回归
  -> CE(rebalanced) + 3x Huber
  -> 保存 full backbone checkpoint
```

推荐目录：

```text
/root/autodl-tmp/imagenet_mini/train
/root/autodl-tmp/imagenet_mini/val
```

训练命令：

```bash
cd code
bash scripts/train_phase2_full.sh \
  /root/autodl-tmp/imagenet_mini/train \
  /root/autodl-tmp/imagenet_mini/val
```

等价展开命令：

```bash
cd code
/root/miniconda3/bin/python train.py \
  --method inst_fusion \
  --stage full \
  --dataset imagenet_mini \
  --data_dir /root/autodl-tmp/imagenet_mini/train \
  --val_data_dir /root/autodl-tmp/imagenet_mini/val \
  --val_freq 5 \
  --fineSize 256 \
  --batch_size 32 \
  --nThreads 8 \
  --lr 1e-4 \
  --huber_weight 3.0 \
  --niter 30 \
  --niter_decay 30 \
  --max_epochs 100 \
  --grad_clip_norm 5.0 \
  --nan_lr_factor 0.1 \
  --nan_max_retries 3 \
  --early_stop_patience 6 \
  --save_epoch_freq 5 \
  --monitor_dir results/jiandu \
  --monitor_num 50 \
  --monitor_freq 5 \
  --name inst_full \
  --gpu_ids 0
```

新训练会产生：

```text
code/checkpoints/inst_full/best_net_G.pth
code/checkpoints/inst_full/latest_net_G.pth
code/checkpoints/inst_full/recovery_clean_net_G.pth
code/checkpoints/inst_full/<epoch>_net_G.pth
```

当前本地已精简后，只保留：

```text
code/checkpoints/inst_full/30_net_G.pth
code/checkpoints/inst_full/80_net_G.pth
code/checkpoints/inst_full/best_net_G.pth
```

推荐 Phase 3 使用 `80_net_G.pth` 作为 `--full_ckpt`。

监督产物：

```text
code/results/jiandu/inst_full/options.json
code/results/jiandu/inst_full/loss_history.csv
code/results/jiandu/inst_full/metrics.csv
code/results/jiandu/inst_full/epoch_005/*.png
```

其中 `epoch_xxx` 目录保存固定随机抽取样本的：

```text
real_gray
fake_rgb      # out_class 解码结果，主结果
fake_rgb_reg  # out_reg 回归结果，对照结果
real_rgb
```

`loss_history.csv` 会记录 `loss_G`、`loss_ce`、`loss_huber`、`loss_huber_weighted` 和当前学习率，便于判断 Huber 权重、重加权参数和学习率是否合理。`metrics.csv` 记录 `train_loss_G`、`val_loss_G`、`overfit_gap_G`，用于判断是否过拟合；PSNR/SSIM 只写入固定抽样监控，不用于选择 best。

当前本地已精简后，只保留：

```text
code/results/jiandu/inst_full/loss_history.csv
code/results/jiandu/inst_full/nan_events.csv
code/results/jiandu/inst_full/epoch_030
code/results/jiandu/inst_full/epoch_060
```

## Phase 3 架构

Phase 3 的核心是把文本条件注入 Phase 2 full backbone 的 decoder：

```text
灰度图 L
  -> 冻结的 InstanceGenerator encoder/decoder
  -> conv8_3 后接 TextCrossAttentionBlock
  -> conv9_3 后接 TextCrossAttentionBlock
  -> 输出 ab 颜色

文本 prompt/caption
  -> 冻结的 CLIP ViT-B/32 text encoder
  -> token-level features: (N, 77, 512)
  -> Linear projection
  -> 作为 Cross-Attention 的 Key/Value
```

Attention 关系：

- 图像特征作为 Query。
- CLIP token 特征作为 Key/Value。
- `conv8_3` 注入语义较强的中层颜色控制。
- `conv9_3` 注入更靠近输出的局部颜色修正。
- `InstanceGenerator` 与 CLIP 都冻结，只训练新增的两个 `TextCrossAttentionBlock`。

具体接入方式如下：

1. 文本编码

```text
prompt/caption
  -> open_clip tokenizer
  -> token ids: (N, 77)
  -> CLIP token_embedding + positional_embedding
  -> CLIP text transformer
  -> ln_final
  -> text_tokens: (N, 77, 512)
  -> padding_mask: (N, 77)
```

这里保留完整 token 序列，而不是只取 EOS pooled 向量。原因是颜色控制常常和局部词相关，例如 `red car`、`blue sky`，token-level features 更适合 Cross-Attention。

2. `conv8_3` 注入

```text
conv8_3: (N, 256, H/4, W/4)
  -> flatten spatial
  -> query: (N, H/4*W/4, 256)
text_tokens: (N, 77, 512)
  -> Linear(512, 256)
  -> key/value: (N, 77, 256)
MultiheadAttention(query, key, value)
  -> residual add
  -> reshape back: (N, 256, H/4, W/4)
```

随后 `model_class(conv8_3)` 产生 `out_class`，所以文本会直接影响 313-bin 颜色分类分支。

3. `conv9_3` 注入

```text
conv9_3: (N, 128, H/2, W/2)
  -> query: (N, H/2*W/2, 128)
text_tokens: (N, 77, 512)
  -> Linear(512, 128)
  -> key/value: (N, 77, 128)
MultiheadAttention(query, key, value)
  -> residual add
  -> reshape back: (N, 128, H/2, W/2)
```

随后 `model_out(conv10_2)` 产生 `out_reg`，所以文本也会影响连续 ab 回归分支。

4. 稳定性设计

`TextCrossAttentionBlock` 的 `out_proj` 使用 zero-init。训练刚开始时，attention 分支输出接近 0，残差连接使模块近似恒等映射：

```text
result = image_feature + zero_initialized_attention_delta
```

这样 Phase 3 初始状态接近已训练好的 Phase 2 backbone，不会一开始破坏已有上色能力。

## Phase 3 数据预处理

Phase 3 采用 `color-word + object + instance mask` 的区域监督。COCO caption 只提供句子底稿；目标 object、bbox、segmentation mask 来自 `instances_*.json`。

输入数据：

```text
/root/autodl-tmp/coco2017/train2017
/root/autodl-tmp/coco2017/val2017
/root/autodl-tmp/coco2017/annotations/instances_train2017.json
/root/autodl-tmp/coco2017/annotations/captions_train2017.json
/root/autodl-tmp/coco2017/annotations/instances_val2017.json
/root/autodl-tmp/coco2017/annotations/captions_val2017.json
```

预处理脚本：

```bash
cd code
/root/miniconda3/bin/python scripts/build_phase3_color_object_jsonl.py \
  --img_dir /root/autodl-tmp/coco2017/train2017 \
  --instances_file /root/autodl-tmp/coco2017/annotations/instances_train2017.json \
  --captions_file /root/autodl-tmp/coco2017/annotations/captions_train2017.json \
  --out_file data/phase3_color_object_no_person_train.jsonl \
  --exclude_categories person
```

筛选逻辑：

```text
iscrowd == 0
segmentation 是 polygon
object != person
instance area / image area >= 0.01
caption 中能匹配 object 或同义词
mask 内 HSV 主色置信度 >= 0.05
```

说明：`person` 被排除不是因为 COCO 的 person mask 不准确，而是因为人体、衣服、肤色、姿态和遮挡经常混在同一个 instance mask 内。用单一 HSV 主色改写成 `color-word + person`，容易把衣服颜色、肤色或阴影错误绑定到 `person` 语义上，反而污染文本控制训练。

同义词表示例：

```text
car: car, vehicle, auto
airplane: airplane, plane, aircraft
couch: couch, sofa
dining table: dining table, table
surfboard: surfboard, surf board
```

每条 JSONL 记录是一条 instance 样本：

```json
{
  "image_path": ".../000000123456.jpg",
  "image_id": 123456,
  "ann_id": 789,
  "object": "car",
  "color": "red",
  "neg_color": "cyan",
  "caption_original": "A car parked near the curb.",
  "caption_pos": "A red car parked near the curb.",
  "caption_neg": "A cyan car parked near the curb.",
  "bbox": [x, y, w, h],
  "segmentation": [[x1, y1, x2, y2, "..."]],
  "width": 640,
  "height": 480
}
```

## Mask 处理

Phase 3 主监督使用 segmentation，不使用 bbox 作为 loss 区域。bbox 只适合快速定位、裁剪或过滤小物体。

mask 缩放必须使用最近邻插值：

```text
原图 mask: H x W, 取值 0/1
mask_full: 256 x 256, nearest neighbor
mask_4x: 64 x 64, nearest neighbor
```

原因是 mask 是离散标签，不是图像颜色。若使用 bilinear，会产生 0.2、0.5、0.7 这类软边界，导致 instance loss 的区域含义不清楚。

`CocoColorObjectDataset` 输出：

```text
rgb_img:     (3, 256, 256)
caption_pos: "A red car ..."
caption_neg: "A cyan car ..."
mask_full:   (1, 256, 256)
mask_4x:     (1, 64, 64)
```

## HSV 颜色词生成

颜色来自 instance mask 内像素，不从整张图统计。规则先判断中性色，再判断 hue：

```text
V < 0.15              -> black
S < 0.20 and V >= .82 -> white
S < 0.20              -> gray
15 <= H < 45 and V < .55 -> brown
H < 15 or H >= 345    -> red
15 <= H < 35          -> orange
35 <= H < 70          -> yellow
70 <= H < 170         -> green
170 <= H < 200        -> cyan
200 <= H < 260        -> blue
260 <= H < 310        -> purple
310 <= H < 345        -> pink
```

如果原 caption 没有颜色词，就把颜色插入 object 前：

```text
"A car parked near the curb."
-> "A red car parked near the curb."
```

如果原 caption 已经有 `color + object`，则替换为 mask 统计出的颜色，减少 caption 与图像颜色冲突。

负样本颜色优先选择互补色或高对比色：

```text
red    -> cyan
yellow -> blue
black  -> white
white  -> black
gray   -> red
```

## Phase 3 训练

推荐直接运行：

```bash
cd code
bash scripts/train_phase3.sh /root/autodl-tmp/coco2017
```

展开后的核心命令：

```bash
cd code
/root/miniconda3/bin/python train_phase3.py \
  --color_object_file data/phase3_color_object_no_person_train.jsonl \
  --val_color_object_file data/phase3_color_object_no_person_val.jsonl \
  --full_ckpt checkpoints/inst_full/80_net_G.pth \
  --fineSize 256 \
  --batch_size 16 \
  --nThreads 4 \
  --lr 1e-4 \
  --huber_weight 3.0 \
  --lambda_inst 1.0 \
  --lambda_rank 0.1 \
  --lambda_outside 0.2 \
  --rank_margin 0.05 \
  --niter 30 \
  --niter_decay 30 \
  --name text_color \
  --gpu_ids 0
```

每个 batch 做双 prompt 前向：

```text
positive flow:
L + caption_pos -> out_class_pos, out_reg_pos

negative flow:
L + caption_neg -> out_class_neg, out_reg_neg
```

只更新 `TextCrossAttentionBlock` 参数；Phase 2 `InstanceGenerator` backbone 和 CLIP text encoder 都冻结。

## Phase 3 Loss

全图正样本重建 loss：

```text
L_global = CE_rebalanced(out_class_pos, gt_ab)
         + 3.0 * Huber(out_reg_pos, gt_ab)
```

instance mask 内重建 loss：

```text
L_inst_ce =
  sum(mask_4x * CE_rebalanced(out_class_pos, gt_ab))
  / sum(mask_4x)

L_inst_huber =
  sum(mask_full * Huber(out_reg_pos, gt_ab))
  / sum(mask_full)

L_inst_rec = L_inst_ce + 3.0 * L_inst_huber
```

正负 prompt ranking loss 使用分类分支解码后的 ab：

```text
ab_cls_pos = decode_313(out_class_pos, T=0.38)
ab_cls_neg = decode_313(out_class_neg, T=0.38)

D_pos = Huber(ab_cls_pos[mask_4x], gt_ab_4x[mask_4x])
D_neg = Huber(ab_cls_neg[mask_4x], gt_ab_4x[mask_4x])

L_rank = max(0, margin + D_pos - D_neg)
margin = 0.05
```

含义：正确颜色 prompt 必须比互补色错误 prompt 更接近 GT。

mask 外一致性 loss：

```text
L_outside_reg = Huber(out_reg_pos[1-mask_full], out_reg_neg[1-mask_full])
L_outside_cls = Huber(ab_cls_pos[1-mask_4x], ab_cls_neg[1-mask_4x])
L_outside = 0.5 * (L_outside_reg + L_outside_cls)
```

含义：`red car` 和 `cyan car` 的差异应该主要发生在 car mask 内，mask 外不要乱染。

最终 loss：

```text
L_total =
  L_global
  + lambda_inst * L_inst_rec
  + lambda_rank * L_rank
  + lambda_outside * L_outside

lambda_inst = 1.0
lambda_rank = 0.1
lambda_outside = 0.2
```

## 数据集选择

推荐分工：

```text
Phase 2 full backbone: ImageNet-Mini
Phase 3 text guidance: COCO Captions + COCO Instances
```

原因：

- ImageNet-Mini 适合训练通用全图上色骨干，因为它不需要文本标注，只需要 RGB 图像。
- COCO Captions 提供自然语言句子底稿，COCO Instances 提供 object、bbox、segmentation mask。
- Phase 3 不直接随机使用 5 条 captions，而是筛选出包含目标 object 的一条 caption，再改写为 `color-word + object`。
- 只用 ImageNet-Mini 训练 Phase 3 不够，因为没有自然语言监督；除非额外生成 caption。
- 只用 COCO 训练 full backbone 也可以，但 COCO 目标偏物体，图像分布不如 ImageNet-Mini 适合做通用预训练。

关于背景描述：COCO captions 不是完全没有 `sky`、`grass`、`street` 这类背景词，但它的标注更偏显著物体，背景颜色和场景属性覆盖不稳定。它的缺陷主要是：

- 同一图像的 5 条 caption 不一定描述颜色。
- 很多 caption 描述物体类别和动作，不描述背景颜色。
- `sky`、`water`、`sunset`、`night` 等场景词覆盖不均匀。
- Caption 可能说 “a person on a beach”，但不说明天空、水面、衣服的颜色。
- 一张图被随机选中某条 caption 时，文本和颜色监督可能很弱。

如果目标是稳定控制 “blue sky”、“green grass”、“sunset background”，更好的方案是：

- 保留 ImageNet-Mini 训练 Phase 2 full backbone。
- Phase 3 用 COCO Captions + Instances 起步。
- 增加固定 prompt 验证集，专门覆盖 sky、grass、water、night、sunset 等背景词。
- 后续可用 BLIP/人工模板为 ImageNet-Mini 或 COCO 图像补充更丰富 caption。
- 对 COCO captions 做筛选或重采样，优先保留含颜色词、场景词、材质词的 caption。
- 为固定验证图人工写多组 prompt，专门测试文本是否真的改变颜色。

## 超参数可视化与控制

需要把“哪些改动合理”变成可比较证据，而不是只看最后几张图。建议分五类记录：

1. 标量曲线

```text
loss/G
loss/ce
loss/huber
val_loss/G
overfit_gap/G
monitor/psnr
monitor/ssim
lr
grad_norm
```

Phase 3 应沿用 Phase 2 的评价原则：best 不由 PSNR/SSIM 决定，而由 `val_loss_G`、`overfit_gap_G` 和固定样本视觉效果共同判断。PSNR/SSIM 只作为固定样本的描述指标。若 `grad_norm` 经常尖峰，优先降低 `lr` 或增大梯度裁剪。

2. 固定样本图像网格

目标设计是每次验证使用同一批灰度图和同一组 prompt，保存：

```text
gray | fake_rgb(class) | fake_rgb_reg(reg) | gt
```

这比随机样本更能判断超参数改动是否真实有效。

当前代码已经支持固定的 `val_color_object_file`，但还没有实现“同一张灰度图配多组 prompt”的矩阵式 monitor。训练前建议补一个 Phase 3 monitor：固定 8-16 张灰度图，固定一组 prompt，每 `val_freq` 保存一张网格图。

3. prompt 控制矩阵

prompt 控制矩阵是一种验证协议，不是额外训练数据。它把同一张灰度图重复输入多组 prompt：

```text
"a red car"
"a blue car"
"a yellow car"
"a car under blue sky"
"a car at night"
```

保存时可以做成矩阵：

```text
rows: fixed gray images
cols: prompts
cell: model output
```

例如专门构造背景 prompt：

```text
"a scene with blue sky"
"a scene with dark night sky"
"a scene at sunset"
"a scene near clear blue water"
"a scene with green grass"
```

观察输出是否随 prompt 变化。如果不同 prompt 输出几乎一样，说明文本条件没有被充分利用。当前代码还没有自动生成这个矩阵，只能通过多次运行 `test_phase3.py` 手动比较，或在训练脚本中新增固定 prompt monitor。

4. 分布统计

记录预测 ab 分布、颜色 bin 使用频率、图像 colorfulness。合理训练应避免两种极端：

- 颜色分布塌缩到灰色附近。
- 颜色过饱和且与图像内容无关。

5. 实验配置记录

每次训练保存完整超参数：

```text
lr
batch_size
num_heads
rebalance_gamma
niter / niter_decay
full_ckpt
clip_arch
attention injection layers
random seed
```

推荐一次只改一个关键变量。优先做这些 ablation：

```text
lr:        1e-4 vs 5e-5 vs 2e-4
num_heads: 2 vs 4 vs 8
注入层:    conv8 only vs conv9 only vs conv8+conv9
gamma:     0.3 vs 0.5 vs 0.7
huber:     1.0 vs 3.0 vs 5.0
```

合理改动的标准：loss 稳定、验证指标不崩、固定样本颜色更自然、prompt 控制更明显。只让训练 loss 下降但图像变灰、变脏、prompt 不生效的改动不合理。

## Phase 3 推理

```bash
cd code
python test_phase3.py \
  --full_ckpt checkpoints/inst_full/80_net_G.pth \
  --test_img_dir data/test \
  --results_img_dir results/phase3 \
  --prompt "a red car under blue sky" \
  --name text_color \
  --which_epoch latest \
  --gpu_ids 0
```

推理时，改变 `--prompt` 应该能改变输出颜色倾向。例如：

```bash
--prompt "a yellow bus on a sunny street"
--prompt "a blue car at night"
--prompt "green grass and a red flower"
```

## 验收标准

- Phase 2 full backbone 能在 ImageNet-Mini 上训练，并使用 `checkpoints/inst_full/80_net_G.pth` 作为 Phase 3 冻结骨干。
- Phase 3 能加载该 full backbone，并冻结 backbone 参数。
- CLIP text encoder 冻结，只输出 token-level features。
- 训练时只有 `TextCrossAttentionBlock` 参数参与优化。
- 同一张灰度图使用不同 prompt 推理时，输出颜色方向有可见变化。

## 不采用的旧方案

旧的 Phase 3 Exemplar 方案是 `--exemplar --ref_img`：输入参考图并做参考色彩迁移。当前 Phase 3 不走这条路线，也不需要 `ExemplarAttention`、`StyleHarmonizer` 或参考图输入。当前目标是文本 prompt 控制上色。
