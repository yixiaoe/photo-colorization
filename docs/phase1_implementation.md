# Phase 1 实现说明

**更新日期：** 2026/05/11  
**对应分支：** `feat/phase1-cnn-color`

---

## 数据流

```
RGB 图像 (N,3,H,W) [0,1]
    │
    ▼ rgb2lab()
Lab 图像 (N,3,H,W)
    ├── L 通道 (N,1,H,W)  → 归一化到 [-1,1]  → 网络输入
    └── ab 通道 (N,2,H,W) → 下采样 ×1/4
                               │
                               ▼ encode_ab_to_zhang2016_bins()
                           GT 类别标签 (N,1,H/4,W/4)  int64  ∈ [0,312]
                               │
                               ▼ CrossEntropyLoss（带重平衡权重）
                             loss_G
```

---

## 网络结构（`CnnColorGenerator`）

| 块 | 层 | 输出通道 | 空间尺寸 | dilation |
|---|---|---|---|---|
| conv1 | Conv×2 + BN + ReLU | 64 | H/2 | 1 |
| conv2 | Conv×2 + BN + ReLU | 128 | H/4 | 1 |
| conv3 | Conv×3 + BN + ReLU | 256 | H/8 | 1 |
| conv4 | Conv×3 + BN + ReLU | 512 | H/8 | 1 |
| conv5 | Conv×3 + BN + ReLU | 512 | H/8 | 2 |
| conv6 | Conv×3 + BN + ReLU | 512 | H/8 | 2 |
| conv7 | Conv×3 + BN + ReLU | 512 | H/8 | 1 |
| conv8 | ConvTranspose + Conv×2 | 256 | H/4 | — |
| pred  | Conv 1×1 | **313** | H/4 | — |

- 输入：`(N, 1, H, W)` L 通道，归一化后范围约 `[-1, 1]`
- 输出：`(N, 313, H/4, W/4)` 原始 logits（未经 softmax）
- 参数量：约 32M

---

## 损失函数

**多项交叉熵 + 类别重平衡权重**

```
prior_mix = (1 - γ) * p_empirical + γ * (1/313)   # γ=0.5
weight_q  = 1 / prior_mix
weight_q  = weight_q / Σ(p_empirical * weight_q)   # 期望归一化
loss = CrossEntropy(logits, gt_class, weight=weight_q)
```

- `p_empirical`：`resources/zhang2016/prior_probs.npy`（313 个 bin 的经验出现概率）
- 重平衡目的：抑制网络偏向灰色/低饱和颜色的倾向

---

## 推理解码：Annealed-Mean

```
probs = softmax(logits / T)          # T=0.38，温度越低颜色越鲜艳
ab_pred = Σ probs_q * pts_in_hull_q  # 加权平均 313 个 bin 中心
```

- 输出 `(N, 2, H/4, W/4)` → 双线性上采样到原始分辨率 `(N, 2, H, W)`
- 温度 `T` 可通过 `--T` 参数调整

---

## util/util.py 接口说明

### Lab ↔ RGB 转换

```python
from util.util import rgb2lab, lab2rgb

# RGB → Lab（归一化）
# rgb:  (N, 3, H, W)  float32，范围 [0, 1]
# 返回: (N, 3, H, W)  L 归一化到 [-1,1]，ab 归一化到 [-1,1]
lab = rgb2lab(rgb, opt)

# Lab（归一化）→ RGB
# lab_rs: (N, 3, H, W)  归一化 Lab
# 返回:   (N, 3, H, W)  float32，范围 [0, 1]（需手动 clamp）
rgb = lab2rgb(lab_rs, opt).clamp(0, 1)
```

### 313-bin 资源加载

```python
from util.util import load_zhang2016_ab_bins, load_zhang2016_prior_probs

pts  = load_zhang2016_ab_bins()      # np.ndarray (313, 2)，ab bin 中心坐标（原始 ab 单位）
prior = load_zhang2016_prior_probs() # np.ndarray (313,)，经验先验概率
```

### 重平衡权重

```python
from util.util import build_zhang2016_rebalance_weights

# gamma: 均匀先验混合比例（论文默认 0.5）
# 返回:  Tensor (313,)  float32，直接传给 CrossEntropyLoss(weight=...)
w = build_zhang2016_rebalance_weights(gamma=0.5, device='cuda')
```

### 训练：ab → 类别标签

```python
from util.util import encode_ab_to_zhang2016_bins

# ab_norm:     (N, 2, H, W)  归一化 ab，范围 [-1, 1]
# pts_in_hull: Tensor (313, 2)，由 load_zhang2016_ab_bins() 转换而来
# ab_norm_val: 归一化系数，与 opt.ab_norm 保持一致（默认 110.）
# 返回:        (N, 1, H, W)  int64，类别索引 ∈ [0, 312]
gt_class = encode_ab_to_zhang2016_bins(ab_norm, pts_in_hull, ab_norm_val=110.)

# 配合 CrossEntropyLoss 使用（需去掉 channel 维度）
loss = nn.CrossEntropyLoss(weight=w)(logits, gt_class[:, 0])
# logits:       (N, 313, H, W)  float32（网络输出，务必 .float() 保证精度）
# gt_class[:,0]:(N, H, W)       int64
```

### 推理：logits → ab

```python
from util.util import decode_zhang2016_annealed_mean

# logits:      (N, 313, H, W)  float32，网络原始输出
# pts_in_hull: Tensor (313, 2)
# T:           温度，越小颜色越鲜艳（论文默认 0.38，可通过 --T 调整）
# ab_norm_val: 与训练一致（默认 110.）
# 返回:        (N, 2, H, W)  归一化 ab，范围约 [-1, 1]
ab_pred = decode_zhang2016_annealed_mean(logits, pts_in_hull, T=0.38, ab_norm_val=110.)

# 上采样回原始分辨率后拼接 L 通道转 RGB
ab_full = F.interpolate(ab_pred, size=(H, W), mode='bilinear', align_corners=False)
fake_rgb = lab2rgb(torch.cat([L, ab_full], dim=1), opt).clamp(0, 1)
```

### 图像保存

```python
from util.util import tensor2im, save_image

# tensor2im: (N, C, H, W) Tensor → (H, W, 3) uint8 numpy（取第 0 张）
arr = tensor2im(rgb_tensor)
save_image(arr, 'output.png')
```

---

## 运行命令

```bash
# 训练（CIFAR-10）
bash scripts/train_phase1.sh /path/to/datasets/cifar-10 cifar10

# 训练（ImageNet-Mini）
bash scripts/train_phase1.sh /path/to/datasets/imagenet_mini imagenet_mini

# 推理
python test.py --method cnn_color --test_img_dir data/test --which_epoch latest
```

---

## 资源文件

| 文件 | 来源 | 用途 |
|------|------|------|
| `code/resources/zhang2016/pts_in_hull.npy` | 官方参考实现 | 313 个 ab bin 中心坐标 |
| `code/resources/zhang2016/prior_probs.npy` | 官方参考实现 | 经验颜色先验概率，用于重平衡权重 |
