# 网络结构调整方案

**日期：** 2026/05/13  
**触发原因：** 阅读参考实现（Su et al. 2020）后发现当前 CnnColorGenerator 与原版存在关键差异

---

## 差异分析

| 结构特征 | 当前实现 | 参考实现 SIGGRAPHGenerator |
|---------|---------|--------------------------|
| 跳跃连接 | 无 | 3 层（conv1→conv10, conv2→conv9, conv3→conv8） |
| 上采样次数 | 1 次 (H/8→H/4) | 3 次 (H/8→H/4→H/2→H) |
| 输出分辨率 | H/4 + 双线性插值 | 全分辨率 |
| 损失 | 仅分类 (313-bin CE) | 分类 + 回归 (Huber) |
| 输出头 | 单头 (313 bins) | 双头 (529 bins class + 2ch reg) |

当前实现是对 Zhang 2016 论文的**简化复现**，而参考代码的 SIGGRAPHGenerator 实际上已融合了 Zhang 2016 backbone + 跳跃连接 + 渐进上采样，是更成熟的基线。

---

## 调整原则

1. **不以 U-Net 为目标**——跳跃连接对上色的边际收益经原论文验证，3 层 add 式 skip 已足够
2. **对齐参考实现**——便于 Phase 2 复用 WeightGenerator 融合逻辑
3. **不改损失函数**——保留 313-bin CE + 退火均值解码体系（已验证可行）
4. **保持向后兼容**——`get_features()` 接口需适配 Phase 3

---

## 具体改动

### 1. CnnColorGenerator 结构升级

```
当前:                       调整为:
conv1 (64, S1→S2)          conv1 (64, S1→S2)
conv2 (128, S1→S2)         conv2 (128, S1→S2)
conv3 (256, S1→S2)         conv3 (256, S1→S2)
conv4~7 (512, dilation)    conv4~7 (512, dilation)      ← 保持不动
conv8 (反卷积 H/8→H/4)     conv8up (反卷积 H/8→H/4)
                              + model3short8(conv3)  → add
                           conv8_3 (256 conv×2)
                           conv9up (反卷积 H/4→H/2)
                              + model2short9(conv2)  → add
                           conv9_3 (128 conv×2)
                           conv10up (反卷积 H/2→H)
                              + model1short10(conv1) → add
                           conv10_3 (128 conv×1)
pred (1×1, 313)            pred (1×1, 313)               ← 保持 313 bins
```

### 2. 输出分辨率变化

- 旧：输出 H/4 → 双线性插值到 H
- 新：**直接输出全分辨率**，去掉后处理插值

### 3. 参数量变化

- 旧：~28M params
- 新：~32M params（增加约 15%，主要来自 decoder 的 conv9/conv10）

### 4. Phase 2 兼容性

`get_features()` 返回的 feature map：
- 旧：conv8 输出的 256ch × H/4
- 新：可配置返回 conv8 或 conv10 的 128ch × H，通过 `--feat_layer` 控制
- 不影响 Phase 2/3 的融合接口设计

### 5. get_features() 接口

新增参数 `feat_layer` 选择特征提取位置：
- `'conv8'` (H/4, 256ch) — 兼容 Phase 3 原设计
- `'conv10'`  (H, 128ch) — 全分辨率特征，适合更精细的 Attention

---

## 涉及文件

| 文件 | 改动内容 |
|------|---------|
| `code/models/networks.py` | `CnnColorGenerator` 增加 3 个跳连接 + 2 个上采样层；修改 `forward`/`get_features` |
| `code/models/cnn_color_model.py` | 若输出分辨率变全分辨率，修改 `set_input` 中 ab 下采样逻辑 |
| `docs/architecture.md` | 更新 Phase 1 数据流图和 Phase 2 网络说明 |
| `docs/phase1_implementation.md` | 更新网络结构表格和数据流描述 |

---

## 不动点

- 313-bin 分类头 + 加权交叉熵 → **不变**
- 退火均值解码推理流程 → **不变**
- Phase 2 的三阶段训练编排 → **不变**
- Phase 3 的 ExemplarAttention 接口 → **不变**
- 数据集管线（`colorization_dataset.py`） → **不变**

---

## 后续影响

### Phase 2（InstFusionModel）
- `InstFusionGenerator` 继承升级后的 `CnnColorGenerator`
- `InstanceGenerator` 返回各层 feature map（对齐参考实现）
- `FusionGenerator` 每层插入 `WeightGenerator`（复用参考实现设计）

### Phase 3（ExemplarAttention）
- `get_features()` 增加 `feat_layer` 参数，选择从 conv8 (H/4) 或 conv10 (H) 提取
- 不影响 Cross-Attention 逻辑
