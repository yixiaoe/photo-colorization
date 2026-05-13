# Phase 2 实现说明

**更新日期：** 2026/05/13  
**对应分支：** `feat/phase2-inst-fusion`

---

## 数据流

```text
stage=full（全图阶段）
  full_rgb -> rgb2lab -> real_L / real_ab
  real_L -> InstFusionGenerator（全图分支）-> fake_ab(1/4 分辨率)
  上采样到原图大小 -> fake_ab_full
  L1(fake_ab_full, real_ab)

stage=instance（实例阶段）
  cropped_rgb -> rgb2lab -> crop_L
  crop_L -> inst_branch + inst_head -> crop_ab(1/4 分辨率)
  按 box_info_4x 回贴到全图特征尺度 -> inst_ab_map
  real_L + inst_ab_map -> 融合预测 -> fake_ab
  上采样 -> fake_ab_full
  L1(fake_ab_full, real_ab)

stage=fusion（融合阶段）
  先加载 full + instance 两阶段权重
  再执行与 instance 相同的融合前向流程
```

---

## 网络结构（InstFusionGenerator）

使用“组合复用”：

- 全图分支：`full_branch = CnnColorGenerator()`
- 实例分支：`inst_branch = CnnColorGenerator()`

两个分支都复用 `CnnColorGenerator` 的特征抽取能力，然后接 Phase 2 模块：

1. `full_head`：`Conv1x1(256->2) + Tanh`，输出全图 `ab`。
2. `inst_head`：`Conv1x1(256->2) + Tanh`，输出实例 `ab`。
3. `FusionGenerator`：3 层卷积，预测逐像素融合权重 `w`（`Sigmoid`）。
4. `WeightGenerator`：按权重融合：`out = inst_pred * w + full_pred * (1 - w)`。

---

## 三阶段训练打通

### 1) `--stage full`

- 目标：全图上色训练。
- 初始化来源：Phase 1 权重（默认路径：`checkpoints/cnn_color/latest_net_G.pth`）。

### 2) `--stage instance`

- 目标：训练实例分支在 crop 区域的颜色预测能力。
- 初始化来源：`full` 阶段权重（默认：`checkpoints/inst_fusion_full/latest_net_G.pth`）。

### 3) `--stage fusion`

- 目标：训练融合模块输出稳定的像素级融合权重。
- 初始化来源：
  - full 权重：`checkpoints/inst_fusion_full/latest_net_G.pth`
  - instance 权重：`checkpoints/inst_fusion_instance/latest_net_G.pth`

说明：

- `train.py` 负责把阶段相关配置（环境变量）传入 `opt`。
- `inst_fusion_model.py` 按 `opt.stage` 执行对应权重加载逻辑。

---

## 混合精度（RTX 4090 优化）

Phase 2 训练已接入 AMP：

- `torch.cuda.amp.autocast(enabled=self._use_amp)`
- `torch.cuda.amp.GradScaler(enabled=self._use_amp)`

训练流程：

```python
with autocast(enabled=self._use_amp):
    self.forward()
    self.loss_G = self.criterion(self.fake_ab_full, self.real_ab)

self.optimizer.zero_grad()
self.scaler.scale(self.loss_G).backward()
self.scaler.step(self.optimizer)
self.scaler.update()
```

效果：

- CUDA 环境（如 RTX 4090）自动启用。
- CPU 环境自动关闭 AMP，不影响兼容性。

---

## 烟雾测试（每阶段 10 iteration）

`train.py` 增加了 smoke test 机制：

- 通过环境变量 `SMOKE_TEST_ITERS` 控制。
- 当 `total_iters >= SMOKE_TEST_ITERS` 时：
  - 立即执行 `model.save_networks('latest')`
  - 打印 smoke test 完成提示
  - 提前退出该阶段

默认脚本使用：`SMOKE_TEST_ITERS=10`。

---

## 运行命令

一键三阶段（推荐，默认每阶段 10 iter）：

```bash
cd code
SMOKE_TEST_ITERS=10 bash scripts/train_phase2.sh ../data/imagenet_mini/train imagenet_mini cnn_color
```

单独跑某一阶段（示例：full）：

```bash
cd code
PHASE1_NAME=cnn_color FULL_STAGE_NAME=inst_fusion_full INSTANCE_STAGE_NAME=inst_fusion_instance SMOKE_TEST_ITERS=10 \
python train.py --method inst_fusion --stage full --dataset imagenet_mini --data_dir ../data/imagenet_mini/train --name inst_fusion_full --fineSize 256 --batch_size 16 --niter 1 --niter_decay 0 --lr 1e-4 --print_freq 1 --save_latest_freq 1000
```

---

## 关键文件

| 文件 | 作用 |
|---|---|
| `code/models/networks.py` | `InstFusionGenerator`、`FusionGenerator`、`WeightGenerator` 定义 |
| `code/models/inst_fusion_model.py` | 三阶段权重初始化、实例回贴融合、AMP 训练逻辑 |
| `code/train.py` | `inst_fusion` 阶段参数下发、smoke test 提前退出 |
| `code/scripts/train_phase2.sh` | 三阶段串行训练脚本（默认 10 iter 烟雾测试） |
