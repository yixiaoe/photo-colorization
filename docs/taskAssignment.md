# 子任务分配表

**项目名称：** 黑白照片上色（三阶段）  
**更新日期：** 2026/05/10（Task-02/03/04/05/06/07 已完成）

---

## 任务总览

| 任务编号 | 任务名称 | Phase | 负责人 | 状态 | 依赖任务 |
|---------|---------|-------|--------|------|---------|
| Task-01 | 环境验证（autoDL 服务器） | 准备 | | ☑ 已完成 | 无 |
| Task-02 | code/ 目录骨架搭建 | 准备 | Yi_CC | ☑ 已完成 | Task-01 |
| Task-03 | options 参数体系 | 准备 | Yi_CC | ☑ 已完成 | Task-02 |
| Task-04 | 数据集管线 | 准备 | Yi_CC | ☑ 已完成 | Task-03 |
| Task-05 | [P1] CnnColorGenerator 网络实现 | Phase 1 | Yi_CC | ☑ 已完成 | Task-03 |
| Task-06 | [P1] 训练打通（全图上色） | Phase 1 | Yi_CC | ☑ 已完成 | Task-04, Task-05 |
| Task-07 | [P1] 推理闭环与评测 | Phase 1 | Yi_CC | ☑ 已完成 | Task-06 |
| Task-08 | [P2] 双分支网络迁移 | Phase 2 | | ☐ 未开始 | Task-07 |
| Task-09 | [P2] 三阶段训练打通 | Phase 2 | | ☐ 未开始 | Task-04, Task-08 |
| Task-10 | [P2] 推理闭环与评测 | Phase 2 | | ☐ 未开始 | Task-09 |
| Task-11 | [P3] ExemplarAttention 模块 | Phase 3 | | ☐ 未开始 | Task-07 |
| Task-12 | [P3] Exemplar 推理验证 | Phase 3 | | ☐ 未开始 | Task-11 |

---

## 使用说明

1. **认领任务：** 在"负责人"列填写姓名
2. **更新状态：** 将 ☐ 改为 ☑
3. **注意依赖：** 开始任务前确保依赖任务已完成

---

## 任务详细说明

### Task-01：环境验证（已完成）
- PyTorch 2.0.0 + Python 3.8 + CUDA 11.8 确认可用
- `torchvision.models.detection.maskrcnn_resnet50_fpn` 可用
- 无需安装 Detectron2

---

### Task-02：code/ 目录骨架搭建 ☑
**负责人：** Yi_CC  
**交付文件：**
- `code/train.py`、`code/test.py`（主入口）
- `code/models/`：`__init__.py`、`base_model.py`、`cnn_color_model.py`（占位）、`inst_fusion_model.py`（占位）、`networks.py`（占位）
- `code/util/`：`util.py`、`visualizer.py`
- `code/scripts/`：`train_phase1.sh`、`train_phase2.sh`、`test.sh`、`setup.sh`

---

### Task-03：options 参数体系 ☑
**负责人：** Yi_CC  
**交付文件：** `code/options/base_options.py`、`code/options/train_options.py`（含 `TestOptions`）  
**已实现参数：**
- `--method`：`cnn_color` | `inst_fusion`
- `--stage`：`full` | `instance` | `fusion`（仅 inst_fusion 使用）
- `--dataset`：`imagenet_mini` | `cifar10`
- `--exemplar`：flag，是否启用参考图 Bonus
- `--ref_img`：参考图路径（exemplar 模式）
- `--name`、`--data_dir`、`--fineSize`、`--batch_size` 等通用参数

---

### Task-04：数据集管线 ☑
**负责人：** Yi_CC  
**交付文件：** `code/datasets/colorization_dataset.py`  
**已实现 Dataset 类：**
- `ColorizationDataset`：全图，cnn_color + inst_fusion `full` 阶段
- `InstanceDataset`：在线随机 crop，inst_fusion `instance` 阶段（无需预计算 npz）
- `FusionDataset`：全图 + torchvision Mask R-CNN 在线 bbox，inst_fusion `fusion` 阶段
- `TestDataset`：推理用，支持两种 method
- `create_dataset(opt, stage, split)` 工厂函数统一入口

---

### Task-05：[P1] CnnColorGenerator 网络实现
**文件：** `code/models/networks.py`、`code/models/cnn_color_model.py`  
**内容：**
- `CnnColorGenerator`：L(1ch) → 313 ab bins 概率图，Conv+BN+ReLU
- 损失：多项交叉熵 + 类别重平衡权重（经验色彩分布）
- 推理：annealed-mean 解码（温度参数 T=0.38）
- 模型工厂 `__init__.py` 注册 `cnn_color`

---

### Task-06：[P1] 训练打通
**文件：** `code/train.py`（`--method cnn_color`）、`code/scripts/train_phase1.sh`  
**内容：**
- 单阶段训练：全图 L → ab，多项交叉熵 + 重平衡
- TensorBoard 日志：loss 曲线
- 烟雾测试：10 iteration，无报错，权重正常保存至 `checkpoints/cnn_color/`

---

### Task-07：[P1] 推理闭环与评测
**文件：** `code/test.py`（`--method cnn_color`）  
**内容：**
- 加载 Phase 1 权重，对灰度图输出上色结果
- 评测：
  1. 主观色彩合理性（视觉检查）
  2. ab 色调分布直方图相似度（巴氏距离）
- Phase 1 验收通过后进入 Phase 2

---

### Task-08：[P2] 双分支网络迁移
**文件：** `code/models/networks.py`、`code/models/inst_fusion_model.py`  
**内容：**
- `InstFusionGenerator`：继承/复用 `CnnColorGenerator`，加载 Phase 1 权重
- `FusionGenerator`：3-conv 预测逐像素融合权重
- `WeightGenerator`：实例与全图加权融合
- 模型工厂注册 `inst_fusion`

---

### Task-09：[P2] 三阶段训练打通
**文件：** `code/train.py`（`--method inst_fusion`）、`code/scripts/train_phase2.sh`  
**内容：**
- `--stage full`：全图上色（从 Phase 1 权重初始化）
- `--stage instance`：实例 crop 上色（从 full 权重初始化）
- `--stage fusion`：融合模块训练（从 full + instance 权重初始化）
- 混合精度：`torch.cuda.amp.autocast`（RTX 4090 优化）
- 烟雾测试：每阶段 10 iteration，权重正常保存

---

### Task-10：[P2] 推理闭环与评测
**文件：** `code/test.py`（`--method inst_fusion`）  
**内容：**
- 有 bbox → 融合推理（instance + full）；无 bbox → 全图回退
- bbox 检测使用 torchvision Mask R-CNN（在线推理，无需离线 npz）
- 对比 Phase 1 结果，评测色彩合理性提升（尤其多物体场景）

---

### Task-11：[P3] ExemplarAttention + StyleHarmonizer 模块
**文件：** `code/models/networks.py`（新增 `ExemplarAttention`、`StyleHarmonizer`）  
**内容：**
- `ExemplarAttention`：从参考图提取 Color Palette（色彩 patch embedding），通过 Cross-Attention 迁移至灰度图语义区域
- `cnn_color + exemplar`：Cross-Attention 插入一次（深层特征处）
- `inst_fusion + exemplar`：全图分支与实例分支各调用一次 `ExemplarAttention`，再接 `StyleHarmonizer`
- `StyleHarmonizer`（**创新点，可选**）：两分支完成 Cross-Attention 后，以全图分支特征为 K/V，实例分支特征为 Q，做一次分支间 Cross-Attention，使实例色彩风格向全局风格空间对齐，再送入 `FusionGenerator`；通过 `--harmonize` flag 启用，不开启时两分支特征直接进入 `FusionGenerator`

---

### Task-12：[P3] Exemplar 推理验证
**文件：** `code/test.py`（`--exemplar --ref_img <path>`）  
**内容：**
- 验证：切换不同参考图，输出颜色风格随之变化
- 定性对比：同一灰度图 + 不同参考图 → 颜色差异显著
- 消融对比：`inst_fusion + exemplar（无 StyleHarmonizer）` vs `inst_fusion + exemplar（有 StyleHarmonizer）`，验证分支间风格协调效果
