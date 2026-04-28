# 子任务分配表

**项目名称：** 黑白照片上色复现（InstColorization CVPR 2020）  
**更新日期：** 2026/04/28

---

## 任务分配表

| 任务编号 | 任务名称 | 负责人 | 状态 | 依赖任务 |
|---------|---------|--------|------|---------|
| Task-01 | 环境配置与 RTX4090 适配 | | ☐ 未开始<br>☐ 进行中<br>☐ 已完成 | 无 |
| Task-02 | code/ 目录骨架搭建 | | ☐ 未开始<br>☐ 进行中<br>☐ 已完成 | Task-01 |
| Task-03 | options 参数体系迁移 | | ☐ 未开始<br>☐ 进行中<br>☐ 已完成 | Task-02 |
| Task-04 | 数据集与 bbox 管线 | | ☐ 未开始<br>☐ 进行中<br>☐ 已完成 | Task-03 |
| Task-05 | 模型与网络结构迁移 | | ☐ 未开始<br>☐ 进行中<br>☐ 已完成 | Task-03 |
| Task-06 | 训练三阶段调度打通 | | ☐ 未开始<br>☐ 进行中<br>☐ 已完成 | Task-04, Task-05 |
| Task-07 | 推理闭环打通 | | ☐ 未开始<br>☐ 进行中<br>☐ 已完成 | Task-04, Task-05 |
| Task-08 | 端到端验收与结果对比 | | ☐ 未开始<br>☐ 进行中<br>☐ 已完成 | Task-06, Task-07 |

---

## 使用说明

1. **认领任务：** 在"负责人"列填写姓名
2. **更新状态：** 将对应行的 ☐ 改为 ☑
3. **注意依赖：** 开始任务前确保依赖任务已完成

---

## 任务详细说明

### Task-01：环境配置与 RTX4090 适配
**目标：** 在远程服务器上建立可用的训练推理环境  
**内容：**
- 基于 `InstColorization-master(reference)/env.yml` 与 `scripts/install.sh` 评估可用性
- 若旧版 torch/detectron2 与 CUDA 不兼容，进行成组版本升级（torch + torchvision + detectron2）
- 记录最终可用的环境版本组合

---

### Task-02：code/ 目录骨架搭建
**目标：** 建立 `code/` 目录结构，搭好入口文件空壳  
**内容：**
- 按 `architecture.md` 建立目录与空文件
- 确保 `scripts/train.sh` 命令可解析（参数路径正确）

---

### Task-03：options 参数体系迁移
**文件：** `code/options/base_options.py`、`code/options/train_options.py`  
**内容：**
- 对齐参考实现的参数接口，适配 `code/` 路径约定
- 保持 `--stage`、`--name`、`--train_img_dir`、`--fineSize` 等核心参数不变

---

### Task-04：数据集与 bbox 管线
**文件：** `code/datasets/fusion_dataset.py`、`code/image_util.py`  
**内容：**
- 迁移 `Training_Full/Instance/Fusion_Dataset` 与 `Fusion_Testing_Dataset`
- 迁移 `gen_maskrcnn_bbox_fromPred`、`get_box_info` 等工具函数
- 补充 bbox 完整性检查（启动前扫描 npz 缺失情况）

---

### Task-05：模型与网络结构迁移
**文件：** `code/models/__init__.py`、`models/base_model.py`、`models/train_model.py`、`models/fusion_model.py`、`models/networks.py`  
**内容：**
- 迁移模型工厂与三类模型（train/fusion）
- 迁移 `SIGGRAPHGenerator`、`InstanceGenerator`、`FusionGenerator`、`WeightGenerator`
- 可根据复现需要做适配性调整（非照搬）

---

### Task-06：训练三阶段调度打通
**文件：** `code/train.py`、`code/scripts/train.sh`  
**内容：**
- 打通 `full -> instance -> fusion` 三阶段
- 每阶段能完成：初始化 → 前向 → 反向 → 保存权重
- 烟雾测试：每阶段跑少量 iteration 验证无报错

---

### Task-07：推理闭环打通
**文件：** `code/inference_bbox.py`、`code/test_fusion.py`  
**内容：**
- 打通 bbox 预测 → 融合上色输出完整链路
- 验证单图端到端可运行，输出图片正常

---

### Task-08：端到端验收与结果对比
**内容：**
- 使用预训练权重跑完整推理，与参考实现结果对比
- 记录定量指标（如 PSNR/FID）或视觉对比截图
- 更新 `progress.md` 完成状态
