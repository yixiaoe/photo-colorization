# 项目模型复现简要计划（基于 InstColorization 参考实现）

## Context
当前目标是在本仓库中复现 `Aware_Image_Colorization_CVPR_2020_paper` 的基线效果，并为后续优化提供可迭代代码底座。约束是：`InstColorization-master(reference)/` 仅作参考，不可修改，且不必完全照搬，可根据复现环境做适当更改与优化；项目代码应落在 `code/`；训练将在远程 RTX4090 服务器运行，因此需要优先处理环境兼容与最小闭环可运行性。

## 推荐实施路径（5 步）
1. **搭建最小代码骨架（先可跑，再完善）**  
   在 `code/` 建立与参考实现一致的最小入口与目录：训练入口、推理入口、模型工厂、数据集、options。先保证命令和参数协议兼容。

2. **先打通训练三阶段调度**  
   对齐 `InstColorization-master(reference)/scripts/train.sh` 与 `train.py` 的 `full -> instance -> fusion` 流程，只做最小可运行链路：能初始化、前向、反向、保存权重。

3. **优先落地 bbox 依赖的数据管线**  
   对齐 `fusion_dataset.py` 的三类训练集/测试集逻辑，严格复用 `<img_dir>_bbox/*.npz` 约定，补齐缺失检查与早期失败提示，避免训练中途因 bbox 缺失崩溃。

4. **打通推理闭环并与训练解耦验证**  
   按 `inference_bbox.py -> test_fusion.py` 顺序完成最小推理闭环，确保“预测 bbox -> 融合上色输出”可单独运行，作为训练前/后快速验收工具。

5. **完成 RTX4090 环境适配并冻结复现命令**  
   基于 `env.yml`、`scripts/install.sh` 制定“可运行环境方案”（必要时升级 torch/cuda/detectron2 组合），最终固化一组可重复执行的命令与目录约定。

## code 架构建议（最小映射）
- `code/train.py`：训练主入口（stage 调度）
- `code/inference_bbox.py`：bbox 预测入口
- `code/test_fusion.py`：融合推理入口
- `code/options/base_options.py`、`code/options/train_options.py`：参数中心
- `code/models/__init__.py`：模型工厂
- `code/models/train_model.py`、`code/models/fusion_model.py`、`code/models/networks.py`：训练/推理模型与网络
- `code/datasets/fusion_dataset.py`：训练/测试数据适配（含 bbox npz 读取）
- `code/scripts/train.sh`：三阶段训练编排脚本

## 需复用的参考实现能力（禁止重造协议）
- 三阶段训练切换与参数入口：`InstColorization-master(reference)/train.py`、`options/train_options.py`
- 模型动态创建模式：`InstColorization-master(reference)/models/__init__.py`
- 融合推理输入协议（crop/full + box_info 多尺度）：`InstColorization-master(reference)/fusion_dataset.py`、`models/fusion_model.py`
- 推理闭环顺序：`InstColorization-master(reference)/inference_bbox.py`、`test_fusion.py`

## 关键风险与缓解
1. **RTX4090 与旧依赖强耦合风险（最高）**  
   参考实现依赖旧版 torch/detectron2，可能与 4090 驱动/CUDA 不兼容。  
   **缓解**：先做最小环境探测；不通则采用“版本迁移方案”（成组升级 torch/cuda/detectron2），保持命令行参数不变。

2. **bbox 是 instance/fusion 的硬前置**  
   缺失或格式不一致会导致数据集阶段失败。  
   **缓解**：训练前增加 bbox 完整性检查（数量/文件名一一对应）。

3. **权重目录命名与脚本默认值不一致**  
   推理可能因 checkpoint 名称不匹配失败。  
   **缓解**：统一 checkpoints 命名约定并在入口参数中显式传递。

## 验证方案（端到端）
1. **最小闭环验证**：
   - 先跑 `inference_bbox` 生成测试集 bbox；
   - 再跑 `test_fusion` 产出结果图，确认无报错且有可视化输出。
2. **阶段性训练验收**：
   - `full/instance/fusion` 各跑小步数（烟雾测试）；
   - 每阶段完成后检查权重文件产出与下阶段加载。
3. **远程服务器验收（RTX4090）**：
   - 连续运行若干 step，确认显存占用稳定、无版本/算子错误。
4. **回归验收**：
   - 在同一小样本上重复执行推理，确认结果与流程稳定可复现。