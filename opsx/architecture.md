# 项目架构说明

**更新日期：** 2026/04/28

---

## 整体结构

```
photo-colorization/
├── InstColorization-master(reference)/   # 参考实现（只读，不可修改）
├── code/                                 # 项目复现代码（主要开发区）
├── opsx/                                 # 计划/进度文档
└── paper/                                # 论文资料
```

---

## code/ 目录架构

```
code/
├── train.py                  # 训练主入口（stage 调度）
├── inference_bbox.py         # bbox 预测入口（Detectron2）
├── test_fusion.py            # 融合推理入口
├── image_util.py             # bbox npz 读取、box_info 计算
├── options/
│   ├── base_options.py       # 基础参数解析
│   └── train_options.py      # 训练/测试参数
├── models/
│   ├── __init__.py           # 模型工厂（按 --model 动态加载）
│   ├── base_model.py         # 基类（save/load/scheduler）
│   ├── train_model.py        # 三阶段训练逻辑
│   ├── fusion_model.py       # 推理融合逻辑
│   └── networks.py           # 网络结构定义
├── datasets/
│   └── fusion_dataset.py     # 训练/测试 Dataset（含 bbox npz 依赖）
├── util/
│   ├── util.py               # Lab/RGB 转换、hint/mask 生成
│   └── visualizer.py         # 训练可视化（visdom）
└── scripts/
    ├── train.sh              # 三阶段训练编排
    ├── test_mask.sh          # 推理闭环脚本
    ├── install.sh            # 环境安装
    └── download_model.sh     # 权重下载
```

---

## 端到端流程

```
输入图片
   │
   ▼
inference_bbox.py          ← Detectron2 Mask R-CNN
   │  输出: <img_dir>_bbox/*.npz
   ▼
test_fusion.py
   ├── Fusion_Testing_Dataset（读取图片 + bbox）
   ├── FusionModel（加载 G / GF / GComp 权重）
   │     ├── 有 bbox → 融合推理（instance + full）
   │     └── 无 bbox → 全图回退
   └── 输出上色结果图
```

## 训练三阶段

| Stage | Dataset | 训练目标 | 初始权重来源 |
|-------|---------|---------|------------|
| full | Training_Full_Dataset | 全图上色分支 netG | siggraph_retrained |
| instance | Training_Instance_Dataset | 实例 crop 分支 netG | coco_full/latest_net_G |
| fusion | Training_Fusion_Dataset | 融合模块 netGF | coco_full + coco_instance |

---

## 关键依赖关系

- `Training_Instance_Dataset` / `Training_Fusion_Dataset` / `Fusion_Testing_Dataset` 均硬依赖 `<img_dir>_bbox/*.npz`
- 训练 stage=instance/fusion 前必须先完成 bbox 预测
- 推理前需确认 checkpoints 目录命名与脚本参数一致
