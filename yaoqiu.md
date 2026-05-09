# lby 分支 Task-05 到 Task-07 完成要求

## 0. 开发目标

本分支 `lby` 的任务是在 `yixiaoe/photo-colorization` 仓库中完成 Phase 1：复现 Zhang et al. 2016 论文《Colorful Image Colorization》。

本阶段只做 2016 论文基线，不做 Phase 2 的 Mask R-CNN 实例分支，也不做 Phase 3 的 exemplar attention。

目标闭环：

1. 输入彩色训练图，转换到 CIE Lab 空间
2. 使用 L 通道作为模型输入
3. 将 ab 通道量化为颜色类别
4. 训练 `Zhang2016Generator` 预测每个像素的 ab 颜色类别分布
5. 使用类别重平衡缓解低饱和、偏灰输出
6. 推理时用 annealed-mean 解码得到连续 ab
7. 合成 L + ab，保存 RGB 上色结果
8. 输出基础评测结果和可视化图片

## 1. 当前仓库基础

仓库已经具备以下基础文件：

- `code/train.py`：训练主入口
- `code/test.py`：推理主入口
- `code/options/base_options.py`：基础参数
- `code/options/train_options.py`：训练和测试参数
- `code/datasets/colorization_dataset.py`：数据集读取与灰度图生成
- `code/models/base_model.py`：模型基类，包含 save/load/scheduler
- `code/models/__init__.py`：按 `--method` 动态加载模型
- `code/models/networks.py`：网络定义文件，目前仍是占位
- `code/models/zhang2016_model.py`：Phase 1 模型逻辑，目前仍是占位
- `code/scripts/train_phase1.sh`：Phase 1 训练脚本
- `code/scripts/test.sh`：推理脚本
- `paper/Colorful_Image_Colorizasion_2016_paper.pdf`：2016 论文

因此 Task-05 到 Task-07 的主要工作不是重新搭工程，而是补齐 Phase 1 的网络、训练、推理和评测。

## 1.1 数据、资源和实验产物目录约定

本分支采用以下目录边界：

- `data/`：只存放本地数据集和测试图片，不提交数据集文件到 Git
- `resources/`：存放复现所需的静态资源，例如 313 类颜色中心和颜色先验
- `checkpoints/`：存放训练得到的模型权重
- `results/`：存放推理输出图片
- `experiments/`：存放实验记录、loss 导出和评测结果

当前已经迁移 Zhang2016 官方资源：

- `resources/zhang2016/pts_in_hull.npy`：313 个 in-gamut ab 颜色中心，形状为 `(313, 2)`
- `resources/zhang2016/prior_probs.npy`：官方颜色类别先验，形状为 `(313,)`

这两个文件来自本地官方实现：

```text
/Users/liubingyi/Documents/CV/official_colorization_2016/resources/pts_in_hull.npy
/Users/liubingyi/Documents/CV/official_colorization_2016/resources/prior_probs.npy
```

Task-05 到 Task-07 优先使用官方 `prior_probs.npy` 完成论文复现闭环。后续如果需要统计本项目数据集的颜色分布，应另存为 `prior_probs_imagenet_mini.npy` 等文件，不覆盖官方 baseline。

## 2. Task-05：Zhang2016Generator 网络实现

### 2.1 需要修改的文件

- `code/models/networks.py`
- `code/models/zhang2016_model.py`
- 必要时补充 `code/util/util.py`

### 2.2 网络输入输出要求

`Zhang2016Generator` 必须满足：

- 输入：Lab 空间的 L 通道，形状为 `N x 1 x H x W`
- 输出：每个像素对应的 ab 颜色类别 logits，形状为 `N x Q x H' x W'`
- `Q = 313`
- 使用 `resources/zhang2016/pts_in_hull.npy` 中的 313 个 in-gamut ab bins
- 输出应是 logits，不在网络内部直接做 softmax，方便训练时使用交叉熵

### 2.3 网络结构要求

需要复现 2016 论文的核心思想：

- 使用全卷积 CNN
- 基本模块为 `Conv2d + BatchNorm2d + ReLU`
- 逐步扩大感受野以理解图像语义
- 最后输出 ab 颜色类别分布

实现时可以先采用工程可跑通版本：

- 保持输入输出空间尺寸可对齐
- 优先保证训练和推理闭环
- 网络层命名清晰，便于后续 Phase 2 复用

建议提供以下类或函数：

- `Zhang2016Generator`
- `define_G(opt)`
- 网络初始化函数，例如 normal/xavier/kaiming

### 2.4 颜色量化要求

必须使用 Zhang2016 的 313 个 ab bin 思路：

- 使用 `resources/zhang2016/pts_in_hull.npy`，形状为 `313 x 2`
- 训练时将真实 ab 映射到最近的颜色 bin
- 推理时将预测概率分布映射回 ab 坐标

现有 `opt.ab_quant` 规则网格只能作为临时调试工具，不能作为 Task-05 到 Task-07 的最终验收路径。

### 2.5 类别重平衡要求

2016 论文的关键点之一是类别重平衡，不能只做普通交叉熵。

需要完成：

- 根据 `resources/zhang2016/prior_probs.npy` 得到每个颜色 bin 的权重
- 低频颜色权重大，高频颜色权重小
- 训练 loss 使用 weighted cross entropy
- 代码中应能清楚区分普通 CE 和 rebalanced CE

建议按论文的平滑先验思路实现：

```text
prior_mix = (1 - gamma) * prior_probs + gamma / Q
weight = 1 / prior_mix
weight = weight / sum(prior_probs * weight)
```

其中 `Q = 313`，`gamma` 可以先使用论文常见设置 `0.5`。如果后续要用 ImageNet-Mini 重新统计先验，应新增文件并通过参数切换，不覆盖官方先验。

### 2.6 annealed-mean 解码要求

推理时不能只用 argmax。需要实现 annealed-mean：

1. 对 logits 除以温度参数 `T`
2. 做 softmax 得到颜色类别概率
3. 用概率对 ab bin 坐标做加权平均

参数要求：

- 默认温度 `T = 0.38`
- 温度应能通过代码常量或参数清楚调整

### 2.7 Task-05 验收标准

完成后应满足：

- `code/models/networks.py` 中存在可实例化的 `Zhang2016Generator`
- `define_G(opt)` 在 `--method zhang2016` 时返回该网络
- `code/models/zhang2016_model.py` 能创建 `netG`
- 一次 forward 不报错
- 输入 batch 的 L 通道能输出 ab logits
- logits 的类别维度与量化类别数一致

## 3. Task-06：Phase 1 训练打通

### 3.1 需要修改的文件

- `code/models/zhang2016_model.py`
- `code/train.py`
- `code/scripts/train_phase1.sh`
- 必要时补充 `code/util/visualizer.py`

### 3.2 训练数据流

训练时数据流必须为：

1. `ColorizationDataset` 返回 `rgb_img` 和 `gray_img`
2. 使用 `rgb_img` 转换得到 Lab
3. L 通道作为输入 `real_A`
4. ab 通道作为真实目标 `real_B`
5. 将 `real_B` 量化为类别标签
6. `netG(real_A)` 输出颜色类别 logits
7. 使用 weighted cross entropy 计算 loss
8. 反向传播并更新 `netG`

### 3.3 `Zhang2016Model` 必须实现的方法

`code/models/zhang2016_model.py` 中至少需要实现：

- `initialize(opt)`
- `set_input(data)`
- `forward()`
- `optimize_parameters()`
- `get_current_losses()`
- `get_current_visuals()`

### 3.4 checkpoint 要求

训练时必须能保存：

- `checkpoints/<name>/latest_net_G.pth`
- epoch checkpoint，例如 `checkpoints/<name>/1_net_G.pth`

建议 Phase 1 默认实验名使用：

```bash
--name zhang2016
```

### 3.5 TensorBoard / 日志要求

训练需要至少记录：

- 总 loss
- weighted cross entropy loss
- 当前 epoch / iteration

如果 `Visualizer` 已经支持 TensorBoard，则直接接入；如果没有，需要保证命令行日志可读。

### 3.6 烟雾测试要求

必须能完成 10 iteration 训练，不报错，且能保存权重。

推荐命令：

```bash
cd code
python train.py \
  --method zhang2016 \
  --stage full \
  --dataset imagenet_mini \
  --data_dir ../data/imagenet_mini/train \
  --name zhang2016 \
  --batch_size 2 \
  --niter 1 \
  --niter_decay 0 \
  --print_freq 1 \
  --save_latest_freq 10 \
  --max_dataset_size 20
```

如果使用 CIFAR-10：

```bash
cd code
python train.py \
  --method zhang2016 \
  --stage full \
  --dataset cifar10 \
  --data_dir ../data/cifar10 \
  --name zhang2016 \
  --batch_size 4 \
  --niter 1 \
  --niter_decay 0 \
  --print_freq 1 \
  --save_latest_freq 10 \
  --max_dataset_size 40
```

本地 CPU 烟雾测试命令：

```bash
cd code
python train.py \
  --method zhang2016 \
  --stage full \
  --dataset imagenet_mini \
  --data_dir ../data/imagenet_mini/train \
  --name zhang2016 \
  --gpu_ids -1 \
  --batch_size 1 \
  --niter 1 \
  --niter_decay 0 \
  --print_freq 1 \
  --save_latest_freq 10 \
  --max_dataset_size 10
```

### 3.7 Task-06 验收标准

完成后应满足：

- `python train.py --method zhang2016 ...` 能启动
- 训练 loss 能正常输出
- 10 iteration 烟雾测试无报错
- checkpoint 能保存到 `checkpoints/zhang2016/`
- 重新运行时可以加载已有 checkpoint

## 4. Task-07：Phase 1 推理闭环与评测

### 4.1 需要修改的文件

- `code/models/zhang2016_model.py`
- `code/test.py`
- `code/util/util.py`
- `code/scripts/test.sh`
- 可新增评测辅助函数，但应保持范围清晰

### 4.2 推理数据流

推理时数据流必须为：

1. `TestDataset` 读取测试图片
2. 将图片转换到 Lab
3. 提取 L 通道
4. 加载 `latest_net_G.pth`
5. `netG(L)` 输出 ab 类别 logits
6. 使用 annealed-mean 解码得到 ab
7. 合成 L + ab
8. 转换回 RGB
9. 保存上色图片

### 4.3 输出图片要求

推理结果至少保存：

- 输入灰度图
- 预测上色图

建议视觉 key：

- `gray`
- `fake_color`
- 如果测试图本身是彩色图，也可以输出 `real_color`

默认输出目录：

```text
results/images/
```

### 4.4 评测要求

Task-07 至少完成两类评测：

1. 主观视觉检查
   - 颜色是否自然
   - 是否大面积偏灰
   - 天空、草地、人脸、动物、车辆等常见物体颜色是否合理

2. ab 直方图相似度
   - 对预测图和真实图统计 ab 色调分布
   - 计算巴氏距离或 KL 散度
   - 输出每张图或平均分数

注意：ab 直方图相似度需要真实彩色图作为参照。如果测试目录只有纯黑白照片，则只做主观视觉检查，不计算真实分布指标。

建议优先实现巴氏距离：

```text
Bhattacharyya distance = -ln(sum(sqrt(hist_pred * hist_gt)))
```

### 4.5 推理测试命令

推荐命令：

```bash
cd code
python test.py \
  --method zhang2016 \
  --test_img_dir ../data/imagenet_mini/test \
  --name zhang2016 \
  --which_epoch latest \
  --results_img_dir ../results/zhang2016
```

### 4.6 Task-07 验收标准

完成后应满足：

- `python test.py --method zhang2016 ...` 能加载 Phase 1 权重
- 能对测试目录中的图片逐张推理
- 能保存 RGB 上色结果
- 输出图片没有明显通道顺序错误
- 输出图片不是纯灰图或全黑图
- 至少能给出主观样例对比
- 至少能输出一种 ab 分布相似度指标

## 5. 代码质量要求

### 5.1 不要破坏现有接口

必须保持这些入口可用：

- `python train.py --method zhang2016`
- `python test.py --method zhang2016`
- `create_model(opt)`
- `create_dataset(opt, stage, split)`

### 5.2 文件范围控制

本阶段优先只改：

- `code/models/networks.py`
- `code/models/zhang2016_model.py`
- `code/util/util.py`
- `code/train.py`
- `code/test.py`
- `code/scripts/train_phase1.sh`
- `code/scripts/test.sh`

除非确实必要，不修改 Phase 2/Phase 3 文件。

### 5.3 设备兼容

代码必须支持：

- GPU：`--gpu_ids 0`
- CPU：`--gpu_ids -1`

如果没有 CUDA，烟雾测试应至少能在 CPU 上跑小 batch。

### 5.4 尺寸兼容

需要注意：

- 网络输出尺寸可能小于输入尺寸
- loss 计算前需要让 label 与 logits 的空间尺寸一致
- 推理时 ab 需要上采样回 L 通道原尺寸
- 合成 RGB 前必须确认 Lab 数值范围正确

### 5.5 数值范围

需要统一：

- RGB tensor 范围：`[0, 1]`
- L 通道归一化方式：遵循 `opt.l_cent` 和 `opt.l_norm`
- ab 归一化方式：遵循 `opt.ab_norm`
- 输出 RGB 需要 clamp 到 `[0, 1]`

## 6. 最终交付物

完成 Task-05 到 Task-07 后，本分支应至少包含：

- 可运行的 `Zhang2016Generator`
- 可训练的 `Zhang2016Model`
- weighted cross entropy loss
- annealed-mean 解码
- Phase 1 训练脚本
- Phase 1 推理脚本
- 10 iteration 烟雾测试记录
- 至少一组灰度输入和上色输出结果
- 至少一种 ab 分布评测结果

## 7. 本阶段不做的内容

以下内容不属于当前 Task-05 到 Task-07 的交付范围：

- Task-08 双分支网络迁移
- Task-09 Phase 2 三阶段训练
- Task-10 Phase 2 融合推理
- Task-11 ExemplarAttention
- Task-12 Exemplar 推理验证
- Detectron2 安装
- 离线 bbox npz 预处理

## 8. 建议执行顺序

1. 在 `networks.py` 实现 `Zhang2016Generator`
2. 在 `zhang2016_model.py` 实现训练 forward 和 loss
3. 跑一次单 batch forward 检查 shape
4. 跑 10 iteration 训练烟雾测试
5. 确认 checkpoint 保存
6. 实现 annealed-mean 解码和 visuals
7. 跑 `test.py` 保存上色结果
8. 实现 ab 直方图评测
9. 整理样例输出和运行命令
