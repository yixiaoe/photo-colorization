# Photo Colorization

Instance-aware black-and-white photo colorization with FiLM semantic conditioning.

## Overview

PyTorch implementation of two colorization methods:

- **Phase 1 — CNN Colorization** (Zhang et al. 2016): L-channel → 313-bin ab probability distribution, rebalanced cross-entropy, annealed-mean decoding
- **Phase 2** (Su et al. CVPR 2020): Dual-branch architecture with Mask R-CNN detections, per-layer soft-attention fusion of full-image and instance features. **We introduce FiLM conditioning** (conv4–conv7) to inject Mask R-CNN class labels into the instance branch, so the network knows what object it is coloring — the original feeds raw crops without semantic cues

## Project Structure

```
photo-colorization/
├── code/
│   ├── train.py                   # Training entry point
│   ├── test.py                    # Inference entry point
│   ├── models/
│   │   ├── networks.py            # Network architectures
│   │   ├── cnn_color_model.py     # Phase 1 model
│   │   ├── inst_fusion_model.py   # Phase 2 model
│   │   └── base_model.py
│   ├── options/
│   │   ├── base_options.py
│   │   └── train_options.py
│   ├── data_process/
│   │   └── colorization_dataset.py
│   ├── util/
│   │   └── util.py                # Lab/RGB conversion, 313-bin utils
│   ├── scripts/
│   │   └── train_phase2.sh
│   └── checkpoints/               # Trained weights
├── docs/
└── paper/                         # Reference papers
```

## Inference

```bash
# Phase 1 baseline
cd code
python test.py --method cnn_color --name cnn_color_imagenet --which_epoch 60 \
    --test_img_dir datasets/test --results_img_dir results

# Phase 2 — full backbone only (single-stage result)
python test.py --method inst_fusion --stage full \
    --name inst_fusion_full --which_epoch 80 \
    --test_img_dir datasets/test --results_img_dir results

# Phase 2 — per-instance crops (FiLM instance branch only)
python test.py --method inst_fusion --stage instance \
    --name inst_fusion_instance --which_epoch 25 \
    --test_img_dir datasets/test --results_img_dir results

# Phase 2 — full pipeline (Mask R-CNN → FiLM → Fusion → colorized output)
python test.py --method inst_fusion --stage fusion \
    --name inst_fusion_fusion --which_epoch 25 \
    --full_ckpt checkpoints/inst_fusion_full/80_net_G.pth \
    --inst_ckpt checkpoints/inst_fusion_instance/25_net_G.pth \
    --test_img_dir datasets/test --results_img_dir results
```

## Phase 1 Training

```bash
cd code
python train.py --method cnn_color --dataset imagenet_mini \
    --name cnn_color_imagenet --data_dir datasets/imagenet_mini \
    --niter 100 --niter_decay 100 --lr 1e-4 --batch_size 16
```

## Phase 2 Training (Three-Stage)

```bash
cd code

# Stage 1 — Full-image backbone (COCO2017, ~12h on RTX 5090)
python train.py --method inst_fusion --stage full \
    --name inst_fusion_full --data_dir datasets/coco/train2017 \
    --niter 50 --niter_decay 50 --lr 3e-5 --batch_size 32

# Stage 2 — Instance branch with FiLM (COCO2017 GT crops, ~18h)
python train.py --method inst_fusion --stage instance \
    --name inst_fusion_instance \
    --data_dir datasets/coco/train2017 \
    --ann_file datasets/coco/annotations/instances_train2017.json \
    --full_ckpt checkpoints/inst_fusion_full/net_G.pth \
    --niter 20 --niter_decay 20 --lr 3e-5 --lr_backbone 1e-5 --batch_size 32

# Stage 3 — Fusion WeightGenerators (requires offline bbox cache)
python train.py --method inst_fusion --stage fusion \
    --name inst_fusion_fusion \
    --data_dir datasets/coco/train2017 \
    --bbox_cache datasets/bbox_cache_train2017.json \
    --full_ckpt checkpoints/inst_fusion_full/net_G.pth \
    --inst_ckpt checkpoints/inst_fusion_instance/net_G.pth \
    --niter 15 --niter_decay 15 --lr 2e-5 --batch_size 1
```


## Loss

All stages use **CE + 3×Huber**. CE drives vivid color prediction via rebalanced 313-bin classification; Huber regularizes spatial consistency.

## Requirements

- PyTorch 2.0+, Python 3.8+
- CUDA 11.8+ (training), CPU inference supported
- RTX 4090/5090 recommended (24GB+ VRAM)
- No Detectron2 (torchvision Mask R-CNN)

## Trained Weights

| File | Description |
|------|-------------|
| `inst_fusion_full/80_net_G.pth` | Stage 1 backbone (COCO, ~31M params) |
| `inst_fusion_instance/25_net_G.pth` | Stage 2 FiLM instance (COCO, ~31M+270K) |
| `inst_fusion_fusion/25_net_G.pth` | Stage 3 fusion (COCO + bbox) |
| `cnn_color_imagenet/60_net_G.pth` | Phase 1 baseline |
| `mask_rcnn/maskrcnn_resnet50_fpn.pth` | Mask R-CNN detector |
