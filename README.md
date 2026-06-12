# Photo Colorization

Instance-aware black-and-white photo colorization with FiLM semantic conditioning and CLIP text-guided control.

## Overview

PyTorch implementation of three colorization methods:

- **Phase 1 — CNN Colorization** (Zhang et al. 2016): L-channel → 313-bin ab probability distribution, rebalanced cross-entropy, annealed-mean decoding
- **Phase 2** (Su et al. CVPR 2020): Dual-branch architecture with Mask R-CNN detections, per-layer soft-attention fusion of full-image and instance features. **We introduce FiLM conditioning** (conv4–conv7) to inject Mask R-CNN class labels into the instance branch, so the network knows what object it is coloring — the original feeds raw crops without semantic cues
- **Phase 3 — CLIP Text-Guided Colorization**: A lightweight TextAdapter (~2.89M params) sits on top of frozen Phase 2 weights. Users assign a natural-language color prompt to each Mask R-CNN instance (e.g. `inst:0=a red dog`); the adapter modulates instance features at 5 decoder layers via FiLM, trained with a ranking loss that forces the positive prompt to produce more accurate colors than a negative counterpart. Phase 2 code and weights are untouched.

## Project Structure

```
photo-colorization/
├── code/
│   ├── train.py                        # Training entry point (Phase 1/2)
│   ├── test.py                         # Inference entry point (all phases)
│   ├── models/
│   │   ├── networks.py                 # Phase 1/2 architectures
│   │   ├── cnn_color_model.py          # Phase 1 model
│   │   ├── inst_fusion_model.py        # Phase 2 model
│   │   ├── text_color_model.py         # Phase 3 model
│   │   ├── text_color_networks.py      # Phase 3: TextAdapter + TextColorPipeline
│   │   └── base_model.py
│   ├── options/
│   │   ├── base_options.py
│   │   └── train_options.py
│   ├── data_process/
│   │   ├── colorization_dataset.py     # Phase 1/2 dataset
│   │   └── text_color_dataset.py       # Phase 3 dataset (COCO + captions)
│   ├── util/
│   │   ├── util.py                     # Lab/RGB conversion, 313-bin utils
│   │   ├── clip_encoder.py             # Phase 3: CLIP text encoder wrapper
│   │   └── maskrcnn_helper.py          # Phase 3: Mask R-CNN with masks
│   ├── scripts/
│   │   ├── train_phase2.sh
│   │   ├── train_phase3.sh             # Phase 3 training launcher
│   │   ├── build_phase3_jsonl.py       # COCO → JSONL (HSV color words)
│   │   └── cache_clip_embeddings.py    # Pre-encode CLIP text embeddings
│   └── checkpoints/                    # Trained weights
├── docs/
└── paper/                              # Reference papers
```

## Web Demo

A localhost web interface is available for interactive colorization without using the command line.

```bash
cd code
python app.py
# Open http://localhost:5000 in your browser
```

**Usage:**
1. Upload a grayscale or colour photo (jpg/png, ≤ 4 MB)
2. Select a method — Phase 1 (CNN), Phase 2 (FiLM, default), or Phase 3 (Text)
3. **Phase 2:** detected instances are listed; click any instance card after colorizing to view its pre-fusion branch result
4. **Phase 3:** assign a colour prompt to each instance via colour swatches or free text (e.g. `a red dog`); click **Reset** to restore the default prompt
5. Click **Colorize** and compare the result against the original / grayscale input
6. Download the colorized image with the **Download** button

> Colour images are automatically converted to grayscale for inference; the result panel shows original, grayscale, and colorized side by side.

---

## Inference

```bash
# Phase 3 — text-guided colorization (recommended)
cd code
python test.py --method text_color \
    --which_epoch 10 --name phase3_text_color \
    --full_ckpt   checkpoints/inst_fusion_full/80_net_G.pth \
    --inst_ckpt   checkpoints/inst_fusion_instance/25_net_G.pth \
    --fusion_ckpt checkpoints/inst_fusion_fusion/25_net_G.pth \
    --image datasets/test/dog.png \
    --prompt "inst:0=a brown dog" \
    --results_img_dir results/phase3 \
    --gpu_ids 0
    # --fineSize 256 (default, do NOT use 224)
    # --ab_cap 0.45  (default, suppresses over-saturated blobs)
    # Supported colors: red/yellow/orange/brown/green/gray/blue/purple
    # Unsupported: black/white

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


## Phase 3 Training

```bash
cd code
# 1. Build JSONL from COCO2017 annotations
python scripts/build_phase3_jsonl.py \
    --instances_file datasets/coco/annotations/instances_train2017.json \
    --img_dir datasets/coco/train2017 \
    --out_file datasets/phase3/color_object_train.jsonl

# 2. Pre-cache CLIP text embeddings
python scripts/cache_clip_embeddings.py

# 3. Train (10 epochs, ~18h on RTX 5090)
bash scripts/train_phase3.sh
```

## Loss

**Phase 1/2:** CE + 3×Huber. CE drives vivid color prediction via rebalanced 313-bin classification; Huber regularizes spatial consistency.

**Phase 3:** adds a ranking loss `max(0, margin + KL(gt||pos) − KL(gt||neg))` over 313-bin probabilities. Each step runs two forwards (positive + negative prompt); the adapter is trained to make the positive prompt produce a color distribution closer to ground truth than the negative.

## Requirements

- PyTorch 2.0+, Python 3.8+
- CUDA 11.8+ (training), CPU inference supported
- RTX 4090/5090 recommended (24GB+ VRAM)
- No Detectron2 (torchvision Mask R-CNN)

## Trained Weights

| File | Description |
|------|-------------|
| `inst_fusion_full/80_net_G.pth` | Phase 2 Stage 1 backbone (COCO, ~31M params) |
| `inst_fusion_instance/25_net_G.pth` | Phase 2 Stage 2 FiLM instance (COCO, ~31M+270K) |
| `inst_fusion_fusion/25_net_G.pth` | Phase 2 Stage 3 fusion (COCO + bbox) |
| `phase3_text_color/10_net_T.pth` | **Phase 3 TextAdapter final** (COCO, 2.89M params, 11MB) |
| `cnn_color_imagenet/60_net_G.pth` | Phase 1 baseline |
| `clip/ViT-B-32-quickgelu.pt` | CLIP text encoder (Phase 3, text tower only) |
| `mask_rcnn/maskrcnn_resnet50_fpn.pth` | Mask R-CNN detector |
