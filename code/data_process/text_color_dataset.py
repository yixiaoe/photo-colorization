"""
Phase 3 — COCO Captions 数据集。

本模块实现 CocoCaptionDataset，用于加载 COCO2017 图像及其对应的
自然语言描述（captions），为文本引导上色提供训练数据。

数据格式：
  - 图像：COCO train2017 JPEG 文件
  - 标注：captions_train2017.json（每张图 5 条英文描述）

每次 __getitem__ 调用时，从该图像的 5 条 caption 中随机选取 1 条，
增强训练多样性。返回格式：
  {'rgb_img': Tensor(3, H, W), 'caption': str}

支持 train/val split 的不同数据增强策略：
  - train: RandomResizedCrop + RandomHorizontalFlip
  - val:   Resize to fixed size

约束：不修改 data_process/colorization_dataset.py，作为独立文件存在。
"""
import os
import json
import random

import numpy as np
from PIL import Image

import torch
import torch.utils.data as Data
import torchvision.transforms as T


def _collect_images(root):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = []
    for dirpath, _, fnames in os.walk(root):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


class CocoCaptionDataset(Data.Dataset):
    """
    COCO images + captions for text-guided colorization training.

    Each sample returns an RGB image tensor and one randomly chosen caption.
    """

    def __init__(self, img_dir, caption_file, fine_size=256, split='train',
                 max_dataset_size=float('inf')):
        self.img_dir = img_dir
        self.fine_size = fine_size

        with open(caption_file) as f:
            coco = json.load(f)

        id2file = {img['id']: img['file_name'] for img in coco['images']}

        # build image_id → captions mapping
        self._captions = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self._captions:
                self._captions[img_id] = []
            self._captions[img_id].append(ann['caption'])

        # build ordered list of (file_path, image_id)
        self._samples = []
        for img_id, fname in id2file.items():
            if img_id not in self._captions:
                continue
            fpath = os.path.join(img_dir, fname)
            self._samples.append((fpath, img_id))

        if max_dataset_size < float('inf'):
            self._samples = self._samples[:int(max_dataset_size)]

        if split == 'train':
            self.spatial_tfm = T.Compose([
                T.RandomResizedCrop(fine_size, scale=(0.6, 1.0),
                                    ratio=(3/4, 4/3),
                                    interpolation=Image.BILINEAR),
                T.RandomHorizontalFlip(),
            ])
        else:
            self.spatial_tfm = T.Resize(
                (fine_size, fine_size), interpolation=Image.BILINEAR)

        self.tensor_tfm = T.ToTensor()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        fpath, img_id = self._samples[idx]
        pil_img = Image.open(fpath).convert('RGB')
        pil_img = self.spatial_tfm(pil_img)
        rgb_img = self.tensor_tfm(pil_img)

        captions = self._captions[img_id]
        caption = random.choice(captions)

        return {
            'rgb_img': rgb_img,
            'caption': caption,
        }
