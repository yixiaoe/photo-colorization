"""Dataset for Phase 3 color-object instance supervision."""
import json
import random

import torch
import torch.utils.data as Data
import torchvision.transforms.functional as TF
from PIL import Image

from data_process.color_object_utils import (
    resize_binary_mask,
    segmentation_to_mask,
)


class CocoColorObjectDataset(Data.Dataset):
    """
    COCO samples preprocessed into color-word + object prompts.

    Each JSONL record must contain:
      image_path, width, height, segmentation, caption_pos, caption_neg.
    """

    def __init__(self, records_file, fine_size=256, split='train',
                 max_dataset_size=float('inf'), random_flip=True):
        self.records_file = records_file
        self.fine_size = int(fine_size)
        self.split = split
        self.random_flip = random_flip and split == 'train'

        self.records = []
        with open(records_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        if max_dataset_size < float('inf'):
            self.records = self.records[:int(max_dataset_size)]

    def __len__(self):
        return len(self.records)

    def _load_mask(self, record):
        mask = segmentation_to_mask(
            record['segmentation'],
            width=record['width'],
            height=record['height'])
        return mask

    def __getitem__(self, idx):
        record = self.records[idx]
        pil_img = Image.open(record['image_path']).convert('RGB')
        mask = self._load_mask(record)

        pil_img = TF.resize(
            pil_img, [self.fine_size, self.fine_size],
            interpolation=TF.InterpolationMode.BILINEAR)
        mask_full = resize_binary_mask(mask, (self.fine_size, self.fine_size))

        if self.random_flip and random.random() < 0.5:
            pil_img = TF.hflip(pil_img)
            mask_full = mask_full[:, ::-1].copy()

        mask_4x = resize_binary_mask(
            mask_full, (self.fine_size // 4, self.fine_size // 4))

        rgb_img = TF.to_tensor(pil_img)
        mask_full_t = torch.from_numpy(mask_full).float().unsqueeze(0)
        mask_4x_t = torch.from_numpy(mask_4x).float().unsqueeze(0)

        return {
            'rgb_img': rgb_img,
            'caption': record['caption_pos'],
            'caption_pos': record['caption_pos'],
            'caption_neg': record['caption_neg'],
            'mask_full': mask_full_t,
            'mask_4x': mask_4x_t,
            'object': record.get('object', ''),
            'color': record.get('color', ''),
            'neg_color': record.get('neg_color', ''),
            'image_id': record.get('image_id', -1),
            'ann_id': record.get('ann_id', -1),
        }
