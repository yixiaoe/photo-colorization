"""
Phase 3 dataset: COCO full images + per-instance color/object captions.

Loads JSONL produced by scripts/build_phase3_jsonl.py. Records sharing
the same image_id are grouped into one sample, so each __getitem__
returns full_rgb plus N instance crops, all aligned at fineSize.

Returned dict (consumed by TextColorModel.set_input):
    full_rgb       (1, 3, H, W)
    cropped_rgb    (N, 3, H, W)
    class_labels   (N,)  long
    box_info       (N, 6) long  - bbox info at H
    box_info_2x    (N, 6) long  - at H/2
    box_info_4x    (N, 6) long  - at H/4
    box_info_8x    (N, 6) long  - at H/8
    masks_full     (N, 1, H, W) float in {0, 1}
    masks_4x       (N, 1, H/4, W/4)
    class_names    list[str] of len N
    caption_pos    list[str] of len N
    caption_neg    list[str] of len N
    caption_bg_pos str
    caption_bg_neg str
    empty_box      bool
    file_id        str
"""
import json
import os
import random
from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

# read-only reuse — DO NOT mutate this module
from data_process.colorization_dataset import get_box_info


# Caption templates (same set as scripts/cache_clip_embeddings.py)
INST_TEMPLATES = [
    'a {c} {o}',
    'a photo of a {c} {o}',
    'the {o} is {c}',
    '{c} {o}',
    'a {c} coloured {o}',
]

BG_POS_TEMPLATES = [
    'outdoor scene',
    'a sunny day',
    'a blue sky',
    'a green grass field',
]

BG_NEG_TEMPLATES = [
    'indoor room',
    'at night',
    'a dark scene',
    'a white wall',
]


def _make_inst_caption(color: str, class_name: str, rng: random.Random) -> str:
    return rng.choice(INST_TEMPLATES).format(c=color, o=class_name)


def _make_bg_captions(rng: random.Random):
    pos = rng.choice(BG_POS_TEMPLATES)
    neg = rng.choice(BG_NEG_TEMPLATES)
    return pos, neg


class TextColorCocoDataset(Data.Dataset):
    """
    Records JSONL → grouped-by-image sample provider.

    Args:
        opt:            options namespace (fineSize, ab_norm, etc.)
        jsonl_file:     path to JSONL produced by build_phase3_jsonl.py
        instances_file: optional COCO instances JSON; required to fetch
                        segmentation mask via pycocotools (mandatory in
                        practice, validated at construction)
        img_dir:        directory holding the COCO images
        max_inst:       hard cap on instances per image (after sort by area)
        split:          'train' or 'val' (controls augmentation)
    """

    def __init__(self, opt, jsonl_file: str, instances_file: str,
                 img_dir: str, max_inst: int = 8, split: str = 'train'):
        self.opt = opt
        self.size = opt.fineSize
        self.split = split
        self.max_inst = max_inst
        self.img_dir = img_dir

        if COCO is None:
            raise ImportError('pycocotools is required for TextColorCocoDataset')

        # ── load JSONL and group by image_id ──────────────────────────
        self.records_by_image = defaultdict(list)
        with open(jsonl_file) as f:
            for line in f:
                rec = json.loads(line)
                self.records_by_image[rec['image_id']].append(rec)

        self.image_ids = sorted(self.records_by_image.keys())
        # Drop oversize groups; keep the top-max_inst by area_ratio
        for iid in self.image_ids:
            recs = self.records_by_image[iid]
            if len(recs) > max_inst:
                recs.sort(key=lambda r: -r['area_ratio'])
                self.records_by_image[iid] = recs[:max_inst]

        print(f'[TextColorCocoDataset] {jsonl_file}: '
              f'{len(self.image_ids)} images / '
              f'{sum(len(v) for v in self.records_by_image.values())} instances')

        # ── COCO for mask rasterisation ───────────────────────────────
        if not os.path.isfile(instances_file):
            raise FileNotFoundError(
                f'instances_file not found: {instances_file}')
        print(f'[TextColorCocoDataset] loading COCO masks from {instances_file}')
        self.coco = COCO(instances_file)
        # Build ann_id -> ann dict for fast lookup
        self._ann_by_id = {a['id']: a for a in self.coco.dataset['annotations']}

        self.resize_pil = T.Resize((self.size, self.size),
                                   interpolation=Image.BILINEAR)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        rng = random.Random(idx + (1234 if self.split == 'train' else 0))
        image_id = self.image_ids[idx]
        records: List[dict] = self.records_by_image[image_id]

        # ── load full image ───────────────────────────────────────────
        fname = records[0]['image_file']
        fpath = os.path.join(self.img_dir, fname)
        pil_orig = Image.open(fpath).convert('RGB')
        orig_W, orig_H = pil_orig.size

        # spatial transform — train: random hflip; resize to fineSize
        do_flip = self.split == 'train' and rng.random() < 0.5
        pil_resized = self.resize_pil(pil_orig)
        if do_flip:
            pil_resized = pil_resized.transpose(Image.FLIP_LEFT_RIGHT)
        full_rgb = self.to_tensor(pil_resized)        # (3, H, W)

        # ── per-instance bbox / mask / caption ────────────────────────
        sz = self.size
        N = len(records)
        cropped_rgb_list = []
        class_labels = []
        class_names = []
        captions_pos, captions_neg = [], []
        box_info    = np.zeros((N, 6), dtype=np.int64)
        box_info_2x = np.zeros((N, 6), dtype=np.int64)
        box_info_4x = np.zeros((N, 6), dtype=np.int64)
        box_info_8x = np.zeros((N, 6), dtype=np.int64)
        masks_full = torch.zeros((N, 1, sz, sz), dtype=torch.float32)

        for i, rec in enumerate(records):
            x, y, w, h = rec['bbox']
            # bbox in original image coords; clip to valid range
            x0 = max(0, int(round(x)))
            y0 = max(0, int(round(y)))
            x1 = min(orig_W, int(round(x + w)))
            y1 = min(orig_H, int(round(y + h)))
            if x1 <= x0 or y1 <= y0:
                x1, y1 = x0 + 1, y0 + 1

            # take crop from the ORIGINAL image (better resolution)
            crop_pil = pil_orig.crop((x0, y0, x1, y1))
            crop_pil = self.resize_pil(crop_pil)
            if do_flip:
                crop_pil = crop_pil.transpose(Image.FLIP_LEFT_RIGHT)
            cropped_rgb_list.append(self.to_tensor(crop_pil))

            # bbox after horizontal flip is mirrored: x0' = orig_W - x1, x1' = orig_W - x0
            if do_flip:
                fx0 = orig_W - x1
                fx1 = orig_W - x0
                fy0, fy1 = y0, y1
            else:
                fx0, fy0, fx1, fy1 = x0, y0, x1, y1

            # box_info at four scales
            box_info[i]    = get_box_info((fx0, fy0, fx1, fy1), pil_orig.size, sz)
            box_info_2x[i] = get_box_info((fx0, fy0, fx1, fy1), pil_orig.size, sz // 2)
            box_info_4x[i] = get_box_info((fx0, fy0, fx1, fy1), pil_orig.size, sz // 4)
            box_info_8x[i] = get_box_info((fx0, fy0, fx1, fy1), pil_orig.size, sz // 8)

            # mask at full image resolution (orig_H, orig_W), resize to fineSize
            ann = self._ann_by_id.get(rec['ann_id'])
            if ann is None:
                # fall back to bbox mask
                m_full = np.zeros((orig_H, orig_W), dtype=np.uint8)
                m_full[y0:y1, x0:x1] = 1
            else:
                m_full = self.coco.annToMask(ann)
                if m_full.shape != (orig_H, orig_W):
                    # mask uses meta-W/H; resize to actual image size
                    m_pil = Image.fromarray(m_full * 255).resize(
                        (orig_W, orig_H), resample=Image.NEAREST)
                    m_full = (np.asarray(m_pil) > 127).astype(np.uint8)

            m_pil = Image.fromarray(m_full * 255).resize(
                (sz, sz), resample=Image.NEAREST)
            m_arr = (np.asarray(m_pil) > 127).astype(np.float32)
            if do_flip:
                m_arr = m_arr[:, ::-1].copy()
            masks_full[i, 0] = torch.from_numpy(m_arr)

            class_labels.append(int(rec['class_id']))
            class_names.append(rec['class_name'])
            captions_pos.append(
                _make_inst_caption(rec['color_pos'], rec['class_name'], rng))
            captions_neg.append(
                _make_inst_caption(rec['color_neg'], rec['class_name'], rng))

        cropped_rgb = torch.stack(cropped_rgb_list)
        class_labels_t = torch.tensor(class_labels, dtype=torch.long)

        # downsampled mask for CE / KL at H/4 resolution
        masks_4x = F.interpolate(masks_full, size=(sz // 4, sz // 4),
                                 mode='nearest')

        cap_bg_pos, cap_bg_neg = _make_bg_captions(rng)

        return {
            'full_rgb':       full_rgb.unsqueeze(0),                   # (1, 3, H, W)
            'cropped_rgb':    cropped_rgb,                              # (N, 3, H, W)
            'class_labels':   class_labels_t,                           # (N,)
            'class_names':    class_names,                              # list[str]
            'box_info':       torch.from_numpy(box_info),
            'box_info_2x':    torch.from_numpy(box_info_2x),
            'box_info_4x':    torch.from_numpy(box_info_4x),
            'box_info_8x':    torch.from_numpy(box_info_8x),
            'masks_full':     masks_full,                               # (N, 1, H, W)
            'masks_4x':       masks_4x,                                 # (N, 1, H/4, W/4)
            'caption_pos':    captions_pos,
            'caption_neg':    captions_neg,
            'caption_bg_pos': cap_bg_pos,
            'caption_bg_neg': cap_bg_neg,
            'empty_box':      False,
            'file_id':        os.path.splitext(fname)[0],
        }


def text_color_collate(batch):
    """
    batch_size must be 1 for Phase 3 (variable-N instances per image).
    The default collate would try to stack across the batch dim, which
    fails because N varies. We just unwrap the single-item list.
    """
    assert len(batch) == 1, (
        'TextColorCocoDataset requires batch_size=1 (N instances vary per image)')
    return batch[0]
