import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from data_process.color_object_dataset import CocoColorObjectDataset
from data_process.color_object_utils import (
    choose_caption_for_object,
    dominant_hsv_color,
    ensure_color_object_caption,
    negative_color_for,
    resize_binary_mask,
)
from models.text_color_model import _masked_ab_huber_loss
from scripts.build_phase3_color_object_jsonl import build_records


def test_resize_binary_mask_uses_nearest_neighbor():
    mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)

    resized = resize_binary_mask(mask, (4, 4))

    assert set(np.unique(resized).tolist()) == {0, 1}
    assert resized[:2, :2].sum() == 4
    assert resized[2:, 2:].sum() == 0


def test_hsv_color_rules_detect_neutral_and_hue_colors():
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    yellow = np.zeros((8, 8, 3), dtype=np.uint8)
    yellow[:, :] = [255, 220, 0]
    mask = np.ones((8, 8), dtype=np.uint8)

    assert dominant_hsv_color(black, mask).color == 'black'
    assert dominant_hsv_color(yellow, mask).color == 'yellow'


def test_caption_selection_and_color_object_insertion():
    captions = [
        'A street with people walking.',
        'A car parked near the curb.',
    ]

    selected = choose_caption_for_object(captions, 'car')
    rewritten = ensure_color_object_caption(selected, 'red', 'car')

    assert selected == 'A car parked near the curb.'
    assert rewritten == 'A red car parked near the curb.'


def test_negative_color_prefers_complementary_hue():
    assert negative_color_for('yellow') == 'blue'
    assert negative_color_for('red') == 'cyan'
    assert negative_color_for('black') == 'white'


def test_color_object_dataset_returns_aligned_binary_masks():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, '000001.jpg')
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        arr[:, :] = [255, 0, 0]
        Image.fromarray(arr).save(img_path)

        record = {
            'image_path': img_path,
            'image_id': 1,
            'ann_id': 10,
            'object': 'car',
            'color': 'red',
            'neg_color': 'cyan',
            'caption_pos': 'A red car.',
            'caption_neg': 'A cyan car.',
            'width': 10,
            'height': 10,
            'bbox': [0, 0, 5, 5],
            'segmentation': [[0, 0, 4, 0, 4, 4, 0, 4]],
        }
        records_path = os.path.join(tmp, 'records.jsonl')
        with open(records_path, 'w') as f:
            f.write(json.dumps(record) + '\n')

        ds = CocoColorObjectDataset(records_path, fine_size=8, split='val')
        item = ds[0]

        assert item['rgb_img'].shape == (3, 8, 8)
        assert item['mask_full'].shape == (1, 8, 8)
        assert item['mask_4x'].shape == (1, 2, 2)
        assert set(torch.unique(item['mask_full']).tolist()) <= {0.0, 1.0}
        assert set(torch.unique(item['mask_4x']).tolist()) <= {0.0, 1.0}
        assert item['caption_pos'] == 'A red car.'
        assert item['caption_neg'] == 'A cyan car.'


def test_masked_ab_huber_loss_ignores_unmasked_pixels():
    pred = torch.zeros(1, 2, 2, 2)
    target = torch.zeros(1, 2, 2, 2)
    target[:, :, 1, 1] = 1.0
    mask = torch.zeros(1, 1, 2, 2)
    mask[:, :, 0, 0] = 1.0

    masked_loss = _masked_ab_huber_loss(pred, target, mask)
    full_loss = _masked_ab_huber_loss(pred, target, torch.ones_like(mask))

    assert masked_loss.item() == 0.0
    assert full_loss.item() > 0.0


def test_build_records_excludes_person_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir = os.path.join(tmp, 'images')
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.path.join(img_dir, '000001.jpg')
        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        arr[:, :] = [255, 0, 0]
        Image.fromarray(arr).save(img_path)

        instances = {
            'images': [{'id': 1, 'file_name': '000001.jpg',
                        'width': 16, 'height': 16}],
            'categories': [
                {'id': 1, 'name': 'person'},
                {'id': 3, 'name': 'car'},
            ],
            'annotations': [
                {
                    'id': 10,
                    'image_id': 1,
                    'category_id': 1,
                    'bbox': [0, 0, 8, 8],
                    'segmentation': [[0, 0, 7, 0, 7, 7, 0, 7]],
                    'area': 64,
                    'iscrowd': 0,
                },
                {
                    'id': 11,
                    'image_id': 1,
                    'category_id': 3,
                    'bbox': [8, 8, 7, 7],
                    'segmentation': [[8, 8, 15, 8, 15, 15, 8, 15]],
                    'area': 49,
                    'iscrowd': 0,
                },
            ],
        }
        captions = {
            'images': instances['images'],
            'annotations': [
                {'id': 1, 'image_id': 1,
                 'caption': 'A person standing next to a car.'},
            ],
        }
        instances_file = os.path.join(tmp, 'instances.json')
        captions_file = os.path.join(tmp, 'captions.json')
        out_file = os.path.join(tmp, 'records.jsonl')
        with open(instances_file, 'w') as f:
            json.dump(instances, f)
        with open(captions_file, 'w') as f:
            json.dump(captions, f)

        args = SimpleNamespace(
            img_dir=img_dir,
            instances_file=instances_file,
            captions_file=captions_file,
            out_file=out_file,
            min_area_ratio=0.0,
            min_color_conf=0.0,
            max_records=0,
            exclude_categories=['person'],
        )

        written, skipped = build_records(args)
        records = [json.loads(line) for line in open(out_file)]

        assert written == 1
        assert skipped['excluded_category'] == 1
        assert records[0]['object'] == 'car'
