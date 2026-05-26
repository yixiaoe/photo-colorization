import os
import json
import tempfile

import torch
import numpy as np
from PIL import Image

from data_process.text_color_dataset import CocoCaptionDataset


def _create_mock_coco(tmp_dir):
    """Create a minimal COCO captions structure for testing."""
    img_dir = os.path.join(tmp_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    # create 3 dummy images
    for i in range(3):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(img_dir, f'{i:06d}.jpg'))

    # create captions JSON
    coco = {
        'images': [
            {'id': 0, 'file_name': '000000.jpg'},
            {'id': 1, 'file_name': '000001.jpg'},
            {'id': 2, 'file_name': '000002.jpg'},
        ],
        'annotations': [
            {'id': 0, 'image_id': 0, 'caption': 'a red car'},
            {'id': 1, 'image_id': 0, 'caption': 'a vehicle in red'},
            {'id': 2, 'image_id': 1, 'caption': 'blue sky with clouds'},
            {'id': 3, 'image_id': 2, 'caption': 'a green tree'},
            {'id': 4, 'image_id': 2, 'caption': 'tree with leaves'},
        ],
    }
    caption_file = os.path.join(tmp_dir, 'captions.json')
    with open(caption_file, 'w') as f:
        json.dump(coco, f)

    return img_dir, caption_file


def test_dataset_length():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir, cap_file = _create_mock_coco(tmp)
        ds = CocoCaptionDataset(img_dir, cap_file, fine_size=64, split='val')
        assert len(ds) == 3


def test_dataset_item_format():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir, cap_file = _create_mock_coco(tmp)
        ds = CocoCaptionDataset(img_dir, cap_file, fine_size=64, split='val')
        item = ds[0]
        assert 'rgb_img' in item
        assert 'caption' in item
        assert item['rgb_img'].shape == (3, 64, 64)
        assert isinstance(item['caption'], str)
        assert len(item['caption']) > 0


def test_dataset_max_size():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir, cap_file = _create_mock_coco(tmp)
        ds = CocoCaptionDataset(img_dir, cap_file, fine_size=64,
                                max_dataset_size=2)
        assert len(ds) == 2


def test_dataset_caption_randomness():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir, cap_file = _create_mock_coco(tmp)
        ds = CocoCaptionDataset(img_dir, cap_file, fine_size=64, split='val')
        # image 0 has 2 captions; sample many times to check randomness
        captions_seen = set()
        for _ in range(50):
            item = ds[0]
            captions_seen.add(item['caption'])
        assert len(captions_seen) >= 2
