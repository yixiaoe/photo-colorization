"""
Build Phase 3 color-object instance records from COCO annotations.

Example:
  python scripts/build_phase3_color_object_jsonl.py \
      --img_dir /root/autodl-tmp/coco2017/train2017 \
      --instances_file /root/autodl-tmp/coco2017/annotations/instances_train2017.json \
      --captions_file /root/autodl-tmp/coco2017/annotations/captions_train2017.json \
      --out_file data/phase3_color_object_no_person_train.jsonl
"""
import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_process.color_object_utils import (
    choose_caption_for_object,
    dominant_hsv_color,
    ensure_color_object_caption,
    negative_color_for,
    segmentation_to_mask,
)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _build_one_record(ann, args, images, categories, captions_by_img,
                      exclude_categories):
    if ann.get('iscrowd', 0) != 0:
        return None, 'crowd'
    if not isinstance(ann.get('segmentation'), list):
        return None, 'non_polygon_segmentation'

    img_info = images.get(ann['image_id'])
    object_name = categories.get(ann['category_id'])
    if img_info is None or object_name is None:
        return None, 'missing_image_or_category'
    if object_name.lower() in exclude_categories:
        return None, 'excluded_category'

    area_ratio = ann.get('area', 0.0) / max(
        1.0, float(img_info['width'] * img_info['height']))
    if area_ratio < args.min_area_ratio:
        return None, 'small_area'

    caption = choose_caption_for_object(
        captions_by_img.get(ann['image_id'], []), object_name)
    if caption is None:
        return None, 'caption_without_object'

    image_path = os.path.join(args.img_dir, img_info['file_name'])
    if not os.path.isfile(image_path):
        return None, 'missing_image_file'

    try:
        with Image.open(image_path) as pil_img:
            rgb = np.asarray(pil_img.convert('RGB'))
            mask = segmentation_to_mask(
                ann['segmentation'], img_info['width'], img_info['height'])
    except Exception:
        return None, 'mask_or_image_error'

    color_est = dominant_hsv_color(rgb, mask)
    if color_est.confidence < args.min_color_conf:
        return None, 'low_color_conf'

    color = color_est.color
    neg_color = negative_color_for(color)
    caption_pos = ensure_color_object_caption(
        caption, color, object_name)
    caption_neg = ensure_color_object_caption(
        caption, neg_color, object_name)

    record = {
        'image_path': image_path,
        'file_name': img_info['file_name'],
        'image_id': ann['image_id'],
        'ann_id': ann['id'],
        'category_id': ann['category_id'],
        'object': object_name,
        'color': color,
        'neg_color': neg_color,
        'caption_original': caption,
        'caption_pos': caption_pos,
        'caption_neg': caption_neg,
        'bbox': ann.get('bbox'),
        'segmentation': ann['segmentation'],
        'width': img_info['width'],
        'height': img_info['height'],
        'area': ann.get('area'),
        'color_conf': color_est.confidence,
        'hue': color_est.hue,
        'saturation': color_est.saturation,
        'value': color_est.value,
    }
    return record, None


def build_records(args):
    inst = _load_json(args.instances_file)
    caps = _load_json(args.captions_file)
    exclude_categories = {
        c.strip().lower()
        for c in getattr(args, 'exclude_categories', ['person'])
        if c.strip()
    }

    images = {img['id']: img for img in inst['images']}
    categories = {cat['id']: cat['name'] for cat in inst['categories']}
    captions_by_img = defaultdict(list)
    for ann in caps['annotations']:
        captions_by_img[ann['image_id']].append(ann['caption'])

    written = 0
    skipped = defaultdict(int)
    os.makedirs(os.path.dirname(args.out_file) or '.', exist_ok=True)

    def process_ann(ann):
        return _build_one_record(
            ann, args, images, categories, captions_by_img, exclude_categories)

    def write_result(result, out):
        nonlocal written
        record, skip_key = result
        if skip_key is not None:
            skipped[skip_key] += 1
            return False
        out.write(json.dumps(record, ensure_ascii=False) + '\n')
        written += 1
        return bool(args.max_records and written >= args.max_records)

    num_workers = getattr(args, 'num_workers', 1)
    with open(args.out_file, 'w') as out:
        if num_workers <= 1:
            for ann in inst['annotations']:
                if write_result(process_ann(ann), out):
                    break
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for result in executor.map(process_ann, inst['annotations']):
                    if write_result(result, out):
                        break

    return written, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--instances_file', required=True)
    parser.add_argument('--captions_file', required=True)
    parser.add_argument('--out_file', required=True)
    parser.add_argument('--min_area_ratio', type=float, default=0.01)
    parser.add_argument('--min_color_conf', type=float, default=0.05)
    parser.add_argument('--max_records', type=int, default=0,
                        help='0 means no limit')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of preprocessing threads')
    parser.add_argument('--exclude_categories', nargs='*', default=['person'],
                        help='COCO category names to skip; default: person')
    args = parser.parse_args()

    written, skipped = build_records(args)
    print(f'wrote {written} records to {args.out_file}')
    if skipped:
        print('skipped:')
        for key, value in sorted(skipped.items()):
            print(f'  {key}: {value}')


if __name__ == '__main__':
    main()
