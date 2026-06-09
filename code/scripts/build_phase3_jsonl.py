"""
Build the Phase 3 (image, instance) JSONL corpus from COCO annotations.

For every COCO instance that passes the filtering rules below, write one
JSONL line containing:
    image_id, ann_id, image_file, class_id, color_pos, color_neg, bbox, area_ratio

Mask information is reconstructed on the fly in the dataset (we do NOT
store mask RLE here; pycocotools can rasterise from the original ann).

Filtering (see docs/phase3_plan.md §6.1):
  iscrowd == 0
  segmentation is polygon
  area / image_area >= 0.01    (>= 0.005 for animals/vehicles)
  HSV main-colour confidence >= 0.05
  person:  confidence >= 0.30   (skin/clothing colour unstable)

HSV colour word:
  V < 0.15                          -> black
  S < 0.20 and V >= 0.82            -> white
  S < 0.20                          -> gray
  15 <= H < 45  and V < 0.55        -> brown
  H < 15  or H >= 345               -> red
  15 <= H < 35                      -> orange
  35 <= H < 70                      -> yellow
  70 <= H < 170                     -> green
  170 <= H < 200                    -> cyan
  200 <= H < 260                    -> blue
  260 <= H < 310                    -> purple
  310 <= H < 345                    -> pink

Hue is averaged circularly (cos/sin weighted by S*V).

Negative-colour sampling:
  50% complementary  (red <-> cyan, etc.)
  30% neighbouring   (red <-> orange / pink)
  20% random other

Usage (run from code/):
  python scripts/build_phase3_jsonl.py \
      --instances_file datasets/coco/annotations/instances_train2017.json \
      --img_dir        datasets/coco/train2017 \
      --out_file       datasets/phase3/color_object_train.jsonl
"""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image

try:
    from pycocotools.coco import COCO
except ImportError:
    raise SystemExit('pycocotools not installed. pip install pycocotools')


# ── colour word logic ─────────────────────────────────────────────────────────

ALL_COLORS = [
    'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink',
    'brown', 'black', 'white', 'gray',
]

COMPLEMENT = {
    'red':    'cyan',   'cyan':   'red',
    'orange': 'blue',   'blue':   'orange',
    'yellow': 'purple', 'purple': 'yellow',
    'green':  'pink',   'pink':   'green',
    'brown':  'cyan',
    'black':  'white',  'white':  'black',
    'gray':   'red',
}

NEIGHBOUR = {
    'red':    ['orange', 'pink'],
    'orange': ['red', 'yellow', 'brown'],
    'yellow': ['orange', 'green', 'brown'],
    'green':  ['yellow', 'cyan'],
    'cyan':   ['green', 'blue'],
    'blue':   ['cyan', 'purple'],
    'purple': ['blue', 'pink'],
    'pink':   ['purple', 'red'],
    'brown':  ['orange', 'red'],
    'black':  ['gray'],
    'white':  ['gray'],
    'gray':   ['white', 'black'],
}

# COCO category id (1..90) -> human-readable name (subset). We always
# fall back to the COCO supercategory if not in this table.
CLASS_NAMES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
    50: 'spoon', 51: 'bowl',
    52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
    62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
}


def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0,1] of shape (...,3) -> hsv of shape (...,3), H in degrees [0,360)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask = delta > 0
    rc = (cmax == r) & mask
    gc = (cmax == g) & mask
    bc = (cmax == b) & mask
    h[rc] = ((g[rc] - b[rc]) / delta[rc]) % 6
    h[gc] = (b[gc] - r[gc]) / delta[gc] + 2
    h[bc] = (r[bc] - g[bc]) / delta[bc] + 4
    h = h * 60.0 % 360.0

    s = np.where(cmax > 0, delta / np.maximum(cmax, 1e-8), 0)
    v = cmax
    return np.stack([h, s, v], axis=-1)


def infer_color(rgb_patch: np.ndarray, mask_patch: np.ndarray, min_conf=0.05):
    """
    rgb_patch:  (H, W, 3) uint8 or float in [0, 1]
    mask_patch: (H, W) bool / uint8 {0, 1}
    returns: (color_word, confidence)  or  (None, 0.0) if below threshold
    """
    if rgb_patch.dtype == np.uint8:
        rgb_patch = rgb_patch.astype(np.float32) / 255.0
    mask = mask_patch.astype(bool)
    if mask.sum() == 0:
        return None, 0.0

    hsv = rgb_to_hsv_np(rgb_patch)
    H, S, V = hsv[..., 0][mask], hsv[..., 1][mask], hsv[..., 2][mask]

    # neutral cases use overall stats inside the mask
    if V.mean() < 0.15:
        return 'black', max(0.0, 0.15 - V.mean()) * 4 + 0.2
    if S.mean() < 0.20 and V.mean() >= 0.82:
        return 'white', (V.mean() - 0.82) * 4 + 0.2
    if S.mean() < 0.20:
        return 'gray', (0.20 - S.mean()) * 4 + 0.2

    # weighted circular mean of hue, weights = S * V
    w = (S * V).astype(np.float32)
    w_sum = float(w.sum())
    if w_sum < 1e-6:
        return None, 0.0
    rad = np.deg2rad(H)
    cx = float((w * np.cos(rad)).sum() / w_sum)
    sy = float((w * np.sin(rad)).sum() / w_sum)
    h_mean = float(np.rad2deg(np.arctan2(sy, cx)) % 360.0)

    # confidence = mean weight in mask + magnitude of resultant vector
    r_mag = float(np.sqrt(cx * cx + sy * sy))      # [0, 1]
    conf = 0.5 * w.mean() + 0.5 * r_mag

    v_mean = float(V.mean())

    if 15 <= h_mean < 45 and v_mean < 0.55:
        color = 'brown'
    elif h_mean < 15 or h_mean >= 345:
        color = 'red'
    elif 15 <= h_mean < 35:
        color = 'orange'
    elif 35 <= h_mean < 70:
        color = 'yellow'
    elif 70 <= h_mean < 170:
        color = 'green'
    elif 170 <= h_mean < 200:
        color = 'cyan'
    elif 200 <= h_mean < 260:
        color = 'blue'
    elif 260 <= h_mean < 310:
        color = 'purple'
    elif 310 <= h_mean < 345:
        color = 'pink'
    else:
        color = 'red'   # unreachable, just in case

    if conf < min_conf:
        return None, conf
    return color, conf


def sample_negative_color(pos_color: str, rng: random.Random) -> str:
    """50% complement, 30% neighbour, 20% random other."""
    p = rng.random()
    if p < 0.50:
        return COMPLEMENT.get(pos_color, 'gray')
    if p < 0.80:
        opts = NEIGHBOUR.get(pos_color, ['gray'])
        return rng.choice(opts)
    pool = [c for c in ALL_COLORS if c != pos_color]
    return rng.choice(pool)


# ── per-annotation processing ─────────────────────────────────────────────────

def process_ann(coco: COCO, ann: dict, img_meta: dict, img_dir: str,
                rng: random.Random, min_area_ratio: float,
                person_min_conf: float = 0.15,
                generic_min_conf: float = 0.04):
    """Return (record_or_None, reason). reason is empty string on success."""
    if ann.get('iscrowd', 0):
        return None, 'iscrowd'
    seg = ann.get('segmentation')
    if not seg:
        return None, 'no_segmentation'

    W_meta = img_meta['width']
    H_meta = img_meta['height']
    img_area = W_meta * H_meta
    if img_area <= 0:
        return None, 'bad_image_size'
    area_ratio = float(ann['area']) / float(img_area)
    if area_ratio < min_area_ratio:
        return None, 'area_too_small'

    cat_id = int(ann['category_id'])
    class_name = CLASS_NAMES.get(cat_id, 'object')
    min_conf = person_min_conf if class_name == 'person' else generic_min_conf

    fname = img_meta['file_name']
    fpath = os.path.join(img_dir, fname)
    if not os.path.isfile(fpath):
        return None, 'image_missing'
    try:
        pil = Image.open(fpath).convert('RGB')
    except Exception:
        return None, 'image_open_fail'

    # use actual PIL size for both image and mask space
    W, H = pil.size

    # rasterise mask via pycocotools — pycocotools handles both polygon
    # lists and RLE dicts; we no longer pre-filter on the seg type.
    try:
        mask_full = coco.annToMask(ann)        # (H_meta, W_meta) uint8
    except Exception:
        return None, 'annToMask_fail'

    # if PIL image and metadata disagree, resize the mask to actual image space
    if mask_full.shape[0] != H or mask_full.shape[1] != W:
        m_pil = Image.fromarray((mask_full * 255).astype(np.uint8)).resize(
            (W, H), resample=Image.NEAREST)
        mask_full = (np.asarray(m_pil) > 127).astype(np.uint8)

    rgb_full = np.asarray(pil)                 # (H, W, 3) uint8
    color, conf = infer_color(rgb_full, mask_full, min_conf=min_conf)
    if color is None:
        return None, 'low_color_conf'

    color_neg = sample_negative_color(color, rng)

    x, y, w, h = ann['bbox']
    rec = {
        'image_id':   int(ann['image_id']),
        'ann_id':     int(ann['id']),
        'image_file': fname,
        'class_id':   cat_id,
        'class_name': class_name,
        'color_pos':  color,
        'color_neg':  color_neg,
        'color_conf': round(float(conf), 4),
        'bbox':       [float(x), float(y), float(w), float(h)],
        'area_ratio': round(area_ratio, 5),
    }
    return rec, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances_file', required=True,
                    help='COCO instances JSON (e.g. instances_train2017.json)')
    ap.add_argument('--img_dir',        required=True)
    ap.add_argument('--out_file',       required=True)
    ap.add_argument('--min_area_ratio', type=float, default=0.005,
                    help='instance_area / image_area lower bound')
    ap.add_argument('--person_min_conf',  type=float, default=0.15)
    ap.add_argument('--generic_min_conf', type=float, default=0.04)
    ap.add_argument('--max_records',    type=int,   default=0,
                    help='cap on output records (0 = no cap, for debug)')
    ap.add_argument('--seed',           type=int,   default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)) or '.', exist_ok=True)

    print(f'[load] COCO instances: {args.instances_file}')
    coco = COCO(args.instances_file)
    img_meta = {i['id']: i for i in coco.dataset['images']}
    anns = coco.dataset['annotations']
    print(f'[stat] images={len(img_meta)}  annotations={len(anns)}')

    kept = 0
    skipped = 0
    color_hist = {c: 0 for c in ALL_COLORS}
    reason_hist: dict = {}

    with open(args.out_file, 'w') as f:
        for n, ann in enumerate(anns):
            im = img_meta.get(ann['image_id'])
            if im is None:
                skipped += 1
                reason_hist['image_meta_missing'] = reason_hist.get('image_meta_missing', 0) + 1
                continue
            rec, reason = process_ann(
                coco, ann, im, args.img_dir, rng, args.min_area_ratio,
                person_min_conf=args.person_min_conf,
                generic_min_conf=args.generic_min_conf,
            )
            if rec is None:
                skipped += 1
                reason_hist[reason] = reason_hist.get(reason, 0) + 1
            else:
                f.write(json.dumps(rec) + '\n')
                kept += 1
                color_hist[rec['color_pos']] += 1

            if (n + 1) % 5000 == 0:
                top_reason = max(reason_hist.items(), key=lambda kv: kv[1]) \
                    if reason_hist else ('-', 0)
                print(f'  [{n+1:7d}/{len(anns)}] kept={kept}  skipped={skipped}  '
                      f'top_skip="{top_reason[0]}"({top_reason[1]})')

            if args.max_records and kept >= args.max_records:
                print(f'  [cap] reached --max_records={args.max_records}, stopping')
                break

    print(f'\nDone.  kept={kept}  skipped={skipped}  -> {args.out_file}')
    print('skip reasons:')
    for r, c in sorted(reason_hist.items(), key=lambda kv: -kv[1]):
        print(f'  {r:24s}  {c:7d}')
    print('color distribution:')
    for c, n in sorted(color_hist.items(), key=lambda kv: -kv[1]):
        if n:
            print(f'  {c:8s}  {n:7d}')


if __name__ == '__main__':
    main()
