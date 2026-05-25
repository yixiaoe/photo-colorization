"""
Offline Mask R-CNN bbox precomputation.

Reads images from datasets/coco/train2017/, saves detections to
datasets/bbox_cache_train2017.json (or any path via --output).

JSON format:
  { "<basename>": [[x0,y0,x1,y1,label], ...], ... }
  Empty list means no detection above threshold.

Supports resume: already-present keys are skipped on re-run.

Usage (run from code/):
  python scripts/precompute_bbox.py \
      --img_dir  datasets/coco/train2017 \
      --output   datasets/bbox_cache_train2017.json \
      --ckpt     checkpoints/mask_rcnn/maskrcnn_resnet50_fpn.pth
"""
import argparse
import json
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image


def collect_images(root):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = []
    for dirpath, _, fnames in os.walk(root):
        for f in sorted(fnames):
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def load_model(ckpt, device):
    if ckpt and os.path.isfile(ckpt):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        print(f'[MaskRCNN] loaded local ckpt: {ckpt}')
    else:
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
        print('[MaskRCNN] downloaded from torchvision hub')
    return model.to(device).eval()


@torch.no_grad()
def detect(img, model, device, box_num, score_thresh):
    t = TF.to_tensor(img).unsqueeze(0).to(device)
    pred   = model(t)[0]
    boxes  = pred['boxes'].cpu().numpy().astype(int)
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy().astype(int)

    keep = scores >= score_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if len(boxes) > box_num:
        top = np.argsort(scores)[-box_num:]
        boxes, labels = boxes[top], labels[top]

    return [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(l)]
            for b, l in zip(boxes, labels)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir',      default='datasets/coco/train2017',
                    help='COCO train2017 image directory')
    ap.add_argument('--output',       default='datasets/bbox_cache_train2017.json',
                    help='output JSON cache path')
    ap.add_argument('--ckpt',         default='checkpoints/mask_rcnn/maskrcnn_resnet50_fpn.pth',
                    help='local Mask R-CNN checkpoint')
    ap.add_argument('--score_thresh', type=float, default=0.5)
    ap.add_argument('--box_num',      type=int,   default=8)
    ap.add_argument('--gpu',          type=int,   default=0)
    ap.add_argument('--save_every',   type=int,   default=1000,
                    help='flush JSON to disk every N newly processed images')
    args = ap.parse_args()

    device = (torch.device(f'cuda:{args.gpu}')
              if args.gpu >= 0 and torch.cuda.is_available()
              else torch.device('cpu'))
    print(f'device : {device}')
    print(f'img_dir: {args.img_dir}')
    print(f'output : {args.output}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

    # load existing cache for resumption
    cache = {}
    if os.path.isfile(args.output):
        with open(args.output) as f:
            cache = json.load(f)
        print(f'resume: {len(cache)} images already cached')

    model  = load_model(args.ckpt, device)
    paths  = collect_images(args.img_dir)
    total  = len(paths)
    print(f'total images found: {total}')

    new_cnt = 0
    for i, path in enumerate(paths):
        key = os.path.basename(path)
        if key in cache:
            continue

        try:
            img  = Image.open(path).convert('RGB')
            dets = detect(img, model, device, args.box_num, args.score_thresh)
        except Exception as e:
            print(f'[warn] {path}: {e}')
            dets = []

        cache[key] = dets
        new_cnt += 1

        if new_cnt % 200 == 0:
            print(f'  [{i+1:6d}/{total}] {(i+1)/total*100:5.1f}%  '
                  f'new={new_cnt}  boxes={len(dets)}  {key}')

        if new_cnt % args.save_every == 0:
            with open(args.output, 'w') as f:
                json.dump(cache, f)
            print(f'  [checkpoint] saved {len(cache)} entries -> {args.output}')

    # final flush
    with open(args.output, 'w') as f:
        json.dump(cache, f)

    n_empty = sum(1 for v in cache.values() if not v)
    print(f'\nDone.')
    print(f'  total cached : {len(cache)}')
    print(f'  with boxes   : {len(cache) - n_empty}')
    print(f'  empty (0 det): {n_empty}')
    print(f'  output       : {args.output}')


if __name__ == '__main__':
    main()
