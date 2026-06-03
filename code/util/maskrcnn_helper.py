"""
Mask R-CNN inference helper that also returns per-instance binary masks.

Phase 2 `data_process.colorization_dataset._predict_bbox` discards the
mask outputs. Phase 3 needs them for visualisation and (optionally) for
runtime instance segmentation. We reuse the Phase-2 model singleton to
avoid loading two copies.
"""
import numpy as np
import torch
import torchvision.transforms.functional as TF

# read-only reuse of the Phase-2 singleton; do NOT mutate that module
from data_process.colorization_dataset import _get_maskrcnn


# COCO 1..90 → human-readable name. Index 0 is reserved (__background__).
COCO_CLASSES = [
    '__background__',
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
    'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
]


def class_name(label_id: int) -> str:
    if 0 <= label_id < len(COCO_CLASSES):
        n = COCO_CLASSES[label_id]
        return n if n not in ('N/A', '__background__') else 'object'
    return 'object'


@torch.no_grad()
def predict_with_masks(pil_img, device, box_num=8, score_thresh=0.5,
                       mask_thresh=0.5):
    """
    Run Mask R-CNN and return per-instance results sorted by score desc.

    Returns:
      list of dicts, each:
        {
          'box':   (x0, y0, x1, y1)  int  – original image coords,
          'label': int                    – COCO class id (1..90),
          'score': float,
          'mask':  np.ndarray (H, W) uint8 {0,1}  – original image resolution,
        }
    """
    model = _get_maskrcnn(device)
    tensor = TF.to_tensor(pil_img).unsqueeze(0).to(device)
    pred = model(tensor)[0]

    boxes  = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy().astype(int)
    masks  = pred['masks'].cpu().numpy()        # (N, 1, H, W) float, soft

    keep = scores >= score_thresh
    boxes, scores, labels, masks = boxes[keep], scores[keep], labels[keep], masks[keep]

    if len(boxes) > box_num:
        top = np.argsort(scores)[-box_num:][::-1]   # descending
        boxes, scores, labels, masks = boxes[top], scores[top], labels[top], masks[top]
    else:
        # already sorted desc by torchvision, but ensure it
        order = np.argsort(scores)[::-1]
        boxes, scores, labels, masks = boxes[order], scores[order], labels[order], masks[order]

    out = []
    for box, score, label, mask in zip(boxes, scores, labels, masks):
        x0, y0, x1, y1 = box.astype(int)
        m = (mask[0] >= mask_thresh).astype(np.uint8)
        out.append({
            'box':   (int(x0), int(y0), int(x1), int(y1)),
            'label': int(label),
            'score': float(score),
            'mask':  m,
        })
    return out
