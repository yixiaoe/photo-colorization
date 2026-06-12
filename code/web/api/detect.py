"""
POST /api/detect
Input:  { "image_b64": "<base64 PNG/JPG>" }
Output: { "instances": [...], "preview_b64": "...", "is_color": bool }
"""
import base64
import io

import numpy as np
from flask import Blueprint, request, jsonify
from PIL import Image, ImageDraw, ImageFont

detect_bp = Blueprint('detect', __name__, url_prefix='')

# colour palette for bbox drawing (cycles through instances)
_BBOX_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
]


def _is_color(pil_img: Image.Image) -> bool:
    """Return True if the image appears to be a colour photo."""
    arr = np.array(pil_img.convert('RGB')).astype(float)
    # compute per-pixel channel std; colour images have significant variation
    channel_std = arr.std(axis=2).mean()
    return float(channel_std) > 5.0


def _draw_preview(pil_img: Image.Image, instances: list) -> Image.Image:
    """Draw coloured bboxes + labels on a copy of the image."""
    preview = pil_img.convert('RGB').copy()
    draw = ImageDraw.Draw(preview)
    W, H = preview.size
    lw = max(2, int(min(W, H) * 0.004))

    for inst in instances:
        color = _BBOX_COLORS[inst['id'] % len(_BBOX_COLORS)]
        x0, y0, x1, y1 = inst['bbox']
        draw.rectangle([x0, y0, x1, y1], outline=color, width=lw)
        label = f"inst:{inst['id']} {inst['label']} {inst['score']:.2f}"
        # draw label background
        tx, ty = x0, max(0, y0 - 18)
        draw.rectangle([tx, ty, tx + len(label) * 7, ty + 16],
                       fill=color)
        draw.text((tx + 2, ty), label, fill='white')
    return preview


def _pil_to_b64(pil_img: Image.Image, fmt='PNG') -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


@detect_bp.route('/api/detect', methods=['POST'])
def detect():
    data = request.get_json(force=True)
    if not data or 'image_b64' not in data:
        return jsonify({'error': 'image_b64 required'}), 400

    # decode image
    try:
        img_bytes = base64.b64decode(data['image_b64'])
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'image decode failed: {e}'}), 400

    is_color = _is_color(pil_img)

    # run Mask R-CNN
    import torch
    from util.maskrcnn_helper import predict_with_masks, class_name
    device = torch.device('cpu')
    try:
        detections = predict_with_masks(pil_img, device, box_num=8, score_thresh=0.5)
    except Exception as e:
        return jsonify({'error': f'Mask R-CNN failed: {e}'}), 500

    instances = []
    for d in detections:
        x0, y0, x1, y1 = [int(v) for v in d['box']]
        instances.append({
            'id':    int(detections.index(d)),
            'label': class_name(d['label']),
            'score': round(float(d['score']), 3),
            'bbox':  [x0, y0, x1, y1],
        })

    preview = _draw_preview(pil_img, instances)
    preview_b64 = _pil_to_b64(preview)

    return jsonify({
        'instances':   instances,
        'preview_b64': preview_b64,
        'is_color':    is_color,
    })
