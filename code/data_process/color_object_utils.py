"""Utilities for Phase 3 color-object instance supervision."""
import math
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


COLOR_WORDS = [
    'black', 'white', 'gray', 'red', 'orange', 'yellow',
    'green', 'cyan', 'blue', 'purple', 'pink', 'brown',
]

COLOR_HUE_CENTERS = {
    'red': 0.0,
    'orange': 25.0,
    'yellow': 52.5,
    'green': 120.0,
    'cyan': 185.0,
    'blue': 232.0,
    'purple': 285.0,
    'pink': 330.0,
    'brown': 25.0,
}

NEUTRAL_NEGATIVES = {
    'black': 'white',
    'white': 'black',
    'gray': 'red',
    'brown': 'blue',
}

COCO80_SYNONYMS = {
    'person': ['person', 'people', 'man', 'woman', 'child'],
    'bicycle': ['bicycle', 'bike'],
    'car': ['car', 'vehicle', 'auto'],
    'motorcycle': ['motorcycle', 'motorbike', 'bike'],
    'airplane': ['airplane', 'plane', 'aircraft'],
    'bus': ['bus'],
    'train': ['train'],
    'truck': ['truck'],
    'boat': ['boat', 'ship'],
    'traffic light': ['traffic light'],
    'fire hydrant': ['fire hydrant', 'hydrant'],
    'stop sign': ['stop sign'],
    'parking meter': ['parking meter'],
    'bench': ['bench'],
    'bird': ['bird'],
    'cat': ['cat'],
    'dog': ['dog'],
    'horse': ['horse'],
    'sheep': ['sheep'],
    'cow': ['cow'],
    'elephant': ['elephant'],
    'bear': ['bear'],
    'zebra': ['zebra'],
    'giraffe': ['giraffe'],
    'backpack': ['backpack', 'bag'],
    'umbrella': ['umbrella'],
    'handbag': ['handbag', 'bag', 'purse'],
    'tie': ['tie'],
    'suitcase': ['suitcase', 'luggage'],
    'frisbee': ['frisbee'],
    'skis': ['skis', 'ski'],
    'snowboard': ['snowboard'],
    'sports ball': ['sports ball', 'ball'],
    'kite': ['kite'],
    'baseball bat': ['baseball bat', 'bat'],
    'baseball glove': ['baseball glove', 'glove'],
    'skateboard': ['skateboard'],
    'surfboard': ['surfboard', 'surf board'],
    'tennis racket': ['tennis racket', 'racket'],
    'bottle': ['bottle'],
    'wine glass': ['wine glass', 'glass'],
    'cup': ['cup'],
    'fork': ['fork'],
    'knife': ['knife'],
    'spoon': ['spoon'],
    'bowl': ['bowl'],
    'banana': ['banana'],
    'apple': ['apple'],
    'sandwich': ['sandwich'],
    'orange': ['orange'],
    'broccoli': ['broccoli'],
    'carrot': ['carrot'],
    'hot dog': ['hot dog'],
    'pizza': ['pizza'],
    'donut': ['donut', 'doughnut'],
    'cake': ['cake'],
    'chair': ['chair'],
    'couch': ['couch', 'sofa'],
    'potted plant': ['potted plant', 'plant'],
    'bed': ['bed'],
    'dining table': ['dining table', 'table'],
    'toilet': ['toilet'],
    'tv': ['tv', 'television'],
    'laptop': ['laptop'],
    'mouse': ['mouse'],
    'remote': ['remote'],
    'keyboard': ['keyboard'],
    'cell phone': ['cell phone', 'phone'],
    'microwave': ['microwave'],
    'oven': ['oven'],
    'toaster': ['toaster'],
    'sink': ['sink'],
    'refrigerator': ['refrigerator', 'fridge'],
    'book': ['book'],
    'clock': ['clock'],
    'vase': ['vase'],
    'scissors': ['scissors'],
    'teddy bear': ['teddy bear'],
    'hair drier': ['hair drier', 'hair dryer'],
    'toothbrush': ['toothbrush'],
}


@dataclass
class ColorEstimate:
    color: str
    confidence: float
    hue: float | None
    saturation: float
    value: float


def _resampling(name):
    if hasattr(Image, 'Resampling'):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


def resize_binary_mask(mask, size):
    """Resize a binary mask with nearest-neighbor interpolation."""
    arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if isinstance(size, int):
        size = (size, size)
    pil = Image.fromarray(arr, mode='L')
    out = pil.resize(size, resample=_resampling('NEAREST'))
    return (np.asarray(out) > 0).astype(np.uint8)


def segmentation_to_mask(segmentation, width, height):
    """Convert COCO polygon segmentation to a binary mask."""
    mask = Image.new('L', (int(width), int(height)), 0)
    draw = ImageDraw.Draw(mask)
    if isinstance(segmentation, list):
        for poly in segmentation:
            if len(poly) < 6:
                continue
            points = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(points, outline=1, fill=1)
    else:
        raise ValueError('Only polygon COCO segmentations are supported')
    return np.asarray(mask, dtype=np.uint8)


def rgb_to_hsv_np(rgb):
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    nonzero = delta > 1e-8
    rmax = nonzero & (cmax == r)
    gmax = nonzero & (cmax == g)
    bmax = nonzero & (cmax == b)
    hue[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6
    hue[gmax] = (b[gmax] - r[gmax]) / delta[gmax] + 2
    hue[bmax] = (r[bmax] - g[bmax]) / delta[bmax] + 4
    hue *= 60.0

    sat = np.zeros_like(cmax)
    sat[cmax > 1e-8] = delta[cmax > 1e-8] / cmax[cmax > 1e-8]
    val = cmax
    return hue, sat, val


def _hue_to_color(hue, value):
    if 15 <= hue < 45 and value < 0.55:
        return 'brown'
    if hue < 15 or hue >= 345:
        return 'red'
    if hue < 35:
        return 'orange'
    if hue < 70:
        return 'yellow'
    if hue < 170:
        return 'green'
    if hue < 200:
        return 'cyan'
    if hue < 260:
        return 'blue'
    if hue < 310:
        return 'purple'
    return 'pink'


def dominant_hsv_color(rgb, mask, black_v=0.15, gray_s=0.20,
                       white_v=0.82, min_color_pixels=8):
    """Estimate an object's main color from pixels inside its mask."""
    rgb = np.asarray(rgb)
    mask = np.asarray(mask).astype(bool)
    pixels = rgb[mask]
    if pixels.size == 0:
        return ColorEstimate('gray', 0.0, None, 0.0, 0.0)

    hue, sat, val = rgb_to_hsv_np(pixels)
    med_s = float(np.median(sat))
    med_v = float(np.median(val))

    if med_v < black_v:
        return ColorEstimate('black', 1.0, None, med_s, med_v)
    if med_s < gray_s:
        color = 'white' if med_v >= white_v else 'gray'
        return ColorEstimate(color, 1.0, None, med_s, med_v)

    valid = (sat >= gray_s) & (val >= black_v)
    if valid.sum() < min_color_pixels:
        return ColorEstimate('gray', 0.0, None, med_s, med_v)

    weights = sat[valid] * val[valid]
    radians = np.deg2rad(hue[valid])
    x = float(np.sum(np.cos(radians) * weights))
    y = float(np.sum(np.sin(radians) * weights))
    mean_hue = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    color = _hue_to_color(mean_hue, med_v)
    confidence = float(valid.sum()) / float(mask.sum())
    return ColorEstimate(color, confidence, mean_hue, med_s, med_v)


def _term_variants(object_name, synonyms=None):
    terms = list((synonyms or COCO80_SYNONYMS).get(object_name, [object_name]))
    out = []
    for term in terms:
        out.append(term)
        if not term.endswith('s'):
            out.append(term + 's')
    return sorted(set(out), key=len, reverse=True)


def _term_regex(term):
    words = [re.escape(w) for w in term.split()]
    return r'\b' + r'\s+'.join(words) + r'\b'


def choose_caption_for_object(captions, object_name, synonyms=None):
    """Return the first caption that mentions object_name or one synonym."""
    terms = _term_variants(object_name, synonyms)
    for caption in captions:
        for term in terms:
            if re.search(_term_regex(term), caption, flags=re.IGNORECASE):
                return caption
    return None


def ensure_color_object_caption(caption, color, object_name, synonyms=None):
    """Ensure a caption contains '<color> <object>' for the target object."""
    if not caption:
        return f'a {color} {object_name}'

    color_group = '|'.join(re.escape(c) for c in COLOR_WORDS)
    terms = _term_variants(object_name, synonyms)
    for term in terms:
        term_pattern = _term_regex(term)
        term_body = r'\s+'.join(re.escape(w) for w in term.split())
        colored = re.compile(
            r'\b(?:' + color_group + r')\s+(' + term_body + r')\b',
            flags=re.IGNORECASE)
        if colored.search(caption):
            return colored.sub(lambda m: f'{color} {m.group(1)}',
                               caption, count=1)

        match = re.search(term_pattern, caption, flags=re.IGNORECASE)
        if match:
            return caption[:match.start()] + color + ' ' + caption[match.start():]

    return f'{caption.rstrip()} {color} {object_name}'


def _hue_distance(a, b):
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def negative_color_for(color):
    """Pick a complementary or high-contrast negative color word."""
    if color in NEUTRAL_NEGATIVES:
        return NEUTRAL_NEGATIVES[color]
    hue = COLOR_HUE_CENTERS.get(color)
    if hue is None:
        return 'red'
    comp = (hue + 180.0) % 360.0
    candidates = {
        k: v for k, v in COLOR_HUE_CENTERS.items()
        if k != color and k != 'brown'
    }
    return min(candidates, key=lambda k: _hue_distance(candidates[k], comp))
