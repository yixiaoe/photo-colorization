"""
Unified colorization dataset.

Supports:
  - ImageNet-Mini  (folder of JPEG images, any structure)
  - CIFAR-10       (torchvision download)
  - COCO2017       (folder + JSON annotation, for stage=instance/fusion)

Stage routing:
  'full'     → full image, returns rgb_img + gray_img
  'instance' → GT-bbox instance crop with class_label (COCO) or random crop (others)
  'fusion'   → full image + bbox info (offline cache or online Mask R-CNN fallback)

FusionDataset bbox loading priority:
  1. --bbox_cache JSON (recommended): precomputed by scripts/precompute_bbox.py,
     loaded once at init, no GPU needed in workers → supports --nThreads > 0
  2. Online Mask R-CNN (fallback when --bbox_cache is empty):
     runs in __getitem__, requires --nThreads 0 to avoid CUDA fork issues
"""
import os
import json
import random
from os.path import join, isfile

import numpy as np
from PIL import Image
from skimage import color

import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F


# ── image helpers ──────────────────────────────────────────────────────────────

def _load_rgb(path):
    img = Image.open(path).convert('RGB')
    return img


def _to_gray_rgb(pil_img):
    """Return (rgb PIL, gray-as-RGB PIL)."""
    arr = np.asarray(pil_img)
    gray = np.round(color.rgb2gray(arr) * 255).astype(np.uint8)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    return pil_img, Image.fromarray(gray_rgb)


# ── bbox via torchvision Mask R-CNN (lazy-loaded singleton) ──────────────────

_maskrcnn = None

_MASKRCNN_LOCAL = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'checkpoints', 'mask_rcnn', 'maskrcnn_resnet50_fpn.pth'
)

def _get_maskrcnn(device):
    global _maskrcnn
    if _maskrcnn is None:
        if os.path.isfile(_MASKRCNN_LOCAL):
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
            state = torch.load(_MASKRCNN_LOCAL, map_location='cpu')
            model.load_state_dict(state)
            print(f'[MaskRCNN] Loaded from {_MASKRCNN_LOCAL}')
        else:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            )
            print('[MaskRCNN] Downloaded from torchvision hub')
        _maskrcnn = model.to(device).eval()
    return _maskrcnn


@torch.no_grad()
def _predict_bbox(pil_img, device, box_num=8, score_thresh=0.5):
    """Return list of (x0, y0, x1, y1, label) tuples from Mask R-CNN."""
    model = _get_maskrcnn(device)
    tensor = F.to_tensor(pil_img).unsqueeze(0).to(device)
    preds = model(tensor)[0]
    boxes  = preds['boxes'].cpu().numpy().astype(int)
    scores = preds['scores'].cpu().numpy()
    labels = preds['labels'].cpu().numpy().astype(int)
    mask   = scores >= score_thresh
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
    if len(boxes) > box_num:
        idx = np.argsort(scores)[-box_num:]
        boxes, labels = boxes[idx], labels[idx]
    return [(tuple(b), int(l)) for b, l in zip(boxes, labels)]


# ── box geometry helper (mirrors reference image_util.get_box_info) ───────────

def get_box_info(pred_bbox, original_size, final_size):
    """
    Args:
        pred_bbox: (x0, y0, x1, y1) in original pixel coords
        original_size: (W, H) of the source image (PIL convention)
        final_size: target spatial resolution (square)
    Returns:
        [L_pad, R_pad, T_pad, B_pad, rw, rh]
          where rw = box width at final_size, rh = box height at final_size
    """
    W, H = original_size
    x0, y0, x1, y1 = pred_bbox
    sx = int(x0 / W * final_size)
    sy = int(y0 / H * final_size)
    ex = int(x1 / W * final_size)
    ey = int(y1 / H * final_size)
    rw = max(ex - sx, 1)
    rh = max(ey - sy, 1)
    if ex - sx < 1:
        if final_size - ex > 1: ex += 1
        else: sx -= 1
        rw = 1
    if ey - sy < 1:
        if final_size - ey > 1: ey += 1
        else: sy -= 1
        rh = 1
    return [sx, final_size - ex, sy, final_size - ey, rw, rh]


# ── ImageNet-Mini / generic file list ────────────────────────────────────────

def _collect_images(root):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = []
    for dirpath, _, fnames in os.walk(root):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(join(dirpath, f))
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return sorted(paths)


# ── COCO annotation loader ────────────────────────────────────────────────────

def _load_coco_instances(ann_file, img_dir, min_area=1024, min_side=32):
    """
    Parse COCO JSON and return a list of
      (img_path, bbox_xywh, category_id)
    tuples, filtered by min area and min side length.
    """
    with open(ann_file) as f:
        coco = json.load(f)

    id2file = {img['id']: img['file_name'] for img in coco['images']}
    records = []
    for ann in coco['annotations']:
        if ann.get('iscrowd', 0):
            continue
        x, y, w, h = ann['bbox']
        if w * h < min_area or w < min_side or h < min_side:
            continue
        fpath = join(img_dir, id2file[ann['image_id']])
        if not os.path.isfile(fpath):
            continue
        records.append((fpath, (x, y, w, h), int(ann['category_id'])))
    return records


# ── Dataset classes ───────────────────────────────────────────────────────────

class ColorizationDataset(Data.Dataset):
    """
    Full-image dataset for Phase 1 (cnn_color) and Phase 2 stage='full'.
    Returns {'rgb_img': Tensor(3,H,W), 'gray_img': Tensor(3,H,W)}.
    """
    def __init__(self, opt, split='train'):
        self.opt = opt
        sz = opt.fineSize

        if split == 'train':
            self.spatial_tfm = T.Compose([
                T.RandomResizedCrop(sz, scale=(0.6, 1.0), ratio=(3/4, 4/3),
                                    interpolation=Image.BILINEAR),
                T.RandomHorizontalFlip(),
            ])
        else:
            self.spatial_tfm = T.Resize((sz, sz), interpolation=Image.BILINEAR)

        self.tensor_tfm = T.ToTensor()

        if opt.dataset == 'cifar10':
            self.ds = torchvision.datasets.CIFAR10(
                root=opt.data_dir, train=(split == 'train'),
                download=True, transform=None)
            self._cifar = True
        else:
            self._cifar = False
            self.paths = _collect_images(opt.data_dir)
            if opt.max_dataset_size < float('inf'):
                self.paths = self.paths[:int(opt.max_dataset_size)]

    def __len__(self):
        return len(self.ds) if self._cifar else len(self.paths)

    def __getitem__(self, idx):
        if self._cifar:
            pil_img, _ = self.ds[idx]
        else:
            pil_img = _load_rgb(self.paths[idx])

        pil_img = self.spatial_tfm(pil_img)
        rgb_img, gray_img = _to_gray_rgb(pil_img)
        return {
            'rgb_img':  self.tensor_tfm(rgb_img),
            'gray_img': self.tensor_tfm(gray_img),
        }


class InstanceDataset(Data.Dataset):
    """
    Phase 2 stage='instance'.

    When dataset='coco' (i.e. data_dir points to COCO images and
    opt.ann_file is given): loads GT bboxes + GT category labels from COCO.

    Fallback (imagenet_mini / cifar10): random crop without class_label
    (uses label=0 as placeholder).
    """

    def __init__(self, opt, split='train'):
        self.opt = opt
        sz = opt.fineSize
        self.tensor_tfm = T.ToTensor()
        self.resize_tfm = T.Resize((sz, sz), interpolation=Image.BILINEAR)

        # --- COCO path ---
        ann_file = getattr(opt, 'ann_file', '')
        self._coco_mode = False
        if ann_file and os.path.isfile(ann_file):
            self._coco_mode = True
            self.records = _load_coco_instances(ann_file, opt.data_dir)
            if opt.max_dataset_size < float('inf'):
                self.records = self.records[:int(opt.max_dataset_size)]
            print(f'[InstanceDataset] COCO mode: {len(self.records)} instances')
            return

        # --- fallback: random-crop ---
        if split == 'train':
            self.spatial_tfm = T.Compose([
                T.RandomResizedCrop(sz, scale=(0.3, 1.0), ratio=(3/4, 4/3),
                                    interpolation=Image.BILINEAR),
                T.RandomHorizontalFlip(),
            ])
        else:
            self.spatial_tfm = T.Resize((sz, sz), interpolation=Image.BILINEAR)

        if opt.dataset == 'cifar10':
            self.ds = torchvision.datasets.CIFAR10(
                root=opt.data_dir, train=(split == 'train'),
                download=True, transform=None)
            self._cifar = True
        else:
            self._cifar = False
            self.paths = _collect_images(opt.data_dir)
            if opt.max_dataset_size < float('inf'):
                self.paths = self.paths[:int(opt.max_dataset_size)]

    def __len__(self):
        if self._coco_mode:
            return len(self.records)
        return len(self.ds) if getattr(self, '_cifar', False) else len(self.paths)

    def __getitem__(self, idx):
        if self._coco_mode:
            fpath, (x, y, w, h), cat_id = self.records[idx]
            pil_img = _load_rgb(fpath)
            # crop instance region (x,y,w,h → PIL crop = (x,y,x+w,y+h))
            crop = pil_img.crop((x, y, x + w, y + h))
            # random horizontal flip
            if random.random() < 0.5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            crop = self.resize_tfm(crop)
            rgb_img, _ = _to_gray_rgb(crop)
            return {
                'rgb_img':    self.tensor_tfm(rgb_img),
                'gray_img':   self.tensor_tfm(rgb_img),   # kept for API compat
                'class_label': torch.tensor(cat_id, dtype=torch.long),
            }

        # fallback: random crop, label=0
        if getattr(self, '_cifar', False):
            pil_img, _ = self.ds[idx]
        else:
            pil_img = _load_rgb(self.paths[idx])
        pil_img = self.spatial_tfm(pil_img)
        rgb_img, gray_img = _to_gray_rgb(pil_img)
        return {
            'rgb_img':    self.tensor_tfm(rgb_img),
            'gray_img':   self.tensor_tfm(gray_img),
            'class_label': torch.tensor(0, dtype=torch.long),
        }


class FusionDataset(Data.Dataset):
    """
    Phase 2 stage='fusion'.

    Bbox source (in priority order):
      1. Offline JSON cache (--bbox_cache): loaded at init, worker-safe.
      2. Online Mask R-CNN fallback: called per item, requires nThreads=0.
    """
    def __init__(self, opt, split='train', box_num=8):
        self.opt = opt
        self.box_num = box_num
        sz = opt.fineSize
        self.final_size = sz
        self.tfm = T.Compose([
            T.Resize((sz, sz), interpolation=Image.BILINEAR),
            T.ToTensor(),
        ])

        # --- offline bbox cache (preferred) ---
        bbox_cache_path = getattr(opt, 'bbox_cache', '')
        self._bbox_cache = {}
        if bbox_cache_path and os.path.isfile(bbox_cache_path):
            with open(bbox_cache_path) as f:
                self._bbox_cache = json.load(f)
            print(f'[FusionDataset] bbox cache loaded: '
                  f'{len(self._bbox_cache)} entries from {bbox_cache_path}')
        else:
            # online fallback: need CUDA in workers → force nThreads=0
            self.device = torch.device(
                f'cuda:{opt.gpu_ids[0]}' if opt.gpu_ids else 'cpu')
            print('[FusionDataset] no bbox cache — using online Mask R-CNN '
                  '(ensure --nThreads 0)')

        if opt.dataset == 'cifar10':
            self.ds = torchvision.datasets.CIFAR10(
                root=opt.data_dir, train=(split == 'train'),
                download=True, transform=None)
            self._cifar = True
        else:
            self._cifar = False
            self.paths = _collect_images(opt.data_dir)
            if opt.max_dataset_size < float('inf'):
                self.paths = self.paths[:int(opt.max_dataset_size)]

    def __len__(self):
        return len(self.ds) if self._cifar else len(self.paths)

    def _get_detections(self, pil_img, basename):
        """Return list of ((x0,y0,x1,y1), label) from cache or online."""
        if self._bbox_cache:
            raw = self._bbox_cache.get(basename, [])
            # raw entry: [x0, y0, x1, y1, label]
            return [((r[0], r[1], r[2], r[3]), r[4]) for r in raw]
        else:
            return _predict_bbox(pil_img, self.device, self.box_num)

    def __getitem__(self, idx):
        if self._cifar:
            pil_img, _ = self.ds[idx]
            basename = f'cifar_{idx}.png'
        else:
            pil_img  = _load_rgb(self.paths[idx])
            basename = os.path.basename(self.paths[idx])

        rgb_img, gray_img = _to_gray_rgb(pil_img)
        detections = self._get_detections(rgb_img, basename)

        full_rgb  = self.tfm(rgb_img)
        full_gray = self.tfm(gray_img)

        n = len(detections)
        output = {
            'full_rgb':  full_rgb.unsqueeze(0),
            'full_gray': full_gray.unsqueeze(0),
            'empty_box': n == 0,
        }

        if n > 0:
            sz = self.final_size
            box_info    = np.zeros((n, 6), dtype=np.int64)
            box_info_2x = np.zeros((n, 6), dtype=np.int64)
            box_info_4x = np.zeros((n, 6), dtype=np.int64)
            box_info_8x = np.zeros((n, 6), dtype=np.int64)
            cropped_rgb_list = []
            labels = []

            for i, (bbox, label) in enumerate(detections):
                x0, y0, x1, y1 = bbox
                box_info[i]    = get_box_info(bbox, rgb_img.size, sz)
                box_info_2x[i] = get_box_info(bbox, rgb_img.size, sz // 2)
                box_info_4x[i] = get_box_info(bbox, rgb_img.size, sz // 4)
                box_info_8x[i] = get_box_info(bbox, rgb_img.size, sz // 8)
                cropped_rgb_list.append(self.tfm(rgb_img.crop((x0, y0, x1, y1))))
                labels.append(label)

            output['cropped_rgb']   = torch.stack(cropped_rgb_list)
            output['class_labels']  = torch.tensor(labels, dtype=torch.long)
            output['box_info']      = torch.from_numpy(box_info)
            output['box_info_2x']   = torch.from_numpy(box_info_2x)
            output['box_info_4x']   = torch.from_numpy(box_info_4x)
            output['box_info_8x']   = torch.from_numpy(box_info_8x)

        return output


class TestDataset(Data.Dataset):
    """
    Inference dataset. Returns full image + optional online bbox.
    Used by test.py for both cnn_color and inst_fusion.
    """
    def __init__(self, opt, box_num=8):
        self.opt = opt
        self.box_num = box_num
        sz = opt.fineSize
        self.final_size = sz
        self.tfm = T.Compose([
            T.Resize((sz, sz), interpolation=Image.BILINEAR),
            T.ToTensor(),
        ])
        self.paths = _collect_images(opt.test_img_dir)
        self.device = torch.device(
            f'cuda:{opt.gpu_ids[0]}' if opt.gpu_ids else 'cpu')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        pil_img = _load_rgb(path)
        file_id = os.path.splitext(os.path.basename(path))[0]
        orig_W, orig_H = pil_img.size

        rgb_img, gray_img = _to_gray_rgb(pil_img)
        output = {
            'rgb_img':  self.tfm(rgb_img),
            'gray_img': self.tfm(gray_img),
            'file_id':  file_id,
            'empty_box': True,
            'orig_size': torch.tensor([orig_H, orig_W]),
        }

        if self.opt.method == 'inst_fusion':
            detections = _predict_bbox(pil_img, self.device, self.box_num)
            if detections:
                sz = self.final_size
                n = len(detections)
                box_info    = np.zeros((n, 6), dtype=np.int64)
                box_info_2x = np.zeros((n, 6), dtype=np.int64)
                box_info_4x = np.zeros((n, 6), dtype=np.int64)
                box_info_8x = np.zeros((n, 6), dtype=np.int64)
                cropped = []
                labels  = []
                for i, (bbox, label) in enumerate(detections):
                    box_info[i]    = get_box_info(bbox, pil_img.size, sz)
                    box_info_2x[i] = get_box_info(bbox, pil_img.size, sz // 2)
                    box_info_4x[i] = get_box_info(bbox, pil_img.size, sz // 4)
                    box_info_8x[i] = get_box_info(bbox, pil_img.size, sz // 8)
                    x0, y0, x1, y1 = bbox
                    cropped.append(self.tfm(pil_img.crop((x0, y0, x1, y1))))
                    labels.append(label)
                output['empty_box']    = False
                output['cropped_img']  = torch.stack(cropped)
                output['class_labels'] = torch.tensor(labels, dtype=torch.long)
                output['box_info']     = torch.from_numpy(box_info)
                output['box_info_2x']  = torch.from_numpy(box_info_2x)
                output['box_info_4x']  = torch.from_numpy(box_info_4x)
                output['box_info_8x']  = torch.from_numpy(box_info_8x)

        return output


# ── factory ───────────────────────────────────────────────────────────────────

def create_dataset(opt, stage='full', split='train'):
    """Return the appropriate Dataset instance for the given stage."""
    if split == 'test':
        return TestDataset(opt)
    if stage == 'instance':
        return InstanceDataset(opt, split)
    if stage == 'fusion':
        return FusionDataset(opt, split)
    return ColorizationDataset(opt, split)
