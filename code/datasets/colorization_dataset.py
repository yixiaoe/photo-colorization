"""
Unified colorization dataset.

Supports:
  - ImageNet-Mini  (folder of JPEG images, any structure)
  - CIFAR-10       (torchvision download)

Stage routing:
  'full'     → full image, returns rgb_img + gray_img
  'instance' → random-crop instance region (no offline bbox needed)
  'fusion'   → full image + online Mask R-CNN bbox (torchvision)
"""
import os
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

def _get_maskrcnn(device):
    global _maskrcnn
    if _maskrcnn is None:
        _maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(device).eval()
    return _maskrcnn


@torch.no_grad()
def _predict_bbox(pil_img, device, box_num=8, score_thresh=0.5):
    """Return list of (x0, y0, x1, y1) tuples from Mask R-CNN."""
    model = _get_maskrcnn(device)
    tensor = F.to_tensor(pil_img).unsqueeze(0).to(device)
    preds = model(tensor)[0]
    boxes  = preds['boxes'].cpu().numpy().astype(int)
    scores = preds['scores'].cpu().numpy()
    mask   = scores >= score_thresh
    boxes, scores = boxes[mask], scores[mask]
    if len(boxes) > box_num:
        idx    = np.argsort(scores)[-box_num:]
        boxes  = boxes[idx]
    return [tuple(b) for b in boxes]


# ── box geometry helper (mirrors reference image_util.get_box_info) ───────────

def get_box_info(pred_bbox, original_size, final_size):
    """
    Args:
        pred_bbox: (x0, y0, x1, y1) in original pixel coords
        original_size: (W, H) of the source image (PIL convention)
        final_size: target spatial resolution (square)
    Returns:
        [L_pad, R_pad, T_pad, B_pad, rh, rw]
    """
    W, H = original_size
    x0, y0, x1, y1 = pred_bbox
    sx = int(x0 / W * final_size)
    sy = int(y0 / H * final_size)
    ex = int(x1 / W * final_size)
    ey = int(y1 / H * final_size)
    rh = max(ex - sx, 1)
    rw = max(ey - sy, 1)
    if ex - sx < 1:
        if final_size - ex > 1: ex += 1
        else: sx -= 1
        rh = 1
    if ey - sy < 1:
        if final_size - ey > 1: ey += 1
        else: sy -= 1
        rw = 1
    return [sx, final_size - ex, sy, final_size - ey, rh, rw]


# ── ImageNet-Mini file list ───────────────────────────────────────────────────

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


# ── Dataset classes ───────────────────────────────────────────────────────────

class ColorizationDataset(Data.Dataset):
    """
    Full-image dataset for Phase 1 (cnn_color) and Phase 2 stage='full'.
    Returns {'rgb_img': Tensor(3,H,W), 'gray_img': Tensor(3,H,W)}.
    """
    def __init__(self, opt, split='train'):
        self.opt = opt
        sz = opt.fineSize
        self.tfm = T.Compose([
            T.Resize((sz, sz), interpolation=Image.BILINEAR),
            T.ToTensor(),
        ])
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

        rgb_img, gray_img = _to_gray_rgb(pil_img)
        return {
            'rgb_img':  self.tfm(rgb_img),
            'gray_img': self.tfm(gray_img),
        }


class InstanceDataset(Data.Dataset):
    """
    Phase 2 stage='instance'.
    Returns a random square crop of each image (simulating instance crops).
    No offline bbox required.
    """
    def __init__(self, opt, split='train'):
        self.opt = opt
        sz = opt.fineSize
        self.tfm = T.Compose([
            T.Resize((sz, sz), interpolation=Image.BILINEAR),
            T.ToTensor(),
        ])
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

        W, H = pil_img.size
        crop_ratio = random.uniform(0.3, 1.0)
        cw = int(W * crop_ratio)
        ch = int(H * crop_ratio)
        x0 = random.randint(0, W - cw)
        y0 = random.randint(0, H - ch)
        pil_img = pil_img.crop((x0, y0, x0 + cw, y0 + ch))

        rgb_img, gray_img = _to_gray_rgb(pil_img)
        return {
            'rgb_img':  self.tfm(rgb_img),
            'gray_img': self.tfm(gray_img),
        }


class FusionDataset(Data.Dataset):
    """
    Phase 2 stage='fusion'.
    Returns full image + online Mask R-CNN bbox info.
    Uses torchvision detection, no npz files required.
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
        self.device = torch.device(
            f'cuda:{opt.gpu_ids[0]}' if opt.gpu_ids else 'cpu')

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

        rgb_img, gray_img = _to_gray_rgb(pil_img)
        bboxes = _predict_bbox(rgb_img, self.device, self.box_num)

        full_rgb  = self.tfm(rgb_img)
        full_gray = self.tfm(gray_img)

        n = len(bboxes)
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
            cropped_rgb_list  = []
            cropped_gray_list = []

            for i, bbox in enumerate(bboxes):
                x0, y0, x1, y1 = bbox
                box_info[i]    = get_box_info(bbox, rgb_img.size, sz)
                box_info_2x[i] = get_box_info(bbox, rgb_img.size, sz // 2)
                box_info_4x[i] = get_box_info(bbox, rgb_img.size, sz // 4)
                box_info_8x[i] = get_box_info(bbox, rgb_img.size, sz // 8)
                crop = (x0, y0, x1, y1)
                cropped_rgb_list.append(self.tfm(rgb_img.crop(crop)))
                cropped_gray_list.append(self.tfm(gray_img.crop(crop)))

            output['cropped_rgb']  = torch.stack(cropped_rgb_list)
            output['cropped_gray'] = torch.stack(cropped_gray_list)
            output['box_info']     = torch.from_numpy(box_info)
            output['box_info_2x']  = torch.from_numpy(box_info_2x)
            output['box_info_4x']  = torch.from_numpy(box_info_4x)
            output['box_info_8x']  = torch.from_numpy(box_info_8x)

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

        full_img = self.tfm(pil_img)
        output = {
            'full_img': full_img.unsqueeze(0),
            'file_id':  file_id,
            'empty_box': True,
        }

        if self.opt.method == 'inst_fusion':
            bboxes = _predict_bbox(pil_img, self.device, self.box_num)
            if bboxes:
                sz = self.final_size
                n = len(bboxes)
                box_info    = np.zeros((n, 6), dtype=np.int64)
                box_info_2x = np.zeros((n, 6), dtype=np.int64)
                box_info_4x = np.zeros((n, 6), dtype=np.int64)
                box_info_8x = np.zeros((n, 6), dtype=np.int64)
                cropped = []
                for i, bbox in enumerate(bboxes):
                    box_info[i]    = get_box_info(bbox, pil_img.size, sz)
                    box_info_2x[i] = get_box_info(bbox, pil_img.size, sz // 2)
                    box_info_4x[i] = get_box_info(bbox, pil_img.size, sz // 4)
                    box_info_8x[i] = get_box_info(bbox, pil_img.size, sz // 8)
                    x0, y0, x1, y1 = bbox
                    cropped.append(self.tfm(pil_img.crop((x0, y0, x1, y1))))
                output['empty_box']    = False
                output['cropped_img']  = torch.stack(cropped)
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
