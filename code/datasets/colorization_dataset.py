"""Unified dataset and color-space helpers for colorization tasks."""
import os
import random
from os.path import join

import numpy as np
from PIL import Image

import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from util.util import rgb2xyz, xyz2lab, lab2xyz, xyz2rgb


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_rgb(path):
    return Image.open(path).convert("RGB")


def _collect_images(root):
    paths = []
    for dirpath, _, fnames in os.walk(root):
        for name in fnames:
            if os.path.splitext(name)[1].lower() in IMG_EXTS:
                paths.append(join(dirpath, name))
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return sorted(paths)


def _resize_tensor_transform(size):
    return T.Compose([
        T.Resize((size, size), interpolation=Image.BILINEAR),
        T.ToTensor(),
    ])


def _rgb_pil_to_lab_tensors(pil_img):
    """Convert RGB PIL image to Lab tensors.

    Returns:
        L_img:  (1, H, W), float32, [0, 100]
        ab_img: (2, H, W), float32, roughly [-128, 127]
    """
    rgb_t = TF.to_tensor(pil_img).unsqueeze(0)  # (1,3,H,W) in [0,1]
    lab_t = xyz2lab(rgb2xyz(rgb_t)).squeeze(0)  # (3,H,W), unnormalised Lab
    L = lab_t[0:1].float()
    ab = lab_t[1:3].float()
    return L, ab


def _rgb_pil_to_gray_rgb_pil(pil_img):
    gray = TF.rgb_to_grayscale(TF.to_tensor(pil_img), num_output_channels=3)
    return T.ToPILImage()(gray)


def lab_tensors_to_rgb_tensor(L_img, ab_img):
    """Convert Lab tensors to RGB tensor in [0,1].

    Args:
        L_img:  (B, 1, H, W)
        ab_img: (B, 2, H, W)
    Returns:
        rgb: (B, 3, H, W), float32 in [0,1]
    """
    lab = torch.cat([L_img, ab_img], dim=1)
    rgb = xyz2rgb(lab2xyz(lab))
    return rgb.clamp(0.0, 1.0)


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
    model = _get_maskrcnn(device)
    tensor = TF.to_tensor(pil_img).unsqueeze(0).to(device)
    preds = model(tensor)[0]
    boxes = preds["boxes"].cpu().numpy().astype(int)
    scores = preds["scores"].cpu().numpy()
    keep = scores >= score_thresh
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) > box_num:
        idx = np.argsort(scores)[-box_num:]
        boxes = boxes[idx]
    return [tuple(b) for b in boxes]


def get_box_info(pred_bbox, original_size, final_size):
    W, H = original_size
    x0, y0, x1, y1 = pred_bbox
    sx = int(x0 / W * final_size)
    sy = int(y0 / H * final_size)
    ex = int(x1 / W * final_size)
    ey = int(y1 / H * final_size)
    rh = max(ex - sx, 1)
    rw = max(ey - sy, 1)
    if ex - sx < 1:
        if final_size - ex > 1:
            ex += 1
        else:
            sx -= 1
        rh = 1
    if ey - sy < 1:
        if final_size - ey > 1:
            ey += 1
        else:
            sy -= 1
        rw = 1
    return [sx, final_size - ex, sy, final_size - ey, rh, rw]


class _BaseImageDataset(Data.Dataset):
    def __init__(self, opt, split="train"):
        self.opt = opt
        self.tfm = _resize_tensor_transform(opt.fineSize)
        if opt.dataset == "cifar10":
            self.ds = torchvision.datasets.CIFAR10(
                root=opt.data_dir,
                train=(split == "train"),
                download=True,
                transform=None,
            )
            self._cifar = True
            self.paths = None
        else:
            self._cifar = False
            self.ds = None
            self.paths = _collect_images(opt.data_dir)
            if opt.max_dataset_size < float("inf"):
                self.paths = self.paths[: int(opt.max_dataset_size)]

    def __len__(self):
        return len(self.ds) if self._cifar else len(self.paths)

    def _get_pil(self, idx):
        if self._cifar:
            pil_img, _ = self.ds[idx]
            return pil_img
        return _load_rgb(self.paths[idx])


class ColorizationDataset(_BaseImageDataset):
    """Dataset for zhang2016 training.

    Returns keys expected by Zhang2016Model:
      - L_img:  (1,H,W)
      - ab_img: (2,H,W)
    Also returns rgb/gray for compatibility and visual logging.
    """

    def __getitem__(self, idx):
        pil_img = self._get_pil(idx)
        rgb_tensor = self.tfm(pil_img)
        gray_tensor = self.tfm(_rgb_pil_to_gray_rgb_pil(pil_img))

        # Lab tensors are computed from resized RGB to align spatially.
        resized_rgb_pil = T.ToPILImage()(rgb_tensor)
        L_img, ab_img = _rgb_pil_to_lab_tensors(resized_rgb_pil)

        return {
            "rgb_img": rgb_tensor,
            "gray_img": gray_tensor,
            "L_img": L_img,
            "ab_img": ab_img,
        }


class InstanceDataset(_BaseImageDataset):
    def __getitem__(self, idx):
        pil_img = self._get_pil(idx)
        W, H = pil_img.size
        crop_ratio = random.uniform(0.3, 1.0)
        cw = int(W * crop_ratio)
        ch = int(H * crop_ratio)
        x0 = random.randint(0, W - cw)
        y0 = random.randint(0, H - ch)
        crop = pil_img.crop((x0, y0, x0 + cw, y0 + ch))

        rgb_tensor = self.tfm(crop)
        gray_tensor = self.tfm(_rgb_pil_to_gray_rgb_pil(crop))
        resized_rgb_pil = T.ToPILImage()(rgb_tensor)
        L_img, ab_img = _rgb_pil_to_lab_tensors(resized_rgb_pil)

        return {
            "rgb_img": rgb_tensor,
            "gray_img": gray_tensor,
            "L_img": L_img,
            "ab_img": ab_img,
        }


class FusionDataset(_BaseImageDataset):
    def __init__(self, opt, split="train", box_num=8):
        super().__init__(opt, split)
        self.box_num = box_num
        self.final_size = opt.fineSize
        self.device = torch.device(f"cuda:{opt.gpu_ids[0]}" if opt.gpu_ids else "cpu")

    def __getitem__(self, idx):
        pil_img = self._get_pil(idx)
        rgb_img = pil_img
        gray_img = _rgb_pil_to_gray_rgb_pil(pil_img)
        bboxes = _predict_bbox(rgb_img, self.device, self.box_num)

        full_rgb = self.tfm(rgb_img)
        full_gray = self.tfm(gray_img)

        n = len(bboxes)
        output = {
            "full_rgb": full_rgb.unsqueeze(0),
            "full_gray": full_gray.unsqueeze(0),
            "empty_box": n == 0,
        }

        if n > 0:
            sz = self.final_size
            box_info = np.zeros((n, 6), dtype=np.int64)
            box_info_2x = np.zeros((n, 6), dtype=np.int64)
            box_info_4x = np.zeros((n, 6), dtype=np.int64)
            box_info_8x = np.zeros((n, 6), dtype=np.int64)
            cropped_rgb_list = []
            cropped_gray_list = []

            for i, bbox in enumerate(bboxes):
                x0, y0, x1, y1 = bbox
                box_info[i] = get_box_info(bbox, rgb_img.size, sz)
                box_info_2x[i] = get_box_info(bbox, rgb_img.size, sz // 2)
                box_info_4x[i] = get_box_info(bbox, rgb_img.size, sz // 4)
                box_info_8x[i] = get_box_info(bbox, rgb_img.size, sz // 8)
                crop = (x0, y0, x1, y1)
                cropped_rgb_list.append(self.tfm(rgb_img.crop(crop)))
                cropped_gray_list.append(self.tfm(gray_img.crop(crop)))

            output["cropped_rgb"] = torch.stack(cropped_rgb_list)
            output["cropped_gray"] = torch.stack(cropped_gray_list)
            output["box_info"] = torch.from_numpy(box_info)
            output["box_info_2x"] = torch.from_numpy(box_info_2x)
            output["box_info_4x"] = torch.from_numpy(box_info_4x)
            output["box_info_8x"] = torch.from_numpy(box_info_8x)

        return output


class TestDataset(Data.Dataset):
    def __init__(self, opt, box_num=8):
        self.opt = opt
        self.box_num = box_num
        self.final_size = opt.fineSize
        self.tfm = _resize_tensor_transform(opt.fineSize)
        self.paths = _collect_images(opt.test_img_dir)
        self.device = torch.device(f"cuda:{opt.gpu_ids[0]}" if opt.gpu_ids else "cpu")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        pil_img = _load_rgb(path)
        file_id = os.path.splitext(os.path.basename(path))[0]

        resized_rgb = self.tfm(pil_img)
        L_img, _ = _rgb_pil_to_lab_tensors(T.ToPILImage()(resized_rgb))

        output = {
            "full_img": resized_rgb.unsqueeze(0),
            "L_img": L_img,
            "file_id": file_id,
            "empty_box": True,
        }

        if self.opt.method == "inst2020":
            bboxes = _predict_bbox(pil_img, self.device, self.box_num)
            if bboxes:
                sz = self.final_size
                n = len(bboxes)
                box_info = np.zeros((n, 6), dtype=np.int64)
                box_info_2x = np.zeros((n, 6), dtype=np.int64)
                box_info_4x = np.zeros((n, 6), dtype=np.int64)
                box_info_8x = np.zeros((n, 6), dtype=np.int64)
                cropped = []
                for i, bbox in enumerate(bboxes):
                    box_info[i] = get_box_info(bbox, pil_img.size, sz)
                    box_info_2x[i] = get_box_info(bbox, pil_img.size, sz // 2)
                    box_info_4x[i] = get_box_info(bbox, pil_img.size, sz // 4)
                    box_info_8x[i] = get_box_info(bbox, pil_img.size, sz // 8)
                    x0, y0, x1, y1 = bbox
                    cropped.append(self.tfm(pil_img.crop((x0, y0, x1, y1))))
                output["empty_box"] = False
                output["cropped_img"] = torch.stack(cropped)
                output["box_info"] = torch.from_numpy(box_info)
                output["box_info_2x"] = torch.from_numpy(box_info_2x)
                output["box_info_4x"] = torch.from_numpy(box_info_4x)
                output["box_info_8x"] = torch.from_numpy(box_info_8x)

        return output


def create_dataset(opt, stage="full", split="train"):
    if split == "test":
        return TestDataset(opt, box_num=getattr(opt, "box_num", 8))
    if stage == "instance":
        return InstanceDataset(opt, split)
    if stage == "fusion":
        return FusionDataset(opt, split)
    return ColorizationDataset(opt, split)
