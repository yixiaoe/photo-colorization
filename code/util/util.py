"""Color-space utilities and Lab/RGB conversion for PyTorch tensors."""
import os
import numpy as np
import torch
from PIL import Image


# ── Lab ↔ RGB (GPU-compatible) ────────────────────────────────────────────────

def rgb2xyz(rgb):
    mask = (rgb > .04045).float().to(rgb.device)
    rgb_ = ((rgb + .055) / 1.055) ** 2.4 * mask + rgb / 12.92 * (1 - mask)
    x = .412453 * rgb_[:, 0] + .357580 * rgb_[:, 1] + .180423 * rgb_[:, 2]
    y = .212671 * rgb_[:, 0] + .715160 * rgb_[:, 1] + .072169 * rgb_[:, 2]
    z = .019334 * rgb_[:, 0] + .119193 * rgb_[:, 1] + .950227 * rgb_[:, 2]
    return torch.stack([x, y, z], dim=1)


def xyz2rgb(xyz):
    r =  3.24048134 * xyz[:, 0] - 1.53715152 * xyz[:, 1] - 0.49853633 * xyz[:, 2]
    g = -0.96925495 * xyz[:, 0] + 1.87599    * xyz[:, 1] + 0.04155593 * xyz[:, 2]
    b =  0.05564664 * xyz[:, 0] - 0.20404134 * xyz[:, 1] + 1.05731107 * xyz[:, 2]
    rgb = torch.clamp(torch.stack([r, g, b], dim=1), min=0)
    mask = (rgb > .0031308).float().to(rgb.device)
    return (1.055 * rgb ** (1 / 2.4) - 0.055) * mask + 12.92 * rgb * (1 - mask)


def xyz2lab(xyz):
    sc = torch.tensor([0.95047, 1.0, 1.08883], device=xyz.device)[None, :, None, None]
    xyz_s = xyz / sc
    mask = (xyz_s > .008856).float().to(xyz.device)
    xyz_i = xyz_s ** (1 / 3) * mask + (7.787 * xyz_s + 16 / 116) * (1 - mask)
    L = 116 * xyz_i[:, 1] - 16
    a = 500 * (xyz_i[:, 0] - xyz_i[:, 1])
    b = 200 * (xyz_i[:, 1] - xyz_i[:, 2])
    return torch.stack([L, a, b], dim=1)


def lab2xyz(lab):
    y = (lab[:, 0] + 16) / 116
    x = lab[:, 1] / 500 + y
    z = y - lab[:, 2] / 200
    z = torch.clamp(z, min=0)
    out = torch.stack([x, y, z], dim=1)
    mask = (out > .2068966).float().to(lab.device)
    out = out ** 3 * mask + (out - 16 / 116) / 7.787 * (1 - mask)
    sc = torch.tensor([0.95047, 1.0, 1.08883], device=lab.device)[None, :, None, None]
    return out * sc


def rgb2lab(rgb, opt):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:, [0]] - opt.l_cent) / opt.l_norm
    ab_rs = lab[:, 1:] / opt.ab_norm
    return torch.cat([l_rs, ab_rs], dim=1)


def lab2rgb(lab_rs, opt):
    l  = lab_rs[:, [0]] * opt.l_norm + opt.l_cent
    ab = lab_rs[:, 1:] * opt.ab_norm
    return xyz2rgb(lab2xyz(torch.cat([l, ab], dim=1)))


# ── ab quantisation ───────────────────────────────────────────────────────────

def encode_ab_ind(data_ab, opt):
    """Nx2xHxW ∈ [-1,1]  →  Nx1xHxW index ∈ [0, Q)"""
    data_ab_rs = torch.round((data_ab * opt.ab_norm + opt.ab_max) / opt.ab_quant)
    return data_ab_rs[:, [0]] * opt.A + data_ab_rs[:, [1]]


def decode_ind_ab(data_q, opt):
    """Nx1xHxW index  →  Nx2xHxW ∈ [-1,1]"""
    data_a = (data_q // opt.A).float()
    data_b = (data_q  % opt.A).float()
    ab = torch.cat([data_a, data_b], dim=1)
    return (ab * opt.ab_quant - opt.ab_max) / opt.ab_norm


def decode_max_ab(data_ab_quant, opt):
    """NxQxHxW → Nx2xHxW  (argmax decoding)"""
    data_q = torch.argmax(data_ab_quant, dim=1, keepdim=True)
    return decode_ind_ab(data_q, opt)


def decode_mean(data_ab_quant, opt):
    """NxQxHxW → Nx2xHxW  (mean decoding)"""
    N, Q, H, W = data_ab_quant.shape
    a_range = torch.arange(-opt.ab_max, opt.ab_max + opt.ab_quant,
                           step=opt.ab_quant, device=data_ab_quant.device
                           ).float()[None, :, None, None]
    quant = data_ab_quant.view(N, int(opt.A), int(opt.A), H, W)
    a_inf = torch.sum(torch.sum(quant, dim=2) * a_range, dim=1, keepdim=True)
    b_inf = torch.sum(torch.sum(quant, dim=1) * a_range, dim=1, keepdim=True)
    return torch.cat([a_inf, b_inf], dim=1) / opt.ab_norm


def get_colorization_data(data_raw, opt, ab_thresh=5.):
    """Convert raw RGB tensor to {'A': L, 'B': ab} in normalised Lab space."""
    data = {}
    data_lab = rgb2lab(data_raw, opt)
    data['A'] = data_lab[:, [0]]      # L channel
    data['B'] = data_lab[:, 1:]       # ab channels
    if ab_thresh > 0:
        thresh = ab_thresh / opt.ab_norm
        ab_range = (torch.max(data['B'].flatten(2), dim=2)[0]
                    - torch.min(data['B'].flatten(2), dim=2)[0])
        mask = ab_range.sum(dim=1) >= thresh
        data['A'] = data['A'][mask]
        data['B'] = data['B'][mask]
        if data['A'].shape[0] == 0:
            return None
    return data


# ── image I/O helpers ─────────────────────────────────────────────────────────

def tensor2im(t, imtype=np.uint8):
    arr = t[0].cpu().float().numpy()
    if arr.shape[0] == 1:
        arr = np.tile(arr, (3, 1, 1))
    arr = np.clip(arr.transpose(1, 2, 0), 0, 1) * 255
    return arr.astype(imtype)


def save_image(arr, path):
    Image.fromarray(arr).save(path)


def mkdirs(paths):
    if isinstance(paths, list):
        for p in paths:
            os.makedirs(p, exist_ok=True)
    else:
        os.makedirs(paths, exist_ok=True)
