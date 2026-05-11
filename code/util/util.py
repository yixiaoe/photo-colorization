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


# ── Zhang 2016 313-bin utilities ──────────────────────────────────────────────

_RESOURCE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'resources', 'zhang2016')


def load_zhang2016_ab_bins():
    """Load (313, 2) array of ab cluster centres."""
    path = os.path.join(_RESOURCE_DIR, 'pts_in_hull.npy')
    return np.load(path).astype(np.float32)


def load_zhang2016_prior_probs():
    """Load (313,) empirical prior probability over ab bins."""
    path = os.path.join(_RESOURCE_DIR, 'prior_probs.npy')
    return np.load(path).astype(np.float32)


def build_zhang2016_rebalance_weights(gamma=0.5, device='cpu'):
    """
    Compute per-class rebalance weights from the empirical ab prior.
    w = 1 / prior_mix,  prior_mix = (1-γ)*p + γ*(1/313)
    Returns Tensor (313,) on device.
    """
    prior = load_zhang2016_prior_probs()
    Q = prior.shape[0]
    uni = np.ones(Q, dtype=np.float32) / Q
    prior_mix = (1 - gamma) * prior + gamma * uni
    # avoid div-by-zero for bins that never appear
    prior_mix = np.clip(prior_mix, 1e-8, None)
    weights = 1.0 / prior_mix
    weights = weights / np.sum(prior * weights)   # re-normalise expectation to 1
    return torch.tensor(weights, dtype=torch.float32, device=device)


def encode_ab_to_zhang2016_bins(ab_norm, pts_in_hull, ab_norm_val=110.):
    """
    Encode normalised ab tensor to nearest-neighbour bin indices.

    Args:
        ab_norm:     Nx2xHxW  in [-1, 1]  (i.e. actual_ab / ab_norm_val)
        pts_in_hull: (313, 2) Tensor of ab cluster centres in raw ab units
        ab_norm_val: scalar used to normalise (default 110.)
    Returns:
        Nx1xHxW  int64 class indices in [0, 312]
    """
    N, _, H, W = ab_norm.shape
    ab = ab_norm * ab_norm_val                     # → raw ab  Nx2xHxW
    ab_flat = ab.permute(0, 2, 3, 1).reshape(-1, 2)   # (N*H*W, 2)
    # nearest neighbour via L2 distance
    pts = pts_in_hull.to(ab_flat.device)           # (313, 2)
    dists = torch.cdist(ab_flat.float(), pts.float())  # (N*H*W, 313)
    idx = dists.argmin(dim=1)                      # (N*H*W,)
    return idx.reshape(N, 1, H, W)


def decode_zhang2016_annealed_mean(logits, pts_in_hull, T=0.38, ab_norm_val=110.):
    """
    Annealed-mean decoding: softmax(logits / T) weighted sum over cluster centres.

    Args:
        logits:      Nx313xHxW  raw network output
        pts_in_hull: (313, 2) Tensor
        T:           temperature (default 0.38 per Zhang et al.)
        ab_norm_val: value used to normalise output to [-1, 1]
    Returns:
        Nx2xHxW  ab predictions in [-1, 1]
    """
    N, Q, H, W = logits.shape
    probs = torch.softmax(logits.float() / T, dim=1)        # Nx313xHxW, float32
    pts = pts_in_hull.to(device=logits.device, dtype=torch.float32)  # (313, 2)
    # weighted sum: (N, 313, H*W) x (313, 2) → (N, H*W, 2)
    probs_flat = probs.view(N, Q, H * W).permute(0, 2, 1)  # (N, H*W, 313)
    ab_flat = torch.matmul(probs_flat, pts)                 # (N, H*W, 2)
    ab = ab_flat.permute(0, 2, 1).view(N, 2, H, W)         # (N, 2, H, W)
    return ab / ab_norm_val


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
