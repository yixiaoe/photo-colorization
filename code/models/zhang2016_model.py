"""
Phase 1 model — Zhang et al., ECCV 2016
"Colorful Image Colorization"

Responsibilities of THIS file
------------------------------
- ECCVGenerator        : encoder-decoder network  (L → Q-bin logits)
- ColorfulLoss         : reweighted multinomial cross-entropy  (Eq. 2-4)
- annealed_mean_decode : probability → ab point estimate  (Eq. 5)
- Zhang2016Model       : BaseModel wrapper (train / infer / visualise)

Color-space conversion (RGB ↔ Lab) lives entirely in dataset.py.
This file only consumes and produces tensors:
  input  : L_img   (B, 1, H, W) float32  [0, 100]
  target : ab_img  (B, 2, H, W) float32  [-128, 127]
  output : ab_pred (B, 2, H, W) float32
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from ..datasets.colorization_dataset import lab_tensors_to_rgb_tensor   # visualisation only


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ab quantization — 313 in-gamut bin centres
# ─────────────────────────────────────────────────────────────────────────────

def _build_ab_gamut(grid=10, L_val=50.0):
    """
    Enumerate (a, b) pairs on a regular grid; keep only in-gamut entries.
    Returns ndarray (Q, 2) with Q ≈ 313.
    """
    from skimage import color as skcolor
    a_vals   = np.arange(-110, 110 + grid, grid)
    b_vals   = np.arange(-110, 110 + grid, grid)
    ab_pairs = np.array([[a, b] for a in a_vals for b in b_vals],
                        dtype=np.float32)
    L_col    = np.full((len(ab_pairs), 1), L_val, dtype=np.float32)
    lab      = np.concatenate([L_col, ab_pairs], axis=1)
    rgb      = skcolor.lab2rgb(lab[np.newaxis])[0]
    in_gamut = np.all((rgb >= 0) & (rgb <= 1), axis=1)
    return ab_pairs[in_gamut]


AB_GAMUT = _build_ab_gamut()   # (313, 2)  – module-level constant
Q        = len(AB_GAMUT)       # 313


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Soft-encoding  (paper footnote 2)
# ─────────────────────────────────────────────────────────────────────────────

def soft_encode(ab_gt, sigma=5.0, K=5):
    """
    Soft-encode ground-truth ab map into a Q-class distribution.

    Args:
        ab_gt : (B, 2, H, W) – ground-truth ab channels
        sigma : Gaussian bandwidth
        K     : number of nearest neighbours

    Returns:
        Z : (B, Q, H, W) – soft target distribution (sums to 1 over Q)
    """
    B, _, H, W = ab_gt.shape
    device     = ab_gt.device
    centres    = torch.from_numpy(AB_GAMUT).to(device)         # (Q, 2)

    ab_flat    = ab_gt.permute(0, 2, 3, 1).reshape(-1, 2)      # (B*H*W, 2)
    diff       = ab_flat.unsqueeze(1) - centres.unsqueeze(0)    # (N, Q, 2)
    dist2      = (diff ** 2).sum(-1)                            # (N, Q)

    topk_dist2, topk_idx = dist2.topk(K, dim=1, largest=False) # (N, K)
    weights = torch.exp(-topk_dist2 / (2 * sigma ** 2))
    weights = weights / weights.sum(dim=1, keepdim=True)

    Z_flat = torch.zeros(B * H * W, Q, device=device)
    Z_flat.scatter_(1, topk_idx, weights)
    return Z_flat.reshape(B, H, W, Q).permute(0, 3, 1, 2)      # (B, Q, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Class-rarity reweighting  (Eq. 3-4)
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(prior_probs=None, lam=0.5, device='cpu'):
    """
    Eq. (4):  w  ∝  1 / ((1-λ)·p̃ + λ/Q),   E[w] = 1.

    Args:
        prior_probs : ndarray (Q,) empirical colour distribution.
                      Pass None to fall back to uniform (equivalent to λ=1).
        lam         : mixing weight with uniform distribution
        device      : torch device string

    Returns:
        weights : FloatTensor (Q,)
    """
    if prior_probs is None:
        prior_probs = np.ones(Q, dtype=np.float32) / Q
    prior_probs = prior_probs.astype(np.float32)
    mixed = (1.0 - lam) * prior_probs + lam / Q
    w     = 1.0 / mixed
    w     = w / (prior_probs * w).sum()   # normalise: E[w] = 1
    return torch.tensor(w, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Loss  (Eq. 2)
# ─────────────────────────────────────────────────────────────────────────────

class ColorfulLoss(nn.Module):
    """
    Reweighted multinomial cross-entropy.

    Args (forward):
        Zhat  : (B, Q, H, W) – raw network logits
        ab_gt : (B, 2, H, W) – ground-truth ab at full resolution
    """
    def __init__(self, prior_probs=None, lam=0.5):
        super().__init__()
        w = compute_class_weights(prior_probs, lam)   # (Q,)
        self.register_buffer('class_weights', w)

    def forward(self, Zhat, ab_gt):
        _, _, Hout, Wout = Zhat.shape

        # align ground-truth to network output resolution
        ab_ds = F.interpolate(ab_gt, size=(Hout, Wout),
                              mode='bilinear', align_corners=False)

        # soft target  Z  ∈ [0,1]^(B,Q,H,W)
        Z = soft_encode(ab_ds, sigma=5.0, K=5)

        # per-pixel rarity weight  v(Z_{h,w}) = w_{q*}   (Eq. 3)
        q_star        = Z.argmax(dim=1)                          # (B,H,W)
        pixel_weights = self.class_weights[q_star]               # (B,H,W)

        # cross-entropy: -Σ_q  Z · log softmax(Zhat)
        log_preds = F.log_softmax(Zhat, dim=1)
        ce        = -(Z * log_preds).sum(dim=1)                  # (B,H,W)

        return (pixel_weights * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Annealed-mean decoding  (Eq. 5)
# ─────────────────────────────────────────────────────────────────────────────

def annealed_mean_decode(Zhat, T=0.38):
    """
    Map predicted logits → ab point estimate.

    f_T(z) = softmax(log z / T),   then  ab = E_{f_T}[ab_bin_centres]

    T = 1   → plain mean  (desaturated / sepia tone)
    T → 0   → mode        (vibrant but spatially noisy)
    T = 0.38 → paper default, balances vibrancy and coherence

    Args:
        Zhat : (B, Q, H, W) raw logits
        T    : temperature ∈ (0, 1]

    Returns:
        ab : (B, 2, H, W)
    """
    log_z   = F.log_softmax(Zhat, dim=1)                          # numerically stable
    probs   = F.softmax(log_z / T, dim=1)                         # re-annealed
    centres = torch.from_numpy(AB_GAMUT).to(Zhat.device)          # (Q, 2)
    return torch.einsum('bqhw,qc->bchw', probs, centres)          # (B, 2, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Network
# ─────────────────────────────────────────────────────────────────────────────

class ECCVGenerator(nn.Module):
    """
    8-block VGG-style encoder with dilated convolutions in blocks 5-6.

    Input : L channel  (B, 1, H, W)   values in [0, 100]
    Output: raw logits (B, Q, H, W)   Q = 313  (full resolution after ×4 up)
    """
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super().__init__()

        def blk(*layers): return nn.Sequential(*layers)

        self.model1 = blk(
            nn.Conv2d(1,   64,  3, stride=1, padding=1), nn.ReLU(True),
            nn.Conv2d(64,  64,  3, stride=2, padding=1), nn.ReLU(True),
            norm_layer(64),
        )
        self.model2 = blk(
            nn.Conv2d(64,  128, 3, stride=1, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(True),
            norm_layer(128),
        )
        self.model3 = blk(
            nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(True),
            norm_layer(256),
        )
        self.model4 = blk(
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            norm_layer(512),
        )
        # dilated blocks — expand receptive field without downsampling
        self.model5 = blk(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(True),
            norm_layer(512),
        )
        self.model6 = blk(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(True),
            norm_layer(512),
        )
        self.model7 = blk(
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            norm_layer(512),
        )
        # decoder: ×2 ConvTranspose, then ×4 bilinear → restored to input res
        self.model8 = blk(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),                     nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),                     nn.ReLU(True),
            nn.Conv2d(256, Q,   1),
        )
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear',
                                     align_corners=False)

    def forward(self, L):
        """
        Args:
            L : (B, 1, H, W)  L channel in [0, 100]
        Returns:
            logits : (B, Q, H, W)
        """
        x = (L - 50.0) / 100.0    # normalise to ≈ [-0.5, 0.5]
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        x = self.model4(x)
        x = self.model5(x)
        x = self.model6(x)
        x = self.model7(x)
        x = self.model8(x)         # spatial: H/4 × W/4
        return self.upsample4(x)   # spatial: H × W


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Zhang2016Model
# ─────────────────────────────────────────────────────────────────────────────

class Zhang2016Model(BaseModel):
    """
    BaseModel wrapper for Phase 1 training and inference.

    Expected opt fields
    -------------------
    opt.lr           : learning rate          (default 3e-5)
    opt.beta1        : Adam β1                (default 0.9)
    opt.lam          : reweighting λ          (default 0.5)
    opt.temperature  : annealed-mean T        (default 0.38)
    opt.prior_probs  : path to .npy (Q,) empirical ab prior, or None
    """

    def name(self):
        return 'Zhang2016Model'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--lr',          type=float, default=3e-5)
        parser.add_argument('--beta1',       type=float, default=0.9)
        parser.add_argument('--lam',         type=float, default=0.5)
        parser.add_argument('--temperature', type=float, default=0.38)
        parser.add_argument('--prior_probs', type=str,   default=None,
                            help='path to empirical ab prior .npy, shape (313,)')
        return parser

    def initialize(self, opt):
        super().initialize(opt)
        self.model_names = ['G']
        self.netG        = ECCVGenerator().to(self.device)

        prior = None
        if getattr(opt, 'prior_probs', None) and os.path.isfile(opt.prior_probs):
            prior = np.load(opt.prior_probs).astype(np.float32)
        self.criterion  = ColorfulLoss(
            prior_probs=prior,
            lam=getattr(opt, 'lam', 0.5),
        ).to(self.device)

        self.temperature = getattr(opt, 'temperature', 0.38)
        self.loss_G      = torch.tensor(0.0)

        if self.isTrain:
            self.optimizers = [
                torch.optim.Adam(
                    self.netG.parameters(),
                    lr=getattr(opt, 'lr', 3e-5),
                    betas=(getattr(opt, 'beta1', 0.9), 0.99),
                )
            ]
            self.setup_schedulers()

    # ── data interface ────────────────────────────────────────────────────

    def set_input(self, data):
        """
        Consumes output from ColorizationDataset / InstanceDataset.
        No color conversion — dataset.py already produced Lab tensors.

            data['L_img']  : (B, 1, H, W)  [0, 100]
            data['ab_img'] : (B, 2, H, W)  [-128, 127]
        """
        self.input_L = data['L_img'].to(self.device)    # (B, 1, H, W)
        self.real_ab = data.get('ab_img')
        if self.real_ab is not None:
            self.real_ab = self.real_ab.to(self.device)  # (B, 2, H, W)

    # ── forward / backward ────────────────────────────────────────────────

    def forward(self):
        self.pred_logits = self.netG(self.input_L)       # (B, Q, H, W)

    def backward_G(self):
        if self.real_ab is None:
            raise RuntimeError('ab_img is required for training Zhang2016Model.')
        self.loss_G = self.criterion(self.pred_logits, self.real_ab)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        for opt in self.optimizers:
            opt.zero_grad()
        self.backward_G()
        for opt in self.optimizers:
            opt.step()

    # ── monitoring ────────────────────────────────────────────────────────

    def get_current_losses(self):
        return {'G': self.loss_G.item()}

    def get_current_visuals(self):
        """
        Returns [0,1] RGB tensors for logging / visualisation.
        Lab → RGB reconstruction is handled by dataset.lab_tensors_to_rgb_tensor,
        keeping color-space logic out of this file.
        """
        with torch.no_grad():
            pred_ab = annealed_mean_decode(
                self.pred_logits, T=self.temperature).clamp(-110, 110)

        visuals = {
            'fake_rgb': lab_tensors_to_rgb_tensor(self.input_L, pred_ab),
            'input_L':  self.input_L.expand(-1, 3, -1, -1) / 100.0,
        }
        if self.real_ab is not None:
            visuals['real_rgb'] = lab_tensors_to_rgb_tensor(self.input_L, self.real_ab)
        return visuals

    # ── inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def colorize(self, L_tensor, temperature=None):
        """
        Standalone inference.

        Args:
            L_tensor    : (B, 1, H, W)  L channel in [0, 100]
                          (obtain via dataset._rgb_pil_to_lab_tensors)
            temperature : override self.temperature if given
        Returns:
            ab_pred : (B, 2, H, W)
        """
        T = temperature if temperature is not None else self.temperature
        self.netG.eval()
        logits = self.netG(L_tensor.to(self.device))
        return annealed_mean_decode(logits, T=T)
