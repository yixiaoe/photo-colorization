"""Phase 1 model: full-image CNN colorization (Zhang et al. 2016)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .networks import CnnColorGenerator
from util.util import (
    rgb2lab, lab2rgb,
    load_zhang2016_ab_bins,
    build_zhang2016_rebalance_weights,
    encode_ab_to_zhang2016_bins,
    decode_zhang2016_annealed_mean,
)


class CnnColorModel(BaseModel):
    def name(self):
        return 'CnnColorModel'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # --T and --rebalance_gamma are defined in train_options.py / test_options.py
        return parser

    def initialize(self, opt):
        super().initialize(opt)
        self.model_names = ['G']

        # network
        self.netG = CnnColorGenerator().to(self.device)

        # 313 ab bin centres — kept as buffer on device
        pts = load_zhang2016_ab_bins()           # (313, 2) float32 np
        self.pts_in_hull = torch.tensor(pts, dtype=torch.float32, device=self.device)

        if self.isTrain:
            # rebalance weights
            w = build_zhang2016_rebalance_weights(
                gamma=opt.rebalance_gamma, device=self.device)
            # cross-entropy loss with per-class weights
            # label smoothing is handled implicitly by soft NN encoding;
            # we use hard nearest-neighbour labels here for simplicity
            self.criterion = nn.CrossEntropyLoss(weight=w).to(self.device)

            self.optimizer = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]
            self.setup_schedulers()

        # AMP scaler for mixed-precision (active only when CUDA available)
        self._use_amp = (len(opt.gpu_ids) > 0 and torch.cuda.is_available())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)

    # ── data loading ──────────────────────────────────────────────────────────

    def set_input(self, data):
        """
        data: dict from ColorizationDataset
          'rgb_img':  (N, 3, H, W) float32 in [0, 1]
        """
        rgb = data['rgb_img'].to(self.device)
        lab = rgb2lab(rgb, self.opt)                   # (N, 3, H, W) normalised Lab
        self.real_L  = lab[:, [0]]                    # (N, 1, H, W)
        self.real_ab = lab[:, 1:]                      # (N, 2, H, W)

        # ground-truth 313-bin class labels at 1/4 resolution (network output res)
        H, W = self.real_L.shape[2], self.real_L.shape[3]
        ab_down = F.interpolate(self.real_ab, size=(H // 4, W // 4), mode='bilinear',
                                align_corners=False)
        # (N, 1, H/4, W/4) int64
        self.gt_ab_class = encode_ab_to_zhang2016_bins(
            ab_down, self.pts_in_hull, ab_norm_val=self.opt.ab_norm)

    # ── forward / backward ────────────────────────────────────────────────────

    def forward(self):
        with torch.cuda.amp.autocast(enabled=self._use_amp):
            logits = self.netG(self.real_L)                # (N, 313, H/4, W/4)
        # always store as float32 so loss & decode never see half-precision
        self.pred_ab_logits = logits.float()

    def backward(self):
        # gt_ab_class: (N, 1, H/4, W/4) → (N, H/4, W/4) for CrossEntropyLoss
        gt = self.gt_ab_class[:, 0]                        # (N, H/4, W/4) int64
        self.loss_G = self.criterion(self.pred_ab_logits, gt)
        self.scaler.scale(self.loss_G).backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    # ── reporting ─────────────────────────────────────────────────────────────

    def get_current_losses(self):
        return {'G': self.loss_G.detach().item()}

    def get_current_visuals(self):
        """
        Returns RGB images (all in [0, 1]) for TensorBoard / saving:
          'real_gray': input L as grey RGB
          'fake_rgb':  predicted colourisation
          'real_rgb':  ground-truth colour image
        """
        with torch.no_grad():
            # decode: annealed-mean at temperature T (default from resources/defaults.py)
            pred_ab = decode_zhang2016_annealed_mean(
                self.pred_ab_logits, self.pts_in_hull,
                T=self.opt.T, ab_norm_val=self.opt.ab_norm)   # (N, 2, H/4, W/4)

            # upsample ab back to full resolution
            H, W = self.real_L.shape[2], self.real_L.shape[3]
            pred_ab_full = F.interpolate(pred_ab, size=(H, W), mode='bilinear',
                                         align_corners=False)

            fake_lab = torch.cat([self.real_L, pred_ab_full], dim=1)
            fake_rgb = lab2rgb(fake_lab, self.opt).clamp(0, 1)

            real_lab = torch.cat([self.real_L, self.real_ab], dim=1)
            real_rgb = lab2rgb(real_lab, self.opt).clamp(0, 1)

            # grey: repeat L channel 3 times, un-normalise to [0, 1]
            gray = (self.real_L * self.opt.l_norm + self.opt.l_cent) / 100.
            gray = gray.expand(-1, 3, -1, -1).clamp(0, 1)

        return {
            'real_gray': gray,
            'fake_rgb':  fake_rgb,
            'real_rgb':  real_rgb,
        }
