"""Phase 1 Zhang et al. 2016 model wrapper."""
import torch
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks
from util.util import (
    build_zhang2016_rebalance_weights,
    decode_zhang2016_annealed_mean,
    encode_ab_to_zhang2016_bins,
    lab2rgb,
    load_zhang2016_ab_bins,
    load_zhang2016_prior_probs,
    rgb2lab,
)


class Zhang2016Model(BaseModel):
    def name(self):
        return 'Zhang2016Model'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        super().initialize(opt)
        self.model_names = ['G']
        self.netG = networks.define_G(opt)
        self.pts_in_hull = load_zhang2016_ab_bins(self.device)
        self.prior_probs = load_zhang2016_prior_probs(self.device)
        self.class_weights = build_zhang2016_rebalance_weights(
            self.prior_probs, gamma=0.5)
        self.loss_names = ['G_total', 'G_CE']

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, getattr(opt, 'beta2', 0.999)),
            )
            self.optimizers = [self.optimizer_G]
            self.setup_schedulers()

    def set_input(self, data):
        raw = None
        for key in ('rgb_img', 'full_img', 'gray_img', 'full_gray'):
            if key in data:
                raw = data[key]
                break
        if raw is None:
            raise KeyError('Zhang2016Model expected one of rgb_img/full_img/gray_img/full_gray')

        if raw.dim() == 5 and raw.size(1) == 1:
            raw = raw.squeeze(1)
        self.input_rgb = raw.to(self.device).float()
        lab = rgb2lab(self.input_rgb, self.opt)
        self.real_A = lab[:, [0]]
        self.real_B = lab[:, 1:]

    def forward(self):
        self.pred_logits = self.netG(self.real_A)
        if hasattr(self, 'fake_B'):
            del self.fake_B
        return self.pred_logits

    def compute_target_labels(self):
        """Downsample ground-truth ab and encode it as 313-bin labels."""
        target_ab = self.real_B
        if target_ab.shape[-2:] != self.pred_logits.shape[-2:]:
            target_ab = F.interpolate(
                target_ab,
                size=self.pred_logits.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        self.target_labels = encode_ab_to_zhang2016_bins(
            target_ab, self.pts_in_hull, self.opt)
        return self.target_labels

    def backward_G(self):
        self.compute_target_labels()
        self.loss_G_CE = F.cross_entropy(
            self.pred_logits,
            self.target_labels,
            weight=self.class_weights,
        )
        self.loss_G_total = self.loss_G_CE
        self.loss_G_total.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def decode_prediction(self, temperature=0.38):
        """Decode current logits to full-resolution normalised ab."""
        fake_B = decode_zhang2016_annealed_mean(
            self.pred_logits, self.pts_in_hull, self.opt, temperature=temperature)
        if fake_B.shape[-2:] != self.real_A.shape[-2:]:
            fake_B = F.interpolate(
                fake_B, size=self.real_A.shape[-2:], mode='bilinear', align_corners=False)
        self.fake_B = fake_B
        return fake_B

    def get_current_visuals(self):
        if not hasattr(self, 'pred_logits'):
            self.forward()
        if not hasattr(self, 'fake_B'):
            self.decode_prediction()

        zero_ab = torch.zeros_like(self.fake_B)
        visuals = {
            'gray': lab2rgb(torch.cat([self.real_A, zero_ab], dim=1), self.opt),
            'fake_color': lab2rgb(torch.cat([self.real_A, self.fake_B], dim=1), self.opt),
        }
        if hasattr(self, 'real_B') and self.real_B.shape[-2:] == self.real_A.shape[-2:]:
            visuals['real_color'] = lab2rgb(torch.cat([self.real_A, self.real_B], dim=1), self.opt)
        return visuals

    def get_current_losses(self):
        return {
            'G_total': float(self.loss_G_total.detach().cpu()),
            'G_CE': float(self.loss_G_CE.detach().cpu()),
        }
