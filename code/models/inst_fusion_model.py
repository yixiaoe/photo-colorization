"""Phase 2 model: dual-branch instance/full-image fusion."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .networks import InstFusionGenerator
from util.util import rgb2lab, lab2rgb


class InstFusionModel(BaseModel):
    def name(self):
        return 'InstFusionModel'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        super().initialize(opt)
        self.model_names = ['G']
        self.netG = InstFusionGenerator().to(self.device)
        self.criterion = nn.L1Loss().to(self.device)
        self._use_amp = (len(opt.gpu_ids) > 0 and torch.cuda.is_available())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)

        self._load_stage_backbone()

        if self.isTrain:
            self.optimizer = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]
            self.setup_schedulers()

    def _load_stage_backbone(self):
        phase1_name = getattr(self.opt, 'phase1_name', 'cnn_color')
        full_name = getattr(self.opt, 'full_stage_name', 'inst_fusion_full')
        inst_name = getattr(self.opt, 'instance_stage_name', 'inst_fusion_instance')
        stage = getattr(self.opt, 'stage', 'full')

        if stage == 'full':
            ckpt = os.path.join(self.opt.checkpoints_dir, phase1_name, 'latest_net_G.pth')
            if os.path.isfile(ckpt):
                state = torch.load(ckpt, map_location=self.device)
                self.netG.load_phase1_weights(state, strict=False)
                print(f'Loaded Stage(full) init from {ckpt}')
            return

        if stage == 'instance':
            ckpt = os.path.join(self.opt.checkpoints_dir, full_name, 'latest_net_G.pth')
            if os.path.isfile(ckpt):
                state = torch.load(ckpt, map_location=self.device)
                self.netG.load_state_dict(state, strict=False)
                print(f'Loaded Stage(instance) init from {ckpt}')
            return

        if stage == 'fusion':
            full_ckpt = os.path.join(self.opt.checkpoints_dir, full_name, 'latest_net_G.pth')
            inst_ckpt = os.path.join(self.opt.checkpoints_dir, inst_name, 'latest_net_G.pth')
            if os.path.isfile(full_ckpt):
                state_full = torch.load(full_ckpt, map_location=self.device)
                self.netG.load_state_dict(state_full, strict=False)
                print(f'Loaded Stage(fusion) full init from {full_ckpt}')
            if os.path.isfile(inst_ckpt):
                state_inst = torch.load(inst_ckpt, map_location=self.device)
                self.netG.load_state_dict(state_inst, strict=False)
                print(f'Loaded Stage(fusion) instance init from {inst_ckpt}')

    def _compose_instance_ab_map(self, pred_crop_ab, box_info, out_hw):
        bsz = self.real_L.size(0)
        h, w = out_hw
        out = torch.zeros((bsz, 2, h, w), device=self.device)
        cnt = torch.zeros((bsz, 1, h, w), device=self.device)

        for b in range(bsz):
            n = pred_crop_ab[b].size(0)
            for i in range(n):
                info = box_info[b, i]
                left = int(info[0].item())
                top = int(info[2].item())
                rh = max(int(info[5].item()), 1)
                rw = max(int(info[4].item()), 1)

                patch = F.interpolate(pred_crop_ab[b, i:i + 1], size=(rh, rw), mode='bilinear', align_corners=False)
                right = min(left + rw, w)
                bottom = min(top + rh, h)
                patch = patch[:, :, :bottom - top, :right - left]
                out[b:b + 1, :, top:bottom, left:right] += patch
                cnt[b:b + 1, :, top:bottom, left:right] += 1.0

        cnt = torch.clamp(cnt, min=1.0)
        return out / cnt

    def set_input(self, data):
        if 'rgb_img' in data:
            rgb = data['rgb_img'].to(self.device)
            lab = rgb2lab(rgb, self.opt)
            self.real_L = lab[:, [0]]
            self.real_ab = lab[:, 1:]
            self.has_inst = ('cropped_img' in data and not bool(data.get('empty_box', True)))
            if self.has_inst:
                crop_rgb = data['cropped_img'].to(self.device)
                n = crop_rgb.shape[1]
                crop_lab = rgb2lab(crop_rgb.view(-1, 3, crop_rgb.size(-2), crop_rgb.size(-1)), self.opt)
                self.crop_L = crop_lab[:, [0]].view(rgb.size(0), n, 1, crop_rgb.size(-2), crop_rgb.size(-1))
                self.box_info_4x = data['box_info_4x'].to(self.device)
            return

        full_rgb = data['full_rgb'].to(self.device)
        if full_rgb.dim() == 5:
            full_rgb = full_rgb.squeeze(1)
        full_lab = rgb2lab(full_rgb, self.opt)
        self.real_L = full_lab[:, [0]]
        self.real_ab = full_lab[:, 1:]

        self.has_inst = ('cropped_rgb' in data and not bool(data.get('empty_box', True)))
        if self.has_inst:
            crop_rgb = data['cropped_rgb'].to(self.device)
            n = crop_rgb.shape[1]
            crop_lab = rgb2lab(crop_rgb.view(-1, 3, crop_rgb.size(-2), crop_rgb.size(-1)), self.opt)
            self.crop_L = crop_lab[:, [0]].view(full_rgb.size(0), n, 1, crop_rgb.size(-2), crop_rgb.size(-1))
            self.box_info_4x = data['box_info_4x'].to(self.device)

    def forward(self):
        if not getattr(self, 'has_inst', False):
            self.fake_ab = self.netG(self.real_L)
            self.fake_ab_full = F.interpolate(self.fake_ab, size=self.real_L.shape[-2:], mode='bilinear', align_corners=False)
            return

        bsz, n, _, h, w = self.crop_L.shape
        crop_L = self.crop_L.view(bsz * n, 1, h, w)
        crop_feat = self.netG.inst_branch.get_features(crop_L)
        crop_ab = self.netG.inst_head(crop_feat).view(bsz, n, 2, crop_feat.size(-2), crop_feat.size(-1))
        inst_map = self._compose_instance_ab_map(crop_ab, self.box_info_4x, out_hw=(self.real_L.size(-2) // 4, self.real_L.size(-1) // 4))

        self.fake_ab = self.netG(self.real_L, inst_ab_map=inst_map)
        self.fake_ab_full = F.interpolate(self.fake_ab, size=self.real_L.shape[-2:], mode='bilinear', align_corners=False)

    def optimize_parameters(self):
        with torch.cuda.amp.autocast(enabled=self._use_amp):
            self.forward()
            self.loss_G = self.criterion(self.fake_ab_full, self.real_ab)
        self.optimizer.zero_grad()
        self.scaler.scale(self.loss_G).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def get_current_losses(self):
        return {'G': self.loss_G.detach().item()} if hasattr(self, 'loss_G') else {}

    def get_current_visuals(self):
        with torch.no_grad():
            fake_rgb = lab2rgb(torch.cat([self.real_L, self.fake_ab_full], dim=1), self.opt).clamp(0, 1)
            real_rgb = lab2rgb(torch.cat([self.real_L, self.real_ab], dim=1), self.opt).clamp(0, 1)
            gray = (self.real_L * self.opt.l_norm + self.opt.l_cent) / 100.0
            gray = gray.expand(-1, 3, -1, -1).clamp(0, 1)
        return {
            'real_gray': gray,
            'fake_rgb': fake_rgb,
            'real_rgb': real_rgb,
        }
