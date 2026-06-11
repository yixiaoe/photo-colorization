"""
Phase 3 model: text-conditioned colorization on top of frozen Phase 2.

Pipeline
  TextColorPipeline = FiLMInstanceGenerator(frozen) + FusionPipeline(frozen)
                    + InstanceTextAdapter(trainable)
                    + BgTextAdapter(trainable)

Each optimisation step does **two forwards** through TextColorPipeline,
one with the positive captions, one with the negatives, and combines:
  L_global  + λ_inst * L_inst_rec
            + λ_rank * L_rank
            + λ_outside * L_outside
with λ_rank / λ_outside on a cosine warmup schedule. The Phase 2 module
is frozen (eval mode, no_grad) so gradients only flow into the adapters.
"""
import math
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .networks import FiLMInstanceGenerator, FusionPipeline
from .text_color_networks import TextColorPipeline
from util.util import (
    rgb2lab, lab2rgb,
    load_zhang2016_ab_bins,
    build_zhang2016_rebalance_weights,
    encode_ab_bins_soft,
    encode_ab_bins_hard,
    decode_zhang2016_annealed_mean,
)
from util.clip_encoder import CLIPTextEncoder


# ── loss helpers ──────────────────────────────────────────────────────────────

def _huber(x, y, delta=0.01):
    d = (x - y).abs()
    return torch.where(d < delta, 0.5 * d ** 2 / delta, d - 0.5 * delta)


def _masked_mean(loss_map, mask):
    """mask broadcastable to loss_map; returns scalar."""
    m_sum = mask.sum().clamp(min=1.0)
    return (loss_map * mask).sum() / m_sum


def _rebalanced_ce_map(logits, gt_soft, gt_hard, rebalance_w):
    """
    Per-pixel rebalanced CE map for masked-mean averaging.
    Returns (N, H, W) loss with same H/W as logits' spatial dims.
    """
    log_p = F.log_softmax(logits.float().clamp(-100., 100.), dim=1).clamp(min=-100.)
    ce = -(gt_soft * log_p).sum(dim=1)               # (N, H, W)
    return ce * rebalance_w[gt_hard]                  # (N, H, W)


def _kl_soft_pred(gt_soft, pred_soft, eps=1e-8):
    """KL(gt || pred) per pixel, both (N, 313, H, W)."""
    return (gt_soft * (gt_soft.clamp(min=eps).log()
                       - pred_soft.clamp(min=eps).log())).sum(dim=1)  # (N, H, W)


class TextColorModel(BaseModel):
    def name(self):
        return 'TextColorModel'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    # ── init ──────────────────────────────────────────────────────────────────

    def initialize(self, opt):
        super().initialize(opt)
        self.model_names = ['T']    # only the text-color pipeline saves; Phase 2 is frozen

        # Phase 2 networks (frozen)
        self.netInst   = FiLMInstanceGenerator(num_classes=91, embed_dim=64).to(self.device).eval()
        self.netFusion = FusionPipeline().to(self.device).eval()
        self._load_phase2_ckpts(opt)

        # the wrapper holding trainable adapter (v2: instance only)
        adapter_hidden = getattr(opt, 'adapter_hidden', 512)
        self.netT = TextColorPipeline(self.netInst, self.netFusion,
                                      clip_dim=512,
                                      hidden_dim=adapter_hidden).to(self.device)
        print(f'[TextColorModel] trainable params: '
              f'{self.netT.num_trainable_parameters():,}')

        # CLIP text tower
        clip_cache = getattr(opt, 'clip_cache', '') or None
        self.clip = CLIPTextEncoder(device=self.device, cache_path=clip_cache)

        # 313 ab bin centres
        pts = load_zhang2016_ab_bins()
        self.pts_in_hull = torch.tensor(pts, dtype=torch.float32, device=self.device)

        # AMP — opt-in via --use_amp; off by default because Phase 2's frozen
        # BN can produce NaN under fp16 on some CUDA configs, and the adapter
        # is only ~1M params so fp32 is essentially free.
        use_amp_flag = getattr(opt, 'use_amp', False)
        self._use_amp = (use_amp_flag and len(opt.gpu_ids) > 0
                         and torch.cuda.is_available())
        self.scaler = torch.amp.GradScaler('cuda', enabled=self._use_amp)
        print(f'[TextColorModel] AMP: {"enabled" if self._use_amp else "disabled"}')

        if self.isTrain:
            self.rebalance_w = build_zhang2016_rebalance_weights(
                gamma=opt.rebalance_gamma, device=self.device)
            self.optimizer = torch.optim.Adam(
                self.netT.get_trainable_params(),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]
            self.setup_schedulers()
            # epoch counter for loss warmup
            self._epoch = 0

    def _load_phase2_ckpts(self, opt):
        # instance ckpt → FiLMInstanceGenerator
        if opt.inst_ckpt and os.path.isfile(opt.inst_ckpt):
            state = torch.load(opt.inst_ckpt, map_location=self.device)
            missing, unexpected = self.netInst.load_state_dict(state, strict=False)
            print(f'[load] inst_ckpt {opt.inst_ckpt}: '
                  f'missing={len(missing)} unexpected={len(unexpected)}')
        else:
            print(f'[warn] inst_ckpt not found: {opt.inst_ckpt}')

        # fusion ckpt → FusionPipeline (WG + heads + backbone copy)
        if opt.fusion_ckpt and os.path.isfile(opt.fusion_ckpt):
            state = torch.load(opt.fusion_ckpt, map_location=self.device)
            self.netFusion.load_state_dict(state, strict=False)
            print(f'[load] fusion_ckpt {opt.fusion_ckpt}')

        # full ckpt → overwrite FusionPipeline backbone (mirrors Phase 2 init)
        if opt.full_ckpt and os.path.isfile(opt.full_ckpt):
            state = torch.load(opt.full_ckpt, map_location=self.device)
            self.netFusion.load_state_dict(state, strict=False)
            print(f'[load] full_ckpt {opt.full_ckpt} -> FusionPipeline backbone')

    # ── train/eval helpers ────────────────────────────────────────────────────

    def train(self):
        self.netT.train()         # only adapters; netInst/netFusion stay eval (handled internally)

    def eval(self):
        self.netT.eval()
        self.netInst.eval()
        self.netFusion.eval()

    def set_epoch(self, epoch):
        self._epoch = epoch

    # ── loss warmup schedule ──────────────────────────────────────────────────

    def _warmup_factor(self) -> float:
        """Cosine warmup factor in [0, 1] for λ_rank / λ_outside."""
        warm_start = self.opt.rank_warmup_epoch
        warm_len   = max(1, self.opt.rank_warmup_len)
        e = self._epoch
        if e < warm_start:
            return 0.0
        if e >= warm_start + warm_len:
            return 1.0
        t = (e - warm_start) / warm_len
        return 0.5 - 0.5 * math.cos(math.pi * t)

    # ── input ────────────────────────────────────────────────────────────────

    def set_input(self, data):
        """
        data: dict produced by TextColorCocoDataset (batch_size=1, but the
              collate fn already unwrapped the list).
        """
        self.empty_box = bool(data.get('empty_box', True))
        self.file_id = data.get('file_id', 'sample')

        full_rgb = data['full_rgb'].to(self.device)        # (1, 3, H, W)
        if full_rgb.dim() == 5:
            full_rgb = full_rgb.squeeze(0)
        lab_full = rgb2lab(full_rgb, self.opt)
        self.real_L_full  = lab_full[:, [0]]                # (1, 1, H, W)
        self.real_ab_full = lab_full[:, 1:]                 # (1, 2, H, W)

        # GT soft / hard labels at H/4 for CE / KL
        if self.isTrain:
            H, W = self.real_L_full.shape[2], self.real_L_full.shape[3]
            ab_down = F.interpolate(self.real_ab_full,
                                    size=(H // 4, W // 4),
                                    mode='bilinear', align_corners=False)
            self.gt_ab_soft = encode_ab_bins_soft(
                ab_down, self.pts_in_hull, ab_norm_val=self.opt.ab_norm)
            self.gt_ab_hard = encode_ab_bins_hard(
                ab_down, self.pts_in_hull, ab_norm_val=self.opt.ab_norm)[:, 0]  # (1, H/4, W/4)

        if not self.empty_box:
            cropped_rgb = data['cropped_rgb'].to(self.device)
            if cropped_rgb.dim() == 5:
                cropped_rgb = cropped_rgb.squeeze(0)
            lab_inst = rgb2lab(cropped_rgb, self.opt)
            self.inst_L = lab_inst[:, [0]]                  # (N, 1, H, W)

            self.class_labels = data['class_labels'].to(self.device)
            self.box_info     = data['box_info'].to(self.device)
            self.box_info_2x  = data['box_info_2x'].to(self.device)
            self.box_info_4x  = data['box_info_4x'].to(self.device)
            self.box_info_8x  = data['box_info_8x'].to(self.device)

            masks_full = data['masks_full'].to(self.device)  # (N, 1, H, W)
            masks_4x   = data['masks_4x'  ].to(self.device)  # (N, 1, H/4, W/4)
            # union over instances → (1, 1, H, W)  and  (1, 1, H/4, W/4)
            self.mask_full = masks_full.max(dim=0, keepdim=True)[0].clamp(0., 1.).squeeze(1)  # (1, H, W)
            self.mask_4x   = masks_4x.max(dim=0, keepdim=True)[0].clamp(0., 1.).squeeze(1)    # (1, H/4, W/4)

            # CLIP encode instance captions
            self.text_inst_pos = self.clip.encode(data['caption_pos'])  # (N, 512)
            self.text_inst_neg = self.clip.encode(data['caption_neg'])  # (N, 512)
            self.captions_pos = data['caption_pos']
            self.captions_neg = data['caption_neg']
        else:
            self.inst_L = None
            self.class_labels = None
            self.box_info = self.box_info_2x = self.box_info_4x = self.box_info_8x = None
            sz = self.real_L_full.shape[2]
            self.mask_full = torch.zeros(1, sz, sz, device=self.device)
            self.mask_4x   = torch.zeros(1, sz // 4, sz // 4, device=self.device)
            self.text_inst_pos = None
            self.text_inst_neg = None
            self.captions_pos = []
            self.captions_neg = []

        # CLIP encode bg captions (always one each)
        self.text_bg_pos = self.clip.encode([data['caption_bg_pos']])    # (1, 512)
        self.text_bg_neg = self.clip.encode([data['caption_bg_neg']])
        self.caption_bg_pos = data['caption_bg_pos']
        self.caption_bg_neg = data['caption_bg_neg']

    # ── forward (twice) ──────────────────────────────────────────────────────

    def _forward_once(self, text_inst, text_bg):
        box_info_list = ([self.box_info, self.box_info_2x,
                          self.box_info_4x, self.box_info_8x]
                         if not self.empty_box else None)
        return self.netT(
            self.real_L_full,
            self.inst_L,
            self.class_labels,
            box_info_list,
            text_inst,
            text_bg,
            empty_box=self.empty_box,
        )

    def forward(self):
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            out_class_pos, out_reg_pos = self._forward_once(
                self.text_inst_pos, self.text_bg_pos)
            if self.isTrain:
                out_class_neg, out_reg_neg = self._forward_once(
                    self.text_inst_neg, self.text_bg_neg)
            else:
                out_class_neg, out_reg_neg = None, None

        self.pred_class_pos = out_class_pos.float()
        self.pred_reg_pos   = out_reg_pos.float()
        self.pred_class_neg = out_class_neg.float() if out_class_neg is not None else None
        self.pred_reg_neg   = out_reg_neg.float()   if out_reg_neg   is not None else None

    # ── backward ──────────────────────────────────────────────────────────────

    def backward(self):
        opt = self.opt
        rebalance_w = self.rebalance_w
        pts = self.pts_in_hull

        # references
        gt_soft = self.gt_ab_soft                 # (1, 313, H/4, W/4)
        gt_hard = self.gt_ab_hard                 # (1, H/4, W/4)
        gt_ab   = self.real_ab_full               # (1, 2, H, W)

        # pre-compute GT ab at H/4 for huber on classification head
        H, W = gt_ab.shape[2:]
        gt_ab_4x = F.interpolate(gt_ab, size=(H // 4, W // 4),
                                 mode='bilinear', align_corners=False)

        # ── 1. global reconstruction (pos) ─────────────────────────────
        ce_map_pos_4x = _rebalanced_ce_map(
            self.pred_class_pos, gt_soft, gt_hard, rebalance_w)
        huber_pos = _huber(self.pred_reg_pos, gt_ab).mean(dim=1)   # (1, H, W)

        loss_global = (ce_map_pos_4x.mean()
                       + opt.huber_weight * huber_pos.mean())

        # ── 2. instance-mask reconstruction (pos) ──────────────────────
        m_full_b = self.mask_full   # (1, H, W)
        m_4x_b   = self.mask_4x     # (1, H/4, W/4)

        loss_inst_rec = (_masked_mean(ce_map_pos_4x, m_4x_b)
                         + opt.huber_weight * _masked_mean(huber_pos, m_full_b))

        # ── 3. ranking on 313-bin probabilities (pos < neg) ────────────
        p_pos = F.softmax(self.pred_class_pos.float(), dim=1)
        p_neg = F.softmax(self.pred_class_neg.float(), dim=1)
        D_pos = _masked_mean(_kl_soft_pred(gt_soft, p_pos), m_4x_b)
        D_neg = _masked_mean(_kl_soft_pred(gt_soft, p_neg), m_4x_b)
        loss_rank = torch.clamp(opt.rank_margin + D_pos - D_neg, min=0.0)

        # ── 4. mask-outside consistency (pos/neg vs GT) ────────────────
        out_mask_full = 1.0 - m_full_b
        out_mask_4x   = 1.0 - m_4x_b
        ce_map_neg_4x = _rebalanced_ce_map(
            self.pred_class_neg, gt_soft, gt_hard, rebalance_w)
        huber_neg = _huber(self.pred_reg_neg, gt_ab).mean(dim=1)

        L_outside = 0.25 * (
            _masked_mean(huber_pos, out_mask_full)
            + _masked_mean(huber_neg, out_mask_full)
            + _masked_mean(ce_map_pos_4x, out_mask_4x)
            + _masked_mean(ce_map_neg_4x, out_mask_4x)
        )

        # ── 5. warmup-weighted total ───────────────────────────────────
        w = self._warmup_factor()
        loss_total = (loss_global
                      + opt.lambda_inst * loss_inst_rec
                      + (w * opt.lambda_rank) * loss_rank
                      + (w * opt.lambda_outside) * L_outside)

        # cache for reporting
        self.loss_G        = loss_total
        self.loss_global   = loss_global.detach()
        self.loss_inst_rec = loss_inst_rec.detach()
        self.loss_rank     = loss_rank.detach()
        self.loss_outside  = L_outside.detach()
        self.warmup        = w

    # ── BN snapshot/restore (defensive, mirror Phase 2 fusion) ───────

    def _snapshot_bn_stats(self):
        snap = {}
        for owner_name, owner in (('inst', self.netInst), ('fusion', self.netFusion)):
            for name, m in owner.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    snap[f'{owner_name}.{name}'] = (
                        m.running_mean.clone(), m.running_var.clone())
        return snap

    def _restore_bn_stats(self, snap):
        for owner_name, owner in (('inst', self.netInst), ('fusion', self.netFusion)):
            for name, m in owner.named_modules():
                key = f'{owner_name}.{name}'
                if isinstance(m, nn.BatchNorm2d) and key in snap:
                    m.running_mean.copy_(snap[key][0])
                    m.running_var.copy_(snap[key][1])

    def optimize_parameters(self):
        bn_snap = self._snapshot_bn_stats()
        self.forward()
        self.backward()
        if torch.isnan(self.loss_G) or torch.isinf(self.loss_G):
            self._restore_bn_stats(bn_snap)
            self.optimizer.zero_grad()
            self._nan_count = getattr(self, '_nan_count', 0) + 1
            # advance the AMP scale only if the scaler has actually been
            # used (its internal _scale gets lazily created by scaler.scale)
            if self.scaler.is_enabled() and self.scaler.get_scale() is not None:
                try:
                    self.scaler.update()
                except (AssertionError, RuntimeError):
                    # scaler still unused — nothing to advance
                    pass
            return
        self._ok_count = getattr(self, '_ok_count', 0) + 1
        self.optimizer.zero_grad()
        self.scaler.scale(self.loss_G).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.netT.get_trainable_params(),
                                       max_norm=5.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    # ── reporting ────────────────────────────────────────────────────────────

    def get_current_losses(self):
        ok  = getattr(self, '_ok_count', 0)
        nan = getattr(self, '_nan_count', 0)
        nan_rate = nan / max(1, ok + nan)
        return {
            'G':        self.loss_G.detach().item(),
            'global':   self.loss_global.item(),
            'inst_rec': self.loss_inst_rec.item(),
            'rank':     self.loss_rank.item(),
            'outside':  self.loss_outside.item(),
            'warmup':   self.warmup,
            'nan%':     nan_rate * 100.0,
        }

    def get_current_visuals(self):
        with torch.no_grad():
            pred_ab = decode_zhang2016_annealed_mean(
                self.pred_class_pos, self.pts_in_hull,
                T=self.opt.T, ab_norm_val=self.opt.ab_norm)
            H, W = self.real_L_full.shape[2:]
            pred_ab_up = F.interpolate(pred_ab, size=(H, W),
                                       mode='bilinear', align_corners=False)
            fake_lab_pos = torch.cat([self.real_L_full, pred_ab_up], dim=1)
            fake_rgb_pos = lab2rgb(fake_lab_pos, self.opt).clamp(0, 1)

            real_lab = torch.cat([self.real_L_full, self.real_ab_full], dim=1)
            real_rgb = lab2rgb(real_lab, self.opt).clamp(0, 1)

            gray = (self.real_L_full * self.opt.l_norm + self.opt.l_cent) / 100.
            gray = gray.expand(-1, 3, -1, -1).clamp(0, 1)

            visuals = {
                'real_gray': gray,
                'fake_rgb_pos': fake_rgb_pos,
                'real_rgb':  real_rgb,
            }

            # counter-factual visual when we have the negative output
            if self.pred_class_neg is not None:
                pred_ab_neg = decode_zhang2016_annealed_mean(
                    self.pred_class_neg, self.pts_in_hull,
                    T=self.opt.T, ab_norm_val=self.opt.ab_norm)
                pred_ab_neg_up = F.interpolate(pred_ab_neg, size=(H, W),
                                               mode='bilinear', align_corners=False)
                fake_lab_neg = torch.cat([self.real_L_full, pred_ab_neg_up], dim=1)
                visuals['fake_rgb_neg'] = lab2rgb(fake_lab_neg, self.opt).clamp(0, 1)
                visuals['diff_pos_neg'] = (fake_rgb_pos - visuals['fake_rgb_neg']).abs()

        return visuals

    # ── checkpoint I/O (only the trainable adapters) ────────────────────────

    def save_networks(self, epoch):
        """Save only the adapter; Phase-2 weights are external and frozen.
        v2: bg_adapter removed — instance adapter only."""
        sd = {
            'inst_adapter': self.netT.inst_adapter.state_dict(),
        }
        path = os.path.join(self.save_dir, f'{epoch}_net_T.pth')
        torch.save(sd, path)
        latest = os.path.join(self.save_dir, 'latest_net_T.pth')
        torch.save(sd, latest)

    def load_networks(self, epoch):
        path = os.path.join(self.save_dir, f'{epoch}_net_T.pth')
        if not os.path.isfile(path):
            path = os.path.join(self.save_dir, 'latest_net_T.pth')
        if not os.path.isfile(path):
            alt = getattr(self.opt, 'adapter_ckpt', '')
            if alt and os.path.isfile(alt):
                path = alt
        if os.path.isfile(path):
            sd = torch.load(path, map_location=self.device)
            # v2: only inst_adapter; silently ignore bg_adapter if present
            # (e.g. when loading a v1 ckpt for comparison — its inst_adapter
            # shape won't match anyway so the load will fail loudly there)
            self.netT.inst_adapter.load_state_dict(sd['inst_adapter'])
            if 'bg_adapter' in sd:
                print(f'[load] ignoring bg_adapter from v1-format ckpt {path}')
            print(f'[load] adapter ckpt: {path}')
        else:
            print(f'[warn] adapter ckpt not found: {path}')
