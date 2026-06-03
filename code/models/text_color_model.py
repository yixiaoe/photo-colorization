"""
Phase 3 — 文本引导上色训练/推理模型。

本模块实现 TextColorModel，继承 BaseModel，负责：
  - 初始化：创建 ClipTextColorGenerator，加载 Phase 2 stage-full 骨干权重并冻结，
    创建 CLIPTextEncoder（冻结），设置仅针对 cross-attention 参数的优化器
  - 数据处理：将 RGB 图像转 Lab 空间，将文本 caption 通过 CLIP 编码为 token 特征
  - 前向传播：灰度 L 通道 + 文本 token → Generator → 313-bin 分类 + ab 回归
  - 损失计算：复用 Phase 2 的 _ce_huber_loss（CE rebalanced + weighted Huber）
  - 可视化：输出灰度图、预测彩色图、真实彩色图的对比

训练策略：
  - 冻结 CLIP + InstanceGenerator backbone（~26M 参数）
  - 仅训练 2 个 TextCrossAttentionBlock（~609K 参数）
  - 使用 AMP 混合精度 + 梯度裁剪（max_norm=5.0）

约束：不修改任何 Phase 1/Phase 2 文件，通过导入复用已有组件。
"""
import os
import torch
import torch.nn.functional as F

from models.base_model import BaseModel
from models.text_color_networks import ClipTextColorGenerator
from models.inst_fusion_model import _ce_huber_loss
from util.clip_encoder import CLIPTextEncoder
from util.util import (
    rgb2lab,
    lab2rgb,
    load_zhang2016_ab_bins,
    build_zhang2016_rebalance_weights,
    encode_ab_bins_soft,
    encode_ab_bins_hard,
    decode_zhang2016_annealed_mean,
)


def _masked_mean(values, mask, eps=1e-6):
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(1)
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=eps)


def _masked_ab_huber_loss(pred_ab, target_ab, mask, delta=0.01):
    """Huber loss averaged only over pixels selected by mask."""
    diff = (pred_ab.float() - target_ab.float()).abs()
    huber = torch.where(diff < delta,
                        0.5 * diff ** 2 / delta,
                        diff - 0.5 * delta)
    huber = huber.mean(dim=1, keepdim=True)
    return _masked_mean(huber, mask)


def _masked_ce_loss(logits_class, gt_ab_soft, gt_ab_hard,
                    rebalance_w, mask_4x):
    logits_class = logits_class.float().clamp(-100., 100.)
    log_p = F.log_softmax(logits_class, dim=1).clamp(min=-100.)
    ce = -(gt_ab_soft * log_p).sum(dim=1, keepdim=True)
    pw = rebalance_w[gt_ab_hard].unsqueeze(1)
    return _masked_mean(ce * pw, mask_4x)


def _instance_text_losses(pred_class_pos, pred_reg_pos,
                          pred_class_neg, pred_reg_neg,
                          gt_ab_full, gt_ab_4x,
                          gt_ab_soft, gt_ab_hard, rebalance_w,
                          mask_full, mask_4x, pts_in_hull, ab_norm,
                          huber_weight=3.0, margin=0.05):
    """Losses for color-word + object + instance-mask supervision."""
    inst_ce = _masked_ce_loss(
        pred_class_pos, gt_ab_soft, gt_ab_hard, rebalance_w, mask_4x)
    inst_huber = _masked_ab_huber_loss(pred_reg_pos, gt_ab_full, mask_full)
    inst_rec = inst_ce + huber_weight * inst_huber

    ab_cls_pos = decode_zhang2016_annealed_mean(
        pred_class_pos, pts_in_hull, T=0.38, ab_norm_val=ab_norm)
    ab_cls_neg = decode_zhang2016_annealed_mean(
        pred_class_neg, pts_in_hull, T=0.38, ab_norm_val=ab_norm)
    dist_pos = _masked_ab_huber_loss(ab_cls_pos, gt_ab_4x, mask_4x)
    dist_neg = _masked_ab_huber_loss(ab_cls_neg, gt_ab_4x, mask_4x)
    rank = torch.relu(torch.as_tensor(margin, device=dist_pos.device)
                      + dist_pos - dist_neg)

    outside_full = 1.0 - mask_full
    outside_4x = 1.0 - mask_4x
    outside_reg = _masked_ab_huber_loss(
        pred_reg_pos, pred_reg_neg, outside_full)
    outside_cls = _masked_ab_huber_loss(
        ab_cls_pos, ab_cls_neg, outside_4x)
    outside = 0.5 * (outside_reg + outside_cls)

    return {
        'inst_ce': inst_ce,
        'inst_huber': inst_huber,
        'inst_rec': inst_rec,
        'rank': rank,
        'outside': outside,
    }


class TextColorModel(BaseModel):

    def name(self):
        return 'TextColorModel'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--full_ckpt', type=str,
                            default='checkpoints/inst_full/80_net_G.pth',
                            help='Phase 2 stage-full 80_net_G.pth')
        parser.add_argument('--clip_arch', type=str,
                            default='ViT-B-32-quickgelu')
        parser.add_argument('--clip_pretrained_path', type=str,
                            default='checkpoints/clip/open_clip_model.safetensors',
                            help='local open_clip checkpoint path')
        parser.add_argument('--num_heads', type=int, default=4)
        parser.add_argument('--rebalance_gamma', type=float, default=0.5)
        parser.add_argument('--huber_weight', type=float, default=3.0)
        parser.add_argument('--lambda_inst', type=float, default=1.0)
        parser.add_argument('--lambda_rank', type=float, default=0.1)
        parser.add_argument('--lambda_outside', type=float, default=0.2)
        parser.add_argument('--rank_margin', type=float, default=0.05)
        return parser

    def initialize(self, opt):
        super().initialize(opt)
        self.model_names = ['G']

        # 313-bin ab centres
        pts = load_zhang2016_ab_bins()
        self.pts_in_hull = torch.tensor(pts, dtype=torch.float32,
                                        device=self.device)

        # AMP
        self._use_amp = (len(opt.gpu_ids) > 0 and torch.cuda.is_available())
        self.scaler = torch.amp.GradScaler('cuda', enabled=self._use_amp)

        # Generator
        num_heads = getattr(opt, 'num_heads', 4)
        self.netG = ClipTextColorGenerator(num_heads=num_heads).to(self.device)

        # Load backbone weights
        full_ckpt = getattr(opt, 'full_ckpt', '')
        if full_ckpt and os.path.isfile(full_ckpt):
            state = torch.load(full_ckpt, map_location=self.device)
            self.netG.load_backbone_weights(state)
            print(f'[TextColor] Loaded backbone from {full_ckpt}')
        else:
            print('[TextColor] Warning: --full_ckpt not provided; random init')

        self.netG.freeze_backbone()

        # CLIP text encoder
        clip_arch = getattr(opt, 'clip_arch', 'ViT-B-32-quickgelu')
        clip_pretrained_path = getattr(opt, 'clip_pretrained_path', '')
        self.clip_encoder = CLIPTextEncoder(
            arch=clip_arch, pretrained_path=clip_pretrained_path,
            device=str(self.device))

        # Optimizer (only xattn params)
        if opt.isTrain:
            self.rebalance_w = build_zhang2016_rebalance_weights(
                gamma=getattr(opt, 'rebalance_gamma', 0.5),
                device=self.device)
            self.optimizer = torch.optim.Adam(
                self.netG.get_trainable_params(),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]
            self.setup_schedulers()

    def set_input(self, data):
        rgb = data['rgb_img'].to(self.device)
        lab = rgb2lab(rgb, self.opt)
        self.real_L = lab[:, [0]]
        self.real_ab = lab[:, 1:]
        self.use_instance_prompt = (
            'caption_pos' in data and 'caption_neg' in data and
            'mask_full' in data and 'mask_4x' in data)

        if self.use_instance_prompt:
            self.mask_full = data['mask_full'].to(self.device).float()
            self.mask_4x = data['mask_4x'].to(self.device).float()
            captions_pos = self._normalise_captions(data['caption_pos'])
            captions_neg = self._normalise_captions(data['caption_neg'])
            self.text_tokens_pos, self.padding_mask_pos = (
                self.clip_encoder.encode(captions_pos))
            self.text_tokens_neg, self.padding_mask_neg = (
                self.clip_encoder.encode(captions_neg))
        else:
            captions = self._normalise_captions(data['caption'])
            self.text_tokens, self.padding_mask = (
                self.clip_encoder.encode(captions))

        if self.opt.isTrain:
            self._encode_ab_targets()

    @staticmethod
    def _normalise_captions(captions):
        if isinstance(captions, torch.Tensor):
            return [str(c) for c in captions]
        if isinstance(captions, (list, tuple)):
            return list(captions)
        return [captions]

    def _encode_ab_targets(self):
        H, W = self.real_L.shape[2], self.real_L.shape[3]
        ab_down = F.interpolate(self.real_ab, size=(H // 4, W // 4),
                                mode='bilinear', align_corners=False)
        self.real_ab_4x = ab_down
        self.gt_ab_soft = encode_ab_bins_soft(
            ab_down, self.pts_in_hull, ab_norm_val=self.opt.ab_norm)
        self.gt_ab_hard = encode_ab_bins_hard(
            ab_down, self.pts_in_hull,
            ab_norm_val=self.opt.ab_norm)[:, 0]

    def forward(self):
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            if getattr(self, 'use_instance_prompt', False):
                out_class_pos, out_reg_pos = self.netG(
                    self.real_L, self.text_tokens_pos, self.padding_mask_pos)
                out_class_neg, out_reg_neg = self.netG(
                    self.real_L, self.text_tokens_neg, self.padding_mask_neg)
                self.pred_class = out_class_pos.float()
                self.pred_ab = out_reg_pos.float()
                self.pred_class_neg = out_class_neg.float()
                self.pred_ab_neg = out_reg_neg.float()
            else:
                out_class, out_reg = self.netG(
                    self.real_L, self.text_tokens, self.padding_mask)
                self.pred_class = out_class.float()
                self.pred_ab = out_reg.float()

    def backward(self):
        self.loss_global, parts = _ce_huber_loss(
            self.pred_class, self.pred_ab,
            self.gt_ab_soft, self.gt_ab_hard,
            self.rebalance_w, self.opt.ab_norm, self.pts_in_hull,
            huber_weight=getattr(self.opt, 'huber_weight', 3.0),
            return_components=True)
        self.loss_ce = parts['ce']
        self.loss_huber = parts['huber']
        self.loss_huber_weighted = parts['huber_weighted']
        self.loss_G = self.loss_global

        if getattr(self, 'use_instance_prompt', False):
            inst_parts = _instance_text_losses(
                pred_class_pos=self.pred_class,
                pred_reg_pos=self.pred_ab,
                pred_class_neg=self.pred_class_neg,
                pred_reg_neg=self.pred_ab_neg,
                gt_ab_full=self.real_ab,
                gt_ab_4x=self.real_ab_4x,
                gt_ab_soft=self.gt_ab_soft,
                gt_ab_hard=self.gt_ab_hard,
                rebalance_w=self.rebalance_w,
                mask_full=self.mask_full,
                mask_4x=self.mask_4x,
                pts_in_hull=self.pts_in_hull,
                ab_norm=self.opt.ab_norm,
                huber_weight=getattr(self.opt, 'huber_weight', 3.0),
                margin=getattr(self.opt, 'rank_margin', 0.05))
            self.loss_inst_rec = inst_parts['inst_rec']
            self.loss_inst_ce = inst_parts['inst_ce']
            self.loss_inst_huber = inst_parts['inst_huber']
            self.loss_rank = inst_parts['rank']
            self.loss_outside = inst_parts['outside']
            self.loss_G = (
                self.loss_global
                + getattr(self.opt, 'lambda_inst', 1.0) * self.loss_inst_rec
                + getattr(self.opt, 'lambda_rank', 0.1) * self.loss_rank
                + getattr(self.opt, 'lambda_outside', 0.2) * self.loss_outside
            )

    def optimize_parameters(self):
        self.forward()
        self.backward()
        if torch.isnan(self.loss_G) or torch.isinf(self.loss_G):
            self.optimizer.zero_grad()
            return
        self.optimizer.zero_grad()
        self.scaler.scale(self.loss_G).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.netG.get_trainable_params(), max_norm=5.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def get_current_losses(self):
        losses = {
            'G': self.loss_G.detach().item(),
            'global': self.loss_global.detach().item(),
            'ce': self.loss_ce.detach().item(),
            'huber': self.loss_huber.detach().item(),
            'huber_weighted': self.loss_huber_weighted.detach().item(),
        }
        if getattr(self, 'use_instance_prompt', False):
            losses.update({
                'inst_rec': self.loss_inst_rec.detach().item(),
                'inst_ce': self.loss_inst_ce.detach().item(),
                'inst_huber': self.loss_inst_huber.detach().item(),
                'rank': self.loss_rank.detach().item(),
                'outside': self.loss_outside.detach().item(),
            })
        return losses

    def get_current_visuals(self):
        with torch.no_grad():
            H, W = self.real_L.shape[2], self.real_L.shape[3]

            pred_ab_class = decode_zhang2016_annealed_mean(
                self.pred_class, self.pts_in_hull, T=0.38,
                ab_norm_val=self.opt.ab_norm)
            pred_ab_class_up = F.interpolate(pred_ab_class, size=(H, W),
                                             mode='bilinear', align_corners=False)
            fake_lab = torch.cat([self.real_L, pred_ab_class_up], dim=1)
            fake_rgb = lab2rgb(fake_lab, self.opt).clamp(0, 1)

            pred_ab_reg = F.interpolate(self.pred_ab, size=(H, W),
                                        mode='bilinear', align_corners=False)
            fake_lab_reg = torch.cat([self.real_L, pred_ab_reg], dim=1)
            fake_rgb_reg = lab2rgb(fake_lab_reg, self.opt).clamp(0, 1)

            real_lab = torch.cat([self.real_L, self.real_ab], dim=1)
            real_rgb = lab2rgb(real_lab, self.opt).clamp(0, 1)
            gray = (self.real_L * self.opt.l_norm + self.opt.l_cent) / 100.
            gray = gray.expand(-1, 3, -1, -1).clamp(0, 1)
        return {
            'real_gray': gray,
            'fake_rgb': fake_rgb,
            'fake_rgb_reg': fake_rgb_reg,
            'real_rgb': real_rgb,
        }

    def train(self):
        self.netG.train()

    def eval(self):
        self.netG.eval()
