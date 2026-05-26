"""
Phase 3 — 文本引导上色训练/推理模型。

本模块实现 TextColorModel，继承 BaseModel，负责：
  - 初始化：创建 ClipTextColorGenerator，加载 Phase 2 stage-full 骨干权重并冻结，
    创建 CLIPTextEncoder（冻结），设置仅针对 cross-attention 参数的优化器
  - 数据处理：将 RGB 图像转 Lab 空间，将文本 caption 通过 CLIP 编码为 token 特征
  - 前向传播：灰度 L 通道 + 文本 token → Generator → 313-bin 分类 + ab 回归
  - 损失计算：复用 Phase 2 的 _ce_huber_loss（CE rebalanced + 10×Huber）
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
)


class TextColorModel(BaseModel):

    def name(self):
        return 'TextColorModel'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--full_ckpt', type=str, default='',
                            help='Phase 2 stage-full net_G.pth')
        parser.add_argument('--clip_arch', type=str, default='ViT-B-32')
        parser.add_argument('--num_heads', type=int, default=4)
        parser.add_argument('--rebalance_gamma', type=float, default=0.5)
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
        clip_arch = getattr(opt, 'clip_arch', 'ViT-B-32')
        self.clip_encoder = CLIPTextEncoder(
            arch=clip_arch, device=str(self.device))

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

        # CLIP encode captions
        captions = data['caption']
        if isinstance(captions, torch.Tensor):
            captions = [str(c) for c in captions]
        elif isinstance(captions, (list, tuple)):
            captions = list(captions)
        else:
            captions = [captions]
        self.text_tokens, self.padding_mask = self.clip_encoder.encode(captions)

        if self.opt.isTrain:
            self._encode_ab_targets()

    def _encode_ab_targets(self):
        H, W = self.real_L.shape[2], self.real_L.shape[3]
        ab_down = F.interpolate(self.real_ab, size=(H // 4, W // 4),
                                mode='bilinear', align_corners=False)
        self.gt_ab_soft = encode_ab_bins_soft(
            ab_down, self.pts_in_hull, ab_norm_val=self.opt.ab_norm)
        self.gt_ab_hard = encode_ab_bins_hard(
            ab_down, self.pts_in_hull,
            ab_norm_val=self.opt.ab_norm)[:, 0]

    def forward(self):
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            out_class, out_reg = self.netG(
                self.real_L, self.text_tokens, self.padding_mask)
        self.pred_class = out_class.float()
        self.pred_ab = out_reg.float()

    def backward(self):
        self.loss_G = _ce_huber_loss(
            self.pred_class, self.pred_ab,
            self.gt_ab_soft, self.gt_ab_hard,
            self.rebalance_w, self.opt.ab_norm, self.pts_in_hull)

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
        return {'G': self.loss_G.detach().item()}

    def get_current_visuals(self):
        with torch.no_grad():
            H, W = self.real_L.shape[2], self.real_L.shape[3]
            pred_ab_up = F.interpolate(self.pred_ab, size=(H, W),
                                       mode='bilinear', align_corners=False)
            fake_lab = torch.cat([self.real_L, pred_ab_up], dim=1)
            fake_rgb = lab2rgb(fake_lab, self.opt).clamp(0, 1)
            real_lab = torch.cat([self.real_L, self.real_ab], dim=1)
            real_rgb = lab2rgb(real_lab, self.opt).clamp(0, 1)
            gray = (self.real_L * self.opt.l_norm + self.opt.l_cent) / 100.
            gray = gray.expand(-1, 3, -1, -1).clamp(0, 1)
        return {'real_gray': gray, 'fake_rgb': fake_rgb, 'real_rgb': real_rgb}

    def train(self):
        self.netG.train()

    def eval(self):
        self.netG.eval()

