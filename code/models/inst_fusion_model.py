"""
Phase 2 model: instance-aware colorization with FiLM conditioning.

Three-stage training controlled by --stage:
  full      – train InstanceGenerator on full images (backbone warm-up)
  instance  – train FiLMInstanceGenerator on GT-cropped instances (COCO)
  fusion    – train FusionPipeline WeightGenerators; both branches frozen
"""
import os
import torch
import torch.nn.functional as F

from .base_model import BaseModel
from .networks import InstanceGenerator, FiLMInstanceGenerator, FusionPipeline
from util.util import (
    rgb2lab, lab2rgb,
    load_zhang2016_ab_bins,
    build_zhang2016_rebalance_weights,
    encode_ab_bins_soft,
    encode_ab_bins_hard,
)


# ── shared loss helper ────────────────────────────────────────────────────────

def _ce_huber_loss(logits_class, pred_ab, gt_ab_soft, gt_ab_hard, rebalance_w,
                   ab_norm, pts_in_hull):
    """
    Combined CE (rebalanced) + 10× Huber loss used in Stage 1 & 2.

    logits_class: (N, 313, H/4, W/4)
    pred_ab:      (N, 2,   H,   W)   – Tanh output [-1,1]
    gt_ab_soft:   (N, 313, H/4, W/4) – soft probability targets
    gt_ab_hard:   (N,      H/4, W/4) – nearest-bin indices
    rebalance_w:  (313,)             – per-bin weights
    ab_norm:      scalar normalisation constant
    pts_in_hull:  (313, 2) Tensor
    """
    # CE
    logits_class = logits_class.float().clamp(-100., 100.)
    log_p = F.log_softmax(logits_class, dim=1).clamp(min=-100.)
    ce = -(gt_ab_soft * log_p).sum(dim=1)                 # (N, H/4, W/4)
    pw = rebalance_w[gt_ab_hard]                           # (N, H/4, W/4)
    loss_ce = (ce * pw).mean()

    # Huber on full-resolution ab
    H, W = pred_ab.shape[2:]
    gt_ab_full = F.interpolate(
        # gt_ab_hard → (N,1,H/4,W/4) → upsample, but we need ab from gt soft
        # We reconstruct gt_ab from the soft targets' argmax or use bilinear upsample
        # of the raw ab: pass it via the caller. Here we accept gt_ab_soft at H/4
        # and the caller provides pred_ab at H → we downscale pred_ab to H/4 for Huber.
        pred_ab, size=(H // 4, W // 4), mode='bilinear', align_corners=False)

    # decode gt_ab at H/4 from hard indices for Huber
    pts = pts_in_hull.to(pred_ab.device)                   # (313, 2)
    hard_flat = gt_ab_hard.reshape(-1)                     # (N*H4*W4,)
    gt_ab_pts = pts[hard_flat].reshape(
        gt_ab_hard.shape[0], gt_ab_hard.shape[1], gt_ab_hard.shape[2], 2
    ).permute(0, 3, 1, 2) / ab_norm                        # (N,2,H/4,W/4)

    diff = (gt_ab_full - gt_ab_pts).abs()
    delta = 0.01
    loss_huber = torch.where(diff < delta,
                             0.5 * diff ** 2 / delta,
                             diff - 0.5 * delta).mean()

    return loss_ce + 10.0 * loss_huber


class InstFusionModel(BaseModel):

    def name(self):
        return 'InstFusionModel'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_classes', type=int, default=91,
                            help='FiLM label vocabulary size (91 for COCO)')
        parser.add_argument('--embed_dim', type=int, default=64,
                            help='FiLM class embedding dimension')
        parser.add_argument('--full_ckpt', type=str, default='',
                            help='path to stage-full net_G.pth, used to init stage-instance')
        parser.add_argument('--inst_ckpt', type=str, default='',
                            help='path to stage-instance net_G.pth, used to init stage-fusion')
        parser.add_argument('--lr_backbone', type=float, default=5e-5,
                            help='lr for backbone in stage-instance (FiLM layers use --lr)')
        parser.add_argument('--box_num', type=int, default=8,
                            help='max instances per image in fusion stage')
        return parser

    # ── initialise ────────────────────────────────────────────────────────────

    def initialize(self, opt):
        super().initialize(opt)
        self.stage = opt.stage
        self.model_names = ['G']

        # 313-bin ab centres
        pts = load_zhang2016_ab_bins()
        self.pts_in_hull = torch.tensor(pts, dtype=torch.float32, device=self.device)

        # AMP
        self._use_amp = (len(opt.gpu_ids) > 0 and torch.cuda.is_available())
        self.scaler = torch.amp.GradScaler('cuda', enabled=self._use_amp)

        if self.stage == 'full':
            self._init_full(opt)
        elif self.stage == 'instance':
            self._init_instance(opt)
        elif self.stage == 'fusion':
            self._init_fusion(opt)
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

    def _init_full(self, opt):
        self.netG = InstanceGenerator().to(self.device)
        self._load_phase1_weights(opt)

        if self.isTrain:
            self.rebalance_w = build_zhang2016_rebalance_weights(
                gamma=getattr(opt, 'rebalance_gamma', 0.5), device=self.device)
            self.optimizer = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]
            self.setup_schedulers()

    def _init_instance(self, opt):
        self.netG = FiLMInstanceGenerator(
            num_classes=opt.num_classes, embed_dim=opt.embed_dim
        ).to(self.device)

        # load full-stage backbone weights
        if opt.full_ckpt and os.path.isfile(opt.full_ckpt):
            state = torch.load(opt.full_ckpt, map_location=self.device)
            missing, unexpected = self.netG.load_state_dict(state, strict=False)
            print(f'[inst] Loaded full-stage weights: '
                  f'{len(missing)} missing, {len(unexpected)} unexpected keys')
        else:
            print('[inst] Warning: --full_ckpt not provided; random init')

        if self.isTrain:
            self.rebalance_w = build_zhang2016_rebalance_weights(
                gamma=getattr(opt, 'rebalance_gamma', 0.5), device=self.device)
            # two-group lr: lower for backbone, higher for FiLM
            self.optimizer = torch.optim.Adam([
                {'params': self.netG.get_backbone_params(), 'lr': opt.lr_backbone},
                {'params': self.netG.get_film_params(),     'lr': opt.lr},
            ], betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]
            self.setup_schedulers()

    def _init_fusion(self, opt):
        self.netG = FusionPipeline().to(self.device)

        # load full-stage backbone weights into FusionPipeline
        if opt.full_ckpt and os.path.isfile(opt.full_ckpt):
            state = torch.load(opt.full_ckpt, map_location=self.device)
            self.netG.load_backbone_weights(state)
            print(f'[fusion] Loaded backbone weights from {opt.full_ckpt}')
        else:
            print('[fusion] Warning: --full_ckpt not provided for fusion stage')

        # load instance network separately (stored on model for forward)
        self.netInst = FiLMInstanceGenerator(
            num_classes=opt.num_classes, embed_dim=opt.embed_dim
        ).to(self.device)
        if opt.inst_ckpt and os.path.isfile(opt.inst_ckpt):
            state = torch.load(opt.inst_ckpt, map_location=self.device)
            self.netInst.load_state_dict(state)
            print(f'[fusion] Loaded instance-net weights from {opt.inst_ckpt}')
        else:
            print('[fusion] Warning: --inst_ckpt not provided for fusion stage')

        self.netG.freeze_backbone()
        for p in self.netInst.parameters():
            p.requires_grad_(False)
        self.netInst.eval()

        if self.isTrain:
            self.optimizer = torch.optim.Adam(
                self.netG.get_trainable_params(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]
            self.setup_schedulers()

    def _load_phase1_weights(self, opt):
        """Try to load Phase 1 weights into InstanceGenerator (partial match)."""
        p1_path = os.path.join(opt.checkpoints_dir, 'cnn_color', 'latest_net_G.pth')
        if os.path.isfile(p1_path):
            state = torch.load(p1_path, map_location=self.device)
            missing, unexpected = self.netG.load_state_dict(state, strict=False)
            print(f'[full] Loaded Phase-1 weights: '
                  f'{len(missing)} missing, {len(unexpected)} unexpected keys')
        else:
            print(f'[full] Phase-1 checkpoint not found at {p1_path}; random init')

    # ── input loading ─────────────────────────────────────────────────────────

    def set_input(self, data):
        if self.stage == 'full':
            self._set_input_full(data)
        elif self.stage == 'instance':
            self._set_input_instance(data)
        elif self.stage == 'fusion':
            self._set_input_fusion(data)

    def _set_input_full(self, data):
        rgb = data['rgb_img'].to(self.device)
        lab = rgb2lab(rgb, self.opt)
        self.real_L  = lab[:, [0]]
        self.real_ab = lab[:, 1:]
        if self.isTrain:
            self._encode_ab_targets()

    def _set_input_instance(self, data):
        """
        data from InstanceDataset (COCO) must contain:
          'rgb_img'     (N,3,H,W)
          'class_label' (N,)  int64 GT COCO category id
        """
        rgb = data['rgb_img'].to(self.device)
        lab = rgb2lab(rgb, self.opt)
        self.real_L      = lab[:, [0]]
        self.real_ab     = lab[:, 1:]
        self.class_label = data['class_label'].to(self.device)  # (N,)
        if self.isTrain:
            self._encode_ab_targets()

    def _set_input_fusion(self, data):
        """
        data from FusionDataset (COCO) must contain:
          'full_rgb'      (1,3,H,W)
          'class_labels'  (N,)   int64 labels per detected instance
          'cropped_rgb'   (N,3,H,W)
          'box_info'      (N,6) at H
          'box_info_2x'   (N,6) at H/2
          'box_info_4x'   (N,6) at H/4
          'box_info_8x'   (N,6) at H/8
          'empty_box'     bool
        """
        rgb_full = data['full_rgb'].to(self.device)     # (1,3,H,W) or (N_batch,...)
        # handle batch dim from DataLoader
        if rgb_full.dim() == 5:
            rgb_full = rgb_full.squeeze(1)

        lab_full = rgb2lab(rgb_full, self.opt)
        self.real_L_full  = lab_full[:, [0]]             # (1,1,H,W)
        self.real_ab_full = lab_full[:, 1:]              # (1,2,H,W)

        self.empty_box = bool(data.get('empty_box', True))

        if not self.empty_box:
            cropped_rgb = data['cropped_rgb'].to(self.device)  # (1,N,3,H,W) or (N,3,H,W)
            if cropped_rgb.dim() == 5:
                cropped_rgb = cropped_rgb.squeeze(0)           # → (N,3,H,W)

            lab_inst = rgb2lab(cropped_rgb, self.opt)
            self.inst_L = lab_inst[:, [0]]                     # (N,1,H,W)

            # DataLoader wraps each tensor in a batch dim → squeeze it off
            self.class_labels = data['class_labels'].to(self.device).squeeze(0)  # (N,)
            self.box_info     = data['box_info'].to(self.device).squeeze(0)      # (N,6)
            self.box_info_2x  = data['box_info_2x'].to(self.device).squeeze(0)
            self.box_info_4x  = data['box_info_4x'].to(self.device).squeeze(0)
            self.box_info_8x  = data['box_info_8x'].to(self.device).squeeze(0)

    def _encode_ab_targets(self):
        H, W = self.real_L.shape[2], self.real_L.shape[3]
        ab_down = F.interpolate(self.real_ab, size=(H // 4, W // 4),
                                mode='bilinear', align_corners=False)
        self.gt_ab_soft = encode_ab_bins_soft(
            ab_down, self.pts_in_hull, ab_norm_val=self.opt.ab_norm)   # (N,313,H/4,W/4)
        self.gt_ab_hard = encode_ab_bins_hard(
            ab_down, self.pts_in_hull, ab_norm_val=self.opt.ab_norm)[:, 0]  # (N,H/4,W/4)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self):
        if self.stage == 'full':
            self._forward_full()
        elif self.stage == 'instance':
            self._forward_instance()
        elif self.stage == 'fusion':
            self._forward_fusion()

    def _forward_full(self):
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            out_class, out_reg, _ = self.netG(self.real_L)
        self.pred_class = out_class.float()
        self.pred_ab    = out_reg.float()

    def _forward_instance(self):
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            out_class, out_reg, _ = self.netG(self.real_L, self.class_label)
        self.pred_class = out_class.float()
        self.pred_ab    = out_reg.float()

    def _forward_fusion(self):
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            if not self.empty_box:
                # batched forward over all instances at once — one GPU kernel launch
                _, _, fm_batch = self.netInst(self.inst_L, self.class_labels)
                # split batch dim back into per-instance dicts for WeightGenerator
                N = self.inst_L.shape[0]
                inst_feats = [
                    {k: fm_batch[k][[i]].detach() for k in fm_batch}
                    for i in range(N)
                ]
                box_info_list = [self.box_info, self.box_info_2x,
                                 self.box_info_4x, self.box_info_8x]
            else:
                inst_feats = []
                box_info_list = []

            fused_ab = self.netG(self.real_L_full, inst_feats, box_info_list)

        self.pred_ab_fused = fused_ab.float()

    # ── backward ──────────────────────────────────────────────────────────────

    def backward(self):
        if self.stage in ('full', 'instance'):
            self.loss_G = _ce_huber_loss(
                self.pred_class, self.pred_ab,
                self.gt_ab_soft, self.gt_ab_hard,
                self.rebalance_w, self.opt.ab_norm, self.pts_in_hull)
        else:  # fusion
            H, W = self.real_ab_full.shape[2:]
            pred_down = F.interpolate(self.pred_ab_fused, size=(H // 4, W // 4),
                                      mode='bilinear', align_corners=False)
            gt_down   = F.interpolate(self.real_ab_full,  size=(H // 4, W // 4),
                                      mode='bilinear', align_corners=False)
            diff = (pred_down - gt_down).abs()
            delta = 0.01
            self.loss_G = torch.where(diff < delta,
                                      0.5 * diff ** 2 / delta,
                                      diff - 0.5 * delta).mean()

    def optimize_parameters(self):
        self.forward()
        # compute loss without backward first, skip bad batch entirely
        self.backward()
        if torch.isnan(self.loss_G) or torch.isinf(self.loss_G):
            self.optimizer.zero_grad()
            return
        self.optimizer.zero_grad()
        self.scaler.scale(self.loss_G).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=5.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    # ── reporting ─────────────────────────────────────────────────────────────

    def get_current_losses(self):
        return {'G': self.loss_G.detach().item()}

    def get_current_visuals(self):
        with torch.no_grad():
            if self.stage == 'fusion':
                real_L  = self.real_L_full
                real_ab = self.real_ab_full
                pred_ab = self.pred_ab_fused
            else:
                real_L  = self.real_L
                real_ab = self.real_ab
                pred_ab = self.pred_ab

            H, W = real_L.shape[2], real_L.shape[3]
            pred_ab_up = F.interpolate(pred_ab, size=(H, W),
                                       mode='bilinear', align_corners=False)

            fake_lab = torch.cat([real_L, pred_ab_up], dim=1)
            fake_rgb = lab2rgb(fake_lab, self.opt).clamp(0, 1)

            real_lab = torch.cat([real_L, real_ab], dim=1)
            real_rgb = lab2rgb(real_lab, self.opt).clamp(0, 1)

            gray = (real_L * self.opt.l_norm + self.opt.l_cent) / 100.
            gray = gray.expand(-1, 3, -1, -1).clamp(0, 1)

        return {
            'real_gray': gray,
            'fake_rgb':  fake_rgb,
            'real_rgb':  real_rgb,
        }

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def save_networks(self, epoch):
        # netG is always saved
        super().save_networks(epoch)
        # also export a bare state_dict file for easy loading by later stages
        if self.stage in ('full', 'instance'):
            bare_path = os.path.join(self.save_dir, 'net_G.pth')
            torch.save(self.netG.state_dict(), bare_path)

    def train(self):
        self.netG.train()
        if self.stage == 'fusion':
            self.netInst.eval()  # keep frozen

    def eval(self):
        self.netG.eval()
        if self.stage == 'fusion':
            self.netInst.eval()
