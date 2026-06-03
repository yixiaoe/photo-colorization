"""
Quick visualization: compare regression head vs classification head output.

Usage (from code/):
  python scripts/visualize_heads.py \
      --ckpt checkpoints/inst_full/best_net_G.pth \
      --img data/test/头像.jpg \
      --out results/head_compare/
"""
import os
import sys
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.networks import InstanceGenerator
from util.util import (
    rgb2lab, lab2rgb, load_zhang2016_ab_bins,
    decode_zhang2016_annealed_mean, save_image, tensor2im,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--out', type=str, default='results/head_compare')
    parser.add_argument('--T', type=float, default=0.38)
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    net = InstanceGenerator().to(device)
    state = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(state)
    net.eval()
    print(f'Loaded checkpoint: {args.ckpt}')

    # load and preprocess image
    pil_img = Image.open(args.img).convert('RGB')
    tfm = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    rgb = tfm(pil_img).unsqueeze(0).to(device)  # (1,3,H,W)

    # create a simple opt namespace for rgb2lab/lab2rgb
    class Opt:
        ab_norm = 110.
        l_norm = 100.
        l_cent = 50.
    opt = Opt()

    lab = rgb2lab(rgb, opt)
    real_L = lab[:, [0]]   # (1,1,H,W)
    real_ab = lab[:, 1:]   # (1,2,H,W)

    # forward
    with torch.no_grad():
        out_class, out_reg, _ = net(real_L)

    H, W = real_L.shape[2], real_L.shape[3]

    # --- regression head ---
    pred_ab_reg = F.interpolate(out_reg, size=(H, W),
                                mode='bilinear', align_corners=False)
    fake_lab_reg = torch.cat([real_L, pred_ab_reg], dim=1)
    fake_rgb_reg = lab2rgb(fake_lab_reg, opt).clamp(0, 1)

    # --- classification head (annealed mean) ---
    pts = torch.tensor(load_zhang2016_ab_bins(), dtype=torch.float32,
                       device=device)
    pred_ab_cls = decode_zhang2016_annealed_mean(
        out_class, pts, T=args.T, ab_norm_val=opt.ab_norm)
    pred_ab_cls_up = F.interpolate(pred_ab_cls, size=(H, W),
                                   mode='bilinear', align_corners=False)
    fake_lab_cls = torch.cat([real_L, pred_ab_cls_up], dim=1)
    fake_rgb_cls = lab2rgb(fake_lab_cls, opt).clamp(0, 1)

    # --- ground truth ---
    real_lab_full = torch.cat([real_L, real_ab], dim=1)
    real_rgb_out = lab2rgb(real_lab_full, opt).clamp(0, 1)

    # --- gray ---
    gray = (real_L * opt.l_norm + opt.l_cent) / 100.
    gray = gray.expand(-1, 3, -1, -1).clamp(0, 1)

    # save
    basename = os.path.splitext(os.path.basename(args.img))[0]
    save_image(tensor2im(gray), os.path.join(args.out, f'{basename}_gray.png'))
    save_image(tensor2im(fake_rgb_reg), os.path.join(args.out, f'{basename}_reg.png'))
    save_image(tensor2im(fake_rgb_cls), os.path.join(args.out, f'{basename}_cls_T{args.T}.png'))
    save_image(tensor2im(real_rgb_out), os.path.join(args.out, f'{basename}_gt.png'))

    print(f'Saved to {args.out}/')
    print(f'  {basename}_gray.png       — input grayscale')
    print(f'  {basename}_reg.png        — regression head (out_reg)')
    print(f'  {basename}_cls_T{args.T}.png — classification head (annealed mean T={args.T})')
    print(f'  {basename}_gt.png         — ground truth')


if __name__ == '__main__':
    main()
