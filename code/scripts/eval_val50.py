"""
Evaluate Phase 2 (full-stage) on N COCO val images using classification head.

Produces per-image visualizations (gray, pred, gt) and a PSNR bar chart.

Usage (from code/):
  python scripts/eval_val50.py \
      --ckpt checkpoints/inst_full/best_net_G.pth \
      --val_dir /path/to/COCO2017/val2017 \
      --out_dir checkpoints/inst_full \
      --num_images 50 --T 0.38
"""
import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.networks import InstanceGenerator
from util.util import (
    rgb2lab, lab2rgb, load_zhang2016_ab_bins,
    decode_zhang2016_annealed_mean, save_image, tensor2im,
)
from util.metrics import compute_psnr, compute_ssim


class Opt:
    ab_norm = 110.
    l_norm = 100.
    l_cent = 50.


def collect_images(directory):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    paths = []
    for f in sorted(os.listdir(directory)):
        if os.path.splitext(f)[1].lower() in exts:
            paths.append(os.path.join(directory, f))
    return paths


def plot_metrics(image_ids, psnr_list, ssim_list, out_path):
    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(image_ids))
    bars = ax.bar(x, psnr_list, color='steelblue', alpha=0.8, width=0.8)

    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)
    ax.axhline(mean_psnr, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean PSNR = {mean_psnr:.2f} dB')

    ax.set_xlabel('Image Index')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'Val50 Evaluation  |  Mean PSNR={mean_psnr:.2f} dB  '
                 f'Mean SSIM={mean_ssim:.4f}')
    ax.set_xticks(x[::5])
    ax.set_xticklabels([str(i) for i in x[::5]])
    ax.legend(loc='lower right')
    ax.set_ylim(bottom=max(0, min(psnr_list) - 2))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved metrics plot: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True,
                        help='path to InstanceGenerator state_dict')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='COCO val2017 image folder')
    parser.add_argument('--out_dir', type=str, default='checkpoints/inst_full',
                        help='output directory (default: checkpoints/inst_full)')
    parser.add_argument('--num_images', type=int, default=50)
    parser.add_argument('--T', type=float, default=0.38,
                        help='annealed-mean temperature')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = Opt()

    # load model
    net = InstanceGenerator().to(device)
    state = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(state)
    net.eval()
    print(f'Loaded checkpoint: {args.ckpt}')

    # pick N images deterministically
    all_paths = collect_images(args.val_dir)
    if len(all_paths) == 0:
        print(f'[error] no images found in {args.val_dir}')
        return
    random.seed(args.seed)
    paths = random.sample(all_paths, min(args.num_images, len(all_paths)))
    print(f'Selected {len(paths)} images (seed={args.seed})')

    # prepare output dirs
    vis_dir = os.path.join(args.out_dir, 'vis_val50')
    os.makedirs(vis_dir, exist_ok=True)

    pts = torch.tensor(load_zhang2016_ab_bins(), dtype=torch.float32,
                       device=device)
    tfm = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])

    image_ids = []
    psnr_list = []
    ssim_list = []

    for i, path in enumerate(paths):
        image_id = os.path.splitext(os.path.basename(path))[0]
        pil_img = Image.open(path).convert('RGB')
        rgb = tfm(pil_img).unsqueeze(0).to(device)

        lab = rgb2lab(rgb, opt)
        real_L = lab[:, [0]]
        real_ab = lab[:, 1:]

        with torch.no_grad():
            out_class, _, _ = net(real_L)

        H, W = real_L.shape[2], real_L.shape[3]
        pred_ab = decode_zhang2016_annealed_mean(
            out_class, pts, T=args.T, ab_norm_val=opt.ab_norm)
        pred_ab_up = F.interpolate(pred_ab, size=(H, W),
                                   mode='bilinear', align_corners=False)

        fake_lab = torch.cat([real_L, pred_ab_up], dim=1)
        fake_rgb = lab2rgb(fake_lab, opt).clamp(0, 1)

        real_lab = torch.cat([real_L, real_ab], dim=1)
        real_rgb = lab2rgb(real_lab, opt).clamp(0, 1)

        gray = (real_L * opt.l_norm + opt.l_cent) / 100.
        gray = gray.expand(-1, 3, -1, -1).clamp(0, 1)

        save_image(tensor2im(gray),
                   os.path.join(vis_dir, f'{image_id}_gray.png'))
        save_image(tensor2im(fake_rgb),
                   os.path.join(vis_dir, f'{image_id}_pred.png'))
        save_image(tensor2im(real_rgb),
                   os.path.join(vis_dir, f'{image_id}_gt.png'))

        psnr = compute_psnr(fake_rgb, real_rgb)
        ssim = compute_ssim(fake_rgb, real_rgb)
        image_ids.append(image_id)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        if (i + 1) % 10 == 0:
            print(f'  [{i + 1}/{len(paths)}] '
                  f'PSNR={psnr:.2f}  SSIM={ssim:.4f}')

    metrics_path = os.path.join(args.out_dir, 'val50_metrics.png')
    plot_metrics(image_ids, psnr_list, ssim_list, metrics_path)

    print(f'\nMean PSNR = {np.mean(psnr_list):.2f} dB '
          f'(std {np.std(psnr_list):.2f})')
    print(f'Mean SSIM = {np.mean(ssim_list):.4f} '
          f'(std {np.std(ssim_list):.4f})')
    print(f'Per-image PNGs saved to: {vis_dir}/')


if __name__ == '__main__':
    main()
