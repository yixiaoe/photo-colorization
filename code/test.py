"""Inference entry point.

Usage:
  # Phase 1
  python test.py --method cnn_color --test_img_dir data/test

  # Phase 2
  python test.py --method inst_fusion --test_img_dir data/test

  # Phase 3 Bonus (exemplar)
  python test.py --method cnn_color --exemplar --ref_img ref.jpg --test_img_dir data/test
"""
import os
import numpy as np
import torch

from util.check_deps import ensure_requirements
ensure_requirements()

from options.train_options import TestOptions
from data_process.colorization_dataset import create_dataset
from models import create_model
from util.util import save_image, tensor2im, lab2rgb, rgb2lab


def _ab_histogram(ab_tensor, bins=32, ab_range=(-1, 1)):
    """Compute 2-D ab histogram from (1, 2, H, W) tensor. Returns flat (bins*bins,) array."""
    a = ab_tensor[0, 0].cpu().numpy().flatten()
    b = ab_tensor[0, 1].cpu().numpy().flatten()
    h, _, _ = np.histogram2d(a, b, bins=bins, range=[ab_range, ab_range])
    h = h / (h.sum() + 1e-8)
    return h.flatten()


def bhattacharyya_distance(p, q):
    """Bhattacharyya distance between two normalised histograms."""
    bc = np.sum(np.sqrt(p * q + 1e-10))
    return -np.log(bc + 1e-10)


def main():
    opt = TestOptions().parse()
    opt.isTrain = False

    dataset = create_dataset(opt, split='test')
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=opt.nThreads)

    opt.model = opt.method
    model = create_model(opt)
    model.eval()
    model.load_networks(opt.which_epoch)

    os.makedirs(opt.results_img_dir, exist_ok=True)

    bd_scores = []

    for i, data in enumerate(loader):
        if i >= opt.how_many:
            break

        model.set_input(data)
        with torch.no_grad():
            model.forward()

        visuals = model.get_current_visuals()
        file_id = data.get('file_id', [f'{i:05d}'])[0]

        for name, img_tensor in visuals.items():
            arr  = tensor2im(img_tensor)
            path = os.path.join(opt.results_img_dir, f'{file_id}_{name}.png')
            save_image(arr, path)

        # Bhattacharyya distance on ab histograms
        if 'fake_rgb' in visuals and 'real_rgb' in visuals:
            fake_lab = rgb2lab(visuals['fake_rgb'], opt)
            real_lab = rgb2lab(visuals['real_rgb'], opt)
            h_fake = _ab_histogram(fake_lab[:, 1:])
            h_real = _ab_histogram(real_lab[:, 1:])
            bd = bhattacharyya_distance(h_fake, h_real)
            bd_scores.append(bd)

        if (i + 1) % 10 == 0:
            n = min(len(dataset), opt.how_many)
            print(f'Processed {i + 1} / {n}')

    if bd_scores:
        mean_bd = float(np.mean(bd_scores))
        print(f'\nBhattacharyya distance (ab histogram)  mean = {mean_bd:.4f}  '
              f'(lower → more similar colour distribution)')

    print('Inference complete. Results saved to', opt.results_img_dir)


if __name__ == '__main__':
    main()
