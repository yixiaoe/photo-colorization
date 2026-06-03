"""Inference entry point.

Usage:
  # Phase 1
  python test.py --method cnn_color --test_img_dir data/test

  # Phase 2
  python test.py --method inst_fusion --test_img_dir data/test

  # Phase 3 (text_color)
  python test.py --method text_color \
      --image dog_on_grass.jpg \
      --prompt "inst:0=a black labrador" \
      --prompt "inst:1=green grass" \
      --prompt "bg=sunset sky" \
      --adapter_ckpt checkpoints/phase3_text_color/latest_net_T.pth \
      --full_ckpt   checkpoints/inst_fusion_full/80_net_G.pth \
      --inst_ckpt   checkpoints/inst_fusion_instance/25_net_G.pth \
      --fusion_ckpt checkpoints/inst_fusion_fusion/25_net_G.pth

  # legacy: --exemplar bonus
  python test.py --method cnn_color --exemplar --ref_img ref.jpg --test_img_dir data/test
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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


def _test_instance_stage(opt, model, dataset, loader):
    """Per-instance colorization test for stage=instance."""
    os.makedirs(opt.results_img_dir, exist_ok=True)
    total_inst = 0

    for i, data in enumerate(loader):
        if i >= opt.how_many:
            break

        file_id = data.get('file_id', [f'{i:05d}'])[0]

        if data.get('empty_box', True):
            print(f'[{file_id}] no instances detected, skipping')
            continue

        cropped = data['cropped_img']
        if cropped.dim() == 5:
            cropped = cropped.squeeze(0)
        labels = data['class_labels'].squeeze(0)
        N = cropped.shape[0]

        for j in range(N):
            item = {
                'rgb_img':     cropped[j:j+1],
                'class_label': labels[j:j+1],
            }
            model.set_input(item)
            with torch.no_grad():
                model.forward()

            visuals = model.get_current_visuals()
            for name, img_tensor in visuals.items():
                arr  = tensor2im(img_tensor)
                path = os.path.join(opt.results_img_dir,
                                    f'{file_id}_inst{j}_{name}.png')
                save_image(arr, path)
            total_inst += 1

        if (i + 1) % 10 == 0:
            n = min(len(dataset), opt.how_many)
            print(f'Processed {i + 1} / {n} images ({total_inst} instances)')

    print(f'\nInstance test complete. {total_inst} instances saved to '
          f'{opt.results_img_dir}')


# ── Phase 3 text-color inference ─────────────────────────────────────────────

def _parse_prompts(prompt_list):
    inst_prompts, bg_prompt = {}, None
    for p in prompt_list:
        if '=' not in p:
            print(f'[warn] ignoring malformed --prompt {p!r} (need "key=value")')
            continue
        key, val = p.split('=', 1)
        key = key.strip()
        if key.startswith('inst:'):
            try:
                idx = int(key.split(':', 1)[1])
                inst_prompts[idx] = val.strip()
            except ValueError:
                print(f'[warn] ignoring bad instance prompt key {key!r}')
        elif key == 'bg':
            bg_prompt = val.strip()
        else:
            print(f'[warn] unknown prompt key {key!r}')
    return inst_prompts, bg_prompt


def _load_image_paths(opt):
    if opt.image:
        return [opt.image]
    if opt.test_img_dir and os.path.isdir(opt.test_img_dir):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        return sorted([os.path.join(opt.test_img_dir, f)
                       for f in os.listdir(opt.test_img_dir)
                       if os.path.splitext(f)[1].lower() in exts])[:opt.how_many]
    print('[error] either --image or --test_img_dir is required')
    sys.exit(1)


def _test_text_color(opt, model):
    from util.maskrcnn_helper import predict_with_masks, class_name
    from data_process.colorization_dataset import get_box_info
    from util.util import decode_zhang2016_annealed_mean

    inst_prompts, bg_prompt = _parse_prompts(opt.prompt)
    device = model.device
    sz = opt.fineSize
    os.makedirs(opt.results_img_dir, exist_ok=True)

    for img_path in _load_image_paths(opt):
        file_id = os.path.splitext(os.path.basename(img_path))[0]
        pil = Image.open(img_path).convert('RGB')
        orig_W, orig_H = pil.size

        # Mask R-CNN detection
        detections = predict_with_masks(
            pil, device, box_num=opt.box_num, score_thresh=opt.score_thresh)
        print(f'\n[{file_id}] detected {len(detections)} instances:')
        for i, d in enumerate(detections):
            cls = class_name(d['label'])
            fallback = f'a {cls}'
            used = inst_prompts.get(i, fallback)
            print(f'  inst:{i:2d} {cls:<14} score={d["score"]:.2f}  prompt="{used}"')
        if bg_prompt is not None:
            print(f'  bg              prompt="{bg_prompt}"')

        # Build pipeline inputs
        pil_resized = pil.resize((sz, sz), resample=Image.BILINEAR)
        full_rgb = torch.from_numpy(
            np.asarray(pil_resized)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        full_rgb = full_rgb.to(device)
        lab_full = rgb2lab(full_rgb, opt)
        real_L_full  = lab_full[:, [0]]

        empty_box = len(detections) == 0
        if not empty_box:
            inst_crops, class_labels = [], []
            n = len(detections)
            box_info    = np.zeros((n, 6), dtype=np.int64)
            box_info_2x = np.zeros_like(box_info)
            box_info_4x = np.zeros_like(box_info)
            box_info_8x = np.zeros_like(box_info)
            captions = []
            for i, d in enumerate(detections):
                x0, y0, x1, y1 = d['box']
                x0 = max(0, x0); y0 = max(0, y0)
                x1 = min(orig_W, x1); y1 = min(orig_H, y1)
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                crop = pil.crop((x0, y0, x1, y1)).resize((sz, sz), resample=Image.BILINEAR)
                inst_crops.append(
                    torch.from_numpy(np.asarray(crop)).permute(2, 0, 1).float() / 255.0)
                class_labels.append(d['label'])
                box_info[i]    = get_box_info((x0, y0, x1, y1), pil.size, sz)
                box_info_2x[i] = get_box_info((x0, y0, x1, y1), pil.size, sz // 2)
                box_info_4x[i] = get_box_info((x0, y0, x1, y1), pil.size, sz // 4)
                box_info_8x[i] = get_box_info((x0, y0, x1, y1), pil.size, sz // 8)
                captions.append(inst_prompts.get(i, f'a {class_name(d["label"])}'))

            cropped_rgb = torch.stack(inst_crops).to(device)
            lab_inst = rgb2lab(cropped_rgb, opt)
            inst_L = lab_inst[:, [0]]
            class_labels_t = torch.tensor(class_labels, dtype=torch.long, device=device)
            box_info_list = [
                torch.from_numpy(box_info).to(device),
                torch.from_numpy(box_info_2x).to(device),
                torch.from_numpy(box_info_4x).to(device),
                torch.from_numpy(box_info_8x).to(device),
            ]
            text_inst = model.clip.encode(captions)
        else:
            inst_L = None
            class_labels_t = None
            box_info_list = None
            text_inst = None

        text_bg = model.clip.encode([bg_prompt or ''])

        with torch.no_grad():
            out_class, _ = model.netT(
                real_L_full, inst_L, class_labels_t, box_info_list,
                text_inst, text_bg, empty_box=empty_box,
            )

        pred_ab = decode_zhang2016_annealed_mean(
            out_class.float(), model.pts_in_hull, T=opt.T, ab_norm_val=opt.ab_norm)
        pred_ab_up = F.interpolate(pred_ab, size=(sz, sz),
                                   mode='bilinear', align_corners=False)
        fake_lab = torch.cat([real_L_full, pred_ab_up], dim=1)
        fake_rgb = lab2rgb(fake_lab, opt).clamp(0, 1)
        fake_rgb_full = F.interpolate(fake_rgb, size=(orig_H, orig_W),
                                      mode='bilinear', align_corners=False)
        save_image(tensor2im(fake_rgb_full),
                   os.path.join(opt.results_img_dir, f'{file_id}_fake.png'))
        print(f'  -> {opt.results_img_dir}/{file_id}_fake.png')


def main():
    opt = TestOptions().parse()
    opt.isTrain = False
    opt.model = opt.method

    model = create_model(opt)
    model.eval()
    model.load_networks(opt.which_epoch)

    if opt.method == 'text_color':
        _test_text_color(opt, model)
        return

    dataset = create_dataset(opt, split='test')
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=opt.nThreads)

    if opt.method == 'inst_fusion' and getattr(opt, 'stage', 'fusion') == 'instance':
        _test_instance_stage(opt, model, dataset, loader)
        return

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

        # upsample outputs to original image resolution if available
        orig_size = data.get('orig_size')
        if orig_size is not None:
            orig_H, orig_W = int(orig_size[0, 0]), int(orig_size[0, 1])
            visuals = {
                k: F.interpolate(v, size=(orig_H, orig_W),
                                 mode='bilinear', align_corners=False)
                for k, v in visuals.items()
            }

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
