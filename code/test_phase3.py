"""
Phase 3 — 文本引导上色推理入口。

本脚本接受用户输入的文本描述（--prompt），对测试图像进行文本控制的上色。
推理流程：
  1. 加载训练好的 TextColorModel（backbone + xattn 权重）
  2. 对每张测试图像：灰度 L 通道 + 文本 prompt → Generator → 预测 ab
  3. 合并 L + ab → RGB，保存结果图像

输出文件命名：{原文件名}_fake_rgb.png / _real_gray.png / _real_rgb.png

Usage:
  python test_phase3.py \\
      --full_ckpt checkpoints/inst_full/net_G.pth \\
      --test_img_dir data/test \\
      --prompt "a red car under blue sky" \\
      --name text_color --gpu_ids 0
"""
import os
import torch
import torch.nn.functional as F

from util.check_deps import ensure_requirements
ensure_requirements()

from options.phase3_options import Phase3TestOptions
from models.text_color_model import TextColorModel
from util.util import save_image, tensor2im, rgb2lab, lab2rgb
from PIL import Image
import torchvision.transforms as T


def _collect_images(root):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = []
    for dirpath, _, fnames in os.walk(root):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def main():
    opt = Phase3TestOptions().parse()
    opt.isTrain = False

    model = TextColorModel()
    model.initialize(opt)
    model.eval()
    model.load_networks(opt.which_epoch)

    os.makedirs(opt.results_img_dir, exist_ok=True)

    paths = _collect_images(opt.test_img_dir)
    tfm = T.Compose([
        T.Resize((opt.fineSize, opt.fineSize), interpolation=Image.BILINEAR),
        T.ToTensor(),
    ])

    prompt = opt.prompt if opt.prompt else "a colorful scene"

    for i, path in enumerate(paths):
        if i >= opt.how_many:
            break

        pil_img = Image.open(path).convert('RGB')
        rgb_tensor = tfm(pil_img).unsqueeze(0)

        data = {'rgb_img': rgb_tensor, 'caption': [prompt]}
        model.set_input(data)
        with torch.no_grad():
            model.forward()

        visuals = model.get_current_visuals()
        file_id = os.path.splitext(os.path.basename(path))[0]

        for name, img_tensor in visuals.items():
            arr = tensor2im(img_tensor)
            out_path = os.path.join(
                opt.results_img_dir, f'{file_id}_{name}.png')
            save_image(arr, out_path)

        if (i + 1) % 10 == 0:
            print(f'Processed {i+1} / {min(len(paths), opt.how_many)}')

    print(f'Done. Results saved to {opt.results_img_dir}')


if __name__ == '__main__':
    main()
