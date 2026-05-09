"""Inference entry point.

Usage:
  # Phase 1
  python test.py --method zhang2016 --test_img_dir data/test

  # Phase 2
  python test.py --method inst2020 --test_img_dir data/test

  # Phase 3 Bonus (exemplar)
  python test.py --method zhang2016 --exemplar --ref_img ref.jpg --test_img_dir data/test
"""
import os
import torch

from options.train_options import TestOptions
from datasets.colorization_dataset import create_dataset
from models import create_model
from util.util import save_image, tensor2im


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

    for i, data in enumerate(loader):
        if i >= opt.how_many:
            break
        model.set_input(data)
        with torch.no_grad():
            model.forward()
        visuals = model.get_current_visuals()
        file_id = data['file_id'][0]
        for name, img_tensor in visuals.items():
            arr  = tensor2im(img_tensor)
            path = os.path.join(opt.results_img_dir, f'{file_id}_{name}.png')
            save_image(arr, path)
        if (i + 1) % 10 == 0:
            print(f'Processed {i + 1} / {min(len(dataset), opt.how_many)}')

    print('Inference complete. Results saved to', opt.results_img_dir)


if __name__ == '__main__':
    main()
