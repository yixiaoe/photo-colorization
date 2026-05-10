"""Training entry point.

Usage:
  # Phase 1
  python train.py --method cnn_color --dataset imagenet_mini --data_dir data/train

  # Phase 2 (run full → instance → fusion in order)
  python train.py --method inst_fusion --stage full     --data_dir data/train
  python train.py --method inst_fusion --stage instance --data_dir data/train
  python train.py --method inst_fusion --stage fusion   --data_dir data/train
"""
import time
import torch

from options.train_options import TrainOptions
from datasets.colorization_dataset import create_dataset
from models import create_model
from util.visualizer import Visualizer


def main():
    opt = TrainOptions().parse()

    dataset = create_dataset(opt, stage=opt.stage, split='train')
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.nThreads,
        drop_last=True,
    )

    opt.model = opt.method   # cnn_color or inst_fusion
    model = create_model(opt)
    model.train()

    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):
        for i, data in enumerate(loader):
            total_iters += 1
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                loss_str = '  '.join(f'{k}: {v:.4f}' for k, v in losses.items())
                print(f'[epoch {epoch}  iter {total_iters}]  {loss_str}')
                visualizer.plot_losses(losses, total_iters)

            if total_iters % opt.save_latest_freq == 0:
                model.save_networks('latest')

        if (epoch + 1) % opt.save_epoch_freq == 0:
            model.save_networks(epoch + 1)

        model.update_learning_rate()

    visualizer.close()
    print('Training complete.')


if __name__ == '__main__':
    main()
