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
import os
import torch

from util.check_deps import ensure_requirements
ensure_requirements()

from options.train_options import TrainOptions
from data_process.colorization_dataset import create_dataset
from models import create_model
from util.visualizer import Visualizer


def main():
    opt = TrainOptions().parse()
    smoke_test_iters = int(os.environ.get('SMOKE_TEST_ITERS', '0'))

    dataset = create_dataset(opt, stage=opt.stage, split='train')
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.nThreads,
        drop_last=True,
    )

    opt.model = opt.method   # cnn_color or inst_fusion
    if opt.method == 'inst_fusion':
        # stage-wise checkpoint linkage for Task-09
        opt.phase1_name = os.environ.get('PHASE1_NAME', 'cnn_color')
        opt.full_stage_name = os.environ.get('FULL_STAGE_NAME', 'inst_fusion_full')
        opt.instance_stage_name = os.environ.get('INSTANCE_STAGE_NAME', 'inst_fusion_instance')
    model = create_model(opt)
    model.train()

    visualizer = Visualizer(opt)
    total_iters = 0
    avg_losses  = {}           # EMA-smoothed losses for console display

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):
        epoch_start = time.time()

        for i, data in enumerate(loader):
            total_iters += 1
            model.set_input(data)
            model.optimize_parameters()

            losses = model.get_current_losses()

            # EMA smoothing
            alpha = opt.avg_loss_alpha
            for k, v in losses.items():
                avg_losses[k] = alpha * avg_losses.get(k, v) + (1 - alpha) * v

            if total_iters % opt.print_freq == 0:
                loss_str = '  '.join(f'{k}: {v:.4f}' for k, v in avg_losses.items())
                print(f'[epoch {epoch+1}  iter {total_iters}]  {loss_str}')
                visualizer.plot_losses(avg_losses, total_iters)

            if total_iters % opt.save_latest_freq == 0:
                model.save_networks('latest')
                # log visual samples to TensorBoard
                visuals = model.get_current_visuals()
                visualizer.plot_images(visuals, total_iters)

            if smoke_test_iters > 0 and total_iters >= smoke_test_iters:
                model.save_networks('latest')
                print(f'Smoke test reached {smoke_test_iters} iterations, checkpoint saved.')
                visualizer.close()
                return

        if (epoch + 1) % opt.save_epoch_freq == 0:
            model.save_networks(epoch + 1)

        model.update_learning_rate()
        elapsed = time.time() - epoch_start
        print(f'Epoch {epoch+1} done in {elapsed:.0f}s')

    visualizer.close()
    print('Training complete.')


if __name__ == '__main__':
    main()
