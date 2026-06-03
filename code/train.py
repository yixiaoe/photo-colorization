"""Training entry point.

Usage:
  # Phase 1
  python train.py --method cnn_color --dataset imagenet_mini --data_dir data/train

  # Phase 2 (run full → instance → fusion in order)
  python train.py --method inst_fusion --stage full     --data_dir data/train
  python train.py --method inst_fusion --stage instance --data_dir data/train
  python train.py --method inst_fusion --stage fusion   --data_dir data/train

  # Phase 3 (text_color, frozen on top of Phase 2)
  python train.py --method text_color \
      --jsonl_file data/phase3_color_object_train.jsonl \
      --instances_file datasets/coco/annotations/instances_train2017.json \
      --img_dir datasets/coco/train2017 \
      --full_ckpt   checkpoints/inst_fusion_full/80_net_G.pth \
      --inst_ckpt   checkpoints/inst_fusion_instance/25_net_G.pth \
      --fusion_ckpt checkpoints/inst_fusion_fusion/25_net_G.pth \
      --name phase3_text_color --batch_size 1 --lr 1e-4 --beta1 0.5 \
      --niter 30 --niter_decay 30
"""
import time
import torch

from util.check_deps import ensure_requirements
ensure_requirements()

from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer


def _build_loader(opt, split):
    """Return (dataset, loader) — Phase 3 needs a different dataset/collate."""
    if opt.method == 'text_color':
        from data_process.text_color_dataset import (
            TextColorCocoDataset, text_color_collate,
        )
        jsonl = opt.jsonl_file if split == 'train' else opt.val_jsonl_file
        ann   = opt.instances_file if split == 'train' else opt.val_instances_file
        idir  = opt.img_dir if split == 'train' else opt.val_img_dir
        ds = TextColorCocoDataset(
            opt, jsonl_file=jsonl, instances_file=ann,
            img_dir=idir, max_inst=opt.max_inst, split=split,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=1, shuffle=(split == 'train'),
            num_workers=opt.nThreads, drop_last=False,
            pin_memory=(len(opt.gpu_ids) > 0),
            collate_fn=text_color_collate,
        )
        return ds, loader

    # Phase 1/2 path
    from data_process.colorization_dataset import create_dataset
    ds = create_dataset(opt, stage=opt.stage, split=split)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=opt.batch_size, shuffle=(split == 'train'),
        num_workers=opt.nThreads, drop_last=True,
        pin_memory=(len(opt.gpu_ids) > 0),
        persistent_workers=(opt.nThreads > 0),
    )
    return ds, loader


def _run_validation(model, val_loader, max_iters=50):
    model.eval()
    n = 0
    sums = {}
    with torch.no_grad():
        for data in val_loader:
            if n >= max_iters:
                break
            model.set_input(data)
            model.forward()
            model.backward()
            for k, v in model.get_current_losses().items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
    model.train()
    return {k: v / max(1, n) for k, v in sums.items()}


def main():
    opt = TrainOptions().parse()

    _, train_loader = _build_loader(opt, 'train')

    # Phase 3 has an optional val loader
    val_loader = None
    if opt.method == 'text_color' and opt.val_jsonl_file:
        try:
            _, val_loader = _build_loader(opt, 'val')
        except (FileNotFoundError, ValueError) as e:
            print(f'[warn] validation set unavailable: {e}')

    opt.model = opt.method   # cnn_color / inst_fusion / text_color
    model = create_model(opt)
    if opt.epoch_count > 0:
        model.load_networks(opt.epoch_count)
    model.train()

    visualizer = Visualizer(opt)
    total_iters = 0
    avg_losses  = {}

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)
        epoch_start = time.time()

        for i, data in enumerate(train_loader):
            total_iters += 1
            model.set_input(data)
            model.optimize_parameters()

            losses = model.get_current_losses()

            # EMA smoothing — skip NaN/Inf so the display doesn't get poisoned
            alpha = opt.avg_loss_alpha
            for k, v in losses.items():
                if v != v or v == float('inf'):
                    continue
                avg_losses[k] = alpha * avg_losses.get(k, v) + (1 - alpha) * v

            if total_iters % opt.print_freq == 0:
                loss_str = '  '.join(f'{k}: {v:.4f}' for k, v in avg_losses.items())
                print(f'[epoch {epoch+1}  iter {total_iters}]  {loss_str}')
                visualizer.plot_losses(avg_losses, total_iters)

            if total_iters % opt.save_latest_freq == 0:
                model.save_networks('latest')
                visuals = model.get_current_visuals()
                visualizer.plot_images(visuals, total_iters)

        if (epoch + 1) % opt.save_epoch_freq == 0:
            model.save_networks(epoch + 1)

        if val_loader is not None and (epoch + 1) % opt.save_epoch_freq == 0:
            val_losses = _run_validation(model, val_loader)
            print('  [val] ' + '  '.join(f'{k}: {v:.4f}' for k, v in val_losses.items()))
            visualizer.plot_losses({f'val_{k}': v for k, v in val_losses.items()},
                                   total_iters)

        model.update_learning_rate()
        elapsed = time.time() - epoch_start
        print(f'Epoch {epoch+1} done in {elapsed:.0f}s')

    visualizer.close()
    print('Training complete.')


if __name__ == '__main__':
    main()
