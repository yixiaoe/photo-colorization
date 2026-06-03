"""Training entry point.

Usage:
  # Phase 1
  python train.py --method cnn_color --dataset imagenet_mini --data_dir data/train

  # Phase 2 (run full → instance → fusion in order)
  python train.py --method inst_fusion --stage full     --data_dir data/train
  python train.py --method inst_fusion --stage instance --data_dir data/train
  python train.py --method inst_fusion --stage fusion   --data_dir data/train
"""
import csv
import json
import math
import os
import random
import time
import torch

from options.train_options import TrainOptions
from data_process.colorization_dataset import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import save_image, tensor2im
from util.metrics import compute_psnr, compute_ssim


def _append_csv_row(csv_path, row):
    csv_path = os.fspath(csv_path)
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _save_monitor_visuals(monitor_dir, epoch, sample_id, visuals):
    out_dir = os.path.join(os.fspath(monitor_dir), f'epoch_{epoch:03d}')
    os.makedirs(out_dir, exist_ok=True)
    for name in ('real_gray', 'fake_rgb', 'fake_rgb_reg', 'real_rgb'):
        if name not in visuals:
            continue
        out_path = os.path.join(out_dir, f'{sample_id}_{name}.png')
        save_image(tensor2im(visuals[name]), out_path)


def _is_finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _has_nonfinite_loss(losses):
    return any(not _is_finite_number(v) for v in losses.values())


def _is_better_val_loss(val_loss, best_val_loss, min_delta):
    if not _is_finite_number(val_loss):
        return False
    if best_val_loss is None:
        return True
    return float(val_loss) < float(best_val_loss) - float(min_delta)


def _reduce_model_learning_rates(model, factor):
    new_lrs = []
    for optimizer in getattr(model, 'optimizers', []):
        for group in optimizer.param_groups:
            group['lr'] *= factor
            new_lrs.append(group['lr'])

    for scheduler in getattr(model, 'schedulers', []):
        if hasattr(scheduler, 'base_lrs'):
            scheduler.base_lrs = [lr * factor for lr in scheduler.base_lrs]

    return new_lrs


def _save_model_state(model, label):
    os.makedirs(model.save_dir, exist_ok=True)
    for name in model.model_names:
        net = getattr(model, 'net' + name)
        path = os.path.join(model.save_dir, f'{label}_net_{name}.pth')
        torch.save(net.state_dict(), path)


def _load_model_state(model, label):
    for name in model.model_names:
        net = getattr(model, 'net' + name)
        path = os.path.join(model.save_dir, f'{label}_net_{name}.pth')
        state = torch.load(path, map_location=model.device)
        net.load_state_dict(state)


def _mean(values):
    return sum(values) / len(values) if values else None


def _build_monitor_loader(opt, dataset):
    if not getattr(opt, 'monitor_dir', '') or opt.monitor_num <= 0:
        return None
    count = min(opt.monitor_num, len(dataset))
    rng = random.Random(opt.monitor_seed)
    indices = rng.sample(range(len(dataset)), count)
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        subset, batch_size=1, shuffle=False, num_workers=0)


def _save_monitor_epoch(model, monitor_loader, monitor_dir, epoch):
    psnr_scores = []
    ssim_scores = []
    reg_psnr_scores = []
    reg_ssim_scores = []
    for i, data in enumerate(monitor_loader):
        model.set_input(data)
        with torch.no_grad():
            model.forward()
        visuals = model.get_current_visuals()
        sample_id = data.get('file_id', [f'sample_{i:03d}'])
        if isinstance(sample_id, (list, tuple)):
            sample_id = sample_id[0]
        _save_monitor_visuals(monitor_dir, epoch, str(sample_id), visuals)
        if 'fake_rgb' in visuals and 'real_rgb' in visuals:
            fake = visuals['fake_rgb'].clamp(0, 1)
            real = visuals['real_rgb'].clamp(0, 1)
            psnr_scores.append(compute_psnr(fake, real))
            ssim_scores.append(compute_ssim(fake, real))
        if 'fake_rgb_reg' in visuals and 'real_rgb' in visuals:
            fake_reg = visuals['fake_rgb_reg'].clamp(0, 1)
            real = visuals['real_rgb'].clamp(0, 1)
            reg_psnr_scores.append(compute_psnr(fake_reg, real))
            reg_ssim_scores.append(compute_ssim(fake_reg, real))

    return {
        'monitor_num': len(psnr_scores),
        'monitor_psnr': _mean(psnr_scores),
        'monitor_ssim': _mean(ssim_scores),
        'monitor_reg_psnr': _mean(reg_psnr_scores),
        'monitor_reg_ssim': _mean(reg_ssim_scores),
    }


def _add_loss_totals(totals, losses):
    for k, v in losses.items():
        if _is_finite_number(v):
            totals[k] = totals.get(k, 0.0) + float(v)


def _mean_losses(totals, count, prefix):
    if count == 0:
        return {}
    return {f'{prefix}_{k}': v / count for k, v in totals.items()}


def _can_eval_losses(model):
    return model.name() in ('InstFusionModel', 'TextColorModel')


def _evaluate_validation_losses(model, val_loader):
    if not _can_eval_losses(model):
        return {}, False

    totals = {}
    count = 0
    has_nonfinite = False
    with torch.no_grad():
        for val_data in val_loader:
            model.set_input(val_data)
            model.forward()
            model.backward()
            losses = model.get_current_losses()
            if _has_nonfinite_loss(losses):
                has_nonfinite = True
                continue
            _add_loss_totals(totals, losses)
            count += 1
    return _mean_losses(totals, count, 'val_loss'), has_nonfinite


def main():
    opt = TrainOptions().parse()
    monitor_root = os.path.join(opt.monitor_dir, opt.name) \
        if getattr(opt, 'monitor_dir', '') else ''
    if monitor_root:
        os.makedirs(monitor_root, exist_ok=True)
        with open(os.path.join(monitor_root, 'options.json'), 'w') as f:
            json.dump(vars(opt), f, indent=2, sort_keys=True, default=str)

    dataset = create_dataset(opt, stage=opt.stage, split='train')
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.nThreads,
        drop_last=True,
        pin_memory=(len(opt.gpu_ids) > 0),
        persistent_workers=(opt.nThreads > 0),
    )

    opt.model = opt.method   # cnn_color or inst_fusion
    model = create_model(opt)
    if opt.epoch_count > 0:
        model.load_networks(opt.epoch_count)
    model.train()

    visualizer = Visualizer(opt)
    total_iters = 0
    avg_losses  = {}           # EMA-smoothed losses for console display
    best_val_loss = None
    bad_val_checks = 0
    nan_retries = 0
    stop_training = False
    total_epochs = min(opt.niter + opt.niter_decay, opt.max_epochs)
    if opt.niter + opt.niter_decay > total_epochs:
        print(f'[Train] epoch cap active: running {total_epochs} epochs')

    # build val loader once if val_data_dir is provided
    val_loader = None
    monitor_loader = None
    val_dataset = None
    if getattr(opt, 'val_data_dir', '') and opt.val_data_dir:
        import copy
        val_opt = copy.copy(opt)
        val_opt.data_dir = opt.val_data_dir
        val_dataset = create_dataset(val_opt, stage=opt.stage, split='val')
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.nThreads, drop_last=False,
        )
    if monitor_root:
        monitor_dataset = val_dataset
        if monitor_dataset is None:
            monitor_dataset = create_dataset(opt, stage=opt.stage, split='val')
        monitor_loader = _build_monitor_loader(opt, monitor_dataset)

    _save_model_state(model, 'recovery_clean')

    for epoch in range(opt.epoch_count, total_epochs):
        epoch_start = time.time()
        epoch_loss_totals = {}
        epoch_loss_count = 0

        for i, data in enumerate(loader):
            total_iters += 1
            model.set_input(data)
            model.optimize_parameters()

            losses = model.get_current_losses()
            if _has_nonfinite_loss(losses):
                nan_retries += 1
                current_lr = model.optimizers[0].param_groups[0]['lr']
                _load_model_state(model, 'recovery_clean')
                new_lrs = _reduce_model_learning_rates(
                    model, opt.nan_lr_factor)
                model.train()
                next_lr = new_lrs[0] if new_lrs else current_lr
                print(f'[NaN] epoch {epoch+1} iter {total_iters}: '
                      f'restore recovery_clean, lr {current_lr:.3e} -> '
                      f'{next_lr:.3e}')
                if monitor_root:
                    _append_csv_row(
                        os.path.join(monitor_root, 'nan_events.csv'),
                        {'epoch': epoch + 1, 'step': total_iters,
                         'retry': nan_retries, 'old_lr': current_lr,
                         'new_lr': next_lr})
                if nan_retries >= opt.nan_max_retries:
                    print('[NaN] maximum recovery attempts reached; stopping.')
                    stop_training = True
                    break
                continue

            _add_loss_totals(epoch_loss_totals, losses)
            epoch_loss_count += 1

            # EMA smoothing — skip NaN/Inf to avoid polluting the display
            alpha = opt.avg_loss_alpha
            for k, v in losses.items():
                if not _is_finite_number(v):
                    continue
                avg_losses[k] = alpha * avg_losses.get(k, v) + (1 - alpha) * v

            if total_iters % opt.print_freq == 0:
                loss_str = '  '.join(f'{k}: {v:.4f}' for k, v in avg_losses.items())
                print(f'[epoch {epoch+1}  iter {total_iters}]  {loss_str}')
                visualizer.plot_losses(avg_losses, total_iters)
                row = {'epoch': epoch + 1, 'step': total_iters}
                row.update({f'loss_{k}': v for k, v in losses.items()})
                row['lr'] = model.optimizers[0].param_groups[0]['lr']
                if monitor_root:
                    _append_csv_row(
                        os.path.join(monitor_root, 'loss_history.csv'), row)

            if total_iters % opt.save_latest_freq == 0:
                model.save_networks('latest')
                # log visual samples to TensorBoard
                visuals = model.get_current_visuals()
                visualizer.plot_images(visuals, total_iters)

        if stop_training:
            break

        if (epoch + 1) % opt.save_epoch_freq == 0:
            model.save_networks(epoch + 1)

        # ── validation ────────────────────────────────────────────────────
        if val_loader is not None and (epoch + 1) % opt.val_freq == 0:
            model.eval()
            val_losses, val_has_nonfinite = _evaluate_validation_losses(
                model, val_loader)
            model.train()

            train_losses = _mean_losses(
                epoch_loss_totals, epoch_loss_count, 'train_loss')
            metrics_row = {'epoch': epoch + 1, 'step': total_iters}
            metrics_row.update(train_losses)
            metrics_row.update(val_losses)
            if 'train_loss_G' in metrics_row and 'val_loss_G' in metrics_row:
                metrics_row['overfit_gap_G'] = (
                    metrics_row['val_loss_G'] - metrics_row['train_loss_G'])
                print(f"[Val] epoch {epoch+1}  "
                      f"train_G={metrics_row['train_loss_G']:.4f}  "
                      f"val_G={metrics_row['val_loss_G']:.4f}  "
                      f"gap={metrics_row['overfit_gap_G']:.4f}")
                visualizer.plot_losses({
                    'train_loss_G': metrics_row['train_loss_G'],
                    'val_loss_G': metrics_row['val_loss_G'],
                    'overfit_gap_G': metrics_row['overfit_gap_G'],
                }, total_iters)
                if _is_better_val_loss(
                        metrics_row['val_loss_G'], best_val_loss,
                        opt.early_stop_min_delta):
                    best_val_loss = metrics_row['val_loss_G']
                    bad_val_checks = 0
                    model.save_networks('best')
                    print(f'  -> new best checkpoint saved '
                          f'(val_loss_G {best_val_loss:.4f})')
                else:
                    bad_val_checks += 1
                    print(f'  -> no val loss improvement '
                          f'({bad_val_checks}/{opt.early_stop_patience})')
                    if (opt.early_stop_patience > 0 and
                            bad_val_checks >= opt.early_stop_patience):
                        print('[EarlyStop] validation loss stopped improving.')
                        stop_training = True

                metrics_row['best_val_loss_G'] = best_val_loss
                metrics_row['bad_val_checks'] = bad_val_checks
            if val_has_nonfinite:
                metrics_row['val_nonfinite_loss'] = 1
                print('[Val] skipped non-finite validation loss batches')
            if len(metrics_row) > 2:
                if monitor_root:
                    _append_csv_row(
                        os.path.join(monitor_root, 'metrics.csv'), metrics_row)

        if (monitor_loader is not None and opt.monitor_freq > 0 and
                (epoch + 1) % opt.monitor_freq == 0):
            model.eval()
            monitor_metrics = _save_monitor_epoch(
                model, monitor_loader, monitor_root, epoch + 1)
            model.train()
            if monitor_root:
                row = {'epoch': epoch + 1, 'step': total_iters}
                row.update(monitor_metrics)
                _append_csv_row(
                    os.path.join(monitor_root, 'monitor_metrics.csv'), row)

        if stop_training:
            break

        if epoch_loss_count > 0:
            _save_model_state(model, 'recovery_clean')

        model.update_learning_rate()
        elapsed = time.time() - epoch_start
        print(f'Epoch {epoch+1} done in {elapsed:.0f}s')

    visualizer.close()
    print('Training complete.')


if __name__ == '__main__':
    main()
