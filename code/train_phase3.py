"""
Phase 3 — 文本引导上色训练入口。

单卡:  python train_phase3.py --color_object_file ... --gpu_ids 0
多卡:  torchrun --nproc_per_node=N train_phase3.py --color_object_file ... --gpu_ids 0
"""
import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from options.phase3_options import Phase3TrainOptions
from data_process.text_color_dataset import CocoCaptionDataset
from data_process.color_object_dataset import CocoColorObjectDataset
from models.text_color_model import TextColorModel, _masked_ab_huber_loss
from util.visualizer import Visualizer
from util.metrics import compute_psnr
from util.util import decode_zhang2016_annealed_mean
from scripts.eval_phase3_prompt_control import edge_aware_total_variation


def _init_ddp():
    """Returns (local_rank, world_size, is_ddp). No-op if LOCAL_RANK not set."""
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank < 0 or not torch.cuda.is_available():
        return 0, 1, False
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), True


def _set_seed(seed, rank=0):
    if seed is None:
        return
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    local_rank, world_size, is_ddp = _init_ddp()
    is_main = (local_rank == 0)

    opt = Phase3TrainOptions().parse()
    # force device to this rank's GPU
    if is_ddp:
        opt.gpu_ids = [local_rank]
    _set_seed(getattr(opt, 'seed', None), rank=local_rank)

    # ── dataset ───────────────────────────────────────────────────────────
    if getattr(opt, 'color_object_file', ''):
        dataset = CocoColorObjectDataset(
            records_file=opt.color_object_file,
            fine_size=opt.fineSize,
            split='train',
            max_dataset_size=opt.max_dataset_size,
        )
    else:
        if not getattr(opt, 'caption_file', ''):
            raise ValueError('Phase 3 requires --color_object_file or --caption_file')
        dataset = CocoCaptionDataset(
            img_dir=opt.data_dir,
            caption_file=opt.caption_file,
            fine_size=opt.fineSize,
            split='train',
            max_dataset_size=opt.max_dataset_size,
        )

    train_sampler = DistributedSampler(dataset, shuffle=True) if is_ddp else None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=opt.nThreads,
        drop_last=True,
        pin_memory=(len(opt.gpu_ids) > 0),
        persistent_workers=(opt.nThreads > 0),
    )

    # ── model ─────────────────────────────────────────────────────────────
    model = TextColorModel()
    model.initialize(opt)
    if opt.epoch_count > 0:
        model.load_networks(opt.epoch_count)
    model.train()

    if is_ddp:
        model.netG = DDP(model.netG, device_ids=[local_rank],
                         find_unused_parameters=False)

    # ── val loader (rank-0 only) ───────────────────────────────────────────
    val_loader = None
    if is_main:
        if getattr(opt, 'val_color_object_file', ''):
            val_dataset = CocoColorObjectDataset(
                records_file=opt.val_color_object_file,
                fine_size=opt.fineSize,
                split='val',
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=opt.batch_size, shuffle=False,
                num_workers=opt.nThreads, drop_last=False,
            )
        elif getattr(opt, 'val_data_dir', '') and opt.val_data_dir:
            val_caption = getattr(opt, 'val_caption_file', '')
            if val_caption:
                val_dataset = CocoCaptionDataset(
                    img_dir=opt.val_data_dir,
                    caption_file=val_caption,
                    fine_size=opt.fineSize,
                    split='val',
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=opt.batch_size, shuffle=False,
                    num_workers=opt.nThreads, drop_last=False,
                )

    visualizer = Visualizer(opt) if is_main else None
    total_iters = 0
    avg_losses = {}
    best_score = None

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):
        epoch_start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for i, data in enumerate(loader):
            total_iters += 1
            model.set_input(data)
            model.optimize_parameters()

            if is_main:
                losses = model.get_current_losses()
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
                    visualizer.plot_images(model.get_current_visuals(), total_iters)

        if is_main and (epoch + 1) % opt.save_epoch_freq == 0:
            model.save_networks(epoch + 1)

        # ── validation (rank-0 only) ───────────────────────────────────────
        if is_ddp:
            dist.barrier()

        if is_main and val_loader is not None and (epoch + 1) % opt.val_freq == 0:
            model.eval()
            psnr_scores = []
            rank_success_list, neg_minus_pos_list, outside_delta_list, tv_list = [], [], [], []
            margin = getattr(opt, 'rank_margin', 0.05)
            with torch.no_grad():
                for val_data in val_loader:
                    model.set_input(val_data)
                    model.forward()
                    visuals = model.get_current_visuals()
                    if 'fake_rgb' in visuals and 'real_rgb' in visuals:
                        psnr_scores.append(
                            compute_psnr(visuals['fake_rgb'].clamp(0, 1),
                                         visuals['real_rgb'].clamp(0, 1)))
                    if model.use_instance_prompt and hasattr(model, 'pred_class_neg'):
                        ab_pos = decode_zhang2016_annealed_mean(
                            model.pred_class, model.pts_in_hull,
                            T=0.38, ab_norm_val=model.opt.ab_norm)
                        ab_neg = decode_zhang2016_annealed_mean(
                            model.pred_class_neg, model.pts_in_hull,
                            T=0.38, ab_norm_val=model.opt.ab_norm)
                        gt_ab_4x = F.interpolate(
                            model.real_ab, size=ab_pos.shape[-2:],
                            mode='bilinear', align_corners=False)
                        mask_4x = model.mask_4x
                        for b in range(ab_pos.shape[0]):
                            pd = _masked_ab_huber_loss(
                                ab_pos[b:b+1], gt_ab_4x[b:b+1], mask_4x[b:b+1]).item()
                            nd = _masked_ab_huber_loss(
                                ab_neg[b:b+1], gt_ab_4x[b:b+1], mask_4x[b:b+1]).item()
                            od = _masked_ab_huber_loss(
                                ab_pos[b:b+1], ab_neg[b:b+1],
                                1.0 - mask_4x[b:b+1]).item()
                            tv = edge_aware_total_variation(
                                ab_pos[b:b+1], model.real_L[b:b+1])
                            rank_success_list.append(float(nd >= pd + margin))
                            neg_minus_pos_list.append(nd - pd)
                            outside_delta_list.append(od)
                            tv_list.append(tv['edge_aware_tv_ab'])
            model.train()
            val_psnr = sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0.0
            if psnr_scores:
                print(f'[Val] epoch {epoch+1}  PSNR = {val_psnr:.2f} dB')
                visualizer.plot_losses({'val_psnr': val_psnr}, total_iters)
            if rank_success_list:
                rank_rate = sum(rank_success_list) / len(rank_success_list)
                mean_nmp = sum(neg_minus_pos_list) / len(neg_minus_pos_list)
                mean_od = sum(outside_delta_list) / len(outside_delta_list)
                mean_tv = sum(tv_list) / len(tv_list)
                score = (rank_rate, mean_nmp, -mean_od, -mean_tv, val_psnr)
                print(f'[Val] rank_rate={rank_rate:.3f}  nmp={mean_nmp:.4f}'
                      f'  out_delta={mean_od:.4f}  tv={mean_tv:.4f}')
                visualizer.plot_losses({'val_rank_rate': rank_rate,
                                        'val_neg_minus_pos': mean_nmp}, total_iters)
            elif psnr_scores:
                score = (0.0, 0.0, 0.0, 0.0, val_psnr)
            else:
                score = None
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                model.save_networks('best')
                label = (f'rank_rate={score[0]:.3f} psnr={val_psnr:.2f}'
                         if rank_success_list else f'PSNR={val_psnr:.2f}')
                print(f'  → new best checkpoint saved ({label})')

        model.update_learning_rate()
        if is_main:
            elapsed = time.time() - epoch_start
            print(f'Epoch {epoch+1} done in {elapsed:.0f}s')

    if is_main:
        visualizer.close()
        print('Training complete.')
    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
