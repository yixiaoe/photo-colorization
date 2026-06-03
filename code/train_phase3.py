"""
Phase 3 — 文本引导上色训练入口。

本脚本是 Phase 3 的独立训练入口，与 train.py（Phase 1/2）完全解耦。
训练流程：
  1. 加载 CocoColorObjectDataset（color-object JSONL）或旧 CocoCaptionDataset
  2. 创建 TextColorModel（含冻结 CLIP + 冻结 InstanceGenerator + 可训练 xattn）
  3. 每个 batch：图像转 Lab，caption 经 CLIP 编码，前向 → 损失 → 反向更新 xattn
  4. 定期保存 checkpoint 和 TensorBoard 可视化

训练策略：
  - 仅训练 Cross-Attention 参数（~609K），其余全部冻结
  - 使用 CE(rebalanced) + 3×Huber 损失（与 Phase 2 一致）
  - 支持 AMP 混合精度加速

Usage:
  python train_phase3.py \\
      --color_object_file data/phase3_color_object_train.jsonl \\
      --val_color_object_file data/phase3_color_object_val.jsonl \\
      --full_ckpt checkpoints/inst_full/80_net_G.pth \\
      --fineSize 256 --batch_size 16 --nThreads 4 \\
      --lr 1e-4 --niter 30 --niter_decay 30 \\
      --name text_color --gpu_ids 0
"""
import time
import torch

from options.phase3_options import Phase3TrainOptions
from data_process.text_color_dataset import CocoCaptionDataset
from data_process.color_object_dataset import CocoColorObjectDataset
from models.text_color_model import TextColorModel
from util.visualizer import Visualizer
from util.metrics import compute_psnr


def main():
    opt = Phase3TrainOptions().parse()

    if getattr(opt, 'color_object_file', ''):
        dataset = CocoColorObjectDataset(
            records_file=opt.color_object_file,
            fine_size=opt.fineSize,
            split='train',
            max_dataset_size=opt.max_dataset_size,
        )
    else:
        if not getattr(opt, 'caption_file', ''):
            raise ValueError(
                'Phase 3 requires --color_object_file or --caption_file')
        dataset = CocoCaptionDataset(
            img_dir=opt.data_dir,
            caption_file=opt.caption_file,
            fine_size=opt.fineSize,
            split='train',
            max_dataset_size=opt.max_dataset_size,
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.nThreads,
        drop_last=True,
        pin_memory=(len(opt.gpu_ids) > 0),
        persistent_workers=(opt.nThreads > 0),
    )

    model = TextColorModel()
    model.initialize(opt)
    if opt.epoch_count > 0:
        model.load_networks(opt.epoch_count)
    model.train()

    visualizer = Visualizer(opt)
    total_iters = 0
    avg_losses = {}
    best_psnr = -1.0

    # build val loader once if val_data_dir is provided
    val_loader = None
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

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):
        epoch_start = time.time()

        for i, data in enumerate(loader):
            total_iters += 1
            model.set_input(data)
            model.optimize_parameters()

            losses = model.get_current_losses()
            alpha = opt.avg_loss_alpha
            for k, v in losses.items():
                if v != v or v == float('inf'):
                    continue
                avg_losses[k] = alpha * avg_losses.get(k, v) + (1-alpha) * v

            if total_iters % opt.print_freq == 0:
                loss_str = '  '.join(
                    f'{k}: {v:.4f}' for k, v in avg_losses.items())
                print(f'[epoch {epoch+1}  iter {total_iters}]  {loss_str}')
                visualizer.plot_losses(avg_losses, total_iters)

            if total_iters % opt.save_latest_freq == 0:
                model.save_networks('latest')
                visuals = model.get_current_visuals()
                visualizer.plot_images(visuals, total_iters)

        if (epoch + 1) % opt.save_epoch_freq == 0:
            model.save_networks(epoch + 1)

        # ── validation ────────────────────────────────────────────────────
        if val_loader is not None and (epoch + 1) % opt.val_freq == 0:
            model.eval()
            psnr_scores = []
            with torch.no_grad():
                for val_data in val_loader:
                    model.set_input(val_data)
                    model.forward()
                    visuals = model.get_current_visuals()
                    if 'fake_rgb' in visuals and 'real_rgb' in visuals:
                        psnr_scores.append(
                            compute_psnr(visuals['fake_rgb'].clamp(0, 1),
                                         visuals['real_rgb'].clamp(0, 1))
                        )
            model.train()
            if psnr_scores:
                val_psnr = sum(psnr_scores) / len(psnr_scores)
                print(f'[Val] epoch {epoch+1}  PSNR = {val_psnr:.2f} dB')
                visualizer.plot_losses({'val_psnr': val_psnr}, total_iters)
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    model.save_networks('best')
                    print(f'  → new best checkpoint saved (PSNR {best_psnr:.2f} dB)')

        model.update_learning_rate()
        elapsed = time.time() - epoch_start
        print(f'Epoch {epoch+1} done in {elapsed:.0f}s')

    visualizer.close()
    print('Training complete.')


if __name__ == '__main__':
    main()
