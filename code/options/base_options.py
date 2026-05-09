import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # ── data ──────────────────────────────────────────────────────────
        parser.add_argument('--data_dir', type=str, default='data/train',
                            help='path to training image folder')
        parser.add_argument('--dataset', type=str, default='imagenet_mini',
                            choices=['imagenet_mini', 'cifar10'],
                            help='dataset type')
        parser.add_argument('--fineSize', type=int, default=256,
                            help='image size fed to the network')
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--nThreads', type=int, default=4)
        parser.add_argument('--max_dataset_size', type=int, default=float('inf'))

        # ── model ─────────────────────────────────────────────────────────
        parser.add_argument('--method', type=str, default='zhang2016',
                            choices=['zhang2016', 'inst2020'],
                            help='colorization method')
        parser.add_argument('--ngf', type=int, default=64)
        parser.add_argument('--norm', type=str, default='batch',
                            choices=['batch', 'instance', 'none'])
        parser.add_argument('--no_dropout', action='store_true')
        parser.add_argument('--init_type', type=str, default='normal',
                            choices=['normal', 'xavier', 'kaiming', 'orthogonal'])

        # ── Lab color space ───────────────────────────────────────────────
        parser.add_argument('--ab_norm', type=float, default=110.)
        parser.add_argument('--ab_max', type=float, default=110.)
        parser.add_argument('--ab_quant', type=float, default=10.)
        parser.add_argument('--l_norm', type=float, default=100.)
        parser.add_argument('--l_cent', type=float, default=50.)
        parser.add_argument('--mask_cent', type=float, default=0.5)

        # ── checkpoint / output ───────────────────────────────────────────
        parser.add_argument('--name', type=str, default='experiment',
                            help='experiment name; checkpoints saved to checkpoints/<name>/')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        parser.add_argument('--results_dir', type=str, default='./results')
        parser.add_argument('--which_epoch', type=str, default='latest')

        # ── misc ──────────────────────────────────────────────────────────
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='comma-separated GPU ids; -1 for CPU')
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--half', action='store_true',
                            help='use FP16 inference')

        self.initialized = True
        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        else:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt = parser.parse_args()
        opt.isTrain = self.isTrain

        # derived color-bin counts
        opt.A = int(2 * opt.ab_max / opt.ab_quant + 1)
        opt.B = opt.A

        # GPU setup
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for s in str_ids:
            gid = int(s)
            if gid >= 0:
                opt.gpu_ids.append(gid)
        if opt.gpu_ids:
            torch.cuda.set_device(opt.gpu_ids[0])

        # ensure output dirs exist
        os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
        os.makedirs(opt.results_dir, exist_ok=True)

        if opt.verbose:
            self._print_options(opt, parser)

        self.opt = opt
        return opt

    def _print_options(self, opt, parser):
        lines = ['--- Options ---']
        for k, v in sorted(vars(opt).items()):
            default = parser.get_default(k)
            tag = '' if v == default else f'\t[default: {default}]'
            lines.append(f'{k:>25}: {str(v):<30}{tag}')
        lines.append('--- End ---')
        print('\n'.join(lines))
