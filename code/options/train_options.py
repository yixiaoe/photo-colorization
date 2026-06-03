from .base_options import BaseOptions
from resources.defaults import T, REBALANCE_GAMMA


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        self.isTrain = True

        # ── phase / stage ─────────────────────────────────────────────────
        parser.add_argument('--stage', type=str, default='full',
                            choices=['full', 'instance', 'fusion'],
                            help='training stage (full: Phase1+Phase2-full; '
                                 'instance/fusion: Phase2 only)')

        # ── optimisation ──────────────────────────────────────────────────
        parser.add_argument('--niter', type=int, default=100,
                            help='epochs at base learning rate')
        parser.add_argument('--niter_decay', type=int, default=100,
                            help='epochs to linearly decay lr to 0')
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--lr_policy', type=str, default='lambda',
                            choices=['lambda', 'step', 'plateau'])
        parser.add_argument('--epoch_count', type=int, default=0)
        parser.add_argument('--max_epochs', type=int, default=100,
                            help='hard cap on total epochs')
        parser.add_argument('--grad_clip_norm', type=float, default=5.0,
                            help='clip gradient norm; set <=0 to disable')
        parser.add_argument('--nan_lr_factor', type=float, default=0.1,
                            help='multiply lr by this factor after NaN/Inf loss')
        parser.add_argument('--nan_max_retries', type=int, default=3,
                            help='stop after this many NaN/Inf recovery attempts')
        parser.add_argument('--early_stop_patience', type=int, default=6,
                            help='validation checks without improvement before stop')
        parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                            help='minimum val loss improvement for early stopping')

        # ── Phase 1 specific ─────────────────────────────────────────────
        parser.add_argument('--T', type=float, default=T,
                            help='annealed-mean temperature for inference decoding')
        parser.add_argument('--rebalance_gamma', type=float, default=REBALANCE_GAMMA,
                            help='prior-mix gamma for class rebalance weights')
        parser.add_argument('--huber_weight', type=float, default=3.0,
                            help='weight for the ab regression Huber loss')

        # ── Phase 2 specific ─────────────────────────────────────────────
        parser.add_argument('--num_classes', type=int, default=91,
                            help='FiLM label vocabulary size (91 for COCO 1-90)')
        parser.add_argument('--embed_dim', type=int, default=64,
                            help='FiLM class embedding dimension')
        parser.add_argument('--full_ckpt', type=str, default='',
                            help='stage-full checkpoint (net_G.pth) to warm-start stage-instance')
        parser.add_argument('--inst_ckpt', type=str, default='',
                            help='stage-instance checkpoint (net_G.pth) to init fusion inst-net')
        parser.add_argument('--lr_backbone', type=float, default=5e-5,
                            help='backbone lr in stage-instance (FiLM layers use --lr)')
        parser.add_argument('--ann_file', type=str, default='',
                            help='COCO annotation JSON for stage=instance (GT bbox+label)')
        parser.add_argument('--box_num', type=int, default=8,
                            help='max instances per image in fusion stage')

        # ── validation ────────────────────────────────────────────────────
        parser.add_argument('--val_data_dir', type=str, default='',
                            help='validation image folder; skip val if empty')
        parser.add_argument('--val_freq', type=int, default=5,
                            help='run validation every N epochs')

        # ── logging ───────────────────────────────────────────────────────
        parser.add_argument('--print_freq', type=int, default=100,
                            help='console log frequency (iterations)')
        parser.add_argument('--save_latest_freq', type=int, default=2000,
                            help='save latest checkpoint every N iters')
        parser.add_argument('--save_epoch_freq', type=int, default=20,
                            help='save checkpoint every N epochs')
        parser.add_argument('--avg_loss_alpha', type=float, default=0.986,
                            help='EMA smoothing for displayed loss')
        parser.add_argument('--monitor_dir', type=str, default='results/jiandu',
                            help='folder for CSV metrics and fixed sample visualisations')
        parser.add_argument('--monitor_num', type=int, default=50,
                            help='number of fixed samples saved for supervision')
        parser.add_argument('--monitor_freq', type=int, default=5,
                            help='save supervision samples every N validation epochs')
        parser.add_argument('--monitor_seed', type=int, default=123,
                            help='fixed random seed for monitor sample selection')

        return parser


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        self.isTrain = False

        parser.add_argument('--stage', type=str, default='full',
                            choices=['full', 'instance', 'fusion'],
                            help='model stage for inst_fusion method')
        parser.add_argument('--T', type=float, default=T,
                            help='annealed-mean temperature for inference decoding')
        parser.add_argument('--rebalance_gamma', type=float, default=REBALANCE_GAMMA)
        parser.add_argument('--test_img_dir', type=str, default='data/test',
                            help='folder of test images')
        parser.add_argument('--results_img_dir', type=str, default='results/images',
                            help='folder to save colorized outputs')
        parser.add_argument('--how_many', type=int, default=200,
                            help='max number of test images to process')
        parser.add_argument('--box_num', type=int, default=8,
                            help='max instances per image (inst_fusion only)')

        # ── Exemplar Bonus ────────────────────────────────────────────────
        parser.add_argument('--exemplar', action='store_true',
                            help='enable exemplar-based colorization (Phase 3)')
        parser.add_argument('--ref_img', type=str, default='',
                            help='path to reference style image (--exemplar mode)')
        parser.add_argument('--harmonize', action='store_true',
                            help='enable StyleHarmonizer between branches (inst_fusion + exemplar only)')

        return parser
