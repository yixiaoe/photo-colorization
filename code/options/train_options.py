from .base_options import BaseOptions


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

        # ── logging ───────────────────────────────────────────────────────
        parser.add_argument('--print_freq', type=int, default=100,
                            help='console log frequency (iterations)')
        parser.add_argument('--save_latest_freq', type=int, default=2000,
                            help='save latest checkpoint every N iters')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='save checkpoint every N epochs')
        parser.add_argument('--avg_loss_alpha', type=float, default=0.986,
                            help='EMA smoothing for displayed loss')

        return parser


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        self.isTrain = False

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
