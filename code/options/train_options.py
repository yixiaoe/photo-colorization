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
        parser.add_argument('--lr', type=float, default=3e-5)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--lr_policy', type=str, default='lambda',
                            choices=['lambda', 'step', 'plateau'])
        parser.add_argument('--epoch_count', type=int, default=0)

        # ── Phase 1 specific ─────────────────────────────────────────────
        parser.add_argument('--T', type=float, default=T,
                            help='annealed-mean temperature for inference decoding')
        parser.add_argument('--rebalance_gamma', type=float, default=REBALANCE_GAMMA,
                            help='prior-mix gamma for class rebalance weights')

        # ── Phase 2 specific ─────────────────────────────────────────────
        parser.add_argument('--num_classes', type=int, default=91,
                            help='FiLM label vocabulary size (91 for COCO 1-90)')
        parser.add_argument('--embed_dim', type=int, default=64,
                            help='FiLM class embedding dimension')
        parser.add_argument('--full_ckpt', type=str, default='',
                            help='stage-full checkpoint (net_G.pth) to warm-start stage-instance / Phase 3')
        parser.add_argument('--inst_ckpt', type=str, default='',
                            help='stage-instance checkpoint (net_G.pth) to init fusion inst-net / Phase 3')
        parser.add_argument('--lr_backbone', type=float, default=1e-5,
                            help='backbone lr in stage-instance (FiLM layers use --lr)')
        parser.add_argument('--ann_file', type=str, default='',
                            help='COCO annotation JSON for stage=instance (GT bbox+label)')
        parser.add_argument('--box_num', type=int, default=8,
                            help='max instances per image in fusion stage')

        # ── Phase 3 specific (text_color) ────────────────────────────────
        parser.add_argument('--fusion_ckpt', type=str, default='',
                            help='Phase 2 stage-fusion checkpoint (FusionPipeline) for text_color')
        parser.add_argument('--jsonl_file', type=str, default='',
                            help='Phase 3 training JSONL from scripts/build_phase3_jsonl.py')
        parser.add_argument('--val_jsonl_file', type=str, default='',
                            help='Phase 3 validation JSONL')
        parser.add_argument('--instances_file', type=str, default='',
                            help='COCO instances_*.json for Phase 3 mask rasterisation')
        parser.add_argument('--val_instances_file', type=str, default='',
                            help='COCO val instances_*.json')
        parser.add_argument('--val_img_dir', type=str, default='',
                            help='Phase 3 val image directory')
        parser.add_argument('--img_dir', type=str, default='',
                            help='Phase 3 train image directory (overrides --data_dir)')
        parser.add_argument('--clip_cache', type=str, default='datasets/phase3/clip_text_cache.pt',
                            help='Phase 3 precomputed CLIP embedding cache (optional)')
        parser.add_argument('--max_inst', type=int, default=8,
                            help='Phase 3 max instances per image')
        parser.add_argument('--huber_weight', type=float, default=3.0)
        parser.add_argument('--lambda_inst', type=float, default=1.0)
        parser.add_argument('--lambda_rank', type=float, default=0.1)
        parser.add_argument('--lambda_outside', type=float, default=0.2)
        parser.add_argument('--rank_margin', type=float, default=0.05)
        parser.add_argument('--rank_warmup_epoch', type=int, default=5,
                            help='Phase 3 epochs at λ=0 before linear warmup begins')
        parser.add_argument('--rank_warmup_len', type=int, default=5,
                            help='Phase 3 cosine warmup length after zero phase')
        parser.add_argument('--use_amp', action='store_true',
                            help='Phase 3: enable AMP (off by default — frozen BN can NaN under fp16)')

        # ── logging ───────────────────────────────────────────────────────
        parser.add_argument('--print_freq', type=int, default=100,
                            help='console log frequency (iterations)')
        parser.add_argument('--save_latest_freq', type=int, default=2000,
                            help='save latest checkpoint every N iters')
        parser.add_argument('--save_epoch_freq', type=int, default=20,
                            help='save checkpoint every N epochs')
        parser.add_argument('--avg_loss_alpha', type=float, default=0.986,
                            help='EMA smoothing for displayed loss')

        return parser


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        self.isTrain = False

        parser.add_argument('--stage', type=str, default='fusion',
                            choices=['full', 'instance', 'fusion'],
                            help='which stage checkpoint to load for inst_fusion inference '
                                 '(default: fusion — uses FusionPipeline + FiLMInstanceGenerator)')
        parser.add_argument('--full_ckpt', type=str, default='',
                            help='stage-full checkpoint for fusion inference')
        parser.add_argument('--inst_ckpt', type=str, default='',
                            help='stage-instance checkpoint for fusion inference')
        parser.add_argument('--num_classes', type=int, default=91)
        parser.add_argument('--embed_dim',   type=int, default=64)
        parser.add_argument('--lr_backbone', type=float, default=5e-5)
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

        # ── Phase 3 (text_color) inference ────────────────────────────────
        # --full_ckpt / --inst_ckpt are already defined above (shared with Phase 2)
        parser.add_argument('--adapter_ckpt', type=str, default='',
                            help='trained Phase 3 adapter (latest_net_T.pth)')
        parser.add_argument('--fusion_ckpt', type=str, default='',
                            help='Phase 3: frozen Phase 2 stage-fusion ckpt')
        parser.add_argument('--clip_cache', type=str, default='datasets/phase3/clip_text_cache.pt')
        parser.add_argument('--image', type=str, default='',
                            help='Phase 3: single input image (alternative to --test_img_dir)')
        parser.add_argument('--prompt', action='append', default=[],
                            help='Phase 3: "inst:i=..." or "bg=...", repeatable')
        parser.add_argument('--score_thresh', type=float, default=0.5,
                            help='Phase 3: Mask R-CNN score threshold')

        # ── Exemplar Bonus (legacy, kept for back-compat) ─────────────────
        parser.add_argument('--exemplar', action='store_true',
                            help='legacy: enable exemplar-based colorization')
        parser.add_argument('--ref_img', type=str, default='',
                            help='legacy: reference style image')
        parser.add_argument('--harmonize', action='store_true',
                            help='legacy: StyleHarmonizer')

        return parser
