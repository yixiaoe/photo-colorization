"""
Phase 3 — 命令行参数定义。

本模块定义 Phase 3 专用的训练和测试参数，继承 BaseOptions 获取
通用参数（data_dir, fineSize, gpu_ids 等），并新增：

Phase3TrainOptions:
  --color_object_file : Phase 3 color-object JSONL 路径
  --caption_file  : COCO captions JSON 路径（旧整图 caption 路径）
  --full_ckpt     : Phase 2 stage-full 预训练权重路径
  --clip_arch     : CLIP 模型架构（默认 ViT-B-32）
  --num_heads     : Cross-Attention 头数（默认 4）
  --niter/niter_decay : 训练 epoch 数（30+30 线性衰减）

Phase3TestOptions:
  --prompt        : 用户输入的文本描述（控制上色方向）
  --test_img_dir  : 测试图像目录
  --results_img_dir : 输出结果目录

约束：不修改 options/base_options.py 或 options/train_options.py。
"""
from options.base_options import BaseOptions


class Phase3TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        self.isTrain = True

        # phase 3 specific
        parser.add_argument('--color_object_file', type=str, default='',
                            help='Phase 3 color-object JSONL records')
        parser.add_argument('--caption_file', type=str, default='',
                            help='COCO captions JSON (legacy whole-image captions)')
        parser.add_argument('--full_ckpt', type=str,
                            default='checkpoints/inst_full/80_net_G.pth',
                            help='Phase 2 stage-full 80_net_G.pth')
        parser.add_argument('--clip_arch', type=str, default='ViT-B-32-quickgelu',
                            help='CLIP model architecture')
        parser.add_argument('--clip_pretrained_path', type=str,
                            default='checkpoints/clip/open_clip_model.safetensors',
                            help='local open_clip checkpoint path')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='cross-attention heads')
        parser.add_argument('--rebalance_gamma', type=float, default=0.5)
        parser.add_argument('--huber_weight', type=float, default=3.0)
        parser.add_argument('--lambda_inst', type=float, default=1.0)
        parser.add_argument('--lambda_rank', type=float, default=0.1)
        parser.add_argument('--lambda_outside', type=float, default=0.2)
        parser.add_argument('--rank_margin', type=float, default=0.05)
        parser.add_argument('--seed', type=int, default=None,
                            help='optional random seed for controlled ablations')

        # optimisation
        parser.add_argument('--niter', type=int, default=30)
        parser.add_argument('--niter_decay', type=int, default=30)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--lr_policy', type=str, default='lambda',
                            choices=['lambda', 'step', 'plateau'])
        parser.add_argument('--epoch_count', type=int, default=0)

        # logging
        parser.add_argument('--print_freq', type=int, default=100)
        parser.add_argument('--save_latest_freq', type=int, default=2000)
        parser.add_argument('--save_epoch_freq', type=int, default=10)
        parser.add_argument('--avg_loss_alpha', type=float, default=0.986)

        # validation
        parser.add_argument('--val_data_dir', type=str, default='',
                            help='COCO val2017 image folder; skip val if empty')
        parser.add_argument('--val_caption_file', type=str, default='',
                            help='captions_val2017.json; required if val_data_dir set')
        parser.add_argument('--val_color_object_file', type=str, default='',
                            help='Phase 3 val color-object JSONL records')
        parser.add_argument('--val_freq', type=int, default=5,
                            help='run validation every N epochs')

        return parser


class Phase3TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        self.isTrain = False

        parser.add_argument('--full_ckpt', type=str,
                            default='checkpoints/inst_full/80_net_G.pth',
                            help='Phase 2 stage-full 80_net_G.pth')
        parser.add_argument('--clip_arch', type=str,
                            default='ViT-B-32-quickgelu')
        parser.add_argument('--clip_pretrained_path', type=str,
                            default='checkpoints/clip/open_clip_model.safetensors')
        parser.add_argument('--num_heads', type=int, default=4)
        parser.add_argument('--prompt', type=str, default='',
                            help='text prompt for colorization')
        parser.add_argument('--test_img_dir', type=str, default='data/test')
        parser.add_argument('--results_img_dir', type=str,
                            default='results/phase3')
        parser.add_argument('--how_many', type=int, default=200)

        return parser
