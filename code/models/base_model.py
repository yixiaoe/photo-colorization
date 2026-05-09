"""Abstract base class for all models."""
import os
import torch


class BaseModel:
    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device(
            f'cuda:{self.gpu_ids[0]}' if self.gpu_ids else 'cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

    # ── interface (subclasses must implement) ─────────────────────────────
    def name(self):
        return 'BaseModel'

    def set_input(self, data):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def optimize_parameters(self):
        raise NotImplementedError

    def get_current_losses(self):
        return {}

    def get_current_visuals(self):
        return {}

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    # ── checkpoint helpers ────────────────────────────────────────────────
    def save_networks(self, epoch):
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            path = os.path.join(self.save_dir, f'{epoch}_net_{name}.pth')
            torch.save(net.state_dict(), path)
            latest = os.path.join(self.save_dir, f'latest_net_{name}.pth')
            torch.save(net.state_dict(), latest)

    def load_networks(self, epoch):
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            path = os.path.join(self.save_dir, f'{epoch}_net_{name}.pth')
            if not os.path.isfile(path):
                path = os.path.join(self.save_dir, f'latest_net_{name}.pth')
            if os.path.isfile(path):
                state = torch.load(path, map_location=self.device)
                net.load_state_dict(state)
                print(f'Loaded {path}')
            else:
                print(f'[Warning] checkpoint not found: {path}')

    # ── scheduler helpers ─────────────────────────────────────────────────
    def setup_schedulers(self):
        self.schedulers = [
            self._get_scheduler(opt) for opt in self.optimizers]

    def _get_scheduler(self, optimizer):
        opt = self.opt
        if opt.lr_policy == 'lambda':
            def rule(epoch):
                return 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)
        elif opt.lr_policy == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        else:
            raise NotImplementedError(opt.lr_policy)

    def update_learning_rate(self):
        for sch in self.schedulers:
            sch.step()

    def train(self):
        for name in self.model_names:
            getattr(self, 'net' + name).train()

    def eval(self):
        for name in self.model_names:
            getattr(self, 'net' + name).eval()
