"""TensorBoard visualizer for training."""
import os
import torch
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    def __init__(self, opt):
        log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.name = opt.name

    def plot_losses(self, losses: dict, step: int):
        for tag, val in losses.items():
            self.writer.add_scalar(f'loss/{tag}', val, step)

    def plot_images(self, imgs: dict, step: int):
        for tag, tensor in imgs.items():
            # tensor: B×C×H×W, values in [0,1]
            self.writer.add_images(tag, tensor.clamp(0, 1), step)

    def close(self):
        self.writer.close()
