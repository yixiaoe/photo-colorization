"""Image quality metrics for colorization evaluation."""
import math
import torch
import torch.nn.functional as F


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR between two RGB [0,1] tensors (N,3,H,W). Returns batch-mean dB."""
    with torch.no_grad():
        mse = F.mse_loss(pred.float(), gt.float(), reduction='none')
        mse = mse.mean(dim=[1, 2, 3])          # (N,)
        psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
        return float(psnr.mean().item())


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor,
                 window_size: int = 11, sigma: float = 1.5) -> float:
    """SSIM between two RGB [0,1] tensors (N,3,H,W). Returns batch-mean."""
    with torch.no_grad():
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pred, gt = pred.float(), gt.float()

        # Gaussian kernel
        coords = torch.arange(window_size, dtype=torch.float32,
                              device=pred.device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.outer(g).unsqueeze(0).unsqueeze(0)   # (1,1,k,k)
        kernel = kernel.expand(pred.shape[1], 1, window_size, window_size)

        pad = window_size // 2
        mu1 = F.conv2d(pred, kernel, padding=pad, groups=pred.shape[1])
        mu2 = F.conv2d(gt,   kernel, padding=pad, groups=gt.shape[1])

        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad,
                             groups=pred.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(gt * gt,     kernel, padding=pad,
                             groups=gt.shape[1]) - mu2_sq
        sigma12   = F.conv2d(pred * gt,   kernel, padding=pad,
                             groups=pred.shape[1]) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim_map.mean().item())
