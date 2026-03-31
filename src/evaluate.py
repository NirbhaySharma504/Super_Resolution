"""
Evaluation utilities for ESRGAN.
Computes PSNR, SSIM, MAE metrics.
"""
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from .config import HR_SIZE


def compute_metrics(sr, hr):
    """Compute PSNR, SSIM, MAE for a single image pair (C,H,W numpy arrays)."""
    s = np.clip(np.transpose(sr, (1, 2, 0)), 0, 1)
    h = np.clip(np.transpose(hr, (1, 2, 0)), 0, 1)
    return {
        "psnr": psnr_metric(h, s, data_range=1.),
        "ssim": ssim_metric(h, s, data_range=1., channel_axis=2),
        "mae": np.mean(np.abs(s - h)),
    }


@torch.no_grad()
def evaluate(model, dl, dev):
    """Evaluate model on a DataLoader, returning average PSNR/SSIM/MAE."""
    model.eval()
    tp, ts, tm, n = 0., 0., 0., 0
    for lr, hr, _ in dl:
        sr = model(lr.to(dev)).cpu().numpy()
        hn = hr.numpy()
        for i in range(sr.shape[0]):
            m = compute_metrics(sr[i], hn[i])
            tp += m["psnr"]
            ts += m["ssim"]
            tm += m["mae"]
            n += 1
    return {"psnr": tp / n, "ssim": ts / n, "mae": tm / n}


@torch.no_grad()
def bicubic_baseline(dl):
    """Compute bicubic interpolation baseline metrics."""
    tp, ts, tm, n = 0., 0., 0., 0
    for lr, hr, _ in dl:
        b = F.interpolate(lr, size=HR_SIZE, mode='bicubic', align_corners=False).numpy()
        hn = hr.numpy()
        for i in range(b.shape[0]):
            m = compute_metrics(b[i], hn[i])
            tp += m["psnr"]
            ts += m["ssim"]
            tm += m["mae"]
            n += 1
    return {"psnr": tp / n, "ssim": ts / n, "mae": tm / n}
