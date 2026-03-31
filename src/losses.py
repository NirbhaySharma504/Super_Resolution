"""
Loss functions for ESRGAN training.
VGGPerceptualLoss and relativistic adversarial loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        self.fe = nn.Sequential(*list(vgg.features[:16])).eval()
        for p in self.fe.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        return F.l1_loss(
            self.fe((sr - self.mean) / self.std),
            self.fe((hr - self.mean) / self.std),
        )


def rel_adv_loss(dr, df, is_disc=True):
    """Relativistic average adversarial loss."""
    if is_disc:
        rl = F.binary_cross_entropy_with_logits(
            dr - df.mean(0, keepdim=True), torch.ones_like(dr))
        fl = F.binary_cross_entropy_with_logits(
            df - dr.mean(0, keepdim=True), torch.zeros_like(df))
    else:
        rl = F.binary_cross_entropy_with_logits(
            dr - df.mean(0, keepdim=True), torch.zeros_like(dr))
        fl = F.binary_cross_entropy_with_logits(
            df - dr.mean(0, keepdim=True), torch.ones_like(df))
    return (rl + fl) / 2
