"""
Training loops for ESRGAN.
L1 pretraining and GAN fine-tuning with AMP support.
"""
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import (
    LR_G, LR_D, BETA1, BETA2, USE_AMP, OUTPUT_DIR,
    LAMBDA_PIXEL, LAMBDA_PERCEPTUAL, LAMBDA_ADVERSARIAL,
)
from .evaluate import evaluate
from .losses import VGGPerceptualLoss, rel_adv_loss


def train_pretrain(gen, tl, vl, dev, epochs):
    """Phase 1: L1 Pre-training (AMP enabled)."""
    print("--- Phase 1: L1 Pre-training (AMP) ---")
    opt = optim.Adam(gen.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, verbose=True)
    scaler = GradScaler(enabled=USE_AMP)
    hist = {"train_loss": [], "val_psnr": [], "val_ssim": []}
    best = 0
    for ep in range(1, epochs + 1):
        gen.train()
        el = 0
        t0 = time.time()
        for lr, hr, _ in tqdm(tl, desc=f"Pretrain Ep {ep}/{epochs}", leave=False):
            lr, hr = lr.to(dev), hr.to(dev)
            with autocast(enabled=USE_AMP):
                loss = F.l1_loss(gen(lr), hr)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            el += loss.item()
        hist["train_loss"].append(el / len(tl))
        if ep % 5 == 0 or ep == 1:
            vm = evaluate(gen, vl, dev)
            hist["val_psnr"].append(vm["psnr"])
            hist["val_ssim"].append(vm["ssim"])
            sch.step(-vm["psnr"])
            print(f"  Ep{ep:3d}/{epochs} L1:{el / len(tl):.6f} PSNR:{vm['psnr']:.2f} SSIM:{vm['ssim']:.4f} {time.time() - t0:.1f}s")
            if vm["psnr"] > best:
                best = vm["psnr"]
                torch.save(gen.state_dict(), OUTPUT_DIR / "gen_pretrained.pth")
    print(f"  Best PSNR: {best:.2f}")
    return hist


def train_gan(gen, disc, tl, vl, dev, epochs):
    """Phase 2: GAN fine-tuning (AMP enabled)."""
    print("--- Phase 2: GAN (AMP) ---")
    ploss = VGGPerceptualLoss().to(dev)
    og = optim.Adam(gen.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    od = optim.Adam(disc.parameters(), lr=LR_D, betas=(BETA1, BETA2))
    sg = optim.lr_scheduler.ReduceLROnPlateau(og, patience=10, factor=0.5, verbose=True)
    sd = optim.lr_scheduler.ReduceLROnPlateau(od, patience=10, factor=0.5, verbose=True)
    scg = GradScaler(enabled=USE_AMP)
    scd = GradScaler(enabled=USE_AMP)
    hist = {
        "g_total": [], "g_pixel": [], "g_perceptual": [],
        "g_adversarial": [], "d_loss": [], "val_psnr": [], "val_ssim": [],
    }
    best = 0
    for ep in range(1, epochs + 1):
        gen.train()
        disc.train()
        eg, ed, egp, egpc, ega = 0, 0, 0, 0, 0
        t0 = time.time()
        for lr, hr, _ in tl:
            lr, hr = lr.to(dev), hr.to(dev)
            with autocast(enabled=USE_AMP):
                sr = gen(lr)
                dr = disc(hr)
                df = disc(sr.detach())
                dl_ = rel_adv_loss(dr, df, True)
            od.zero_grad(set_to_none=True)
            scd.scale(dl_).backward()
            scd.step(od)
            scd.update()
            with autocast(enabled=USE_AMP):
                dr2 = disc(hr).detach()
                df2 = disc(sr)
                lp = F.l1_loss(sr, hr)
                lpc = ploss(sr, hr)
                la = rel_adv_loss(dr2, df2, False)
                gl = LAMBDA_PIXEL * lp + LAMBDA_PERCEPTUAL * lpc + LAMBDA_ADVERSARIAL * la
            og.zero_grad(set_to_none=True)
            scg.scale(gl).backward()
            scg.step(og)
            scg.update()
            eg += gl.item()
            ed += dl_.item()
            egp += lp.item()
            egpc += lpc.item()
            ega += la.item()
        nb = len(tl)
        hist["g_total"].append(eg / nb)
        hist["g_pixel"].append(egp / nb)
        hist["g_perceptual"].append(egpc / nb)
        hist["g_adversarial"].append(ega / nb)
        hist["d_loss"].append(ed / nb)
        if ep % 5 == 0 or ep == 1:
            vm = evaluate(gen, vl, dev)
            hist["val_psnr"].append(vm["psnr"])
            hist["val_ssim"].append(vm["ssim"])
            sg.step(-vm["psnr"])
            sd.step(-vm["psnr"])
            print(
                f"  Ep{ep:3d}/{epochs} G:{eg / nb:.4f}"
                f"(p:{egp / nb:.4f} pc:{egpc / nb:.4f} a:{ega / nb:.4f}) "
                f"D:{ed / nb:.4f} PSNR:{vm['psnr']:.2f} SSIM:{vm['ssim']:.4f} "
                f"{time.time() - t0:.1f}s"
            )
            if vm["psnr"] > best:
                best = vm["psnr"]
                torch.save(
                    {"gen": gen.state_dict(), "disc": disc.state_dict()},
                    OUTPUT_DIR / "esrgan_best.pth",
                )
    print(f"  Best PSNR: {best:.2f}")
    return hist
