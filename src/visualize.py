"""
Visualization utilities for ESRGAN Super Resolution.
Dataset samples, training curves, and SR comparisons.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .config import HR_SIZE, OUTPUT_DIR
from .evaluate import compute_metrics

def plot_dataset_samples(test_loader, save_path=None):
    """Plot LR vs HR sample visualization."""
    fig, ax = plt.subplots(4, 6, figsize=(24, 16))
    fig.suptitle("LR (64x64) vs HR (125x125)", fontsize=16, fontweight='bold')
    ch = ["ECAL", "HCAL-in", "HCAL-out"]
    slr, shr, sy = next(iter(test_loader))
    for i in range(4):
        lbl = "Quark" if sy[i].item() == 1 else "Gluon"
        for c in range(3):
            ax[i, c].imshow(slr[i, c].numpy(), cmap='inferno', vmin=0, vmax=0.3)
            ax[i, c].axis('off')
            if i == 0:
                ax[i, c].set_title(f"LR-{ch[c]}", fontweight='bold')
            if c == 0:
                ax[i, c].set_ylabel(lbl)
            ax[i, c + 3].imshow(shr[i, c].numpy(), cmap='inferno', vmin=0, vmax=0.3)
            ax[i, c + 3].axis('off')
            if i == 0:
                ax[i, c + 3].set_title(f"HR-{ch[c]}", fontweight='bold')
    plt.tight_layout()
    if save_path is None:
        save_path = OUTPUT_DIR / "dataset_samples.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_training_curves(pretrain_hist, gan_hist, save_path=None):
    """Plot training loss/metric curves."""
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("ESRGAN Training", fontsize=16, fontweight='bold')
    
    # Pretrain curves
    ax[0,0].plot(pretrain_hist["train_loss"], 'b-', lw=1.5)
    ax[0,0].set_title("Pretrain L1")
    ax[0,0].grid(True, alpha=0.3)
    
    ev = list(range(0, len(pretrain_hist["val_psnr"]) * 5, 5))
    if pretrain_hist["val_psnr"]: 
        ax[0,1].plot(ev, pretrain_hist["val_psnr"], 'g-o', ms=3, lw=1.5)
    ax[0,1].set_title("Pretrain PSNR")
    ax[0,1].grid(True, alpha=0.3)
    
    if pretrain_hist["val_ssim"]: 
        ax[0,2].plot(ev, pretrain_hist["val_ssim"], 'r-o', ms=3, lw=1.5)
    ax[0,2].set_title("Pretrain SSIM")
    ax[0,2].grid(True, alpha=0.3)
    
    # GAN curves
    for k, l in [("g_total", "G"), ("g_pixel", "Pix"), ("g_perceptual", "Perc"), ("g_adversarial", "Adv")]:
        if k in gan_hist:
            ax[1,0].plot(gan_hist[k], label=l, lw=1.5)
    if "d_loss" in gan_hist:
        ax[1,0].plot(gan_hist["d_loss"], label="D", lw=1.5, ls='--')
    ax[1,0].legend(fontsize=8)
    ax[1,0].set_title("GAN Losses")
    ax[1,0].grid(True, alpha=0.3)
    
    ev2 = list(range(0, len(gan_hist.get("val_psnr", [])) * 5, 5))
    if gan_hist.get("val_psnr"): 
        ax[1,1].plot(ev2, gan_hist["val_psnr"], 'g-o', ms=3, lw=1.5)
    ax[1,1].set_title("GAN PSNR")
    ax[1,1].grid(True, alpha=0.3)
    
    if gan_hist.get("val_ssim"): 
        ax[1,2].plot(ev2, gan_hist["val_ssim"], 'r-o', ms=3, lw=1.5)
    ax[1,2].set_title("GAN SSIM")
    ax[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path is None:
        save_path = OUTPUT_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_sr_comparison(model, test_loader, device, num_samples=8, save_path=None):
    """Plot LR | Bicubic | ESRGAN | HR."""
    model.eval()
    ns = num_samples
    fig, ax = plt.subplots(ns, 4, figsize=(20, 5 * ns))
    fig.suptitle("LR | Bicubic | ESRGAN | HR", fontsize=16, fontweight='bold', y=1.01)
    ct = ["LR(64)", "Bicubic(125)", "ESRGAN(125)", "HR(125)"]
    
    with torch.no_grad():
        lb, hb, yb = next(iter(test_loader))
        lb, hb, yb = lb[:ns], hb[:ns], yb[:ns]
        sb = model(lb.to(device)).cpu()
        bb = F.interpolate(lb, size=HR_SIZE, mode='bicubic', align_corners=False)
        
    for i in range(min(ns, lb.shape[0])):
        # Visualize the first channel (ECAL) as an example
        imgs = [lb[i, 0].numpy(), np.clip(bb[i, 0].numpy(), 0, 1), np.clip(sb[i, 0].numpy(), 0, 1), hb[i, 0].numpy()]
        lbl = "Q" if yb[i].item() == 1 else "G"
        for j, im in enumerate(imgs):
            ax[i, j].imshow(im, cmap='inferno', vmin=0, vmax=0.3)
            ax[i, j].axis('off')
            if i == 0: 
                ax[i, j].set_title(ct[j], fontsize=12, fontweight='bold')
        
        # Compute metrics over all channels for the annotation
        sm = compute_metrics(sb[i].numpy(), hb[i].numpy())
        bm = compute_metrics(bb[i].numpy(), hb[i].numpy())
        
        ax[i, 0].text(.02, .98, lbl, transform=ax[i, 0].transAxes, fontsize=10, va='top', color='w', bbox=dict(boxstyle='round', fc='k', alpha=.7))
        ax[i, 1].text(.02, .98, f"{bm['psnr']:.1f}dB", transform=ax[i, 1].transAxes, fontsize=9, va='top', color='w', bbox=dict(boxstyle='round', fc='k', alpha=.7))
        ax[i, 2].text(.02, .98, f"{sm['psnr']:.1f}dB", transform=ax[i, 2].transAxes, fontsize=9, va='top', color='w', bbox=dict(boxstyle='round', fc='k', alpha=.7))
        
    plt.tight_layout()
    if save_path is None:
        save_path = OUTPUT_DIR / "sr_comparisons.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_channel_comparison(model, test_loader, device, num_samples=4, save_path=None):
    """Plot SR vs HR per channel."""
    model.eval()
    chn = ["ECAL", "HCAL-in", "HCAL-out"]
    nc = num_samples
    fig, ax = plt.subplots(nc, 6, figsize=(30, 5 * nc))
    fig.suptitle("SR vs HR per channel", fontsize=16, fontweight='bold', y=1.01)
    
    with torch.no_grad():
        l2, h2, _ = next(iter(test_loader))
        l2, h2 = l2[:nc], h2[:nc]
        s2 = model(l2.to(device)).cpu()
        
    for i in range(min(nc, l2.shape[0])):
        for c in range(3):
            ax[i, c*2].imshow(np.clip(s2[i, c].numpy(), 0, 1), cmap='inferno', vmin=0, vmax=0.3)
            ax[i, c*2].axis('off')
            if i == 0: 
                ax[i, c*2].set_title(f"SR-{chn[c]}")
                
            ax[i, c*2+1].imshow(h2[i, c].numpy(), cmap='inferno', vmin=0, vmax=0.3)
            ax[i, c*2+1].axis('off')
            if i == 0: 
                ax[i, c*2+1].set_title(f"HR-{chn[c]}")
                
    plt.tight_layout()
    if save_path is None:
        save_path = OUTPUT_DIR / "channel_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_metric_distributions(model, test_loader, device, save_path=None):
    """Plot PSNR and SSIM distributions by class and method."""
    model.eval()
    pq_, pg_, sq_, sg_, bpq, bpg, bsq, bsg = [], [], [], [], [], [], [], []
    
    with torch.no_grad():
        for lr, hr, y in test_loader:
            sr = model(lr.to(device)).cpu().numpy()
            bic = F.interpolate(lr, size=HR_SIZE, mode='bicubic', align_corners=False).numpy()
            hn = hr.numpy()
            yn = y.numpy()
            for i in range(sr.shape[0]):
                ms = compute_metrics(sr[i], hn[i])
                mb = compute_metrics(bic[i], hn[i])
                if yn[i] == 1: 
                    pq_.append(ms["psnr"]); sq_.append(ms["ssim"])
                    bpq.append(mb["psnr"]); bsq.append(mb["ssim"])
                else: 
                    pg_.append(ms["psnr"]); sg_.append(ms["ssim"])
                    bpg.append(mb["psnr"]); bsg.append(mb["ssim"])
                    
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Metrics Distributions", fontsize=16, fontweight='bold')
    
    ax[0,0].hist(pq_, 40, alpha=.6, label=f"Q({np.mean(pq_):.1f})", color='blue')
    ax[0,0].hist(pg_, 40, alpha=.6, label=f"G({np.mean(pg_):.1f})", color='red')
    ax[0,0].set_title("PSNR by class")
    ax[0,0].legend()
    ax[0,0].grid(True, alpha=.3)
    
    ax[0,1].hist(sq_, 40, alpha=.6, label=f"Q({np.mean(sq_):.4f})", color='blue')
    ax[0,1].hist(sg_, 40, alpha=.6, label=f"G({np.mean(sg_):.4f})", color='red')
    ax[0,1].set_title("SSIM by class")
    ax[0,1].legend()
    ax[0,1].grid(True, alpha=.3)
    
    ap = pq_ + pg_
    abp = bpq + bpg
    ax[1,0].hist(ap, 40, alpha=.6, label=f"ESRGAN({np.mean(ap):.1f})", color='green')
    ax[1,0].hist(abp, 40, alpha=.6, label=f"Bic({np.mean(abp):.1f})", color='orange')
    ax[1,0].set_title("PSNR: ESRGAN vs Bicubic")
    ax[1,0].legend()
    ax[1,0].grid(True, alpha=.3)
    
    an = sq_ + sg_
    abn = bsq + bsg
    ax[1,1].hist(an, 40, alpha=.6, label=f"ESRGAN({np.mean(an):.4f})", color='green')
    ax[1,1].hist(abn, 40, alpha=.6, label=f"Bic({np.mean(abn):.4f})", color='orange')
    ax[1,1].set_title("SSIM: ESRGAN vs Bicubic")
    ax[1,1].legend()
    ax[1,1].grid(True, alpha=.3)
    
    plt.tight_layout()
    if save_path is None:
        save_path = OUTPUT_DIR / "metric_distributions.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_difference_maps(model, test_loader, device, num_samples=4, save_path=None):
    """Plot absolute difference maps between HR and SR/Bicubic."""
    model.eval()
    ns = num_samples
    fig, ax = plt.subplots(ns, 3, figsize=(15, 5 * ns))
    fig.suptitle("Error Maps: HR vs Bicubic & ESRGAN", fontsize=16, fontweight='bold', y=1.01)
    
    with torch.no_grad():
        lb, hb, _ = next(iter(test_loader))
        lb, hb = lb[:ns], hb[:ns]
        sb = model(lb.to(device)).cpu()
        bb = F.interpolate(lb, size=HR_SIZE, mode='bicubic', align_corners=False)
        
    for i in range(min(ns, lb.shape[0])):
        # We compute mean absolute error across the 3 channels
        err_bicubic = np.mean(np.abs(hb[i].numpy() - bb[i].numpy()), axis=0)
        err_esrgan = np.mean(np.abs(hb[i].numpy() - sb[i].numpy()), axis=0)
        hr_mean = np.mean(hb[i].numpy(), axis=0)
        
        # Display HR
        ax[i, 0].imshow(hr_mean, cmap='inferno')
        ax[i, 0].axis('off')
        if i == 0: ax[i, 0].set_title("HR (Mean over channels)", fontsize=12)
        
        # Display Bicubic Error
        im1 = ax[i, 1].imshow(err_bicubic, cmap='magma')
        ax[i, 1].axis('off')
        if i == 0: ax[i, 1].set_title("Bicubic Abs Error", fontsize=12)
        fig.colorbar(im1, ax=ax[i, 1], fraction=0.046, pad=0.04)
        
        # Display ESRGAN Error
        im2 = ax[i, 2].imshow(err_esrgan, cmap='magma')
        ax[i, 2].axis('off')
        if i == 0: ax[i, 2].set_title("ESRGAN Abs Error", fontsize=12)
        fig.colorbar(im2, ax=ax[i, 2], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    if save_path is None:
        save_path = OUTPUT_DIR / "difference_maps.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
