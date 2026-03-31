"""
Data pipeline for Super Resolution ESRGAN.
Memory-efficient Dataset (memmap, log1p normalization), Arrow-native I/O, data loaders.
"""
import os
import gc
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

from .config import (
    DATA_DIR, NUM_CHANNELS, LR_SIZE, HR_SIZE,
    PARQUET_FILES, BATCH_SIZE, NUM_WORKERS,
)


class JetImageDataset(Dataset):
    """Memory-efficient Dataset: memmap on disk, log1p normalize on-the-fly."""

    def __init__(self, lr_path, hr_path, y_path, indices, norm_stats=None, augment=False):
        self.lr_path, self.hr_path, self.y_path = lr_path, hr_path, y_path
        self.lr, self.hr, self.y = None, None, None
        self.indices, self.norm_stats, self.augment = indices, norm_stats, augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.lr is None:
            self.lr = np.load(self.lr_path, mmap_mode='r')
            self.hr = np.load(self.hr_path, mmap_mode='r')
            self.y = np.load(self.y_path, mmap_mode='r')
        i = self.indices[idx]
        lr = torch.from_numpy(np.array(self.lr[i], dtype=np.float32, copy=True))
        hr = torch.from_numpy(np.array(self.hr[i], dtype=np.float32, copy=True))
        label = torch.tensor(float(self.y[i]))
        if self.norm_stats is not None:
            for c in range(lr.shape[0]):
                cmax = self.norm_stats[c]['max']
                lr[c] = torch.log1p(lr[c].clamp(min=0)) / np.log1p(cmax)
                hr[c] = torch.log1p(hr[c].clamp(min=0)) / np.log1p(cmax)
        if self.augment and torch.rand(1).item() > 0.5:
            lr = torch.flip(lr, [-1])
            hr = torch.flip(hr, [-1])
        if self.augment and torch.rand(1).item() > 0.5:
            lr = torch.flip(lr, [-2])
            hr = torch.flip(hr, [-2])
        return lr, hr, label


def preprocess_parquet_to_npy(parquet_files, npy_dir):
    """Parquet -> memmap .npy via Arrow flatten (no to_pylist). Cached."""
    npy_dir = Path(npy_dir)
    npy_dir.mkdir(exist_ok=True)
    lr_p = str(npy_dir / "all_lr.npy")
    hr_p = str(npy_dir / "all_hr.npy")
    y_p = str(npy_dir / "all_y.npy")
    if all(os.path.exists(p) for p in [lr_p, hr_p, y_p]):
        n = np.load(y_p, mmap_mode='r').shape[0]
        print(f"  Cached ({n} samples)")
        return lr_p, hr_p, y_p, n
    total = sum(pq.ParquetFile(str(DATA_DIR / f)).metadata.num_rows for f in parquet_files)
    print(f"  Total rows: {total}")
    lr_mm = np.lib.format.open_memmap(lr_p, mode='w+', dtype=np.float32,
                                       shape=(total, NUM_CHANNELS, LR_SIZE, LR_SIZE))
    hr_mm = np.lib.format.open_memmap(hr_p, mode='w+', dtype=np.float32,
                                       shape=(total, NUM_CHANNELS, HR_SIZE, HR_SIZE))
    y_mm = np.lib.format.open_memmap(y_p, mode='w+', dtype=np.float32, shape=(total,))

    def arrow_to_np(col, spatial_size):
        flat = col
        for _ in range(3):
            flat = flat.flatten()
        return flat.to_numpy(zero_copy_only=False).astype(np.float32).reshape(
            -1, NUM_CHANNELS, spatial_size, spatial_size)

    off = 0
    last_print = 0
    for fp in parquet_files:
        print(f"  {fp}...")
        for batch in pq.ParquetFile(str(DATA_DIR / fp)).iter_batches(batch_size=500):
            bs = batch.num_rows
            lr_mm[off:off + bs] = arrow_to_np(batch.column("X_jets_LR"), LR_SIZE)
            hr_mm[off:off + bs] = arrow_to_np(batch.column("X_jets"), HR_SIZE)
            y_mm[off:off + bs] = batch.column("y").to_numpy(zero_copy_only=False).astype(np.float32)
            off += bs
            if off - last_print >= 5000:
                print(f"    {off}/{total}")
                last_print = off
    del lr_mm, hr_mm, y_mm
    gc.collect()
    print(f"  Done!")
    return lr_p, hr_p, y_p, total


def compute_norm_stats(lr_path, hr_path, n_train, chunk=5000):
    """Per-channel max sequentially from TRAINING data to avoid disk thrashing."""
    lr = np.load(lr_path, mmap_mode='r')
    hr = np.load(hr_path, mmap_mode='r')
    stats = {}
    for c in range(NUM_CHANNELS):
        cmax = float('-inf')
        for s in range(0, n_train, chunk):
            e = min(s + chunk, n_train)
            cmax = max(cmax, float(lr[s:e, c].max()), float(hr[s:e, c].max()))
        stats[c] = {'max': cmax}
        print(f"  Ch{c}: [Max: {cmax:.4f}]")
    del lr, hr
    return stats


def create_data_loaders(parquet_files=None, data_dir=None):
    """
    Preprocess data, compute norm stats, create train/val/test DataLoaders.
    Returns: (train_loader, val_loader, test_loader, norm_stats, lr_path, hr_path, y_path, n_total)
    """
    if parquet_files is None:
        parquet_files = PARQUET_FILES
    if data_dir is None:
        data_dir = DATA_DIR

    print("Preprocessing parquet -> memmap .npy...")
    lr_path, hr_path, y_path, n_total = preprocess_parquet_to_npy(parquet_files, data_dir / "npy_cache")

    lr_mmap = np.load(lr_path, mmap_mode='r')
    hr_mmap = np.load(hr_path, mmap_mode='r')
    y_mmap = np.load(y_path, mmap_mode='r')
    print(f"Samples: {n_total} | LR:{lr_mmap.shape} HR:{hr_mmap.shape}")
    print(f"Quarks: {(y_mmap == 1).sum():.0f} | Gluons: {(y_mmap == 0).sum():.0f}")

    # Split FIRST, then compute norms from train only
    sub_total = min(35000, n_total)
    n_train = int(0.8 * sub_total)
    n_val = int(0.1 * sub_total)

    indices = np.arange(sub_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:sub_total]

    print("\nNorm stats (train only):")
    norm_stats = compute_norm_stats(lr_path, hr_path, n_train)

    train_ds = JetImageDataset(lr_path, hr_path, y_path, train_idx, norm_stats, augment=True)
    val_ds = JetImageDataset(lr_path, hr_path, y_path, val_idx, norm_stats)
    test_ds = JetImageDataset(lr_path, hr_path, y_path, test_idx, norm_stats)
    kw = dict(num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, **kw)
    print(f"Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")

    return train_loader, val_loader, test_loader, norm_stats, lr_path, hr_path, y_path, n_total
