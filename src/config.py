"""
Configuration for Super Resolution ESRGAN pipeline.
All constants and hyperparameters.
"""
import os
from pathlib import Path
import numpy as np
import torch

# --- Paths ---
DATA_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Parquet files ---
PARQUET_FILES = [
    "QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272_LR.parquet",
    "QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540_LR.parquet",
    "QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494_LR.parquet",
]

# --- Image dimensions ---
LR_SIZE = 64
HR_SIZE = 125
NUM_CHANNELS = 3

# --- Training ---
BATCH_SIZE = 16
NUM_WORKERS = 4
PRETRAIN_EPOCHS = 10
GAN_EPOCHS = 20
USE_AMP = True

# --- Optimizers ---
LR_G = 1e-4
LR_D = 1e-4
BETA1 = 0.9
BETA2 = 0.999

# --- Loss weights ---
LAMBDA_PIXEL = 1.0
LAMBDA_PERCEPTUAL = 0.006
LAMBDA_ADVERSARIAL = 0.005

# --- Model architecture ---
NUM_RRDB = 8
NUM_FEATURES = 64
GROWTH_CHANNELS = 32

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
