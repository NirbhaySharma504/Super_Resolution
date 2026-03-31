# Super Resolution of CMS Calorimeter Jets using ESRGAN (GSoC Task 2b)

This repository contains the implementation of an Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) designed to upscale 64x64 resolution jet images to 125x125 matrices. The dataset consists of three-channel images (Tracks, ECAL, HCAL) for two classes of particles: quarks and gluons.

## 🚀 Quickstart & How to Run

To quickly reproduce the evaluation results without retraining from scratch, follow these steps:

1. **Download the Dataset:** Download the required CMS jet `.parquet` files and place them in the root directory alongside the Jupyter notebook.
2. **Fetch Model Weights:** The pre-trained model weights (`esrgan_best.pth`, ~50MB) are included in this repository. Ensure they are located in the `outputs/` directory.
3. **Run the Notebook:** Open and execute all cells in `Task_2b_Super_Resolution_GAN.ipynb`. The notebook will automatically:
    * Parse the raw `.parquet` files and cache them into memory-mapped `.npy` files for efficient loading.
    * Load the pre-trained generator weights from the `outputs/` folder.
    * Run the evaluation to display the visual and quantitative comparisons (PSNR, SSIM, MAE).

## 📁 Project Structure

Ensure your working directory matches the following structure before running the notebook:

```text
├── outputs/
│   ├── esrgan_best.pth      # Included pre-trained GAN weights
│   └── pretrain_best.pth    # Included L1 pre-training weights
├── src/
│   ├── config.py            # Centralized hyperparameters and paths
│   ├── data.py              # Memmap dataset loading & Arrow-native Parquet reading
│   ├── evaluate.py          # PSNR, SSIM, and MAE metric utilities
│   ├── losses.py            # VGG Perceptual & Relativistic Adversarial Loss
│   ├── models.py            # ESRGAN Generator (RRDB) & VGG Discriminator
│   ├── train.py             # Phase 1 & Phase 2 training loops (AMP support)
│   └── visualize.py         # Plotting utilities for training curves and SR comparison
├── QCDToGGQQ_IMGjet_RH1all_jet0_run...parquet  # Raw datasets (Place these here)
└── Task_2b_Super_Resolution_GAN.ipynb          # Main execution notebook
```

The project is modularized for clarity and reproducibility:
* `src/config.py`: Centralized hyperparameters, path definitions, and device configuration.
* `src/data.py`: Memory-efficient dataset loading using `numpy.memmap` and Arrow-native Parquet reading to handle hardware constraints smoothly.
* `src/models.py`: Defines the ESRGAN Generator (utilizing Residual-in-Residual Dense Blocks without Batch Norm) and the VGG-style Discriminator.
* `src/losses.py`: Implementation of the VGG Perceptual Loss and the Relativistic Average Adversarial Loss.
* `src/train.py`: Training loops separated into Phase 1 (L1 Pre-training) and Phase 2 (GAN fine-tuning) with Automatic Mixed Precision (AMP) support.
* `src/evaluate.py`: Utilities for computing PSNR, SSIM, and MAE metrics.

## Performance & Results

The ESRGAN model was evaluated against a standard Bicubic interpolation baseline on a held-out test set. The GAN successfully learned to reconstruct high-frequency details, resulting in sharper energy deposits that closely mimic the ground-truth high-resolution data.

| Metric | Bicubic Baseline | ESRGAN (Ours) | Absolute Gain |
| :--- | :--- | :--- | :--- |
| **PSNR** | 44.35 dB | **45.28 dB** | + 0.93 dB |
| **SSIM** | 0.9828 | **0.9847** | + 0.0019 |
| **MAE** | 0.000665 | **0.000414** | - 0.000251 |

## Hardware Considerations

To train effectively on limited hardware resources, the following optimizations were made:
1.  **On-the-fly Memory Mapping:** Raw parquet data is cached to `.npy` files and loaded via `mmap_mode='r'` to prevent out-of-memory (OOM) errors during data loading.
2.  **Automatic Mixed Precision (AMP):** Utilizing `torch.cuda.amp` to reduce VRAM footprint and speed up training without sacrificing gradient precision.