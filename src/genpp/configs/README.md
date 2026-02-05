# Configuration Structure Documentation

This folder contains the Hydra configuration files for the GenPP project. The configuration is organized hierarchically to promote reusability and maintainability.

## Structure

``` ascii
configs/
├── base_chen.yaml           # Base configuration for Chen model (uses wb2_full_pad_x)
├── base_drn.yaml            # Base configuration for DRN model (uses wb2_full_minmax)
├── base_emos.yaml           # Base configuration for EMOS model (uses wb2_cut)
├── base_engression.yaml     # Base configuration for Engression model (uses wb2_full_pad_x)
├── base_fm_unet.yaml        # Base configuration for FM UNet model (uses wb2_full_pad_xy)
├── base_fm_uvit.yaml        # Base configuration for FM UViT model (uses wb2_full_pad_xy)
├── base_autoencoder.yaml    # Base configuration for AutoEncoder (FID)
├── base_classifierencoder.yaml # Base configuration for Classifier (FID)
├── data/                    # Data configurations (flattened, self-contained)
│   ├── wb2_full.yaml            # WeatherBench2 full, standard spatial, standard preprocess
│   ├── wb2_full_minmax.yaml     # WeatherBench2 full, standard spatial, minmax wind preprocess
│   ├── wb2_full_pad_x.yaml      # WeatherBench2 full, padded X, standard preprocess
│   ├── wb2_full_pad_xy.yaml     # WeatherBench2 full, padded X+Y, standard preprocess
│   ├── wb2_cut.yaml             # WeatherBench2 cut (2 vars), no preprocess (for EMOS)
│   ├── icon_full.yaml           # ICON full dataset configuration
│   ├── ae_dataset.yaml          # AutoEncoder dataset (for FID)
│   └── classification_dataset.yaml # Classification dataset (for FID)
├── model/                   # Model configurations
│   ├── cnn_chen_direct.yaml     # CNN-based Chen model
│   ├── drn.yaml                 # Deep Residual Network model
│   ├── emos.yaml                # EMOS model configuration
│   ├── fm_unet_direct.yaml      # Flow Matching UNet model
│   ├── fm_uvit_direct.yaml      # Flow Matching UViT model
│   ├── lr_scheduler/            # Learning rate scheduler configurations
│   │   ├── constant.yaml            # Constant learning rate
│   │   ├── oneCycleLR.yaml          # One cycle learning rate scheduler
│   │   └── warmupCosAnnealing.yaml  # Warmup cosine annealing scheduler
│   ├── optimizer/               # Optimizer configurations
│   │   └── adamW.yaml               # AdamW optimizer configuration
│   └── loss_fn/                 # Loss function configurations
├── trainer/                 # PyTorch Lightning trainer configurations
│   └── default.yaml             # Standard training setup
├── logger/                  # Logger configurations
│   ├── csv.yaml                 # CSV logger
│   ├── none.yaml                # No logging
│   └── wandb.yaml               # Weights & Biases logger
└── hydra/                   # Hydra framework configurations
    └── standard.yaml            # Standard Hydra configuration
```
