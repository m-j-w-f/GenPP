# Configuration Structure Documentation

This folder contains the Hydra configuration files for the GenPP project. The configuration is organized hierarchically to promote reusability and maintainability.

## Structure

``` ascii
configs/
├── base_chen.yaml              # Base configuration for Chen model (uses wb2_full_pad_x)
├── base_drn.yaml               # Base configuration for DRN model (uses wb2_full_minmax)
├── base_emos.yaml              # Base configuration for EMOS model (uses wb2_cut)
├── base_engression.yaml        # Base configuration for Engression model (uses wb2_full_pad_x)
├── base_fm_unet.yaml           # Base configuration for FM UNet model (uses wb2_full_pad_xy)
├── base_fm_uvit.yaml           # Base configuration for FM UViT model (uses wb2_full_pad_xy)
├── base_fm_uvit_cfg.yaml       # Base configuration for FM UViT with classifier-free guidance
├── base_autoencoder.yaml       # Base configuration for AutoEncoder (FID)
├── base_classifierencoder.yaml # Base configuration for Classifier (FID)
├── data/                       # Data configurations (flattened, self-contained)
│   ├── wb2_full.yaml               # WeatherBench2 full, standard spatial, standard preprocess
│   ├── wb2_full_minmax.yaml        # WeatherBench2 full, standard spatial, minmax wind preprocess
│   ├── wb2_full_pad_x.yaml         # WeatherBench2 full, padded X, standard preprocess
│   ├── wb2_full_pad_xy.yaml        # WeatherBench2 full, padded X+Y, standard preprocess
│   ├── wb2_cut.yaml                # WeatherBench2 cut (2 vars), no preprocess (for EMOS)
│   ├── icon_full.yaml              # ICON full dataset configuration
│   ├── icon_full_minmax.yaml       # ICON full, minmax wind preprocess
│   ├── icon_full_pad_x.yaml        # ICON full, padded X
│   ├── icon_full_pad_xy.yaml       # ICON full, padded X+Y
│   ├── icon_full_spatial.yaml      # ICON full with spatial preprocessing
│   ├── icon_full_spatial_minmax.yaml # ICON full, spatial + minmax wind preprocess
│   ├── icon_cut.yaml               # ICON cut (reduced vars)
│   ├── ae_dataset.yaml             # AutoEncoder dataset (for FID)
│   ├── classification_dataset.yaml # Classification dataset (for FID)
│   └── dataloader/                 # Dataloader configurations
│       ├── debug.yaml                  # Debug dataloader (small batches)
│       ├── simple.yaml                 # Simple dataloader
│       └── standard.yaml               # Standard dataloader
├── model/                      # Model configurations
│   ├── base_chen.yaml              # Base Chen model config
│   ├── base_engression.yaml        # Base Engression model config
│   ├── base_fm_unet.yaml           # Base FM UNet model config
│   ├── base_fm_uvit.yaml           # Base FM UViT model config
│   ├── base_fm_uvit_cfg.yaml       # Base FM UViT CFG model config
│   ├── cnn_chen_direct.yaml        # CNN-based Chen model (direct)
│   ├── cnn_chen_noise.yaml         # CNN-based Chen model (noise)
│   ├── cnn_engression_direct.yaml  # CNN-based Engression model (direct)
│   ├── cnn_engression_noise.yaml   # CNN-based Engression model (noise)
│   ├── drn.yaml                    # Deep Residual Network model
│   ├── emos.yaml                   # EMOS model configuration
│   ├── fm_unet_direct.yaml         # Flow Matching UNet model (direct)
│   ├── fm_unet_noise.yaml          # Flow Matching UNet model (noise)
│   ├── fm_uvit_direct.yaml         # Flow Matching UViT model (direct)
│   ├── fm_uvit_noise.yaml          # Flow Matching UViT model (noise)
│   ├── fm_uvit_cfg_direct.yaml     # Flow Matching UViT CFG (direct)
│   ├── fm_uvit_cfg_noise.yaml      # Flow Matching UViT CFG (noise)
│   ├── autoencoder.yaml            # AutoEncoder model (for FID)
│   ├── classifierencoder.yaml      # Classifier model (for FID)
│   ├── lr_scheduler/               # Learning rate scheduler configurations
│   │   ├── constant.yaml               # Constant learning rate
│   │   ├── oneCycleLR.yaml             # One cycle learning rate scheduler
│   │   ├── reduceLROnPlateau.yaml      # Reduce LR on plateau scheduler
│   │   └── warmupReduceLROnPlateau.yaml # Warmup + reduce LR on plateau
│   ├── optimizer/                  # Optimizer configurations
│   │   └── adamW.yaml                  # AdamW optimizer configuration
│   └── loss_fn/                    # Loss function configurations
│       ├── energy_score.yaml           # Energy score loss
│       ├── rbf_score.yaml              # RBF kernel score loss
│       ├── patchwise_energy_score.yaml # Patchwise energy score
│       ├── patchwise_rbf_score.yaml    # Patchwise RBF score
│       ├── multiscale_energy_score.yaml # Multiscale energy score
│       ├── multiscale_rbf_score.yaml   # Multiscale RBF score
│       ├── multiscale_patchwise_energy_score.yaml # Multiscale patchwise energy
│       └── multiscale_patchwise_rbf_score.yaml    # Multiscale patchwise RBF
├── trainer/                    # PyTorch Lightning trainer configurations
│   ├── default.yaml                # Standard training setup
│   ├── debug.yaml                  # Debug trainer (fast dev run)
│   ├── debug_gpu.yaml              # Debug trainer with GPU
│   ├── lr_finder.yaml              # Learning rate finder
│   └── overfit.yaml                # Overfit batches (debugging)
├── logger/                     # Logger configurations
│   ├── csv.yaml                    # CSV logger
│   └── wandb.yaml                  # Weights & Biases logger
└── hydra/                      # Hydra framework configurations
    └── standard.yaml               # Standard Hydra configuration
```
