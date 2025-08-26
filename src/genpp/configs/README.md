# Configuration Structure Documentation

This folder contains the Hydra configuration files for the GenPP project. The configuration is organized hierarchically to promote reusability and maintainability.

## Caveats

When using the _fast weatherbench datasets, the batchsize is not determied by `data.dataloader.batch_size` but by `data.batch_size`. This is due to the fact that when the data is batched by xbatchers MapDataset the Dataloader itself should receive `None` as the batch dim. But for the fast variants, a TensorDataset is used under the hood, where the Dataloader determines the batch size.

## Structure

``` ascii
configs/
├── base_chen.yaml           # Base configuration for Chen model
├── base_drn.yaml           # Base configuration for DRN model
├── base_emos.yaml          # Base configuration for EMOS model
├── base_fmunet.yaml        # Base configuration for FMUNet model
├── data/                   # Data-related configurations
│   ├── weatherbench2_cut.yaml      # Cut WeatherBench2 dataset config (t2m + ws only)
│   ├── weatherbench2_cut_fast.yaml # Fast version of cut WeatherBench2 dataset config
│   ├── weatherbench2_full.yaml     # Full WeatherBench2 dataset config (all variables)
│   ├── weatherbench2_full_fast.yaml # Fast version of full WeatherBench2 dataset config
│   ├── dataloader/         # DataLoader configurations
│   │   └── standard.yaml       # Standard dataloader configuration
│   ├── spatial/            # Spatial dimension configurations
│   │   ├── padded_xy.yaml      # Padded configuration for x and y
│   │   ├── padded_y.yaml       # Padded configuration for y only
│   │   ├── patchwise.yaml      # Patchwise spatial configuration
│   │   └── standard.yaml       # Standard spatial configuration
│   ├── splits/             # Train/val/test split configurations
│   │   └── standard.yaml       # Standard data splits
│   └── preprocess/         # Preprocessing pipeline configurations
│       ├── minMaxWind.yaml     # MinMax preprocessing for wind variables
│       ├── none.yaml           # No preprocessing
│       └── standard.yaml       # Standard preprocessing pipeline
├── model/                  # Model configurations
│   ├── cnn_chen.yaml       # CNN-based Chen model
│   ├── drn.yaml            # Deep Residual Network model
│   ├── emos.yaml           # EMOS model configuration
│   ├── fm_cnn.yaml         # Flow Matching CNN model
│   ├── lr_scheduler/       # Learning rate scheduler configurations
│   │   ├── constant.yaml       # Constant learning rate
│   │   ├── oneCycleLR.yaml     # One cycle learning rate scheduler
│   │   └── warmupCosAnnealing.yaml # Warmup cosine annealing scheduler
│   └── optimizer/          # Optimizer configurations
│       └── adamW.yaml          # AdamW optimizer configuration
├── trainer/                # PyTorch Lightning trainer configurations
│   └── default.yaml        # Standard training setup
├── logger/                 # Logger configurations
│   ├── csv.yaml            # CSV logger
│   ├── none.yaml           # No logging
│   └── wandb.yaml          # Weights & Biases logger
└── hydra/                  # Hydra framework configurations
    └── standard.yaml       # Standard Hydra configuration
```
