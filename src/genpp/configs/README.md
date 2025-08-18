# Configuration Structure Documentation

This folder contains the Hydra configuration files for the GenPP project. The configuration is organized hierarchically to promote reusability and maintainability.

## Structure

```
configs/
├── config.yaml             # Main configuration file
├── data/                   # Data-related configurations
│   ├── weatherbench2.yaml      # Main WeatherBench2 dataset config (variable selection support)
│   ├── weatherbench2_full.yaml # Full WeatherBench2 dataset config (all variables)
│   ├── weatherbench2_cut.yaml  # Cut WeatherBench2 dataset config (t2m + ws only)
│   ├── dataloader/         # DataLoader configurations
│   ├── spatial/            # Spatial dimension configurations
│   ├── splits/             # Train/val/test split configurations
│   └── preprocess/         # Preprocessing pipeline configurations
├── model/                  # Model configurations
│   ├── cnn_chen.yaml       # CNN-based Chen model
│   └── fc_chen.yaml        # Fully connected Chen model
├── trainer/                # PyTorch Lightning trainer configurations
│   └── default.yaml        # Standard training setup
├── logger/                 # Logger configurations
│   ├── default.yaml        # Weights & Biases logger
│   ├── csv.yaml            # CSV logger
│   └── none.yaml           # No logging
└── preprocess/             # Preprocessing pipeline configurations
    └── standard.yaml       # Standard preprocessing pipeline
```
