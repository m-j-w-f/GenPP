# GenPP: Spatially Coherent Probabilistic Weather Forecasts via Deep Generative Post-Processing

This repository contains code for post-processing numerical weather predictions using deep generative models to produce spatially coherent probabilistic forecasts.

## Environment

This project uses [Pixi](https://prefix.dev/docs/pixi/) for reproducible environment management.

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Create and activate the environment
pixi install

# For GPU support (Linux with CUDA 12.4+)
pixi install -e gpu
```

## Data

### WeatherBench 2

Place WeatherBench 2 data in:

``` bash
src/genpp/data/weatherbench2/
```

#### Downloading

A download script exist:

``` bash
src/genpp/data/weatherbench2/download.py
```

As well as a script for flattening the downloaded data:

``` bash
src/genpp/data/weatherbench2/flat_and_aggr.py
```

### ICON

Place ICON data in:

``` bash
src/genpp/data/icon/data/
```

#### Regridding

Scripts for regriding exist and should be run in that order:

``` bash
src/genpp/data/icon/cdo_scripts/get_remap_grid_ens.sh
src/genpp/data/icon/cdo_scripts/get_remap_grid_rea.sh

src/genpp/data/icon/cdo_scripts/launch_interpolate_ens_jobs.sh
src/genpp/data/icon/cdo_scripts/launch_interpolate_ens_members_jobs.sh
src/genpp/data/icon/cdo_scripts/launch_interpolate_rea_jobs.sh
```

#### Convert to PyTorch Tensors

Scripts for conversion exist and should be run before model trianing once.

``` bash
src/genpp/data/icon/scripts/launch_tensor_jobs.sh
```

## Models

Model implementations are in `src/genpp/models/`:

### Conditional Generative Models (`models/cgm/`)

- **Flow Matching** (`cgm/fm/`) - Flow matching models with UNet (`fm_cnn.py`) and UViT (`fm_uvit.py`) backbones, including classifier-free guidance (`cfg.py`)
- **Chen** (`cgm/chen/chen.py`) - CNN-based generative model
- **Engression** (`cgm/engression/`) - Engression model with CNN backbone (`cnn.py`)
- **Utilities** (`cgm/utils/`) - Base generative module (`base_generative_module.py`) and temperature-dependent scaling (`td_scaling.py`)

### Distributional Regression (`models/distributionalRegression/`)

- **DRN** (`drn.py`) - Deep Residual Network for distributional regression
- **EMOS** (`emos.py`) - Ensemble Model Output Statistics baseline
- **Distributions** (`distributions.py`) - Parametric distribution implementations
- **Meta** (`meta.py`) - Meta-learning utilities

### Scoring Rules (`models/scores/`)

- **Energy Score** (`energy.py`) - Energy score implementation
- **CRPS** (`crps.py`) - Continuous Ranked Probability Score
- **RBF Score** (`rbf.py`) - Radial basis function kernel score
- **Variogram** (`variogram.py`) - Variogram-based scoring
- **Kernels** (`kernels/`) - Kernel implementations (L2, RBF)

### Other

- `base_module.py` - Base Lightning module
- `layers.py` - Custom neural network layers
- `loss.py` - Loss function implementations

## Data

Data loading and preprocessing are in `src/genpp/data/`:

### WeatherBench 2 (`data/weatherbench2/`)

- `zarr_dataset.py` - Dataset for the FID models
- `fast_dataset_simple.py` - Dataset for post-processing models
- `flat_and_aggr.py` - Flattening and Aggregation of data
- `download.py` - Data download utilities

**Data path:** `src/genpp/data/weatherbench2/`

### ICON (`data/icon/`)

- `dataset.py` - ICON dataset implementation
- `scripts/` - Data processing scripts
- `cdo_scripts/` - CDO (Climate Data Operators) processing scripts

**Data path:** `src/genpp/data/icon/data/`

### Preprocessing (`preproc/`)

- `preprocessors.py` - Data preprocessor implementations
- `transforms.py` - Data transformations (normalization, etc.)

## Configs

Configuration is managed via [Hydra](https://hydra.cc/). See [`src/genpp/configs/README.md`](src/genpp/configs/README.md) for detailed documentation.

## Training

Train a model using Hydra configuration:

```bash
# Train Flow Matching UViT on WeatherBench2
pixi run python src/genpp/train.py --config-name=base_fm_uvit

# Train with config overrides
pixi run python src/genpp/train.py --config-name=base_fm_uvit trainer.max_epochs=100 model.lr=1e-4

# Train DRN baseline
pixi run python src/genpp/train.py --config-name=base_drn

# Train EMOS baseline
pixi run python src/genpp/train.py --config-name=base_emos
```

### Best Model Configurations

Pre-configured commands for the best-performing models are available in:

- **WeatherBench 2**: [`src/genpp/scripts/train_best_wb2.sh`](src/genpp/scripts/train_best_wb2.sh) - Best configurations for all models (EMOS, DRN, LNGM, Engression, FM-UNet, FM-UViT)
- **ICON**: [`src/genpp/scripts/qsub_gpu_job_fast.sh`](src/genpp/scripts/qsub_gpu_job_fast.sh) - Best configurations for ICON models (designed for PBS/NQSV batch system)

### Pixi Tasks

```bash
pixi run test          # Run all tests
pixi run unittest      # Run unit tests only
pixi run wandb-agent   # Launch W&B sweep agent
pixi run launch-gpu-job # Launch GPU job on cluster
```

## Evaluation

Evaluation scripts are in `src/genpp/eval/`:

- `wb2/` - WeatherBench 2 evaluation scripts
- `icon/` - ICON evaluation scripts
- `FID/` - Fréchet Inception Distance evaluation
- `permutation_importance.py` - Feature importance analysis
- `permutation_importance_drn.py` - Feature importance for DRN
- `utils.py` - Evaluation utilities

## Visualization

Plotting notebooks and scripts are in `src/genpp/plots/`:

- `00scores_table.ipynb` / `00scores_table_icon.ipynb` - Score summary tables
- `02plot_scores.ipynb` - Score visualizations
- `03plot_samples.ipynb` / `03plot_sample_icon.py` - Sample visualizations
- `06plot_spatial_errors_wb2.ipynb` / `06plot_spatial_errors_icon.py` - Spatial error analysis
- `07plot_feature_importance.ipynb` - Feature importance plots
- `08plot_histograms.ipynb` - Histogram visualizations

## Experiment Tracking

Experiments are tracked with Weights & Biases:

- **Project Dashboard**: https://wandb.ai/feik/genpp/

Configure W&B logging in your training run:

```bash
pixi run python src/genpp/train.py --config-name=base_fm_uvit logger=wandb
```

## Project Structure

``` txt
src/genpp/
├── train.py              # Main training entrypoint
├── configs/              # Hydra configuration files
├── data/                 # Data loading and preprocessing
│   ├── weatherbench2/    # WB2 datasets and download scripts
│   ├── icon/             # ICON dataset implementation
│   └── utils.py          # Data utilities
├── models/               # Model implementations
│   ├── base_module.py    # Base Lightning module
│   ├── layers.py         # Custom layers
│   ├── loss.py           # Loss functions
│   ├── cgm/              # Conditional generative models
│   │   ├── fm/           # Flow matching (UNet, UViT, CFG)
│   │   ├── chen/         # Chen model
│   │   ├── engression/   # Engression model
│   │   └── utils/        # Shared utilities
│   ├── distributionalRegression/  # DRN, EMOS, distributions
│   └── scores/           # Scoring rules (CRPS, energy, RBF, variogram)
├── preproc/              # Preprocessors and transforms
├── eval/                 # Evaluation scripts (WB2, ICON, FID)
├── plots/                # Visualization notebooks
├── scripts/              # Shell scripts for cluster jobs
└── sweeps/               # Hyperparameter sweep configs
```
