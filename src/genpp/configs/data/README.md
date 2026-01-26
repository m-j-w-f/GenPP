# Data Configurations

This directory contains dataset-specific configurations organized by dataset type.

## Directory Structure

### Dataset-Specific Configurations

- **icon/** - Configuration files for ICON dataset
  - `icon_forecast.yaml` - Main ICON forecast configuration
  - `spatial/` - ICON-specific spatial configurations
    - `standard.yaml` - Standard spatial config (no padding/transforms)
  - `splits/` - ICON-specific train/val/test splits
    - `standard.yaml` - Standard time splits (train: <=2021, val: 2022, test: >=2023)
  - Uses `ForecastDataModule` from `genpp.data.icon.dataset`
  
- **weatherbench2/** - Configuration files for WeatherBench2 dataset
  - `weatherbench2_full_fast.yaml` - Full variable set configuration
  - `weatherbench2_cut_fast.yaml` - Reduced variable set (inherits from full)
  - `ae_dataset.yaml` - Configuration for AutoEncoder (used in FID computation)
  - `classification_dataset.yaml` - Configuration for classifier
  - `spatial/` - WeatherBench2-specific spatial configurations
    - `standard.yaml` - Standard spatial config (31x37 grid, no padding)
    - `padded_x.yaml` - Padded in x dimension only
    - `padded_xy.yaml` - Padded in both x and y dimensions
  - `splits/` - WeatherBench2-specific train/val/test splits
    - `standard.yaml` - Standard time-based splits
  - Uses various data modules including:
    - `FastWeatherBench2DataModule` for optimized tensor loading
    - `ZarrDataModule` for zarr-based datasets
    - `ZarrClassificationDataModule` for classification tasks

### Shared Configurations

- **dataloader/** - DataLoader configurations (shared across datasets)
  - `standard.yaml` - Standard settings (4 workers, prefetch, forkserver)
  - `simple.yaml` - Simple settings (4 workers, basic options)
  - `debug.yaml` - Debug settings (minimal workers for debugging)
  
- **preprocess/** - Preprocessing configurations (shared across datasets)
  - `standard.yaml` - Standard preprocessing with meta features
  - `minMaxWind.yaml` - Min-max normalization for wind variables
  - `none.yaml` - No preprocessing

## Usage

Configurations are referenced in Hydra using the dataset-specific paths:

```yaml
# For WeatherBench2 with spatial/splits configs
defaults:
  - data/dataloader: standard
  - data/weatherbench2/spatial: padded_xy
  - data/weatherbench2/splits: standard
  - data/preprocess: standard
  - data: weatherbench2/weatherbench2_full_fast

# For ICON with spatial/splits configs
defaults:
  - data/dataloader: standard
  - data/icon/spatial: standard
  - data/icon/splits: standard
  - data/preprocess: standard  # Optional, ICON doesn't use preprocess by default
  - data: icon/icon_forecast
```

## Config Structure Rules

1. **Spatial and splits configs are dataset-specific** - Each dataset has its own `spatial/` and `splits/` subdirectories with configs tailored to that dataset's characteristics (grid dimensions, time ranges, etc.)

2. **Dataloader and preprocess configs are shared** - These configs contain general settings that work across datasets and are located directly under `data/`.

3. **Config references** - Dataset configs reference shared configs using `${data.dataloader.*}` and `${data.preprocess.*}`, and reference their own spatial/splits using `${data.spatial.*}` and `${data.splits.*}`.

## Dataset-Specific Keys

Configurations differ between datasets based on their specific requirements:
- **Grid dimensions**: WeatherBench2 uses 31x37 grid, ICON uses dynamic dimensions
- **Variable naming**: Different naming conventions for weather variables
- **Time splits**: Different year ranges and splitting logic
- **Spatial transforms**: Dataset-specific padding and transformation needs

When creating new configs, place them in the appropriate dataset subdirectory.
