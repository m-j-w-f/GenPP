# Data Configurations

This directory contains dataset-specific configurations organized by dataset type.

## Directory Structure

- **icon/** - Configuration files for ICON dataset
  - Contains configs for the ICON weather prediction model dataset
  - Uses `ForecastDataModule` from `genpp.data.icon.dataset`
  
- **weatherbench2/** - Configuration files for WeatherBench2 dataset
  - Contains configs for WeatherBench2 benchmark dataset
  - Uses various data modules including:
    - `FastWeatherBench2DataModule` for optimized tensor loading
    - `ZarrDataModule` for zarr-based datasets
    - `ZarrClassificationDataModule` for classification tasks
  - Files:
    - `weatherbench2_full_fast.yaml` - Full variable set configuration
    - `weatherbench2_cut_fast.yaml` - Reduced variable set (inherits from full)
    - `ae_dataset.yaml` - Configuration for AutoEncoder (used in FID computation)
    - `classification_dataset.yaml` - Configuration for classifier

- **dataloader/** - DataLoader configurations (shared across datasets)
- **preprocess/** - Preprocessing configurations (shared across datasets)
- **spatial/** - Spatial configuration options (shared across datasets)
- **splits/** - Dataset split configurations (train/val/test)

## Usage

Configurations can be referenced in Hydra using the new paths:

```yaml
# For WeatherBench2
defaults:
  - data: weatherbench2/weatherbench2_full_fast

# For ICON (when configs are created)
defaults:
  - data: icon/icon_forecast
```

## Dataset-Specific Keys

Configurations may differ between datasets based on their specific requirements:
- File paths
- Axis/dimension names
- Variable naming conventions
- Data module parameters

When creating new configs, place them in the appropriate dataset subdirectory.
