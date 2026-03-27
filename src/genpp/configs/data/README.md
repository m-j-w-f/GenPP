# Data Configurations

This directory contains dataset-specific configurations organized as flat config files.

## Directory Structure

### Dataset Configurations

- **WeatherBench2 Configs:**
  - `wb2_full.yaml` - Full variable set with standard spatial and standard preprocessing
  - `wb2_full_minmax.yaml` - Full variables with minmax wind preprocessing (for DRN)
  - `wb2_full_pad_x.yaml` - Full variables with X padding (for CNN models)
  - `wb2_full_pad_xy.yaml` - Full variables with X and Y padding (for flow matching)
  - `wb2_cut.yaml` - Reduced variables with no preprocessing (for EMOS)
  - `ae_dataset.yaml` - Configuration for AutoEncoder (used in FID computation)
  - `classification_dataset.yaml` - Configuration for classifier
  - Uses `FastWeatherBench2DataModule` for optimized tensor loading

- **ICON Configs:**
  - `icon_full.yaml` - Full ICON forecast configuration
  - Uses `ForecastDataModule` from `genpp.data.icon.dataset`

## Usage

Each dataset configuration is completely self-contained with spatial, splits, preprocessing, and dataloader settings:

```yaml
# WeatherBench2 with padding on X (for CNN models)
defaults:
  - data: wb2_full_pad_x

# WeatherBench2 with reduced variables and no preprocessing (for EMOS)
defaults:
  - data: wb2_cut

# WeatherBench2 with minmax preprocessing (for DRN)
defaults:
  - data: wb2_full_minmax

# ICON forecast
defaults:
  - data: icon_full
```

### Switching Datasets via Command Line

```bash
# Use WeatherBench2 full with X padding
python train.py -cn base_chen data=wb2_full_pad_x

# Use WeatherBench2 cut (reduced variables, no preprocessing)
python train.py -cn base_emos data=wb2_cut

# Use ICON
python train.py -cn base_chen data=icon_full
```

## Available Configs Summary

| Config | Dataset | Variables | Spatial | Preprocessing |
|--------|---------|-----------|---------|---------------|
| `wb2_full` | WeatherBench2 | Full (30) | Standard (31x37) | Standard |
| `wb2_full_minmax` | WeatherBench2 | Full (30) | Standard (31x37) | MinMax Wind |
| `wb2_full_pad_x` | WeatherBench2 | Full (30) | Padded X (32x40) | Standard |
| `wb2_full_pad_xy` | WeatherBench2 | Full (30) | Padded X+Y (32x40) | Standard |
| `wb2_cut` | WeatherBench2 | Reduced (2) | Standard (31x37) | None |
| `icon_full` | ICON | Full (43) | Dynamic grid | N/A |
| `ae_dataset` | WeatherBench2 | 2 vars | N/A | N/A |
| `classification_dataset` | WeatherBench2 | 2 vars | N/A | N/A |

## Config Structure

Each dataset config contains all necessary settings:
- **spatial**: Grid dimensions and transforms (padding)
- **splits**: Train/val/test time ranges
- **preprocess**: Preprocessing pipelines (x_preprocessing, y_preprocessing)
- **x_select_variables / y_select_variables**: Input/output variable selection
- **features**: Feature configuration
- **dataloader settings**: Workers, prefetch, pin_memory, etc.
- **module**: DataModule target and parameters
