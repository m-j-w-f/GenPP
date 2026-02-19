# Plan: Permutation-Based Feature Importance for WB2

## Overview

This document proposes a plan to implement **permutation-based feature importance assessment** for the WeatherBench2 (WB2) pipeline. The approach permutes one input channel at a time per prediction (i.e., per sample with a unique `(time, prediction_timedelta)` pair) and measures the resulting degradation in model performance, thereby quantifying each channel's contribution.

## Background

### Current Data Structure

The WB2 input tensor `x` has shape `(sample, feature, longitude, latitude)` where:

- **`sample`**: Stacked from `(time × prediction_timedelta)` — each calendar day produces 5 samples (lead times day+1 through day+5).
- **`feature`**: Input channels (e.g., 63 in the full config: 30 variables × 2 aggregation stats [mean, std] + 3 metadata features).
- **`longitude`** / **`latitude`**: Spatial grid (37 × 31).

Feature categories (tracked in `feature_metadata`):
| Category | Examples | Indices |
| --- | --- | --- |
| `all_var_mean` | `2m_temperature+statistic_mean`, `10m_wind_speed+statistic_mean` | `all_var_mean_indices` |
| `all_var_std` | `2m_temperature+statistic_std`, `10m_wind_speed+statistic_std` | `all_var_std_indices` |
| `meta_vars` | `sin_prediction_time`, `cos_prediction_time`, `latitude`, `longitude` | `meta_var_indices` |
| `pixel_idx` | `pixel_idx` | `pixel_idx_index` |

### What "Permute One Channel Per Prediction" Means

For a given channel index `c`, for each sample (identified by its unique `(time, prediction_timedelta)` pair):
1. **Shuffle the spatial values** of channel `c` — permute the flattened `(longitude, latitude)` grid and reshape back.
2. This breaks the spatial signal for that channel while preserving:
   - The marginal distribution of channel values.

Each sample gets an independent random permutation. The transform is stateless per-sample.

## Proposed Implementation

### 1. `PermuteChannel` Transform in `preproc/transforms.py`

**Rationale**: The existing transform framework (`Transform` ABC → applied via `TransformTensorDataset.__getitem__`) is the natural place for on-the-fly data perturbation. A `PermuteChannel` transform fits the existing pattern: it takes a tensor and returns a tensor with one channel's spatial values shuffled.

```python
class PermuteChannel(Transform):
    """Permute the spatial values of a single channel for feature importance analysis.

    Args:
        channel_index (int): Index of the feature channel to permute.
        seed (int | None): Optional random seed for reproducibility.
    """

    def __init__(self, channel_index: int, seed: int | None = None):
        self.channel_index = channel_index
        self.seed = seed

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Permute spatial dimensions of the specified channel.

        Args:
            data: Tensor with shape (feature, longitude, latitude).

        Returns:
            Tensor with the specified channel's spatial values shuffled.
        """
        result = data.clone()
        channel = result[self.channel_index]          # (lon, lat)
        flat = channel.flatten()
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        perm = torch.randperm(flat.numel(), generator=generator)
        result[self.channel_index] = flat[perm].reshape(channel.shape)
        return result
```

**Key Design Decisions**:
- Operates on a single sample tensor `(feature, lon, lat)` — this is the shape at `__getitem__` time, before batching.
- Uses `torch.Generator` with an optional seed so experiments can be reproduced.
- Clones the input to avoid in-place mutation of cached tensors.
- The transform is **stateless per-sample** — each sample gets an independent random permutation. This is the correct design since each sample corresponds to a unique `(time, prediction_timedelta)` prediction.

**Why a Transform (vs. a Preprocessor)**:
- **Transforms** are applied on-the-fly during `__getitem__` and do not modify the cached data. This is essential — we never want to persist permuted data.
- **Preprocessors** are applied once during `prepare_data()` and affect the cached `.pt` files. Permuting at the preprocessor level would corrupt the cache.
- The transform can be composed via `Pipe` with existing transforms (e.g., `Pad`).

### 2. Integration with `Pipe`

The `PermuteChannel` transform can be composed in a `Pipe` pipeline:

```python
# Example: permute channel 0, then pad
pipe = Pipe([PermuteChannel(channel_index=0, seed=42), Pad(padding=(1,1,1,1))])
```

### 3. Importance Script: `eval/permutation_importance.py`

A standalone script that orchestrates the full importance assessment. The script directly controls which channel to permute by programmatically injecting a `PermuteChannel` transform into the dataset's `x_transform` for each channel iteration — no YAML config changes needed.

**High-Level Algorithm**:
```
1. Load a trained model from WandB (reuse logic from cgm_predict_eval.py).
2. Run baseline prediction + energy score computation (no permutation, no rescaling needed).
3. For each input channel c in [0, ..., n_features-1]:
   a. Create a PermuteChannel(channel_index=c) transform.
   b. Inject it into the dataset's x_transform (via Pipe) directly in the script.
   c. Run prediction + energy score computation with the permuted data.
   d. Record: importance[c] = (energy_score_permuted - energy_score_baseline) / energy_score_baseline.
4. Aggregate and output results (CSV and/or console).
```

**Proposed CLI**:
```bash
python -m genpp.eval.permutation_importance \
    --run-path feik/genpp/abc123 \
    --split val \
    --device 0 \
    --batch-size 16 \
    --n-repeats 5 \
    --seed 42 \
    -v
```

**Key Arguments**:
| Argument | Description |
| --- | --- |
| `--run-path` | WandB run path(s) to load the trained model |
| `--split` | Dataset split to evaluate on (`val` or `test`) |
| `--n-repeats` | Number of permutation repeats per channel (for robust estimates) |
| `--seed` | Base random seed for reproducibility |
| `--channels` | Optional: subset of channel indices to permute (default: all) |
| `--output` | Output CSV path |

**Implementation Strategy**:
- **Reuse heavily** from `cgm_predict_eval.py`: model loading, config parsing, datamodule setup, scoring logic.
- Extract shared utilities (model loading, scoring) into `eval/utils.py` if not already there.
- The script programmatically creates a `PermuteChannel` transform for the current channel, wraps it with any existing `x_transform` in a `Pipe`, and passes it to `TransformTensorDataset` before running prediction. This is done in a loop over all channels directly in the script.
- Energy score is computed without rescaling the model output (can use raw model predictions directly).

**Output Format** (CSV):
```csv
channel_index,channel_name,category,baseline_es,permuted_es,importance,importance_std
0,2m_temperature+statistic_mean,predicted_var_mean,0.52,0.89,0.71,0.02
1,10m_wind_speed+statistic_mean,predicted_var_mean,0.52,0.61,0.17,0.01
...
```

Where `importance = (permuted_es - baseline_es) / baseline_es`.

## Implementation Checklist

- [ ] Add `PermuteChannel` class to `preproc/transforms.py`
- [ ] Add unit tests for `PermuteChannel` in `tests/test_preproc/test_transforms.py`
- [ ] Create `src/genpp/eval/permutation_importance.py` script
- [ ] Refactor shared model-loading logic from `cgm_predict_eval.py` into reusable utilities (if needed)
- [ ] Add integration test for importance script (mock model + small synthetic data)
- [ ] Document usage in README or script docstring
- [ ] Validate on a real WB2 model run

## Risks and Considerations

1. **Runtime**: With 63 channels × 5 repeats × full evaluation = 315 forward passes. This could be slow for large models. Mitigation: `--channels` flag to subset.
2. **Memory**: Cloning tensors in `PermuteChannel.transform()` doubles per-sample memory. This is negligible at the sample level.
3. **Statistical Significance**: Single permutation per channel may be noisy. Multiple repeats (`--n-repeats`) with different seeds addresses this.
