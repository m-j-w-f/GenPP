# Plan: Permutation-Based Feature Importance for WB2

## Overview

This document proposes a plan to implement **permutation-based feature importance assessment** for the WeatherBench2 (WB2) pipeline. The approach permutes one input channel at a time (per day) and measures the resulting degradation in model performance, thereby quantifying each channel's contribution.

## Background

### Current Data Structure

The WB2 input tensor `x` has shape `(sample, feature, longitude, latitude)` where:

- **`sample`**: Stacked from `(time × prediction_timedelta)` — each calendar day produces 5 samples (lead times day+1 through day+5).
- **`feature`**: Input channels (e.g., 63 in the full config: 30 variables × 2 aggregation stats [mean, std] + 3 metadata features).
- **`longitude`** / **`latitude`**: Spatial grid (37 × 31).

Feature categories (tracked in `feature_metadata`):
| Category | Examples | Indices |
|---|---|---|
| `all_var_mean` | `2m_temperature+statistic_mean`, `10m_wind_speed+statistic_mean` | `all_var_mean_indices` |
| `all_var_std` | `2m_temperature+statistic_std`, `10m_wind_speed+statistic_std` | `all_var_std_indices` |
| `meta_vars` | `sin_prediction_time`, `cos_prediction_time`, `latitude`, `longitude` | `meta_var_indices` |
| `pixel_idx` | `pixel_idx` | `pixel_idx_index` |

### What "Permute One Channel Per Day" Means

For a given channel index `c`:
1. Group all samples by their originating calendar day (i.e., samples that share the same `time` coordinate but differ by `prediction_timedelta`).
2. Within each day-group, **shuffle the spatial values** of channel `c` (permute the `(longitude, latitude)` grid) — all 5 lead-time samples from that day get the same permutation applied.
3. This breaks the spatial signal for that channel while preserving:
   - The marginal distribution of channel values.
   - The temporal coherence within a day (all lead times see the same permuted field).

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
- **Per-day consistency** (same permutation for all 5 lead times of a day) is **not** handled at the transform level. This is intentional — the transform is stateless per-sample. Per-day grouping is handled by the importance script (see Section 3).

**Why a Transform (vs. a Preprocessor)**:
- **Transforms** are applied on-the-fly during `__getitem__` and do not modify the cached data. This is essential — we never want to persist permuted data.
- **Preprocessors** are applied once during `prepare_data()` and affect the cached `.pt` files. Permuting at the preprocessor level would corrupt the cache.
- The transform can be composed via `Pipe` with existing transforms (e.g., `Pad`).

**Limitations of Transform-Only Approach**:
- The current `TransformTensorDataset` applies the **same** transform to every sample. To enforce per-day consistency, we would need either:
  - (a) A stateful transform that tracks which day-group the current sample belongs to (complex, fragile).
  - (b) A custom dataset wrapper or a modified evaluation loop that sets the seed per day-group (recommended — see Script section).
- For the initial implementation, **per-sample random permutation** (no day-grouping) is a valid and simpler baseline. Per-day grouping can be added later.

### 2. Integration with `Pipe` and Config

The `PermuteChannel` transform can be composed in a `Pipe` pipeline:

```python
# Example: permute channel 0, then pad
pipe = Pipe([PermuteChannel(channel_index=0, seed=42), Pad(padding=(1,1,1,1))])
```

For Hydra config-driven usage, it can be instantiated like other transforms:

```yaml
x_transform:
  _target_: genpp.preproc.transforms.Pipe
  transforms:
    - _target_: genpp.preproc.transforms.PermuteChannel
      channel_index: 0
      seed: 42
    - _target_: genpp.preproc.transforms.Pad
      padding: [1, 1, 1, 1]
```

### 3. Importance Script: `eval/permutation_importance.py`

A standalone script that orchestrates the full importance assessment:

**High-Level Algorithm**:
```
1. Load a trained model from WandB (reuse logic from cgm_predict_eval.py).
2. Run baseline prediction + scoring (no permutation).
3. For each input channel c in [0, ..., n_features-1]:
   a. Create a PermuteChannel(channel_index=c) transform.
   b. Inject it into the dataset's x_transform (via Pipe).
   c. Run prediction + scoring with the permuted data.
   d. Record score degradation: importance[c] = score_permuted - score_baseline.
4. Aggregate and output results (CSV, WandB, or console).
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
|---|---|
| `--run-path` | WandB run path(s) to load the trained model |
| `--split` | Dataset split to evaluate on (`val` or `test`) |
| `--n-repeats` | Number of permutation repeats per channel (for robust estimates) |
| `--seed` | Base random seed for reproducibility |
| `--channels` | Optional: subset of channel indices to permute (default: all) |
| `--metric` | Scoring metric to use (`crps`, `energy_score`, default: `crps`) |
| `--skip-meta` | Skip metadata channels (lat, lon, sin/cos time) |
| `--output` | Output CSV path |

**Implementation Strategy**:
- **Reuse heavily** from `cgm_predict_eval.py`: model loading, config parsing, datamodule setup, scoring logic.
- Extract shared utilities (model loading, scoring) into `eval/utils.py` if not already there.
- The script modifies `cfg.data.module.dataset_config.{split}.x_transform` before calling `datamodule.setup()` to inject the `PermuteChannel` transform.
- For per-day consistency: the script can set the `seed` parameter of `PermuteChannel` based on the day index. Since the dataset is ordered by `(time, prediction_timedelta)`, samples `[i*5 : (i+1)*5]` belong to the same day and can share a seed.

**Output Format** (CSV):
```csv
channel_index,channel_name,category,baseline_crps,permuted_crps,importance,importance_std
0,2m_temperature+statistic_mean,predicted_var_mean,0.52,0.89,0.37,0.02
1,10m_wind_speed+statistic_mean,predicted_var_mean,0.52,0.61,0.09,0.01
...
```

### 4. Per-Day Permutation Strategy (Advanced)

For strict per-day consistency (all 5 lead times of a day share the same permuted field):

**Option A — Seed-Based (Recommended)**:
In the importance script, when creating the `PermuteChannel` transform, pass a seed derived from the day index:
```python
# In a custom dataset wrapper or modified __getitem__:
day_index = sample_index // 5  # 5 lead times per day
transform = PermuteChannel(channel_index=c, seed=base_seed + day_index)
```
This requires a small modification to `TransformTensorDataset` to pass the sample index to the transform, or a wrapper dataset.

**Option B — Pre-Permuted Tensor**:
Before prediction, clone the full `x` tensor and apply permutation at the tensor level (outside the dataset):
```python
x_permuted = x_tensor.clone()
for day_start in range(0, len(x_tensor), 5):
    perm = torch.randperm(lon * lat)
    for offset in range(5):
        idx = day_start + offset
        flat = x_permuted[idx, channel].flatten()
        x_permuted[idx, channel] = flat[perm].reshape(lon, lat)
```
This avoids modifying the transform/dataset infrastructure but requires more memory.

**Recommendation**: Start with per-sample permutation (simpler, uses the `PermuteChannel` transform directly). Add per-day grouping in a follow-up if results suggest it matters.

## Implementation Checklist

- [ ] Add `PermuteChannel` class to `preproc/transforms.py`
- [ ] Add unit tests for `PermuteChannel` in `tests/test_preproc/test_transforms.py`
- [ ] Create `src/genpp/eval/permutation_importance.py` script
- [ ] Refactor shared model-loading logic from `cgm_predict_eval.py` into reusable utilities (if needed)
- [ ] Add integration test for importance script (mock model + small synthetic data)
- [ ] Document usage in README or script docstring
- [ ] Validate on a real WB2 model run

## Risks and Considerations

1. **Runtime**: With 63 channels × 5 repeats × full evaluation = 315 forward passes. This could be slow for large models. Mitigation: `--channels` flag to subset, `--skip-meta` to skip metadata channels.
2. **Memory**: Cloning tensors in `PermuteChannel.transform()` doubles per-sample memory. This is negligible at the sample level.
3. **Statistical Significance**: Single permutation per channel may be noisy. Multiple repeats (`--n-repeats`) with different seeds addresses this.
4. **Per-Day vs. Per-Sample**: Per-sample permutation breaks the within-day spatial consistency. For the initial version this is acceptable; per-day grouping can be added later.
