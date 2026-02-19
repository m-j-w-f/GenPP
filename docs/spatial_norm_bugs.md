# Spatial Normalization (`norm_mode=spatial`) — Possible Bug Sources

When training a Flow Matching model with `data.norm_mode=spatial` via:

```bash
pixi run -e gpu python src/genpp/train.py --config-name base_fm_uvit \
  model=fm_uvit_noise model.internal_td_scaling=std \
  model/lr_scheduler=reduceLROnPlateau \
  data=icon_full_pad_xy data.norm_mode=spatial \
  data.dataloader.num_workers=10 data.batch_size=8 \
  model.backbone.depth=2 model.backbone.embed_dim=128 \
  model.optimizer.lr=0.001
```

The model produces NaN/Inf scores. With `per_variable` mode (default), everything works fine.

---

## Bug 1 (CRITICAL): Division by zero — spatial `std` can be zero at individual grid cells

**Location:** `_compute_tensor_stats` in `dataset.py` (line ~780-783) and `__getitem__` (lines 350-352, 367-369, 389-391)

**Problem:**  
In `per_variable` mode, the std is computed across **all spatial positions and all samples**, so it averages over `N_samples × H × W` values — practically never zero.  
In `spatial` mode, the std is computed **per grid cell** across only `N_samples` values (one value per training tensor). If a variable is constant or near-constant at a specific grid cell across all training samples (e.g., albedo at a water grid cell is always 0), the spatial std for that cell will be **exactly zero or extremely close to zero**.

When `__getitem__` normalizes via:
```python
all_vars_mean[slice_idx] = (all_vars_mean_slice - mean) / std  # std can be 0 at some (c, x, y)
```
this produces **Inf** or **NaN** values that propagate through the entire training pipeline.

**Why it causes NaN/Inf:**
- `0 / 0 = NaN`
- `nonzero / 0 = Inf`
- These values propagate into the loss function and gradients, causing NaN/Inf scores.

**Fix:** Clamp the spatial std to a small epsilon before dividing, e.g. `std = torch.clamp(std, min=1e-6)`.

---

## Bug 2 (CRITICAL): Padding + spatial normalization mismatch

**Location:** `__getitem__` in `dataset.py` (lines 402-411)

**Problem:**  
The `icon_full_pad_xy` config applies a `Pad` transform with `mode="constant"` (zero-padding) **after** normalization. When using spatial normalization:

1. The data is normalized using stats of shape `[c, x, y]` (original spatial dimensions).
2. Then padding is applied, adding **zero-valued border pixels** to the spatial dimensions.
3. In the normalized space, a value of 0 means `(x - mean) / std = 0`, i.e. `x = mean` — but the padded zeros actually represent **no data**, not `x = mean`.

For the **ground truth (y)**, this means the padded regions contain zeros in normalized space. During training, the FM noise model computes:
```python
x_1 = ground_truth - nwp_fc["predicted_vars_mean"]
```
Both `ground_truth` and `predicted_vars_mean` will be 0 in padded regions, so `x_1 = 0` there — which seems fine. But the `internal_td_scaling` divides by `scale`:
```python
x_1 = x_1 / scale
```
If scale is derived from spatial statistics that don't account for padding, this could introduce inconsistencies.

**Note:** With the current `icon_full_pad_xy.yaml`, padding is `[0, 0, 0, 0]` so this bug only manifests if non-zero padding is used. But it's a latent issue.

---

## Bug 3 (LIKELY): MinMax normalization — division by zero when `max == min` spatially

**Location:** `__getitem__` (lines 357-359, 374-376, 396-398)

**Problem:**  
In minmax normalization:
```python
all_vars_mean[slice_idx] = (all_vars_mean_slice - all_min) / (all_max - all_min)
```
In spatial mode, `all_max[c, x, y] - all_min[c, x, y]` can be **exactly zero** for specific grid cells where the variable has the same value across all training samples. This produces Inf/NaN.

**Note:** This only applies if minmax normalization is used. The reported command uses default `zscore`, so this is relevant only for minmax configs but should still be guarded.

---

## Bug 4 (POSSIBLE): Numerical precision — variance computation can go negative

**Location:** `_compute_tensor_stats` (lines 780-783)

**Problem:**  
The variance is computed using the formula:
```python
var = (tensor_sum_sq / tensor_count) - (mean ** 2)
```
This is the "textbook" formula, which is known to be **numerically unstable**. When values are large and the variance is small, `sum_sq / N` and `mean²` can be very close, and their difference can become **slightly negative** due to floating-point cancellation — even when using double precision.

`torch.sqrt(negative_value)` produces **NaN**.

In `per_variable` mode, the aggregation is over `N_samples × H × W` values (millions), making the estimates very stable. In `spatial` mode, aggregation is over only `N_samples` values (potentially hundreds or thousands), making the estimates much more susceptible to numerical instability.

**Fix:** Use Welford's online algorithm or clamp variance to zero: `var = torch.clamp(var, min=0.0)`.

---

## Bug 5 (POSSIBLE): `internal_td_scaling.fit()` may produce invalid scales with spatially-normalized data

**Location:** `td_scaling.py`, `FixedTDScaling.fit()` (lines 200-251) and `LinearAbsTDScaling.fit()` (lines 89-141)

**Problem:**  
The TD scaling fit methods compute `diff = obs - nwp` and then regress `abs(diff)` against lead time. With spatial normalization:

- The magnitude of `diff` varies spatially because each grid cell has its own normalization scale.
- In `per_variable` mode, all grid cells within a variable share the same scale, so `diff` is uniformly scaled.
- In `spatial` mode, some grid cells may have very large normalized values (due to near-zero std), while others are normal.
- The regression `abs(diff) ~ td` can be dominated by these extreme outlier cells.
- The resulting `scale` may not properly normalize the deviations, leading to extreme values in `x_1 / scale` during training.

If the fitted scale is very small (because the average abs error is small globally but some cells are extreme), dividing by this scale amplifies the extreme cells even further, leading to NaN/Inf.

---

## Bug 6 (POSSIBLE): Double precision to float conversion loses spatial extremes

**Location:** `_compute_tensor_stats` (lines 783-784)

**Problem:**  
The spatial sums are accumulated in double precision, but the final mean and std are cast back to float32:
```python
std = torch.sqrt(var).float()
mean = mean.float()
```
For spatial mode, the mean values can be very large (e.g., pressure ~100000 Pa at a specific grid cell), and converting from double to float can introduce precision loss. More importantly, extremely small std values near the float32 epsilon (~1e-7) may become zero after conversion, triggering Bug 1.

---

## Bug 7 (UNLIKELY but worth checking): `tensor_min` / `tensor_max` type mismatch in spatial mode

**Location:** `_compute_tensor_stats` (lines 747-753)

**Problem:**  
In spatial mode, the initial `tensor_min` and `tensor_max` are created with `.clone()` (float32), while `tensor_sum` and `tensor_sum_sq` use `.double()`. Later, `torch.minimum(tensor_min, tensor)` compares float32 tensors, which should be fine. However, the min/max tensors remain float32 while mean/std are computed in double and then cast to float32. This inconsistency is unlikely to cause issues but could be confusing.

---

## Summary / Priority

| # | Bug | Severity | Likely cause of NaN/Inf? |
|---|-----|----------|--------------------------|
| 1 | Spatial std = 0 → division by zero | **CRITICAL** | **YES** — most likely cause |
| 4 | Negative variance → NaN from sqrt | **HIGH** | **YES** — second most likely |
| 3 | MinMax `max - min = 0` → division by zero | HIGH | Only if minmax is used |
| 5 | TD scaling fit produces bad scales | MEDIUM | Possible amplification |
| 6 | Float32 conversion loses small std | MEDIUM | Contributes to Bug 1 |
| 2 | Padding + spatial norm mismatch | LOW | Only with non-zero padding |
| 7 | Type mismatch in min/max | LOW | Unlikely |

**Recommended investigation order:** Bug 1 → Bug 4 → Bug 5 → Bug 6
