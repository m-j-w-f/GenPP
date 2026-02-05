# Bug Analysis: Chen and Engression Model Validation Score Differences

## Problem Statement
For the Chen (LNGM) and Engression models, the reported validation scores for variable0 (temperature) and variable1 (wind speed) are very different. This analysis investigates potential bugs in:
- Scaling issues
- Loss calculation
- Input scales
- Variable selection

**Context:**
- Temperature (T_2M) is in Kelvin, typical values ~275K
- Wind speed (VMAX_10M) is in m/s, typical values 0-15 m/s
- Both use z-score normalization by default

---

## Potential Bugs

### 🔴 BUG 1: CRITICAL - Variable Order Mismatch in `predicted_var_mean_indices`

**Location:** `src/genpp/data/icon/dataset.py` lines 1064-1066

**Issue:** The index extraction iterates through `all_var_mean_names` in order, selecting indices where names appear in `y_select_variables`. However, this produces indices in the order they appear in `all_var_mean_names`, NOT in `y_select_variables` order.

```python
predicted_var_mean_indices = [
    i for i, name in enumerate(all_var_mean_names) if name in y_select_variables
]
```

**Why this matters:**
- `y_select_variables = ['T_2M+height_2.0', 'VMAX_10M+height_2_10.0']` (temperature first)
- If `all_var_mean_names` has these in different order, the extracted indices would point to variables in the wrong order

**Current Status:** In the current config, both lists have temperature before wind speed, so this is NOT the bug. But it's a fragile design.

**Verification:**
```python
# Add debug logging to ForecastDataset.__getitem__
print(f"predicted_var_mean_indices: {self.feature_metadata['predicted_var_mean_indices']}")
print(f"predicted_var_mean_names: {self.feature_metadata['predicted_var_mean_names']}")
print(f"all_var_mean_names: {self.feature_metadata['all_var_mean_names']}")
```

---

### 🔴 BUG 2: CRITICAL - Asymmetric Normalization Between X and Y Variables

**Location:** `src/genpp/data/icon/dataset.py` 

**Issue:** The normalization statistics for X variables and Y variables are computed from different data:
- X (forecast): Statistics computed from FC (forecast) tensors
- Y (reanalysis): Statistics computed from REA (reanalysis) tensors

If the forecast mean/std differs significantly from reanalysis mean/std, the normalized residual (prediction - target) may have very different scales for each variable.

**Example scenario:**
- Temperature forecast might have bias, so `fc_mean` ≠ `rea_mean`
- Wind speed forecast might be calibrated, so `fc_mean` ≈ `rea_mean`
- Result: Temperature residual is systematically biased, wind speed is not

**Verification:**
```python
# Compare normalization statistics
print(f"FC mean (T_2M): {norm_stats['all_mean'][25]}")  # index 25 for T_2M
print(f"REA mean (T_2M): {norm_stats['rea_mean'][0]}")  # index 0 in y_select order

print(f"FC mean (VMAX_10M): {norm_stats['all_mean'][28]}")  # index 28 for VMAX
print(f"REA mean (VMAX_10M): {norm_stats['rea_mean'][1]}")  # index 1 in y_select order
```

---

### 🟠 BUG 3: HIGH - Missing mean_correction in CNNEngressionDirectModel

**Location:** `src/genpp/models/cgm/engression/cnn.py` lines 610-643

**Issue:** The `CNNEngressionDirectModel` inherits from `BaseEngressionDirectModel` but overrides the `forward()` method without using the `mean_correction` layer.

**BaseEngressionDirectModel.forward():**
```python
nwp_mean = self.crop(x["all_vars_mean"])  # First crop all_vars_mean for mean_correction
nwp_mean = self.crop(x["predicted_vars_mean"]) + self.mean_correction(nwp_mean)  # Then combine
result = nwp_mean_expanded + self.crop(samples)
```
Note: `nwp_mean` is intentionally overwritten on the second line - the first assignment is used as input to `mean_correction()`.

**CNNEngressionDirectModel.forward():**
```python
means = x["predicted_vars_mean"].unsqueeze(1)  # No mean_correction!
res = means + samples
```

**Impact:** The `mean_correction` layer in the base class is created but never used. This layer is supposed to learn a correction based on all input variables, which could help handle variable-specific biases.

**Recommendation:** Either:
1. Remove the unused `mean_correction` from base class if CNNEngressionDirectModel doesn't need it
2. Or use it consistently in the overridden forward() method

---

### 🟠 BUG 4: HIGH - Input Channel Mismatch Between Models

**Location:** Config files and model initialization

**Issue:** The Chen and Engression models use different input channel configurations:

**Chen Model (`base_chen.yaml`):**
```yaml
in_features: ${data.features.input_features}  # 39 variables
```

**Engression Model (`base_engression.yaml`):**
```yaml
num_in_vars: ${data.features.input_features}  # 39 variables
in_channels: ${data.features.input_concat}     # 39*2 + 4 = 82 channels
```

**The difference:**
- Chen uses `in_features` = 39 (just the input variables)
- Engression uses `in_channels` = 82 (input_features * aggregation_levels + meta_features)

**Impact:** Engression gets mean+std of all variables plus meta features, while Chen might only get means.

**Verification:** Check what `_prepare_forward_inputs` uses in Chen vs `prepare_input` in Engression.

---

### 🟡 BUG 5: MODERATE - LocallyConnected2D Weight Initialization

**Location:** `src/genpp/models/layers.py` line 31

**Issue:** The weight initialization uses `torch.randn(height, width, in_features, out_features)` which doesn't account for fan-in/fan-out.

```python
self.weight = nn.Parameter(torch.randn(height, width, in_features, out_features))
```

**Impact:** With 39 input features, the variance of outputs could be very high initially. This affects both variables equally but could slow learning.

**Recommendation:** Use Xavier or Kaiming initialization:
```python
self.weight = nn.Parameter(
    torch.randn(height, width, in_features, out_features) / math.sqrt(in_features)
)
```

---

### 🟡 BUG 6: MODERATE - Energy Score Doesn't Weight Variables Equally

**Location:** `src/genpp/models/scores/kernels/l2.py` lines 37-65

**Issue:** The Energy Score with `normalize=False` (default) computes:
```python
sq_diff_sum = torch.sum(diff**2, dim=-1)  # Sum over all variables and pixels
```

This means variables with larger spatial variance contribute more to the total loss.

If temperature has a larger spatial gradient (more variation across the grid) than wind speed, temperature will dominate the loss.

**Verification:**
```python
# Check per-variable energy scores
val_loss_var_0  # Temperature
val_loss_var_1  # Wind speed
# If ratio is > 10:1, this is likely the issue
```

**Recommendation:** Use `normalize=True` in EnergyScore:
```python
EnergyScore(normalize=True)  # Uses mean instead of sum
```

---

### 🟡 BUG 7: MODERATE - Residual Connection with Different Scales

**Location:** `src/genpp/models/cgm/chen/chen.py` lines 806-809

```python
res = pred_mean + std_samples
```

**Issue:** The residual connection assumes `pred_mean` and `std_samples` are in the same scale. But:
- `pred_mean` is the output of `mean_model` which processes normalized inputs
- `std_samples` are generated from the noise decoder

If the mean_model learns to output values on a different scale than what std_samples produces, one variable's predictions could be dominated by the mean while another is dominated by the samples.

**Verification:**
```python
# During validation, log the magnitudes
print(f"pred_mean std: {pred_mean.std()}")
print(f"std_samples std: {std_samples.std()}")
# Should be similar for healthy training
```

---

## Debugging Steps

### Step 1: Print Normalization Statistics
Add logging to compare FC and REA statistics for the two target variables.

### Step 2: Verify Variable Ordering
Confirm that `predicted_vars_mean`, `rea`, and the normalization statistics all have variables in the same order.

### Step 3: Compare Per-Variable Losses
If `val_loss_var_0 >> val_loss_var_1` or vice versa, investigate why one variable is harder to predict.

### Step 4: Check Input/Output Magnitudes
Log the mean/std of model inputs and outputs at each stage of the forward pass.

### Step 5: Run with `normalize=True`
Try training with `EnergyScore(normalize=True)` to see if this balances the variables.

### Step 6: Swap Variable Order
If the problem follows the variable (e.g., temperature is always bad), it's a data issue.
If the problem follows the position (e.g., variable 0 is always bad), it's a model issue.

---

## Most Likely Root Causes

1. **Different normalization statistics between FC and REA** - If forecast and reanalysis have different biases, the normalized targets have different scales for each variable.

2. **Energy Score dominance** - Without `normalize=True`, the variable with more spatial variation dominates the loss.

3. **Residual connection imbalance** - The mean_model might learn to predict one variable better than the other, creating an imbalance in the residual.

---

## Recommended Fixes

1. **Add assertions to verify variable order consistency**
2. **Use `EnergyScore(normalize=True)` for training**
3. **Log per-variable statistics during training**
4. **Consider variable-wise loss weighting if needed**
