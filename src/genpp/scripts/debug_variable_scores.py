"""Debugging script to investigate validation score differences between variables.

This script helps diagnose issues with the Chen and Engression models where
variable0 (temperature) and variable1 (wind speed) have very different validation scores.

Usage:
    python -m genpp.scripts.debug_variable_scores --data-dir /path/to/icon/data

The script will:
1. Load normalization statistics and compare FC vs REA statistics for each variable
2. Check variable ordering consistency
3. Load a sample batch and verify tensor shapes and values
"""

import argparse
import pickle
from pathlib import Path

import torch


def compare_normalization_stats(norm_stats_path: Path, feature_metadata_path: Path) -> None:
    """Compare normalization statistics between forecast and reanalysis."""
    print("=" * 80)
    print("NORMALIZATION STATISTICS ANALYSIS")
    print("=" * 80)
    
    # weights_only=False is needed because norm_stats contains tensors created by GenPP
    # This script is for local debugging with trusted data files only
    norm_stats = torch.load(norm_stats_path, weights_only=False)
    
    with open(feature_metadata_path, "rb") as f:
        feature_metadata = pickle.load(f)
    
    # Get variable names
    all_var_names = feature_metadata.get("all_var_mean_names", [])
    y_var_names = feature_metadata.get("predicted_var_mean_names", [])
    predicted_var_indices = feature_metadata.get("predicted_var_mean_indices", [])
    
    print(f"\nTarget variables (y_select_variables): {y_var_names}")
    print(f"Predicted var indices in all_vars: {predicted_var_indices}")
    
    # Check ordering
    print("\n--- Variable Ordering Check ---")
    for i, idx in enumerate(predicted_var_indices):
        name_in_all = all_var_names[idx] if idx < len(all_var_names) else "OUT OF BOUNDS"
        name_in_y = y_var_names[i] if i < len(y_var_names) else "OUT OF BOUNDS"
        match = "✓" if name_in_all == name_in_y else "✗ MISMATCH"
        print(f"  Position {i}: all_var_names[{idx}] = {name_in_all}, y_var_names[{i}] = {name_in_y} {match}")
    
    # Compare FC and REA statistics
    print("\n--- FC vs REA Statistics Comparison ---")
    print("(Differences indicate potential normalization bias)")
    
    for i, var_name in enumerate(y_var_names):
        fc_idx = predicted_var_indices[i]
        
        # FC statistics (forecast)
        fc_mean = norm_stats["all_mean"][fc_idx].item()
        fc_std = norm_stats["all_std"][fc_idx].item()
        
        # REA statistics (reanalysis/target)
        rea_mean = norm_stats["rea_mean"][i].item()
        rea_std = norm_stats["rea_std"][i].item()
        
        mean_diff = abs(fc_mean - rea_mean)
        std_ratio = fc_std / rea_std if rea_std > 0 else float("inf")
        
        print(f"\n  Variable {i}: {var_name}")
        print(f"    FC  mean: {fc_mean:12.4f}, std: {fc_std:12.4f}")
        print(f"    REA mean: {rea_mean:12.4f}, std: {rea_std:12.4f}")
        print(f"    Mean difference: {mean_diff:.4f}")
        print(f"    Std ratio (FC/REA): {std_ratio:.4f}")
        
        if mean_diff > 0.5:
            print(f"    ⚠️  Warning: Large mean difference - normalized residual will have bias!")
        if std_ratio < 0.8 or std_ratio > 1.2:
            print(f"    ⚠️  Warning: Std ratio far from 1.0 - variables may be on different scales!")


def check_sample_batch(data_dir: Path) -> None:
    """Load and check a sample batch from the dataset."""
    print("\n" + "=" * 80)
    print("SAMPLE BATCH ANALYSIS")
    print("=" * 80)
    
    fc_tensor_dir = data_dir / "tensors" / "fc"
    rea_tensor_dir = data_dir / "tensors" / "rea"
    
    # Get first FC and REA files
    fc_files = sorted(fc_tensor_dir.glob("fc_*.pt"))
    rea_files = sorted(rea_tensor_dir.glob("rea_*.pt"))
    
    if not fc_files or not rea_files:
        print("No tensor files found!")
        return
    
    print(f"\nLoading sample FC tensor: {fc_files[0].name}")
    # weights_only=False is needed for tensors created by GenPP preprocessing
    fc_tensor = torch.load(fc_files[0], weights_only=False)
    print(f"  Shape: {fc_tensor.shape}")
    
    print(f"\nLoading sample REA tensor: {rea_files[0].name}")
    # weights_only=False is needed for tensors created by GenPP preprocessing
    rea_tensor = torch.load(rea_files[0], weights_only=False)
    print(f"  Shape: {rea_tensor.shape}")
    
    # Check REA tensor values (raw, before normalization)
    print("\n--- REA Tensor Statistics (Raw, before normalization) ---")
    for i in range(rea_tensor.shape[0]):
        mean_val = rea_tensor[i].mean().item()
        std_val = rea_tensor[i].std().item()
        min_val = rea_tensor[i].min().item()
        max_val = rea_tensor[i].max().item()
        print(f"  Channel {i}: mean={mean_val:.2f}, std={std_val:.2f}, min={min_val:.2f}, max={max_val:.2f}")
        
        # Check for expected ranges
        if i == 0:  # Temperature
            if 250 < mean_val < 300:
                print(f"    ✓ Looks like temperature in Kelvin (expected ~275K)")
            else:
                print(f"    ⚠️  Unexpected range for temperature!")
        elif i == 1:  # Wind speed
            if 0 < mean_val < 20:
                print(f"    ✓ Looks like wind speed in m/s (expected 0-15 m/s)")
            else:
                print(f"    ⚠️  Unexpected range for wind speed!")


def main():
    parser = argparse.ArgumentParser(description="Debug validation score differences")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to ICON data directory containing tensors/",
    )
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    # Find norm_stats and feature_metadata files
    tensor_dir = data_dir / "tensors"
    norm_stats_files = list(tensor_dir.glob("norm_stats_*.pt"))
    fc_metadata = tensor_dir / "fc" / "feature_metadata.pkl"
    
    if not norm_stats_files:
        print("Error: No norm_stats file found!")
        return
    
    if not fc_metadata.exists():
        print(f"Error: Feature metadata not found at {fc_metadata}")
        return
    
    print(f"Using norm_stats: {norm_stats_files[0]}")
    print(f"Using metadata: {fc_metadata}")
    
    compare_normalization_stats(norm_stats_files[0], fc_metadata)
    check_sample_batch(data_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nIf you see large mean differences or std ratios far from 1.0,")
    print("this indicates the forecast and reanalysis have different statistics.")
    print("This can cause one variable to have much higher normalized error than the other.")


if __name__ == "__main__":
    main()
