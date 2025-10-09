"""Compute statistics for LHCb variables with outlier removal.

How to run:
    pixi run python src/hepattn/experiments/lhcb/compute_statistics.py \
        --data-paths /data/bfys/shenzh/4Dtracking/hepattn/data/lhcb/version3_hdf5/train.h5 \
                     /data/bfys/shenzh/4Dtracking/hepattn/data/lhcb/version3_hdf5/val.h5 \
        --config src/hepattn/experiments/lhcb/config/base.yaml \
        --output src/hepattn/experiments/lhcb/config/lhcb_var_transform_temp.yaml \
        --cut-fraction 0
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml


def remove_outliers_percentile(data, cut_fraction):
    """Remove outliers using percentile-based method.

    Args:
        data: 1D array of values
        cut_fraction: Fraction to cut from each tail
    Returns:
        Cleaned data and mask
    """
    lower_percentile = cut_fraction * 100
    upper_percentile = (1 - cut_fraction) * 100

    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)

    mask = (data >= lower) & (data <= upper)
    return data[mask], mask


def compute_statistics(data, cut_fraction=0.01):
    """Compute statistics with outlier removal.

    Args:
        data: 1D array of values
        cut_fraction: Fraction to cut from each tail (default 0.01 = 1%)

    Returns:
        Tuple of (result_dict, data_original, data_clean, data_transformed)
    """
    # Remove NaN and inf
    data_original = data[np.isfinite(data)]

    if len(data_original) == 0:
        return None

    # Check if data is boolean (only 0 and 1)
    unique_vals = np.unique(data_original)
    if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
        print("  Skipping boolean field")
        return None

    # Remove outliers
    data_clean, _mask = remove_outliers_percentile(data_original, cut_fraction=cut_fraction)
    n_outliers = len(data_original) - len(data_clean)
    outlier_frac = n_outliers / len(data_original)

    print(f"  Removed {n_outliers:,} outliers ({outlier_frac * 100:.2f}%)")

    # Compute statistics on cleaned data
    mean = np.mean(data_clean)
    std = np.std(data_clean)

    print(f"  Mean: {mean:.6f}, Std: {std:.6f}")

    # Apply standardization transformation
    data_transformed = (data_clean - mean) / std

    # Store results
    result = {
        "mean": float(mean),
        "std": float(std),
        "n_samples": len(data_clean),
        "outlier_fraction": float(outlier_frac),
    }

    return (result, data_original, data_clean, data_transformed)


def plot_before_after(data_clean, data_transformed, field_name, result, save_dir):
    """Plot histogram before and after transformation."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Before transformation (after outlier removal)
    ax_before = axes[0]
    ax_before.hist(data_clean, bins=100, alpha=0.7, color="blue", edgecolor="black")
    ax_before.axvline(result["mean"], color="red", linestyle="--", linewidth=2, label=f"Mean: {result['mean']:.4f}")
    ax_before.axvline(result["mean"] + result["std"], color="orange", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Std: {result['std']:.4f}")
    ax_before.axvline(result["mean"] - result["std"], color="orange", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_before.set_xlabel(field_name, fontsize=13)
    ax_before.set_ylabel("Counts", fontsize=13)
    ax_before.set_title(f"Before Transformation\nmu={result['mean']:.4f}, sigma={result['std']:.4f}", fontsize=14, fontweight="bold")
    ax_before.legend(fontsize=11)
    ax_before.grid(True, alpha=0.3)

    # Right plot: After transformation (standardized)
    ax_after = axes[1]
    ax_after.hist(data_transformed, bins=100, alpha=0.7, color="green", edgecolor="black")
    ax_after.axvline(0, color="red", linestyle="--", linewidth=2, label="Mean: 0.00")
    ax_after.axvline(1, color="orange", linestyle="--", linewidth=1.5, alpha=0.8, label="Std: 1.00")
    ax_after.axvline(-1, color="orange", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_after.set_xlabel(f"{field_name} (standardized)", fontsize=13)
    ax_after.set_ylabel("Counts", fontsize=13)
    ax_after.set_title("After Transformation\nmu=0.00, sigma=1.00", fontsize=14, fontweight="bold")
    ax_after.legend(fontsize=11)
    ax_after.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(
        f"{field_name} - Samples: {result['n_samples']:,} | Outliers removed: {result['outlier_fraction'] * 100:.2f}%", fontsize=15, fontweight="bold"
    )

    # Save plot
    save_path = save_dir / f"{field_name}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved plot to {save_path}")


def compute_statistics_from_hdf5(hdf5_paths, input_fields, target_fields, cut_fraction=0.01):
    """Compute statistics from HDF5 files.

    Args:
        hdf5_paths: List of paths to HDF5 files (train, val, etc.)
        input_fields: Dict with 'tracks' key containing list of track field names
        target_fields: Dict with 'vertex' key containing list of vertex field names
        cut_fraction: Fraction to cut from each tail (default 0.01 = 1%)

    Returns:
        dict: Statistics for each field
    """
    print(f"\nProcessing {len(hdf5_paths)} files:")
    for path in hdf5_paths:
        print(f"  - {path}")
    print(f"Outlier cut fraction: {cut_fraction * 100:.1f}%")

    statistics = {}

    # Process track fields
    if "tracks" in input_fields:
        separator = "=" * 80
        print(f"\n{separator}")
        print(f"Processing TRACK fields: {input_fields['tracks']}")
        print(f"{separator}")

        for field in input_fields["tracks"]:
            print(f"\nProcessing track field: {field}...")

            # Collect data from all files
            all_data = []
            total_samples = 0

            for hdf5_path in hdf5_paths:
                with h5py.File(hdf5_path, "r") as f:
                    # Load all data for this field from tracks
                    data = f[f"tracks/{field}"][:]
                    all_data.append(data)
                    total_samples += len(data)
                    print(f"  Loaded {len(data):,} samples from {hdf5_path}")

            # Concatenate all data
            data_combined = np.concatenate(all_data)
            print(f"  Total: {total_samples:,} samples")
            print(f"  Raw range: [{data_combined.min():.4f}, {data_combined.max():.4f}]")

            # Compute statistics
            result_tuple = compute_statistics(data_combined, cut_fraction=cut_fraction)

            # Skip boolean fields
            if result_tuple is None:
                continue

            result, _data_original, data_clean, data_transformed = result_tuple

            statistics[field] = result

            # Create plots directory
            plot_dir = Path("plots/lhcb_statistics")
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Plot before and after transformation
            plot_before_after(data_clean, data_transformed, field, result, plot_dir)

    # Process vertex fields
    if "vertex" in target_fields:
        separator = "=" * 80
        print(f"\n{separator}")
        print(f"Processing VERTEX fields: {target_fields['vertex']}")
        print(f"{separator}")

        for field in target_fields["vertex"]:
            print(f"\nProcessing vertex field: {field}...")

            # Collect data from all files
            all_data = []
            total_samples = 0

            for hdf5_path in hdf5_paths:
                with h5py.File(hdf5_path, "r") as f:
                    # Load all data for this field from vertices
                    data = f[f"vertices/{field}"][:]
                    all_data.append(data)
                    total_samples += len(data)
                    print(f"  Loaded {len(data):,} samples from {hdf5_path}")

            # Concatenate all data
            data_combined = np.concatenate(all_data)
            print(f"  Total: {total_samples:,} samples")
            print(f"  Raw range: [{data_combined.min():.4f}, {data_combined.max():.4f}]")

            # Compute statistics
            result_tuple = compute_statistics(data_combined, cut_fraction=cut_fraction)

            # Skip boolean fields
            if result_tuple is None:
                continue

            result, _data_original, data_clean, data_transformed = result_tuple

            statistics[field] = result

            # Create plots directory
            plot_dir = Path("plots/lhcb_statistics")
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Plot before and after transformation
            plot_before_after(data_clean, data_transformed, field, result, plot_dir)

    return statistics


def write_yaml_config(statistics, output_path):
    """Write statistics to YAML configuration file."""
    config = {}

    for field, stats in statistics.items():
        config[field] = {
            "type": "std",
            "mean": stats["mean"],
            "std": stats["std"],
        }

    # Write to file using Path
    output_path = Path(output_path)
    with output_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Wrote configuration to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute statistics for LHCb variables")
    parser.add_argument(
        "--data-paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to HDF5 files (train, val, etc.)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base config YAML file to read field names from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/hepattn/experiments/lhcb/config/lhcb_var_transform.yaml",
        help="Output YAML file path",
    )
    parser.add_argument(
        "--cut-fraction",
        type=float,
        default=0.005,
        help="Fraction to cut from each tail for outlier removal (default 0.005 = 0.5%)",
    )

    args = parser.parse_args()

    # Read config to get selected fields
    config_path = Path(args.config)
    with config_path.open() as f:
        config = yaml.safe_load(f)

    # Extract fields from config
    input_fields = config["data"]["inputs"]
    target_fields = config["data"]["targets"]

    print(f"\nReading field names from config: {args.config}")
    print(f"Found {len(input_fields.get('tracks', []))} input track fields")
    print(f"Found {len(target_fields.get('vertex', []))} target vertex fields")

    # Compute statistics
    statistics = compute_statistics_from_hdf5(
        args.data_paths,
        input_fields=input_fields,
        target_fields=target_fields,
        cut_fraction=args.cut_fraction,
    )

    # Write YAML config
    write_yaml_config(statistics, args.output)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Processed {len(statistics)} fields")
    print(f"Outlier cut fraction: {args.cut_fraction * 100:.1f}% from each tail")

    print("\nPlots saved to: plots/lhcb_statistics/")
    print("Configuration saved to:", args.output)


if __name__ == "__main__":
    main()
