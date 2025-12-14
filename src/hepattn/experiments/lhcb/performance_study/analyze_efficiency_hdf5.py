import operator
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from hepattn.experiments.lhcb.performance_study.utils import (
    MAX_VERTICES,
    build_true_vertex_info,
    calculate_clopper_pearson_errors,
)


def analyze_vertex_efficiency(
    hdf5_path,
    output_dir="output",
    overlap_threshold=0.7,
    min_tracks=4,
    max_tracks=200,
    bin_width=5,
    selected_vertex_class=0,  # 0=PV, 1=SV, 2=null
    method_name="HepATTn",
):
    """Analyze vertex reconstruction efficiency from HDF5 predictions.

    Uses track-based matching: a match is successful if both recall and precision
    of shared tracks exceed the threshold.

    Args:
        hdf5_path: Path to the HDF5 file containing predictions and targets
        output_dir: Output directory for plots and CSV files
        overlap_threshold: Track overlap threshold for matching (both recall and precision)
        min_tracks: Minimum track count for binning
        max_tracks: Maximum track count for binning
        bin_width: Width of each bin
        selected_vertex_class: Vertex class to analyze (0=PV, 1=SV, 2=null)
        method_name: Name of the method for comparison plots
    """
    # Create equal-width bins for track count
    num_bins = (max_tracks - min_tracks) // bin_width + 1
    bins = [(min_tracks + i * bin_width, min_tracks + (i + 1) * bin_width - 1) for i in range(num_bins)]

    # Initialize per-bin counters (using bin tuples as keys)
    true_vertices_per_bin = dict.fromkeys(bins, 0)
    matched_vertices_per_bin = dict.fromkeys(bins, 0)

    print(f"Loading HDF5 file: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as file:
        event_ids = sorted(int(key) for key in file)
        print(f"Found {len(event_ids)} events")

        for event_id in tqdm(event_ids, desc="Processing events"):
            event_data = file[str(event_id)]

            # Extract targets and predictions from HDF5
            targets = event_data["targets"]
            target_vertex_valid = targets["vertex_valid"][0]
            target_vertex_tracks_valid = targets["vertex_tracks_valid"][0]
            target_vertex_class = targets["vertex_class"][0]
            tracks_valid = targets["tracks_valid"][0]

            predictions = event_data["preds"]["final"]
            pred_vertex_valid = predictions["vertex_classification"]["vertex_valid"][0]
            pred_vertex_tracks_valid = predictions["track_assignment"]["vertex_tracks_valid"][0]
            pred_vertex_class = predictions["vertex_classification"]["vertex_class"][0]

            # Build true vertex info
            true_vertex_info = build_true_vertex_info(
                target_vertex_valid,
                target_vertex_class,
                target_vertex_tracks_valid,
                tracks_valid,
                selected_vertex_class,
                min_tracks,
            )

            # Process vertices and count true/matched vertices
            _process_vertices_for_efficiency(
                true_vertex_info,
                pred_vertex_valid,
                pred_vertex_class,
                pred_vertex_tracks_valid,
                tracks_valid,
                selected_vertex_class,
                overlap_threshold,
                bins,
                true_vertices_per_bin,
                matched_vertices_per_bin,
            )

    # Calculate efficiency and errors
    efficiencies, efficiency_errors = _calculate_efficiencies(bins, true_vertices_per_bin, matched_vertices_per_bin)

    # Calculate overall efficiency and error bounds
    overall_stats = _calculate_overall_efficiency(true_vertices_per_bin, matched_vertices_per_bin)

    # Prepare plotting data (filter bins with no data)
    plot_data = _prepare_plot_data(bins, efficiencies, efficiency_errors, true_vertices_per_bin, matched_vertices_per_bin)

    if plot_data is None:
        print("No efficiency data to plot")
        return

    # Create and save plot
    _create_efficiency_plot(
        plot_data,
        bin_width,
        overlap_threshold,
        output_dir,
        method_name,
    )

    # Print summary results
    _print_results(
        overlap_threshold,
        selected_vertex_class,
        overall_stats,
        plot_data["efficiency"],
        bins,
        efficiency_errors,
        true_vertices_per_bin,
    )

    # Save results to CSV
    _save_efficiency_to_csv(
        bins,
        plot_data,
        efficiencies,
        efficiency_errors,
        true_vertices_per_bin,
        matched_vertices_per_bin,
        overall_stats,
        overlap_threshold,
        output_dir,
        method_name,
    )


def _process_vertices_for_efficiency(
    true_vertex_info,
    pred_vertex_valid,
    pred_vertex_class,
    pred_vertex_tracks_valid,
    tracks_valid,
    selected_vertex_class,
    overlap_threshold,
    bins,
    true_vertices_per_bin,
    matched_vertices_per_bin,
):
    """Process vertices and count both denominator and numerator for efficiency.

    Args:
        true_vertex_info: Dict of vertex_id to {track_indices} (already filtered by min_tracks)
        pred_vertex_valid: Array of predicted vertex validity flags
        pred_vertex_class: Array of predicted vertex class labels
        pred_vertex_tracks_valid: Array of predicted vertex-track assignments
        tracks_valid: Array of track validity flags
        selected_vertex_class: Vertex class to filter (0=PV, 1=SV, 2=null)
        overlap_threshold: Track overlap threshold for matching
        bins: List of (bin_start, bin_end) tuples
        true_vertices_per_bin: Dict to update with true vertex counts (denominator)
        matched_vertices_per_bin: Dict to update with matched vertex counts (numerator)
    """
    # Count true vertices per bin (denominator)
    for vertex_data in true_vertex_info.values():
        num_tracks = len(vertex_data["track_indices"])

        # Update bin counts (denominator)
        for bin_start, bin_end in bins:
            if bin_start <= num_tracks <= bin_end:
                true_vertices_per_bin[bin_start, bin_end] += 1
                break

    # Match predicted vertices and collect all matched true vertex IDs (numerator)
    matched_true_ids = set()
    for pred_idx in range(MAX_VERTICES):
        if not (pred_vertex_valid[pred_idx] and pred_vertex_class[pred_idx] == selected_vertex_class):
            continue

        pred_track_mask = pred_vertex_tracks_valid[pred_idx] & tracks_valid
        pred_track_indices = set(np.where(pred_track_mask)[0])

        if not pred_track_indices:
            continue

        best_match_id = _find_best_match(pred_track_indices, true_vertex_info, overlap_threshold)
        if best_match_id is not None:
            matched_true_ids.add(best_match_id)

    # Fill all collected matched vertices per bin (numerator)
    for matched_id in matched_true_ids:
        num_tracks = len(true_vertex_info[matched_id]["track_indices"])
        for bin_start, bin_end in bins:
            if bin_start <= num_tracks <= bin_end:
                matched_vertices_per_bin[bin_start, bin_end] += 1
                break


def _find_best_match(pred_track_indices, true_vertex_info, overlap_threshold):
    """Find the best matching true vertex for a predicted vertex.

    A match requires BOTH:
    - Recall = intersection / num_true_tracks >= threshold
    - Precision = intersection / num_pred_tracks >= threshold

    Returns:
        Best matching vertex ID or None if no match found
    """
    best_match_id = None
    best_recall = 0.0

    num_pred_tracks = len(pred_track_indices)
    if num_pred_tracks == 0:
        return None

    for vertex_id, vertex_data in true_vertex_info.items():
        true_track_indices = vertex_data["track_indices"]
        num_true_tracks = len(true_track_indices)

        if num_true_tracks == 0:
            continue

        # Calculate overlap metrics
        num_shared = len(pred_track_indices & true_track_indices)
        recall = num_shared / num_true_tracks
        precision = num_shared / num_pred_tracks

        # Both recall and precision must exceed threshold, and recall should be best so far
        if recall >= overlap_threshold and precision >= overlap_threshold and recall > best_recall:
            best_recall = recall
            best_match_id = vertex_id

    return best_match_id


def _calculate_efficiencies(bins, true_vertices_per_bin, matched_vertices_per_bin):
    """Calculate efficiency and Clopper-Pearson confidence intervals for each bin."""
    efficiencies = {}
    errors = {}

    for bin_tuple in bins:
        num_true = true_vertices_per_bin[bin_tuple]
        num_matched = matched_vertices_per_bin[bin_tuple]
        efficiency = num_matched / num_true if num_true > 0 else 0.0
        efficiencies[bin_tuple] = efficiency

        error_lower, error_upper = calculate_clopper_pearson_errors(num_matched, num_true, efficiency)
        errors[bin_tuple] = {"lower": error_lower, "upper": error_upper}

    return efficiencies, errors


def _calculate_overall_efficiency(true_vertices_per_bin, matched_vertices_per_bin):
    """Calculate overall efficiency and error bounds.

    Returns:
        dict: Dictionary containing overall statistics:
            - efficiency: overall efficiency value
            - error_lower: lower error bound
            - error_upper: upper error bound
            - total_true: total number of true vertices
            - total_matched: total number of matched vertices
    """
    total_true = sum(true_vertices_per_bin.values())
    total_matched = sum(matched_vertices_per_bin.values())
    overall_efficiency = total_matched / total_true if total_true > 0 else 0.0

    # Calculate Clopper-Pearson confidence intervals
    error_lower, error_upper = calculate_clopper_pearson_errors(total_matched, total_true, overall_efficiency)

    return {
        "efficiency": overall_efficiency,
        "error_lower": error_lower,
        "error_upper": error_upper,
        "total_true": total_true,
        "total_matched": total_matched,
    }


def _prepare_plot_data(bins, efficiencies, efficiency_errors, true_vertices_per_bin, matched_vertices_per_bin):
    """Prepare data for plotting, filtering out bins with no data."""
    bin_centers = [(start + end) / 2 for start, end in bins]
    num_true_list = [true_vertices_per_bin[bin_tuple] for bin_tuple in bins]
    num_matched_list = [matched_vertices_per_bin[bin_tuple] for bin_tuple in bins]

    # Filter bins with data
    x_values, eff_values, err_lower, err_upper = [], [], [], []
    num_true_filtered, num_matched_filtered = [], []

    for idx, bin_tuple in enumerate(bins):
        if num_true_list[idx] > 0:
            x_values.append(bin_centers[idx])
            eff_values.append(efficiencies[bin_tuple])
            err_lower.append(efficiency_errors[bin_tuple]["lower"])
            err_upper.append(efficiency_errors[bin_tuple]["upper"])
            num_true_filtered.append(num_true_list[idx])
            num_matched_filtered.append(num_matched_list[idx])

    if not x_values:
        return None

    return {
        "x": x_values,
        "efficiency": eff_values,
        "error_lower": err_lower,
        "error_upper": err_upper,
        "num_true_filtered": num_true_filtered,
        "num_matched_filtered": num_matched_filtered,
        "bin_centers": bin_centers,
        "num_true_all": num_true_list,
        "num_matched_all": num_matched_list,
    }


def _create_efficiency_plot(plot_data, bin_width, overlap_threshold, output_dir, method_name):
    """Create and save the efficiency plot with confidence intervals."""
    x = plot_data["x"]
    efficiency = plot_data["efficiency"]
    num_true = plot_data["num_true_filtered"]
    num_matched = plot_data["num_matched_filtered"]

    _, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    # Plot efficiency with confidence intervals
    lower_bounds = [eff - err for eff, err in zip(efficiency, plot_data["error_lower"], strict=True)]
    upper_bounds = [eff + err for eff, err in zip(efficiency, plot_data["error_upper"], strict=True)]

    ax1.plot(x, efficiency, "bo-", linewidth=2, markersize=8, label="Efficiency (Track-based)")
    ax1.fill_between(
        x,
        lower_bounds,
        upper_bounds,
        alpha=0.3,
        color="blue",
        label="Clopper-Pearson 1-sigma CI",
    )

    ax1.set_xlabel("Number of tracks per primary vertex", fontsize=12)
    ax1.set_ylabel("Efficiency", fontsize=12)
    ax1.set_title(
        f"Vertex Reconstruction Efficiency vs Track Count\nOverlap threshold: {overlap_threshold}",
        fontsize=14,
    )
    ax1.set_ylim(0, 1.27)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Add bar charts for counts
    x_array = np.array(x)
    max_count = max(max(num_true), max(num_matched))
    y_max = max_count * 2.3

    ax2.bar(
        x_array,
        num_true,
        bin_width,
        alpha=0.8,
        color="lightblue",
        label="Total true vertices",
        linewidth=0.5,
    )
    ax2.bar(
        x_array,
        num_matched,
        bin_width,
        alpha=0.9,
        color="darkorange",
        label="Matched vertices",
        linewidth=0.5,
    )

    ax2.set_ylim(0, y_max)
    ax2.set_yticklabels([])
    ax2.tick_params(axis="y", which="both", length=0)
    ax2.legend(loc="upper right")

    # Add numerical labels on bars
    for x_pos, n_true, n_matched in zip(x_array, num_true, num_matched, strict=True):
        if n_true > 0:
            ax2.text(
                x_pos,
                n_true + y_max * 0.03,
                str(n_true),
                ha="center",
                va="bottom",
                fontsize=7,
                color="darkblue",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        if n_matched > 0:
            y_pos = n_matched / 2 if n_matched >= max_count * 0.15 else n_matched + y_max * 0.01
            ax2.text(
                x_pos,
                y_pos,
                str(n_matched),
                ha="center",
                va="center",
                fontsize=7,
                weight="bold",
                color="white",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "darkred", "alpha": 0.9, "edgecolor": "none"},
            )

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / f"efficiency_{method_name.lower().replace(' ', '_')}.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()


def _print_results(
    overlap_threshold,
    selected_vertex_class,
    overall_stats,
    efficiency_values,
    bins,
    efficiency_errors,
    true_vertices_per_bin,
):
    """Print efficiency analysis summary to console."""
    class_names = {0: "PV", 1: "SV", 2: "null"}

    print(f"\n{'-' * 60}")
    print("EFFICIENCY ANALYSIS SUMMARY - HDF5 DATA")
    print(f"{'-' * 60}")
    print(f"Track overlap threshold: {overlap_threshold}")
    print(f"Vertex class: {selected_vertex_class} ({class_names.get(selected_vertex_class, 'unknown')})")
    eff = overall_stats["efficiency"]
    err_upper = overall_stats["error_upper"]
    err_lower = overall_stats["error_lower"]
    matched = overall_stats["total_matched"]
    total = overall_stats["total_true"]
    print(f"Overall efficiency: {eff:.3f} +{err_upper:.3f}/-{err_lower:.3f} ({matched}/{total})")

    if efficiency_values:
        # Map efficiency values to bins (only bins with data)
        bins_with_data = [bin_tuple for bin_tuple in bins if true_vertices_per_bin[bin_tuple] > 0]

        # Find best efficiency
        best_idx = int(np.argmax(efficiency_values))
        best_bin = bins_with_data[best_idx]
        best_eff = efficiency_values[best_idx]
        best_err = efficiency_errors[best_bin]

        # Find worst efficiency (excluding zero)
        non_zero_effs = [(i, eff) for i, eff in enumerate(efficiency_values) if eff > 0]
        if non_zero_effs:
            worst_idx, worst_eff = min(non_zero_effs, key=operator.itemgetter(1))
            worst_bin = bins_with_data[worst_idx]
            worst_err = efficiency_errors[worst_bin]
            print(f"Best efficiency: {best_eff:.3f} +{best_err['upper']:.3f}/-{best_err['lower']:.3f} in bin [{best_bin[0]}, {best_bin[1]}]")
            print(f"Worst efficiency: {worst_eff:.3f} +{worst_err['upper']:.3f}/-{worst_err['lower']:.3f} in bin [{worst_bin[0]}, {worst_bin[1]}]")
        else:
            print(f"Best efficiency: {best_eff:.3f} +{best_err['upper']:.3f}/-{best_err['lower']:.3f} in bin [{best_bin[0]}, {best_bin[1]}]")

    print(f"Total true vertices analyzed: {overall_stats['total_true']}")
    print(f"Total matched vertices: {overall_stats['total_matched']}")
    print(f"{'-' * 60}\n")


def _save_efficiency_to_csv(
    bins,
    plot_data,
    efficiencies,
    efficiency_errors,
    true_vertices_per_bin,
    matched_vertices_per_bin,
    overall_stats,
    overlap_threshold,
    output_dir,
    method_name,
):
    """Save efficiency results to CSV files for comparison with other methods."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed per-bin results
    bin_data = []
    for i, bin_tuple in enumerate(bins):
        bin_start, bin_end = bin_tuple
        bin_center = plot_data["bin_centers"][i]
        efficiency = efficiencies[bin_tuple]
        error_lower = efficiency_errors[bin_tuple]["lower"]
        error_upper = efficiency_errors[bin_tuple]["upper"]
        num_true = true_vertices_per_bin[bin_tuple]
        num_matched = matched_vertices_per_bin[bin_tuple]

        bin_data.append({
            "bin_start": bin_start,
            "bin_end": bin_end,
            "bin_center": bin_center,
            "efficiency": efficiency,
            "error_lower": error_lower,
            "error_upper": error_upper,
            "num_true": num_true,
            "num_matched": num_matched,
        })

    df = pd.DataFrame(bin_data)
    csv_path = output_path / f"efficiency_{method_name.lower().replace(' ', '_')}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved bin-level efficiency results to: {csv_path}")

    # Save overall summary
    summary_df = pd.DataFrame([
        {
            "total_matched": overall_stats["total_matched"],
            "total_true": overall_stats["total_true"],
            "overall_efficiency": overall_stats["efficiency"],
            "overall_error_lower": overall_stats["error_lower"],
            "overall_error_upper": overall_stats["error_upper"],
            "overlap_threshold": overlap_threshold,
            "method_name": method_name,
        }
    ])

    summary_path = output_path / f"efficiency_summary_{method_name.lower().replace(' ', '_')}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved overall efficiency summary to: {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze vertex reconstruction efficiency from HDF5 predictions file")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to the HDF5 file containing predictions and targets")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for plots and CSV files")
    parser.add_argument("--overlap_threshold", type=float, default=0.7, help="Track overlap threshold for matching")
    parser.add_argument("--min_tracks", type=int, default=4, help="Minimum track count for binning")
    parser.add_argument("--max_tracks", type=int, default=200, help="Maximum track count for binning")
    parser.add_argument("--bin_width", type=int, default=5, help="Bin width for track count histogram")
    parser.add_argument("--selected_vertex_class", type=int, default=0, help="Vertex class to analyze: 0=PV, 1=SV, 2=null")
    parser.add_argument("--method_name", type=str, default="HepATTn", help="Method name for comparison plots and CSV files")
    args = parser.parse_args()

    analyze_vertex_efficiency(
        hdf5_path=args.hdf5_path,
        output_dir=args.output_dir,
        overlap_threshold=args.overlap_threshold,
        min_tracks=args.min_tracks,
        max_tracks=args.max_tracks,
        bin_width=args.bin_width,
        selected_vertex_class=args.selected_vertex_class,
        method_name=args.method_name,
    )
