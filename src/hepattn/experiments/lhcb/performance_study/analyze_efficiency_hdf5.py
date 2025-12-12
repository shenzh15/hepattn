from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


def load_and_analyze_efficiency_from_hdf5(
    h5_path,
    output_dir="output",
    track_overlap_threshold=0.7,
    min_tracks=4,
    max_tracks=200,
    bin_width=5,
    target_vertex_class=0,  # 0=PV, 1=SV, 2=null
):
    """Load predictions and targets from HDF5 file and analyze vertex reconstruction efficiency.

    Uses track-based matching: if the fraction of shared tracks between predicted and true
    vertex exceeds the threshold, the match is considered successful.

    Args:
        h5_path: Path to the HDF5 file
        output_dir: Output directory for plots
        track_overlap_threshold: Track overlap threshold for matching
        min_tracks: Minimum track count (for binning)
        max_tracks: Maximum track count (for binning)
        bin_width: Bin width
        target_vertex_class: Vertex class to analyze (0=PV, 1=SV, 2=null)

    Returns:
        dict: Efficiency analysis results
    """
    # Define equal-width bins
    track_bins = []
    bin_idx = 0
    while min_tracks + bin_width * bin_idx <= max_tracks:
        bin_start = int(min_tracks + bin_idx * bin_width)
        bin_end = int(min_tracks + (bin_idx + 1) * bin_width - 1)
        track_bins.append((bin_start, bin_end))
        bin_idx += 1

    # Initialize cumulative counters
    bin_labels = [f"[{bin_start}, {bin_end}]" for bin_start, bin_end in track_bins]
    cumulative_true_vertex_counts = dict.fromkeys(bin_labels, 0)  # denominator
    cumulative_matched_counts = dict.fromkeys(bin_labels, 0)  # numerator

    all_event_true_vertex_track_counts = {}  # Store for potential later use

    print(f"Loading HDF5 file: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        event_ids = sorted([int(k) for k in f])
        print(f"Found {len(event_ids)} events")

        for event_idx, event_id in enumerate(event_ids):
            if event_idx % 500 == 0:
                print(f"Processing event {event_idx}/{len(event_ids)}...")

            event_group = f[str(event_id)]

            # Extract targets and predictions
            targets = event_group["targets"]
            true_vertex_valid = targets["vertex_valid"][0]
            true_vertex_tracks = targets["vertex_tracks_valid"][0]
            true_vertex_class = targets["vertex_class"][0]
            tracks_valid = targets["tracks_valid"][0]

            preds = event_group["preds"]["final"]
            pred_vertex_valid = preds["vertex_classification"]["vertex_valid"][0]
            pred_vertex_tracks = preds["track_assignment"]["vertex_tracks_valid"][0]
            pred_vertex_class = preds["vertex_classification"]["vertex_class"][0]

            # Build true vertex info and count tracks
            event_true_vertex_track_counts, true_vertex_info = _process_true_vertices(
                true_vertex_valid,
                true_vertex_class,
                true_vertex_tracks,
                tracks_valid,
                target_vertex_class,
                min_tracks,
                track_bins,
                bin_labels,
                cumulative_true_vertex_counts,
            )

            all_event_true_vertex_track_counts[event_id] = event_true_vertex_track_counts

            # Perform track-based matching and update matched counts
            _process_pred_vertices_and_match(
                pred_vertex_valid,
                pred_vertex_class,
                pred_vertex_tracks,
                tracks_valid,
                target_vertex_class,
                true_vertex_info,
                track_overlap_threshold,
                event_true_vertex_track_counts,
                min_tracks,
                track_bins,
                bin_labels,
                cumulative_matched_counts,
            )

    # Calculate efficiency and errors
    bin_efficiencies, bin_efficiency_errors = _calculate_efficiencies(
        bin_labels,
        cumulative_true_vertex_counts,
        cumulative_matched_counts,
    )

    # Overall efficiency
    total_n_true = sum(cumulative_true_vertex_counts.values())
    total_n_matched = sum(cumulative_matched_counts.values())
    total_efficiency = total_n_matched / total_n_true if total_n_true > 0 else 0.0
    total_efficiency_error = _calculate_total_efficiency_error(total_n_true, total_n_matched, total_efficiency)

    # Prepare and filter plotting data
    plot_data = _prepare_plot_data(
        bin_labels,
        track_bins,
        bin_efficiencies,
        bin_efficiency_errors,
        cumulative_true_vertex_counts,
        cumulative_matched_counts,
    )

    if plot_data is None:
        print("No efficiency data to plot")
        return None

    # Create and save plot
    _create_efficiency_plot(
        plot_data,
        bin_width,
        track_overlap_threshold,
        output_dir,
        total_efficiency,
        total_efficiency_error,
        total_n_true,
        total_n_matched,
        target_vertex_class,
        bin_labels,
        bin_efficiencies,
        bin_efficiency_errors,
    )

    return {
        "bin_labels": bin_labels,
        "track_bins": track_bins,
        "bin_centers": plot_data["bin_centers"],
        "efficiencies": list(bin_efficiencies.values()),
        "efficiency_errors": list(bin_efficiency_errors.values()),
        "n_true_list": plot_data["n_true_list"],
        "n_matched_list": plot_data["n_matched_list"],
        "total_efficiency": total_efficiency,
        "total_efficiency_error": total_efficiency_error,
        "all_event_true_vertex_track_counts": all_event_true_vertex_track_counts,
    }


def _process_true_vertices(
    true_vertex_valid,
    true_vertex_class,
    true_vertex_tracks,
    tracks_valid,
    target_vertex_class,
    min_tracks,
    track_bins,
    bin_labels,
    cumulative_true_vertex_counts,
):
    """Process true vertices and update cumulative counts."""
    event_true_vertex_track_counts = {}
    true_vertex_info = []

    for vtx_idx in range(100):
        if not (true_vertex_valid[vtx_idx] and true_vertex_class[vtx_idx] == target_vertex_class):
            continue

        track_mask = true_vertex_tracks[vtx_idx] & tracks_valid
        n_tracks = int(track_mask.sum())
        track_indices = np.where(track_mask)[0]

        true_vertex_info.append({
            "vertex_id": vtx_idx,
            "n_tracks": n_tracks,
            "track_indices": set(track_indices),
            "vertex_class": int(true_vertex_class[vtx_idx]),
        })
        event_true_vertex_track_counts[vtx_idx] = n_tracks

        if n_tracks >= min_tracks:
            for b_idx, (bin_start, bin_end) in enumerate(track_bins):
                if bin_start <= n_tracks <= bin_end:
                    cumulative_true_vertex_counts[bin_labels[b_idx]] += 1
                    break

    return event_true_vertex_track_counts, true_vertex_info


def _process_pred_vertices_and_match(
    pred_vertex_valid,
    pred_vertex_class,
    pred_vertex_tracks,
    tracks_valid,
    target_vertex_class,
    true_vertex_info,
    track_overlap_threshold,
    event_true_vertex_track_counts,
    min_tracks,
    track_bins,
    bin_labels,
    cumulative_matched_counts,
):
    """Process predicted vertices, perform matching, and update matched counts."""
    matched_true_vertex_ids = set()

    for pred_idx in range(100):
        if not (pred_vertex_valid[pred_idx] and pred_vertex_class[pred_idx] == target_vertex_class):
            continue

        pred_track_mask = pred_vertex_tracks[pred_idx] & tracks_valid
        pred_track_indices = set(np.where(pred_track_mask)[0])

        if len(pred_track_indices) == 0:
            continue

        best_match_id = _find_best_matching_vertex(
            pred_track_indices,
            true_vertex_info,
            track_overlap_threshold,
        )

        if best_match_id is not None:
            matched_true_vertex_ids.add(best_match_id)

    # Add matched true vertices to numerator bins
    for matched_id in matched_true_vertex_ids:
        if matched_id not in event_true_vertex_track_counts:
            continue
        n_tracks = event_true_vertex_track_counts[matched_id]
        if n_tracks >= min_tracks:
            for b_idx, (bin_start, bin_end) in enumerate(track_bins):
                if bin_start <= n_tracks <= bin_end:
                    cumulative_matched_counts[bin_labels[b_idx]] += 1
                    break


def _find_best_matching_vertex(pred_track_indices, true_vertex_info, track_overlap_threshold):
    """Find the best matching true vertex based on track overlap.

    A match requires BOTH:
    - true_overlap_ratio (recall) = intersection / true_n_tracks >= threshold
    - rec_overlap_ratio (purity) = intersection / rec_n_tracks >= threshold
    """
    best_match_id = None
    best_true_overlap = 0

    rec_n_tracks = len(pred_track_indices)
    if rec_n_tracks == 0:
        return None

    for true_vtx in true_vertex_info:
        true_track_indices = true_vtx["track_indices"]
        if len(true_track_indices) == 0:
            continue

        intersection = len(pred_track_indices & true_track_indices)
        true_overlap_ratio = intersection / len(true_track_indices)  # recall
        rec_overlap_ratio = intersection / rec_n_tracks  # purity

        # Both recall and purity must be >= threshold
        if true_overlap_ratio >= track_overlap_threshold and rec_overlap_ratio >= track_overlap_threshold and true_overlap_ratio > best_true_overlap:
            best_true_overlap = true_overlap_ratio
            best_match_id = true_vtx["vertex_id"]

    return best_match_id


def _calculate_efficiencies(bin_labels, cumulative_true_vertex_counts, cumulative_matched_counts):
    """Calculate efficiency and Clopper-Pearson errors for each bin."""
    bin_efficiencies = {}
    bin_efficiency_errors = {}

    for bin_label in bin_labels:
        n_true = cumulative_true_vertex_counts[bin_label]
        n_matched = cumulative_matched_counts[bin_label]
        efficiency = n_matched / n_true if n_true > 0 else 0.0
        bin_efficiencies[bin_label] = efficiency

        if n_true > 0:
            k, n = n_matched, n_true
            lower_bound = beta.ppf(0.1585, k, n - k + 1) if k > 0 else 0.0
            upper_bound = beta.ppf(0.8415, k + 1, n - k) if k < n else 1.0

            if np.isnan(lower_bound) or np.isnan(upper_bound):
                bin_efficiency_errors[bin_label] = {"lower": 0.0, "upper": 0.0}
            else:
                bin_efficiency_errors[bin_label] = {
                    "lower": efficiency - lower_bound,
                    "upper": upper_bound - efficiency,
                }
        else:
            bin_efficiency_errors[bin_label] = {"lower": 0.0, "upper": 0.0}

    return bin_efficiencies, bin_efficiency_errors


def _calculate_total_efficiency_error(total_n_true, total_n_matched, total_efficiency):
    """Calculate overall efficiency error using Clopper-Pearson method."""
    if total_n_true == 0:
        return 0.0

    k_total, n_total = total_n_matched, total_n_true
    lower_total = beta.ppf(0.1585, k_total, n_total - k_total + 1) if k_total > 0 else 0.0
    upper_total = beta.ppf(0.8415, k_total + 1, n_total - k_total) if k_total < n_total else 1.0

    return (total_efficiency - lower_total + upper_total - total_efficiency) / 2


def _prepare_plot_data(
    bin_labels,
    track_bins,
    bin_efficiencies,
    bin_efficiency_errors,
    cumulative_true_vertex_counts,
    cumulative_matched_counts,
):
    """Prepare data for plotting, filtering bins with no data."""
    bin_centers = [(s + e) / 2 for s, e in track_bins]
    n_true_list = [cumulative_true_vertex_counts[label] for label in bin_labels]
    n_matched_list = [cumulative_matched_counts[label] for label in bin_labels]

    x_plot, eff_plot, err_lower_plot, err_upper_plot = [], [], [], []
    n_true_plot, n_matched_plot = [], []

    for idx, center in enumerate(bin_centers):
        if n_true_list[idx] > 0:
            x_plot.append(center)
            eff_plot.append(bin_efficiencies[bin_labels[idx]])
            err_lower_plot.append(bin_efficiency_errors[bin_labels[idx]]["lower"])
            err_upper_plot.append(bin_efficiency_errors[bin_labels[idx]]["upper"])
            n_true_plot.append(n_true_list[idx])
            n_matched_plot.append(n_matched_list[idx])

    if not x_plot:
        return None

    return {
        "x_plot": x_plot,
        "eff_plot": eff_plot,
        "err_lower_plot": err_lower_plot,
        "err_upper_plot": err_upper_plot,
        "n_true_plot": n_true_plot,
        "n_matched_plot": n_matched_plot,
        "bin_centers": bin_centers,
        "n_true_list": n_true_list,
        "n_matched_list": n_matched_list,
    }


def _create_efficiency_plot(
    plot_data,
    bin_width,
    track_overlap_threshold,
    output_dir,
    total_efficiency,
    total_efficiency_error,
    total_n_true,
    total_n_matched,
    target_vertex_class,
    bin_labels,
    bin_efficiencies,
    bin_efficiency_errors,
):
    """Create and save the efficiency plot."""
    x_plot = plot_data["x_plot"]
    eff_plot = plot_data["eff_plot"]
    n_true_plot = plot_data["n_true_plot"]
    n_matched_plot = plot_data["n_matched_plot"]

    _, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    # Efficiency plot with confidence interval
    lower_bounds = [eff - err for eff, err in zip(eff_plot, plot_data["err_lower_plot"], strict=True)]
    upper_bounds = [eff + err for eff, err in zip(eff_plot, plot_data["err_upper_plot"], strict=True)]

    ax1.plot(x_plot, eff_plot, "bo-", linewidth=2, markersize=8, label="Efficiency (Track-based)")
    ax1.fill_between(
        x_plot,
        lower_bounds,
        upper_bounds,
        alpha=0.3,
        color="blue",
        label="Clopper-Pearson 1-sigma CI",
    )

    ax1.set_xlabel("Number of tracks of primary vertex", fontsize=12)
    ax1.set_ylabel("Efficiency", fontsize=12)
    ax1.set_title(
        f"Vertex Reconstruction Efficiency vs Track Count\nTrack overlap threshold: {track_overlap_threshold}",
        fontsize=14,
    )
    ax1.set_ylim(0, 1.27)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Bar plot
    x_pos = np.array(x_plot)
    max_vertex_count = max(max(n_true_plot), max(n_matched_plot))
    ax2_max = max_vertex_count * 2.3

    ax2.bar(
        x_pos,
        n_true_plot,
        bin_width,
        alpha=0.8,
        color="lightblue",
        label="Total true vertices",
        linewidth=0.5,
    )
    ax2.bar(
        x_pos,
        n_matched_plot,
        bin_width,
        alpha=0.9,
        color="darkorange",
        label="Matched vertices",
        linewidth=0.5,
    )

    ax2.set_ylim(0, ax2_max)
    ax2.set_yticklabels([])
    ax2.tick_params(axis="y", which="both", length=0)
    ax2.legend(loc="upper right")

    # Add numerical labels
    for x, n_true, n_matched in zip(x_pos, n_true_plot, n_matched_plot, strict=True):
        if n_true > 0:
            ax2.text(
                x,
                n_true + ax2_max * 0.03,
                str(n_true),
                ha="center",
                va="bottom",
                fontsize=7,
                weight="normal",
                color="darkblue",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        if n_matched > 0:
            y_pos = n_matched / 2
            if n_matched < max_vertex_count * 0.15:
                y_pos = n_matched + ax2_max * 0.01
            ax2.text(
                x,
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
    plot_file = output_path / "vertex_efficiency_hdf5.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    # Print results
    _print_results(
        track_overlap_threshold,
        target_vertex_class,
        total_efficiency,
        total_efficiency_error,
        total_n_true,
        total_n_matched,
        eff_plot,
        bin_labels,
        bin_efficiencies,
        bin_efficiency_errors,
        plot_file,
    )


def _print_results(
    track_overlap_threshold,
    target_vertex_class,
    total_efficiency,
    total_efficiency_error,
    total_n_true,
    total_n_matched,
    eff_plot,
    bin_labels,
    bin_efficiencies,
    bin_efficiency_errors,
    output_path,
):
    """Print efficiency analysis summary."""
    class_names = {0: "PV", 1: "SV", 2: "null"}
    print(f"\n{'-' * 60}")
    print("EFFICIENCY ANALYSIS SUMMARY - HDF5 DATA")
    print(f"{'-' * 60}")
    print(f"Track overlap threshold: {track_overlap_threshold}")
    print(f"Vertex class: {target_vertex_class} ({class_names.get(target_vertex_class, 'unknown')})")
    print(f"Overall efficiency: {total_efficiency:.3f} +/- {total_efficiency_error:.3f} ({total_n_matched}/{total_n_true})")

    if eff_plot:
        best_eff_idx = np.argmax(eff_plot)
        worst_eff_idx = np.argmin([e for e in eff_plot if e > 0] or [0])

        # Find corresponding bin indices
        eff_values = list(bin_efficiencies.values())
        n_true_values = [1 if bin_efficiencies[label] > 0 else 0 for label in bin_labels]

        best_bin_idx = 0
        worst_bin_idx = 0
        plot_count = 0
        for idx, n_true in enumerate(n_true_values):
            if eff_values[idx] > 0 or n_true > 0:
                if plot_count == best_eff_idx:
                    best_bin_idx = idx
                if plot_count == worst_eff_idx:
                    worst_bin_idx = idx
                plot_count += 1

        best_err = bin_efficiency_errors[bin_labels[best_bin_idx]]
        worst_err = bin_efficiency_errors[bin_labels[worst_bin_idx]]

        best_eff = eff_plot[best_eff_idx]
        worst_eff = eff_plot[worst_eff_idx]
        print(f"Best efficiency: {best_eff:.3f} +{best_err['upper']:.3f}/-{best_err['lower']:.3f} in bin {bin_labels[best_bin_idx]}")
        print(f"Worst efficiency: {worst_eff:.3f} +{worst_err['upper']:.3f}/-{worst_err['lower']:.3f} in bin {bin_labels[worst_bin_idx]}")

    print(f"Total true vertices analyzed: {total_n_true}")
    print(f"Total matched vertices: {total_n_matched}")
    print(f"Output saved to: {output_path}")
    print(f"{'-' * 60}\n")


if __name__ == "__main__":
    # Example usage
    h5_file = "epoch=099-val_loss=8.19651_version5_hdf5_eval.h5"
    out_dir = "output"

    results = load_and_analyze_efficiency_from_hdf5(
        h5_path=h5_file,
        output_dir=out_dir,
        track_overlap_threshold=0.7,
        min_tracks=4,
        max_tracks=200,
        bin_width=5,
        target_vertex_class=0,  # 0=PV, 1=SV, 2=null
    )
