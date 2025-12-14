"""Utility functions for vertex reconstruction performance analysis."""

import numpy as np
from scipy.stats import beta

# Constants
MAX_VERTICES = 100  # Maximum number of vertices per event from MaskFormer setup


def build_true_vertex_info(
    target_vertex_valid,
    target_vertex_class,
    target_vertex_tracks_valid,
    tracks_valid,
    selected_vertex_class,
    min_tracks=0,
):
    """Build true vertex information dictionary with optional filtering.

    Args:
        target_vertex_valid: Array of vertex validity flags
        target_vertex_class: Array of vertex class labels
        target_vertex_tracks_valid: Array of vertex-track assignment flags
        tracks_valid: Array of track validity flags
        selected_vertex_class: Vertex class to filter (0=PV, 1=SV, 2=null)
        min_tracks: Minimum track count threshold (default: 0, no filtering)

    Returns:
        dict: vertex_info dict mapping vertex_id to {track_indices}
    """
    true_vertex_info = {}

    for vertex_idx in range(MAX_VERTICES):
        if not (target_vertex_valid[vertex_idx] and target_vertex_class[vertex_idx] == selected_vertex_class):
            continue

        track_mask = target_vertex_tracks_valid[vertex_idx] & tracks_valid
        track_indices = np.where(track_mask)[0]
        num_tracks = len(track_indices)

        # Apply minimum track count filter
        if num_tracks < min_tracks:
            continue

        true_vertex_info[vertex_idx] = {
            "track_indices": set(track_indices),
        }

    return true_vertex_info


def calculate_clopper_pearson_errors(k, n, rate):
    """Calculate Clopper-Pearson 1-sigma confidence interval errors.

    Args:
        k: Number of successes
        n: Total number of trials
        rate: Observed rate (k/n)

    Returns:
        tuple: (error_lower, error_upper)
    """
    if n == 0:
        return 0.0, 0.0

    lower_bound = beta.ppf(0.1585, k, n - k + 1) if k > 0 else 0.0
    upper_bound = beta.ppf(0.8415, k + 1, n - k) if k < n else 1.0

    if np.isnan(lower_bound) or np.isnan(upper_bound):
        return 0.0, 0.0

    return rate - lower_bound, upper_bound - rate
