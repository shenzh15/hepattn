"""Convert LHCb ROOT files to HDF5 format for ML training.

Example usage:
    pixi run python src/hepattn/experiments/lhcb/prep.py -i data/lhcb/version3_small -o data/lhcb/version3_small_hdf5 --train_split 0.8
    pixi run python src/hepattn/experiments/lhcb/prep.py -i data/lhcb/version3 -o data/lhcb/version3_hdf5 --train_split 0.8 --n_workers 8
"""

from multiprocessing import Pool, cpu_count
from pathlib import Path

import h5py
import numpy as np
import uproot
from tqdm import tqdm


def calc_zbl_err(tx, ty, c00):
    """Calculate z-beamline error for track quality cuts.

    Formula: zbl_err = 1 / sqrt(W_00 * tx^2 + W_11 * ty^2)
    where W_00 = W_11 = 1/c00 (by symmetry)

    Parameters
    ----------
    tx, ty : array
        Track slopes (dx/dz, dy/dz)
    c00 : array
        Covariance cov(x,x) = cov(y,y) by symmetry

    Returns:
    --------
    zbl_err : array
        Uncertainty on z-beamline position (np.inf for invalid tracks)
    """
    # Initialize result with inf (will be cut)
    zbl_err = np.full_like(tx, np.inf, dtype=np.float64)

    # Calculate denominator: tx^2 + ty^2
    n = tx * tx + ty * ty

    # Check for valid tracks: n > 0 (not parallel to beam) and c00 > 0 (valid covariance)
    valid_mask = (n > 1e-10) & (c00 > 0)

    if not np.any(valid_mask):
        return zbl_err

    # For valid tracks: zbl_err = 1 / sqrt((tx^2 + ty^2) / c00) = sqrt(c00 / (tx^2 + ty^2))
    zbl_err[valid_mask] = np.sqrt(c00[valid_mask] / n[valid_mask])

    # Set inf for any NaN or inf results (additional safety)
    return np.where(np.isfinite(zbl_err), zbl_err, np.inf)


def _read_single_root_file(args):
    """Read a single ROOT file and return the data."""
    file_idx, root_file_path, branches_to_read = args

    root_file = uproot.open(root_file_path)
    tree = root_file["velo_kalman_filter/monitor_tree"]
    data = tree.arrays(branches_to_read, library="np")

    # Track file index for each track
    file_indices = np.full(len(data["event_number"]), file_idx)

    root_file.close()

    return file_idx, data, file_indices


def convert_root_to_hdf5(input_dir: str, output_dir: str, train_split: float = 0.8, n_workers: int | None = None):
    """Convert ROOT files to HDF5 format, creating train and validation sets.

    Parameters
    ----------
    input_dir : str
        Directory containing ROOT files
    output_dir : str
        Directory where HDF5 files will be saved
    train_split : float
        Fraction of events to use for training (default: 0.8)
    n_workers : int | None
        Number of parallel workers for reading ROOT files (default: cpu_count()-1)
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave one CPU free

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Get all ROOT files
    root_files = sorted(Path(input_dir).glob("*.root"))

    # Branches to read from ROOT files
    branches_to_read = [
        "beamPOCA_tx",
        "beamPOCA_ty",
        "beamPOCA_x",
        "beamPOCA_y",
        "beamPOCA_z",
        "beamPOCA_t",
        "backward",
        "ovtx_x",
        "ovtx_y",
        "ovtx_z",
        "ovtx_t",
        "c00",
        "c20",
        "c22",
        "c55",
        "fromPV",
        "chi2",
        "event_number",
    ]

    # Branches to save to HDF5 (excluding event_number)
    branches_to_save = [b for b in branches_to_read if b != "event_number"]

    print(f"Converting {len(root_files)} ROOT files to HDF5 format...")
    print(f"Output directory: {output_path}")
    print(f"Branches to read: {len(branches_to_read)}")
    print(f"Branches to save: {len(branches_to_save)}")
    print(f"Train/Val split: {train_split * 100:.0f}% / {(1 - train_split) * 100:.0f}%")
    print(f"Using {n_workers} parallel workers for reading ROOT files")

    # First pass: collect all data in parallel
    print("\nStep 1: Reading all ROOT files in parallel...")
    all_data = {branch: [] for branch in branches_to_read}
    all_file_indices = []

    # Prepare arguments for parallel processing
    file_args = [(idx, path, branches_to_read) for idx, path in enumerate(root_files)]

    # Read files in parallel
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(_read_single_root_file, file_args), total=len(root_files), desc="  Reading ROOT files"))

    # Sort results by file index to maintain order
    results.sort(key=lambda x: x[0])

    # Collect data from results
    for file_idx, data, file_indices in results:
        for branch in branches_to_read:
            all_data[branch].append(data[branch])
        all_file_indices.append(file_indices)

    # Concatenate all data
    print("\nStep 2: Concatenating data...")
    for branch in branches_to_read:
        all_data[branch] = np.concatenate(all_data[branch])

    file_indices = np.concatenate(all_file_indices)
    event_numbers = all_data["event_number"]

    # Apply quality cuts
    print("\nStep 2b: Applying quality cuts...")
    n_tracks_before = len(event_numbers)
    print(f"  Tracks before cuts: {n_tracks_before}")

    # Cut 1: c00 < 1
    c00 = all_data["c00"]
    c00_mask = c00 < 0.2
    n_c00 = np.sum(c00_mask)
    print(f"  After c00 < 0.2: {n_c00} tracks ({n_c00 / n_tracks_before * 100:.2f}%)")

    # Cut 2: chi2 < 100
    chi2 = all_data["chi2"]
    chi2_mask = chi2 < 100.0
    n_chi2 = np.sum(chi2_mask)
    print(f"  After chi2 < 100: {n_chi2} tracks ({n_chi2 / n_tracks_before * 100:.2f}%)")

    # Combined quality mask (all cuts must pass)
    quality_mask = c00_mask & chi2_mask

    # Apply mask to all data
    for branch in branches_to_read:
        all_data[branch] = all_data[branch][quality_mask]
    file_indices = file_indices[quality_mask]
    event_numbers = all_data["event_number"]

    n_tracks_after = len(event_numbers)
    n_removed = n_tracks_before - n_tracks_after
    print(f"  Final tracks after all cuts: {n_tracks_after}")
    print(f"  Total removed: {n_removed} ({n_removed / n_tracks_before * 100:.2f}%)")

    # Create unique event IDs (file_idx * 1000 + event_number)
    print("\nStep 3: Creating event splits...")
    # Note that the length of `unique_event_ids` here is the number of all tracks, not the number of events
    unique_event_ids = file_indices * 1000 + event_numbers
    unique_events = np.unique(unique_event_ids)

    # Split events
    n_train_events = int(len(unique_events) * train_split)
    train_events = unique_events[:n_train_events]
    val_events = unique_events[n_train_events:]

    print(f"  Total events: {len(unique_events)}")
    print(f"  Training events: {len(train_events)}")
    print(f"  Validation events: {len(val_events)}")

    # Split data - use np.isin for much faster membership test
    print("\nStep 4: Splitting data...")
    train_mask = np.isin(unique_event_ids, train_events)
    val_mask = ~train_mask

    train_data = {branch: all_data[branch][train_mask] for branch in branches_to_read}
    val_data = {branch: all_data[branch][val_mask] for branch in branches_to_read}

    train_event_ids = unique_event_ids[train_mask]
    val_event_ids = unique_event_ids[val_mask]

    print(f"  Training tracks: {np.sum(train_mask)}")
    print(f"  Validation tracks: {np.sum(val_mask)}")

    # Save files in parallel (train and val are different files)
    print("\nStep 5-6: Saving training and validation sets in parallel...")
    train_file = output_path / "train.h5"
    val_file = output_path / "val.h5"

    # Prepare arguments for parallel saving
    save_args = [
        (train_file, train_data, train_event_ids, branches_to_save, "training"),
        (val_file, val_data, val_event_ids, branches_to_save, "validation"),
    ]

    # Save both files in parallel
    with Pool(processes=2) as pool:
        pool.starmap(_save_hdf5, save_args)

    print("\n✓ Conversion complete!")


def _find_unique_vertices_with_tolerance(x, y, z, t, tol_xyz=0.0001, tol_t=0.0001):
    """Find unique vertices using tolerance-based rounding."""
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])

    # Round to tolerance precision
    decimals_xyz = int(-np.log10(tol_xyz))
    decimals_t = int(-np.log10(tol_t))

    x_rounded = np.round(x, decimals_xyz)
    y_rounded = np.round(y, decimals_xyz)
    z_rounded = np.round(z, decimals_xyz)
    t_rounded = np.round(t, decimals_t)

    # Combine coordinates
    vertex_coords = np.column_stack([x_rounded, y_rounded, z_rounded, t_rounded])

    # Find unique vertices
    unique_vertices, inverse_indices = np.unique(vertex_coords, axis=0, return_inverse=True)

    return unique_vertices, inverse_indices


def _save_hdf5(output_file, data, event_ids, branches, dataset_type):
    """Helper function to save data to HDF5 file."""
    with h5py.File(output_file, "w") as hf:
        # Create tracks group
        tracks_group = hf.create_group("tracks")

        # Store each branch's data
        for branch_name in branches:
            tracks_group.create_dataset(branch_name, data=data[branch_name], compression="lzf")

        # Calculate event indices using unique_event_ids
        unique_events, event_indices = np.unique(event_ids, return_index=True)
        event_indices = np.append(event_indices, len(event_ids))

        # Store event_indices in tracks group
        tracks_group.create_dataset("event_indices", data=event_indices, compression="lzf")

        # Extract and store vertices
        ovtx_x = data["ovtx_x"]
        ovtx_y = data["ovtx_y"]
        ovtx_z = data["ovtx_z"]
        ovtx_t = data["ovtx_t"]
        from_pv = data["fromPV"]

        # Process vertices using vectorized operations
        print(f"  Processing {len(unique_events)} events for {dataset_type}...")
        vertices_result = _process_all_vertices_vectorized(event_ids, unique_events, ovtx_x, ovtx_y, ovtx_z, ovtx_t, from_pv)

        # Create vertices group
        vertices_group = hf.create_group("vertices")
        if vertices_result["num_vertices"] > 0:
            vertices_group.create_dataset("ovtx_x", data=vertices_result["vertices"][:, 0], compression="lzf")
            vertices_group.create_dataset("ovtx_y", data=vertices_result["vertices"][:, 1], compression="lzf")
            vertices_group.create_dataset("ovtx_z", data=vertices_result["vertices"][:, 2], compression="lzf")
            vertices_group.create_dataset("ovtx_t", data=vertices_result["vertices"][:, 3], compression="lzf")
            vertices_group.create_dataset("n_tracks", data=vertices_result["n_tracks"], compression="lzf")
            vertices_group.create_dataset("is_pv", data=vertices_result["is_pv"], compression="lzf")
            vertices_group.create_dataset("vertex_event_indices", data=vertices_result["vertex_event_indices"], compression="lzf")
            tracks_group.create_dataset("map_vertex", data=vertices_result["track_to_vertex"], compression="lzf")

        # Metadata
        hf.attrs["num_tracks"] = len(event_ids)
        hf.attrs["num_events"] = len(unique_events)
        hf.attrs["num_vertices"] = vertices_result["num_vertices"]
        hf.attrs["dataset_type"] = dataset_type
        hf.attrs["branch_names"] = branches

    print(f"  {dataset_type.capitalize()}: {len(event_ids)} tracks, {len(unique_events)} events, {vertices_result['num_vertices']} vertices")


def _process_all_vertices_vectorized(event_ids, unique_events, ovtx_x, ovtx_y, ovtx_z, ovtx_t, from_pv):
    """Process all events' vertices using vectorized operations for better performance."""
    all_vertices = []
    all_vertex_n_tracks = []
    all_vertex_is_pv = []
    vertex_event_indices = []
    all_inverse = np.full(len(event_ids), -1, dtype=np.int64)

    # Process events in batches for better performance
    batch_size = 1000
    num_batches = (len(unique_events) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="    Processing event batches", leave=False):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(unique_events))
        batch_events = unique_events[start_idx:end_idx]

        for event_id in batch_events:
            event_mask = event_ids == event_id
            if not np.any(event_mask):
                vertex_event_indices.append(len(all_vertices))
                continue

            vertex_event_indices.append(len(all_vertices))

            # Get tracks for this event
            event_track_indices = np.where(event_mask)[0]
            event_x = ovtx_x[event_mask]
            event_y = ovtx_y[event_mask]
            event_z = ovtx_z[event_mask]
            event_t = ovtx_t[event_mask]
            event_from_pv = from_pv[event_mask]

            # Find unique vertices
            unique_verts, inverse = _find_unique_vertices_with_tolerance(event_x, event_y, event_z, event_t)

            if len(unique_verts) == 0:
                continue

            # Vectorized: count tracks per vertex
            vertex_counts = np.bincount(inverse)

            # Vectorized: check if all tracks in each vertex are from PV
            # For each vertex, sum up fromPV values and compare with track count
            from_pv_counts = np.bincount(inverse, weights=(event_from_pv == 1).astype(int))
            vertex_is_pv = from_pv_counts == vertex_counts

            # Filter vertices with at least 4 tracks
            valid_mask = vertex_counts >= 4
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                continue

            # Create mapping from old to new vertex indices (vectorized)
            old_to_new_vtx = np.full(len(unique_verts), -1, dtype=np.int64)
            old_to_new_vtx[valid_indices] = np.arange(len(valid_indices))

            # Add valid vertices
            all_vertices.extend(unique_verts[valid_indices])
            all_vertex_n_tracks.extend(vertex_counts[valid_indices])
            all_vertex_is_pv.extend(vertex_is_pv[valid_indices])

            # Vectorized: remap track-to-vertex assignments
            new_vertex_ids = old_to_new_vtx[inverse]
            valid_track_mask = new_vertex_ids >= 0
            all_inverse[event_track_indices[valid_track_mask]] = new_vertex_ids[valid_track_mask]

    vertex_event_indices.append(len(all_vertices))

    return {
        "vertices": np.array(all_vertices) if len(all_vertices) > 0 else np.array([]),
        "n_tracks": np.array(all_vertex_n_tracks, dtype=np.int64) if len(all_vertex_n_tracks) > 0 else np.array([]),
        "is_pv": np.array(all_vertex_is_pv, dtype=bool) if len(all_vertex_is_pv) > 0 else np.array([]),
        "vertex_event_indices": np.array(vertex_event_indices, dtype=np.int64),
        "track_to_vertex": all_inverse,
        "num_vertices": len(all_vertices),
    }


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Convert LHCb ROOT files to HDF5 format")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Input directory containing ROOT files")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory for HDF5 files")
    parser.add_argument("--train_split", type=float, default=0.8, help="Fraction of events for training (default: 0.8)")
    parser.add_argument("--n_workers", type=int, default=None, help="Number of parallel workers for reading (default: cpu_count()-1)")

    args = parser.parse_args()

    convert_root_to_hdf5(args.input_dir, args.output_dir, args.train_split, args.n_workers)
