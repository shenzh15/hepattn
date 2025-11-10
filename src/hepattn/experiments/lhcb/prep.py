"""Convert LHCb ROOT files to HDF5 format for ML training.

Example usage:
    pixi run python src/hepattn/experiments/lhcb/prep.py -i data/lhcb/version3_small -o data/lhcb/version3_small_hdf5 --train_split 0.8
    pixi run python src/hepattn/experiments/lhcb/prep.py -i data/lhcb/version3 -o data/lhcb/version3_hdf5 --train_split 0.8
"""

from pathlib import Path

import h5py
import numpy as np
import uproot


def convert_root_to_hdf5(input_dir: str, output_dir: str, train_split: float = 0.8):
    """Convert ROOT files to HDF5 format, creating train and validation sets.

    Parameters
    ----------
    input_dir : str
        Directory containing ROOT files
    output_dir : str
        Directory where HDF5 files will be saved
    train_split : float
        Fraction of events to use for training (default: 0.8)
    """
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

    # First pass: collect all data
    print("\nStep 1: Reading all ROOT files...")
    all_data = {branch: [] for branch in branches_to_read}
    all_file_indices = []

    for file_idx, root_file_path in enumerate(root_files):
        if file_idx % 10 == 0:
            print(f"  Reading file {file_idx + 1}/{len(root_files)}: {Path(root_file_path).name}")

        root_file = uproot.open(root_file_path)
        tree = root_file["velo_kalman_filter/monitor_tree"]
        data = tree.arrays(branches_to_read, library="np")

        # Append data
        for branch in branches_to_read:
            all_data[branch].append(data[branch])

        # Track file index for each track
        all_file_indices.append(np.full(len(data["event_number"]), file_idx))

        root_file.close()

    # Concatenate all data
    print("\nStep 2: Concatenating data...")
    for branch in branches_to_read:
        all_data[branch] = np.concatenate(all_data[branch])

    file_indices = np.concatenate(all_file_indices)
    event_numbers = all_data["event_number"]

    # Create unique event IDs (file_idx * 1000 + event_number)
    print("\nStep 3: Creating event splits...")
    # Note that the length of `unique_event_ids` here is the number of all tracks, not the number of events
    unique_event_ids = file_indices * 1000 + event_numbers
    unique_events = np.unique(unique_event_ids)

    # Split events
    n_train_events = int(len(unique_events) * train_split)
    train_events = set(unique_events[:n_train_events])
    val_events = set(unique_events[n_train_events:])

    print(f"  Total events: {len(unique_events)}")
    print(f"  Training events: {len(train_events)}")
    print(f"  Validation events: {len(val_events)}")

    # Split data
    print("\nStep 4: Splitting data...")
    train_mask = np.array([eid in train_events for eid in unique_event_ids])
    val_mask = ~train_mask

    train_data = {branch: all_data[branch][train_mask] for branch in branches_to_read}
    val_data = {branch: all_data[branch][val_mask] for branch in branches_to_read}

    train_event_ids = unique_event_ids[train_mask]
    val_event_ids = unique_event_ids[val_mask]

    print(f"  Training tracks: {np.sum(train_mask)}")
    print(f"  Validation tracks: {np.sum(val_mask)}")

    # Save files
    print("\nStep 5: Saving training set...")
    train_file = output_path / "train.h5"
    _save_hdf5(train_file, train_data, train_event_ids, branches_to_save, "training")

    print("\nStep 6: Saving validation set...")
    val_file = output_path / "val.h5"
    _save_hdf5(val_file, val_data, val_event_ids, branches_to_save, "validation")


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

        # Store all vertices information
        all_vertices = []
        all_vertex_n_tracks = []
        all_vertex_is_pv = []
        vertex_event_indices = []
        all_inverse = np.full(len(event_ids), -1, dtype=np.int64)  # Initialize with -1 for all tracks

        for event_id in unique_events:
            event_mask = event_ids == event_id
            if not np.any(event_mask):
                vertex_event_indices.append(len(all_vertices))
                continue

            # Record start index for this event
            vertex_event_indices.append(len(all_vertices))

            # Get indices of tracks in this event
            event_track_indices = np.where(event_mask)[0]

            event_x = ovtx_x[event_mask]
            event_y = ovtx_y[event_mask]
            event_z = ovtx_z[event_mask]
            event_t = ovtx_t[event_mask]
            event_from_pv = from_pv[event_mask]

            # Find unique vertices with tolerance
            unique_verts, inverse = _find_unique_vertices_with_tolerance(event_x, event_y, event_z, event_t)

            # Count tracks per vertex and check if all tracks are from PV
            vertex_counts = np.bincount(inverse)
            vertex_is_pv = np.zeros(len(unique_verts), dtype=bool)

            for vtx_idx in range(len(unique_verts)):
                tracks_in_vertex = inverse == vtx_idx
                # Check if all tracks belonging to this vertex have fromPV==1
                n_from_pv = np.sum(event_from_pv[tracks_in_vertex] == 1)
                n_total = np.sum(tracks_in_vertex)
                vertex_is_pv[vtx_idx] = n_from_pv == n_total

            # Create mapping from old vertex index to new LOCAL vertex index (after filtering)
            old_to_new_vtx = {}
            local_vertex_idx = 0  # Local index within this event

            # Filter vertices with at least 4 tracks
            for old_idx, (vtx, count, is_pv) in enumerate(zip(unique_verts, vertex_counts, vertex_is_pv, strict=True)):
                if count >= 4:
                    old_to_new_vtx[old_idx] = local_vertex_idx
                    all_vertices.append(vtx)
                    all_vertex_n_tracks.append(count)
                    all_vertex_is_pv.append(is_pv)
                    local_vertex_idx += 1

            # Remap inverse indices for tracks belonging to valid vertices (using LOCAL indices)
            for local_track_idx, inv_idx in enumerate(inverse):
                if inv_idx in old_to_new_vtx:
                    global_track_idx = event_track_indices[local_track_idx]
                    all_inverse[global_track_idx] = old_to_new_vtx[inv_idx]

        # Append final index
        vertex_event_indices.append(len(all_vertices))
        vertex_event_indices = np.array(vertex_event_indices)

        # Create vertices group
        vertices_group = hf.create_group("vertices")
        if len(all_vertices) > 0:
            all_vertices = np.array(all_vertices)
            vertices_group.create_dataset("ovtx_x", data=all_vertices[:, 0], compression="lzf")
            vertices_group.create_dataset("ovtx_y", data=all_vertices[:, 1], compression="lzf")
            vertices_group.create_dataset("ovtx_z", data=all_vertices[:, 2], compression="lzf")
            vertices_group.create_dataset("ovtx_t", data=all_vertices[:, 3], compression="lzf")
            vertices_group.create_dataset("n_tracks", data=np.array(all_vertex_n_tracks), compression="lzf")
            vertices_group.create_dataset("is_pv", data=np.array(all_vertex_is_pv), compression="lzf")
            vertices_group.create_dataset("vertex_event_indices", data=vertex_event_indices, compression="lzf")
            tracks_group.create_dataset("map_vertex", data=all_inverse, compression="lzf")

        # Metadata
        hf.attrs["num_tracks"] = len(event_ids)
        hf.attrs["num_events"] = len(unique_events)
        hf.attrs["num_vertices"] = max(0, len(all_vertices))
        hf.attrs["dataset_type"] = dataset_type
        hf.attrs["branch_names"] = branches

    print(f"  Saved {len(event_ids)} tracks, {len(unique_events)} events, {max(0, len(all_vertices))} vertices")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Convert LHCb ROOT files to HDF5 format")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Input directory containing ROOT files")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory for HDF5 files")
    parser.add_argument("--train_split", type=float, default=0.8, help="Fraction of events for training (default: 0.8)")

    args = parser.parse_args()

    convert_root_to_hdf5(args.input_dir, args.output_dir, args.train_split)
