from pathlib import Path

import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, Dataset

from hepattn.utils.scaling import FeatureScaler


class LHCbDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        max_vertices: int = 100,
        scale_dict_path: str | None = None,
    ):
        super().__init__()

        self.filepath = Path(filepath)
        self.inputs = inputs
        self.targets = targets
        self.max_vertices = max_vertices
        self._file = None  # Lazy-loaded file handle per worker

        # Initialize feature scaler if path provided
        self.scaler = FeatureScaler(scale_dict_path) if scale_dict_path else None

        # Load metadata from HDF5 file
        f = self._get_file()
        num_events_available = f.attrs["num_events"]
        self.total_tracks = f.attrs["num_tracks"]
        self.total_vertices = f.attrs["num_vertices"]

        # Build valid event indices (filter out events with too many vertices)
        vertex_event_indices = f["vertices/vertex_event_indices"][:]
        vertex_counts = np.diff(vertex_event_indices)
        valid_event_mask = vertex_counts <= max_vertices
        self.valid_event_indices = np.where(valid_event_mask)[0]

        if num_events > num_events_available:
            msg = f"Requested {num_events} events, but only {num_events_available} are available in {filepath}."
            raise ValueError(msg)

        num_events = len(self.valid_event_indices) if num_events < 0 else min(num_events, len(self.valid_event_indices))

        if num_events == 0:
            raise ValueError("num_events must be greater than 0")

        self.num_events = num_events
        self.valid_event_indices = self.valid_event_indices[:num_events]

    def __len__(self):
        return int(self.num_events)

    def _get_file(self):
        """Lazy open file handle per worker process."""
        if self._file is None:
            self._file = h5py.File(self.filepath, "r")
        return self._file

    def __getitem__(self, idx):
        # Map to valid event index
        event_idx = self.valid_event_indices[idx]

        # Load the event
        tracks, vertices, vertex_tracks_valid = self.load_event(event_idx)

        # Return raw data for collate function to handle
        return {
            "tracks": tracks,
            "vertices": vertices,
            "vertex_tracks_valid": vertex_tracks_valid,
        }

    def load_event(self, idx):
        """Load a single event from HDF5 file."""
        f = self._get_file()

        # Get event boundaries - only load the slices we need
        start_track = f["tracks/event_indices"][idx]
        end_track = f["tracks/event_indices"][idx + 1]

        # Load all track features
        tracks = {}
        if "tracks" in self.inputs:
            for feature in self.inputs["tracks"]:
                tracks[feature] = f[f"tracks/{feature}"][start_track:end_track]

        # Load track-to-vertex mapping
        vertex_tracks_valid = f["tracks/map_vertex"][start_track:end_track]

        # Get vertex boundaries - only load the slices we need
        start_vertex = f["vertices/vertex_event_indices"][idx]
        end_vertex = f["vertices/vertex_event_indices"][idx + 1]

        # Load vertices
        vertices = {}
        if "vertex" in self.targets:
            for feature in self.targets["vertex"]:
                vertices[feature] = f[f"vertices/{feature}"][start_vertex:end_vertex]

        # Load vertex metadata
        vertices["is_pv"] = f["vertices/is_pv"][start_vertex:end_vertex]

        return tracks, vertices, vertex_tracks_valid


def _get_dict_length(d):
    """Safely get length of first value in dict, or 0 if dict is empty."""
    return len(next(iter(d.values()))) if d else 0


def collate_fn(batch, inputs_config, targets_config, max_vertices, scaler=None):
    """Collate function with dynamic padding for tracks and fixed padding for vertices."""
    # Get max number of tracks in this batch
    max_tracks = max(_get_dict_length(sample["tracks"]) for sample in batch)

    batch_size = len(batch)
    inputs = {}
    targets = {}

    # Prepare track features - keep each field separate (InputNet will concatenate)
    if "tracks" in inputs_config:
        # Get track lengths
        tracks_lengths = np.array([_get_dict_length(sample["tracks"]) for sample in batch])

        # Output each field separately as "tracks_field"
        for field in inputs_config["tracks"]:
            padded = np.zeros((batch_size, max_tracks), dtype=np.float32)
            for i, sample in enumerate(batch):
                track_data = sample["tracks"].get(field, np.array([]))
                padded[i, : len(track_data)] = track_data

            # Convert to tensor
            tensor = torch.from_numpy(padded)

            # Apply scaling if available (using original field name)
            if scaler is not None and field in scaler.transforms:
                tensor = scaler.transforms[field].transform(tensor)

            # Output as "tracks_{field}"
            inputs[f"tracks_{field}"] = tensor

        # Create tracks valid mask
        tracks_valid = np.zeros((batch_size, max_tracks), dtype=bool)
        for i, n_tracks in enumerate(tracks_lengths):
            tracks_valid[i, :n_tracks] = True
        inputs["tracks_valid"] = torch.from_numpy(tracks_valid)
        # Also add to targets for cost/loss computation
        targets["tracks_valid"] = torch.from_numpy(tracks_valid)

    # Prepare vertex tensors with fixed padding to max_vertices
    if "vertex" in targets_config:
        # Pre-compute vertex lengths
        vertices_lengths = np.array([_get_dict_length(sample["vertices"]) for sample in batch])

        for field in targets_config["vertex"]:
            # Pre-allocate the full padded array
            padded = np.zeros((batch_size, max_vertices), dtype=bool if field == "is_pv" else np.float32)

            for i, sample in enumerate(batch):
                vertex_data = sample["vertices"].get(field, np.array([]))
                n = len(vertex_data)
                padded[i, :n] = vertex_data

            # Convert to tensor
            tensor = torch.from_numpy(padded)

            # Apply scaling if available (using original field name, skip boolean fields)
            if field != "is_pv" and scaler is not None and field in scaler.transforms:
                tensor = scaler.transforms[field].transform(tensor)

            if field == "is_pv":
                # Store both the boolean version and class version
                # targets["vertex_is_pv"] = tensor
                # Convert to class labels for ObjectClassificationTask:
                # PV (is_pv=True) -> class 0
                # SV (is_pv=False) -> class 1
                # Invalid vertices will be marked as null class (2) by the model
                targets["vertex_class"] = (~tensor).long()  # True->0, False->1
            else:
                targets[f"vertex_{field}"] = tensor

        # Create vertex valid mask using vectorized operations
        vertex_valid = np.zeros((batch_size, max_vertices), dtype=bool)
        for i, n_vertices in enumerate(vertices_lengths):
            vertex_valid[i, :n_vertices] = True
        targets["vertex_valid"] = torch.from_numpy(vertex_valid)

    # Build track-to-vertex assignment mask (batch_size, max_vertices, max_tracks)
    assignment_mask = np.zeros((batch_size, max_vertices, max_tracks), dtype=bool)
    for batch_idx, sample in enumerate(batch):
        vertex_tracks_valid = sample["vertex_tracks_valid"]
        n_vertices = _get_dict_length(sample["vertices"])

        # Filter valid vertex indices
        valid_mask = (vertex_tracks_valid >= 0) & (vertex_tracks_valid < n_vertices)
        valid_vertex_ids = vertex_tracks_valid[valid_mask]
        valid_track_ids = np.where(valid_mask)[0]

        # Use advanced indexing to set True values
        assignment_mask[batch_idx, valid_vertex_ids, valid_track_ids] = True

    # Provide as vertex_tracks_valid for ObjectHitMaskTask
    targets["vertex_tracks_valid"] = torch.from_numpy(assignment_mask)

    return inputs, targets


class LHCbDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        num_workers: int,
        batch_size: int = 1,
        num_train: int = -1,
        num_val: int = -1,
        num_test: int = -1,
        test_path: str | None = None,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.pin_memory = pin_memory
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage in {"fit", "test"}:
            self.train_dataset = LHCbDataset(
                filepath=self.train_path,
                num_events=self.num_train,
                **self.kwargs,
            )

        if stage == "fit":
            self.val_dataset = LHCbDataset(
                filepath=self.val_path,
                num_events=self.num_val,
                **self.kwargs,
            )

            rank_zero_info(f"Created training dataset with {len(self.train_dataset):,} events")
            rank_zero_info(f"Created validation dataset with {len(self.val_dataset):,} events")

        if stage == "test":
            assert self.test_path is not None, "No test file specified, see --data.test_path"

            self.test_dataset = LHCbDataset(
                filepath=self.test_path,
                num_events=self.num_test,
                **self.kwargs,
            )
            rank_zero_info(f"Created test dataset with {len(self.test_dataset):,} events")

    def get_dataloader(self, dataset: LHCbDataset, shuffle: bool):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=lambda batch: collate_fn(batch, dataset.inputs, dataset.targets, max_vertices=dataset.max_vertices, scaler=dataset.scaler),
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dataset, shuffle=False)
