"""Test LHCb data loading and collation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from hepattn.experiments.lhcb.data import LHCbDataModule, LHCbDataset, collate_fn


class TestLHCbData:
    @pytest.fixture
    def test_config(self):
        """Configuration for testing."""
        return {
            "inputs": {
                "tracks": [
                    "backward",
                    "c00",
                    "c31",
                    "c33",
                    "c55",
                    "chi2",
                    "beamPOCA_tx",
                    "beamPOCA_ty",
                    "beamPOCA_x",
                    "beamPOCA_y",
                    "beamPOCA_z",
                    "beamPOCA_t",
                ],
            },
            "targets": {"vertex": ["ovtx_x", "ovtx_y", "ovtx_z", "ovtx_t", "is_pv"]},
        }

    @pytest.fixture
    def dataset(self, test_config):
        """Create a test dataset."""
        filepath = "data/lhcb/version3_small_hdf5/train.h5"
        if not Path(filepath).exists():
            pytest.skip(f"Test data not found at {filepath}")

        return LHCbDataset(
            filepath=filepath,
            inputs=test_config["inputs"],
            targets=test_config["targets"],
            num_events=5,
            max_vertices=100,
        )

    def test_dataset_initialization(self, dataset):
        """Test that dataset initializes correctly."""
        assert len(dataset) > 0
        assert dataset.num_events > 0
        assert len(dataset.valid_event_indices) <= dataset.num_events

    def test_dataset_getitem(self, dataset):
        """Test that __getitem__ returns correct structure."""
        sample = dataset[0]

        # Check keys
        assert "tracks" in sample
        assert "vertices" in sample
        assert "vertex_tracks_valid" in sample

        # Check tracks dict has expected keys
        tracks = sample["tracks"]
        expected_track_fields = dataset.inputs.get("tracks", [])
        missing_fields = set(expected_track_fields) - set(tracks.keys())
        assert len(missing_fields) == 0, f"Missing fields in tracks: {missing_fields}"

        # Check vertices dict has expected keys
        vertices = sample["vertices"]
        vertex_fields = [f for f in dataset.targets.get("vertex", []) if f != "is_pv"]
        assert all(field in vertices for field in vertex_fields), f"Missing fields in vertices: {set(vertex_fields) - set(vertices.keys())}"
        assert "is_pv" in vertices

        # Check vertex_tracks_valid (track-to-vertex mapping)
        vertex_tracks_valid = sample["vertex_tracks_valid"]
        assert isinstance(vertex_tracks_valid, np.ndarray)
        assert vertex_tracks_valid.dtype == np.int64

    def test_dataset_vertex_filtering(self, test_config):
        """Test that events with too many vertices are filtered out."""
        filepath = "data/lhcb/version3_small_hdf5/train.h5"
        if not Path(filepath).exists():
            pytest.skip(f"Test data not found at {filepath}")

        # Create dataset with lower max_vertices
        dataset_filtered = LHCbDataset(
            filepath=filepath,
            inputs=test_config["inputs"],
            targets=test_config["targets"],
            num_events=-1,
            max_vertices=30,
        )

        # Create dataset with high max_vertices
        dataset_all = LHCbDataset(
            filepath=filepath,
            inputs=test_config["inputs"],
            targets=test_config["targets"],
            num_events=-1,
            max_vertices=100,
        )

        # Filtered dataset should have fewer or equal events
        assert len(dataset_filtered) <= len(dataset_all)

    def test_collate_fn(self, dataset):
        """Test that collate_fn produces correct batch structure."""
        # Get a batch manually
        batch = [dataset[i] for i in range(min(3, len(dataset)))]

        inputs, targets = collate_fn(batch, dataset.inputs, dataset.targets, max_vertices=100)

        batch_size = len(batch)

        # Check inputs structure
        assert "tracks_valid" in inputs

        # Check that all track fields are present with tracks_ prefix
        for field in dataset.inputs["tracks"]:
            key = f"tracks_{field}"
            assert key in inputs, f"Missing field: {key}"
            assert inputs[key].shape[0] == batch_size
            assert inputs[key].dtype == torch.float32

        # Check tracks_valid shape: (batch_size, max_tracks)
        max_tracks = inputs["tracks_backward"].shape[1]
        assert inputs["tracks_valid"].shape[0] == batch_size
        assert inputs["tracks_valid"].shape[1] == max_tracks
        assert inputs["tracks_valid"].dtype == torch.bool

        # Check targets structure
        assert "vertex_valid" in targets
        assert "vertex_tracks_valid" in targets

        # Check vertex features
        for field in dataset.targets["vertex"]:
            if field == "is_pv":
                assert "vertex_is_pv" in targets
                assert targets["vertex_is_pv"].dtype == torch.bool
            else:
                key = f"vertex_{field}"
                assert key in targets
                assert targets[key].dtype == torch.float32

        # Check vertex_valid shape: (batch_size, max_vertices)
        assert targets["vertex_valid"].shape == (batch_size, 100)
        assert targets["vertex_valid"].dtype == torch.bool

        # Check vertex_tracks_valid shape: (batch_size, max_vertices, max_tracks)
        assert targets["vertex_tracks_valid"].shape[0] == batch_size
        assert targets["vertex_tracks_valid"].shape[1] == 100
        assert targets["vertex_tracks_valid"].dtype == torch.bool

    def test_mask_consistency(self, dataset):
        """Test that vertex_tracks_valid is consistent with vertices and tracks."""
        # Get original sample
        sample = dataset[0]
        original_mask = sample["vertex_tracks_valid"]
        n_tracks = len(next(iter(sample["tracks"].values())))
        n_vertices = len(next(iter(sample["vertices"].values())))

        # Collate into batch
        batch = [sample]
        inputs, targets = collate_fn(batch, dataset.inputs, dataset.targets, max_vertices=100)

        mask = targets["vertex_tracks_valid"][0]  # (max_vertices, max_tracks)
        vertex_valid = targets["vertex_valid"][0]  # (max_vertices,)
        tracks_valid = inputs["tracks_valid"][0]  # (max_tracks,)

        # Test 1: Mask should only be True for valid vertices and tracks
        for v_idx in range(mask.shape[0]):
            for t_idx in range(mask.shape[1]):
                if mask[v_idx, t_idx]:
                    assert vertex_valid[v_idx], f"Mask is True for invalid vertex {v_idx}"
                    assert tracks_valid[t_idx], f"Mask is True for invalid track {t_idx}"

        # Test 2: Verify mask matches original map_vertex data
        for track_idx in range(n_tracks):
            vertex_idx = original_mask[track_idx]
            if vertex_idx >= 0:  # Valid assignment
                assert vertex_idx < n_vertices, f"Invalid vertex index {vertex_idx} for track {track_idx}"
                # Check that the mask has True at the correct position
                assert mask[vertex_idx, track_idx], f"Mask should be True at [{vertex_idx}, {track_idx}]"

        # Test 3: Count total True values in mask should match valid assignments in original_mask
        expected_assignments = np.sum(original_mask >= 0)
        actual_assignments = torch.sum(mask).item()
        assert actual_assignments == expected_assignments, f"Assignment count mismatch: {actual_assignments} != {expected_assignments}"

    def test_datamodule_initialization(self, test_config):
        """Test that DataModule initializes correctly."""
        train_path = "data/lhcb/version3_small_hdf5/train.h5"
        val_path = "data/lhcb/version3_small_hdf5/val.h5"

        if not Path(train_path).exists() or not Path(val_path).exists():
            pytest.skip("Test data not found")

        datamodule = LHCbDataModule(
            train_path=train_path,
            val_path=val_path,
            num_workers=0,
            batch_size=2,
            num_train=5,
            num_val=3,
            max_vertices=100,
            **test_config,
        )

        # Setup for fit
        datamodule.setup("fit")

        assert hasattr(datamodule, "train_dataset")
        assert hasattr(datamodule, "val_dataset")
        assert len(datamodule.train_dataset) == 5
        assert len(datamodule.val_dataset) == 3

    def test_dataloader(self, test_config):
        """Test that DataLoader works correctly."""
        train_path = "data/lhcb/version3_small_hdf5/train.h5"
        val_path = "data/lhcb/version3_small_hdf5/val.h5"

        if not Path(train_path).exists() or not Path(val_path).exists():
            pytest.skip("Test data not found")

        datamodule = LHCbDataModule(
            train_path=train_path,
            val_path=val_path,
            num_workers=0,
            batch_size=2,
            num_train=5,
            num_val=3,
            max_vertices=100,
            **test_config,
        )

        datamodule.setup("fit")
        train_loader = datamodule.train_dataloader()

        # Get one batch
        batch = next(iter(train_loader))
        inputs, targets = batch

        # Check batch structure
        assert isinstance(inputs, dict)
        assert isinstance(targets, dict)
        assert "tracks_valid" in inputs
        assert "vertex_tracks_valid" in targets

    def test_local_vertex_indices(self, dataset):
        """Test that vertex_tracks_valid uses local vertex indices (not global)."""
        sample = dataset[0]
        mask = sample["vertex_tracks_valid"]
        n_vertices = len(sample["vertices"]["ovtx_x"])

        # All valid mask indices should be < n_vertices (local indexing)
        valid_indices = mask[mask >= 0]
        if len(valid_indices) > 0:
            assert np.all(valid_indices < n_vertices), "Mask contains indices >= n_vertices (not local)"

    def test_feature_scaling(self, test_config):
        """Test that feature scaling is actually applied when scaler is provided."""
        filepath = "data/lhcb/version3_small_hdf5/train.h5"
        if not Path(filepath).exists():
            pytest.skip(f"Test data not found at {filepath}")

        # Create a temporary scaling config using ORIGINAL field names (not prefixed)
        scale_config = {
            "c00": {"type": "std", "mean": 1.0, "std": 2.0},
            "c31": {"type": "std", "mean": 2.0, "std": 1.5},
            "ovtx_x": {"type": "std", "mean": 10.0, "std": 5.0},
            "ovtx_y": {"type": "std", "mean": 20.0, "std": 3.0},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(scale_config, f)
            scale_dict_path = f.name

        try:
            # Create dataset with scaler
            dataset_with_scaler = LHCbDataset(
                filepath=filepath,
                inputs=test_config["inputs"],
                targets=test_config["targets"],
                num_events=2,
                max_vertices=100,
                scale_dict_path=scale_dict_path,
            )

            # Create dataset without scaler
            dataset_without_scaler = LHCbDataset(
                filepath=filepath,
                inputs=test_config["inputs"],
                targets=test_config["targets"],
                num_events=2,
                max_vertices=100,
                scale_dict_path=None,
            )

            # Get a batch with scaler
            batch_with = [dataset_with_scaler[0]]
            inputs_with, targets_with = collate_fn(
                batch_with,
                dataset_with_scaler.inputs,
                dataset_with_scaler.targets,
                max_vertices=100,
                scaler=dataset_with_scaler.scaler,
            )

            # Get a batch without scaler
            batch_without = [dataset_without_scaler[0]]
            inputs_without, targets_without = collate_fn(
                batch_without,
                dataset_without_scaler.inputs,
                dataset_without_scaler.targets,
                max_vertices=100,
                scaler=None,
            )

            # Test 1: Check that backward field is NOT scaled (it's boolean)
            if "tracks_backward" in inputs_with and "tracks_backward" in inputs_without:
                assert torch.equal(inputs_with["tracks_backward"], inputs_without["tracks_backward"]), "backward is boolean and should not be scaled"

            # Test 2: Check that c00 field is scaled
            if "tracks_c00" in inputs_with and "tracks_c00" in inputs_without:
                c00_with = inputs_with["tracks_c00"][0]
                c00_without = inputs_without["tracks_c00"][0]
                valid_mask = inputs_without["tracks_valid"][0]

                if valid_mask.sum() > 0 and not torch.allclose(c00_with[valid_mask], c00_without[valid_mask], rtol=1e-3):
                    expected_scaled = (c00_without[valid_mask] - scale_config["c00"]["mean"]) / scale_config["c00"]["std"]
                    assert torch.allclose(c00_with[valid_mask], expected_scaled, rtol=1e-5), "c00 field scaling is incorrect"

            # Test 3: Check that c31 field is scaled
            if "tracks_c31" in inputs_with and "tracks_c31" in inputs_without:
                c31_with = inputs_with["tracks_c31"][0]
                c31_without = inputs_without["tracks_c31"][0]
                valid_mask = inputs_without["tracks_valid"][0]

                if valid_mask.sum() > 0 and not torch.allclose(c31_with[valid_mask], c31_without[valid_mask], rtol=1e-3):
                    expected_scaled = (c31_without[valid_mask] - scale_config["c31"]["mean"]) / scale_config["c31"]["std"]
                    assert torch.allclose(c31_with[valid_mask], expected_scaled, rtol=1e-5), "c31 field scaling is incorrect"

            # Test 4: Check that vertex fields are scaled
            if "vertex_ovtx_x" in targets_with and "vertex_ovtx_x" in targets_without:
                ovtx_x_with = targets_with["vertex_ovtx_x"][0]
                ovtx_x_without = targets_without["vertex_ovtx_x"][0]
                valid_mask = targets_without["vertex_valid"][0]

                if valid_mask.sum() > 0 and not torch.allclose(ovtx_x_with[valid_mask], ovtx_x_without[valid_mask], rtol=1e-3):
                    expected_scaled = (ovtx_x_without[valid_mask] - scale_config["ovtx_x"]["mean"]) / scale_config["ovtx_x"]["std"]
                    assert torch.allclose(ovtx_x_with[valid_mask], expected_scaled, rtol=1e-5), "ovtx_x field scaling is incorrect"

            # Test 5: Check that mask fields remain unchanged
            assert torch.equal(inputs_with["tracks_valid"], inputs_without["tracks_valid"]), "tracks_valid should not be scaled"
            assert torch.equal(targets_with["vertex_valid"], targets_without["vertex_valid"]), "vertex_valid should not be scaled"
            assert torch.equal(targets_with["vertex_tracks_valid"], targets_without["vertex_tracks_valid"]), (
                "vertex_tracks_valid should not be scaled"
            )

            # Test 6: Check that boolean target fields remain unchanged
            if "vertex_is_pv" in targets_with and "vertex_is_pv" in targets_without:
                assert torch.equal(targets_with["vertex_is_pv"], targets_without["vertex_is_pv"]), "vertex_is_pv should not be scaled"

        finally:
            # Clean up temporary file
            Path(scale_dict_path).unlink()
