"""Test LHCb model metrics."""

import pytest
import torch

from hepattn.experiments.lhcb.model import LHCbModel
from hepattn.models.maskformer import MaskFormer


class TestLHCbModel:
    @pytest.fixture
    def mock_model(self):
        """Create a mock LHCbModel for testing."""
        # Create a minimal mock model
        model = torch.nn.Linear(10, 10)

        lhcb_model = LHCbModel(
            name="test_model",
            model=model,
            lrs_config={"initial": 1e-3, "max": 1e-2, "end": 1e-4, "pct_start": 0.1, "weight_decay": 0.0},
            optimizer="AdamW",
            mtl=False,
        )

        return lhcb_model

    def test_track_overlap_matching_perfect_match(self, mock_model):
        """Test track overlap matching with perfect match scenario."""
        batch_size = 2
        n_pred = 5
        n_true = 5
        n_tracks = 10

        # Create identical predictions and targets (perfect match)
        pred_tracks_mask = torch.zeros(batch_size, n_pred, n_tracks, dtype=torch.bool)
        true_tracks_mask = torch.zeros(batch_size, n_true, n_tracks, dtype=torch.bool)

        # First batch: vertex 0 has tracks 0,1,2; vertex 1 has tracks 3,4
        pred_tracks_mask[0, 0, [0, 1, 2]] = True
        pred_tracks_mask[0, 1, [3, 4]] = True

        true_tracks_mask[0, 0, [0, 1, 2]] = True
        true_tracks_mask[0, 1, [3, 4]] = True

        # Second batch: vertex 0 has tracks 0,1,2,3,4,5
        pred_tracks_mask[1, 0, [0, 1, 2, 3, 4, 5]] = True
        true_tracks_mask[1, 0, [0, 1, 2, 3, 4, 5]] = True

        vertex_valid = torch.zeros(batch_size, n_true, dtype=torch.bool)
        vertex_valid[0, [0, 1]] = True  # First batch has 2 valid vertices
        vertex_valid[1, 0] = True  # Second batch has 1 valid vertex

        matched = mock_model.compute_track_overlap_matching(
            pred_tracks_mask=pred_tracks_mask,
            true_tracks_mask=true_tracks_mask,
            vertex_valid=vertex_valid,
            recall_threshold=0.7,
            precision_threshold=0.7,
        )

        # All valid vertices should be matched with perfect predictions
        assert matched[0, 0] == True, "Vertex 0 in batch 0 should be matched"
        assert matched[0, 1] == True, "Vertex 1 in batch 0 should be matched"
        assert matched[1, 0] == True, "Vertex 0 in batch 1 should be matched"

        # Invalid vertices should not be matched
        assert matched[0, 2:].sum() == 0, "Invalid vertices should not be matched"
        assert matched[1, 1:].sum() == 0, "Invalid vertices should not be matched"

    def test_track_overlap_matching_partial_overlap(self, mock_model):
        """Test track overlap matching with partial overlap."""
        batch_size = 1
        n_pred = 3
        n_true = 3
        n_tracks = 10

        pred_tracks_mask = torch.zeros(batch_size, n_pred, n_tracks, dtype=torch.bool)
        true_tracks_mask = torch.zeros(batch_size, n_true, n_tracks, dtype=torch.bool)

        # True vertex has tracks [0,1,2,3,4] (5 tracks)
        true_tracks_mask[0, 0, [0, 1, 2, 3, 4]] = True

        # Pred vertex 0: tracks [0,1,2] (3/5 = 60% recall, 3/3 = 100% precision) -> NO MATCH
        pred_tracks_mask[0, 0, [0, 1, 2]] = True

        # Pred vertex 1: tracks [0,1,2,3] (4/5 = 80% recall, 4/4 = 100% precision) -> MATCH
        pred_tracks_mask[0, 1, [0, 1, 2, 3]] = True

        # Pred vertex 2: tracks [0,1,2,3,4,5,6] (5/7 = 71% recall, 5/7 = 71% precision) -> MATCH
        pred_tracks_mask[0, 2, [0, 1, 2, 3, 4, 5, 6]] = True

        vertex_valid = torch.ones(batch_size, n_true, dtype=torch.bool)

        matched = mock_model.compute_track_overlap_matching(
            pred_tracks_mask=pred_tracks_mask,
            true_tracks_mask=true_tracks_mask,
            vertex_valid=vertex_valid,
            recall_threshold=0.7,
            precision_threshold=0.7,
        )

        # Vertex 0 should be matched (either by pred 1 or pred 2)
        assert matched[0, 0] == True, "Vertex 0 should be matched with 80% or 71% overlap"

    def test_track_overlap_matching_no_match(self, mock_model):
        """Test track overlap matching when no match meets thresholds."""
        batch_size = 1
        n_pred = 2
        n_true = 2
        n_tracks = 10

        pred_tracks_mask = torch.zeros(batch_size, n_pred, n_tracks, dtype=torch.bool)
        true_tracks_mask = torch.zeros(batch_size, n_true, n_tracks, dtype=torch.bool)

        # True vertex 0: tracks [0,1,2,3,4,5,6,7,8,9] (10 tracks)
        true_tracks_mask[0, 0, :] = True

        # Pred vertex 0: tracks [0,1,2] (3/10 = 30% recall, 3/3 = 100% precision) -> NO MATCH
        pred_tracks_mask[0, 0, [0, 1, 2]] = True

        # Pred vertex 1: tracks [0,1,2,3,4,5] (6/10 = 60% recall, 6/6 = 100% precision) -> NO MATCH
        pred_tracks_mask[0, 1, [0, 1, 2, 3, 4, 5]] = True

        vertex_valid = torch.ones(batch_size, n_true, dtype=torch.bool)

        matched = mock_model.compute_track_overlap_matching(
            pred_tracks_mask=pred_tracks_mask,
            true_tracks_mask=true_tracks_mask,
            vertex_valid=vertex_valid,
            recall_threshold=0.7,
            precision_threshold=0.7,
        )

        # Vertex 0 should NOT be matched (recall too low)
        assert matched[0, 0] == False, "Vertex 0 should not be matched with <70% recall"

    def test_track_overlap_matching_empty_vertices(self, mock_model):
        """Test track overlap matching with empty vertices (no tracks)."""
        batch_size = 1
        n_pred = 2
        n_true = 2
        n_tracks = 10

        pred_tracks_mask = torch.zeros(batch_size, n_pred, n_tracks, dtype=torch.bool)
        true_tracks_mask = torch.zeros(batch_size, n_true, n_tracks, dtype=torch.bool)

        # True vertex 0: no tracks (empty)
        # True vertex 1: tracks [0,1,2]
        true_tracks_mask[0, 1, [0, 1, 2]] = True

        # Pred vertex 0: no tracks (empty)
        # Pred vertex 1: tracks [0,1,2]
        pred_tracks_mask[0, 1, [0, 1, 2]] = True

        vertex_valid = torch.ones(batch_size, n_true, dtype=torch.bool)

        matched = mock_model.compute_track_overlap_matching(
            pred_tracks_mask=pred_tracks_mask,
            true_tracks_mask=true_tracks_mask,
            vertex_valid=vertex_valid,
            recall_threshold=0.7,
            precision_threshold=0.7,
        )

        # Vertex 0 (empty) should not be matched
        assert matched[0, 0] == False, "Empty vertex should not be matched"

        # Vertex 1 should be matched (perfect match)
        assert matched[0, 1] == True, "Vertex 1 should be matched"

    def test_efficiency_metrics_initialization(self, mock_model):
        """Test that efficiency metrics are properly initialized."""
        assert hasattr(mock_model, "pv_efficiency")
        assert hasattr(mock_model, "sv_efficiency")

        # Metrics should be MeanMetric instances
        import torchmetrics as tm

        assert isinstance(mock_model.pv_efficiency, tm.MeanMetric)
        assert isinstance(mock_model.sv_efficiency, tm.MeanMetric)
