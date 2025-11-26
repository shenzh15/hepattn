"""LHCb vertex reconstruction model."""

import torch
import torchmetrics as tm
from torch import nn

from hepattn.models.wrapper import ModelWrapper


class LHCbModel(ModelWrapper):
    """Model wrapper for LHCb vertex reconstruction."""

    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)

        # ======== PV/SV Classification Metrics ========
        # Efficiency metrics for PV and SV (based on track overlap matching)
        self.pv_efficiency = tm.MeanMetric()
        self.sv_efficiency = tm.MeanMetric()

        # Strict PV efficiency: requires predicted vertex to be classified as PV
        self.pv_efficiency_strict = tm.MeanMetric()

        # Fake rate metrics for PV reconstruction
        self.pv_fake_rate_strict = tm.MeanMetric()  # Only pred PVs can match true PVs
        self.pv_fake_rate_relaxed = tm.MeanMetric()  # Any pred vertex can match true PVs

        # Split rate metrics for PV reconstruction (true PV matched by >= 2 pred vertices)
        self.pv_split_rate_strict = tm.MeanMetric()  # Only pred PVs can match
        self.pv_split_rate_relaxed = tm.MeanMetric()  # Any pred vertex can match

        # Merge rate metrics for PV reconstruction (true PV claimed by >= 2 pred vertices with high purity)
        self.pv_merge_rate_strict = tm.MeanMetric()  # Only pred PVs can claim
        self.pv_merge_rate_relaxed = tm.MeanMetric()  # Any pred vertex can claim

    def compute_track_overlap_matching(
        self,
        pred_tracks_mask: torch.Tensor,
        true_tracks_mask: torch.Tensor,
        vertex_valid: torch.Tensor,
        recall_threshold: float = 0.7,
        precision_threshold: float = 0.7,
        pred_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute matching between predicted and true vertices based on track overlap.

        Uses matrix operations to efficiently compute recall and precision for all pairs.

        Args:
            pred_tracks_mask: Predicted track-vertex assignment [B, N_pred, N_tracks]
            true_tracks_mask: Ground truth track-vertex assignment [B, N_true, N_tracks]
            vertex_valid: Valid vertex mask [B, N_true]
            recall_threshold: Minimum recall to consider a match (default: 0.7)
            precision_threshold: Minimum precision to consider a match (default: 0.7)
            pred_valid: Optional mask for valid predicted vertices [B, N_pred]. If provided,
                       only pred vertices where this mask is True can participate in matching.

        Returns:
            matched: Boolean tensor [B, N_true] indicating if each true vertex was matched
        """
        # Compute overlap matrix via matrix multiplication: [B, N_pred, N_tracks] @ [B, N_tracks, N_true]
        # Result: [B, N_pred, N_true] - number of shared tracks between each pred-true pair
        overlap = torch.bmm(pred_tracks_mask.float(), true_tracks_mask.transpose(1, 2).float())  # [B, N_pred, N_true]

        # Count tracks per vertex
        n_true_tracks = true_tracks_mask.sum(dim=2)  # [B, N_true]
        n_pred_tracks = pred_tracks_mask.sum(dim=2)  # [B, N_pred]

        # Compute recall matrix: overlap / n_true_tracks
        # Avoid division by zero
        recall = overlap / (n_true_tracks.unsqueeze(1) + 1e-8)  # [B, N_pred, N_true]

        # Compute precision matrix: overlap / n_pred_tracks
        precision = overlap / (n_pred_tracks.unsqueeze(2) + 1e-8)  # [B, N_pred, N_true]

        # For each true vertex, find if there exists any pred vertex with both recall and precision above threshold
        # Create a match quality matrix
        match_quality = (recall >= recall_threshold) & (precision >= precision_threshold)  # [B, N_pred, N_true]

        # Filter out invalid predicted vertices if pred_valid mask is provided
        if pred_valid is not None:
            match_quality = match_quality & pred_valid.unsqueeze(2)  # [B, N_pred, N_true]

        # Check if any pred vertex matches each true vertex
        matched = match_quality.any(dim=1)  # [B, N_true]

        # Mask out invalid vertices and return
        return matched & vertex_valid

    def compute_fake_rate_strict(
        self,
        pred_tracks_mask: torch.Tensor,
        true_tracks_mask: torch.Tensor,
        pred_classes: torch.Tensor,
        true_classes: torch.Tensor,
        vertex_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Compute strict fake rate for reconstructed PVs using pure matrix operations.

        Only predicted vertices classified as PV can match true PVs.

        For each true PV, the reconstructed PV with maximum recall is matched.
        After all true PVs are matched, the remaining unmatched reconstructed PVs are considered fake.
        Fake rate = number of fake PVs / total number of reconstructed PVs

        Args:
            pred_tracks_mask: Predicted track-vertex assignment [B, N_pred, N_tracks]
            true_tracks_mask: Ground truth track-vertex assignment [B, N_true, N_tracks]
            pred_classes: Predicted vertex classes [B, N_pred]
            true_classes: Ground truth vertex classes [B, N_true]
            vertex_valid: Valid vertex mask [B, N_true]

        Returns:
            fake_rate: Fake rate for each batch element [B]
        """
        # Step 1: Identify PV vertices (class 0)
        pred_is_pv = pred_classes == 0  # [B, N_pred]
        true_is_pv = (true_classes == 0) & vertex_valid  # [B, N_true]

        # Step 2: Compute overlap matrix for all vertices using batch matrix multiplication
        # [B, N_pred, N_tracks] @ [B, N_tracks, N_true] -> [B, N_pred, N_true]
        overlap = torch.bmm(pred_tracks_mask.float(), true_tracks_mask.transpose(1, 2).float())

        # Step 3: Compute recall matrix
        n_true_tracks = true_tracks_mask.sum(dim=2)  # [B, N_true]
        recall = overlap / (n_true_tracks.unsqueeze(1) + 1e-8)  # [B, N_pred, N_true]

        # Step 4: Mask out non-PV vertices by setting their recall to -1
        # This ensures they won't be selected when finding maximum recall (recall range is [0,1])
        recall_masked = recall.clone()
        # Mask predicted non-PVs: set entire row to -1
        recall_masked[~pred_is_pv] = -1.0
        # Mask true non-PVs: set entire column to -1
        recall_masked = torch.where(true_is_pv.unsqueeze(1), recall_masked, -1.0)

        # Step 5: For each true PV, find the predicted vertex with maximum recall
        # max_values: [B, N_true], max_indices: [B, N_true]
        max_recall_values, max_recall_pred_idx = recall_masked.max(dim=1)

        # Step 6: Create matched mask for predicted PVs
        # We need to mark which predicted PVs were selected by any true PV
        b, n_pred, n_true = recall_masked.shape

        # Create batch indices for advanced indexing
        batch_idx = torch.arange(b, device=recall.device).unsqueeze(1).expand(b, n_true)  # [B, N_true]

        # Initialize matched mask
        matched_pred_pvs = torch.zeros(b, n_pred, dtype=torch.bool, device=recall.device)

        # Only mark matches where the true vertex is actually a PV and max recall is valid
        valid_matches = true_is_pv & (max_recall_values > -1.0)  # [B, N_true]

        # Use advanced indexing to mark matched predicted vertices
        # For each valid true PV, mark its best matching predicted vertex as matched
        if valid_matches.any():
            matched_pred_pvs[batch_idx[valid_matches], max_recall_pred_idx[valid_matches]] = True

        # Step 7: Compute fake rate per batch element
        # Count predicted PVs
        n_pred_pv = pred_is_pv.sum(dim=1).float()  # [B]

        # Count fake PVs: predicted PVs that were not matched
        n_fake = (pred_is_pv & ~matched_pred_pvs).sum(dim=1).float()  # [B]

        # Compute fake rate, handling edge cases
        return torch.where(n_pred_pv > 0, n_fake / n_pred_pv, torch.zeros_like(n_pred_pv))

    def compute_fake_rate_relaxed(
        self,
        pred_tracks_mask: torch.Tensor,
        true_tracks_mask: torch.Tensor,
        pred_classes: torch.Tensor,
        true_classes: torch.Tensor,
        vertex_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Compute relaxed fake rate for reconstructed vertices using pure matrix operations.

        Both PV and SV predictions can match true PVs (null predictions are excluded).
        This measures how many predicted non-null vertices are fake.

        For each true PV, any predicted PV/SV with maximum recall is matched.
        After all true PVs are matched, we count how many predicted PV/SV were not matched.
        Fake rate = number of unmatched predicted PV/SV / total number of predicted PV/SV

        Args:
            pred_tracks_mask: Predicted track-vertex assignment [B, N_pred, N_tracks]
            true_tracks_mask: Ground truth track-vertex assignment [B, N_true, N_tracks]
            pred_classes: Predicted vertex classes [B, N_pred]
            true_classes: Ground truth vertex classes [B, N_true]
            vertex_valid: Valid vertex mask [B, N_true]

        Returns:
            fake_rate: Fake rate for each batch element [B]
        """
        # Step 1: Identify true PV vertices and valid predicted vertices (exclude null)
        true_is_pv = (true_classes == 0) & vertex_valid  # [B, N_true]
        pred_not_null = pred_classes != 2  # [B, N_pred] - True for PV and SV, False for null

        # Step 2: Compute overlap matrix for all vertices
        overlap = torch.bmm(pred_tracks_mask.float(), true_tracks_mask.transpose(1, 2).float())

        # Step 3: Compute recall matrix
        n_true_tracks = true_tracks_mask.sum(dim=2)  # [B, N_true]
        recall = overlap / (n_true_tracks.unsqueeze(1) + 1e-8)  # [B, N_pred, N_true]

        # Step 4: Mask out true non-PVs and predicted nulls
        recall_masked = recall.clone()
        recall_masked[~pred_not_null] = -1.0  # Exclude null predictions
        recall_masked = torch.where(true_is_pv.unsqueeze(1), recall_masked, -1.0)  # Only match to true PVs

        # Step 5: For each true PV, find the predicted PV/SV with maximum recall
        max_recall_values, max_recall_pred_idx = recall_masked.max(dim=1)

        # Step 6: Create matched mask for predicted PV/SV vertices
        b, n_pred, n_true = recall_masked.shape
        batch_idx = torch.arange(b, device=recall.device).unsqueeze(1).expand(b, n_true)
        matched_pred_vertices = torch.zeros(b, n_pred, dtype=torch.bool, device=recall.device)

        # Mark matched vertices (only PV/SV can be matched)
        valid_matches = true_is_pv & (max_recall_values > -1.0)
        if valid_matches.any():
            matched_pred_vertices[batch_idx[valid_matches], max_recall_pred_idx[valid_matches]] = True

        # Step 7: Compute fake rate among predicted PV/SV vertices (exclude null)
        # Count predicted PV/SV vertices
        n_pred_not_null = pred_not_null.sum(dim=1).float()  # [B]
        # Count PV/SV vertices that were not matched
        n_fake = (pred_not_null & ~matched_pred_vertices).sum(dim=1).float()  # [B]

        return torch.where(n_pred_not_null > 0, n_fake / n_pred_not_null, torch.zeros_like(n_pred_not_null))

    def compute_split_rate_strict(
        self,
        pred_tracks_mask: torch.Tensor,
        true_tracks_mask: torch.Tensor,
        pred_classes: torch.Tensor,
        true_classes: torch.Tensor,
        vertex_valid: torch.Tensor,
        recall_threshold: float = 0.7,
    ) -> torch.Tensor:
        """Compute strict split rate for true PVs using pure matrix operations.

        Only predicted vertices classified as PV can contribute to splitting.
        A true PV is considered "split" if at least 2 predicted PVs have recall >= threshold.

        Split rate = number of split true PVs / total number of true PVs

        Args:
            pred_tracks_mask: Predicted track-vertex assignment [B, N_pred, N_tracks]
            true_tracks_mask: Ground truth track-vertex assignment [B, N_true, N_tracks]
            pred_classes: Predicted vertex classes [B, N_pred]
            true_classes: Ground truth vertex classes [B, N_true]
            vertex_valid: Valid vertex mask [B, N_true]
            recall_threshold: Minimum recall to count as a match (default: 0.7)

        Returns:
            split_rate: Split rate for each batch element [B]
        """
        # Step 1: Identify PV vertices
        pred_is_pv = pred_classes == 0  # [B, N_pred]
        true_is_pv = (true_classes == 0) & vertex_valid  # [B, N_true]

        # Step 2: Compute overlap and recall matrices
        overlap = torch.bmm(pred_tracks_mask.float(), true_tracks_mask.transpose(1, 2).float())
        n_true_tracks = true_tracks_mask.sum(dim=2)  # [B, N_true]
        recall = overlap / (n_true_tracks.unsqueeze(1) + 1e-8)  # [B, N_pred, N_true]

        # Step 3: Filter recall by predicted PV classification
        recall_filtered = recall.clone()
        recall_filtered[~pred_is_pv] = 0.0  # Only pred PVs can contribute

        # Step 4: Find matches with recall >= threshold
        high_recall_matches = (recall_filtered >= recall_threshold).float()  # [B, N_pred, N_true]

        # Step 5: Count how many pred vertices match each true PV
        n_matches_per_true = high_recall_matches.sum(dim=1)  # [B, N_true]

        # Step 6: Identify split true PVs (>= 2 matches)
        is_split = (n_matches_per_true >= 2) & true_is_pv  # [B, N_true]

        # Step 7: Compute split rate
        n_true_pv = true_is_pv.sum(dim=1).float()  # [B]
        n_split = is_split.sum(dim=1).float()  # [B]

        return torch.where(n_true_pv > 0, n_split / n_true_pv, torch.zeros_like(n_true_pv))

    def compute_split_rate_relaxed(
        self,
        pred_tracks_mask: torch.Tensor,
        true_tracks_mask: torch.Tensor,
        pred_classes: torch.Tensor,
        true_classes: torch.Tensor,
        vertex_valid: torch.Tensor,
        recall_threshold: float = 0.7,
    ) -> torch.Tensor:
        """Compute relaxed split rate for true PVs using pure matrix operations.

        Both PV and SV predictions can contribute to splitting (null predictions are excluded).
        A true PV is considered "split" if at least 2 predicted PV/SV have recall >= threshold.

        Split rate = number of split true PVs / total number of true PVs

        Args:
            pred_tracks_mask: Predicted track-vertex assignment [B, N_pred, N_tracks]
            true_tracks_mask: Ground truth track-vertex assignment [B, N_true, N_tracks]
            pred_classes: Predicted vertex classes [B, N_pred]
            true_classes: Ground truth vertex classes [B, N_true]
            vertex_valid: Valid vertex mask [B, N_true]
            recall_threshold: Minimum recall to count as a match (default: 0.7)

        Returns:
            split_rate: Split rate for each batch element [B]
        """
        # Step 1: Identify true PV vertices and valid predicted vertices (exclude null)
        true_is_pv = (true_classes == 0) & vertex_valid  # [B, N_true]
        pred_not_null = pred_classes != 2  # [B, N_pred] - True for PV and SV, False for null

        # Step 2: Compute overlap and recall matrices for all vertices
        overlap = torch.bmm(pred_tracks_mask.float(), true_tracks_mask.transpose(1, 2).float())
        n_true_tracks = true_tracks_mask.sum(dim=2)  # [B, N_true]
        recall = overlap / (n_true_tracks.unsqueeze(1) + 1e-8)  # [B, N_pred, N_true]

        # Step 3: Filter recall by predicted vertex validity (exclude null)
        recall_filtered = recall.clone()
        recall_filtered[~pred_not_null] = 0.0  # Only PV/SV can contribute

        # Step 4: Find matches with recall >= threshold
        high_recall_matches = (recall_filtered >= recall_threshold).float()  # [B, N_pred, N_true]

        # Step 5: Count how many pred PV/SV match each true PV
        n_matches_per_true = high_recall_matches.sum(dim=1)  # [B, N_true]

        # Step 6: Identify split true PVs (>= 2 matches)
        is_split = (n_matches_per_true >= 2) & true_is_pv  # [B, N_true]

        # Step 7: Compute split rate
        n_true_pv = true_is_pv.sum(dim=1).float()  # [B]
        n_split = is_split.sum(dim=1).float()  # [B]

        return torch.where(n_true_pv > 0, n_split / n_true_pv, torch.zeros_like(n_true_pv))

    def compute_merge_rate_strict(
        self,
        pred_tracks_mask: torch.Tensor,
        true_tracks_mask: torch.Tensor,
        pred_classes: torch.Tensor,
        true_classes: torch.Tensor,
        vertex_valid: torch.Tensor,
        purity_threshold: float = 0.7,
    ) -> torch.Tensor:
        """Compute strict merge rate for true PVs using pure matrix operations.

        Only predicted vertices classified as PV can contribute to merging.
        A true PV is considered "merged" if at least 2 predicted PVs have purity >= threshold.

        Purity (precision) = overlap / n_pred_tracks (measures how much of pred vertex is correct)

        Merge rate = number of merged true PVs / total number of true PVs

        Args:
            pred_tracks_mask: Predicted track-vertex assignment [B, N_pred, N_tracks]
            true_tracks_mask: Ground truth track-vertex assignment [B, N_true, N_tracks]
            pred_classes: Predicted vertex classes [B, N_pred]
            true_classes: Ground truth vertex classes [B, N_true]
            vertex_valid: Valid vertex mask [B, N_true]
            purity_threshold: Minimum purity to count as a match (default: 0.7)

        Returns:
            merge_rate: Merge rate for each batch element [B]
        """
        # Step 1: Identify PV vertices
        pred_is_pv = pred_classes == 0  # [B, N_pred]
        true_is_pv = (true_classes == 0) & vertex_valid  # [B, N_true]

        # Step 2: Compute overlap and purity (precision) matrices
        overlap = torch.bmm(pred_tracks_mask.float(), true_tracks_mask.transpose(1, 2).float())
        n_pred_tracks = pred_tracks_mask.sum(dim=2)  # [B, N_pred]
        purity = overlap / (n_pred_tracks.unsqueeze(2) + 1e-8)  # [B, N_pred, N_true]

        # Step 3: Filter purity by predicted PV classification
        purity_filtered = purity.clone()
        purity_filtered[~pred_is_pv] = 0.0  # Only pred PVs can contribute

        # Step 4: Find matches with purity >= threshold
        high_purity_matches = (purity_filtered >= purity_threshold).float()  # [B, N_pred, N_true]

        # Step 5: Count how many pred vertices match each true PV
        n_matches_per_true = high_purity_matches.sum(dim=1)  # [B, N_true]

        # Step 6: Identify merged true PVs (>= 2 matches)
        is_merged = (n_matches_per_true >= 2) & true_is_pv  # [B, N_true]

        # Step 7: Compute merge rate
        n_true_pv = true_is_pv.sum(dim=1).float()  # [B]
        n_merged = is_merged.sum(dim=1).float()  # [B]

        return torch.where(n_true_pv > 0, n_merged / n_true_pv, torch.zeros_like(n_true_pv))

    def compute_merge_rate_relaxed(
        self,
        pred_tracks_mask: torch.Tensor,
        true_tracks_mask: torch.Tensor,
        pred_classes: torch.Tensor,
        true_classes: torch.Tensor,
        vertex_valid: torch.Tensor,
        purity_threshold: float = 0.7,
    ) -> torch.Tensor:
        """Compute relaxed merge rate for true PVs using pure matrix operations.

        Both PV and SV predictions can contribute to merging (null predictions are excluded).
        A true PV is considered "merged" if at least 2 predicted PV/SV have purity >= threshold.

        Purity (precision) = overlap / n_pred_tracks (measures how much of pred vertex is correct)

        Merge rate = number of merged true PVs / total number of true PVs

        Args:
            pred_tracks_mask: Predicted track-vertex assignment [B, N_pred, N_tracks]
            true_tracks_mask: Ground truth track-vertex assignment [B, N_true, N_tracks]
            pred_classes: Predicted vertex classes [B, N_pred]
            true_classes: Ground truth vertex classes [B, N_true]
            vertex_valid: Valid vertex mask [B, N_true]
            purity_threshold: Minimum purity to count as a match (default: 0.7)

        Returns:
            merge_rate: Merge rate for each batch element [B]
        """
        # Step 1: Identify true PV vertices and valid predicted vertices (exclude null)
        true_is_pv = (true_classes == 0) & vertex_valid  # [B, N_true]
        pred_not_null = pred_classes != 2  # [B, N_pred] - True for PV and SV, False for null

        # Step 2: Compute overlap and purity (precision) matrices for all vertices
        overlap = torch.bmm(pred_tracks_mask.float(), true_tracks_mask.transpose(1, 2).float())
        n_pred_tracks = pred_tracks_mask.sum(dim=2)  # [B, N_pred]
        purity = overlap / (n_pred_tracks.unsqueeze(2) + 1e-8)  # [B, N_pred, N_true]

        # Step 3: Filter purity by predicted vertex validity (exclude null)
        purity_filtered = purity.clone()
        purity_filtered[~pred_not_null] = 0.0  # Only PV/SV can contribute

        # Step 4: Find matches with purity >= threshold
        high_purity_matches = (purity_filtered >= purity_threshold).float()  # [B, N_pred, N_true]

        # Step 5: Count how many pred PV/SV match each true PV
        n_matches_per_true = high_purity_matches.sum(dim=1)  # [B, N_true]

        # Step 6: Identify merged true PVs (>= 2 matches)
        is_merged = (n_matches_per_true >= 2) & true_is_pv  # [B, N_true]

        # Step 7: Compute merge rate
        n_true_pv = true_is_pv.sum(dim=1).float()  # [B]
        n_merged = is_merged.sum(dim=1).float()  # [B]

        return torch.where(n_true_pv > 0, n_merged / n_true_pv, torch.zeros_like(n_true_pv))

    def log_custom_metrics(self, preds, targets, stage):
        """Log custom metrics for vertex reconstruction.

        Args:
            preds: Model predictions dictionary
            targets: Ground truth targets dictionary
            stage: Training stage (train/val/test)
        """
        # Get vertex classification predictions and targets
        if "vertex_classification" not in preds["final"]:
            return

        # Extract predictions and targets
        vertex_preds = preds["final"]["vertex_classification"]

        # Get predicted classes (already computed by predict() method)
        if "vertex_class" not in vertex_preds:
            return
        pred_classes = vertex_preds["vertex_class"]  # [B, N]

        # Get target classes
        if "vertex_class" not in targets:
            return
        target_classes = targets["vertex_class"]  # [B, N]

        # Get valid vertex mask to exclude padding
        vertex_valid = targets.get("vertex_valid", None)  # padding mask [B, N_vertices]

        # ======== Track Overlap-based Efficiency Metrics ========
        # Get predicted and ground truth track-vertex assignments
        if "track_assignment" not in preds["final"]:
            return

        # Get predicted track assignment mask [B, N_vertices, N_tracks]
        pred_assignment = preds["final"]["track_assignment"].get("vertex_tracks_valid")
        if pred_assignment is None:
            return

        # Get ground truth track assignment mask [B, N_vertices, N_tracks]
        true_assignment = targets.get("vertex_tracks_valid")
        if true_assignment is None:
            return

        # Create pred_valid mask: exclude null predictions (class 2)
        pred_not_null = pred_classes != 2  # [B, N_pred] - True for PV and SV, False for null

        # Compute matching based on track overlap (recall >= 70% and precision >= 70%)
        # Only predicted PV/SV vertices can match (null predictions are excluded)
        matched = self.compute_track_overlap_matching(
            pred_tracks_mask=pred_assignment,
            true_tracks_mask=true_assignment,
            vertex_valid=vertex_valid,
            recall_threshold=0.7,
            precision_threshold=0.7,
            pred_valid=pred_not_null,
        )

        # Split efficiency by PV and SV using target classes
        # PV = class 0, SV = class 1
        is_pv = target_classes == 0  # [B, N_vertices]
        is_sv = target_classes == 1  # [B, N_vertices]

        # Mask with vertex_valid to only consider valid vertices
        pv_mask = is_pv & vertex_valid
        sv_mask = is_sv & vertex_valid

        # Compute efficiency for PV
        if pv_mask.any():
            pv_matched = matched[pv_mask]
            pv_eff = pv_matched.float().mean()
            self.pv_efficiency(pv_eff)
            self.log(f"{stage}/pv_efficiency", self.pv_efficiency, sync_dist=True)

        # Compute efficiency for SV
        if sv_mask.any():
            sv_matched = matched[sv_mask]
            sv_eff = sv_matched.float().mean()
            self.sv_efficiency(sv_eff)
            self.log(f"{stage}/sv_efficiency", self.sv_efficiency, sync_dist=True)

        # ======== Strict PV Efficiency (pred must be classified as PV) ========
        # Recompute match quality, but only consider pred vertices classified as PV
        overlap = torch.bmm(pred_assignment.float(), true_assignment.transpose(1, 2).float())
        n_true_tracks = true_assignment.sum(dim=2)
        n_pred_tracks = pred_assignment.sum(dim=2)
        recall = overlap / (n_true_tracks.unsqueeze(1) + 1e-8)
        precision = overlap / (n_pred_tracks.unsqueeze(2) + 1e-8)

        # Match quality matrix
        match_quality = (recall >= 0.7) & (precision >= 0.7)  # [B, N_pred, N_true]

        # Only consider matches where predicted vertex is classified as PV
        pred_is_pv_mask = pred_classes == 0  # [B, N_pred]
        match_quality_strict = match_quality & pred_is_pv_mask.unsqueeze(2)  # [B, N_pred, N_true]

        # Check if any pred PV matches each true vertex
        matched_strict = match_quality_strict.any(dim=1) & vertex_valid  # [B, N_true]

        # Compute strict efficiency for PV
        if pv_mask.any():
            pv_matched_strict = matched_strict[pv_mask]
            pv_eff_strict = pv_matched_strict.float().mean()
            self.pv_efficiency_strict(pv_eff_strict)
            self.log(f"{stage}/pv_efficiency_strict", self.pv_efficiency_strict, sync_dist=True)

        # ======== Fake Rate for PV ========
        # Strict: Only pred PVs can match true PVs
        fake_rates_strict = self.compute_fake_rate_strict(
            pred_tracks_mask=pred_assignment,
            true_tracks_mask=true_assignment,
            pred_classes=pred_classes,
            true_classes=target_classes,
            vertex_valid=vertex_valid,
        )
        mean_fake_rate_strict = fake_rates_strict.mean()
        self.pv_fake_rate_strict(mean_fake_rate_strict)
        self.log(f"{stage}/pv_fake_rate_strict", self.pv_fake_rate_strict, sync_dist=True)

        # Relaxed: Any pred vertex can match true PVs (ignores classification)
        fake_rates_relaxed = self.compute_fake_rate_relaxed(
            pred_tracks_mask=pred_assignment,
            true_tracks_mask=true_assignment,
            pred_classes=pred_classes,
            true_classes=target_classes,
            vertex_valid=vertex_valid,
        )
        mean_fake_rate_relaxed = fake_rates_relaxed.mean()
        self.pv_fake_rate_relaxed(mean_fake_rate_relaxed)
        self.log(f"{stage}/pv_fake_rate_relaxed", self.pv_fake_rate_relaxed, sync_dist=True)

        # ======== Split Rate for PV ========
        # Strict: Only pred PVs can contribute to splitting
        split_rates_strict = self.compute_split_rate_strict(
            pred_tracks_mask=pred_assignment,
            true_tracks_mask=true_assignment,
            pred_classes=pred_classes,
            true_classes=target_classes,
            vertex_valid=vertex_valid,
            recall_threshold=0.7,
        )
        mean_split_rate_strict = split_rates_strict.mean()
        self.pv_split_rate_strict(mean_split_rate_strict)
        self.log(f"{stage}/pv_split_rate_strict", self.pv_split_rate_strict, sync_dist=True)

        # Relaxed: Any pred vertex can contribute to splitting (ignores classification)
        split_rates_relaxed = self.compute_split_rate_relaxed(
            pred_tracks_mask=pred_assignment,
            true_tracks_mask=true_assignment,
            pred_classes=pred_classes,
            true_classes=target_classes,
            vertex_valid=vertex_valid,
            recall_threshold=0.7,
        )
        mean_split_rate_relaxed = split_rates_relaxed.mean()
        self.pv_split_rate_relaxed(mean_split_rate_relaxed)
        self.log(f"{stage}/pv_split_rate_relaxed", self.pv_split_rate_relaxed, sync_dist=True)

        # ======== Merge Rate for PV ========
        # Strict: Only pred PVs can contribute to merging
        merge_rates_strict = self.compute_merge_rate_strict(
            pred_tracks_mask=pred_assignment,
            true_tracks_mask=true_assignment,
            pred_classes=pred_classes,
            true_classes=target_classes,
            vertex_valid=vertex_valid,
            purity_threshold=0.7,
        )
        mean_merge_rate_strict = merge_rates_strict.mean()
        self.pv_merge_rate_strict(mean_merge_rate_strict)
        self.log(f"{stage}/pv_merge_rate_strict", self.pv_merge_rate_strict, sync_dist=True)

        # Relaxed: Any pred vertex can contribute to merging (ignores classification)
        merge_rates_relaxed = self.compute_merge_rate_relaxed(
            pred_tracks_mask=pred_assignment,
            true_tracks_mask=true_assignment,
            pred_classes=pred_classes,
            true_classes=target_classes,
            vertex_valid=vertex_valid,
            purity_threshold=0.7,
        )
        mean_merge_rate_relaxed = merge_rates_relaxed.mean()
        self.pv_merge_rate_relaxed(mean_merge_rate_relaxed)
        self.log(f"{stage}/pv_merge_rate_relaxed", self.pv_merge_rate_relaxed, sync_dist=True)
