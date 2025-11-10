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

        # ============ PV/SV/Null Classification Metrics ============
        # Class definition: 0=PV (Primary Vertex), 1=SV (Secondary Vertex), 2=Null
        num_classes = 3

        # Overall classification metrics
        self.vertex_acc_micro = tm.classification.MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.vertex_acc_macro = tm.classification.MulticlassAccuracy(num_classes=num_classes, average="macro")

        # Per-class detailed metrics
        self.per_class_precision = tm.classification.MulticlassPrecision(num_classes=num_classes, average=None)
        self.per_class_recall = tm.classification.MulticlassRecall(num_classes=num_classes, average=None)

        # Efficiency metrics for PV and SV (based on track overlap matching)
        self.pv_efficiency = tm.MeanMetric()
        self.sv_efficiency = tm.MeanMetric()

    def compute_track_overlap_matching(
        self,
        pred_tracks_mask: torch.Tensor,
        true_tracks_mask: torch.Tensor,
        vertex_valid: torch.Tensor,
        recall_threshold: float = 0.7,
        precision_threshold: float = 0.7,
    ) -> torch.Tensor:
        """Compute matching between predicted and true vertices based on track overlap.

        Uses matrix operations to efficiently compute recall and precision for all pairs.

        Args:
            pred_tracks_mask: Predicted track-vertex assignment [B, N_pred, N_tracks]
            true_tracks_mask: Ground truth track-vertex assignment [B, N_true, N_tracks]
            vertex_valid: Valid vertex mask [B, N_true]
            recall_threshold: Minimum recall to consider a match (default: 0.7)
            precision_threshold: Minimum precision to consider a match (default: 0.7)

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

        # Check if any pred vertex matches each true vertex
        matched = match_quality.any(dim=1)  # [B, N_true]

        # Mask out invalid vertices and return
        return matched & vertex_valid

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
        vertex_valid = targets.get("vertex_valid", None)

        # Filter out padding vertices using the valid mask
        if vertex_valid is not None:
            pred_flat = pred_classes[vertex_valid]
            target_flat = target_classes[vertex_valid]
        else:
            pred_flat = pred_classes.view(-1)
            target_flat = target_classes.view(-1)

        # ======== Overall Classification Metrics ========
        self.vertex_acc_micro(pred_flat, target_flat)
        self.log(f"{stage}/vertex_acc_micro", self.vertex_acc_micro, sync_dist=True)

        self.vertex_acc_macro(pred_flat, target_flat)
        self.log(f"{stage}/vertex_acc_macro", self.vertex_acc_macro, sync_dist=True)

        # ======== Per-class Detailed Metrics ========
        per_class_prec = self.per_class_precision(pred_flat, target_flat)
        per_class_rec = self.per_class_recall(pred_flat, target_flat)

        class_names = ["pv", "sv"]
        for i, class_name in enumerate(class_names):
            self.log(f"{stage}/{class_name}_precision", per_class_prec[i], sync_dist=True)
            self.log(f"{stage}/{class_name}_recall", per_class_rec[i], sync_dist=True)

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

        # Compute matching based on track overlap (recall >= 70% and precision >= 70%)
        matched = self.compute_track_overlap_matching(
            pred_tracks_mask=pred_assignment,
            true_tracks_mask=true_assignment,
            vertex_valid=vertex_valid,
            recall_threshold=0.7,
            precision_threshold=0.7,
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
