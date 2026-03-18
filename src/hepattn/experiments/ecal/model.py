import torch
from torch import nn

from hepattn.models.wrapper import ModelWrapper


class ECALReconstructor(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)

    def log_custom_metrics(self, preds, targets, stage):
        # Use predictions from the final decoder layer
        preds = preds["final"]

        if "cluster_cell_assignment" not in preds:
            return

        # Get predicted and true cell-to-cluster assignment masks
        pred_cell_masks = preds["cluster_cell_assignment"]["cluster_cell_valid"]
        true_cell_masks = targets["cluster_cell_valid"]

        pred_valid = preds["cluster_valid"]["cluster_valid"]
        true_valid = targets["cluster_valid"]

        # Mask out cells that are not on a valid cluster slot
        pred_cell_masks = pred_cell_masks & pred_valid.unsqueeze(-1)
        true_cell_masks = true_cell_masks & true_valid.unsqueeze(-1)

        # True/false positive calculation
        cell_tp = (pred_cell_masks & true_cell_masks).sum(-1)
        cell_p = pred_cell_masks.sum(-1)
        cell_t = true_cell_masks.sum(-1)

        for wp in [0.5, 0.75, 1.0]:
            both_valid = true_valid & pred_valid

            # Efficiency: fraction of true clusters that are well-reconstructed
            effs = ((cell_tp / cell_t.clamp(min=1)) >= wp) & both_valid
            # Purity: fraction of predicted clusters that are not fake
            purs = ((cell_tp / cell_p.clamp(min=1)) >= wp) & both_valid

            eff = effs.float().sum(-1) / true_valid.float().sum(-1).clamp(min=1)
            pur = purs.float().sum(-1) / pred_valid.float().sum(-1).clamp(min=1)

            self.log(f"{stage}/p{wp}_cell_eff", eff.mean())
            self.log(f"{stage}/p{wp}_cell_pur", pur.mean())

        # Log counting info
        pred_num = pred_valid.sum(-1)
        true_num = true_valid.sum(-1)
        self.log(f"{stage}/num_pred_clusters", torch.mean(pred_num.float()))
        self.log(f"{stage}/num_true_clusters", torch.mean(true_num.float()))

        # Log mean number of cells per cluster
        num_cells_per_pred = pred_cell_masks.sum(-1).float()[pred_valid].mean() if pred_valid.any() else torch.tensor(0.0)
        num_cells_per_true = true_cell_masks.sum(-1).float()[true_valid].mean() if true_valid.any() else torch.tensor(0.0)
        self.log(f"{stage}/num_cells_per_pred_cluster", num_cells_per_pred)
        self.log(f"{stage}/num_cells_per_true_cluster", num_cells_per_true)

        # Log energy regression metrics if available
        if "cluster_regression" in preds and "cluster_e" in targets:
            pred_e = preds["cluster_regression"].get("cluster_e")
            true_e = targets.get("cluster_e")
            if pred_e is not None and true_e is not None:
                valid = true_valid
                if valid.any():
                    pred_e_valid = pred_e[valid]
                    true_e_valid = true_e[valid]
                    e_residual = (pred_e_valid - true_e_valid) / true_e_valid.clamp(min=0.001)
                    self.log(f"{stage}/energy_residual_mean", e_residual.mean())
                    self.log(f"{stage}/energy_residual_std", e_residual.std())
