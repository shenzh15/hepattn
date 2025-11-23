import torch
from lightning import Callback, LightningModule, Trainer


class GradientLoggerCallback(Callback):
    def __init__(self, log_every_n_steps=50, log_per_loss_grads=False):
        """Callback to log model gradients during training.

        Args:
            log_every_n_steps (int): Frequency of logging gradients. Logs every `n` steps.
            log_per_loss_grads (bool): If True, compute and log gradient norms for each task separately.
                                       This helps diagnose which tasks contribute most to gradient magnitude.
                                       Uses torch.autograd.grad() which has minimal overhead.
        """
        self.log_every_n_steps = log_every_n_steps
        self.log_per_loss_grads = log_per_loss_grads

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        if trainer.fast_dev_run or stage != "fit":
            return
        kwargs = {"sync_dist": len(trainer.device_ids) > 1}

        def log(metrics, stage):
            for t, loss_value in metrics.items():
                n = f"{stage}_{t}"
                module.log(n, loss_value, **kwargs)

        self.log = log

    def on_after_backward(self, trainer, pl_module):
        """Called after the backward pass in training.
        Logs the gradients of the model's parameters.
        """
        # Check if logging should happen at this step
        if trainer.global_step % self.log_every_n_steps == 0:
            total_grad_magnitude = 0.0
            total_params = 0
            grad_norm_squared = 0.0  # For computing global L2 norm

            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    grad_magnitude = param.grad.norm().item()
                    total_grad_magnitude += grad_magnitude
                    total_params += param.grad.numel()
                    # Accumulate squared norms for global norm (this is what Lightning clips!)
                    grad_norm_squared += grad_magnitude ** 2
                    # Log gradient statistics
                    # grad_mean = param.grad.mean().item()
                    # grad_std = param.grad.std().item()
                    # self.log({f"grad/{name}_mean": grad_mean, f"grad/{name}_std": grad_std}, "gradients")

                # else:
                #     self.log({f"grad/{name}_mean": None, f"grad/{name}_std": None}, "gradients")

            # Log gradient norms
            if total_params > 0:
                avg_grad_magnitude = total_grad_magnitude / total_params
                # Compute global gradient norm (L2 norm) - THIS is what gets clipped by Lightning!
                global_norm = grad_norm_squared ** 0.5
                self.log(
                    {
                        "total_magnitude": total_grad_magnitude,  # L1-like sum (legacy, not clipped)
                        "average_magnitude": avg_grad_magnitude,  # Average (legacy, not clipped)
                        "global_norm": global_norm,  # IMPORTANT: This is compared to gradient_clip_val!
                    },
                    "gradient",
                )

    def _compute_single_loss_grad_norm(self, pl_module, loss_value):
        """Compute gradient norm for a single loss component using torch.autograd.grad.

        This method uses torch.autograd.grad() instead of backward() to avoid
        interfering with the training process. torch.autograd.grad() is purely
        functional and does not modify the .grad attributes.

        Args:
            pl_module: Lightning module
            loss_value: Single loss tensor

        Returns:
            float: L2 gradient norm
        """
        # Get all parameters that require gradients
        params = [p for p in pl_module.parameters() if p.requires_grad]

        # Compute gradients using torch.autograd.grad (no side effects)
        grads = torch.autograd.grad(
            loss_value,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )

        # Compute L2 gradient norm
        grad_norm_squared = 0.0
        for grad in grads:
            if grad is not None:
                param_norm = grad.data.norm(2)
                grad_norm_squared += param_norm.item() ** 2

        return grad_norm_squared ** 0.5

    def compute_per_loss_gradients(self, trainer, pl_module, losses):
        """Compute and log gradient norms for each task (aggregated across all layers).

        This method aggregates losses by task across all layers, then computes gradient norms
        using torch.autograd.grad() which does not modify parameter gradients and thus
        does not interfere with the training process.

        For example, all vertex_classification losses from all layers are summed together.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            losses: Nested dict {layer: {task: {loss_name: loss_value}}}
        """
        # Only log periodically to avoid overhead
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        kwargs = {"sync_dist": len(trainer.device_ids) > 1, "on_step": True, "on_epoch": False}

        # Aggregate losses by task across all layers (for per-task gradients)
        task_aggregated_losses = {}
        # Also aggregate by individual loss component (for per-loss gradients)
        loss_component_aggregated = {}

        for layer_losses in losses.values():
            for task_name, task_losses in layer_losses.items():
                # Aggregate by task
                if task_name not in task_aggregated_losses:
                    task_aggregated_losses[task_name] = []
                layer_task_loss = sum(loss_value for loss_value in task_losses.values() if loss_value.requires_grad)
                if layer_task_loss != 0:
                    task_aggregated_losses[task_name].append(layer_task_loss)

                # Aggregate by individual loss component
                for loss_name, loss_value in task_losses.items():
                    if not loss_value.requires_grad:
                        continue
                    component_key = f"{task_name}/{loss_name}"
                    if component_key not in loss_component_aggregated:
                        loss_component_aggregated[component_key] = []
                    loss_component_aggregated[component_key].append(loss_value)

        # 1. Compute and log gradient norm for each aggregated task
        for task_name, task_loss_list in task_aggregated_losses.items():
            if not task_loss_list:
                continue
            total_task_loss = sum(task_loss_list)
            try:
                grad_norm = self._compute_single_loss_grad_norm(pl_module, total_task_loss)
                pl_module.log(f"per_task_grad_norm/{task_name}", grad_norm, **kwargs)
            except RuntimeError as e:
                print(f"Warning: Could not compute gradient for task {task_name}: {e}")

        # 2. Compute and log gradient norm for each individual loss component
        for component_key, loss_list in loss_component_aggregated.items():
            if not loss_list:
                continue
            total_component_loss = sum(loss_list)
            try:
                grad_norm = self._compute_single_loss_grad_norm(pl_module, total_component_loss)
                pl_module.log(f"per_loss_grad_norm/{component_key}", grad_norm, **kwargs)
                # Also log the weighted loss value for reference
                pl_module.log(f"weighted_loss_value/{component_key}", total_component_loss.item(), **kwargs)
            except RuntimeError as e:
                print(f"Warning: Could not compute gradient for component {component_key}: {e}")
