from lightning import Callback, LightningModule, Trainer


class GradientLoggerCallback(Callback):
    def __init__(self, log_every_n_steps=50, log_per_loss_grads=False):
        """Callback to log model gradients during training.

        Args:
            log_every_n_steps (int): Frequency of logging gradients. Logs every `n` steps.
            log_per_loss_grads (bool): If True, compute and log gradient norms for each loss component separately.
                                       This helps diagnose which losses contribute most to gradient magnitude.
                                       Warning: Adds computational overhead due to multiple backward passes.
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
        """Compute gradient norm for a single loss component.

        Args:
            pl_module: Lightning module
            loss_value: Single loss tensor

        Returns:
            float: L2 gradient norm
        """
        # Zero gradients before computing
        pl_module.zero_grad()

        # Backward pass for this specific loss
        loss_value.backward(retain_graph=True)

        # Compute L2 gradient norm (same as Lightning's global norm)
        grad_norm_squared = 0.0
        for param in pl_module.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                grad_norm_squared += param_norm.item() ** 2

        return grad_norm_squared ** 0.5

    def compute_per_loss_gradients(self, trainer, pl_module, losses):
        """Compute and log gradient norms for each task (aggregated across all layers).

        This method aggregates losses by task across all layers, then computes gradient norms.
        For example, all vertex_classification losses from all layers are summed together.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            losses: Nested dict {layer: {task: {loss_name: loss_value}}}
        """
        # Only log periodically to avoid overhead
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # Store original gradients if they exist
        original_grads = {}
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                original_grads[name] = param.grad.clone()

        kwargs = {"sync_dist": len(trainer.device_ids) > 1, "on_step": True, "on_epoch": False}

        # Aggregate losses by task across all layers
        task_aggregated_losses = {}
        for layer_losses in losses.values():
            for task_name, task_losses in layer_losses.items():
                if task_name not in task_aggregated_losses:
                    task_aggregated_losses[task_name] = []

                # Sum all losses for this task in this layer
                layer_task_loss = sum(loss_value for loss_value in task_losses.values() if loss_value.requires_grad)
                if layer_task_loss != 0:  # Only add non-zero losses
                    task_aggregated_losses[task_name].append(layer_task_loss)

        # Compute gradient norm for each aggregated task
        for task_name, task_loss_list in task_aggregated_losses.items():
            if not task_loss_list:
                continue

            # Sum losses from all layers for this task
            total_task_loss = sum(task_loss_list)

            try:
                grad_norm = self._compute_single_loss_grad_norm(pl_module, total_task_loss)
                pl_module.log(
                    f"per_task_grad_norm/{task_name}",
                    grad_norm,
                    **kwargs
                )
            except RuntimeError as e:
                print(f"Warning: Could not compute gradient for task {task_name}: {e}")

        # Restore original gradients
        pl_module.zero_grad()
        for name, param in pl_module.named_parameters():
            if name in original_grads:
                param.grad = original_grads[name]
