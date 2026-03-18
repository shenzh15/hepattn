from hepattn.callbacks.prediction_writer import PredictionWriter


class ECALPredictionWriter(PredictionWriter):
    def setup(self, trainer, module, stage):
        super().setup(trainer=trainer, pl_module=module, stage=stage)
        if stage != "test":
            return

        self.var_transform = self.dataset.scaler.transforms
        self.cluster_fields = tuple(self.dataset.targets.get("cluster", []))

    def inverse_transform_named_value(self, name, value):
        if not name.startswith("cluster_"):
            return value

        field = name.removeprefix("cluster_")
        if field not in self.var_transform:
            return value

        return self.var_transform[field].inverse_transform(value)

    def inverse_transform_regression_value(self, item_name, name, value):
        if item_name == "outputs" and name == "cluster_regr":
            transformed = value.clone()
            for idx, field in enumerate(self.cluster_fields):
                if field in self.var_transform:
                    transformed[..., idx] = self.var_transform[field].inverse_transform(transformed[..., idx])
            return transformed

        return self.inverse_transform_named_value(name, value)

    def write_items(self, sample_group, item_name, items, idx):
        items_group = sample_group.create_group(item_name)
        for name, value in items.items():
            value = value[idx][None, ...]
            if item_name == "targets":
                value = self.inverse_transform_named_value(name, value)
            self.create_dataset(items_group, name, value)

    def write_layer_task_items(self, sample_group, item_name, items, idx):
        items_group = sample_group.create_group(item_name)
        for layer_name, layer_items in items.items():
            if layer_name not in self.write_layers:
                continue
            layer_group = items_group.create_group(layer_name)
            for task_name, task_items in layer_items.items():
                task_group = layer_group.create_group(task_name)
                for name, value in task_items.items():
                    value = value[idx][None, ...]
                    if task_name == "cluster_regression":
                        value = self.inverse_transform_regression_value(item_name, name, value)
                    self.create_dataset(task_group, name, value)
