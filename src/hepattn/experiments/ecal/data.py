from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from hepattn.utils.lrsm_dataset import LRSMDataset
from hepattn.utils.scaling import FeatureScaler

torch.multiprocessing.set_sharing_strategy("file_system")


class ECALDataset(LRSMDataset):
    def __init__(
        self,
        dirpath: str,
        num_samples: int,
        inputs: dict[str, list[str]],
        targets: dict[str, list[str]],
        scale_dict_path: str,
        input_dtype: str = "float32",
        target_dtype: str = "float32",
        input_pad_value: float = 0.0,
        target_pad_value: float = 0.0,
        force_pad_sizes: dict[str, int] | None = None,
        skip_pad_items: list[str] | None = None,
        sampling_seed: int = 42,
        sample_reject_warn_limit: int = 10,
        min_cluster_energy: float = 0.0,
        min_num_cells: int = 1,
        min_num_clusters: int = 1,
    ):
        super().__init__(
            dirpath,
            num_samples,
            inputs,
            targets,
            input_dtype,
            target_dtype,
            input_pad_value,
            target_pad_value,
            force_pad_sizes,
            skip_pad_items,
            sampling_seed,
            sample_reject_warn_limit,
        )

        scale_path = Path(scale_dict_path)
        if not scale_path.is_absolute():
            scale_path = Path(__file__).resolve().parent / scale_path
        self.scaler = FeatureScaler(str(scale_path))

        self.min_cluster_energy = min_cluster_energy
        self.min_num_cells = min_num_cells
        self.min_num_clusters = min_num_clusters

        event_filenames = sorted(Path(self.dirpath).rglob("reco_ecal_*.npz"))
        num_available_events = len(event_filenames)
        num_requested_events = num_available_events if num_samples == -1 else num_samples
        self.num_samples = min(num_available_events, num_requested_events)

        print(f"Found {num_available_events} available events, {num_requested_events} requested, {self.num_samples} used")

        self.event_filenames = event_filenames[: self.num_samples]

        def filename_to_sample_id(filename):
            parts = filename.stem.replace("reco_ecal_", "").split("_")
            file_idx = int(parts[0])
            event_idx = int(parts[1])
            return file_idx * 100000 + event_idx

        self.sample_ids = [filename_to_sample_id(f) for f in self.event_filenames]
        self.sample_ids_to_filenames = {
            self.sample_ids[i]: str(self.event_filenames[i]) for i in range(len(self.sample_ids))
        }

    def load_sample(self, sample_id: int) -> dict[str, np.ndarray] | None:
        """Loads a single ECAL event from a preprocessed NPZ file."""
        filename = self.sample_ids_to_filenames[sample_id]

        try:
            with np.load(filename, allow_pickle=True) as archive:
                event = {key: archive[key] for key in archive.files}
        except (EOFError, Exception) as e:
            print(f"Error loading sample {sample_id}: {e}")
            return None

        num_cells = len(event["cell.x"])
        num_clusters = len(event["cluster.e"])

        if num_cells < self.min_num_cells:
            return None

        if self.min_cluster_energy > 0:
            valid_mask = event["cluster.e"] >= self.min_cluster_energy
            if valid_mask.sum() < self.min_num_clusters:
                return None

            event["cluster_valid"] = event["cluster_valid"] & valid_mask
            for key in list(event.keys()):
                if key.startswith("cluster."):
                    event[key] = event[key][valid_mask]
            event["cluster_cell_valid"] = event["cluster_cell_valid"][valid_mask]
            num_clusters = valid_mask.sum()

        if num_clusters < self.min_num_clusters:
            return None

        event["cell.x_norm"] = event["cell.x"] / 3000.0
        event["cell.y_norm"] = event["cell.y"] / 3000.0
        event["cell.z_norm"] = event["cell.z"] / 12620.0
        event["cell.dx_norm"] = event["cell.dx"] / 120.0
        event["cell.dy_norm"] = event["cell.dy"] / 120.0
        event["cell.region_norm"] = event["cell.region"] / 7.0

        event["cell_valid"] = np.ones(num_cells, dtype=bool)

        out = {}

        for field in self.inputs.get("cell", []):
            out[f"cell_{field}"] = event[f"cell.{field}"]
        out["cell_valid"] = event["cell_valid"]

        for field in self.targets.get("cluster", []):
            val = torch.as_tensor(event[f"cluster.{field}"], dtype=torch.float32)
            val = self.scaler.transforms[field].transform(val).cpu().numpy()
            out[f"cluster_{field}"] = val
        out["cluster_valid"] = event["cluster_valid"]

        for field in self.targets.get("cluster_cell", []):
            if field == "valid":
                continue

        out["cluster_cell_valid"] = event["cluster_cell_valid"]
        out["sample_id"] = sample_id

        return out


class ECALDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        batch_size: int = 1,
        test_dir: str | None = None,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dset = ECALDataset(dirpath=self.train_dir, num_samples=self.num_train, **self.kwargs)
            self.val_dset = ECALDataset(dirpath=self.val_dir, num_samples=self.num_val, **self.kwargs)
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = ECALDataset(dirpath=self.test_dir, num_samples=self.num_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def get_dataloader(self, dataset: ECALDataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate_fn,
            sampler=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dset)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dset)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dset)
