import comet_ml  # noqa: F401  # Must be imported before torch for automatic logging
import torch
from lightning.pytorch.cli import ArgsType

from hepattn.experiments.lhcb.data import LHCbDataModule

# from hepattn.experiments.lhcb.dummy_model import DummyModel
from hepattn.experiments.lhcb.model import LHCbModel
from hepattn.utils.cli import CLI

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=LHCbModel,
        datamodule_class=LHCbDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
