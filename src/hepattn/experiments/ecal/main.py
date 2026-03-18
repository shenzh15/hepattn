from lightning.pytorch.cli import ArgsType

from hepattn.experiments.ecal.data import ECALDataModule
from hepattn.experiments.ecal.model import ECALReconstructor
from hepattn.utils.cli import CLI


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=ECALReconstructor,
        datamodule_class=ECALDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
