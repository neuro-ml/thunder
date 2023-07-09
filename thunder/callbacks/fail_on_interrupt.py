from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class FailOnInterrupt(Callback):
    """Forces RuntimeError in order for trainer to stop if KeyboardInterrupt was raised"""

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        if isinstance(exception, KeyboardInterrupt):
            raise RuntimeError("Finished run on KeyboardInterrupt") from exception
