from typing import Callable, Union, List, Tuple, Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from more_itertools import collapse, zip_equal, padded
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .utils import to_np


class ThunderModule(LightningModule):
    def __init__(
            self,
            architecture: nn.Module,
            criterion: Callable,
            n_targets: int = 1,
            activation: Callable = nn.Identity(),
            optimizer: Union[List[Optimizer], Optimizer] = None,
            lr_scheduler: Union[List[LRScheduler], LRScheduler] = None,
    ):
        super().__init__()
        self.architecture = architecture
        self.criterion = criterion
        self.n_targets = n_targets
        self.activation = activation
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def transfer_batch_to_device(self, batch: Tuple, device: torch.device, dataloader_idx: int) -> Any:
        if self.trainer.state.stage != "train":
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.architecture(*args, **kwargs)

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> STEP_OUTPUT:
        x, y = batch[: -self.n_targets], batch[-self.n_targets:]
        return self.criterion(self(*x), *y)

    def validation_step(self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        return self.predict(batch, batch_idx, dataloader_idx)

    def test_step(self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        return self.predict(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.predict(batch, batch_idx, dataloader_idx)

    # TODO: rework predict logic
    def predict(self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        x, y = batch[: -self.n_targets], batch[-self.n_targets:]
        x = super().transfer_batch_to_device(x, self.device, dataloader_idx)
        return to_np(self.inference_step(*x), y)

    def inference_step(self, *xs: torch.Tensor) -> Any:
        return self.activation(self(*xs))

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        if not self.optimizer and not self.lr_scheduler:
            raise NotImplementedError(
                "You must specify optimizer or lr_scheduler, "
                "or implement configure_optimizers method"
            )

        _optimizers = list(collapse([self.optimizer]))
        _lr_schedulers = list(collapse([self.lr_scheduler]))
        max_len = max(map(len, (_optimizers, _lr_schedulers)))
        _optimizers = list(padded(_optimizers, None, max_len))
        _lr_schedulers = list(padded(_lr_schedulers, None, max_len))

        optimizers = []
        lr_schedulers = []

        for optimizer, lr_scheduler in zip_equal(_optimizers, _lr_schedulers):
            if callable(lr_scheduler):
                if optimizer is None:
                    raise ValueError(f"The scheduler demands an Optimizer, but received None")
                lr_scheduler = lr_scheduler(optimizer)

            optimizers.append(optimizer if lr_scheduler is None else lr_scheduler.optimizer)
            if lr_scheduler is not None:
                lr_schedulers.append(lr_scheduler)

        if len(optimizers) < len(lr_schedulers):
            raise ValueError(
                "The number of optimizers must be greater or equal to the number of "
                f"lr_schedulers, got {len(optimizers)} and {len(lr_schedulers)}\n"
                f"Optimizers: f{optimizers}\n"
                f"Schedulers: f{lr_schedulers}\n"
            )

        return optimizers, lr_schedulers
