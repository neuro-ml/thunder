from typing import Any, Callable, List, Tuple, Union

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from more_itertools import collapse, padded, zip_equal
from toolz import identity
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from ..predict import BasePredictor, Predictor
from ..utils import squeeze_first
from .utils import maybe_from_np, to_np


class ThunderModule(LightningModule):
    def __init__(
            self,
            architecture: nn.Module,
            criterion: Callable,
            n_targets: int = 1,
            activation: Callable = identity,
            optimizer: Union[List[Optimizer], Optimizer] = None,
            lr_scheduler: Union[List[LRScheduler], LRScheduler] = None,
            predictor: BasePredictor = None,
            n_val_targets: int = None
    ):
        """
        Parameters
        ----------
        architecture: nn.Module
            Model architecture used to conduct forward pass.
        criterion: Callable
            Criterion to optimize.
        n_targets: int
            Number of target values in train and inference batches, if negative, then ...
        activation: Callable
            Final activation function for inference, identity by default.
        optimizer: Union[List[Optimizer], Optimizer]
            Optimizers.
        lr_scheduler: Union[List[LRScheduler], LRScheduler]
            Learning Rate policies.
        predictor: BasePredictor.
            Predictor for inference.
        n_val_targets: int
            Number of target values for inference, if set to None assumes value of `n_targets`.
        """
        super().__init__()
        self.architecture = architecture
        self.criterion = criterion
        self.n_targets = n_targets
        self.n_val_targets = n_targets if n_val_targets is None else n_val_targets
        self.activation = activation
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.predictor = predictor if predictor else Predictor()

    def transfer_batch_to_device(self, batch: Tuple, device: torch.device, dataloader_idx: int) -> Any:
        if self.trainer.state.stage != "train":
            return batch
        return super().transfer_batch_to_device(maybe_from_np(batch, device=device), device, dataloader_idx)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.architecture(*args, **kwargs)

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> STEP_OUTPUT:
        x, y = batch[: -self.n_targets], batch[-self.n_targets:]
        return self.criterion(self(*x), *y)

    def validation_step(self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        return self.inference_step(batch, batch_idx, dataloader_idx)

    def test_step(self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        return self.inference_step(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.inference_step(batch, batch_idx, dataloader_idx)

    def predict(self, x) -> STEP_OUTPUT:
        # TODO: do we need super(). ...?, also consider changing maybe_to_np to smth stricter
        x = maybe_from_np(x, device=self.device)
        if not isinstance(x, (list, tuple)):
            x = (x,)
        return to_np(self.activation(self(*x)))

    def inference_step(self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = map(squeeze_first, (batch[:-self.n_val_targets], batch[-self.n_val_targets:]))
        return self.predictor([x], self.predict)[0], y

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        if not self.optimizer and not self.lr_scheduler:
            raise NotImplementedError(
                "You must specify optimizer or lr_scheduler, "
                "or implement configure_optimizers method"
            )

        _optimizers = list(collapse([self.optimizer]))
        _lr_schedulers = list(collapse([self.lr_scheduler]))
        max_len = max(len(_optimizers), len(_lr_schedulers))
        _optimizers = list(padded(_optimizers, None, max_len))
        _lr_schedulers = list(padded(_lr_schedulers, None, max_len))

        optimizers = []
        lr_schedulers = []

        for optimizer, lr_scheduler in zip_equal(_optimizers, _lr_schedulers):
            if callable(lr_scheduler):
                if optimizer is None:
                    raise ValueError("The scheduler demands an Optimizer, but received None")
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
