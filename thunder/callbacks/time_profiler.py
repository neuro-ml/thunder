from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Literal, Union, overload

from lightning import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from more_itertools import windowed


class TimeProfiler(Callback):
    """
    Parameters
    ----------
    keys : Union[str, bool]
        Optional keys for logging. If set to `True` it will log all keys.
    """
    @overload
    def __init__(self, *keys: str):
        ...

    @overload
    def __init__(self, keys: Literal[True]):
        ...

    def __init__(self, *keys: Union[str, bool]):
        self._default_keys = (
            "train batch",
            "validation batch",
            "train epoch",
            "validation epoch",
            "avg train downtime",
            "avg val downtime",
        )
        self._optional_keys = (
            "backward",
            "optimizer step",
            "total train downtime",
            "total val downtime",
        )

        if len(keys) != 0 and keys[0] is True:
            keys = sorted(self._optional_keys)

        _keys = sorted(set(keys).intersection(self._optional_keys))
        if _keys != sorted(keys):
            raise ValueError(f"TimeProfiler got unknown keys: {set(keys) - set(_keys)}")

        self.keys = sorted(set(keys).union(self._default_keys))
        self.time_stamps: Dict[str, List[datetime]] = defaultdict(list)

    def log_time(self, key: str) -> None:
        self.time_stamps[key].append(datetime.now())

    def compute_time_delta(self) -> Dict[str, float]:
        def delta(t1, t2=None):
            if isinstance(t1, (list, tuple)):
                return (t1[1] - t1[0]).total_seconds()
            return (t2 - t1).total_seconds()

        deltas = {}
        for key, time_stamps in self.time_stamps.items():
            if len(time_stamps) % 2 == 1:
                continue
            deltas[key] = list(map(delta, windowed(time_stamps, 2, step=2)))

            if key == "train batch":
                n_train_batches = len(deltas[key])
            elif key == "validation batch":
                n_val_batches = len(deltas[key])

            deltas[key] = sum(deltas[key]) / len(deltas[key])

        if "train epoch" in deltas:
            deltas["train epoch"] -= deltas["validation epoch"]
            deltas["total train downtime"] = deltas["train epoch"] - n_train_batches * deltas["train batch"]

            deltas["total val downtime"] = deltas["validation epoch"] - n_val_batches * deltas["validation batch"]

            deltas["avg train downtime"] = deltas["total train downtime"] / n_train_batches
            deltas["avg val downtime"] = deltas["total val downtime"] / n_val_batches

        return deltas

    def log_to_logger(self, pl_module, on_epoch: bool = False, clear: bool = True):
        deltas = self.compute_time_delta()
        pl_module.log_dict(
            {f"{self.__class__.__name__}/{k}": v for k, v in deltas.items() if k in self.keys},
            prog_bar=False,
            on_step=not on_epoch,
            on_epoch=on_epoch,
        )
        if clear:
            self.time_stamps.clear()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.log_time("train batch")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_time("train batch")
        self.log_time("optimizer step")
        self.log_to_logger(pl_module, False, False)

    def on_train_epoch_start(self, trainer, pl_module):
        self.log_time("train epoch")

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_time("train epoch")
        self.log_to_logger(pl_module, True, True)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.log_time("validation batch")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_time("validation batch")
        self.log_to_logger(pl_module, False, False)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_time("validation epoch")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_time("validation epoch")

    def on_before_backward(self, trainer, pl_module, loss):
        self.log_time("backward")

    def on_after_backward(self, trainer, pl_module):
        self.log_time("backward")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self.log_time("optimizer step")
        self.log_to_logger(pl_module, False, False)

    def setup(self, trainer, pl_module, stage: str):
        if not trainer.loggers:
            raise MisconfigurationException(f"Cannot use {self.__class__.__name__} callback with no logger")
        if stage == "fit":
            self.time_stamps.clear()

    def teardown(self, trainer, pl_module, stage: str):
        if stage == "fit":
            self.time_stamps.clear()
