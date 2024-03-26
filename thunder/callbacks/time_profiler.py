from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Literal, Union, overload

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
        self.batch_sizes: Dict[str, List[int]] = defaultdict(list)
        self.deltas = dict()

    def log_time(self, key: str) -> None:
        self.time_stamps[key].append(datetime.now())

    def log_batch_size(self, batch, key: str) -> None:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        self.batch_sizes[key].append(len(batch))

    def compute_time_delta(self) -> Dict[str, float]:
        deltas = {}
        for key, time_stamps in self.time_stamps.items():
            deltas[key] = [(t[1] - t[0]).total_seconds() for t in windowed(time_stamps, 2, step=2,
                                                                           fillvalue=time_stamps[-1])]
            deltas[key] = sum(deltas[key]) / len(deltas[key])

        if "train epoch" in deltas:
            if "validation epoch" in deltas:
                deltas["train epoch"] -= deltas["validation epoch"]

            n_train_batches = len(self.batch_sizes["train batch"])
            deltas["total train downtime"] = deltas["train epoch"] - n_train_batches * deltas["train batch"]
            deltas["avg train downtime"] = deltas["total train downtime"] / sum(self.batch_sizes["train batch"])

            if "validation epoch" in deltas:
                n_val_batches = len(self.batch_sizes["validation batch"])
                deltas["total val downtime"] = deltas["validation epoch"] - n_val_batches * deltas["validation batch"]
                deltas["avg val downtime"] = deltas["total val downtime"] / sum(self.batch_sizes["validation batch"])

        self.deltas = deltas

    def log_to_logger(self, pl_module):
        self.compute_time_delta()
        pl_module.log_dict(
            {f"{self.__class__.__name__}/{k}": v for k, v in self.deltas.items() if k in self.keys},
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.time_stamps.clear()
        self.batch_sizes.clear()
        self.deltas.clear()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.log_time("train batch")
        self.log_batch_size(batch, "train batch")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_time("train batch")
        self.log_time("optimizer step")

    def on_train_epoch_start(self, trainer, pl_module):
        self.log_time("train epoch")

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_time("train epoch")
        self.log_to_logger(pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.log_time("validation batch")
        self.log_batch_size(batch, "validation batch")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_time("validation batch")

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

    def setup(self, trainer, pl_module, stage: str):
        if not trainer.loggers:
            raise MisconfigurationException(f"Cannot use {self.__class__.__name__} callback with no logger")
        if stage == "fit":
            self.time_stamps.clear()
            self.batch_sizes.clear()

    def teardown(self, trainer, pl_module, stage: str):
        if stage == "fit":
            self.time_stamps.clear()
            self.batch_sizes.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {"keys": self.keys,
                "time_stamps": self.time_stamps,
                "batch_sizes": self.batch_sizes,
                "deltas": self.deltas}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.deltas = state_dict["deltas"]
        self.time_stamps = state_dict["time_stamps"]
        self.batch_sizes = state_dict["batch_sizes"]
        self.keys = state_dict["keys"]
        super().load_state_dict(state_dict)
