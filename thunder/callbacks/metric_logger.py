from collections import defaultdict
from itertools import chain
from typing import Any, Dict, Callable, Optional

import numpy as np
import torch
from lightning import LightningModule, Trainer, Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from toolz import valmap

from ..torch.utils import to_np


class MetricLogger(Callback):
    def __init__(self, single_metrics: Dict[str, Callable] = None, group_metrics: Dict[str, Callable] = None):
        single_metrics = dict(single_metrics or {})
        group_metrics = dict(group_metrics or {})
        for name in set(single_metrics) & set(group_metrics):
            single_metrics[f'single/{name}'] = single_metrics.pop(name)
            group_metrics[f'group/{name}'] = group_metrics.pop(name)

        self.single_metrics = single_metrics
        self.group_metrics = group_metrics
        self._train_losses = []
        self._single_metric_values = {name: [] for name in single_metrics}
        self._all_predictions = []

    # loss

    def on_train_batch_end(
            self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if outputs is None:
            return

        if isinstance(outputs, torch.Tensor):
            outputs = {'loss': outputs}
        if isinstance(outputs, dict):
            outputs = valmap(to_np, outputs)
        elif isinstance(outputs, (list, tuple)):
            outputs = dict(zip(map(str, range(len(outputs))), map(to_np, outputs)))
        else:
            raise TypeError(f"Unknown type of outputs: {type(outputs[0])}")

        self._train_losses.append(outputs)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        group = defaultdict(list)
        for loss in self._train_losses:
            for k, v in loss.items():
                group[k].append(v)

        names = set(map(len, group.values()))
        if len(names) != 1:
            raise ValueError('Loss names are inconsistent')

        for k, vs in group.items():
            pl_module.log(f'train/{k}', np.mean(vs))

        self._train_losses = []

    # val

    def on_validation_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if self.group_metrics:
            self._all_predictions.extend(zip(*outputs))

        for pred, target in zip(*outputs):
            for name, metric in self.single_metrics.items():
                self._single_metric_values[name].append(metric(pred, target))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        group_metric_values = {}
        if self.group_metrics and self._all_predictions:
            predictions, targets = map(np.asarray, zip(*self._all_predictions))
            group_metric_values = {name: metric(predictions, targets) for name, metric in self.group_metrics.items()}

        single_metric_values = valmap(np.mean, self._single_metric_values)

        self._single_metric_values = {name: [] for name in self.single_metrics}
        self._all_predictions = []

        for k, value in chain(single_metric_values.items(), group_metric_values.items()):
            pl_module.log(f'val/{k}', value)
