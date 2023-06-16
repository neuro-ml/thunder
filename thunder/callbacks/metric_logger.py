from collections import defaultdict
from functools import partial
from inspect import isfunction
from itertools import chain
from typing import Any, Dict, Callable, Optional, List

import numpy as np
import torch
from lightning import LightningModule, Trainer, Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from toolz import valmap, compose

from ..torch.utils import to_np
from ..utils import squeeze_first


class MetricLogger(Callback):
    def __init__(
        self,
        single_metrics: Dict = None,
        group_metrics: Dict[str, Callable] = None,
        aggregate_fn=None,
    ):
        _single_metrics = dict(single_metrics or {})
        group_metrics = dict(group_metrics or {})

        single_metrics = {}
        preprocess: Dict[Callable, List[str]] = defaultdict(list)

        # collect metrics
        for k, v in _single_metrics.items():
            if isinstance(k, str):
                single_metrics.update({k: v})
            elif callable(k) or isinstance(k, tuple) and all(map(callable, k)):
                if isinstance(k, tuple):
                    k = compose(*k)
                if isinstance(v, (list, tuple)):
                    metrics = {_get_func_name(f): f for f in v}
                elif isinstance(v, dict):
                    metrics = v
                else:
                    raise TypeError(
                        f"When passing metrics with preprocessing, metrics should be List[Callable] or Dict[str, Callable], got {type(v)}"
                    )
                preprocess[k] = metrics
                single_metrics.update(metrics)
            else:
                raise TypeError(f"Metric keys should be of type str or Callable, got {type(k)}")

        for name in set(single_metrics) & set(group_metrics):
            single_metrics[f'single/{name}'] = single_metrics.pop(name)
            group_metrics[f'group/{name}'] = group_metrics.pop(name)

        preprocess[_identity] = sorted(set(single_metrics.keys()) - set(chain.from_iterable(preprocess.values())))

        self.single_metrics = single_metrics
        self.group_metrics = group_metrics
        self.preprocess = preprocess
        self._train_losses = []
        self._single_metric_values = {name: [] for name in single_metrics}
        self._all_predictions = []
        self._default_aggregators = {"min": np.min, "max": np.max, "median": np.median, "std": np.std}

        self.aggregate_fn = {"": np.mean}

        if isinstance(aggregate_fn, (str, Callable)):
            aggregate_fn = [aggregate_fn]

        if isinstance(aggregate_fn, (list, tuple)):
            for fn in aggregate_fn:
                if callable(fn):
                    self.aggregate_fn.update({_get_func_name(fn): fn})
                elif isinstance(fn, str):
                    if fn not in self._default_aggregators:
                        raise ValueError(
                            f"Unknown aggregate_fn: {fn}, if passing a str"
                            f", it should be one of {sorted(self._default_aggregators.keys())}"
                        )
                    self.aggregate_fn.update({fn: self._default_aggregators[fn]})
                else:
                    raise TypeError(f"Expected aggregate_fn to be callable or str, got {type(fn)}")

        elif isinstance(aggregate_fn, dict):
            not_callable = {k: v for k, v in filter(lambda it: not callable(it[1]), aggregate_fn.items())}
            if not_callable:
                raise TypeError(f"All aggregators must be callable if you pass a dict, got uncallable {not_callable}")
            self.aggregate_fn.update(aggregate_fn)
        else:
            if aggregate_fn is not None:
                raise ValueError(f"Unknown type of aggrefate_fn: {type(aggregate_fn)}")

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
            for preprocess, metrics_names in self.preprocess.items():
                _pred, _target = preprocess(pred, target)
                for name in metrics_names:
                    self._single_metric_values[name].append(self.single_metrics[name](_pred, _target))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        group_metric_values = {}
        if self.group_metrics and self._all_predictions:
            predictions, targets = zip(*self._all_predictions)
            group_metric_values = {name: metric(predictions, targets) for name, metric in self.group_metrics.items()}

        single_metric_values = {}
        for fn_name, fn in self.aggregate_fn.items():
            prefix = f"{fn_name}/" if fn_name else ""
            single_metric_values.update({f"{prefix}{k}": fn(v) for k, v in self._single_metric_values.items()})

        self._single_metric_values = {name: [] for name in self.single_metrics}
        self._all_predictions = []

        for k, value in chain(single_metric_values.items(), group_metric_values.items()):
            pl_module.log(f'val/{k}', value)


def _get_func_name(function: Callable) -> str:
    if isinstance(function, partial):
        function = function.func

    if isfunction(function):
        return function.__name__
    elif isinstance(function, Callable):
        return function.__class__.__name__

    raise ValueError(f"You must pass a callable object, got f{type(function)}")


def _identity(*args):
    return squeeze_first(args)
