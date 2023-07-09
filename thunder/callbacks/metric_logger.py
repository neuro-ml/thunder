from collections import defaultdict
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from more_itertools import zip_equal
from toolz import compose, valmap

from ..torch.utils import to_np
from ..utils import collect, squeeze_first


class MetricLogger(Callback):
    def __init__(
        self,
        single_metrics: Dict = None,
        group_metrics: Dict[str, Callable] = None,
        aggregate_fn: Union[Dict[str, Callable], str, Callable, List[Union[str, Callable]]] = None,
    ):
        """
        Parameters
        ----------
        single_metrics: Dict
            Metrics that are calculated on each object separately and then aggregated.
        group_metrics: Dict[str, Callable]
            Metrics that are calculated on entire dataset.
        aggregate_fn: Union[Dict[str, Callable], str, Callable, List[Union[str, Callable]]]
            How to aggregate metrics. By default it computes mean value. If yoy specify something,
            then the callback will compute mean and the specified values.
        """
        _single_metrics = dict(single_metrics or {})
        group_metrics = dict(group_metrics or {})

        single_metrics = {}
        preprocess: Dict[Callable, List[str]] = defaultdict(list)

        # collect metrics
        for k, v in _single_metrics.items():
            if isinstance(k, str):
                single_metrics[k] = v
            elif callable(k) or isinstance(k, tuple) and all(map(callable, k)):
                if isinstance(k, tuple):
                    k = compose(*k)

                if isinstance(v, (list, tuple)):
                    metrics = {_get_func_name(f): f for f in v}
                elif isinstance(v, dict):
                    metrics = v
                elif callable(v):
                    metrics = {_get_func_name(v): v}
                else:
                    raise TypeError(
                        f"When passing metrics with preprocessing, metrics should be "
                        f"Callable, List[Callable] or Dict[str, Callable], got {type(v)}"
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
        self._single_metric_values = defaultdict(lambda: {name: [] for name in single_metrics})
        self._all_predictions = defaultdict(list)
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
            not_callable = dict(filter(lambda it: not callable(it[1]), aggregate_fn.items()))
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
        self.evaluate_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.evaluate_epoch(trainer, pl_module, "val")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.evaluate_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.evaluate_epoch(trainer, pl_module, "test")

    def evaluate_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if len(outputs) != 2:
            raise ValueError(f"Expected step output in form of 2 elements (x, y)," f"but received {len(outputs)}")
        xs, ys = outputs
        xs = _recombine_batch(xs) if isinstance(xs, (list, tuple)) else xs
        ys = _recombine_batch(ys) if isinstance(ys, (list, tuple)) else ys

        outputs = (xs, ys)

        if self.group_metrics:
            self._all_predictions[dataloader_idx].extend(zip(*outputs))

        for pred, target in zip(*outputs):
            for preprocess, metrics_names in self.preprocess.items():
                preprocessed = preprocess(pred, target)
                for name in metrics_names:
                    self._single_metric_values[dataloader_idx][name].append(self.single_metrics[name](*preprocessed))

    def evaluate_epoch(self, trainer: Trainer, pl_module: LightningModule, key: str) -> None:
        group_metric_values = {}
        for name, metric in self.group_metrics.items():
            for dataloader_idx, all_predictions in self._all_predictions.items():
                loader_postfix = f"/{dataloader_idx}" if len(self._all_predictions) > 1 else ""
                predictions, targets = zip(*all_predictions)
                group_metric_values[f"{name}{loader_postfix}"] = metric(predictions, targets)

        single_metric_values = {}
        for fn_name, fn in self.aggregate_fn.items():
            for dataloader_idx, metrics in self._single_metric_values.items():
                prefix = f"{fn_name}/" if fn_name else ""
                loader_postfix = f"/{dataloader_idx}" if len(self._single_metric_values) > 1 else ""
                single_metric_values.update({f"{prefix}{k}{loader_postfix}": fn(v) for k, v in metrics.items()})

        self._single_metric_values.clear()
        self._all_predictions.clear()

        for k, value in chain(single_metric_values.items(), group_metric_values.items()):
            pl_module.log(f'{key}/{k}', value)


def _get_func_name(function: Callable) -> str:
    if isinstance(function, partial):
        function = function.func

    if callable(function):
        if hasattr(function, "__name__"):
            return function.__name__
        return function.__class__.__name__

    raise ValueError(f"You must pass a callable object, got f{type(function)}")


def _identity(*args):
    return squeeze_first(args)


@collect
def _recombine_batch(xs: Sequence) -> List:
    yield from map(squeeze_first, zip_equal(*xs))
