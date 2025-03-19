import warnings
from collections import defaultdict
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from more_itertools import zip_equal
from toolz import compose, keymap, valmap

from ..torch.utils import to_np
from ..utils import squeeze_first


class MetricMonitor(Callback):
    def __init__(
            self,
            single_metrics: Dict = None,
            group_metrics: Dict = None,
            aggregate_fn: Union[Dict[str, Callable], str, Callable, List[Union[str, Callable]]] = None,
            log_individual_metrics: bool = False
    ):
        """
        Parameters
        ----------
        single_metrics: Dict
            Metrics that are calculated on each object separately and then aggregated.
        group_metrics: Dict
            Metrics that are calculated on entire dataset.
        aggregate_fn: Union[Dict[str, Callable], str, Callable, List[Union[str, Callable]]]
            How to aggregate metrics. By default, it computes mean value. If yoy specify something,
            then the callback will compute mean and the specified values.
        log_individual_metrics: bool
            If True, logs table for case-wise metrics (if logger has `log_table` method) and saves table to csv file.
        """
        _single_metrics = dict(single_metrics or {})
        _group_metrics = dict(group_metrics or {})

        # metrics = {"metric_name": func}
        # preprocess = {preprocess_func: ["metric_name"]} + {identity: ["metric_name"]}

        single_metrics, single_preprocess = _process_metrics(_single_metrics)
        group_metrics, group_preprocess = _process_metrics(_group_metrics)

        names_to_replace = []
        for name in set(single_metrics) & set(group_metrics):
            single_metrics[f"single/{name}"] = single_metrics.pop(name)
            group_metrics[f"group/{name}"] = group_metrics.pop(name)
            names_to_replace.append(name)

        single_preprocess = valmap(lambda names:
                                   [f"single/{n}" if n in names_to_replace else n for n in names],
                                   single_preprocess)
        group_preprocess = valmap(lambda names:
                                  [f"group/{n}" if n in names_to_replace else n for n in names],
                                  group_preprocess)

        self.single_metrics = single_metrics
        self.group_metrics = group_metrics
        self.single_preprocess = single_preprocess
        self.group_preprocess = group_preprocess
        self._train_losses = []
        self._single_metric_values = defaultdict(lambda: {name: {} for name in single_metrics})
        self._all_predictions = defaultdict(lambda: {prep: [] for prep in self.group_preprocess})
        self._default_aggregators = {"min": np.min, "max": np.max, "median": np.median, "std": np.std}

        self.aggregate_fn = {"": np.mean}
        self.log_individual_metrics = log_individual_metrics

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
                raise TypeError(f"All aggregators must be callable if you pass a dict, got not callable {not_callable}")
            self.aggregate_fn.update(aggregate_fn)
        else:
            if aggregate_fn is not None:
                raise ValueError(f"Unknown type of aggregate_fn: {type(aggregate_fn)}")

    def on_train_batch_end(
            self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if outputs is None:
            return

        if isinstance(outputs, torch.Tensor):
            outputs = {'loss': to_np(outputs)}
        if isinstance(outputs, dict):
            outputs = valmap(to_np, outputs)
        elif isinstance(outputs, (list, tuple)):
            outputs = dict(zip_equal(map(str, range(len(outputs))), map(to_np, outputs)))
        else:
            raise TypeError(f"Unknown type of outputs: {type(outputs[0])}")

        self._train_losses.append(outputs)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        group = defaultdict(list)
        for loss in self._train_losses:
            for k, v in loss.items():
                group[k].append(v)

        n_entries = set(map(len, group.values()))
        if len(n_entries) != 1:
            warnings.warn("Losses are inconsistent, number of entries for each loss: "
                          f"{valmap(len, group)}. "
                          "This can also happen due to rerun experiment, "
                          "however please validate your loss function.")

        for k, vs in group.items():
            pl_module.log(f'train/{k}', np.mean(vs))

        self._train_losses.clear()

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
            batch_idx: Hashable,
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
            batch_idx: Hashable,
            dataloader_idx: int = 0,
    ) -> None:
        if len(outputs) != 2:
            raise ValueError(f"Expected step output in form of 2 elements (x, y), "
                             f"but received {len(outputs)}")
        xs, ys = outputs
        xs = _recombine_batch(xs) if isinstance(xs, (list, tuple)) else xs
        ys = _recombine_batch(ys) if isinstance(ys, (list, tuple)) else ys

        outputs = (ys, xs)

        if self.group_metrics:
            for preprocess in self.group_preprocess.keys():
                self._all_predictions[dataloader_idx][preprocess].extend(
                    preprocess(*args) for args in zip_equal(*outputs)
                )

        for i, (target, pred) in enumerate(zip_equal(*outputs)):
            object_idx = f"{batch_idx}_{i}"
            for preprocess, metrics_names in self.single_preprocess.items():
                preprocessed = preprocess(target, pred)
                for name in metrics_names:
                    self._single_metric_values[dataloader_idx][name][object_idx] = \
                        self.single_metrics[name](*preprocessed)

    def evaluate_epoch(self, trainer: Trainer, pl_module: LightningModule, key: str) -> None:
        self._squeeze_ids_in_single_metrics()

        group_metric_values = {}

        for dataloader_idx, all_predictions in self._all_predictions.items():
            loader_postfix = f"/{dataloader_idx}" if len(self._all_predictions) > 1 else ""
            for preprocess, metrics_names in self.group_preprocess.items():
                preprocessed = [np.asarray(p) for p in zip_equal(*all_predictions[preprocess])]
                for name in metrics_names:
                    group_metric_values[f"{name}{loader_postfix}"] = self.group_metrics[name](*preprocessed)

        single_metric_values = {}
        for fn_name, fn in self.aggregate_fn.items():
            for dataloader_idx, metrics in self._single_metric_values.items():
                prefix = f"{fn_name}/" if fn_name else ""
                loader_postfix = f"/{dataloader_idx}" if len(self._single_metric_values) > 1 else ""

                if self.log_individual_metrics:
                    dataframe = pd.DataFrame(metrics)
                    root_dir = Path(trainer.log_dir) / key
                    root_dir.mkdir(exist_ok=True)
                    for logger in pl_module.loggers:
                        if hasattr(logger, "log_table"):
                            logger.log_table(f"{key}/dataloader_{dataloader_idx}", dataframe=dataframe)

                    dataframe.to_csv(root_dir / f"dataloader_{dataloader_idx}.csv")

                single_metric_values.update({f"{prefix}{k}{loader_postfix}":
                                            fn(list(v.values())) for k, v in metrics.items()})

        self._single_metric_values.clear()
        self._all_predictions.clear()

        for k, value in chain(single_metric_values.items(), group_metric_values.items()):
            pl_module.log(f'{key}/{k}', value)

    def _squeeze_ids_in_single_metrics(self):
        for dataloader_idx, _ in self._all_predictions.items():
            for _, metrics_names in self.single_preprocess.items():
                for name in metrics_names:
                    if all(k.rsplit("_", 1)[1] == "0" for k in
                           self._single_metric_values[dataloader_idx][name].keys()):
                        self._single_metric_values[dataloader_idx][name] = \
                            keymap(lambda k: k.rsplit("_", 1)[0], self._single_metric_values[dataloader_idx][name])


def _get_func_name(function: Callable) -> str:
    if isinstance(function, partial):
        function = function.func

    if callable(function):
        if hasattr(function, "__name__"):
            return function.__name__
        return function.__class__.__name__

    raise ValueError(f"You must pass a callable object, got {type(function)}")


def _identity(*args):
    return squeeze_first(args)


def _recombine_batch(xs: Sequence) -> List:
    return [squeeze_first(x) for x in zip_equal(*xs)]


def _process_metrics(raw_metrics: Dict) -> Tuple[Dict[str, Callable], Dict[Callable, List[str]]]:
    processed_metrics = {}
    preprocess: Dict[Callable, List[str]] = defaultdict(list)

    # collect metrics
    for k, v in raw_metrics.items():
        if isinstance(k, str):
            processed_metrics[k] = v
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
            preprocess[k] = sorted(metrics.keys())
            processed_metrics.update(metrics)
        else:
            raise TypeError(f"Metric keys should be of type str or Callable, got {type(k)}")

    identity_preprocess_metrics = sorted(set(processed_metrics.keys()) - set(chain.from_iterable(preprocess.values())))
    if identity_preprocess_metrics:
        preprocess[_identity] = identity_preprocess_metrics
    return processed_metrics, preprocess
