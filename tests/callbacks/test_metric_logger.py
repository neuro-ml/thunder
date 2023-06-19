from contextlib import nullcontext
from functools import wraps, partial

import numpy as np
import pandas as pd
import pytest
import torch
from lightning import Trainer
from lightning.pytorch.demos.boring_classes import RandomDataset
from lightning.pytorch.loggers import CSVLogger
from more_itertools import collapse
from sklearn.metrics import accuracy_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from thunder import ThunderModule
from thunder.callbacks import MetricLogger


def ravel(metric):
    @wraps(metric)
    def wrapper(x, y):
        return metric(np.ravel(x), np.ravel(y))

    return wrapper


accuracy = ravel(accuracy_score)


class NoOptimModule(ThunderModule):
    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0, requires_grad=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return np.asarray([1, 0]), np.asarray([batch_idx % 2, batch_idx % 3])

    def test_step(self, batch, batch_idx, dataloader_idx = 0):
        return np.asarray([1, 0]), np.asarray([batch_idx % 2, batch_idx % 3])

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=16)

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=16)

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=16)

    def configure_optimizers(self):
        return None


class NoOptimSegm(NoOptimModule):
    def training_step(self, batch, batch_idx):
        x = torch.tensor([[1, 1], [0, 0]])
        y = torch.tensor([[1, 0], [0, 0]])
        return torch.tensor(x == y, dtype=torch.float, requires_grad=True).sum()

    def validation_step(self, batch, batch_idx):
        x = np.asarray([[1, 1], [0, 0]])
        y = np.asarray([[1, 1], [1, 0]])
        return x, y

    def test_step(self, batch, batch_idx):
        x = np.asarray([[1, 1], [0, 0]])
        y = np.asarray([[1, 1], [1, 0]])
        return x, y


def test_single_metrics(tmpdir):
    metric_logger = MetricLogger(single_metrics={"accuracy": accuracy})
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=4,
        limit_val_batches=4,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[metric_logger],
        logger=CSVLogger(tmpdir),
    )
    model = NoOptimSegm(nn.Linear(2, 1), lambda x, y: x + y, -1)
    trainer.fit(model)
    trainer.test(model)

    metrics = pd.read_csv(f"{tmpdir}/lightning_logs/version_0/metrics.csv")

    m = (np.asarray([[1, 1], [0, 0]]) == np.asarray([[1, 0], [0, 0]])).sum()
    assert all([np.allclose(m, metrics["train/loss"].dropna().iloc[i]) for i in range(2)])

    m = np.mean([accuracy(x, y) for y, x in zip(np.asarray([[1, 1], [1, 0]]), np.asarray([[1, 1], [0, 0]]))] * 4)
    assert all([np.allclose(m, metrics["val/accuracy"].dropna().iloc[i]) for i in range(2)])


def test_group_metrics(tmpdir):
    metric_logger = MetricLogger(group_metrics={"accuracy": accuracy})
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=4,
        limit_val_batches=4,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[metric_logger],
        logger=CSVLogger(tmpdir),
    )
    model = NoOptimModule(nn.Linear(2, 1), lambda x, y: x + y, -1)
    trainer.fit(model)
    trainer.test(model)

    metrics = pd.read_csv(f"{tmpdir}/lightning_logs/version_0/metrics.csv")
    assert all([np.allclose(0, metrics["train/loss"].dropna().iloc[i]) for i in range(2)])
    x = np.asarray(list(collapse([[1, 0] for i in range(4)])))
    y = np.asarray(list(collapse([[i % 2, i % 3] for i in range(4)])))
    acc = accuracy(x, y)
    assert all([np.allclose(acc, metrics["val/accuracy"].dropna().iloc[i]) for i in range(2)])


@pytest.mark.parametrize(
    "aggregate_fn, target, exception",
    [
        (None, ["accuracy"], nullcontext()),
        ("std", ["accuracy", "std/accuracy"], nullcontext()),
        (["std", "max"], ["accuracy", "std/accuracy", "max/accuracy"], nullcontext()),
        (["std", "wrong_key"], [], pytest.raises(ValueError, match="wrong_key")),
        (np.sum, ["accuracy", "sum/accuracy"], nullcontext()),
        ([np.std, np.max], ["accuracy", "std/accuracy", "max/accuracy"], nullcontext()),
        ([np.sum, "max"], ["accuracy", "sum/accuracy", "max/accuracy"], nullcontext()),
        ([np.sum, "max", 2], [], pytest.raises(TypeError, match="int")),
        ({"sum": np.sum, "max": max}, ["accuracy", "sum/accuracy", "max/accuracy"], nullcontext()),
        ({"sum": np.sum, "zero": 0}, [], pytest.raises(TypeError, match="zero")),
        (partial(np.sum), ["accuracy", "sum/accuracy"], nullcontext()),
    ],
)
def test_aggregators(aggregate_fn, target, exception, tmpdir):
    metric_logger = None
    with exception:
        metric_logger = MetricLogger(single_metrics={"accuracy": accuracy}, aggregate_fn=aggregate_fn)
    if metric_logger is None:
        return

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=4,
        limit_val_batches=4,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[metric_logger],
        logger=CSVLogger(tmpdir),
    )
    model = NoOptimSegm(nn.Linear(2, 1), lambda x, y: x + y, -1)
    trainer.fit(model)

    columns = pd.read_csv(f"{tmpdir}/lightning_logs/version_0/metrics.csv").columns
    columns = [c.replace("val/", "") for c in columns if "val/" in c]
    assert sorted(columns) == sorted(target), aggregate_fn

    trainer.test(model)
    columns = pd.read_csv(f"{tmpdir}/lightning_logs/version_0/metrics.csv").columns
    columns = [c.replace("test/", "") for c in columns if "test/" in c]
    assert sorted(columns) == sorted(target), aggregate_fn


def accuracy2(*args, **kwargs):
    return accuracy(*args, **kwargs)


@pytest.mark.parametrize(
    "single_metrics, target, exception",
    [
        ({"accuracy": accuracy}, {"accuracy": 0.75}, nullcontext()),
        (
            {lambda x, y: (np.zeros_like(x), y): [accuracy, accuracy2]},
            {"accuracy_score": 0.25, "accuracy2": 0.25},
            nullcontext(),
        ),
        (
            {
                lambda x, y: (np.zeros_like(x), y): {"acc1": accuracy, "acc2": accuracy},
                lambda x, y: (np.ones_like(x) * 2, y): {"acc3": accuracy, "acc4": accuracy},
            },
            {"acc1": 0.25, "acc2": 0.25, "acc3": 0, "acc4": 0},
            nullcontext(),
        ),
        ({lambda *args: args: "std"}, {}, pytest.raises(TypeError, match="str")),
    ],
)
def test_preprocessing(single_metrics, target, exception, tmpdir):
    metric_logger = None
    with exception:
        metric_logger = MetricLogger(single_metrics=single_metrics)
    if metric_logger is None:
        return

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=4,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[metric_logger],
        logger=CSVLogger(tmpdir),
    )
    model = NoOptimSegm(nn.Linear(2, 1), lambda x, y: x + y, -1)
    trainer.fit(model)

    df = pd.read_csv(f"{tmpdir}/lightning_logs/version_0/metrics.csv")
    columns = [c.replace("val/", "") for c in df.columns if "val/" in c]
    assert sorted(columns) == sorted(target.keys())
    assert all(np.allclose(df[f"val/{c}"].iloc[0], target[c]) for c in columns), (df.iloc[0], target)

    trainer.test(model)
    df = pd.read_csv(f"{tmpdir}/lightning_logs/version_0/metrics.csv")
    columns = [c.replace("test/", "") for c in df.columns if "test/" in c]
    assert sorted(columns) == sorted(target.keys())
    assert all(np.allclose(df[f"test/{c}"].iloc[-1], target[c]) for c in columns), (df.iloc[0], target)