from contextlib import nullcontext
from functools import partial, wraps
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from lightning import Trainer
from lightning.pytorch.demos.boring_classes import RandomDataset
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from more_itertools import collapse
from sklearn.metrics import accuracy_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from thunder import ThunderModule
from thunder.callbacks import MetricMonitor


def ravel(metric):
    @wraps(metric)
    def wrapper(x, y):
        return metric(np.ravel(x), np.ravel(y))

    return wrapper


accuracy = ravel(accuracy_score)
recall = ravel(recall_score)


class NoOptimModule(ThunderModule):
    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0, requires_grad=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return np.asarray([1, 0]), np.asarray([batch_idx % 2, batch_idx % 3])

    def test_step(self, batch, batch_idx, dataloader_idx=0):
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
    metric_monitor = MetricMonitor(single_metrics={"accuracy": accuracy})
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=4,
        limit_val_batches=4,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[metric_monitor],
        logger=CSVLogger(tmpdir),
    )
    model = NoOptimSegm(nn.Linear(2, 1), lambda x, y: x + y, -1)
    trainer.fit(model)
    trainer.test(model)

    metrics = pd.read_csv(f"{tmpdir}/lightning_logs/version_0/metrics.csv")

    m = (np.asarray([[1, 1], [0, 0]]) == np.asarray([[1, 0], [0, 0]])).sum()
    assert all(np.allclose(m, metrics["train/loss"].dropna().iloc[i]) for i in range(2))

    m = np.mean([accuracy(x, y) for y, x in zip(np.asarray([[1, 1], [1, 0]]), np.asarray([[1, 1], [0, 0]]))] * 4)
    assert all(np.allclose(m, metrics["val/accuracy"].dropna().iloc[i]) for i in range(2))


def test_group_metrics(tmpdir):
    metric_monitor = MetricMonitor(group_metrics={"accuracy": accuracy})
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=4,
        limit_val_batches=4,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[metric_monitor],
        logger=CSVLogger(tmpdir),
    )
    model = NoOptimModule(nn.Linear(2, 1), lambda x, y: x + y, -1)
    trainer.fit(model)
    trainer.test(model)

    metrics = pd.read_csv(f"{tmpdir}/lightning_logs/version_0/metrics.csv")
    assert all(np.allclose(0, metrics["train/loss"].dropna().iloc[i]) for i in range(2))
    x = np.asarray(list(collapse([[1, 0] for i in range(4)])))
    y = np.asarray(list(collapse([[i % 2, i % 3] for i in range(4)])))
    acc = accuracy(x, y)
    assert all(np.allclose(acc, metrics["val/accuracy"].dropna().iloc[i]) for i in range(2))


def test_group_metrics_with_preprocessing(tmpdir):
    metric_monitor = MetricMonitor(group_metrics={lambda y, x: (y > 0.5, x): accuracy_score})
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=4,
        limit_val_batches=4,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[metric_monitor],
        logger=CSVLogger(tmpdir),
    )
    model = NoOptimModule(nn.Linear(2, 1), lambda x, y: x + y, -1)
    trainer.fit(model)
    trainer.test(model)

    metrics = pd.read_csv(f"{tmpdir}/lightning_logs/version_0/metrics.csv")
    assert all(np.allclose(0, metrics["train/loss"].dropna().iloc[i]) for i in range(2))
    x = np.asarray(list(collapse([[1, 0] for i in range(4)])))
    y = np.asarray(list(collapse([[i % 2, i % 3] for i in range(4)])))
    acc = accuracy(x, y)
    assert all(np.allclose(acc, metrics["val/accuracy_score"].dropna().iloc[i]) for i in range(2))


def test_group_preprocessing(tmpdir):
    class Module(ThunderModule):
        def __init__(self):
            super().__init__(nn.Linear(2, 1), lambda: ())

        def train_dataloader(self) -> TRAIN_DATALOADERS:
            return DataLoader(RandomDataset(32, 64), batch_size=16)

        def val_dataloader(self) -> EVAL_DATALOADERS:
            return DataLoader(RandomDataset(32, 64), batch_size=3)

        def training_step(self, batch, *args, **kwargs):
            x = torch.mean(batch[0])
            x.requires_grad = True
            return x

        def validation_step(self, batch, *args, **kwargs):
            return batch, torch.randn(3) > 0.5

        def configure_optimizers(self):
            return torch.optim.Adam(self.architecture.parameters())

    def check_preprocess(y, x):
        assert y == 0 or y == 1
        assert tuple(x.shape) == (32,)
        return y, x.mean() > 0

    metric_monitor = MetricMonitor(group_metrics={check_preprocess: accuracy_score})
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=4,
        limit_val_batches=4,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[metric_monitor],
        logger=CSVLogger(tmpdir),
    )
    model = Module()
    trainer.fit(model)


def test_metrics_collision(tmpdir):
    metric_monitor = MetricMonitor(single_metrics={"accuracy": lambda y, x: y + x,
                                                   "recall": lambda y, x: y + x},
                                   group_metrics={"accuracy": lambda y, x: y + x,
                                                  "precision": lambda y, x: y + x})

    assert sorted(metric_monitor.single_metrics.keys()) == sorted(["single/accuracy", "recall"])
    assert sorted(metric_monitor.group_metrics.keys()) == sorted(["group/accuracy", "precision"])
    assert sorted(chain(*metric_monitor.single_preprocess.values())) == sorted(["single/accuracy", "recall"])
    assert sorted(chain(*metric_monitor.group_preprocess.values())) == sorted(["group/accuracy", "precision"])


@pytest.mark.parametrize(
    "aggregate_fn, target, exception",
    [
        (None, ["accuracy"], nullcontext()),
        ("std", ["accuracy", "std/accuracy"], nullcontext()),
        (["std", "max"], ["accuracy", "std/accuracy", "max/accuracy"], nullcontext()),
        (["std", "wrong_key"], [], pytest.raises(ValueError, match="wrong_key")),
        (np.sum, ["accuracy", "sum/accuracy"], nullcontext()),
        ([np.std, np.max], ["accuracy", "std/accuracy", f"{np.max.__name__}/accuracy"], nullcontext()),
        ([np.sum, "max"], ["accuracy", "sum/accuracy", "max/accuracy"], nullcontext()),
        ([np.sum, "max", 2], [], pytest.raises(TypeError, match="int")),
        ({"sum": np.sum, "max": max}, ["accuracy", "sum/accuracy", "max/accuracy"], nullcontext()),
        ({"sum": np.sum, "zero": 0}, [], pytest.raises(TypeError, match="zero")),
        (partial(np.sum), ["accuracy", "sum/accuracy"], nullcontext()),
    ],
)
def test_aggregators(aggregate_fn, target, exception, tmpdir):
    metric_monitor = None
    with exception:
        metric_monitor = MetricMonitor(single_metrics={"accuracy": accuracy}, aggregate_fn=aggregate_fn)
    if metric_monitor is None:
        return

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=4,
        limit_val_batches=4,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[metric_monitor],
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
            {lambda x, y: (x, np.zeros_like(y)): [accuracy, accuracy2]},
            {"accuracy_score": 0.25, "accuracy2": 0.25},
            nullcontext(),
        ),
        (
            {
                lambda x, y: (x, np.zeros_like(y)): {"acc1": accuracy, "acc2": accuracy},
                lambda x, y: (x, np.ones_like(y) * 2): {"acc3": accuracy, "acc4": accuracy},
            },
            {"acc1": 0.25, "acc2": 0.25, "acc3": 0, "acc4": 0},
            nullcontext(),
        ),
        ({lambda *args: args: "std"}, {}, pytest.raises(TypeError, match="str")),
    ],
)
def test_preprocessing(single_metrics, target, exception, tmpdir):
    metric_monitor = None
    with exception:
        metric_monitor = MetricMonitor(single_metrics=single_metrics)
    if metric_monitor is None:
        return

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=4,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[metric_monitor],
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


class MultiSegm(NoOptimSegm):
    def __init__(self, batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def validation_step(self, batch, batch_idx):
        x, y = super().validation_step(batch, batch_idx)
        return (x, x), (y, y)

    def test_step(self, batch, batch_idx):
        x, y = super().test_step(batch, batch_idx)
        return x, (y,)


class MultiSegmCustomBS(MultiSegm):
    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=self.batch_size)


@pytest.mark.parametrize("batch_size", [1, 2, 16])
def test_multioutput(batch_size, tmpdir):
    metric_monitor = MetricMonitor(
        {
            lambda x, y: (
                y[0] if isinstance(y, (tuple, list)) else y,
                x[0] if isinstance(x, (tuple, list)) else x,
            ): accuracy
        }
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=4,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[metric_monitor],
        logger=CSVLogger(tmpdir),
    )
    model = MultiSegmCustomBS(batch_size, nn.Linear(2, 1), lambda x, y: x + y, -1)
    trainer.fit(model)

    target = {"accuracy_score": accuracy(np.asarray([[1, 1], [1, 0]]), np.asarray([[1, 1], [0, 0]]))}

    df = pd.read_csv(f"{tmpdir}/lightning_logs/version_0/metrics.csv")
    columns = [c.replace("val/", "") for c in df.columns if "val/" in c]
    assert all(np.allclose(df[f"val/{c}"].iloc[0], target[c]) for c in columns), (
        df["val/accuracy_score"].iloc[0],
        target,
    )

    trainer.test(model)
    df = pd.read_csv(f"{tmpdir}/lightning_logs/version_0/metrics.csv")
    columns = [c.replace("test/", "") for c in df.columns if "test/" in c]
    assert all(np.allclose(df[f"test/{c}"].iloc[-1], target[c]) for c in columns), (
        df["test/accuracy_score"].iloc[-1],
        target,
    )


class MultiLoaderModule(NoOptimModule):
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            return np.asarray([1, 0]), np.asarray([batch_idx % 2, batch_idx % 3])
        return np.asarray([0, 0]), np.asarray([0, 0])

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            return np.asarray([1, 0]), np.asarray([batch_idx % 2, batch_idx % 3])
        return np.asarray([0, 0]), np.asarray([0, 0])

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=16), DataLoader(RandomDataset(32, 64), batch_size=16)

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=16), DataLoader(RandomDataset(32, 64), batch_size=16)


def test_multiple_loaders(tmpdir):
    metric_monitor = MetricMonitor(single_metrics={"accuracy": accuracy})
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=4,
        limit_val_batches=4,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[metric_monitor],
        logger=CSVLogger(tmpdir),
    )
    model = MultiLoaderModule(nn.Linear(2, 1), lambda x, y: x + y, -1)
    trainer.fit(model)
    trainer.test(model)

    df = pd.read_csv(f"{tmpdir}/lightning_logs/version_0/metrics.csv")
    val_columns = [c for c in df.columns if "val/accuracy" in c]
    test_columns = [c for c in df.columns if "test/accuracy" in c]

    assert len(val_columns) == 2, (val_columns, df.columns)
    assert "val/accuracy/0" in val_columns and "val/accuracy/1" in val_columns
    assert len(test_columns) == 2, (test_columns, df.columns)
    assert "test/accuracy/0" in test_columns and "test/accuracy/1" in test_columns


def test_log_table(tmpdir):
    metric_monitor = MetricMonitor(single_metrics={"accuracy": accuracy}, log_individual_metrics=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=4,
        limit_val_batches=4,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[metric_monitor],
        logger=CSVLogger(tmpdir),
    )
    model = MultiLoaderModule(nn.Linear(2, 1), lambda x, y: x + y, -1)
    trainer.fit(model)
    trainer.test(model)

    root_dir = trainer.default_root_dir
    for p in Path(root_dir).glob("*/dataloader_*"):
        assert str(p.relative_to(root_dir)) not in ["val/dataloader_0", "val/dataloader_1",
                                                    "test/dataloader_0", "test/dataloader_1"]
