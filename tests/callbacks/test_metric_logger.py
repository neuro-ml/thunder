from functools import wraps

import numpy as np
import pandas as pd
import pytest
import torch
from lightning import Trainer
from lightning.pytorch.demos.boring_classes import RandomDataset
from lightning.pytorch.loggers import CSVLogger
from more_itertools import collapse
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader

from thunder import ThunderModule
from thunder.callbacks import MetricLogger


def ravel(metric):
    @wraps(metric)
    def wrapper(x, y):
        return metric(x.ravel(), y.ravel())
    return wrapper


accuracy = ravel(accuracy_score)


class NoOptimModule(ThunderModule):
    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0, requires_grad=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return np.asarray([1, 0]), np.asarray([batch_idx % 2, batch_idx % 3])

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=16)

    def val_dataloader(self):
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


def test_single_metrics(tmpdir):
    metric_logger = MetricLogger(single_metrics={"accuracy": accuracy})
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2,
                      limit_train_batches=4, limit_val_batches=4,
                      enable_checkpointing=False, enable_progress_bar=False,
                      callbacks=[metric_logger], logger=CSVLogger(tmpdir))
    model = NoOptimSegm(nn.Linear(2, 1), lambda x, y: x + y, -1)
    trainer.fit(model)

    metrics = pd.read_csv(f"{tmpdir}/lightning_logs/version_0/metrics.csv")

    m = (np.asarray([[1, 1], [0, 0]]) == np.asarray([[1, 0], [0, 0]])).sum()
    assert all([np.allclose(m, metrics["train/loss"].dropna().iloc[i]) for i in range(2)])

    m = np.mean([accuracy(x, y) for y, x in zip(np.asarray([[1, 1], [1, 0]]), np.asarray([[1, 1], [0, 0]]))] * 4)
    assert all([np.allclose(m, metrics["val/accuracy"].dropna().iloc[i]) for i in range(2)])


def test_group_metrics(tmpdir):
    metric_logger = MetricLogger(group_metrics={"accuracy": accuracy})
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2,
                      limit_train_batches=4, limit_val_batches=4,
                      enable_checkpointing=False, enable_progress_bar=False,
                      callbacks=[metric_logger], logger=CSVLogger(tmpdir))
    model = NoOptimModule(nn.Linear(2, 1), lambda x, y: x + y, -1)
    trainer.fit(model)

    metrics = pd.read_csv(f"{tmpdir}/lightning_logs/version_0/metrics.csv")
    assert all([np.allclose(0, metrics["train/loss"].dropna().iloc[i]) for i in range(2)])
    x = np.asarray(list(collapse([[1, 0] for i in range(4)] * 2)))
    y = np.asarray(list(collapse([[i % 2, i % 3] for i in range(4)] * 2)))
    acc = accuracy(x, y)
    assert all([np.allclose(acc, metrics["val/accuracy"].dropna().iloc[i]) for i in range(2)])
