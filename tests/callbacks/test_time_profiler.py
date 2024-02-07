import csv

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.optim import Adam
from torch.utils.data import DataLoader

from thunder import ThunderModule
from thunder.callbacks import TimeProfiler
from thunder.torch.utils import last_checkpoint


TESTKEYS = [
    tuple(), ("optimizer step",),
    ("backward", "optimizer step", "total train downtime", "total val downtime"),
    (True,),
    ("backward", "wrong key")
]


@pytest.mark.parametrize("keys", TESTKEYS)
def test_init(keys, tmpdir):
    model = BoringModel()

    if "wrong key" in keys:
        with pytest.raises(ValueError, match="unknown"):
            TimeProfiler(*keys)
        return

    time_profiler = TimeProfiler(*keys)

    if len(keys) > 0 and keys[0] is True:
        assert sorted(time_profiler.keys) == sorted(time_profiler._default_keys + time_profiler._optional_keys)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=2,
        log_every_n_steps=1,
        callbacks=[time_profiler],
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model)


def test_time_profiler_no_logger(tmpdir):
    """Test TimeProfiler with no logger in Trainer."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[TimeProfiler()],
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    with pytest.raises(MisconfigurationException, match="Cannot use"):
        trainer.fit(model)


def test_time_profiler_logs_for_different_stages(tmpdir):
    model = BoringModel()

    time_profiler = TimeProfiler(True)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=4,
        limit_test_batches=1,
        log_every_n_steps=1,
        callbacks=[time_profiler],
        logger=CSVLogger(tmpdir),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    # training and validation stages will run
    trainer.fit(model)

    with open(f"{tmpdir}/lightning_logs/version_0/metrics.csv") as csvfile:
        content = csv.reader(csvfile, delimiter=",")
        it = iter(content).__next__()

    # searching for training stage logs
    for key in filter(lambda k: "train" in k, time_profiler.keys):
        assert f"{time_profiler.__class__.__name__}/{key}" in it

    # searching for validation stage logs
    for key in filter(lambda k: "val" in k, time_profiler.keys):
        assert f"{time_profiler.__class__.__name__}/{key}" in it


def test_with_no_validation(tmpdir):
    class Model(BoringModel):
        def val_dataloader(self):
            return []

    model = Model()
    time_profiler = TimeProfiler(True)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=4,
        limit_test_batches=1,
        log_every_n_steps=1,
        callbacks=[time_profiler],
        logger=CSVLogger(tmpdir),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    # training and validation stages will run
    trainer.fit(model)

    with open(f"{tmpdir}/lightning_logs/version_0/metrics.csv") as csvfile:
        content = csv.reader(csvfile, delimiter=",")
        it = iter(content).__next__()

    # searching for training stage logs
    for key in filter(lambda k: "train" in k, time_profiler.keys):
        assert f"{time_profiler.__class__.__name__}/{key}" in it

    # searching for validation stage logs
    for key in filter(lambda k: "val" in k, time_profiler.keys):
        assert f"{time_profiler.__class__.__name__}/{key}" not in it


@pytest.mark.parametrize("args", [(True,), ("backward",)])
def test_load_from_checkpoint(args, tmpdir):
    """
    Checks whether state `TimeProfiler` is restored properly after experiment fails.
    """
    class Dataset(RandomDataset):
        def __getitem__(self, item):
            return super().__getitem__(item), torch.randn(1)[0]

    ERR_MSG = "Baby it's time to fail."

    FAILED = [False]  # Marks if training has failed.

    class Module(ThunderModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def training_step(self, batch, batch_idx):
            if FAILED[0]:
                assert self.trainer.current_epoch >= 2
            out = super().training_step(batch, batch_idx)
            if self.trainer.current_epoch == 2 and not FAILED[0]:
                raise RuntimeError(ERR_MSG)
            return out

    model = torch.nn.Sequential(torch.nn.Linear(3, 1))

    optimizer = Adam(model.parameters(), lr=1)
    loader = DataLoader(Dataset(3, 64), batch_size=2)
    module = Module(model, lambda x, y: x.mean(), optimizer=optimizer)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=5,
        limit_train_batches=2,
        callbacks=[ModelCheckpoint(save_last=True), TimeProfiler(*args)],
        logger=CSVLogger(tmpdir),
        enable_progress_bar=False,
    )

    with pytest.raises(RuntimeError, match=ERR_MSG):
        trainer.fit(module, loader, ckpt_path="last")

    FAILED[0] = True
    module = Module(model, lambda x, y: x.mean(), optimizer=optimizer)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=5,
        limit_train_batches=2,
        callbacks=[TimeProfiler(*args), ModelCheckpoint(save_last=True)],
        logger=CSVLogger(tmpdir),
        enable_progress_bar=False,
    )
    trainer.fit(module, loader, ckpt_path=last_checkpoint(tmpdir))
