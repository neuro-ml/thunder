import csv

import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from thunder.callbacks import TimeProfiler


TESTKEYS = [
    tuple(), ("optimizer step",),
    ("backward", "optimizer step", "total train downtime", "total val downtime"),
    (True,),
    ("backward", "wrong key")
]


@pytest.mark.parametrize("keys", TESTKEYS)
def test_init(keys, tmpdir):
    """Test TPU stats are logged using a logger."""
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
