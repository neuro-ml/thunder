import numpy as np
import pytest
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import RandomDataset
from lightning.pytorch.loggers import CSVLogger
from more_itertools import zip_equal
from torch import nn
from torch.nn import Sequential
from torch.optim import Adam
from torch.utils.data import DataLoader

from thunder import ThunderModule
from thunder.policy import Multiply, Policy, Schedule, Switch
from thunder.torch.utils import last_checkpoint


@pytest.fixture
def model():
    return Sequential(nn.Linear(3, 2), nn.Linear(2, 1))


@pytest.fixture
def optim_single_group(model):
    return Adam(model.parameters(), lr=1)


@pytest.fixture
def optim(model):
    return Adam([{"params": model[0].parameters(), "lr": 1e-3}, {"params": model[1].parameters(), "lr": 1e-4}])


def test_init(optim, optim_single_group):
    multiply = Multiply([{0: 1} for _ in range(3)])
    schedule = Schedule([np.cos for _ in range(3)])
    switch = Switch({1: 10}, [1, 2, 3])

    with pytest.raises(ValueError, match="Got 3"):
        multiply(optim)
    with pytest.raises(ValueError, match="Got 3"):
        multiply(optim_single_group)
    with pytest.raises(ValueError, match="Got 3"):
        schedule(optim)
    with pytest.raises(ValueError, match="Got 3"):
        schedule(optim_single_group)
    with pytest.raises(ValueError, match="Got 3"):
        switch(optim)
    with pytest.raises(ValueError, match="Got 3"):
        switch(optim_single_group)
    with pytest.raises(ValueError, match="Got 2"):
        Switch([{1: 10}, {1: 10}], [1, 2, 3])(optim_single_group)


def test_custom_policy(optim, tmpdir):
    class NewPolicy(Policy):
        def get_lr(self):
            return [pg["lr"] * 0 for pg in self.optimizer.param_groups]

        def state_dict(self):
            return super().state_dict()

        def load_state_dict(self, state_dict):
            super().load_state_dict(state_dict)

    policy = NewPolicy()(optim)
    check_scheduler(policy, [0, 0, 0, 0, 0])
    check_scheduler_saving(NewPolicy()(optim), policy, optim, tmpdir)


@pytest.mark.parametrize(
    "mapping,targets",
    [
        ({0: 0.1, 3: 0.1}, [[0.1, 10], [0.1, 10], [0.1, 10], [0.01, 1], [0.01, 1]]),
        ({1: 0.1, 3: 0.1}, [[1, 100], [0.1, 10], [0.1, 10], [0.01, 1], [0.01, 1]]),
        ([{1: 0.1, 3: 0.1}, {0: 0.1, 3: 1000}], [[1, 10], [0.1, 10], [0.1, 10], [0.01, 10000], [0.01, 10000]]),
        ({}, [[1, 100] for _ in range(5)]),
    ],
)
def test_multiply(mapping, targets, optim, tmpdir):
    scheduler = Multiply(mapping, [1, 100])
    scheduler(optim)
    check_scheduler(scheduler, targets)

    new_scheduler = Multiply(mapping)
    check_scheduler_saving(new_scheduler, scheduler, optim, tmpdir)


@pytest.mark.parametrize("mapping,targets", [(np.cos, [[np.cos(i), np.cos(i)] for i in range(5)]),
                                             ([np.cos, np.sin], [[np.cos(i), np.sin(i)] for i in range(5)])])
def test_schedule(mapping, targets, optim, tmpdir):
    scheduler = Schedule(mapping, [1, 100])
    scheduler(optim)
    check_scheduler(scheduler, targets)

    new_scheduler = Schedule(mapping)
    check_scheduler_saving(new_scheduler, scheduler, optim, tmpdir)


@pytest.mark.parametrize(
    "mapping,lr_init,targets",
    [
        ({0: 1, 1: 1, 2: 10, 3: 4, 4: 5}, 100, [[1, 1], [1, 1], [10, 10], [4, 4], [5, 5]]),
        ({0: 2, 3: 6}, 100, [[2, 2], [2, 2], [2, 2], [6, 6], [6, 6]]),
        ({2: 0}, 100, [[100, 100], [100, 100], [0, 0], [0, 0], [0, 0]]),
        ([{0: 1, 1: 1, 2: 10, 3: 4, 4: 5}, {0: 1, 1: 3, 2: 56, 3: 4, 4: 5}], [100, 50],
         [[1, 1], [1, 3], [10, 56], [4, 4], [5, 5]]),
        ([{0: 2, 3: 23}, {0: 2, 3: 6}], [100, 50], [[2, 2], [2, 2], [2, 2], [23, 6], [23, 6]]),
        ({2: 0}, [100, 50], [[100, 50], [100, 50], [0, 0], [0, 0], [0, 0]]),
    ],
)
def test_switch(mapping, lr_init, targets, optim, tmpdir):
    scheduler = Switch(mapping, lr_init)
    scheduler(optim)
    check_scheduler(scheduler, targets)

    new_scheduler = Switch(mapping, 0.0001)
    check_scheduler_saving(new_scheduler, scheduler, optim, tmpdir)


@pytest.mark.parametrize(
    "lr_scheduler, mapping, lr_mapping",
    [
        (Switch({0: 1, 1: 2, 2: 10, 3: 4, 4: 5}, lr_init=1),
         {0: 1, 1: 2, 2: 10, 3: 4, 4: 5}, {0: 1, 1: 2, 2: 10, 3: 4, 4: 5}),
        (Multiply({0: 1, 1: 1, 2: 10, 3: 4, 4: 5}, lr_init=1),
         {0: 1, 1: 1, 2: 10, 3: 4, 4: 5}, {0: 1, 1: 1, 2: 10, 3: 40, 4: 200}),
        (Schedule(lambda x: x + 1, lr_init=1), lambda x: x + 1, {i: i + 1 for i in range(5)}),
    ],
)
def test_load_from_checkpoint(lr_scheduler, mapping, lr_mapping, model, tmpdir):
    """
    Checks whether state of schedulers is restored properly after experiment fails.
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
            """

            """
            if FAILED[0]:
                assert self.trainer.current_epoch >= 2
            out = super().training_step(batch, batch_idx)
            if self.trainer.current_epoch == 2 and not FAILED[0]:
                raise RuntimeError(ERR_MSG)
            if callable(lr_mapping):
                lr = lr_mapping(self.trainer.current_epoch)
            else:
                lr = lr_mapping[self.trainer.current_epoch]
            assert lr == self.optimizer.param_groups[0]["lr"]
            return out

    optimizer = Adam(model.parameters(), lr=1)
    loader = DataLoader(Dataset(3, 64), batch_size=2)
    module = Module(model, lambda x, y: x.mean(), optimizer=optimizer, lr_scheduler=lr_scheduler)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=5,
        limit_train_batches=2,
        callbacks=[ModelCheckpoint(save_last=True)],
        logger=CSVLogger(tmpdir),
        enable_progress_bar=False,
    )

    with pytest.raises(RuntimeError, match=ERR_MSG):
        trainer.fit(module, loader, ckpt_path="last")

    FAILED[0] = True
    optimizer = Adam(model.parameters(), lr=1)
    loader = DataLoader(Dataset(3, 64), batch_size=2)
    new_lr_scheduler = lr_scheduler.__class__(mapping, lr_init=1)
    module = Module(model, lambda x, y: x.mean(), optimizer=optimizer, lr_scheduler=new_lr_scheduler)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=5,
        limit_train_batches=2,
        callbacks=[ModelCheckpoint(save_last=True)],
        logger=CSVLogger(tmpdir),
        enable_progress_bar=False,
    )
    trainer.fit(module, loader, ckpt_path=last_checkpoint(tmpdir))


def check_scheduler(scheduler, targets):
    for i, target in zip_equal(range(5), targets):
        assert np.allclose([pg["lr"] for pg in scheduler.optimizer.param_groups], target), f"epoch {i}"
        scheduler.optimizer.step()
        scheduler.step()


def check_scheduler_saving(new_scheduler, scheduler, optim, tmpdir):
    torch.save(scheduler.state_dict(), f"{tmpdir}/scheduler.pth")
    new_scheduler(optim)
    new_scheduler.load_state_dict(torch.load(f"{tmpdir}/scheduler.pth"))
    assert new_scheduler.state_dict() == scheduler.state_dict()
