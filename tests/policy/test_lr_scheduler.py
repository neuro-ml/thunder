import numpy as np
import pytest
import torch
from more_itertools import zip_equal
from torch import nn
from torch.nn import Sequential
from torch.optim import Adam

from thunder.policy import Multiply, Policy, Schedule, Switch


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
    _test_scheduler(policy, [0, 0, 0, 0, 0])
    _test_scheduler_saving(NewPolicy()(optim), policy, optim, tmpdir)


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
    _test_scheduler(scheduler, targets)

    new_scheduler = Multiply(mapping)
    _test_scheduler_saving(new_scheduler, scheduler, optim, tmpdir)


@pytest.mark.parametrize("mapping,targets", [(np.cos, [[np.cos(i), np.cos(i)] for i in range(5)]),
                                             ([np.cos, np.sin], [[np.cos(i), np.sin(i)] for i in range(5)])])
def test_schedule(mapping, targets, optim, tmpdir):
    scheduler = Schedule(mapping, [1, 100])
    scheduler(optim)
    _test_scheduler(scheduler, targets)

    new_scheduler = Schedule(mapping)
    _test_scheduler_saving(new_scheduler, scheduler, optim, tmpdir)


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
    _test_scheduler(scheduler, targets)

    new_scheduler = Switch(mapping, 0.0001)
    _test_scheduler_saving(new_scheduler, scheduler, optim, tmpdir)


def _test_scheduler(scheduler, targets):
    for i, target in zip_equal(range(5), targets):
        assert np.allclose([pg["lr"] for pg in scheduler.optimizer.param_groups], target), f"epoch {i}"
        scheduler.optimizer.step()
        scheduler.step()


def _test_scheduler_saving(new_scheduler, scheduler, optim, tmpdir):
    torch.save(scheduler.state_dict(), f"{tmpdir}/scheduler.pth")
    new_scheduler(optim)
    new_scheduler.load_state_dict(torch.load(f"{tmpdir}/scheduler.pth"))
    assert new_scheduler.state_dict() == scheduler.state_dict()
