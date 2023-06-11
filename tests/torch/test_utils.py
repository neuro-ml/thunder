from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import torch
from lightning import LightningModule
from more_itertools import zip_equal

from thunder import ThunderModule
from thunder.torch.utils import get_device, to_np, maybe_from_np


@pytest.mark.parametrize(
    "value, target",
    [
        (
            {"x": torch.Tensor([1, 2, 3]), "y": torch.from_numpy(np.ones((3, 3)))},
            {"x": np.asarray([1, 2, 3]).astype(np.float32), "y": np.ones((3, 3))}
        ),
        ([torch.ones(10, 11), torch.zeros(5, 5)], [np.ones((10, 11)), np.zeros((5, 5))]),
        ((torch.ones(10, 11), torch.zeros(5, 5)), (np.ones((10, 11)), np.zeros((5, 5)))),
        (torch.eye(3), np.eye(3)),
    ],
)
def test_to_np(value, target):
    if isinstance(value, dict):
        for (k1, v1), (k2, v2) in zip_equal(to_np(value).items(), target.items()):
            assert k1 == k2 and np.all(v1 == v2)
    elif isinstance(value, (list, tuple)):
        value = to_np(value)
        assert isinstance(value, (list, tuple)), type(value)
        for v1, v2 in zip_equal(value, target):
            assert np.all(v1 == v2)
    else:
        assert np.all(to_np(value) == target)


@pytest.mark.parametrize(
    "value, target",
    [
        (
            {"x": np.asarray([1, 2, 3]).astype(np.float32), "y": np.ones((3, 3))},
            {"x": torch.Tensor([1, 2, 3]), "y": torch.from_numpy(np.ones((3, 3)))}
        ),
        ([np.ones((10, 11)), np.zeros((5, 5))], [torch.ones(10, 11), torch.zeros(5, 5)]),
        ((np.ones((10, 11)), np.zeros((5, 5))), (torch.ones(10, 11), torch.zeros(5, 5))),
        (np.eye(3), torch.eye(3)),
    ],
)
def test_maybe_from_np(value, target):
    if isinstance(value, dict):
        for (k1, v1), (k2, v2) in zip_equal(maybe_from_np(value, device="cpu").items(), target.items()):
            assert k1 == k2 and (v1 == v2).all(), (v1, v2)
    elif isinstance(value, (list, tuple)):
        value = maybe_from_np(value, device="cpu")
        assert isinstance(value, (list, tuple)), type(value)
        for v1, v2 in zip_equal(value, target):
            assert (v1 == v2).all()
    else:
        assert (maybe_from_np(value, device="cpu") == target).all()


@pytest.mark.parametrize(
    "x, expected",
    [
        (torch.eye(3), does_not_raise()),
        (torch.nn.Linear(2, 1), does_not_raise()),
        (LightningModule(), does_not_raise()),
        (ThunderModule(torch.nn.Linear(2, 1), lambda x, y: x + y), does_not_raise()),
        ([], pytest.raises(TypeError, match="list")),
    ],
)
def test_get_device(x, expected):
    with expected:
        assert get_device(x) == torch.device("cpu")
