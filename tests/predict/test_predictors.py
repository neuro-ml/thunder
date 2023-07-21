import numpy as np
import torch
import torch.nn as nn

from thunder.predict import ExtraDims
from thunder.torch.utils import maybe_from_np, to_np


def predict(x: np.ndarray, model: nn.Module) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = maybe_from_np(x)
        return to_np(model(x))


def test_predictor_single_object():
    def predict(z):
        assert z.ndim == 5
        assert z.shape[1] == z.shape[3] == 1
        return np.zeros_like(z)

    predictor = ExtraDims(1, 3)
    x = np.random.randn(2, 3, 4)
    y, = predictor.run([x], predict)
    assert (y == np.zeros_like(x)).all()


class SplitLinear(nn.Module):
    def __init__(self, single_mode: bool):
        super().__init__()
        self.l1 = nn.Linear(4, 1)
        self.l2 = nn.Linear(4, 1)
        self.single_mode = single_mode

    def forward(self, x) -> tuple:
        if self.single_mode:
            return self.l1(x), self.l2(x)
        return self.l1(x[0]), self.l2(x[1])


def test_predictor_multi_object():
    predictor = Predictor()

    model = reinit(SplitLinear(True))
    batch = [np.random.randn(1, 4).astype(np.float32) for _ in range(8)]

    for x in batch:
        stream = predictor(x)
        y = stream.send(predict(next(stream), model))

        assert y[0].shape == (1, 1) and y[1].shape == (1, 1)
        assert np.allclose(y, np.zeros_like(y))


def test_map():
    predictor = Map(Predictor())
    model = reinit(torch.nn.Linear(4, 1))
    batch = [np.random.randn(1, 4).astype(np.float32) for _ in range(8)]

    stream = predictor(batch)
    for x in stream:
        y = stream.send(predict(x, model))
        assert y.shape == (1, 1)
        assert np.allclose(y, np.zeros_like(y))


def test_chain():
    class PlusOne(Predictor):
        def backward(self, y):
            return y + 1

    predictor = Chain(Predictor(), PlusOne())

    model = reinit(torch.nn.Linear(4, 1))
    batch = [np.random.randn(1, 4).astype(np.float32) for _ in range(8)]

    for x in batch:
        stream = predictor(x)
        y = stream.send(predict(next(stream), model))

        assert y.shape == (1, 1)
        assert np.allclose(y, np.ones_like(y))


def reinit(model: nn.Module, constant: float = 0.0) -> nn.Module:
    for m in model.modules():
        if hasattr(m, "weight"):
            m.weight.data.fill_(constant)
        if hasattr(m, "bias"):
            m.bias.data.fill_(constant)

    return model
