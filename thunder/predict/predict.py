from typing import Union, Tuple, Iterable, Any

import numpy as np
import torch
from toolz import compose, compose_left

Array = Union[np.ndarray, torch.Tensor]
ArrayLike = Union[Array, Tuple[Array, ...]]


class BasePredictor:
    def forward(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError("Method `forward` must be implemented.")

    def backward(self, y: ArrayLike) -> Any:
        raise NotImplementedError("Method `must` must be implemented.")

    def __call__(self, *args, **kwargs) -> Iterable:
        return self.run(*args, **kwargs)

    def run(self, x: ArrayLike) -> ArrayLike:
        y = yield self.forward(x)
        yield self.backward(y)


class Predictor(BasePredictor):
    def forward(self, x: ArrayLike) -> ArrayLike:
        return x

    def backward(self, y: ArrayLike) -> Any:
        return y


class Map(BasePredictor):
    def __init__(self, predictor: BasePredictor):
        super().__init__()
        self.predictor = predictor

    def run(self, xs: Iterable[ArrayLike]) -> Iterable[ArrayLike]:
        for x in xs:
            yield from self.predictor(x)


class Chain(BasePredictor):
    def __init__(self, *predictors: BasePredictor):
        super().__init__()
        self._composed_forward = compose_left(*[p.forward for p in predictors])
        self._composed_backward = compose(*[p.backward for p in predictors])

    def forward(self, x: ArrayLike) -> ArrayLike:
        return self._composed_forward(x)

    def backward(self, y: ArrayLike) -> Any:
        return self._composed_backward(y)
