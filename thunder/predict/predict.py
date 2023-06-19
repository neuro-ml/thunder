from abc import ABC, abstractmethod
from typing import Callable, Iterable


class BasePredictor(ABC):
    @abstractmethod
    def forward(self, batches: Iterable) -> Iterable:
        raise NotImplementedError("You must implement forward method")

    @abstractmethod
    def backward(self, predicts: Iterable) -> Iterable:
        raise NotImplementedError("You must implement backward method")

    def __call__(self, batches: Iterable, predict_fn: Callable) -> Iterable:
        return self.run(batches, predict_fn)

    def run(self, batches: Iterable, predict_fn: Callable) -> Iterable:
        return self.backward(map(predict_fn, self.forward(batches)))


class InfinitePredictor(BasePredictor):
    def forward(self, batches: Iterable) -> Iterable:
        yield from batches

    def backward(self, predicts: Iterable) -> Iterable:
        yield from predicts


class Predictor(InfinitePredictor):
    def run(self, batches: Iterable, predict_fn: Callable) -> Iterable:
        return tuple(super().run(batches, predict_fn))
