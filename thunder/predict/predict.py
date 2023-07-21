from abc import ABC, abstractmethod
from typing import Callable, Iterable

from toolz import compose


class BasePredictor(ABC):
    """Base class for all predictors."""

    @abstractmethod
    def forward(self, batches: Iterable) -> Iterable:
        """Process stream of batches before model inference."""
        raise NotImplementedError("You must implement forward method")

    @abstractmethod
    def backward(self, predicts: Iterable) -> Iterable:
        """Post-process stream of predictions."""
        raise NotImplementedError("You must implement backward method")

    def __call__(self, batches: Iterable, predict_fn: Callable) -> Iterable:
        return self.run(batches, predict_fn)

    def run(self, batches: Iterable, predict_fn: Callable) -> Iterable:
        """Runs preprocessing, inference and postprocessing."""
        return self.backward(map(predict_fn, self.forward(batches)))


class InfinitePredictor(BasePredictor):
    """Useful for running inference on infinite stream of data."""

    def forward(self, batches: Iterable) -> Iterable:
        yield from batches

    def backward(self, predicts: Iterable) -> Iterable:
        yield from predicts


class Predictor(InfinitePredictor):
    """Assumes using finite amount of data for inference to be run on."""

    def run(self, batches: Iterable, predict_fn: Callable) -> Iterable:
        return tuple(super().run(batches, predict_fn))


class Decorated(Predictor):
    """
    Decorates inference function
    Example
    -----------
    Decorated(f, g, h)
    # inside Decorated
    predict_fn = f(g(h(predict_fn)))
    """

    def __init__(self, *decorators: Callable):
        self.decorators = compose(*decorators)

    def run(self, batches: Iterable, predict_fn: Callable) -> Iterable:
        return super().run(batches, self.decorators(predict_fn))
