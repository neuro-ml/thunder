from abc import ABC, abstractmethod
from itertools import chain
from typing import Callable, Iterable

from more_itertools import zip_equal, mark_ends, chunked, split_after
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


class Decorated(InfinitePredictor):
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


class SlidingWindow(BasePredictor):
    def __init__(self, patcher: Callable, combiner: Callable, batch_size: int = 1, **patcher_kwargs):
        super().__init__()
        self.patcher = patcher
        self.batch_size = batch_size
        self.combiner = combiner
        self.patcher_kwargs = patcher_kwargs

    def forward(self, batches: Iterable) -> Iterable:
        recombined_batches = map(lambda batch: zip_equal(*batch), batches)

        def get_patches(batches):
            for xs in chain.from_iterable(batches):
                patches = self.patcher(*xs, **self.patcher_kwargs)
                yield from mark_ends(patches)

        yield from chunked(get_patches(recombined_batches), self.batch_size)

    def backward(self, predicts: Iterable) -> Iterable:
        patched_images = chain.from_iterable(map(lambda p: zip(*p), predicts))
        for patched_image in split_after(patched_images, lambda pim: pim[1]):
            yield self.combiner(patched_image)

    def run(self, batches: Iterable, predict_fn: Callable) -> Iterable:
        def run_predict(batches, predict_fn):
            for batch_of_patches in self.forward(batches):
                firsts, lasts, patches = zip_equal(*batch_of_patches)
                preds = predict_fn(patches)
                yield firsts, lasts, preds

        return self.backward(run_predict(batches, predict_fn))
