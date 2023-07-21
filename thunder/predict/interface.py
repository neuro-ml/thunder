from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, Iterable, Any, Optional, Callable

from more_itertools import chunked, zip_equal
from toolz import peek

__all__ = 'Predictor', 'Chain'


class Predictor(ABC):
    @abstractmethod
    def forward(self, values: Iterable) -> Iterable[Tuple[Any, Any]]:
        raise NotImplementedError

    @abstractmethod
    def backward(self, pairs: Iterable[Tuple[Any, Any]]) -> Iterable:
        raise NotImplementedError

    def run(self, values: Iterable, predict: Callable[[Any], Any]) -> Iterable:
        def process(pairs):
            for value, context in pairs:
                yield predict(value), context

        return self.backward(process(self.forward(values)))


class Chain(Predictor):
    def __init__(self, *predictors: Predictor):
        super().__init__()
        self.predictors = predictors

    def forward(self, values: Iterable) -> Iterable[Tuple[Any, Any]]:
        # this is a bit hacky, but necessary
        #  we don't want to store anything in the class fields,
        #  so we need a way to safely pass a state from forward to backward

        def wrap(predictor, xs):
            for v, c in predictor.forward(xs):
                contexts[predictor].append(c)
                yield v

        # TODO: queue
        contexts = defaultdict(list)
        for p in self.predictors:
            values = wrap(p, values)

        for value in values:
            yield value, contexts

    def backward(self, pairs: Iterable[Tuple[Any, Any]]) -> Iterable:
        try:
            (_, contexts), pairs = peek(pairs)
        except StopIteration:
            return ()

        values = (x for x, _ in pairs)
        for predictor in self.predictors:
            values = predictor.backward((v, contexts[predictor].pop(0)) for v in values)

        return values
