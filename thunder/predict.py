from abc import ABC, abstractmethod
from typing import Callable, Any


class Predictor(ABC):
    def __init__(self, predict_fn: Callable):
        self.predict_fn = predict_fn

    def __call__(self, x) -> Any:
        return self.predict(x)

    @abstractmethod
    def predict(self, x) -> Any:
        return self.predict_fn(x)


class DefaultPredictor(Predictor):
    def predict(self, x) -> Any:
        return super().predict(x)
