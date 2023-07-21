from typing import Iterable, Tuple, Any

import numpy as np
import torch

from .interface import Predictor

__all__ = 'ExtraDims',


class ExtraDims(Predictor):
    def __init__(self, *axis):
        super().__init__()
        self.axis = axis

    def forward(self, values: Iterable) -> Iterable[Tuple[Any, Any]]:
        for value in values:
            if isinstance(value, (np.ndarray, np.generic)):
                value = np.expand_dims(value, self.axis)
            else:
                # we basically add axes from the greatest to the smallest
                for ax in sorted(np.normalize_axis_tuple(self.axis, value.ndim), reverse=True):
                    value = torch.unsqueeze(value, ax)

            yield value, None

    def backward(self, pairs: Iterable[Tuple[Any, Any]]) -> Iterable:
        for value, _ in pairs:
            if isinstance(value, (np.ndarray, np.generic)):
                value = np.squeeze(value, self.axis)
            else:
                for ax in sorted(np.normalize_axis_tuple(self.axis, value.ndim), reverse=True):
                    value = torch.squeeze(value, ax)

            yield value
