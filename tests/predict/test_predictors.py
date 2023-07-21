import numpy as np
from more_itertools import ilen

from thunder.predict import Decorated


def test_decorated():
    def predict(x):
        return x.sum() ** 2

    predictor = Decorated(lambda f: (lambda x: f(x + 1)))
    x = np.ones((2, 10))

    assert all(y == 400 for y in predictor(x, predict)) and ilen(predictor(x, predict)) == 2
